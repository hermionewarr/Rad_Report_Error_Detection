#!/usr/bin/env python3
"""Visualise BERT inconsistency attributions over findings and conclusion text.

Samples TP / TN / FP / FN examples from a predictions CSV and writes PNG + PDF heatmaps
per example. Two attribution methods:

  cls — final-layer CLS→token attention (qualitative, often diffuse)
  ig  — Integrated Gradients w.r.t. the predicted-class logit (sharper word alignment)

Word colours use offset-based alignment from the same tokenisation as inference.

Typical usage (from project root, PTCT3 env):

  python scripts/visualize_bert_attention.py \\
    --predictions_csv .../results/hard_predictions.csv \\
    --mode lat \\
    --method both

If --output_dir is omitted, plots go to <run>/plots/attention_attribution_<mode>/ inferred
from the predictions CSV path (or --run_dir). Each figure is saved as both .png and .pdf.

BERT inconsistency eval runs this automatically after saving predictions (use
bert_inconsistency_detector.py --skip_attention_attribution to disable).
Row filter: lat/site/severity/cross_pair_full keep matching error types; hard/all
keep every error type in the CSV.

Condensed paper summaries (top-10 words + keyword-context panel) are written to the
same attention_attribution_<mode>/ folder: summary_compact_panel.png, summary_top_words_ig.png,
summary_confusion_top_words_ig.png (2×2 TP/FN/FP/TN grid with findings|conclusion top words),
summary_top_words_cls_by_error_type.png (NF vs findings | severity | site side-by-side),
summary_laterality_patterns.png (lat) or summary_<mode>_term_lift.png (severity/site).
For all/hard/cross_pair_full, both summary_severity_term_lift.png and summary_site_term_lift.png
are written; the compact panel shows the top lifts from both families combined.
Use --skip_summary or --summary_max_rows to control cost (~1–3 min for 120 reports).
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

BERT_MODEL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BERT_MODEL_ROOT  # flat project root

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paths import resolve_run_layout_from_artifact  # noqa: E402
from src.bert_inconsistency_detector import (  # noqa: E402
    FINDINGS_SPECIAL,
    CONCLUSION_SPECIAL,
    SITE_SWAP_TERMS,
    SEVERITY_FLIP_TERMS,
    TOKENIZER_PATH,
    explain_consistency,
    init_tokenizer,
    load_consistency_classifier,
    normalize_error_type_name,
    row_included_in_eval_mode,
)

def _save_figure(fig: Figure, output_path: Path, *, dpi: int = 160) -> None:
    """Save figure as PNG and a matching PDF beside it."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")


BUCKET_SPECS = (
    ("tp", "is_tp0", "TP — detected inconsistency"),
    ("tn", "is_tn0", "TN — correct consistent"),
    ("fp", "is_fp0", "FP — false alarm"),
    ("fn", "is_fn0", "FN — missed inconsistency"),
)

# Inconsistency families for stratified top-word summaries (hard / all eval).
ERROR_TYPE_SUMMARY_GROUPS = (
    ("nf_vs_findings", "NF vs findings", frozenset({"NF_v_Dis_swap"})),
    ("severity", "Severity flip", frozenset({"severity_flip"})),
    ("site", "Site swap", frozenset({"site_swap"})),
)

METHOD_LABELS = {
    "cls": "CLS attention (final layer, mean heads)",
    "ig": "Integrated Gradients (predicted-class logit)",
}

# Light neutral → saturated orange (no dark maroon) so highlights stay legible.
CMAP = LinearSegmentedColormap.from_list(
    "attention_readable",
    ["#f8f9fa", "#fff8e1", "#ffe082", "#ffb74d", "#ffa726", "#fb8c00"],
    N=256,
)
HIGHLIGHT_TEXT_COLOR = "#1a1a1a"
BODY_FONT_SIZE = 9.0
MAX_WORDS_PER_SECTION = 220
FIG_WIDTH = 14.0
LINE_HEIGHT = 0.021
HEADER_HEIGHT = 0.15
SECTION_GAP = 0.018
CHARS_PER_LINE = 98
SUMMARY_TOP_N = 10
CONFUSION_GRID_TOP_N = 8
TERM_LIFT_MIN_PRESENT = 10
SUMMARY_BAR_COLOR = "#fb8c00"
FINDINGS_BAR_COLOR = "#1976d2"
CONCLUSION_BAR_COLOR = "#fb8c00"
SUMMARY_GRID_COLOR = "#f0f0f0"

# Confusion-matrix layout (inconsistent-positive): rows=true, cols=pred.
CONFUSION_CELL_LAYOUT = (
    (0, 0, "tp", "TP"),
    (0, 1, "fn", "FN"),
    (1, 0, "fp", "FP"),
    (1, 1, "tn", "TN"),
)

# Minimal stopwords for aggregate word ranking (keep left/right and clinical terms).
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "no",
        "not",
        "there",
        "this",
        "that",
        "as",
        "by",
        "from",
        "it",
        "its",
    }
)


def _safe_int(value: str, default: int = 0) -> int:
    try:
        return int((value or "").strip())
    except ValueError:
        return default


def load_prediction_rows(csv_path: Path, mode: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        required = {"findings", "conclusion", "error_type"}
        missing = required - fields
        if missing:
            raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")
        for row in reader:
            findings = (row.get("findings") or "").strip()
            conclusion = (row.get("conclusion") or "").strip()
            if not findings or not conclusion:
                continue
            error_type = (row.get("error_type") or "unknown").strip()
            if not row_included_in_eval_mode(error_type, mode):
                continue
            rows.append(row)
    return rows


def bucket_rows(rows: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    out: Dict[str, List[Dict[str, str]]] = {name: [] for name, _, _ in BUCKET_SPECS}
    for row in rows:
        for name, col, _ in BUCKET_SPECS:
            if _safe_int(row.get(col, "0")) == 1:
                out[name].append(row)
    return out


def sample_rows(
    pool: List[Dict[str, str]],
    n: int,
    rng: random.Random,
) -> List[Dict[str, str]]:
    if n <= 0 or not pool:
        return []
    if len(pool) <= n:
        return list(pool)
    return rng.sample(pool, n)


def _section_token_indices(
    token_scores: Sequence[Tuple[str, float]],
) -> Tuple[List[int], List[int]]:
    """Return token indices belonging to findings / conclusion sections."""
    findings_tok = FINDINGS_SPECIAL.strip("[]")
    conclusion_tok = CONCLUSION_SPECIAL.strip("[]")
    section = None
    findings_idx: List[int] = []
    conclusion_idx: List[int] = []
    skip = {findings_tok, conclusion_tok, "[CLS]", "[SEP]", "", "<s>", "</s>"}

    for i, (tok, _) in enumerate(token_scores):
        bare = tok.strip()
        if bare in {findings_tok, FINDINGS_SPECIAL}:
            section = "findings"
            continue
        if bare in {conclusion_tok, CONCLUSION_SPECIAL}:
            section = "conclusion"
            continue
        if bare in skip or (bare.startswith("[") and bare.endswith("]")):
            continue
        if section == "findings":
            findings_idx.append(i)
        elif section == "conclusion":
            conclusion_idx.append(i)
    return findings_idx, conclusion_idx


# Alphabetic tokens with optional internal hyphen/apostrophe; punctuation is not part of the word.
_WORD_SPAN_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*")


def normalize_word_token(word: str) -> str:
    """Lowercase alphabetic token with trailing/leading punctuation removed."""
    m = _WORD_SPAN_RE.search(word.strip())
    return m.group(0).lower() if m else word.strip().lower()


def iter_word_spans(text: str) -> List[Tuple[str, int, int]]:
    """Return (word, start, end) spans; punctuation is excluded from each word."""
    return [(m.group(0), m.start(), m.end()) for m in _WORD_SPAN_RE.finditer(text)]


def aggregate_words_by_offsets(
    section_text: str,
    section_char_start: int,
    offsets: Sequence[Tuple[int, int]],
    token_scores: Sequence[float],
    token_indices: Sequence[int],
) -> List[Tuple[str, float]]:
    """Map token attributions to alphabetic words via character overlap."""
    spans = iter_word_spans(section_text)
    if not spans:
        return []

    out: List[Tuple[str, float]] = []
    for word, local_start, local_end in spans:
        g0 = section_char_start + local_start
        g1 = section_char_start + local_end
        span_scores: List[float] = []
        for ti in token_indices:
            if ti >= len(offsets) or ti >= len(token_scores):
                continue
            a, b = offsets[ti]
            if a == 0 and b == 0:
                continue
            if a < g1 and b > g0:
                span_scores.append(abs(float(token_scores[ti])))
        out.append((word, max(span_scores) if span_scores else 0.0))
    return out


def word_scores_from_explain(
    row: Dict[str, str],
    explain: Dict[str, object],
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    token_scores: List[Tuple[str, float]] = explain["token_scores"]  # type: ignore[assignment]
    offsets: List[Tuple[int, int]] = explain["offsets"]  # type: ignore[assignment]
    f_span, c_span = explain["findings_span"], explain["conclusion_span"]  # type: ignore[misc]
    scores_only = [s for _, s in token_scores]
    f_idx, c_idx = _section_token_indices(token_scores)

    f_words = aggregate_words_by_offsets(
        row["findings"], f_span[0], offsets, scores_only, f_idx
    )
    c_words = aggregate_words_by_offsets(
        row["conclusion"], c_span[0], offsets, scores_only, c_idx
    )
    return f_words, c_words


def _normalize_scores(scores: Sequence[float]) -> List[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi <= lo:
        return [0.5 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def _wrap_word_lines(
    words: List[Tuple[str, float]], chars_per_line: int = 105
) -> List[List[Tuple[str, float, float]]]:
    if not words:
        return []
    words = words[:MAX_WORDS_PER_SECTION]
    norms = _normalize_scores([s for _, s in words])
    lines: List[List[Tuple[str, float, float]]] = []
    current: List[Tuple[str, float, float]] = []
    line_len = 0
    for (word, score), norm in zip(words, norms):
        add = len(word) + (1 if current else 0)
        if current and line_len + add > chars_per_line:
            lines.append(current)
            current = []
            line_len = 0
        current.append((word, score, norm))
        line_len += add
    if current:
        lines.append(current)
    return lines


def _estimate_figure_height(
    findings_lines: int, conclusion_lines: int, header_lines: int = 5
) -> float:
    n_lines = header_lines + findings_lines + conclusion_lines + 3
    return max(4.0, HEADER_HEIGHT + n_lines * LINE_HEIGHT * 38)


def _draw_word_line(
    ax: Axes,
    y: float,
    line: List[Tuple[str, float, float]],
    fontsize: float,
    x_start: float = 0.01,
) -> None:
    x = x_start
    char_w = 0.0062 * (fontsize / BODY_FONT_SIZE)
    for word, _score, norm in line:
        label = word + " "
        facecolor = CMAP(norm)
        ax.text(
            x,
            y,
            label,
            transform=ax.transAxes,
            fontsize=fontsize,
            va="top",
            ha="left",
            color=HIGHLIGHT_TEXT_COLOR,
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor=facecolor,
                edgecolor="#d0d0d0" if norm < 0.12 else "#e65100" if norm > 0.85 else "none",
                linewidth=0.6,
                alpha=0.95,
            ),
        )
        x += len(label) * char_w


def _draw_section(
    ax: Axes,
    y_top: float,
    title: str,
    word_scores: List[Tuple[str, float]],
    chars_per_line: int,
    fontsize: float,
) -> float:
    lines = _wrap_word_lines(word_scores, chars_per_line=chars_per_line)
    ax.text(
        0.01,
        y_top,
        title,
        transform=ax.transAxes,
        fontsize=fontsize + 1.5,
        fontweight="bold",
        va="top",
        ha="left",
    )
    y = y_top - LINE_HEIGHT * 1.15
    for line in lines:
        _draw_word_line(ax, y, line, fontsize=fontsize)
        y -= LINE_HEIGHT
    if len(word_scores) > MAX_WORDS_PER_SECTION:
        ax.text(
            0.01,
            y,
            f"… truncated ({len(word_scores) - MAX_WORDS_PER_SECTION} more words)",
            transform=ax.transAxes,
            fontsize=fontsize,
            va="top",
            color="#666",
        )
        y -= LINE_HEIGHT
    return y - SECTION_GAP


def plot_example_attention(
    title: str,
    row: Dict[str, str],
    explain: Dict[str, object],
    output_path: Path,
    dpi: int = 140,
) -> None:
    method = str(explain.get("attribution", "cls"))
    f_words, c_words = word_scores_from_explain(row, explain)

    f_lines = _wrap_word_lines(f_words)
    c_lines = _wrap_word_lines(c_words)
    fig_h = _estimate_figure_height(len(f_lines), len(c_lines))

    fig: Figure = plt.figure(figsize=(FIG_WIDTH, fig_h))
    ax = fig.add_axes([0.03, 0.02, 0.94, 0.96])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    true_name = row.get("true_label_name") or ""
    pred_name = row.get("pred_label_name") or str(explain.get("prediction", ""))
    p_inc = row.get("prob_inconsistent") or ""
    p_con = row.get("prob_consistent") or ""
    if p_inc or p_con:
        prob_str = f"inconsistent={p_inc}, consistent={p_con}"
    else:
        prob_str = (
            f"inconsistent={explain['inconsistency_prob']:.4f}, "
            f"consistent={explain['consistency_prob']:.4f}"
        )

    method_desc = METHOD_LABELS.get(method, method)
    header = textwrap.dedent(
        f"""\
        {title}
        patient_id={row.get('patient_id', '')}  source_idx={row.get('source_idx', '')}  error_type={row.get('error_type', '')}
        true={true_name}  pred={pred_name}  ({prob_str})
        {method_desc}. Words aligned via tokenizer offsets. Highlight = normalised score (light→orange).
        """
    ).strip()
    ax.text(
        0.01,
        0.99,
        header,
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        ha="left",
        family="monospace",
    )

    y = 0.99 - HEADER_HEIGHT
    y = _draw_section(
        ax, y, "Findings", f_words, chars_per_line=CHARS_PER_LINE, fontsize=BODY_FONT_SIZE
    )
    _draw_section(
        ax, y, "Conclusion", c_words, chars_per_line=CHARS_PER_LINE, fontsize=BODY_FONT_SIZE
    )

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.02, pad=0.01)
    cbar.set_label(f"normalised {method} attribution", fontsize=9)

    _save_figure(fig, output_path, dpi=dpi)
    plt.close(fig)


def collect_examples(
    buckets: Dict[str, List[Dict[str, str]]],
    counts: Dict[str, int],
    rng: random.Random,
) -> List[Tuple[str, str, Dict[str, str]]]:
    selected: List[Tuple[str, str, Dict[str, str]]] = []
    for name, _, label in BUCKET_SPECS:
        n = counts.get(name, 0)
        for i, row in enumerate(sample_rows(buckets[name], n, rng)):
            eid = f"{name}_{i}"
            selected.append((eid, f"{label} (#{i + 1})", row))
    return selected


def attribution_row_filter_mode(eval_mode: str) -> str:
    """Map eval CLI mode to row_included_in_eval_mode filter."""
    if eval_mode in {"hard", "all"}:
        return "all"
    return eval_mode


def default_attention_attribution_dir(predictions_csv: str, eval_mode: str) -> Path:
    plots_dir = Path(predictions_csv).resolve().parent.parent / "plots"
    out = plots_dir / f"attention_attribution_{eval_mode}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def resolve_output_dir(
    predictions_csv: Path,
    model_dir: Path,
    mode: str,
    explicit_output: Optional[str],
    explicit_run_dir: Optional[str],
) -> Path:
    if explicit_output:
        out = Path(explicit_output).resolve()
        out.mkdir(parents=True, exist_ok=True)
        return out

    layout = None
    if explicit_run_dir:
        root = Path(explicit_run_dir).resolve()
        layout = resolve_run_layout_from_artifact(root / "results")
        if layout is None and (root / "model").is_dir():
            from paths import RunLayout

            layout = RunLayout(
                root=root,
                model=root / "model",
                results=root / "results",
                plots=root / "plots",
                logs=root / "logs",
            )
    else:
        layout = resolve_run_layout_from_artifact(predictions_csv)
        if layout is None:
            layout = resolve_run_layout_from_artifact(model_dir)

    if layout is not None:
        out = layout.plots / f"attention_attribution_{mode}"
        out.mkdir(parents=True, exist_ok=True)
        return out

    out = BERT_MODEL_ROOT / "analysis_figures" / f"bert_attention_{mode}"
    out.mkdir(parents=True, exist_ok=True)
    return out.resolve()


def write_attribution_manifest(
    output_dir: Path,
    *,
    run_name: str,
    predictions_csv: Path,
    model_dir: Path,
    mode: str,
    method: str,
    ig_steps: int,
    plot_files: Sequence[str],
) -> None:
    manifest = output_dir / "attention_attribution_source.txt"
    manifest.write_text(
        "\n".join(
            [
                "BERT inconsistency attribution plots",
                f"run: {run_name}",
                f"predictions_csv: {predictions_csv}",
                f"model_dir: {model_dir}",
                f"eval_mode_filter: {mode}",
                f"method: {method}",
                f"ig_steps: {ig_steps}",
                "",
                "Plots:",
                *[f"  {name}" for name in plot_files],
            ]
        ),
        encoding="utf-8",
    )


def _word_ok_for_summary(word: str) -> bool:
    bare = normalize_word_token(word)
    if len(bare) < 2 or not bare.replace("-", "").replace("'", "").isalpha():
        return False
    return bare not in _STOPWORDS


def _error_type_summary_group(row: Dict[str, str]) -> Optional[str]:
    et = normalize_error_type_name(row.get("error_type", ""))
    for key, _, types in ERROR_TYPE_SUMMARY_GROUPS:
        if et in types:
            return key
    return None


def sample_rows_for_error_type_groups(
    rows: Sequence[Dict[str, str]],
    per_group_max: int,
    rng: random.Random,
) -> Dict[str, List[Dict[str, str]]]:
    """Sample up to per_group_max reports per inconsistency family."""
    pools: Dict[str, List[Dict[str, str]]] = {key: [] for key, _, _ in ERROR_TYPE_SUMMARY_GROUPS}
    for row in rows:
        group = _error_type_summary_group(row)
        if group:
            pools[group].append(row)
    return {
        key: sample_rows(pools[key], per_group_max, rng)
        for key, _, _ in ERROR_TYPE_SUMMARY_GROUPS
    }


def sample_rows_for_summary(
    buckets: Dict[str, List[Dict[str, str]]],
    max_rows: int,
    rng: random.Random,
) -> List[Dict[str, str]]:
    if max_rows <= 0:
        return []
    per_bucket = max(1, max_rows // len(BUCKET_SPECS))
    picked: List[Dict[str, str]] = []
    seen: set[Tuple[str, str, str]] = set()
    for name, _, _ in BUCKET_SPECS:
        for row in sample_rows(buckets[name], per_bucket, rng):
            key = (
                row.get("patient_id", ""),
                row.get("source_idx", ""),
                row.get("error_type", ""),
            )
            if key in seen:
                continue
            seen.add(key)
            picked.append(row)
    if len(picked) > max_rows:
        picked = rng.sample(picked, max_rows)
    return picked


def _binary_pair_pattern(text: str, term_a: str, term_b: str) -> str:
    has_a = bool(re.search(rf"\b{re.escape(term_a)}\b", text, re.IGNORECASE))
    has_b = bool(re.search(rf"\b{re.escape(term_b)}\b", text, re.IGNORECASE))
    if has_a and has_b:
        return f"both {term_a} & {term_b}"
    if has_a:
        return f"{term_a} only"
    if has_b:
        return f"{term_b} only"
    return "neither"


def _report_text(row: Dict[str, str]) -> str:
    """Findings + conclusion (same text the model sees, minus special tokens)."""
    findings = (row.get("findings") or "").strip()
    conclusion = (row.get("conclusion") or "").strip()
    if findings and conclusion:
        return f"{findings} {conclusion}"
    return findings or conclusion


def _term_present(text: str, term: str) -> bool:
    return bool(re.search(rf"\b{re.escape(term)}\b", text, re.IGNORECASE))


def _row_prob_inconsistent(row: Dict[str, str]) -> float:
    raw = (row.get("prob_inconsistent") or "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    raw = (row.get("prob_consistent") or "").strip()
    if raw:
        try:
            return 1.0 - float(raw)
        except ValueError:
            pass
    return float("nan")


def _outcome_bucket(row: Dict[str, str]) -> Optional[str]:
    for name, col, _ in BUCKET_SPECS:
        if _safe_int(row.get(col, "0")) == 1:
            return name
    return None


def _accumulate_word_scores(
    totals: Dict[str, float],
    counts: Dict[str, int],
    words: Sequence[Tuple[str, float]],
) -> None:
    for word, score in words:
        if not _word_ok_for_summary(word):
            continue
        key = normalize_word_token(word)
        totals[key] += float(score)
        counts[key] += 1


def _finalize_word_means(
    totals: Dict[str, float], counts: Dict[str, int]
) -> Dict[str, Tuple[float, int]]:
    out: Dict[str, Tuple[float, int]] = {}
    for word, total in totals.items():
        n = counts[word]
        if n:
            out[word] = (total / n, n)
    return out


def _empty_bucket_section_stats() -> Dict[str, Dict[str, Dict[str, float]]]:
    buckets = ("tp", "tn", "fp", "fn")
    return {
        b: {"findings": defaultdict(float), "conclusion": defaultdict(float)} for b in buckets
    }


def _empty_bucket_section_counts() -> Dict[str, Dict[str, Dict[str, int]]]:
    buckets = ("tp", "tn", "fp", "fn")
    return {
        b: {"findings": defaultdict(int), "conclusion": defaultdict(int)} for b in buckets
    }


def compute_attribution_summaries(
    model: object,
    rows: Sequence[Dict[str, str]],
    *,
    ig_steps: int = 50,
) -> Tuple[
    Dict[str, Tuple[float, int]],
    Dict[str, Tuple[float, int]],
    Dict[str, Tuple[float, int]],
    Dict[str, Dict[str, Dict[str, Tuple[float, int]]]],
    Dict[str, Dict[str, Dict[str, Tuple[float, int]]]],
]:
    """One pass: full-report + per-bucket findings/conclusion stats for IG and CLS."""
    report_ig_t: Dict[str, float] = defaultdict(float)
    report_ig_c: Dict[str, int] = defaultdict(int)
    report_ig_dec_t: Dict[str, float] = defaultdict(float)
    report_ig_dec_c: Dict[str, int] = defaultdict(int)
    report_cls_t: Dict[str, float] = defaultdict(float)
    report_cls_c: Dict[str, int] = defaultdict(int)
    bucket_ig_t = _empty_bucket_section_stats()
    bucket_ig_c = _empty_bucket_section_counts()
    bucket_cls_t = _empty_bucket_section_stats()
    bucket_cls_c = _empty_bucket_section_counts()

    for i, row in enumerate(rows, start=1):
        bucket = _outcome_bucket(row)
        explain_ig = explain_consistency(
            model,
            row["findings"],
            row["conclusion"],
            attribution="ig",
            ig_steps=ig_steps,
            ig_target_label=0,
        )
        explain_ig_dec = explain_consistency(
            model,
            row["findings"],
            row["conclusion"],
            attribution="ig",
            ig_steps=ig_steps,
        )
        explain_cls = explain_consistency(
            model,
            row["findings"],
            row["conclusion"],
            attribution="cls",
            ig_steps=ig_steps,
        )
        f_ig, c_ig = word_scores_from_explain(row, explain_ig)
        f_ig_dec, c_ig_dec = word_scores_from_explain(row, explain_ig_dec)
        f_cls, c_cls = word_scores_from_explain(row, explain_cls)

        _accumulate_word_scores(report_ig_t, report_ig_c, list(f_ig) + list(c_ig))
        _accumulate_word_scores(
            report_ig_dec_t, report_ig_dec_c, list(f_ig_dec) + list(c_ig_dec)
        )
        _accumulate_word_scores(report_cls_t, report_cls_c, list(f_cls) + list(c_cls))

        if bucket:
            _accumulate_word_scores(bucket_ig_t[bucket]["findings"], bucket_ig_c[bucket]["findings"], f_ig)
            _accumulate_word_scores(bucket_ig_t[bucket]["conclusion"], bucket_ig_c[bucket]["conclusion"], c_ig)
            _accumulate_word_scores(bucket_cls_t[bucket]["findings"], bucket_cls_c[bucket]["findings"], f_cls)
            _accumulate_word_scores(
                bucket_cls_t[bucket]["conclusion"], bucket_cls_c[bucket]["conclusion"], c_cls
            )

        if i % 25 == 0:
            print(f"  summary attribution: {i}/{len(rows)} rows")

    def finalize_bucket(
        totals: Dict[str, Dict[str, Dict[str, float]]],
        counts: Dict[str, Dict[str, Dict[str, int]]],
    ) -> Dict[str, Dict[str, Dict[str, Tuple[float, int]]]]:
        out: Dict[str, Dict[str, Dict[str, Tuple[float, int]]]] = {}
        for bucket in totals:
            out[bucket] = {}
            for section in ("findings", "conclusion"):
                out[bucket][section] = _finalize_word_means(
                    totals[bucket][section], counts[bucket][section]
                )
        return out

    return (
        _finalize_word_means(report_ig_t, report_ig_c),
        _finalize_word_means(report_ig_dec_t, report_ig_dec_c),
        _finalize_word_means(report_cls_t, report_cls_c),
        finalize_bucket(bucket_ig_t, bucket_ig_c),
        finalize_bucket(bucket_cls_t, bucket_cls_c),
    )


def _word_rank_score(mean: float, count: int, *, rank_by: str) -> float:
    if rank_by == "total":
        return mean * count
    return mean


def _bar_values(
    ranked: Sequence[Tuple[str, float, int]],
    *,
    value: str,
) -> List[float]:
    if value == "total":
        return [mean * count for _, mean, count in ranked]
    return [mean for _, mean, _ in ranked]


def _xlim_for_bar_annotations(
    bar_vals: Sequence[float],
    counts: Sequence[int],
    *,
    frac_pad: float = 0.02,
    per_char_frac: float = 0.016,
    x_pad_frac: float = 0.02,
) -> float:
    """Upper x-limit leaving room for trailing ``n=…`` labels."""
    if not bar_vals:
        return 1.0
    max_bar = max(bar_vals)
    x_pad = max_bar * x_pad_frac
    extents = [
        w + x_pad + max_bar * per_char_frac * len(f"n={n}")
        for w, n in zip(bar_vals, counts)
    ]
    return max(extents) * (1.0 + frac_pad)


def top_n_words(
    word_stats: Dict[str, Tuple[float, int]],
    n: int = SUMMARY_TOP_N,
    *,
    rank_by: str = "mean",
) -> List[Tuple[str, float, int]]:
    ranked = sorted(
        word_stats.items(),
        key=lambda kv: _word_rank_score(kv[1][0], kv[1][1], rank_by=rank_by),
        reverse=True,
    )
    return [(w, mean, cnt) for w, (mean, cnt) in ranked[:n]]


def _draw_mini_word_bars(
    ax: Axes,
    ranked: Sequence[Tuple[str, float, int]],
    *,
    title: str,
    bar_color: str,
    max_words: int = CONFUSION_GRID_TOP_N,
) -> None:
    subset = list(ranked[:max_words])
    if not subset:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=8, color="#888")
        ax.set_title(title, fontsize=8, pad=4)
        ax.axis("off")
        return
    words = [w for w, _, _ in subset][::-1]
    means = [m for _, m, _ in subset][::-1]
    ax.barh(range(len(words)), means, color=bar_color, edgecolor="#444", linewidth=0.35)
    ax.set_yticks(range(len(words)), labels=words, fontsize=7)
    ax.set_title(title, fontsize=8, pad=4)
    ax.tick_params(axis="x", labelsize=6)
    ax.grid(axis="x", alpha=0.2, color=SUMMARY_GRID_COLOR)


def plot_confusion_section_top_words_grid(
    bucket_stats: Dict[str, Dict[str, Dict[str, Tuple[float, int]]]],
    output_path: Path,
    *,
    method_label: str,
    bucket_counts: Optional[Dict[str, int]] = None,
) -> None:
    """2×2 confusion layout; each cell = findings | conclusion top-word bars."""
    fig = plt.figure(figsize=(13.5, 10.5))
    outer = fig.add_gridspec(
        3,
        3,
        width_ratios=[0.06, 1.0, 1.0],
        height_ratios=[0.08, 1.0, 1.0],
        wspace=0.28,
        hspace=0.32,
    )

    fig.text(0.5, 0.97, f"Top {CONFUSION_GRID_TOP_N} words by section — {method_label}", ha="center", fontsize=12)
    fig.text(0.42, 0.92, "Pred: inconsistent", ha="center", fontsize=10, fontweight="bold")
    fig.text(0.78, 0.92, "Pred: consistent", ha="center", fontsize=10, fontweight="bold")

    for row_idx, row_label in enumerate(("True: inconsistent", "True: consistent")):
        fig.text(
            0.02,
            0.72 - row_idx * 0.42,
            row_label,
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
            rotation=90,
        )

    for row_i, col_i, bucket, short in CONFUSION_CELL_LAYOUT:
        cell = outer[row_i + 1, col_i + 1].subgridspec(1, 2, wspace=0.35)
        ax_f = fig.add_subplot(cell[0, 0])
        ax_c = fig.add_subplot(cell[0, 1])

        section_stats = bucket_stats.get(bucket, {})
        f_ranked = top_n_words(section_stats.get("findings", {}), CONFUSION_GRID_TOP_N)
        c_ranked = top_n_words(section_stats.get("conclusion", {}), CONFUSION_GRID_TOP_N)

        n_cell = (bucket_counts or {}).get(bucket, 0)
        title_suffix = f" (n={n_cell})" if n_cell else ""
        _draw_mini_word_bars(
            ax_f,
            f_ranked,
            title=f"{short} — findings{title_suffix}",
            bar_color=FINDINGS_BAR_COLOR,
        )
        _draw_mini_word_bars(
            ax_c,
            c_ranked,
            title=f"{short} — conclusion{title_suffix}",
            bar_color=CONCLUSION_BAR_COLOR,
        )

    _save_figure(fig, output_path, dpi=160)
    plt.close(fig)
    print(f"Saved confusion-grid summary to: {output_path}")
    print(f"Saved confusion-grid summary to: {output_path.with_suffix('.pdf')}")


def plot_top_words_bar(
    ranked: Sequence[Tuple[str, float, int]],
    output_path: Path,
    *,
    title: str,
    method_label: str,
    value: str = "mean",
) -> None:
    if not ranked:
        print(f"Skipping empty top-words plot: {output_path}")
        return
    words = [w for w, _, _ in ranked][::-1]
    bar_vals = _bar_values(ranked, value=value)[::-1]
    counts = [c for _, _, c in ranked][::-1]

    fig_h = max(3.2, 0.38 * len(words) + 1.2)
    fig, ax = plt.subplots(figsize=(6.5, fig_h))
    y_pos = range(len(words))
    bars = ax.barh(y_pos, bar_vals, color=SUMMARY_BAR_COLOR, edgecolor="#e65100", linewidth=0.5)
    ax.set_yticks(y_pos, labels=words, fontsize=10)
    if value == "total":
        ax.set_xlabel(f"total {method_label} (mean × n, full report)", fontsize=10)
    else:
        ax.set_xlabel(f"mean {method_label} (full report)", fontsize=10)
    ax.set_title(title, fontsize=11, pad=8)
    ax.grid(axis="x", alpha=0.25, color=SUMMARY_GRID_COLOR)
    x_pad = max(bar_vals) * 0.02 if bar_vals else 0.0
    for bar, n in zip(bars, counts):
        ax.text(
            bar.get_width() + x_pad,
            bar.get_y() + bar.get_height() / 2,
            f"n={n}",
            va="center",
            ha="left",
            fontsize=8,
            color="#444",
        )
    ax.set_xlim(0, _xlim_for_bar_annotations(bar_vals, counts))
    plt.tight_layout()
    _save_figure(fig, output_path, dpi=160)
    plt.close(fig)
    print(f"Saved attribution summary plot to: {output_path}")
    print(f"Saved attribution summary plot to: {output_path.with_suffix('.pdf')}")


def compute_report_word_stats(
    model: object,
    rows: Sequence[Dict[str, str]],
    *,
    attribution: str,
    ig_steps: int = 50,
    ig_target_label: Optional[int] = None,
    label: str = "",
) -> Dict[str, Tuple[float, int]]:
    """Aggregate full-report word scores for one attribution method."""
    totals: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    if not rows:
        return {}

    for i, row in enumerate(rows, start=1):
        kwargs: Dict[str, object] = {
            "attribution": attribution,
            "ig_steps": ig_steps,
        }
        if ig_target_label is not None:
            kwargs["ig_target_label"] = ig_target_label
        explain = explain_consistency(
            model,
            row["findings"],
            row["conclusion"],
            **kwargs,
        )
        f_scores, c_scores = word_scores_from_explain(row, explain)
        _accumulate_word_scores(totals, counts, list(f_scores) + list(c_scores))
        if label and i % 25 == 0:
            print(f"  {label}: {i}/{len(rows)} rows")

    return _finalize_word_means(totals, counts)


def compute_error_type_word_summaries(
    model: object,
    rows_by_group: Dict[str, List[Dict[str, str]]],
    *,
    attribution: str,
    ig_steps: int = 50,
    ig_target_label: Optional[int] = None,
    method_label: str,
) -> Dict[str, Dict[str, Tuple[float, int]]]:
    """Per inconsistency-family word stats (CLS or IG)."""
    out: Dict[str, Dict[str, Tuple[float, int]]] = {}
    for key, _, _ in ERROR_TYPE_SUMMARY_GROUPS:
        group_rows = rows_by_group.get(key) or []
        if not group_rows:
            continue
        print(
            f"  error-type {key} ({method_label}): "
            f"n={len(group_rows)} reports"
        )
        out[key] = compute_report_word_stats(
            model,
            group_rows,
            attribution=attribution,
            ig_steps=ig_steps,
            ig_target_label=ig_target_label,
            label=f"error-type {key} ({method_label})",
        )
    return out


def plot_error_type_top_words_grid(
    ranked_by_group: Dict[str, Sequence[Tuple[str, float, int]]],
    group_counts: Dict[str, int],
    output_path: Path,
    *,
    method_label: str,
    value: str = "total",
    top_n: int = SUMMARY_TOP_N,
) -> None:
    """Side-by-side top-word bars for NF vs findings, severity, and site errors."""
    panels = [
        (key, title)
        for key, title, _ in ERROR_TYPE_SUMMARY_GROUPS
        if ranked_by_group.get(key)
    ]
    if not panels:
        print(f"Skipping error-type top-words grid (no data): {output_path}")
        return

    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(6.2 * len(panels), 4.6),
        squeeze=False,
    )
    fig.suptitle(
        f"Top {top_n} report words by inconsistency type — {method_label}",
        fontsize=12,
        y=1.02,
    )

    for ax, (group_key, group_title) in zip(axes[0], panels):
        ranked = list(ranked_by_group[group_key][:top_n])
        words = [w for w, _, _ in ranked][::-1]
        bar_vals = _bar_values(ranked, value=value)[::-1]
        counts = [c for _, _, c in ranked][::-1]
        n_reports = group_counts.get(group_key, 0)

        y_pos = range(len(words))
        bars = ax.barh(
            y_pos,
            bar_vals,
            color=SUMMARY_BAR_COLOR,
            edgecolor="#e65100",
            linewidth=0.5,
        )
        ax.set_yticks(y_pos, labels=words, fontsize=9)
        if value == "total":
            ax.set_xlabel(f"total {method_label}\n(mean × n)", fontsize=8)
        else:
            ax.set_xlabel(f"mean {method_label}", fontsize=8)
        ax.set_title(f"{group_title}\n(n={n_reports} reports)", fontsize=10, pad=6)
        ax.grid(axis="x", alpha=0.25, color=SUMMARY_GRID_COLOR)
        x_pad = max(bar_vals) * 0.02 if bar_vals else 0.0
        for bar, n in zip(bars, counts):
            ax.text(
                bar.get_width() + x_pad,
                bar.get_y() + bar.get_height() / 2,
                f"n={n}",
                va="center",
                ha="left",
                fontsize=7,
                color="#444",
            )
        ax.set_xlim(0, _xlim_for_bar_annotations(bar_vals, counts))

    plt.tight_layout()
    _save_figure(fig, output_path, dpi=160)
    plt.close(fig)
    print(f"Saved error-type top-words grid to: {output_path}")
    print(f"Saved error-type top-words grid to: {output_path.with_suffix('.pdf')}")


def _mean_prob_by_pattern(
    rows: Sequence[Dict[str, str]],
    pattern_fn,
) -> Dict[str, Dict[str, Tuple[float, int]]]:
    """pattern -> true_label_name -> (mean prob_inconsistent, n)."""
    buckets: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        pattern = pattern_fn(row)
        label = (row.get("true_label_name") or "").strip() or (
            "inconsistent" if _safe_int(row.get("true_label", "1")) == 0 else "consistent"
        )
        p = _row_prob_inconsistent(row)
        if p == p:
            buckets[pattern][label].append(p)
    out: Dict[str, Dict[str, Tuple[float, int]]] = {}
    for pattern, by_label in buckets.items():
        out[pattern] = {}
        for label, vals in by_label.items():
            out[pattern][label] = (sum(vals) / len(vals), len(vals))
    return out


def plot_laterality_pattern_heatmap(
    rows: Sequence[Dict[str, str]],
    output_path: Path,
    *,
    title: str,
) -> None:
    pattern_order = ("left only", "right only", "both left & right", "neither")
    label_order = ("consistent", "inconsistent")

    def pattern_fn(row: Dict[str, str]) -> str:
        return _binary_pair_pattern(_report_text(row), "left", "right")

    stats = _mean_prob_by_pattern(rows, pattern_fn)
    matrix = []
    annotations: List[List[str]] = []
    for label in label_order:
        row_vals: List[float] = []
        row_ann: List[str] = []
        for pattern in pattern_order:
            mean, n = stats.get(pattern, {}).get(label, (float("nan"), 0))
            row_vals.append(mean if mean == mean else 0.0)
            row_ann.append(f"{mean:.2f}\nn={n}" if n else "—")
        matrix.append(row_vals)
        annotations.append(row_ann)

    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pattern_order)), labels=pattern_order, fontsize=9)
    ax.set_yticks(range(len(label_order)), labels=[f"true {l}" for l in label_order], fontsize=9)
    ax.set_title(title, fontsize=11, pad=8)
    for i in range(len(label_order)):
        for j in range(len(pattern_order)):
            val = matrix[i][j]
            color = "white" if val > 0.55 else "black"
            ax.text(j, i, annotations[i][j], ha="center", va="center", fontsize=8, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("mean P(inconsistent)", fontsize=9)
    plt.tight_layout()
    _save_figure(fig, output_path, dpi=160)
    plt.close(fig)
    print(f"Saved pattern summary plot to: {output_path}")
    print(f"Saved pattern summary plot to: {output_path.with_suffix('.pdf')}")


def compute_term_lifts(
    rows: Sequence[Dict[str, str]],
    terms: Sequence[str],
    *,
    max_terms: Optional[int] = 12,
    min_present: int = 0,
) -> List[Tuple[str, float, int, int]]:
    """Δ mean P(inconsistent) when term present vs absent; ranked by |lift|."""
    seen: set[str] = set()
    unique_terms: List[str] = []
    for term in terms:
        key = term.lower()
        if key not in seen:
            seen.add(key)
            unique_terms.append(term)

    lifts: List[Tuple[str, float, int, int]] = []
    for term in unique_terms:
        present: List[float] = []
        absent: List[float] = []
        for row in rows:
            p = _row_prob_inconsistent(row)
            if p != p:
                continue
            if _term_present(_report_text(row), term):
                present.append(p)
            else:
                absent.append(p)
        if present and absent:
            lift = (sum(present) / len(present)) - (sum(absent) / len(absent))
            lifts.append((term, lift, len(present), len(absent)))
    if min_present > 0:
        lifts = [(t, l, np, na) for t, l, np, na in lifts if np >= min_present]
    lifts.sort(key=lambda x: abs(x[1]), reverse=True)
    if max_terms is not None:
        lifts = lifts[:max_terms]
    return lifts


def _xlim_for_signed_lift_bars(
    values: Sequence[float],
    lifts: Sequence[Tuple[str, float, int, int]],
    *,
    frac_pad: float = 0.12,
    per_char_frac: float = 0.018,
) -> Tuple[float, float]:
    """Symmetric-ish x-limits for signed lift bars with count annotations."""
    if not values:
        return (-0.1, 0.1)
    pos_max = max((v for v in values if v > 0), default=0.0)
    neg_min = min((v for v in values if v < 0), default=0.0)
    span = max(pos_max, -neg_min, 0.05)
    max_label = max((len(f"n+{np}, n−{na}") for _, _, np, na in lifts), default=10)
    margin = span * (frac_pad + per_char_frac * max_label)
    left = min(neg_min * 1.08, 0.0) - (margin if neg_min < 0 else 0.0)
    right = max(pos_max * 1.08, 0.0) + margin
    return left, right


def combined_swap_term_lifts(
    rows: Sequence[Dict[str, str]],
    *,
    per_category: int = 4,
    max_total: int = 8,
) -> List[Tuple[str, float]]:
    """Top severity + site lifts for compact all-mode panel (prefixed labels)."""
    sev = [
        (f"sev: {term}", lift)
        for term, lift, _, _ in compute_term_lifts(
            rows,
            SEVERITY_FLIP_TERMS,
            max_terms=per_category,
            min_present=TERM_LIFT_MIN_PRESENT,
        )
    ]
    site = [
        (f"site: {term}", lift)
        for term, lift, _, _ in compute_term_lifts(
            rows, SITE_SWAP_TERMS, max_terms=per_category, min_present=TERM_LIFT_MIN_PRESENT
        )
    ]
    combined = sev + site
    combined.sort(key=lambda x: abs(x[1]), reverse=True)
    return combined[:max_total]


def plot_term_presence_lift(
    rows: Sequence[Dict[str, str]],
    terms: Sequence[str],
    output_path: Path,
    *,
    title: str,
    max_terms: Optional[int] = None,
    min_present: int = TERM_LIFT_MIN_PRESENT,
) -> None:
    """Bar chart: Δ mean P(inconsistent) when term present vs absent in full report."""
    lifts = compute_term_lifts(rows, terms, max_terms=max_terms, min_present=min_present)
    if not lifts:
        print(f"Skipping term-lift plot (insufficient data): {output_path}")
        return

    terms_plot = [t for t, _, _, _ in lifts][::-1]
    values = [v for _, v, _, _ in lifts][::-1]
    counts = [(np, na) for _, _, np, na in lifts][::-1]

    fig_h = max(3.5, 0.32 * len(terms_plot) + 1.2)
    fig, ax = plt.subplots(figsize=(7.4, fig_h))
    y_pos = range(len(terms_plot))
    colors = ["#c62828" if v > 0 else "#1565c0" for v in values]
    bars = ax.barh(y_pos, values, color=colors, edgecolor="#333333", linewidth=0.4)
    ax.axvline(0, color="#666", linewidth=0.8)
    ax.set_yticks(y_pos, labels=terms_plot, fontsize=9)
    ax.set_xlabel("Δ mean P(inconsistent) when term in report (present − absent)", fontsize=9)
    ax.set_title(title, fontsize=11, pad=8)
    ax.grid(axis="x", alpha=0.25, color=SUMMARY_GRID_COLOR)
    span = max((abs(v) for v in values), default=0.05)
    x_pad = span * 0.04
    for bar, val, (n_pres, n_abs) in zip(bars, values, counts):
        label = f"n+{n_pres}, n−{n_abs}"
        if val >= 0:
            ax.text(
                bar.get_width() + x_pad,
                bar.get_y() + bar.get_height() / 2,
                label,
                va="center",
                ha="left",
                fontsize=7,
                color="#444",
            )
        else:
            ax.text(
                bar.get_width() - x_pad,
                bar.get_y() + bar.get_height() / 2,
                label,
                va="center",
                ha="right",
                fontsize=7,
                color="#444",
            )
    ax.set_xlim(_xlim_for_signed_lift_bars(values, lifts))
    plt.tight_layout()
    _save_figure(fig, output_path, dpi=160)
    plt.close(fig)
    print(f"Saved term-lift plot to: {output_path}")
    print(f"Saved term-lift plot to: {output_path.with_suffix('.pdf')}")


def _draw_laterality_panel(ax: Axes, stats: Dict[str, Dict[str, Tuple[float, int]]]):
    pattern_order = ("left only", "right only", "both left & right", "neither")
    label_order = ("consistent", "inconsistent")
    matrix = []
    for label in label_order:
        row_vals = []
        for pattern in pattern_order:
            mean, _n = stats.get(pattern, {}).get(label, (float("nan"), 0))
            row_vals.append(mean if mean == mean else 0.0)
        matrix.append(row_vals)
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pattern_order)), labels=pattern_order, fontsize=7, rotation=15, ha="right")
    ax.set_yticks(range(len(label_order)), labels=[f"true {l}" for l in label_order], fontsize=8)
    ax.set_title("Laterality in full report", fontsize=9)
    for i in range(len(label_order)):
        for j in range(len(pattern_order)):
            mean, n = stats.get(pattern_order[j], {}).get(label_order[i], (float("nan"), 0))
            txt = f"{mean:.2f}\nn={n}" if n else "—"
            color = "white" if matrix[i][j] > 0.55 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=color)
    return im


def plot_summary_panel(
    ranked_ig: Sequence[Tuple[str, float, int]],
    output_path: Path,
    *,
    mode: str,
    n_rows: int,
    pattern_stats: Optional[Dict[str, Dict[str, Tuple[float, int]]]] = None,
    term_lifts: Optional[Sequence[Tuple[str, float]]] = None,
    term_lift_title: Optional[str] = None,
    ig_value: str = "total",
) -> None:
    """Single-row paper figure: top IG words + keyword-context inset."""
    if not ranked_ig:
        return
    fig = plt.figure(figsize=(10.5, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.35)
    ax_words = fig.add_subplot(gs[0, 0])
    words = [w for w, _, _ in ranked_ig][::-1]
    bar_vals = _bar_values(ranked_ig, value=ig_value)[::-1]
    counts = [c for _, _, c in ranked_ig][::-1]
    bars = ax_words.barh(
        range(len(words)), bar_vals, color=SUMMARY_BAR_COLOR, edgecolor="#e65100", linewidth=0.4
    )
    ax_words.set_yticks(range(len(words)), labels=words, fontsize=9)
    if ig_value == "total":
        ax_words.set_xlabel("total IG (mean × n, → inconsistent)", fontsize=9)
    else:
        ax_words.set_xlabel("mean IG (full report, → inconsistent)", fontsize=9)
    ax_words.set_title(f"Top {len(words)} words ({n_rows} sampled reports)", fontsize=10)
    ax_words.grid(axis="x", alpha=0.25)
    x_pad = max(bar_vals) * 0.02 if bar_vals else 0.0
    for bar, n in zip(bars, counts):
        ax_words.text(
            bar.get_width() + x_pad,
            bar.get_y() + bar.get_height() / 2,
            f"n={n}",
            va="center",
            ha="left",
            fontsize=7,
            color="#444",
        )
    ax_words.set_xlim(0, _xlim_for_bar_annotations(bar_vals, counts))

    ax_right = fig.add_subplot(gs[0, 1])
    if mode == "lat" and pattern_stats:
        im = _draw_laterality_panel(ax_right, pattern_stats)
        fig.colorbar(im, ax=ax_right, fraction=0.046, pad=0.04, label="P(inc)")
    elif term_lifts:
        terms = [t for t, _ in term_lifts][::-1]
        vals = [v for _, v in term_lifts][::-1]
        colors = ["#c62828" if v > 0 else "#1565c0" for v in vals]
        ax_right.barh(range(len(terms)), vals, color=colors, edgecolor="#333", linewidth=0.3)
        ax_right.axvline(0, color="#666", linewidth=0.7)
        ax_right.set_yticks(range(len(terms)), labels=terms, fontsize=8)
        ax_right.set_xlabel("Δ P(inc) present−absent", fontsize=8)
        ax_right.set_title(term_lift_title or f"{mode} term presence", fontsize=9)
        ax_right.grid(axis="x", alpha=0.25)
    else:
        ax_right.axis("off")
        ax_right.text(
            0.5,
            0.5,
            "Pattern panel for\nlat / severity / site modes",
            ha="center",
            va="center",
            fontsize=9,
            color="#666",
        )

    fig.suptitle(f"Attribution summary ({mode})", fontsize=12, y=1.02)
    _save_figure(fig, output_path, dpi=160)
    plt.close(fig)
    print(f"Saved combined summary panel to: {output_path}")
    print(f"Saved combined summary panel to: {output_path.with_suffix('.pdf')}")


def write_attribution_summary_tables(
    output_dir: Path,
    *,
    mode: str,
    n_sampled: int,
    top_ig: Sequence[Tuple[str, float, int]],
    top_cls: Sequence[Tuple[str, float, int]],
    pattern_stats: Optional[Dict[str, Dict[str, Tuple[float, int]]]] = None,
    top_ig_decision: Optional[Sequence[Tuple[str, float, int]]] = None,
) -> None:
    lines = [
        "Attribution aggregate summary",
        f"eval_mode: {mode}",
        f"sampled_reports: {n_sampled}",
        f"ig_target: inconsistent (label=0)",
        f"ig_decision_target: model prediction per report",
        f"section: full report (findings + conclusion)",
        "",
        "term_lift: for each keyword, mean P(inconsistent) when the term appears",
        "anywhere in the report minus when it does not (from eval predictions CSV).",
        "This is behavioural correlation, not IG/CLS attribution.",
        "",
        f"Top {SUMMARY_TOP_N} words by total IG (mean × n):",
        f"{'word':<20} {'mean_ig':>10} {'n':>8} {'total_ig':>12}",
    ]
    for word, mean, n in top_ig:
        lines.append(f"{word:<20} {mean:10.6f} {n:8d} {mean * n:12.6f}")
    if top_ig_decision:
        lines.extend(["", f"Top {SUMMARY_TOP_N} words by total IG toward predicted class (mean × n):"])
        lines.append(f"{'word':<20} {'mean_ig':>10} {'n':>8} {'total_ig':>12}")
        for word, mean, n in top_ig_decision:
            lines.append(f"{word:<20} {mean:10.6f} {n:8d} {mean * n:12.6f}")
    lines.extend(["", f"Top {SUMMARY_TOP_N} words by total CLS attention (mean × n):"])
    lines.append(f"{'word':<20} {'mean_cls':>10} {'n':>8} {'total_cls':>12}")
    for word, mean, n in top_cls:
        lines.append(f"{word:<20} {mean:10.6f} {n:8d} {mean * n:12.6f}")

    if pattern_stats:
        lines.extend(["", "Laterality pattern × true label (mean P inconsistent):"])
        for pattern, by_label in pattern_stats.items():
            parts = [
                f"{label}={mean:.3f}(n={n})" for label, (mean, n) in sorted(by_label.items())
            ]
            lines.append(f"  {pattern}: " + ", ".join(parts))

    txt_path = output_dir / "attribution_summary.txt"
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    csv_path = output_dir / "attribution_summary_top_words.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["word", "mean_ig_inc", "n_ig_inc", "total_ig_inc", "mean_ig_dec", "n_ig_dec", "total_ig_dec", "mean_cls", "n_cls", "total_cls"]
        )
        cls_map = {w: (m, n) for w, m, n in top_cls}
        ig_map = {w: (m, n) for w, m, n in top_ig}
        ig_dec_map = {w: (m, n) for w, m, n in (top_ig_decision or [])}
        all_words = list(
            dict.fromkeys(
                [w for w, _, _ in top_ig]
                + [w for w, _, _ in (top_ig_decision or [])]
                + [w for w, _, _ in top_cls]
            )
        )
        for word in all_words:
            ig_m, ig_n = ig_map.get(word, ("", ""))
            ig_dec_m, ig_dec_n = ig_dec_map.get(word, ("", ""))
            cls_m, cls_n = cls_map.get(word, ("", ""))
            total_ig = ig_m * ig_n if ig_m != "" and ig_n != "" else ""
            total_ig_dec = ig_dec_m * ig_dec_n if ig_dec_m != "" and ig_dec_n != "" else ""
            total_cls = cls_m * cls_n if cls_m != "" and cls_n != "" else ""
            writer.writerow([word, ig_m, ig_n, total_ig, ig_dec_m, ig_dec_n, total_ig_dec, cls_m, cls_n, total_cls])
    print(f"Saved attribution summary tables to: {txt_path}")


def run_attribution_summary(
    model: object,
    rows: Sequence[Dict[str, str]],
    output_dir: Path,
    *,
    mode: str,
    ig_steps: int = 50,
    max_rows: int = 120,
    seed: int = 42,
) -> None:
    """Condensed paper-style aggregates under attention_attribution_<mode>/."""
    if max_rows <= 0 or not rows:
        return

    rng = random.Random(seed)
    buckets = bucket_rows(list(rows))
    sampled = sample_rows_for_summary(buckets, max_rows, rng)
    if not sampled:
        print("Skipping attribution summary: no rows to sample.")
        return

    print(
        f"\n=== Attribution summary (mode={mode}, n={len(sampled)}/{len(rows)} reports) ==="
    )

    ig_stats, ig_decision_stats, cls_stats, bucket_ig, bucket_cls = compute_attribution_summaries(
        model, sampled, ig_steps=ig_steps
    )
    top_ig = top_n_words(ig_stats, SUMMARY_TOP_N, rank_by="total")
    top_ig_decision = top_n_words(ig_decision_stats, SUMMARY_TOP_N, rank_by="total")
    top_cls = top_n_words(cls_stats, SUMMARY_TOP_N, rank_by="total")

    sampled_bucket_counts = {
        name: sum(1 for row in sampled if _outcome_bucket(row) == name)
        for name, _, _ in BUCKET_SPECS
    }
    plot_confusion_section_top_words_grid(
        bucket_ig,
        output_dir / "summary_confusion_top_words_ig.png",
        method_label="IG → inconsistent",
        bucket_counts=sampled_bucket_counts,
    )
    plot_confusion_section_top_words_grid(
        bucket_cls,
        output_dir / "summary_confusion_top_words_cls.png",
        method_label="CLS attention",
        bucket_counts=sampled_bucket_counts,
    )

    plot_top_words_bar(
        top_ig,
        output_dir / "summary_top_words_ig.png",
        title=f"Top {SUMMARY_TOP_N} report words — IG → inconsistent",
        method_label="IG",
        value="total",
    )
    plot_top_words_bar(
        top_ig_decision,
        output_dir / "summary_top_words_ig_decision.png",
        title=f"Top {SUMMARY_TOP_N} report words — IG → predicted class",
        method_label="IG",
        value="total",
    )
    plot_top_words_bar(
        top_cls,
        output_dir / "summary_top_words_cls.png",
        title=f"Top {SUMMARY_TOP_N} report words — CLS attention",
        method_label="CLS attention",
        value="total",
    )

    pattern_stats = None
    term_lifts: Optional[List[Tuple[str, float]]] = None
    term_lift_title: Optional[str] = None
    if mode == "lat":

        def lat_pattern(row: Dict[str, str]) -> str:
            return _binary_pair_pattern(_report_text(row), "left", "right")

        pattern_stats = _mean_prob_by_pattern(sampled, lat_pattern)
        plot_laterality_pattern_heatmap(
            sampled,
            output_dir / "summary_laterality_patterns.png",
            title="Report laterality vs P(inconsistent)",
        )
    elif mode in {"severity", "site"}:
        terms = SEVERITY_FLIP_TERMS if mode == "severity" else SITE_SWAP_TERMS
        plot_term_presence_lift(
            sampled,
            terms,
            output_dir / f"summary_{mode}_term_lift.png",
            title=f"Report {mode} terms — present vs absent (n+ ≥ {TERM_LIFT_MIN_PRESENT})",
        )
        term_lifts = [
            (term, lift)
            for term, lift, _, _ in compute_term_lifts(
                sampled, terms, max_terms=8, min_present=TERM_LIFT_MIN_PRESENT
            )
        ]
    elif mode in {"all", "hard", "cross_pair_full"}:
        plot_term_presence_lift(
            sampled,
            SEVERITY_FLIP_TERMS,
            output_dir / "summary_severity_term_lift.png",
            title=f"Report severity terms — present vs absent (n+ ≥ {TERM_LIFT_MIN_PRESENT})",
        )
        plot_term_presence_lift(
            sampled,
            SITE_SWAP_TERMS,
            output_dir / "summary_site_term_lift.png",
            title=f"Report site terms — present vs absent (n+ ≥ {TERM_LIFT_MIN_PRESENT})",
        )
        term_lifts = combined_swap_term_lifts(sampled)
        term_lift_title = "severity + site term presence"

    plot_summary_panel(
        top_ig,
        output_dir / "summary_compact_panel.png",
        mode=mode,
        n_rows=len(sampled),
        pattern_stats=pattern_stats,
        term_lifts=term_lifts,
        term_lift_title=term_lift_title,
        ig_value="total",
    )

    write_attribution_summary_tables(
        output_dir,
        mode=mode,
        n_sampled=len(sampled),
        top_ig=top_ig,
        top_cls=top_cls,
        pattern_stats=pattern_stats,
        top_ig_decision=top_ig_decision,
    )

    if mode in {"all", "hard", "cross_pair_full"}:
        per_group_max = max(10, max_rows // len(ERROR_TYPE_SUMMARY_GROUPS))
        error_type_samples = sample_rows_for_error_type_groups(rows, per_group_max, rng)
        error_type_counts = {
            key: len(error_type_samples.get(key) or [])
            for key, _, _ in ERROR_TYPE_SUMMARY_GROUPS
        }

        cls_by_group = compute_error_type_word_summaries(
            model,
            error_type_samples,
            attribution="cls",
            ig_steps=ig_steps,
            method_label="CLS attention",
        )
        ranked_cls = {
            key: top_n_words(stats, SUMMARY_TOP_N, rank_by="total")
            for key, stats in cls_by_group.items()
        }
        plot_error_type_top_words_grid(
            ranked_cls,
            error_type_counts,
            output_dir / "summary_top_words_cls_by_error_type.png",
            method_label="CLS attention",
            value="total",
        )

        ig_by_group = compute_error_type_word_summaries(
            model,
            error_type_samples,
            attribution="ig",
            ig_steps=ig_steps,
            ig_target_label=0,
            method_label="IG → inconsistent",
        )
        ranked_ig = {
            key: top_n_words(stats, SUMMARY_TOP_N, rank_by="total")
            for key, stats in ig_by_group.items()
        }
        plot_error_type_top_words_grid(
            ranked_ig,
            error_type_counts,
            output_dir / "summary_top_words_ig_by_error_type.png",
            method_label="IG → inconsistent",
            value="total",
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions_csv", type=str, required=True)
    p.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Model directory (default: <run>/model inferred from predictions CSV).",
    )
    p.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Optional run root; used to locate plots/ when output_dir is omitted.",
    )
    p.add_argument(
        "--mode",
        type=str,
        default="hard",
        choices=["all", "hard", "lat", "site", "severity", "cross_pair_full"],
    )
    p.add_argument(
        "--method",
        type=str,
        default="both",
        choices=["cls", "ig", "both"],
        help="Attribution method: cls, ig (Integrated Gradients), or both.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for PNG files (default: <run>/plots/attention_attribution_<mode>/).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_tp", type=int, default=2)
    p.add_argument("--n_tn", type=int, default=2)
    p.add_argument("--n_fp", type=int, default=2)
    p.add_argument("--n_fn", type=int, default=2)
    p.add_argument("--dpi", type=int, default=140)
    p.add_argument(
        "--ig_steps",
        type=int,
        default=50,
        help="Riemann steps for Integrated Gradients (method ig or both).",
    )
    p.add_argument(
        "--summary_only",
        action="store_true",
        help="Skip per-example heatmaps; only write aggregate summary plots.",
    )
    p.add_argument(
        "--skip_summary",
        action="store_true",
        help="Skip condensed top-word / pattern summary plots.",
    )
    p.add_argument(
        "--summary_max_rows",
        type=int,
        default=120,
        help="Max eval reports for aggregate IG/CLS summaries (stratified TP/TN/FP/FN).",
    )
    return p.parse_args()


def _methods_to_run(method: str) -> List[str]:
    if method == "both":
        return ["cls", "ig"]
    return [method]


def run_attention_attribution(
    predictions_csv: str,
    model_dir: str,
    *,
    mode: str = "hard",
    output_dir: Optional[str] = None,
    run_dir: Optional[str] = None,
    method: str = "both",
    seed: int = 42,
    n_tp: int = 2,
    n_tn: int = 2,
    n_fp: int = 2,
    n_fn: int = 2,
    dpi: int = 140,
    ig_steps: int = 50,
    run_summary: bool = True,
    summary_max_rows: int = 120,
    summary_only: bool = False,
) -> Path:
    """Sample TP/TN/FP/FN rows (filtered by eval mode) and write attribution PNGs."""
    predictions_path = Path(predictions_csv).resolve()
    model_path = Path(model_dir).resolve()
    filter_mode = attribution_row_filter_mode(mode)

    layout = resolve_run_layout_from_artifact(predictions_path)
    methods = _methods_to_run(method)
    if output_dir:
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = resolve_output_dir(
            predictions_path,
            model_path,
            mode,
            None,
            run_dir,
        )
    run_name = layout.root.name if layout else out_dir.parent.name

    init_tokenizer(TOKENIZER_PATH)
    rows = load_prediction_rows(predictions_path, mode=filter_mode)
    if not rows:
        raise ValueError(
            f"No rows left after mode={filter_mode!r} filter in {predictions_path}"
        )

    buckets = bucket_rows(rows)
    rng = random.Random(seed)
    counts = {"tp": n_tp, "tn": n_tn, "fp": n_fp, "fn": n_fn}
    picked: List[Tuple[str, str, Dict[str, str]]] = []
    if not summary_only:
        picked = collect_examples(buckets, counts, rng)

    for name, _, _ in BUCKET_SPECS:
        need = counts[name]
        have = len(buckets[name])
        if need > have and need > 0:
            print(f"Warning: only {have} {name.upper()} examples available (requested {need}).")

    print(f"\n=== Attention attribution (mode={mode}, filter={filter_mode}) ===")
    print(f"Run: {run_name}")
    print(f"Predictions: {predictions_path}")
    print(f"Model: {model_path}")
    print(f"Output: {out_dir}")
    print(
        f"Loaded {len(rows)} rows, plotting {len(picked)} examples "
        f"x {len(methods)} method(s): {', '.join(methods)}."
    )
    model = load_consistency_classifier(str(model_path))

    written: List[Path] = []
    for eid, title, row in picked:
        for attn_method in methods:
            out = explain_consistency(
                model,
                row["findings"],
                row["conclusion"],
                attribution=attn_method,
                ig_steps=ig_steps,
            )
            png_path = out_dir / f"{eid}_{attn_method}.png"
            plot_example_attention(title, row, out, png_path, dpi=dpi)
            written.append(png_path)
            print(
                f"  {eid}_{attn_method}: {row.get('error_type')} "
                f"true={row.get('true_label_name')} pred={out['prediction']} -> {png_path.name}"
            )

    summary_files: List[str] = []
    if run_summary and mode in {"lat", "site", "severity", "hard", "all", "cross_pair_full"}:
        run_attribution_summary(
            model,
            rows,
            out_dir,
            mode=mode,
            ig_steps=ig_steps,
            max_rows=summary_max_rows,
            seed=seed,
        )
        summary_files = [
            "summary_top_words_ig.png",
            "summary_top_words_ig_decision.png",
            "summary_top_words_cls.png",
            "summary_confusion_top_words_ig.png",
            "summary_confusion_top_words_cls.png",
            "summary_compact_panel.png",
            "attribution_summary.txt",
            "attribution_summary_top_words.csv",
        ]
        if mode == "lat":
            summary_files.append("summary_laterality_patterns.png")
        elif mode in {"severity", "site"}:
            summary_files.append(f"summary_{mode}_term_lift.png")
        elif mode in {"all", "hard", "cross_pair_full"}:
            summary_files.extend(
                [
                    "summary_severity_term_lift.png",
                    "summary_site_term_lift.png",
                    "summary_top_words_cls_by_error_type.png",
                    "summary_top_words_ig_by_error_type.png",
                ]
            )

    write_attribution_manifest(
        out_dir,
        run_name=run_name,
        predictions_csv=predictions_path,
        model_dir=model_path,
        mode=mode,
        method=method,
        ig_steps=ig_steps,
        plot_files=[p.name for p in written],
    )
    manifest_path = out_dir / "attention_attribution_source.txt"
    if summary_files:
        with open(manifest_path, "a", encoding="utf-8") as f:
            f.write("\nSummary aggregates:\n")
            for name in summary_files:
                f.write(f"  {name}\n")
    print(f"Wrote {len(written)} PNGs to {out_dir}")
    print(f"Manifest: {manifest_path}")
    return out_dir


def main() -> None:
    args = parse_args()
    predictions_csv = Path(args.predictions_csv).resolve()

    layout = resolve_run_layout_from_artifact(predictions_csv)
    if args.model_dir:
        model_dir = Path(args.model_dir).resolve()
    elif layout is not None:
        model_dir = layout.model.resolve()
    else:
        raise ValueError(
            "Could not infer model_dir from predictions_csv; pass --model_dir explicitly."
        )

    run_attention_attribution(
        str(predictions_csv),
        str(model_dir),
        mode=args.mode,
        output_dir=args.output_dir,
        run_dir=args.run_dir,
        method=args.method,
        seed=args.seed,
        n_tp=args.n_tp,
        n_tn=args.n_tn,
        n_fp=args.n_fp,
        n_fn=args.n_fn,
        dpi=args.dpi,
        ig_steps=args.ig_steps,
        run_summary=not args.skip_summary,
        summary_max_rows=args.summary_max_rows,
        summary_only=args.summary_only,
    )


if __name__ == "__main__":
    main()
