#!/usr/bin/env python3
"""Lexical-overlap null models for findings/conclusion inconsistency detection.

Baselines (no learned parameters except optional Jaccard threshold on val):

  keyword  – flag pair-side / severity-tier contradiction using the same term
             families used to construct lat/site/severity corruptions
  overlap  – bag-of-words Jaccard; inconsistent if similarity < threshold
  union    – inconsistent if keyword or low-overlap fires

Writes BERT-compatible hard_predictions.csv and eval summaries via predictions_eval.

Examples (repo root, PTCT3):

  python scripts/eval_lexical_null.py

  python scripts/eval_lexical_null.py \\
    --eval_csv data/eval_datasets/hard_negative_eval_v6_single_swap.csv \\
    --modes keyword,overlap,union
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paths import (  # noqa: E402
    eval_data_path,
    setup_run,
)
from src.bert_inconsistency_detector import (  # noqa: E402
    LATERALITY_SWAP_RULES,
    SITE_SWAP_RULES,
)
from src.predictions_eval import (  # noqa: E402
    write_eval_artifacts,
)
from src.severity_flip import (  # noqa: E402
    SEVERITY_SCALAR_RULES,
    SEVERITY_TIERS,
)

SCRIPT_ID = "lexical-null"
# No dedicated null-model run category; land under llm_baseline (rule baselines).
RUN_CATEGORY = "llm_baseline"
DEFAULT_EVAL = "hard_negative_eval_v6_single_swap.csv"
DEFAULT_VAL = "val_hard_negative_eval.csv"
DEFAULT_MODES = ("keyword", "overlap", "union")

TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?", re.IGNORECASE)
WORD_RE = re.compile(r"\b[a-z0-9][a-z0-9\-]*\b", re.IGNORECASE)

# 1 = consistent, 0 = inconsistent (matches BERT binary labels)
CONSISTENT = 1
INCONSISTENT = 0


def _pair_rules() -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()
    for rules in (LATERALITY_SWAP_RULES, SITE_SWAP_RULES, SEVERITY_SCALAR_RULES):
        for left, right, _ in rules:
            key = (left.lower(), right.lower())
            if key not in seen and (key[1], key[0]) not in seen:
                seen.add(key)
                pairs.append((left, right))
    return pairs


PAIRS = _pair_rules()


@dataclass(frozen=True)
class NullScores:
    """Scores and decision for one findings/conclusion pair."""

    jaccard: float
    keyword_incongruent: bool
    n_pair_conflicts: int
    severity_conflict: bool
    score_inconsistent: float  # higher => more inconsistent
    pred_label: int


def tokens(text: str) -> Set[str]:
    return {m.group(0).lower() for m in TOKEN_RE.finditer(text or "")}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def count_term(text: str, term: str) -> int:
    return len(re.findall(rf"\b{re.escape(term)}\b", text or "", flags=re.IGNORECASE))


def side_of_pair(text: str, a: str, b: str) -> Optional[str]:
    """Dominant side of an antonym pair, or None if neither term appears."""
    ca, cb = count_term(text, a), count_term(text, b)
    if ca == 0 and cb == 0:
        return None
    if ca > cb:
        return "a"
    if cb > ca:
        return "b"
    return "tie"


def pair_conflicts(findings: str, conclusion: str, pairs: Sequence[Tuple[str, str]] = PAIRS) -> int:
    n = 0
    for a, b in pairs:
        f_side = side_of_pair(findings, a, b)
        c_side = side_of_pair(conclusion, a, b)
        if f_side in {"a", "b"} and c_side in {"a", "b"} and f_side != c_side:
            n += 1
    return n


def severity_tier_indices(text: str) -> Set[int]:
    present: Set[int] = set()
    for i, bag in enumerate(SEVERITY_TIERS):
        for term in bag:
            if count_term(text, term) > 0:
                present.add(i)
                break
    return present


def severity_tier_conflict(findings: str, conclusion: str) -> bool:
    """Cross-tier severity mismatch between sections (mild vs severe family, etc.)."""
    f_tiers = severity_tier_indices(findings)
    c_tiers = severity_tier_indices(conclusion)
    if not f_tiers or not c_tiers:
        return False
    return f_tiers.isdisjoint(c_tiers)


def keyword_incongruent(findings: str, conclusion: str) -> Tuple[bool, int, bool]:
    n_pairs = pair_conflicts(findings, conclusion)
    sev = severity_tier_conflict(findings, conclusion)
    return (n_pairs > 0 or sev), n_pairs, sev


def score_pair(
    findings: str,
    conclusion: str,
    *,
    mode: str,
    jaccard_threshold: float,
) -> NullScores:
    jac = jaccard(tokens(findings), tokens(conclusion))
    kw, n_pairs, sev = keyword_incongruent(findings, conclusion)
    low_overlap = jac < jaccard_threshold

    if mode == "keyword":
        pred = INCONSISTENT if kw else CONSISTENT
        score_inc = float(n_pairs + int(sev))
    elif mode == "overlap":
        pred = INCONSISTENT if low_overlap else CONSISTENT
        score_inc = 1.0 - jac
    elif mode == "union":
        pred = INCONSISTENT if (kw or low_overlap) else CONSISTENT
        # Prefer keyword evidence; fall back to 1-jac for ranking.
        score_inc = max(float(n_pairs + int(sev)), 1.0 - jac if low_overlap else 0.0)
    else:
        raise ValueError(f"Unknown mode {mode!r}")

    return NullScores(
        jaccard=jac,
        keyword_incongruent=kw,
        n_pair_conflicts=n_pairs,
        severity_conflict=sev,
        score_inconsistent=score_inc,
        pred_label=pred,
    )


def normalize_true_label(row: Dict[str, str]) -> int:
    if "label" in row and str(row["label"]).strip() != "":
        return int(float(row["label"]))
    if "true_label" in row and str(row["true_label"]).strip() != "":
        return int(float(row["true_label"]))
    et = (row.get("error_type") or "").strip().lower()
    if et in {"consistent", "positive"}:
        return CONSISTENT
    return INCONSISTENT


def normalize_error_type(error_type: str) -> str:
    et = (error_type or "unknown").strip()
    if et == "positive":
        return "consistent"
    return et


def load_eval_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def balanced_accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    # recall on inconsistent (0) and consistent (1)
    tp0 = fp0 = fn0 = tn0 = 0
    for t, p in zip(y_true, y_pred):
        if t == INCONSISTENT and p == INCONSISTENT:
            tp0 += 1
        elif t == CONSISTENT and p == INCONSISTENT:
            fp0 += 1
        elif t == INCONSISTENT and p == CONSISTENT:
            fn0 += 1
        else:
            tn0 += 1
    rec0 = tp0 / (tp0 + fn0) if (tp0 + fn0) else 0.0
    rec1 = tn0 / (tn0 + fp0) if (tn0 + fp0) else 0.0
    return 0.5 * (rec0 + rec1)


def calibrate_jaccard_threshold(
    rows: Sequence[Dict[str, str]],
    grid: Optional[Sequence[float]] = None,
) -> Tuple[float, float]:
    """Pick Jaccard threshold maximizing balanced accuracy on val rows."""
    if grid is None:
        grid = [i / 100.0 for i in range(1, 50)]  # 0.01 .. 0.49
    sims: List[float] = []
    labels: List[int] = []
    for r in rows:
        sims.append(jaccard(tokens(r.get("findings", "")), tokens(r.get("conclusion", ""))))
        labels.append(normalize_true_label(r))

    best_thr = 0.15
    best_ba = -1.0
    for thr in grid:
        preds = [INCONSISTENT if s < thr else CONSISTENT for s in sims]
        ba = balanced_accuracy(labels, preds)
        if ba > best_ba:
            best_ba = ba
            best_thr = thr
    return best_thr, best_ba


def predict_rows(
    rows: Sequence[Dict[str, str]],
    *,
    mode: str,
    jaccard_threshold: float,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for r in rows:
        findings = r.get("findings", "") or ""
        conclusion = r.get("conclusion", "") or ""
        true_label = normalize_true_label(r)
        scores = score_pair(
            findings,
            conclusion,
            mode=mode,
            jaccard_threshold=jaccard_threshold,
        )
        pred = scores.pred_label
        # Soft probs from score for compatibility with AUC tooling / ranking.
        # keyword: map conflict count; overlap: 1-jaccard; clip to (0,1).
        if mode == "keyword":
            p_inc = min(1.0, 0.5 + 0.25 * scores.n_pair_conflicts + 0.25 * int(scores.severity_conflict))
            if not scores.keyword_incongruent:
                p_inc = max(0.0, min(0.49, 1.0 - scores.jaccard))
        elif mode == "overlap":
            p_inc = 1.0 - scores.jaccard
        else:
            p_kw = min(1.0, 0.5 + 0.25 * scores.n_pair_conflicts + 0.25 * int(scores.severity_conflict))
            if not scores.keyword_incongruent:
                p_kw = 0.0
            p_inc = max(p_kw, 1.0 - scores.jaccard)

        out.append(
            {
                "patient_id": r.get("patient_id", ""),
                "source_idx": r.get("source_idx", ""),
                "findings": findings,
                "conclusion": conclusion,
                "error_type": normalize_error_type(r.get("error_type", "")),
                "incidence": r.get("incidence", ""),
                "true_label": true_label,
                "pred_label": pred,
                "true_label_name": "inconsistent" if true_label == INCONSISTENT else "consistent",
                "pred_label_name": "inconsistent" if pred == INCONSISTENT else "consistent",
                "prob_inconsistent": float(p_inc),
                "prob_consistent": float(1.0 - p_inc),
                "is_correct": int(pred == true_label),
                "is_tp0": int(true_label == INCONSISTENT and pred == INCONSISTENT),
                "is_fp0": int(true_label == CONSISTENT and pred == INCONSISTENT),
                "is_fn0": int(true_label == INCONSISTENT and pred == CONSISTENT),
                "is_tn0": int(true_label == CONSISTENT and pred == CONSISTENT),
                "jaccard": f"{scores.jaccard:.6f}",
                "keyword_incongruent": int(scores.keyword_incongruent),
                "n_pair_conflicts": scores.n_pair_conflicts,
                "severity_conflict": int(scores.severity_conflict),
            }
        )
    return out


PRED_FIELDS = [
    "patient_id",
    "source_idx",
    "findings",
    "conclusion",
    "error_type",
    "incidence",
    "true_label",
    "pred_label",
    "true_label_name",
    "pred_label_name",
    "prob_inconsistent",
    "prob_consistent",
    "is_correct",
    "is_tp0",
    "is_fp0",
    "is_fn0",
    "is_tn0",
    "jaccard",
    "keyword_incongruent",
    "n_pair_conflicts",
    "severity_conflict",
]


def write_predictions(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PRED_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_prediction_rows(
    rows: Sequence[Dict[str, object]],
    *,
    mode: str,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """Return (overall_row, by_error_type_rows) for paper tables / CSVs."""
    by_type: Dict[str, Counter] = {}
    overall = Counter()
    for r in rows:
        et = str(r["error_type"])
        t, p = int(r["true_label"]), int(r["pred_label"])
        c = by_type.setdefault(et, Counter())
        c["n"] += 1
        c["correct"] += int(t == p)
        overall["n"] += 1
        overall["correct"] += int(t == p)
        if t == INCONSISTENT:
            overall["n_inc"] += 1
            overall["hit_inc"] += int(p == INCONSISTENT)
            c["n_inc"] = c.get("n_inc", 0) + 1
            c["hit_inc"] = c.get("hit_inc", 0) + int(p == INCONSISTENT)
        else:
            overall["n_con"] += 1
            overall["hit_con"] += int(p == CONSISTENT)
            c["n_con"] = c.get("n_con", 0) + 1
            c["hit_con"] = c.get("hit_con", 0) + int(p == CONSISTENT)

    n = overall["n"] or 1
    n_inc = overall["n_inc"] or 0
    n_con = overall["n_con"] or 0
    acc = overall["correct"] / n
    rec_inc = (overall["hit_inc"] / n_inc) if n_inc else 0.0
    rec_con = (overall["hit_con"] / n_con) if n_con else 0.0
    fa = 1.0 - rec_con
    balanced = 0.5 * (rec_inc + rec_con)

    overall_row: Dict[str, object] = {
        "mode": mode,
        "n_total": overall["n"],
        "n_inconsistent": n_inc,
        "n_consistent": n_con,
        "accuracy": round(acc, 6),
        "accuracy_pct": round(100 * acc, 2),
        "inconsistent_recall": round(rec_inc, 6),
        "inconsistent_recall_pct": round(100 * rec_inc, 2),
        "consistent_recall": round(rec_con, 6),
        "consistent_recall_pct": round(100 * rec_con, 2),
        "false_alarm_rate": round(fa, 6),
        "false_alarm_rate_pct": round(100 * fa, 2),
        "balanced_accuracy": round(balanced, 6),
        "balanced_accuracy_pct": round(100 * balanced, 2),
    }

    by_rows: List[Dict[str, object]] = []
    for et in sorted(by_type):
        c = by_type[et]
        n_et = c["n"]
        et_acc = c["correct"] / n_et if n_et else 0.0
        if c.get("n_inc", 0):
            detection = c["hit_inc"] / c["n_inc"]
            metric_name = "detection_rate"
            metric_val = detection
        else:
            detection = (c["hit_con"] / c["n_con"]) if c.get("n_con") else 0.0
            metric_name = "consistent_recall"
            metric_val = detection
        by_rows.append(
            {
                "mode": mode,
                "error_type": et,
                "n": n_et,
                "n_inconsistent": c.get("n_inc", 0),
                "n_consistent": c.get("n_con", 0),
                "accuracy": round(et_acc, 6),
                "accuracy_pct": round(100 * et_acc, 2),
                "metric": metric_name,
                "class_recall": round(metric_val, 6),
                "class_recall_pct": round(100 * metric_val, 2),
                "detected": c.get("hit_inc", 0) if c.get("n_inc", 0) else c.get("hit_con", 0),
            }
        )
    return overall_row, by_rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csvs(
    results_dir: Path,
    *,
    mode: str,
    overall_row: Dict[str, object],
    by_rows: Sequence[Dict[str, object]],
) -> None:
    """Write per-mode overall / by-error-type CSVs under results/ and mode/results/."""
    overall_fields = list(overall_row.keys())
    by_fields = list(by_rows[0].keys()) if by_rows else [
        "mode",
        "error_type",
        "n",
        "n_inconsistent",
        "n_consistent",
        "accuracy",
        "accuracy_pct",
        "metric",
        "class_recall",
        "class_recall_pct",
        "detected",
    ]
    write_csv(results_dir / f"metrics_overall_{mode}.csv", [overall_row], overall_fields)
    write_csv(results_dir / f"metrics_by_error_type_{mode}.csv", by_rows, by_fields)
    mode_dir = results_dir.parent / mode / "results"
    write_csv(mode_dir / "metrics_overall.csv", [overall_row], overall_fields)
    write_csv(mode_dir / "metrics_by_error_type.csv", by_rows, by_fields)


def write_comparison_csvs(
    results_dir: Path,
    overall_rows: Sequence[Dict[str, object]],
    by_rows: Sequence[Dict[str, object]],
) -> None:
    """Combined tables across modes (the useful paper-facing summaries)."""
    if overall_rows:
        write_csv(
            results_dir / "metrics_overall.csv",
            overall_rows,
            list(overall_rows[0].keys()),
        )
    if by_rows:
        write_csv(
            results_dir / "metrics_by_error_type.csv",
            by_rows,
            list(by_rows[0].keys()),
        )
        # Wide pivot: one row per error_type, columns mode_class_recall_pct
        modes = sorted({str(r["mode"]) for r in by_rows})
        ets = sorted({str(r["error_type"]) for r in by_rows})
        by_map = {(str(r["mode"]), str(r["error_type"])): r for r in by_rows}
        wide: List[Dict[str, object]] = []
        for et in ets:
            row: Dict[str, object] = {"error_type": et}
            n = None
            for m in modes:
                r = by_map.get((m, et))
                if r is None:
                    row[f"{m}_class_recall_pct"] = ""
                    row[f"{m}_n"] = ""
                else:
                    row[f"{m}_class_recall_pct"] = r["class_recall_pct"]
                    row[f"{m}_n"] = r["n"]
                    n = r["n"]
            row["n"] = n if n is not None else ""
            wide.append(row)
        wide_fields = ["error_type", "n"] + [
            f for m in modes for f in (f"{m}_class_recall_pct", f"{m}_n")
        ]
        write_csv(results_dir / "metrics_by_error_type_wide.csv", wide, wide_fields)


def print_quick_table(
    overall_row: Dict[str, object],
    by_rows: Sequence[Dict[str, object]],
    title: str,
) -> None:
    print(f"\n=== {title} ===")
    print(
        f"overall acc={overall_row['accuracy_pct']}%  "
        f"inconsistent recall={overall_row['inconsistent_recall_pct']}%  "
        f"consistent recall={overall_row['consistent_recall_pct']}%  "
        f"false-alarm rate={overall_row['false_alarm_rate_pct']}%"
    )
    print(f"{'error_type':>22} {'n':>6} {'acc':>8} {'class_recall':>14}")
    for r in by_rows:
        print(
            f"{str(r['error_type']):>22} {int(r['n']):6d} "
            f"{float(r['accuracy_pct']):7.1f}% "
            f"{float(r['class_recall_pct']):13.1f}%"
        )


def parse_modes(s: str) -> List[str]:
    modes = [m.strip().lower() for m in s.split(",") if m.strip()]
    for m in modes:
        if m not in {"keyword", "overlap", "union"}:
            raise ValueError(f"Unknown mode {m!r}; choose keyword, overlap, union")
    return modes


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--eval_csv",
        type=str,
        default=eval_data_path(DEFAULT_EVAL),
        help="Hard eval CSV (default: paper v6 single-swap).",
    )
    p.add_argument(
        "--val_csv",
        type=str,
        default=eval_data_path(DEFAULT_VAL),
        help="Val CSV for Jaccard threshold calibration (empty to skip).",
    )
    p.add_argument(
        "--modes",
        type=str,
        default=",".join(DEFAULT_MODES),
        help="Comma-separated: keyword,overlap,union",
    )
    p.add_argument(
        "--jaccard_threshold",
        type=float,
        default=None,
        help="Fixed Jaccard thr for overlap/union. Default: calibrate on --val_csv.",
    )
    p.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional fixed run name (else timestamped).",
    )
    p.add_argument(
        "--output_run_dir",
        type=str,
        default=None,
        help="Optional fixed output root under trained_models/runs/.",
    )
    p.add_argument(
        "--no_summary",
        action="store_true",
        help="Skip write_eval_artifacts full summary (still writes preds + quick table).",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    modes = parse_modes(args.modes)
    eval_path = Path(args.eval_csv)
    if not eval_path.is_file():
        raise FileNotFoundError(f"Eval CSV not found: {eval_path}")

    eval_rows = load_eval_rows(eval_path)
    print(f"Loaded eval: {eval_path} ({len(eval_rows)} rows)")

    thr = args.jaccard_threshold
    thr_ba = None
    if thr is None and any(m in {"overlap", "union"} for m in modes):
        val_path = Path(args.val_csv) if args.val_csv else None
        if val_path and val_path.is_file():
            val_rows = load_eval_rows(val_path)
            thr, thr_ba = calibrate_jaccard_threshold(val_rows)
            print(
                f"Calibrated Jaccard threshold on {val_path.name}: "
                f"thr={thr:.3f} (val balanced-acc={100*thr_ba:.1f}%)"
            )
        else:
            thr = 0.15
            print(f"No val CSV; using default Jaccard thr={thr:.3f}")
    elif thr is None:
        thr = 0.15

    layout = setup_run(
        SCRIPT_ID,
        tags=modes,
        run_name=args.run_name,
        run_category=RUN_CATEGORY,
        output_model_dir=args.output_run_dir,
    )
    print(f"Run dir: {layout.root}")

    comparison_lines = [
        "Lexical null models",
        f"eval_csv: {eval_path}",
        f"jaccard_threshold: {thr:.4f}",
        f"jaccard_val_balanced_acc: {thr_ba if thr_ba is not None else 'n/a'}",
        f"pair_rules: {len(PAIRS)} (laterality+site+severity-scalar)",
        f"severity_tiers: {len(SEVERITY_TIERS)}",
        "",
    ]

    all_overall: List[Dict[str, object]] = []
    all_by_type: List[Dict[str, object]] = []

    for mode in modes:
        pred_rows = predict_rows(eval_rows, mode=mode, jaccard_threshold=thr)
        pred_path = layout.results / f"hard_predictions_{mode}.csv"
        write_predictions(pred_path, pred_rows)
        print(f"Wrote {pred_path}")

        overall_row, by_rows = summarize_prediction_rows(pred_rows, mode=mode)
        all_overall.append(overall_row)
        all_by_type.extend(by_rows)
        write_summary_csvs(
            layout.results,
            mode=mode,
            overall_row=overall_row,
            by_rows=by_rows,
        )
        print_quick_table(overall_row, by_rows, f"mode={mode}")

        if not args.no_summary:
            # Per-mode subdir for summaries to avoid collisions.
            mode_root = layout.root / mode
            mode_results = mode_root / "results"
            mode_plots = mode_root / "plots"
            mode_results.mkdir(parents=True, exist_ok=True)
            mode_plots.mkdir(parents=True, exist_ok=True)
            # predictions_eval expects CSV under mode results or elsewhere; pass path directly.
            linked = mode_results / "hard_predictions.csv"
            if linked.resolve() != pred_path.resolve():
                linked.write_text(pred_path.read_text(encoding="utf-8"), encoding="utf-8")
            result = write_eval_artifacts(
                str(linked),
                run_name=f"{layout.root.name}_{mode}",
                eval_mode="all",
                eval_csv=str(eval_path),
                source="bert",
                results_dir=mode_results,
                plots_dir=mode_plots,
                write_incidence=False,
                write_confusion=True,
                model_name=f"lexical_null_{mode}",
                summary_path=str(mode_results / "eval_summary.txt"),
            )
            comparison_lines.append(f"--- {mode} ---")
            comparison_lines.append(f"summary: {result.summary_path}")
            comparison_lines.append(
                f"acc={overall_row['accuracy_pct']}% "
                f"inc_recall={overall_row['inconsistent_recall_pct']}% "
                f"con_recall={overall_row['consistent_recall_pct']}% "
                f"fa={overall_row['false_alarm_rate_pct']}%"
            )
            for r in by_rows:
                comparison_lines.append(
                    f"  {r['error_type']}: n={r['n']} "
                    f"{r['metric']}={r['class_recall_pct']}%"
                )
            comparison_lines.append("")

    write_comparison_csvs(layout.results, all_overall, all_by_type)
    print(
        f"Wrote summary CSVs: {layout.results / 'metrics_overall.csv'}, "
        f"{layout.results / 'metrics_by_error_type.csv'}, "
        f"{layout.results / 'metrics_by_error_type_wide.csv'}"
    )

    compare_path = layout.results / "lexical_null_comparison.txt"
    compare_path.write_text("\n".join(comparison_lines) + "\n", encoding="utf-8")
    print(f"\nWrote comparison: {compare_path}")
    print(compare_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
