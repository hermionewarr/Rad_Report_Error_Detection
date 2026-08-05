#!/usr/bin/env python3
"""Cross-prompt and BERT-vs-LLM failure overlap analysis for paper figures.

Joins per-example predictions on (patient_id, source_idx, error_type) and reports:
  - prompt-robust vs prompt-sensitive failure buckets
  - pairwise agreement / error-set Jaccard across prompts
  - failure rates by error_type
  - BERT layer scaling overlap (6 / 8 / 12 by default)
  - configurable BERT vs LLM comparison (default: BERT-6 vs LLM p3)

Typical usage (from repo root, PTCT3 or any env with matplotlib):

  python scripts/analyze_failure_overlap.py

  python scripts/analyze_failure_overlap.py \\
    --llm-prompts 2,3,13 \\
    --llm-models qwen3-32b-gguf,gemma-3-27b-gguf,llama-3.3-70b \\
    --bert-runs 6=20260615_bert_inconsistency-hard-w3-single-cls600-big,\\
8=20260618_bert_inconsistency-hard-single-cls600-8L,\\
12=20260618_bert_inconsistency-hard-single-cls600-12L \\
    --bert-llm-compare bert=6,llm=3

Override auto-discovered LLM suites when needed:

  --llm-suites 3=20260608_llm-suite_qwen3-32b-q6_p3
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
BERT_ROOT = REPO_ROOT
BERT_RUNS_DIR = BERT_ROOT / "trained_models" / "runs" / "bert_binary"
LLM_BASELINE_DIR = BERT_ROOT / "trained_models" / "runs" / "llm_baseline"
FIGURES_DIR = BERT_ROOT / "figures" / "failure_overlap"

DEFAULT_LLM_PROMPTS = ("2", "3", "13")
DEFAULT_LLM_MODELS = (
    ("Qwen3-32B", "qwen3-32b-gguf"),
    ("Gemma3-27B", "gemma-3-27b-gguf"),
    ("Llama3-70B", "llama-3.3-70b"),
)
DEFAULT_BERT_RUNS: Dict[str, str] = {
    "6": "20260615_bert_inconsistency-hard-w3-single-cls600-big",
    "8": "20260618_bert_inconsistency-hard-single-cls600-8L",
    "12": "20260618_bert_inconsistency-hard-single-cls600-12L",
}
DEFAULT_BERT_LLM_COMPARE = ("6", "3")

ERROR_TYPES = (
    "consistent",
    "laterality_swap",
    "site_swap",
    "severity_flip",
    "NF_v_Dis_swap",
    "cross_pair_mismatch",
)

ROW_KEY = ("patient_id", "source_idx", "error_type")

BUCKET_COLORS = {
    "robust_correct": "#2ca02c",
    "prompt_sensitive": "#ffbb78",
    "robust_failure": "#d62728",
}


@dataclass
class ExampleRecord:
    patient_id: str
    source_idx: str
    error_type: str
    findings: str = ""
    conclusion: str = ""
    wrong_by: Dict[str, bool] = field(default_factory=dict)
    fp_by: Dict[str, bool] = field(default_factory=dict)
    fn_by: Dict[str, bool] = field(default_factory=dict)


def _parse_csv_map(raw: Optional[str]) -> Dict[str, str]:
    """Parse 'key=path,key=path' or 'key=run_name,...'."""
    out: Dict[str, str] = {}
    if not raw:
        return out
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"Expected key=value, got: {item!r}")
        key, value = item.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _parse_llm_models(raw: Optional[str]) -> Tuple[Tuple[str, str], ...]:
    if not raw:
        return DEFAULT_LLM_MODELS
    models: List[Tuple[str, str]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            label, slug = item.split("=", 1)
            models.append((label.strip(), slug.strip()))
        else:
            models.append((item, item))
    if not models:
        raise SystemExit("No LLM models parsed from --llm-models")
    return tuple(models)


def _suite_matches_prompt(dir_name: str, prompt_id: str) -> bool:
    pid = re.escape(prompt_id)
    return bool(
        re.search(rf"__p{pid}$", dir_name)
        or re.search(rf"__p{pid}_", dir_name)
        or re.search(rf"_p{pid}$", dir_name)
    )


def find_latest_llm_suite(prompt_id: str, model_slug: Optional[str] = None) -> Optional[Path]:
    candidates: List[Path] = []
    for path in LLM_BASELINE_DIR.iterdir():
        if path.is_dir() and _suite_matches_prompt(path.name, prompt_id):
            candidates.append(path)
    if not candidates:
        return None
    if model_slug:
        with_preds = [
            c
            for c in candidates
            if (c / "models" / model_slug / "results" / "predictions.csv").is_file()
        ]
        if with_preds:
            candidates = with_preds
    return max(candidates, key=lambda p: p.name)


def resolve_llm_predictions(
    prompt_id: str,
    model_slug: str,
    suite_overrides: Dict[str, str],
) -> Path:
    if prompt_id in suite_overrides:
        override = suite_overrides[prompt_id]
        suite = Path(override)
        if not suite.is_absolute():
            suite = LLM_BASELINE_DIR / override
    else:
        suite = find_latest_llm_suite(prompt_id, model_slug=model_slug)
        if suite is None:
            raise FileNotFoundError(
                f"No LLM suite found for prompt p{prompt_id} under {LLM_BASELINE_DIR}. "
                f"Pass --llm-suites {prompt_id}=<suite_dir>."
            )
    pred = suite / "models" / model_slug / "results" / "predictions.csv"
    if not pred.is_file():
        raise FileNotFoundError(f"Missing LLM predictions: {pred}")
    return pred


def resolve_bert_predictions(layer_key: str, run_overrides: Dict[str, str]) -> Path:
    run_name = run_overrides.get(layer_key) or DEFAULT_BERT_RUNS.get(layer_key)
    if not run_name:
        raise SystemExit(f"Unknown BERT layer key {layer_key!r}; pass --bert-runs")
    run_dir = BERT_RUNS_DIR / run_name
    for name in ("hard_predictions.csv", "predictions.csv"):
        path = run_dir / "results" / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No BERT predictions under {run_dir / 'results'}")


def _keep_row(row: dict, *, require_parse_ok: bool) -> bool:
    if require_parse_ok and "parse_ok" in row:
        return (row.get("parse_ok") or "").strip() == "1"
    return True


def _row_wrong(row: dict) -> bool:
    val = (row.get("is_correct") or "").strip()
    if val in ("0", "1"):
        return val == "0"
    return (row.get("pred_label_name") or "") != (row.get("true_label_name") or "")


def _row_fp(row: dict) -> bool:
    if "is_fp0" in row:
        return (row.get("is_fp0") or "").strip() == "1"
    return row.get("error_type") == "consistent" and _row_wrong(row)


def _row_fn(row: dict) -> bool:
    if "is_fn0" in row:
        return (row.get("is_fn0") or "").strip() == "1"
    return row.get("error_type") != "consistent" and _row_wrong(row)


def load_predictions(
    csv_path: Path,
    label: str,
    *,
    require_parse_ok: bool = False,
) -> Dict[Tuple[str, str], ExampleRecord]:
    records: Dict[Tuple[str, str], ExampleRecord] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not _keep_row(row, require_parse_ok=require_parse_ok):
                continue
            key = tuple(row[k] for k in ROW_KEY)
            rec = records.get(key)
            if rec is None:
                rec = ExampleRecord(
                    patient_id=row["patient_id"],
                    source_idx=row["source_idx"],
                    error_type=(row.get("error_type") or "").strip(),
                    findings=row.get("findings", ""),
                    conclusion=row.get("conclusion", ""),
                )
                records[key] = rec
            rec.wrong_by[label] = _row_wrong(row)
            rec.fp_by[label] = _row_fp(row)
            rec.fn_by[label] = _row_fn(row)
    return records


def inner_join_records(
    parts: Sequence[Tuple[str, Dict[Tuple[str, str], ExampleRecord]]],
) -> Dict[Tuple[str, str], ExampleRecord]:
    """Merge records keeping only keys present in every condition."""
    if not parts:
        return {}
    common_keys = set(parts[0][1].keys())
    for _label, records in parts[1:]:
        common_keys &= set(records.keys())
    merged = merge_records(parts)
    return {key: merged[key] for key in common_keys if key in merged}


def align_record_counts(
    parts: Sequence[Tuple[str, Dict[Tuple[str, str], ExampleRecord]]],
) -> None:
    for label, records in parts:
        print(f"  {label}: {len(records)} rows")
    joined = inner_join_records(parts)
    if parts:
        sizes = [len(records) for _label, records in parts]
        if len(set(sizes)) > 1 or len(joined) != sizes[0]:
            print(
                f"  aligned intersection: {len(joined)} rows "
                f"(using examples present in all conditions)"
            )


def merge_records(
    parts: Sequence[Tuple[str, Dict[Tuple[str, str], ExampleRecord]]],
) -> Dict[Tuple[str, str], ExampleRecord]:
    merged: Dict[Tuple[str, str], ExampleRecord] = {}
    for _label, records in parts:
        for key, rec in records.items():
            dst = merged.get(key)
            if dst is None:
                dst = ExampleRecord(
                    patient_id=rec.patient_id,
                    source_idx=rec.source_idx,
                    error_type=rec.error_type,
                    findings=rec.findings,
                    conclusion=rec.conclusion,
                )
                merged[key] = dst
            dst.wrong_by.update(rec.wrong_by)
            dst.fp_by.update(rec.fp_by)
            dst.fn_by.update(rec.fn_by)
    return merged


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0:
        return float("nan")
    po = float(np.mean(a == b))
    p1 = float(np.mean(a))
    p2 = float(np.mean(b))
    pe = p1 * p2 + (1.0 - p1) * (1.0 - p2)
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def error_jaccard(wrong_a: np.ndarray, wrong_b: np.ndarray) -> float:
    union = np.logical_or(wrong_a, wrong_b)
    if not union.any():
        return 1.0
    inter = np.logical_and(wrong_a, wrong_b)
    return float(inter.sum() / union.sum())


def failure_buckets(
    merged: Dict[Tuple[str, str], ExampleRecord],
    labels: Sequence[str],
) -> Dict[str, int]:
    counts = {"robust_correct": 0, "prompt_sensitive": 0, "robust_failure": 0}
    for rec in merged.values():
        if not all(lab in rec.wrong_by for lab in labels):
            continue
        flags = [rec.wrong_by[lab] for lab in labels]
        n_wrong = sum(flags)
        if n_wrong == 0:
            counts["robust_correct"] += 1
        elif n_wrong == len(flags):
            counts["robust_failure"] += 1
        else:
            counts["prompt_sensitive"] += 1
    return counts


def failure_rate_by_error_type(
    merged: Dict[Tuple[str, str], ExampleRecord],
    label: str,
) -> Dict[str, Tuple[int, int]]:
    stats = {et: [0, 0] for et in ERROR_TYPES}
    for rec in merged.values():
        if label not in rec.wrong_by:
            continue
        et = rec.error_type if rec.error_type in stats else "cross_pair_mismatch"
        stats[et][0] += 1
        if rec.wrong_by[label]:
            stats[et][1] += 1
    return {et: (n, w) for et, (n, w) in stats.items() if n > 0}


def ladder_flips(
    merged: Dict[Tuple[str, str], ExampleRecord],
    ordered_labels: Sequence[str],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for left, right in zip(ordered_labels[:-1], ordered_labels[1:]):
        gained = lost = 0
        for rec in merged.values():
            if left not in rec.wrong_by or right not in rec.wrong_by:
                continue
            if rec.wrong_by[left] and not rec.wrong_by[right]:
                gained += 1
            elif not rec.wrong_by[left] and rec.wrong_by[right]:
                lost += 1
        rows.append(
            {
                "from": left,
                "to": right,
                "gained": gained,
                "lost": lost,
                "net": gained - lost,
            }
        )
    return rows


def overlap_counts(
    merged: Dict[Tuple[str, str], ExampleRecord],
    label_a: str,
    label_b: str,
) -> Dict[str, int]:
    both_wrong = bert_only = llm_only = both_correct = 0
    for rec in merged.values():
        if label_a not in rec.wrong_by or label_b not in rec.wrong_by:
            continue
        wa = rec.wrong_by[label_a]
        wb = rec.wrong_by[label_b]
        if wa and wb:
            both_wrong += 1
        elif wa:
            bert_only += 1
        elif wb:
            llm_only += 1
        else:
            both_correct += 1
    return {
        "both_wrong": both_wrong,
        "a_only_wrong": bert_only,
        "b_only_wrong": llm_only,
        "both_correct": both_correct,
        "n": both_wrong + bert_only + llm_only + both_correct,
    }


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_failure_buckets(
    bucket_rows: List[dict],
    output: Path,
    *,
    title: str,
) -> None:
    models = [r["llm_model"] for r in bucket_rows]
    totals = np.array([r["n"] for r in bucket_rows], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = np.zeros(len(models))
    for bucket, color in BUCKET_COLORS.items():
        vals = np.array([r[bucket] for r in bucket_rows], dtype=float)
        pct = np.where(totals > 0, 100.0 * vals / totals, 0.0)
        ax.bar(models, pct, bottom=bottom, label=bucket.replace("_", " "), color=color)
        bottom += pct
    ax.set_ylabel("% of examples")
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_jaccard_heatmap(
    labels: Sequence[str],
    matrix: np.ndarray,
    output: Path,
    *,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(1.2 * len(labels) + 2, 1.0 * len(labels) + 2))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Error-set Jaccard")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_failure_rate_heatmap(
    condition_stats: Dict[str, Dict[str, Tuple[int, int]]],
    output: Path,
    *,
    title: str,
) -> None:
    conditions = list(condition_stats.keys())
    types_present = [et for et in ERROR_TYPES if any(condition_stats[c].get(et) for c in conditions)]
    data = np.full((len(types_present), len(conditions)), np.nan)
    for j, cond in enumerate(conditions):
        for i, et in enumerate(types_present):
            n, w = condition_stats[cond].get(et, (0, 0))
            if n:
                data[i, j] = 100.0 * w / n
    fig, ax = plt.subplots(figsize=(1.1 * len(conditions) + 2, 0.55 * len(types_present) + 2))
    im = ax.imshow(data, vmin=0, vmax=100, cmap="YlOrRd")
    ax.set_xticks(range(len(conditions)))
    ax.set_yticks(range(len(types_present)))
    ax.set_xticklabels(conditions, rotation=30, ha="right")
    ax.set_yticklabels(types_present)
    for i in range(len(types_present)):
        for j in range(len(conditions)):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.0f}", ha="center", va="center", fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Failure rate (%)")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_overlap_bars(
    rows: List[dict],
    output: Path,
    *,
    title: str,
    label_a: str,
    label_b: str,
) -> None:
    models = [r["llm_model"] for r in rows] if "llm_model" in rows[0] else [r["group"] for r in rows]
    key = "llm_model" if "llm_model" in rows[0] else "group"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(models))
    w = 0.25
    both = np.array([r["both_wrong"] for r in rows])
    a_only = np.array([r["a_only_wrong"] for r in rows])
    b_only = np.array([r["b_only_wrong"] for r in rows])
    ax.bar(x - w, both, width=w, label=f"Both wrong ({label_a} & {label_b})")
    ax.bar(x, a_only, width=w, label=f"{label_a} only wrong")
    ax.bar(x + w, b_only, width=w, label=f"{label_b} only wrong")
    ax.set_xticks(x)
    ax.set_xticklabels([r[key] for r in rows], rotation=15, ha="right")
    ax.set_ylabel("Example count")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def export_example_cases(
    merged: Dict[Tuple[str, str], ExampleRecord],
    labels: Sequence[str],
    output: Path,
    *,
    max_per_bucket: int = 25,
) -> None:
    buckets: Dict[str, List[ExampleRecord]] = {
        "robust_failure": [],
        "prompt_sensitive": [],
    }
    for rec in merged.values():
        flags = [rec.wrong_by[lab] for lab in labels if lab in rec.wrong_by]
        if len(flags) != len(labels):
            continue
        n_wrong = sum(flags)
        if n_wrong == len(flags):
            buckets["robust_failure"].append(rec)
        elif 0 < n_wrong < len(flags):
            buckets["prompt_sensitive"].append(rec)

    fieldnames = [
        "bucket",
        "patient_id",
        "source_idx",
        "error_type",
        *labels,
        "findings",
        "conclusion",
    ]
    out_rows: List[dict] = []
    for bucket, recs in buckets.items():
        for rec in recs[:max_per_bucket]:
            row = {
                "bucket": bucket,
                "patient_id": rec.patient_id,
                "source_idx": rec.source_idx,
                "error_type": rec.error_type,
                "findings": rec.findings[:500],
                "conclusion": rec.conclusion[:500],
            }
            for lab in labels:
                wrong = rec.wrong_by.get(lab)
                row[lab] = "wrong" if wrong else "correct"
            out_rows.append(row)
    write_csv(output, fieldnames, out_rows)


def analyze_llm_model(
    llm_label: str,
    model_slug: str,
    prompt_ids: Sequence[str],
    suite_overrides: Dict[str, str],
    output_dir: Path,
) -> Dict[str, object]:
    prompt_labels = [f"p{pid}" for pid in prompt_ids]
    parts: List[Tuple[str, Dict[Tuple[str, str], ExampleRecord]]] = []
    resolved: Dict[str, str] = {}
    for pid in prompt_ids:
        path = resolve_llm_predictions(pid, model_slug, suite_overrides)
        label = f"p{pid}"
        resolved[label] = str(path)
        parts.append((label, load_predictions(path, label, require_parse_ok=True)))
    print(f"  {llm_label} prediction counts:")
    align_record_counts(parts)
    merged = inner_join_records(parts)

    buckets = failure_buckets(merged, prompt_labels)
    n = sum(buckets.values())
    bucket_row = {
        "llm_model": llm_label,
        "n": n,
        **buckets,
        "robust_failure_pct": 100.0 * buckets["robust_failure"] / n if n else 0.0,
        "prompt_sensitive_pct": 100.0 * buckets["prompt_sensitive"] / n if n else 0.0,
    }

    pairwise_rows: List[dict] = []
    wrong_vectors: Dict[str, np.ndarray] = {}
    keys = sorted(merged.keys())
    for lab in prompt_labels:
        wrong_vectors[lab] = np.array([merged[k].wrong_by.get(lab, False) for k in keys])
    for a, b in combinations(prompt_labels, 2):
        wa, wb = wrong_vectors[a], wrong_vectors[b]
        pred_a = ~wa
        pred_b = ~wb
        pairwise_rows.append(
            {
                "llm_model": llm_label,
                "label_a": a,
                "label_b": b,
                "agreement": float(np.mean(pred_a == pred_b)),
                "cohen_kappa": cohen_kappa(pred_a.astype(int), pred_b.astype(int)),
                "error_jaccard": error_jaccard(wa, wb),
                "n": len(keys),
            }
        )

    jaccard = np.eye(len(prompt_labels))
    for i, a in enumerate(prompt_labels):
        for j, b in enumerate(prompt_labels):
            if i < j:
                val = error_jaccard(wrong_vectors[a], wrong_vectors[b])
                jaccard[i, j] = jaccard[j, i] = val

    cond_stats = {
        lab: failure_rate_by_error_type(merged, lab) for lab in prompt_labels
    }
    flips = ladder_flips(merged, prompt_labels)

    model_dir = output_dir / llm_label.replace("/", "_")
    model_dir.mkdir(parents=True, exist_ok=True)
    write_csv(model_dir / "resolved_paths.csv", ["label", "path"], [{"label": k, "path": v} for k, v in resolved.items()])
    write_csv(model_dir / "failure_buckets.csv", list(bucket_row.keys()), [bucket_row])
    write_csv(
        model_dir / "pairwise_prompt_agreement.csv",
        ["llm_model", "label_a", "label_b", "agreement", "cohen_kappa", "error_jaccard", "n"],
        pairwise_rows,
    )
    write_csv(
        model_dir / "ladder_flips.csv",
        ["from", "to", "gained", "lost", "net"],
        flips,
    )
    failure_type_rows = []
    for cond, stats in cond_stats.items():
        for et, (count_n, count_w) in stats.items():
            failure_type_rows.append(
                {
                    "llm_model": llm_label,
                    "condition": cond,
                    "error_type": et,
                    "n": count_n,
                    "failures": count_w,
                    "failure_rate_pct": 100.0 * count_w / count_n if count_n else 0.0,
                }
            )
    write_csv(
        model_dir / "failure_by_error_type.csv",
        ["llm_model", "condition", "error_type", "n", "failures", "failure_rate_pct"],
        failure_type_rows,
    )
    export_example_cases(merged, prompt_labels, model_dir / "example_cases.csv")

    plot_jaccard_heatmap(
        prompt_labels,
        jaccard,
        model_dir / "pairwise_error_jaccard.png",
        title=f"{llm_label}: error-set Jaccard across prompts",
    )
    plot_failure_rate_heatmap(
        cond_stats,
        model_dir / "failure_rate_by_error_type.png",
        title=f"{llm_label}: failure rate by error type",
    )

    return {
        "bucket_row": bucket_row,
        "pairwise_rows": pairwise_rows,
        "merged": merged,
        "prompt_labels": prompt_labels,
        "jaccard": jaccard,
        "cond_stats": cond_stats,
    }


def analyze_bert_layers(
    layer_keys: Sequence[str],
    run_overrides: Dict[str, str],
    output_dir: Path,
) -> Dict[str, object]:
    labels = [f"BERT-{k}L" for k in layer_keys]
    parts: List[Tuple[str, Dict[Tuple[str, str], ExampleRecord]]] = []
    resolved: Dict[str, str] = {}
    for key in layer_keys:
        path = resolve_bert_predictions(key, run_overrides)
        label = f"BERT-{key}L"
        resolved[label] = str(path)
        parts.append((label, load_predictions(path, label)))
    print("BERT layer prediction counts:")
    align_record_counts(parts)
    merged = inner_join_records(parts)

    keys = sorted(merged.keys())
    wrong_vectors = {
        lab: np.array([merged[k].wrong_by.get(lab, False) for k in keys]) for lab in labels
    }

    pairwise_rows: List[dict] = []
    jaccard = np.eye(len(labels))
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i < j:
                val = error_jaccard(wrong_vectors[a], wrong_vectors[b])
                jaccard[i, j] = jaccard[j, i] = val
                pred_a = ~wrong_vectors[a]
                pred_b = ~wrong_vectors[b]
                pairwise_rows.append(
                    {
                        "label_a": a,
                        "label_b": b,
                        "agreement": float(np.mean(pred_a == pred_b)),
                        "cohen_kappa": cohen_kappa(pred_a.astype(int), pred_b.astype(int)),
                        "error_jaccard": val,
                        "n": len(keys),
                    }
                )

    buckets = failure_buckets(merged, labels)
    bert_dir = output_dir / "bert_layers"
    bert_dir.mkdir(parents=True, exist_ok=True)
    write_csv(bert_dir / "resolved_paths.csv", ["label", "path"], [{"label": k, "path": v} for k, v in resolved.items()])
    write_csv(
        bert_dir / "pairwise_layer_agreement.csv",
        ["label_a", "label_b", "agreement", "cohen_kappa", "error_jaccard", "n"],
        pairwise_rows,
    )
    write_csv(
        bert_dir / "failure_buckets.csv",
        ["n", "robust_correct", "prompt_sensitive", "robust_failure"],
        [{"n": sum(buckets.values()), **buckets}],
    )
    plot_jaccard_heatmap(
        labels,
        jaccard,
        bert_dir / "pairwise_error_jaccard.png",
        title="BERT layer scaling: error-set Jaccard (6 / 8 / 12)",
    )
    cond_stats = {lab: failure_rate_by_error_type(merged, lab) for lab in labels}
    plot_failure_rate_heatmap(
        cond_stats,
        bert_dir / "failure_rate_by_error_type.png",
        title="BERT layers: failure rate by error type",
    )
    return {"merged": merged, "labels": labels, "pairwise_rows": pairwise_rows}


def _subset_label(
    records: Dict[Tuple[str, str], ExampleRecord],
    label: str,
) -> Dict[Tuple[str, str], ExampleRecord]:
    """Keep only one condition label on copied records (for pairwise overlap)."""
    out: Dict[Tuple[str, str], ExampleRecord] = {}
    for key, rec in records.items():
        if label not in rec.wrong_by:
            continue
        out[key] = ExampleRecord(
            patient_id=rec.patient_id,
            source_idx=rec.source_idx,
            error_type=rec.error_type,
            findings=rec.findings,
            conclusion=rec.conclusion,
            wrong_by={label: rec.wrong_by[label]},
            fp_by={label: rec.fp_by.get(label, False)},
            fn_by={label: rec.fn_by.get(label, False)},
        )
    return out


def analyze_bert_llm_compare(
    llm_results: Dict[str, dict],
    bert_layer: str,
    llm_prompt: str,
    run_overrides: Dict[str, str],
    output_dir: Path,
) -> List[dict]:
    bert_label = f"BERT-{bert_layer}L"
    llm_prompt_label = f"p{llm_prompt}"
    bert_path = resolve_bert_predictions(bert_layer, run_overrides)
    bert_recs = load_predictions(bert_path, bert_label)

    overlap_rows: List[dict] = []
    for llm_label, result in llm_results.items():
        llm_recs = _subset_label(result["merged"], llm_prompt_label)
        merged = inner_join_records([(bert_label, bert_recs), (llm_prompt_label, llm_recs)])
        counts = overlap_counts(merged, bert_label, llm_prompt_label)
        union_wrong = counts["both_wrong"] + counts["a_only_wrong"] + counts["b_only_wrong"]
        overlap_rows.append(
            {
                "llm_model": llm_label,
                "bert_label": bert_label,
                "llm_prompt": llm_prompt_label,
                **counts,
                "jaccard_errors": (
                    counts["both_wrong"] / union_wrong if union_wrong else 1.0
                ),
            }
        )

    compare_dir = output_dir / "bert_vs_llm"
    compare_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        compare_dir / f"overlap_{bert_label}_vs_{llm_prompt_label}.csv",
        [
            "llm_model",
            "bert_label",
            "llm_prompt",
            "n",
            "both_wrong",
            "a_only_wrong",
            "b_only_wrong",
            "both_correct",
            "jaccard_errors",
        ],
        overlap_rows,
    )
    plot_overlap_bars(
        overlap_rows,
        compare_dir / f"overlap_{bert_label}_vs_{llm_prompt_label}.png",
        title=f"Error overlap: {bert_label} vs LLM {llm_prompt_label}",
        label_a=bert_label,
        label_b=llm_prompt_label,
    )
    return overlap_rows


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--llm-prompts",
        default=",".join(DEFAULT_LLM_PROMPTS),
        help=f"Comma-separated prompt ids (default: {','.join(DEFAULT_LLM_PROMPTS)})",
    )
    p.add_argument(
        "--llm-models",
        default=None,
        help="Comma-separated slugs or Label=slug (default: all three main LLMs)",
    )
    p.add_argument(
        "--llm-suites",
        default=None,
        help="Optional prompt_id=suite_dir overrides, e.g. 3=20260608_llm-suite_qwen3-32b-q6_p3",
    )
    p.add_argument(
        "--bert-runs",
        default=None,
        help="Layer=run_name overrides (default: 6/8/12 cls600 hard-single runs)",
    )
    p.add_argument(
        "--bert-layers",
        default="6,8,12",
        help="BERT layer keys for inter-BERT comparison (default: 6,8,12)",
    )
    p.add_argument(
        "--bert-llm-compare",
        default=f"bert={DEFAULT_BERT_LLM_COMPARE[0]},llm={DEFAULT_BERT_LLM_COMPARE[1]}",
        help="Which BERT layer and LLM prompt to compare (default: bert=6,llm=3)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=FIGURES_DIR,
        help=f"Output directory (default: {FIGURES_DIR})",
    )
    p.add_argument(
        "--skip-missing-llm",
        action="store_true",
        help="Skip LLM models whose prediction files are missing instead of failing",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    prompt_ids = [p.strip() for p in args.llm_prompts.split(",") if p.strip()]
    if not prompt_ids:
        raise SystemExit("No prompts in --llm-prompts")

    llm_models = _parse_llm_models(args.llm_models)
    suite_overrides = _parse_csv_map(args.llm_suites)
    bert_run_overrides = _parse_csv_map(args.bert_runs)
    bert_layers = [x.strip() for x in args.bert_layers.split(",") if x.strip()]

    compare_map = _parse_csv_map(args.bert_llm_compare)
    bert_compare_layer = compare_map.get("bert", DEFAULT_BERT_LLM_COMPARE[0])
    llm_compare_prompt = compare_map.get("llm", DEFAULT_BERT_LLM_COMPARE[1])

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"LLM prompts: {', '.join('p' + p for p in prompt_ids)}")
    print(f"Output: {output_dir}")

    llm_results: Dict[str, dict] = {}
    all_bucket_rows: List[dict] = []
    all_pairwise: List[dict] = []

    for llm_label, slug in llm_models:
        try:
            result = analyze_llm_model(
                llm_label, slug, prompt_ids, suite_overrides, output_dir
            )
        except FileNotFoundError as exc:
            if args.skip_missing_llm:
                print(f"SKIP {llm_label}: {exc}")
                continue
            raise
        llm_results[llm_label] = result
        all_bucket_rows.append(result["bucket_row"])
        all_pairwise.extend(result["pairwise_rows"])
        print(
            f"{llm_label}: robust_failure={result['bucket_row']['robust_failure']} "
            f"({result['bucket_row']['robust_failure_pct']:.1f}%), "
            f"prompt_sensitive={result['bucket_row']['prompt_sensitive']} "
            f"({result['bucket_row']['prompt_sensitive_pct']:.1f}%)"
        )

    if all_bucket_rows:
        write_csv(
            output_dir / "summary_failure_buckets.csv",
            list(all_bucket_rows[0].keys()),
            all_bucket_rows,
        )
        write_csv(
            output_dir / "summary_pairwise_prompt_agreement.csv",
            ["llm_model", "label_a", "label_b", "agreement", "cohen_kappa", "error_jaccard", "n"],
            all_pairwise,
        )
        plot_failure_buckets(
            all_bucket_rows,
            output_dir / "failure_buckets_by_llm.png",
            title="Cross-prompt failure buckets (% of eval examples)",
        )

    if bert_layers:
        print(f"BERT inter-layer comparison: {', '.join(bert_layers)}L")
        bert_result = analyze_bert_layers(bert_layers, bert_run_overrides, output_dir)
        write_csv(
            output_dir / "bert_layers" / "summary_pairwise_layer_agreement.csv",
            ["label_a", "label_b", "agreement", "cohen_kappa", "error_jaccard", "n"],
            bert_result["pairwise_rows"],
        )

    if llm_results:
        print(f"BERT vs LLM: BERT-{bert_compare_layer}L vs LLM p{llm_compare_prompt}")
        overlap_rows = analyze_bert_llm_compare(
            llm_results,
            bert_compare_layer,
            llm_compare_prompt,
            bert_run_overrides,
            output_dir,
        )
        write_csv(
            output_dir / "summary_bert_llm_overlap.csv",
            [
                "llm_model",
                "bert_label",
                "llm_prompt",
                "n",
                "both_wrong",
                "a_only_wrong",
                "b_only_wrong",
                "both_correct",
                "jaccard_errors",
            ],
            overlap_rows,
        )

    config_lines = [
        f"llm_prompts={','.join(prompt_ids)}",
        f"llm_models={args.llm_models or 'default'}",
        f"llm_suites={args.llm_suites or 'auto'}",
        f"bert_runs={args.bert_runs or 'default'}",
        f"bert_layers={','.join(bert_layers)}",
        f"bert_llm_compare=bert={bert_compare_layer},llm={llm_compare_prompt}",
    ]
    (output_dir / "run_config.txt").write_text("\n".join(config_lines) + "\n", encoding="utf-8")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
