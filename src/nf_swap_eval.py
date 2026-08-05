"""NF_v_Dis_swap eval helpers: swap-direction breakdown and reporting."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

try:
    from data.build_eval_datasets import (
        _is_nf_labeled,
        parse_human_nf_label,
    )
    from paths import NO_FINDINGS_DIR, data_path
    from report_split_io import load_report_records
except ImportError:
    import sys

    _repo_root = Path(__file__).resolve().parents[1]
    if str(_repo_root) not in sys.path:
        sys.path.append(str(_repo_root))
    from data.build_eval_datasets import (  # type: ignore
        _is_nf_labeled,
        parse_human_nf_label,
    )
    from paths import NO_FINDINGS_DIR, data_path  # type: ignore
    from report_split_io import load_report_records  # type: ignore


NF_HUMAN_REVIEW_CSV = str(NO_FINDINGS_DIR / "no_findings_test_set_human_review.csv")
NF_SWAP_DIRECTION_ORDER = ("nf_to_disease", "disease_to_nf")
NF_SWAP_DIRECTION_LABELS = {
    "nf_to_disease": "NF → disease (orig NF)",
    "disease_to_nf": "disease → NF (orig disease)",
}


@dataclass(frozen=True)
class NfSwapDirectionContext:
    idx_to_source_row: Dict[int, str]
    nf_labels: Dict[str, bool]
    human_nf_by_source_row: Dict[str, bool]


def _example_attr(example: Any, name: str, default=None):
    if isinstance(example, Mapping):
        return example.get(name, default)
    return getattr(example, name, default)


def infer_report_split_for_eval_csv(eval_csv: Optional[str]) -> tuple[str, str]:
    """Return (report_split_path, nf_labels_split key)."""
    name = Path(eval_csv or "").name.lower()
    if "val_hard" in name or name.startswith("val_"):
        return data_path("val_text_path"), "val"
    return data_path("test_text_path"), "test"


def _load_nf_labels_quiet(csv_path: str, split: str) -> Dict[str, bool]:
    labels: Dict[str, bool] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("split") == split:
                labels[row["conclusion"]] = row["nf_label"] == "1"
    return labels


def _load_human_nf_by_source_row_quiet(
    csv_path: str,
    split: str,
) -> Dict[str, bool]:
    out: Dict[str, bool] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("split") != split:
                continue
            parsed = parse_human_nf_label(row.get("human_nf", ""))
            if parsed is None:
                continue
            out[str(row.get("source_row", "")).strip()] = parsed == 1
    return out


def build_nf_swap_direction_context(
    eval_csv: Optional[str] = None,
    human_review_csv: str = NF_HUMAN_REVIEW_CSV,
) -> NfSwapDirectionContext:
    report_path, split = infer_report_split_for_eval_csv(eval_csv)
    records = load_report_records(report_path)
    idx_to_source_row = {i: str(r.source_row).strip() for i, r in enumerate(records)}
    nf_labels = _load_nf_labels_quiet(
        str(NO_FINDINGS_DIR / "no_findings_labels.csv"),
        split=split,
    )
    human_nf_by_source_row: Dict[str, bool] = {}
    review_path = Path(human_review_csv)
    if review_path.is_file():
        human_nf_by_source_row = _load_human_nf_by_source_row_quiet(
            str(review_path), split=split
        )
    return NfSwapDirectionContext(
        idx_to_source_row=idx_to_source_row,
        nf_labels=nf_labels,
        human_nf_by_source_row=human_nf_by_source_row,
    )


def build_original_conclusion_by_source_idx(examples: Sequence[Any]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for ex in examples:
        if _example_attr(ex, "error_type") not in {"consistent", "positive"}:
            continue
        source_idx = _example_attr(ex, "source_idx")
        if source_idx is None:
            continue
        out[int(source_idx)] = str(_example_attr(ex, "conclusion") or "").strip()
    return out


def resolve_original_is_nf(
    source_idx: Optional[int],
    original_conclusion: str,
    ctx: NfSwapDirectionContext,
) -> Optional[bool]:
    if source_idx is not None:
        source_row = ctx.idx_to_source_row.get(int(source_idx))
        if source_row and source_row in ctx.human_nf_by_source_row:
            return ctx.human_nf_by_source_row[source_row]
    if not original_conclusion:
        return None
    return _is_nf_labeled(original_conclusion, ctx.nf_labels)


def nf_swap_direction_name(orig_is_nf: bool) -> str:
    return "nf_to_disease" if orig_is_nf else "disease_to_nf"


def summarize_nf_swap_by_direction(
    examples: Sequence[Any],
    preds: Union[np.ndarray, Sequence[int]],
    labels: Union[np.ndarray, Sequence[int]],
    *,
    task: str,
    eval_csv: Optional[str] = None,
    nf_multiclass_id: int = 4,
) -> Dict[str, Dict[str, float]]:
    """Per-direction metrics for NF_v_Dis_swap rows aligned with preds/labels."""
    preds_arr = np.asarray(preds, dtype=np.int64)
    labels_arr = np.asarray(labels, dtype=np.int64)
    if len(examples) != len(preds_arr) or len(examples) != len(labels_arr):
        raise ValueError(
            f"Length mismatch: examples={len(examples)} preds={len(preds_arr)} "
            f"labels={len(labels_arr)}"
        )

    ctx = build_nf_swap_direction_context(eval_csv)
    orig_by_idx = build_original_conclusion_by_source_idx(examples)
    buckets: Dict[str, List[int]] = {k: [] for k in NF_SWAP_DIRECTION_ORDER}

    for i, ex in enumerate(examples):
        if _example_attr(ex, "error_type") != "NF_v_Dis_swap":
            continue
        source_idx = _example_attr(ex, "source_idx")
        orig = orig_by_idx.get(int(source_idx)) if source_idx is not None else None
        if orig is None:
            continue
        orig_is_nf = resolve_original_is_nf(source_idx, orig, ctx)
        if orig_is_nf is None:
            continue
        buckets[nf_swap_direction_name(orig_is_nf)].append(i)

    out: Dict[str, Dict[str, float]] = {}
    for direction in NF_SWAP_DIRECTION_ORDER:
        idxs = buckets[direction]
        n = len(idxs)
        if n == 0:
            out[direction] = {
                "n": 0,
                "detect_recall": float("nan"),
                "type_acc": float("nan"),
            }
            continue
        dir_preds = preds_arr[idxs]
        dir_labels = labels_arr[idxs]
        if task == "multiclass":
            detect_recall = float((dir_preds != 0).mean())
            type_acc = float((dir_preds == dir_labels).mean())
        else:
            detect_recall = float((dir_preds == 0).mean())
            type_acc = float((dir_preds == dir_labels).mean())
        out[direction] = {
            "n": n,
            "detect_recall": detect_recall,
            "type_acc": type_acc,
        }
    return out


def format_nf_swap_direction_lines(
    by_direction: Dict[str, Dict[str, float]],
    *,
    task: str,
) -> List[str]:
    """Text lines for NF swap direction table (empty when no NF_v_Dis_swap rows)."""
    if not by_direction or all(int(m.get("n", 0)) == 0 for m in by_direction.values()):
        return []
    type_label = "type_acc" if task == "multiclass" else "label_acc"
    sep = "─" * 72
    lines = [
        "",
        sep,
        "NF_v_Dis_swap by swap direction (from original conclusion class)",
        f"  {'direction':<32} {'n':>6}  {'detect_recall':>14}  {type_label:>10}",
        sep,
    ]
    for direction in NF_SWAP_DIRECTION_ORDER:
        m = by_direction.get(direction, {})
        n = int(m.get("n", 0))
        if n == 0:
            continue
        det = float(m.get("detect_recall", float("nan")))
        typ = float(m.get("type_acc", float("nan")))
        det_str = f"{det:.1%}" if det == det else "   -  "
        typ_str = f"{typ:.1%}" if typ == typ else "   -  "
        lines.append(
            f"  {NF_SWAP_DIRECTION_LABELS[direction]:<32} {n:>6}  {det_str:>14}  {typ_str:>10}"
        )
    lines.append(sep)
    return lines


def print_nf_swap_direction_table(
    by_direction: Dict[str, Dict[str, float]],
    *,
    task: str,
) -> None:
    lines = format_nf_swap_direction_lines(by_direction, task=task)
    if lines:
        print("\n" + "\n".join(lines))
