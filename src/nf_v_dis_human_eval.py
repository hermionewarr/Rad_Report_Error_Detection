"""Human-labeled NF vs disease (NF_v_Dis_swap) eval subset.

Builds consistent + swapped-conclusion pairs from the manually reviewed test
subset (human_nf labels) so BERT and LLM baselines report the same benchmark.
"""
from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    from sklearn.metrics import roc_auc_score
except ImportError:  # pragma: no cover
    roc_auc_score = None  # type: ignore

try:
    from paths import NO_FINDINGS_DIR, data_path
    from report_split_io import load_report_records
except ImportError:
    import sys
    from pathlib import Path

    _repo_root = Path(__file__).resolve().parents[1]
    if str(_repo_root) not in sys.path:
        sys.path.append(str(_repo_root))
    from paths import NO_FINDINGS_DIR, data_path
    from report_split_io import load_report_records


NF_HUMAN_REVIEW_CSV = str(NO_FINDINGS_DIR / "no_findings_test_set_human_review.csv")
DEFAULT_TEST_FILE = data_path("test_text_path")
DEFAULT_SEED = 42
HUMAN_NF_PREDICTIONS_FILENAME = "human_nf_predictions.csv"


@dataclass
class HumanEvalExample:
    findings: str
    conclusion: str
    label: int  # 1 = consistent, 0 = inconsistent
    error_type: str
    patient_id: str = ""
    source_idx: Optional[int] = None


def parse_human_nf_label(val: str) -> Optional[int]:
    v = str(val).strip().lower()
    if v in ("0", "1"):
        return int(v)
    return None


def _optional_int(value) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def build_nf_v_dis_human_eval_examples(
    human_review_csv: str = NF_HUMAN_REVIEW_CSV,
    test_file: str = DEFAULT_TEST_FILE,
    seed: int = DEFAULT_SEED,
) -> List[HumanEvalExample]:
    """Build NF_v_Dis_swap eval pairs using human_nf labels on the reviewed subset."""
    human_rows: List[Dict[str, str]] = []
    with open(human_review_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if parse_human_nf_label(row.get("human_nf", "")) is None:
                continue
            human_rows.append(row)
    if not human_rows:
        return []

    by_source_row = {r.source_row: r for r in load_report_records(test_file)}
    nf_pool = [
        row["conclusion"].strip()
        for row in human_rows
        if parse_human_nf_label(row["human_nf"]) == 1
    ]
    disease_pool = [
        row["conclusion"].strip()
        for row in human_rows
        if parse_human_nf_label(row["human_nf"]) == 0
    ]

    rng = random.Random(seed)
    examples: List[HumanEvalExample] = []
    for row in human_rows:
        source_row = str(row.get("source_row", "")).strip()
        rec = by_source_row.get(source_row)
        if rec is None:
            continue
        human_i = parse_human_nf_label(row["human_nf"])
        assert human_i is not None
        patient_id = (row.get("patient_id") or "").strip()
        source_idx = _optional_int(source_row)
        examples.append(
            HumanEvalExample(
                findings=rec.findings,
                conclusion=rec.conclusion,
                label=1,
                error_type="consistent",
                patient_id=patient_id,
                source_idx=source_idx,
            )
        )
        swap_pool = disease_pool if human_i == 1 else nf_pool
        swap_candidates = [c for c in swap_pool if c != rec.conclusion]
        if not swap_candidates:
            continue
        examples.append(
            HumanEvalExample(
                findings=rec.findings,
                conclusion=rng.choice(swap_candidates),
                label=0,
                error_type="NF_v_Dis_swap",
                patient_id=patient_id,
                source_idx=source_idx,
            )
        )
    return examples


def _row_detected_inconsistent(row: Dict) -> bool:
    """True when the model flagged the pair as inconsistent (binary or multiclass)."""
    if "is_tp0" in row and str(row.get("is_tp0", "")).strip() != "":
        return str(row.get("is_tp0", "")).strip().lower() in {"1", "true", "yes"}
    return int(row.get("true_label", 0)) == 0 and int(row.get("pred_label", 1)) == 0


def summarize_human_nf_predictions(rows: List[Dict]) -> Optional[Dict[str, float]]:
    """NF_v_Dis_swap detection recall on human-labeled subset predictions."""
    swap_rows = [r for r in rows if (r.get("error_type") or "").strip() == "NF_v_Dis_swap"]
    if not swap_rows:
        return None

    detected = sum(1 for r in swap_rows if _row_detected_inconsistent(r))
    n_swap = len(swap_rows)
    n_reviewed = sum(
        1 for r in rows if (r.get("error_type") or "").strip() == "consistent"
    )

    auroc = float("nan")
    if roc_auc_score is not None:
        y_true = [
            1 if (r.get("error_type") or "").strip() == "NF_v_Dis_swap" else 0
            for r in rows
        ]
        scores = [float(r.get("prob_inconsistent", 0.0) or 0.0) for r in rows]
        if len(set(y_true)) > 1:
            auroc = float(roc_auc_score(y_true, scores))

    return {
        "n": n_swap,
        "inconsistent_recall": detected / max(n_swap, 1),
        "auroc": auroc,
        "n_reviewed": n_reviewed,
    }


def format_human_nf_summary_line(metrics: Dict[str, float]) -> str:
    auroc = float(metrics.get("auroc", float("nan")))
    auroc_str = f"{auroc:.3f}" if auroc == auroc else "  -  "
    return (
        f"  {'NF vs Disease Human eval':<26} {int(metrics['n']):>6}  "
        f"{float(metrics['inconsistent_recall']):>13.1%}  {auroc_str:>6}"
    )
