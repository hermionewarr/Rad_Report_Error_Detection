"""Prob/confidence evaluation from row-level LLM predictions CSV."""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )
except ImportError:  # pragma: no cover
    roc_auc_score = None  # type: ignore
    roc_curve = None  # type: ignore
    average_precision_score = None  # type: ignore
    precision_recall_curve = None  # type: ignore

try:
    from src.llm_eval_prompts import (
        CONFIDENCE_SCORE_MAX,
        CONFIDENCE_SCORE_MIN,
    )
except ImportError:
    import sys

    _repo_root = Path(__file__).resolve().parents[1]
    if str(_repo_root) not in sys.path:
        sys.path.append(str(_repo_root))
    from src.llm_eval_prompts import (
        CONFIDENCE_SCORE_MAX,
        CONFIDENCE_SCORE_MIN,
    )

try:
    from paths import (
        RUN_OUTPUT_LOG,
        current_argv,
        default_plot_path,
        default_result_path,
        infer_run_root_from_predictions,
        setup_run,
        tee_run_output,
    )
    from src.predictions_eval import (
        CONSISTENT_LABEL,
        INCONSISTENT_LABEL,
        _balanced_balanced_accuracy,
        _binary_class_recall_rates,
        _binary_macro_f1,
        _fmt_pct,
        _fmt_pct_float,
        _load_predictions,
        _metric_rows,
        _overall_counts,
        _pct,
        _resolve_multiclass,
        _safe_int,
        _weighted_binary_class_accuracy,
        summarize_by_error_type,
    )
except ImportError:
    import sys

    _repo_root = Path(__file__).resolve().parents[1]
    if str(_repo_root) not in sys.path:
        sys.path.append(str(_repo_root))
    from paths import (
        RUN_OUTPUT_LOG,
        current_argv,
        default_plot_path,
        default_result_path,
        infer_run_root_from_predictions,
        setup_run,
        tee_run_output,
    )
    from src.predictions_eval import (
        CONSISTENT_LABEL,
        INCONSISTENT_LABEL,
        _balanced_balanced_accuracy,
        _binary_class_recall_rates,
        _binary_macro_f1,
        _fmt_pct,
        _fmt_pct_float,
        _load_predictions,
        _metric_rows,
        _overall_counts,
        _pct,
        _resolve_multiclass,
        _safe_int,
        _weighted_binary_class_accuracy,
        summarize_by_error_type,
    )


SCRIPT_ID = "predictions-prob-eval"
RUN_CATEGORY = "llm_baseline"

DEFAULT_THRESHOLDS = (
    0.5,
    0.7,
    0.8,
    0.9,
    0.95,
    0.99,
    0.999,
    0.9999,
)
DEFAULT_MARGIN_CUTOFFS = (0.0, 0.5, 0.9, 0.95, 0.99, 0.999, 0.9999)
CALIBRATION_BINS = 10


@dataclass
class ProbEvalResult:
    predictions_csv: str
    run_name: str
    n_total: int
    n_metric: int
    has_confidence: bool
    json_prob_disagree: int
    auroc: float
    auprc: float
    best_balanced_threshold: float
    best_f1_threshold: float
    summary_path: Optional[str] = None
    errors_csv_path: Optional[str] = None
    plot_paths: List[str] = field(default_factory=list)


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        v = float((value or "").strip() if isinstance(value, str) else value)
    except (TypeError, ValueError):
        return default
    if math.isnan(v):
        return default
    return v


def _row_prob_inconsistent(row: Dict) -> float:
    return _safe_float(row.get("prob_inconsistent", ""), default=0.0)


def _row_prob_consistent(row: Dict) -> float:
    return _safe_float(row.get("prob_consistent", ""), default=0.0)


def _row_margin(row: Dict) -> float:
    return abs(_row_prob_inconsistent(row) - _row_prob_consistent(row))


def _prob_pred_label(row: Dict, threshold: float = 0.5) -> int:
    return INCONSISTENT_LABEL if _row_prob_inconsistent(row) >= threshold else CONSISTENT_LABEL


def _rows_with_pred_label(
    rows: Sequence[Dict],
    pred_fn: Callable[[Dict], int],
) -> List[Dict]:
    out: List[Dict] = []
    for row in rows:
        patched = dict(row)
        patched["pred_label"] = str(pred_fn(row))
        out.append(patched)
    return out


def _rows_passing_filter(
    rows: Sequence[Dict],
    keep_fn: Callable[[Dict], bool],
) -> List[Dict]:
    return [r for r in rows if keep_fn(r)]


def _has_prob_columns(rows: Sequence[Dict]) -> bool:
    return bool(rows) and "prob_inconsistent" in rows[0] and "prob_consistent" in rows[0]


def _has_confidence_column(rows: Sequence[Dict]) -> bool:
    return bool(rows) and "confidence" in rows[0]


def _json_prob_disagreement_count(rows: Sequence[Dict]) -> int:
    count = 0
    for row in rows:
        json_pred = _safe_int(row.get("pred_label", ""), default=INCONSISTENT_LABEL)
        prob_pred = _prob_pred_label(row, threshold=0.5)
        if json_pred != prob_pred:
            count += 1
    return count


def _inconsistent_positive_labels(rows: Sequence[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (y_score=prob_inconsistent, y_true=1 if inconsistent else 0)."""
    scores: List[float] = []
    labels: List[int] = []
    for row in rows:
        true_label = _safe_int(row.get("true_label", ""), default=INCONSISTENT_LABEL)
        scores.append(_row_prob_inconsistent(row))
        labels.append(1 if true_label == INCONSISTENT_LABEL else 0)
    return np.asarray(scores, dtype=float), np.asarray(labels, dtype=int)


def _consistent_positive_labels(rows: Sequence[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (y_score=prob_consistent, y_true=1 if consistent else 0)."""
    scores: List[float] = []
    labels: List[int] = []
    for row in rows:
        true_label = _safe_int(row.get("true_label", ""), default=INCONSISTENT_LABEL)
        scores.append(_row_prob_consistent(row))
        labels.append(1 if true_label == CONSISTENT_LABEL else 0)
    return np.asarray(scores, dtype=float), np.asarray(labels, dtype=int)


def _prob_pred_label_consistent_threshold(row: Dict, threshold: float = 0.5) -> int:
    return CONSISTENT_LABEL if _row_prob_consistent(row) >= threshold else INCONSISTENT_LABEL


def _overall_summary_dict(overall: Dict[str, int], by_type: Dict[str, Dict[str, object]]) -> Dict[str, float]:
    inc_recall, con_recall = _binary_class_recall_rates(overall)
    macro_err, bal_bal = _balanced_balanced_accuracy(overall, by_type)
    acc = overall["correct"] / max(overall["n_metric"], 1)
    macro_f1 = _binary_macro_f1(
        overall["tp0"], overall["fp0"], overall["fn0"], overall["tn0"]
    )
    fpr = overall["fp0"] / max(overall["n_consistent"], 1)
    return {
        "acc": acc,
        "det": inc_recall,
        "con": con_recall,
        "fpr": fpr,
        "bal": 0.5 * (inc_recall + con_recall),
        "bal_bal": bal_bal,
        "macro_err": macro_err,
        "macro_f1": macro_f1,
        "weighted_50_50": _weighted_binary_class_accuracy(
            overall, weight_inconsistent=0.5, weight_consistent=0.5
        ),
        "weighted_95_5": _weighted_binary_class_accuracy(
            overall, weight_inconsistent=0.05, weight_consistent=0.95
        ),
        "n": overall["n_metric"],
    }


def _metrics_for_pred_fn(
    rows: Sequence[Dict],
    pred_fn: Callable[[Dict], int],
    *,
    multiclass: bool = False,
) -> Dict[str, float]:
    patched = _rows_with_pred_label(rows, pred_fn)
    overall = _overall_counts(patched, multiclass=multiclass)
    by_type = summarize_by_error_type(patched, multiclass=multiclass)
    return _overall_summary_dict(overall, by_type)


def _metrics_at_threshold(rows: Sequence[Dict], threshold: float, *, multiclass: bool = False) -> Dict[str, float]:
    return _metrics_for_pred_fn(
        rows,
        lambda r, t=threshold: _prob_pred_label(r, threshold=t),
        multiclass=multiclass,
    )


def _class_f1_from_rates(tp: int, fp: int, fn: int) -> float:
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    return 2 * prec * rec / (prec + rec + 1e-8)


def _f1_inconsistent_at_threshold(
    rows: Sequence[Dict],
    threshold: float,
) -> float:
    tp = fp = fn = 0
    for row in rows:
        true_label = _safe_int(row.get("true_label", ""), default=INCONSISTENT_LABEL)
        pred = _prob_pred_label(row, threshold=threshold)
        if true_label == INCONSISTENT_LABEL and pred == INCONSISTENT_LABEL:
            tp += 1
        elif true_label == CONSISTENT_LABEL and pred == INCONSISTENT_LABEL:
            fp += 1
        elif true_label == INCONSISTENT_LABEL and pred == CONSISTENT_LABEL:
            fn += 1
    return _class_f1_from_rates(tp, fp, fn)


def _search_best_threshold(
    rows: Sequence[Dict],
    *,
    multiclass: bool = False,
    thresholds: Optional[Sequence[float]] = None,
) -> Tuple[float, float, float]:
    """Return (best_bal_threshold, best_bal_score, best_f1_threshold)."""
    if thresholds is None:
        scores, _ = _inconsistent_positive_labels(rows)
        if len(scores):
            uniq = np.unique(scores)
            if len(uniq) > 200:
                thresholds = tuple(float(x) for x in np.quantile(scores, np.linspace(0.01, 0.99, 99)))
            else:
                thresholds = tuple(float(x) for x in uniq)
            thresholds = tuple(sorted(set(list(DEFAULT_THRESHOLDS) + list(thresholds))))
        else:
            thresholds = DEFAULT_THRESHOLDS

    best_bal_t = 0.5
    best_bal = float("-inf")
    best_f1_t = 0.5
    best_f1 = float("-inf")
    for threshold in thresholds:
        m = _metrics_at_threshold(rows, threshold, multiclass=multiclass)
        if m["bal_bal"] == m["bal_bal"] and m["bal_bal"] > best_bal:
            best_bal = m["bal_bal"]
            best_bal_t = threshold
        f1 = _f1_inconsistent_at_threshold(rows, threshold)
        if f1 > best_f1:
            best_f1 = f1
            best_f1_t = threshold
    return best_bal_t, best_bal, best_f1_t


def _compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = CALIBRATION_BINS) -> float:
    if len(y_true) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if not np.any(mask):
            continue
        bin_acc = float(np.mean(y_true[mask]))
        bin_conf = float(np.mean(y_prob[mask]))
        ece += np.sum(mask) * abs(bin_acc - bin_conf)
    return float(ece / len(y_true))


def _calibration_bins_from_arrays(
    scores: np.ndarray,
    labels: np.ndarray,
    n_bins: int = CALIBRATION_BINS,
) -> List[Dict[str, float]]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out: List[Dict[str, float]] = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)
        n = int(np.sum(mask))
        if n == 0:
            continue
        out.append(
            {
                "lo": float(lo),
                "hi": float(hi),
                "n": n,
                "mean_pred": float(np.mean(scores[mask])),
                "actual_rate": float(np.mean(labels[mask])),
            }
        )
    return out


def _calibration_bins(
    rows: Sequence[Dict],
    n_bins: int = CALIBRATION_BINS,
) -> List[Dict[str, float]]:
    scores, labels = _inconsistent_positive_labels(rows)
    return _calibration_bins_from_arrays(scores, labels, n_bins=n_bins)


def _calibration_bins_consistent(
    rows: Sequence[Dict],
    n_bins: int = CALIBRATION_BINS,
) -> List[Dict[str, float]]:
    scores, labels = _consistent_positive_labels(rows)
    return _calibration_bins_from_arrays(scores, labels, n_bins=n_bins)


def _confidence_metrics(rows: Sequence[Dict]) -> Dict[int, Dict[str, float]]:
    buckets: Dict[int, List[Dict]] = {}
    for row in rows:
        conf = _safe_int(row.get("confidence", ""), default=-1)
        if conf < 0:
            continue
        buckets.setdefault(conf, []).append(row)
    out: Dict[int, Dict[str, float]] = {}
    for conf, subset in sorted(buckets.items()):
        overall = _overall_counts(subset, multiclass=False)
        by_type = summarize_by_error_type(subset, multiclass=False)
        summary = _overall_summary_dict(overall, by_type)
        out[conf] = {
            "n": float(len(subset)),
            "acc": summary["acc"],
            "bal_bal": summary["bal_bal"],
            "bal": summary["bal"],
            "det": summary["det"],
            "con": summary["con"],
            "fpr": summary["fpr"],
        }
    return out


def _confidence_accuracy(rows: Sequence[Dict]) -> Dict[int, Dict[str, float]]:
    return {
        conf: {"n": int(stats["n"]), "acc": stats["acc"]}
        for conf, stats in _confidence_metrics(rows).items()
    }


def _row_confidence(row: Dict) -> int:
    return _safe_int(row.get("confidence", ""), default=-1)


def _rows_with_min_confidence(rows: Sequence[Dict], min_confidence: int) -> List[Dict]:
    return [r for r in rows if _row_confidence(r) >= min_confidence]


def _confidence_threshold_metrics(rows: Sequence[Dict]) -> List[Dict[str, float]]:
    """Metrics for selective prediction: keep rows with confidence >= tau."""
    n_total = len(rows)
    out: List[Dict[str, float]] = []
    for tau in range(CONFIDENCE_SCORE_MIN, CONFIDENCE_SCORE_MAX + 1):
        kept = _rows_with_min_confidence(rows, tau)
        coverage = 100.0 * len(kept) / max(n_total, 1)
        if not kept:
            out.append(
                {
                    "tau": float(tau),
                    "n": 0.0,
                    "coverage": coverage,
                    "acc": float("nan"),
                    "det": float("nan"),
                    "con": float("nan"),
                    "fpr": float("nan"),
                    "bal": float("nan"),
                    "bal_bal": float("nan"),
                }
            )
            continue
        m = _metrics_for_pred_fn(
            kept,
            lambda r: _safe_int(r.get("pred_label", ""), default=INCONSISTENT_LABEL),
        )
        out.append(
            {
                "tau": float(tau),
                "n": float(len(kept)),
                "coverage": coverage,
                "acc": m["acc"],
                "det": m["det"],
                "con": m["con"],
                "fpr": m["fpr"],
                "bal": m["bal"],
                "bal_bal": m["bal_bal"],
            }
        )
    return out


def _confidence_by_error_type(
    rows: Sequence[Dict],
) -> Tuple[List[str], Dict[str, Dict[int, Dict[str, float]]]]:
    buckets: Dict[str, Dict[int, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    error_types_seen: set[str] = set()
    for row in rows:
        conf = _row_confidence(row)
        if conf < CONFIDENCE_SCORE_MIN:
            continue
        error_type = (row.get("error_type") or "unknown").strip()
        error_types_seen.add(error_type)
        buckets[error_type][conf].append(row)

    def _sort_key(error_type: str) -> Tuple[int, str]:
        return (0 if error_type == "consistent" else 1, error_type)

    error_types = sorted(error_types_seen, key=_sort_key)
    stats: Dict[str, Dict[int, Dict[str, float]]] = {}
    for error_type in error_types:
        stats[error_type] = {}
        for conf in range(CONFIDENCE_SCORE_MIN, CONFIDENCE_SCORE_MAX + 1):
            subset = buckets[error_type].get(conf, [])
            if not subset:
                continue
            overall = _overall_counts(subset, multiclass=False)
            by_type = summarize_by_error_type(subset, multiclass=False)
            summary = _overall_summary_dict(overall, by_type)
            stats[error_type][conf] = {
                "n": float(len(subset)),
                "acc": summary["acc"],
                "det": summary["det"],
                "con": summary["con"],
                "bal_bal": summary["bal_bal"],
            }
    return error_types, stats


def _short_error_type_label(error_type: str) -> str:
    return {
        "cross_pair_mismatch": "cross_pair",
        "laterality_swap": "laterality",
        "site_swap": "site",
        "severity_flip": "severity",
        "NF_v_Dis_swap": "NF_v_Dis",
    }.get(error_type, error_type)


def _collect_forensics_rows(
    rows: Sequence[Dict],
    *,
    margin_cutoff: float = 0.5,
) -> List[Dict]:
    flagged: List[Dict] = []
    for row in rows:
        true_label = _safe_int(row.get("true_label", ""), default=INCONSISTENT_LABEL)
        json_pred = _safe_int(row.get("pred_label", ""), default=INCONSISTENT_LABEL)
        prob_pred = _prob_pred_label(row, threshold=0.5)
        margin = _row_margin(row)
        wrong = true_label != json_pred
        reasons: List[str] = []
        if wrong:
            reasons.append("wrong")
        if margin < margin_cutoff:
            reasons.append("low_margin")
        if json_pred != prob_pred:
            reasons.append("json_prob_disagree")
        conf = _safe_int(row.get("confidence", ""), default=-1)
        if conf >= 4 and margin >= 0.99 and wrong:
            reasons.append("high_conf_wrong")
        if not reasons:
            continue
        flagged.append(
            {
                "patient_id": row.get("patient_id", ""),
                "source_idx": row.get("source_idx", ""),
                "error_type": row.get("error_type", ""),
                "true_label": true_label,
                "pred_label": json_pred,
                "prob_pred_label": prob_pred,
                "prob_inconsistent": _row_prob_inconsistent(row),
                "prob_consistent": _row_prob_consistent(row),
                "margin": margin,
                "confidence": conf if conf >= 0 else "",
                "reasons": ";".join(reasons),
                "findings": (row.get("findings") or "")[:120],
                "conclusion": (row.get("conclusion") or "")[:120],
            }
        )
    return flagged


def _write_forensics_csv(rows: Sequence[Dict], output_path: str) -> None:
    fieldnames = [
        "patient_id",
        "source_idx",
        "error_type",
        "true_label",
        "pred_label",
        "prob_pred_label",
        "prob_inconsistent",
        "prob_consistent",
        "margin",
        "confidence",
        "reasons",
        "findings",
        "conclusion",
    ]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved forensics CSV to: {output_path}")


def plot_roc_curve(rows: Sequence[Dict], output_path: str) -> Optional[float]:
    return _plot_roc_curve_for_score(
        rows,
        output_path,
        score_labels_fn=_inconsistent_positive_labels,
        score_name="prob_inconsistent",
        xlabel="False positive rate (consistent flagged inconsistent)",
        ylabel="True positive rate (inconsistent detected)",
        title="ROC — prob_inconsistent as detection score",
    )


def plot_roc_curve_consistent(rows: Sequence[Dict], output_path: str) -> Optional[float]:
    return _plot_roc_curve_for_score(
        rows,
        output_path,
        score_labels_fn=_consistent_positive_labels,
        score_name="prob_consistent",
        xlabel="False positive rate (inconsistent flagged consistent)",
        ylabel="True positive rate (consistent detected)",
        title="ROC — prob_consistent as score",
    )


def _plot_roc_curve_for_score(
    rows: Sequence[Dict],
    output_path: str,
    *,
    score_labels_fn: Callable[[Sequence[Dict]], Tuple[np.ndarray, np.ndarray]],
    score_name: str,
    xlabel: str,
    ylabel: str,
    title: str,
) -> Optional[float]:
    if roc_auc_score is None or roc_curve is None:
        print(f"Skipping ROC plot for {score_name} (sklearn not installed).")
        return float("nan")
    scores, labels = score_labels_fn(rows)
    if len(np.unique(labels)) < 2:
        print(f"Skipping ROC plot for {score_name} (single class in labels).")
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    auroc = float(roc_auc_score(labels, scores))

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(fpr, tpr, color="#1f77b4", linewidth=2, label=f"{score_name} (AUROC = {auroc:.4f})")
    ax.plot([0, 1], [0, 1], color="#888", linestyle="--", linewidth=1, label="Random")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved ROC plot to: {output_path}")
    return auroc


def plot_pr_curve(rows: Sequence[Dict], output_path: str) -> Optional[float]:
    return _plot_pr_curve_for_score(
        rows,
        output_path,
        score_labels_fn=_inconsistent_positive_labels,
        score_name="prob_inconsistent",
        xlabel="Recall (inconsistent)",
        ylabel="Precision (inconsistent)",
        title="Precision–Recall — prob_inconsistent",
    )


def plot_pr_curve_consistent(rows: Sequence[Dict], output_path: str) -> Optional[float]:
    return _plot_pr_curve_for_score(
        rows,
        output_path,
        score_labels_fn=_consistent_positive_labels,
        score_name="prob_consistent",
        xlabel="Recall (consistent)",
        ylabel="Precision (consistent)",
        title="Precision–Recall — prob_consistent",
    )


def _plot_pr_curve_for_score(
    rows: Sequence[Dict],
    output_path: str,
    *,
    score_labels_fn: Callable[[Sequence[Dict]], Tuple[np.ndarray, np.ndarray]],
    score_name: str,
    xlabel: str,
    ylabel: str,
    title: str,
) -> Optional[float]:
    if average_precision_score is None or precision_recall_curve is None:
        print(f"Skipping PR plot for {score_name} (sklearn not installed).")
        return float("nan")
    scores, labels = score_labels_fn(rows)
    if len(np.unique(labels)) < 2:
        print(f"Skipping PR plot for {score_name} (single class in labels).")
        return float("nan")
    precision, recall, _ = precision_recall_curve(labels, scores)
    auprc = float(average_precision_score(labels, scores))

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(recall, precision, color="#2ca02c", linewidth=2, label=f"AUPRC = {auprc:.4f}")
    baseline = float(np.mean(labels))
    ax.axhline(baseline, color="#888", linestyle="--", linewidth=1, label=f"Prevalence = {baseline:.3f}")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved PR plot to: {output_path}")
    return auprc


def _metrics_at_consistent_threshold(
    rows: Sequence[Dict],
    threshold: float,
    *,
    multiclass: bool = False,
) -> Dict[str, float]:
    return _metrics_for_pred_fn(
        rows,
        lambda r, t=threshold: _prob_pred_label_consistent_threshold(r, threshold=t),
        multiclass=multiclass,
    )


def plot_metrics_vs_threshold(
    rows: Sequence[Dict],
    output_path: str,
    *,
    thresholds: Sequence[float],
    multiclass: bool = False,
    score: str = "inconsistent",
) -> None:
    accs: List[float] = []
    dets: List[float] = []
    cons: List[float] = []
    bals: List[float] = []
    bal_bals: List[float] = []
    metric_fn = _metrics_at_threshold if score == "inconsistent" else _metrics_at_consistent_threshold
    for threshold in thresholds:
        m = metric_fn(rows, threshold, multiclass=multiclass)
        accs.append(100.0 * m["acc"])
        dets.append(100.0 * m["det"])
        cons.append(100.0 * m["con"])
        bals.append(100.0 * m["bal"])
        bal_bals.append(100.0 * m["bal_bal"])

    score_label = "prob_inconsistent" if score == "inconsistent" else "prob_consistent"
    pred_rule = (
        "predict inconsistent if score ≥ t"
        if score == "inconsistent"
        else "predict consistent if score ≥ t"
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = list(thresholds)
    ax.plot(xs, accs, marker="o", label="Accuracy", linewidth=2)
    ax.plot(xs, dets, marker="o", label="Detection (inconsistent recall)", linewidth=2)
    ax.plot(xs, cons, marker="o", label="Consistent recall", linewidth=2)
    ax.plot(xs, bals, marker="o", label="Balanced acc (50/50 recall)", linewidth=2)
    ax.plot(xs, bal_bals, marker="o", label="Balanced-balanced acc", linewidth=2)
    ax.set_xlabel(f"Threshold on {score_label} ({pred_rule})")
    ax.set_ylabel("Rate (%)")
    ax.set_title(f"Metrics vs threshold — {score_label}")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved threshold sweep plot to: {output_path}")


def plot_selective_margin_curve(
    rows: Sequence[Dict],
    output_path: str,
    *,
    margin_cutoffs: Sequence[float],
    multiclass: bool = False,
) -> None:
    coverages: List[float] = []
    accs: List[float] = []
    bals: List[float] = []
    bal_bals: List[float] = []
    dets: List[float] = []
    cons: List[float] = []
    n_total = len(rows)
    for cutoff in margin_cutoffs:
        kept = _rows_passing_filter(rows, lambda r, c=cutoff: _row_margin(r) >= c)
        coverage = 100.0 * len(kept) / max(n_total, 1)
        if not kept:
            coverages.append(coverage)
            accs.append(float("nan"))
            bals.append(float("nan"))
            bal_bals.append(float("nan"))
            dets.append(float("nan"))
            cons.append(float("nan"))
            continue
        m = _metrics_for_pred_fn(kept, lambda r: _safe_int(r.get("pred_label", ""), default=0), multiclass=multiclass)
        coverages.append(coverage)
        accs.append(100.0 * m["acc"])
        bals.append(100.0 * m["bal"])
        bal_bals.append(100.0 * m["bal_bal"])
        dets.append(100.0 * m["det"])
        cons.append(100.0 * m["con"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(coverages, accs, marker="o", label="Accuracy", linewidth=2)
    ax.plot(coverages, bals, marker="o", label="Balanced acc (50/50)", linewidth=2)
    ax.plot(coverages, bal_bals, marker="o", label="Balanced-balanced acc", linewidth=2)
    ax.plot(coverages, dets, marker="o", label="Detection", linewidth=2)
    ax.plot(coverages, cons, marker="o", label="Consistent recall", linewidth=2)
    ax.set_xlabel("Coverage (%) — rows kept (margin ≥ cutoff)")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Selective prediction by token-probability margin")
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 102)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved selective margin plot to: {output_path}")


def _plot_calibration_reliability_panel(
    ax: plt.Axes,
    bins: List[Dict[str, float]],
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    color: str = "#1f77b4",
) -> None:
    if not bins:
        ax.set_title(f"{title}\n(no bins)")
        return
    mean_pred = [b["mean_pred"] for b in bins]
    actual = [b["actual_rate"] for b in bins]
    counts = [b["n"] for b in bins]
    ax.plot([0, 1], [0, 1], color="#888", linestyle="--", linewidth=1, label="Perfect calibration")
    ax.scatter(mean_pred, actual, s=[max(20, min(200, c / 20)) for c in counts], color=color, alpha=0.8)
    for mp, ac, b in zip(mean_pred, actual, bins):
        ax.annotate(f"n={b['n']}", (mp, ac), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)


def plot_calibration_reliability(rows: Sequence[Dict], output_path: str) -> None:
    bins = _calibration_bins(rows)
    if not bins:
        print("Skipping calibration plot (no bins).")
        return
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    _plot_calibration_reliability_panel(
        ax,
        bins,
        xlabel="Mean predicted prob_inconsistent",
        ylabel="Observed inconsistent rate",
        title="Calibration — prob_inconsistent",
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved calibration plot to: {output_path}")


def plot_calibration_reliability_consistent(rows: Sequence[Dict], output_path: str) -> None:
    bins = _calibration_bins_consistent(rows)
    if not bins:
        print("Skipping consistent calibration plot (no bins).")
        return
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    _plot_calibration_reliability_panel(
        ax,
        bins,
        xlabel="Mean predicted prob_consistent",
        ylabel="Observed consistent rate",
        title="Calibration — prob_consistent",
        color="#ff7f0e",
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved consistent calibration plot to: {output_path}")


def plot_calibration_reliability_dual(rows: Sequence[Dict], output_path: str) -> None:
    bins_inc = _calibration_bins(rows)
    bins_con = _calibration_bins_consistent(rows)
    if not bins_inc and not bins_con:
        print("Skipping dual calibration plot (no bins).")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    _plot_calibration_reliability_panel(
        axes[0],
        bins_inc,
        xlabel="Mean predicted prob_inconsistent",
        ylabel="Observed inconsistent rate",
        title="prob_inconsistent",
    )
    _plot_calibration_reliability_panel(
        axes[1],
        bins_con,
        xlabel="Mean predicted prob_consistent",
        ylabel="Observed consistent rate",
        title="prob_consistent",
        color="#ff7f0e",
    )
    fig.suptitle("Calibration reliability diagrams", y=1.02)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved dual calibration plot to: {output_path}")


def _margin_histogram_xlim(all_margins: np.ndarray) -> Tuple[float, float, Optional[float]]:
    """Return (zoom_xmin, zoom_xmax, full_xmax). full_xmax=None when zoom covers full range."""
    if len(all_margins) == 0:
        return 0.0, 1.0, None
    share_below_90 = float(np.mean(all_margins < 0.9))
    share_below_99 = float(np.mean(all_margins < 0.99))
    if share_below_99 < 0.15:
        return 0.95, 1.0005, 1.0
    if share_below_90 < 0.10:
        return 0.85, 1.002, 1.0
    return 0.0, 1.002, None


def plot_margin_by_outcome(rows: Sequence[Dict], output_path: str) -> None:
    correct = [_row_margin(r) for r in rows if _safe_int(r.get("true_label", "")) == _safe_int(r.get("pred_label", ""))]
    wrong = [_row_margin(r) for r in rows if _safe_int(r.get("true_label", "")) != _safe_int(r.get("pred_label", ""))]
    all_margins = np.asarray(correct + wrong, dtype=float)
    zoom_xmin, zoom_xmax, full_xmax = _margin_histogram_xlim(all_margins)

    if full_xmax is not None:
        fig, (ax_full, ax_zoom) = plt.subplots(
            2,
            1,
            figsize=(8, 7),
            height_ratios=[1, 1.6],
            sharey=False,
        )
        full_bins = np.linspace(0.0, full_xmax, 41)
        ax_full.hist(correct, bins=full_bins, alpha=0.6, label=f"Correct (n={len(correct)})", color="#2ca02c")
        ax_full.hist(wrong, bins=full_bins, alpha=0.6, label=f"Wrong (n={len(wrong)})", color="#d62728")
        ax_full.set_xlim(0.0, full_xmax)
        ax_full.set_xlabel("Margin |prob_inconsistent − prob_consistent|")
        ax_full.set_ylabel("Count")
        ax_full.set_title("Full range")
        ax_full.legend(loc="upper right", fontsize=9)
        ax_full.grid(True, alpha=0.3)
        ax = ax_zoom
    else:
        fig, ax = plt.subplots(figsize=(8, 5))

    zoom_bins = np.linspace(zoom_xmin, zoom_xmax, 41)
    n_below = int(np.sum(all_margins < zoom_xmin))
    ax.hist(correct, bins=zoom_bins, alpha=0.6, label=f"Correct (n={len(correct)})", color="#2ca02c")
    ax.hist(wrong, bins=zoom_bins, alpha=0.6, label=f"Wrong (n={len(wrong)})", color="#d62728")
    ax.set_xlim(zoom_xmin, zoom_xmax)
    ax.set_xlabel("Margin |prob_inconsistent − prob_consistent|")
    ax.set_ylabel("Count")
    zoom_title = f"Zoomed [{zoom_xmin:.2f}, {zoom_xmax:.4f}]"
    if n_below > 0:
        zoom_title += f"  ({n_below} rows with margin < {zoom_xmin:.2f} in panel above)"
    ax.set_title(zoom_title if full_xmax is not None else "Token probability margin by outcome")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    if full_xmax is not None:
        fig.suptitle("Token probability margin by outcome", y=1.01)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved margin histogram to: {output_path}")


def plot_confidence_accuracy(conf_stats: Dict[int, Dict[str, float]], output_path: str) -> None:
    _plot_confidence_metric_bars(
        conf_stats,
        output_path,
        metric_key="acc",
        ylabel="Accuracy (%)",
        title="Accuracy by LLM confidence score",
        color="#9467bd",
    )


def plot_confidence_bal_bal_acc(conf_stats: Dict[int, Dict[str, float]], output_path: str) -> None:
    _plot_confidence_metric_bars(
        conf_stats,
        output_path,
        metric_key="bal_bal",
        ylabel="Balanced-balanced accuracy (%)",
        title="Balanced-balanced accuracy by LLM confidence score",
        color="#17becf",
    )


def plot_confidence_fpr(conf_stats: Dict[int, Dict[str, float]], output_path: str) -> None:
    _plot_confidence_metric_bars(
        conf_stats,
        output_path,
        metric_key="fpr",
        ylabel="False positive rate (%)",
        title="False positive rate by LLM confidence score",
        color="#d62728",
    )


def plot_confidence_recall(conf_stats: Dict[int, Dict[str, float]], output_path: str) -> None:
    _plot_confidence_metric_bars(
        conf_stats,
        output_path,
        metric_key="det",
        ylabel="Recall / TPR (%)",
        title="Inconsistent recall (TPR) by LLM confidence score",
        color="#ff7f0e",
    )


def plot_confidence_tnr(conf_stats: Dict[int, Dict[str, float]], output_path: str) -> None:
    _plot_confidence_metric_bars(
        conf_stats,
        output_path,
        metric_key="con",
        ylabel="TNR (%)",
        title="True negative rate (TNR) by LLM confidence score",
        color="#2ca02c",
    )


def plot_confidence_threshold_curve(rows: Sequence[Dict], output_path: str) -> None:
    if not _has_confidence_column(rows):
        print("Skipping confidence threshold curve (no confidence column).")
        return
    conf_rows = [r for r in rows if _row_confidence(r) >= CONFIDENCE_SCORE_MIN]
    if not conf_rows:
        print("Skipping confidence threshold curve (no rows with valid confidence).")
        return

    curve = _confidence_threshold_metrics(conf_rows)
    taus = [int(p["tau"]) for p in curve]
    coverages = [p["coverage"] for p in curve]
    bal_bals = [100.0 * p["bal_bal"] for p in curve]
    accs = [100.0 * p["acc"] for p in curve]
    dets = [100.0 * p["det"] for p in curve]
    cons = [100.0 * p["con"] for p in curve]
    fprs = [100.0 * p["fpr"] for p in curve]

    fig, ax_left = plt.subplots(figsize=(8, 5))
    ax_right = ax_left.twinx()
    x = np.array(taus, dtype=float)
    ax_left.plot(x, bal_bals, marker="o", color="#17becf", linewidth=2.5, label="Bal-balanced acc")
    ax_left.plot(x, accs, marker="s", color="#9467bd", linewidth=1.8, label="Accuracy")
    ax_left.plot(x, dets, marker="^", color="#ff7f0e", linewidth=1.8, label="Recall (TPR)")
    ax_left.plot(x, cons, marker="v", color="#2ca02c", linewidth=1.8, label="TNR")
    ax_left.plot(x, fprs, marker="x", color="#d62728", linewidth=1.8, label="FPR")
    ax_right.plot(
        x,
        coverages,
        marker="D",
        color="#ff7f0e",
        linewidth=2,
        linestyle="--",
        label="Coverage",
    )

    ax_left.set_xlabel("Minimum confidence threshold (keep rows with confidence ≥ τ)")
    ax_left.set_ylabel("Rate (%)")
    ax_right.set_ylabel("Coverage (%)")
    ax_left.set_title("Selective prediction by LLM self-reported confidence")
    ax_left.set_xticks(x, labels=[str(t) for t in taus])
    ax_left.set_ylim(0, 100)
    ax_right.set_ylim(0, 102)
    ax_left.grid(True, alpha=0.3)

    for tau, bal_bal, cov, n in zip(taus, bal_bals, coverages, [int(p["n"]) for p in curve]):
        if bal_bal != bal_bal:
            continue
        ax_left.annotate(
            f"n={n}",
            (tau, bal_bal),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            color="#17becf",
        )
        ax_right.annotate(
            f"{cov:.0f}%",
            (tau, cov),
            textcoords="offset points",
            xytext=(0, -12),
            ha="center",
            fontsize=8,
            color="#ff7f0e",
        )

    handles_left, labels_left = ax_left.get_legend_handles_labels()
    handles_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(handles_left + handles_right, labels_left + labels_right, loc="lower left")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved confidence threshold curve to: {output_path}")


def plot_confidence_by_error_type(rows: Sequence[Dict], output_path: str) -> None:
    if not _has_confidence_column(rows):
        print("Skipping confidence-by-error-type plot (no confidence column).")
        return
    conf_rows = [r for r in rows if _row_confidence(r) >= CONFIDENCE_SCORE_MIN]
    if not conf_rows:
        print("Skipping confidence-by-error-type plot (no rows with valid confidence).")
        return

    error_types, stats = _confidence_by_error_type(conf_rows)
    if not error_types:
        print("Skipping confidence-by-error-type plot (no grouped stats).")
        return

    levels = list(range(CONFIDENCE_SCORE_MIN, CONFIDENCE_SCORE_MAX + 1))
    matrix = np.full((len(error_types), len(levels)), np.nan, dtype=float)
    counts = np.zeros((len(error_types), len(levels)), dtype=int)
    for i, error_type in enumerate(error_types):
        for j, conf in enumerate(levels):
            cell = stats.get(error_type, {}).get(conf)
            if cell is None:
                continue
            matrix[i, j] = 100.0 * cell["acc"]
            counts[i, j] = int(cell["n"])

    ylabels = [_short_error_type_label(et) for et in error_types]
    fig_h = max(4.5, 0.55 * len(error_types) + 1.5)
    fig, ax = plt.subplots(figsize=(8.5, fig_h))
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="#eeeeee")
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=100)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Accuracy within bin (%)")

    ax.set_xticks(np.arange(len(levels)), labels=[str(c) for c in levels])
    ax.set_yticks(np.arange(len(error_types)), labels=ylabels)
    ax.set_xlabel("Self-reported confidence")
    ax.set_ylabel("Error type")
    ax.set_title("Accuracy by error type and confidence")

    for i in range(len(error_types)):
        for j in range(len(levels)):
            n = counts[i, j]
            if n == 0:
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="#666")
                continue
            val = matrix[i, j]
            text = f"{val:.0f}%\nn={n}" if val == val else f"n={n}"
            color = "white" if val == val and val < 45 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confidence-by-error-type plot to: {output_path}")


def plot_confidence_metrics(conf_stats: Dict[int, Dict[str, float]], output_path: str) -> None:
    if not conf_stats:
        print("Skipping confidence metrics plot (no confidence column).")
        return
    levels = list(range(CONFIDENCE_SCORE_MIN, CONFIDENCE_SCORE_MAX + 1))
    tprs: List[float] = []
    bal_bals: List[float] = []
    fprs: List[float] = []
    ns: List[int] = []
    for conf in levels:
        stats = conf_stats.get(conf)
        if stats is None or int(stats.get("n", 0)) == 0:
            ns.append(0)
            tprs.append(float("nan"))
            bal_bals.append(float("nan"))
            fprs.append(float("nan"))
        else:
            ns.append(int(stats["n"]))
            tprs.append(100.0 * stats["det"])
            bal_bals.append(100.0 * stats["bal_bal"])
            fprs.append(100.0 * stats["fpr"])

    x = np.array(levels, dtype=float)
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars_tpr = ax.bar(
        x - width,
        [0.0 if v != v else v for v in tprs],
        width,
        label="Recall (TPR)",
        color="#1f77b4",
        alpha=0.85,
    )
    bars_bal = ax.bar(
        x,
        [0.0 if v != v else v for v in bal_bals],
        width,
        label="Bal-balanced acc",
        color="#9467bd",
        alpha=0.85,
    )
    bars_fpr = ax.bar(
        x + width,
        [0.0 if v != v else v for v in fprs],
        width,
        label="False positive rate",
        color="#d62728",
        alpha=0.85,
    )
    ax.set_xlabel("Self-reported confidence")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(x, labels=[str(c) for c in levels])
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left")
    for bar_group in (bars_tpr, bars_bal, bars_fpr):
        for bar in bar_group:
            height = bar.get_height()
            if height <= 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 1,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    for xi, n in zip(x, ns):
        ax.text(xi, 1, f"n={n}", ha="center", va="bottom", fontsize=8, color="#444")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved confidence metrics plot to: {output_path}")


def plot_confidence_detection_metrics(
    conf_stats: Dict[int, Dict[str, float]],
    output_path: str,
) -> None:
    """Grouped bars: TPR (recall) and FPR by self-reported confidence."""
    if not conf_stats:
        print("Skipping confidence detection metrics plot (no confidence column).")
        return
    metric_specs = (
        ("det", "Recall (TPR)", "#2ca02c"),
        ("fpr", "FPR", "#d62728"),
    )
    levels = list(range(CONFIDENCE_SCORE_MIN, CONFIDENCE_SCORE_MAX + 1))
    ns: List[int] = []
    series: List[List[float]] = [[] for _ in metric_specs]
    for conf in levels:
        stats = conf_stats.get(conf)
        if stats is None or int(stats.get("n", 0)) == 0:
            ns.append(0)
            for vals in series:
                vals.append(float("nan"))
        else:
            ns.append(int(stats["n"]))
            for i, (key, _, _) in enumerate(metric_specs):
                series[i].append(100.0 * stats[key])

    x = np.array(levels, dtype=float)
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    offsets = (-width / 2, width / 2)
    bar_groups = []
    for offset, (vals, (_, label, color)) in zip(offsets, zip(series, metric_specs)):
        bars = ax.bar(
            x + offset,
            [0.0 if v != v else v for v in vals],
            width,
            label=label,
            color=color,
            alpha=0.85,
        )
        bar_groups.append(bars)
    ax.set_xlabel("Self-reported confidence")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(x, labels=[str(c) for c in levels])
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left")
    for bars in bar_groups:
        for bar in bars:
            height = bar.get_height()
            if height <= 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 1,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    for xi, n in zip(x, ns):
        ax.text(xi, 1, f"n={n}", ha="center", va="bottom", fontsize=8, color="#444")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved confidence detection metrics plot to: {output_path}")


def _plot_confidence_metric_bars(
    conf_stats: Dict[int, Dict[str, float]],
    output_path: str,
    *,
    metric_key: str,
    ylabel: str,
    title: str,
    color: str,
) -> None:
    if not conf_stats:
        print(f"Skipping confidence plot for {metric_key} (no confidence column).")
        return
    levels = list(range(CONFIDENCE_SCORE_MIN, CONFIDENCE_SCORE_MAX + 1))
    values: List[float] = []
    ns: List[int] = []
    for conf in levels:
        stats = conf_stats.get(conf)
        if stats is None or int(stats.get("n", 0)) == 0:
            ns.append(0)
            values.append(float("nan"))
        else:
            ns.append(int(stats["n"]))
            values.append(100.0 * stats[metric_key])

    fig, ax = plt.subplots(figsize=(6.5, 5))
    bar_colors = ["#c7c7c7" if n == 0 else color for n in ns]
    x = np.array(levels, dtype=float)
    bars = ax.bar(x, [0.0 if v != v else v for v in values], width=0.7, color=bar_colors, alpha=0.85)
    ax.set_xlabel("Self-reported confidence")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.set_xticks(x, labels=[str(c) for c in levels])
    ax.grid(True, axis="y", alpha=0.3)
    for bar, n, val in zip(bars, ns, values):
        label = f"n={n}" if n == 0 else f"n={n}\n{val:.1f}%"
        y_pos = 1 if n == 0 else (val if val == val else 1)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos + 1,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved confidence plot to: {output_path}")


def plot_confusion_at_thresholds(
    rows: Sequence[Dict],
    output_path: str,
    *,
    thresholds: Sequence[float],
    labels: Sequence[str],
    multiclass: bool = False,
) -> None:
    n_panels = len(thresholds)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    for ax, threshold, label in zip(axes, thresholds, labels):
        patched = _rows_with_pred_label(rows, lambda r, t=threshold: _prob_pred_label(r, threshold=t))
        overall = _overall_counts(patched, multiclass=multiclass)
        matrix = [
            [overall["tp0"], overall["fn0"]],
            [overall["fp0"], overall["tn0"]],
        ]
        im = ax.imshow(matrix, cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks([0, 1], labels=["Pred inc", "Pred con"])
        ax.set_yticks([0, 1], labels=["True inc", "True con"])
        acc = _pct(overall["correct"], overall["n_metric"])
        ax.set_title(f"{label}\nth={threshold:.4g}, acc={acc:.1f}%")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i][j]), ha="center", va="center", color="black")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved confusion matrices to: {output_path}")


def _append_by_error_type_section(
    lines: List[str],
    rows: Sequence[Dict],
    *,
    section_title: str,
    pred_fn: Callable[[Dict], int],
    multiclass: bool = False,
) -> None:
    patched = _rows_with_pred_label(rows, pred_fn)
    by_type = summarize_by_error_type(patched, multiclass=multiclass)
    lines.append("")
    lines.append(section_title)
    lines.append(f"{'error_type':>18} {'N':>6} {'metric':>10} {'rate':>10} {'class_recall':>14}")
    for error_type, stats in by_type.items():
        n = int(stats["n"])
        if error_type == "consistent":
            metric_val = int(stats.get("consistent_recall", 0))
        elif multiclass:
            metric_val = int(stats.get("detected", 0))
        else:
            metric_val = int(stats.get("detected", 0))
        rate = _fmt_pct(metric_val, n)
        class_recall_pct = float(stats.get("class_recall_pct", stats.get("accuracy_pct", float("nan"))))
        class_recall_str = (
            "nan" if class_recall_pct != class_recall_pct else f"{class_recall_pct:.1f}%"
        )
        lines.append(
            f"{error_type:>18} {n:>6} {metric_val:>10} {rate:>10} {class_recall_str:>14}"
        )


def write_prob_eval_report(
    predictions_csv: str,
    output_report: str,
    *,
    run_name: str,
    eval_mode: str = "hard",
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    margin_cutoffs: Sequence[float] = DEFAULT_MARGIN_CUTOFFS,
    prompt: Optional[str] = None,
) -> ProbEvalResult:
    rows_all = _load_predictions(predictions_csv)
    if not _has_prob_columns(rows_all):
        raise ValueError(
            "predictions CSV must contain prob_inconsistent and prob_consistent columns"
        )
    multiclass = _resolve_multiclass(rows_all, prompt)
    if multiclass:
        raise ValueError(
            "predictions_prob_eval currently supports binary predictions only"
        )

    rows = _metric_rows(rows_all)
    has_confidence = _has_confidence_column(rows_all)
    baseline_patched = _rows_with_pred_label(rows, lambda r: _safe_int(r.get("pred_label", ""), default=0))
    baseline_overall = _overall_counts(baseline_patched, multiclass=False)
    baseline_by_type = summarize_by_error_type(baseline_patched, multiclass=False)
    baseline = _overall_summary_dict(baseline_overall, baseline_by_type)

    disagree = _json_prob_disagreement_count(rows)
    scores, labels = _inconsistent_positive_labels(rows)
    auroc = float("nan")
    auprc = float("nan")
    if roc_auc_score is not None and len(np.unique(labels)) >= 2:
        auroc = float(roc_auc_score(labels, scores))
    if average_precision_score is not None and len(np.unique(labels)) >= 2:
        auprc = float(average_precision_score(labels, scores))
    ece = _compute_ece(labels, scores)

    best_bal_t, best_bal, best_f1_t = _search_best_threshold(rows, multiclass=False)
    best_bal_metrics = _metrics_at_threshold(rows, best_bal_t)
    high_margin_cutoff = 0.99
    high_margin_rows = _rows_passing_filter(rows, lambda r: _row_margin(r) >= high_margin_cutoff)
    high_margin_metrics = _metrics_for_pred_fn(
        high_margin_rows,
        lambda r: _safe_int(r.get("pred_label", ""), default=0),
    ) if high_margin_rows else None

    conf_stats = _confidence_metrics(rows) if has_confidence else {}

    lines: List[str] = []
    lines.append("PROB/CONFIDENCE EVAL SUMMARY")
    lines.append(f"Run: {run_name}")
    lines.append(f"Predictions: {predictions_csv}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Eval mode: {eval_mode}")
    lines.append("")
    lines.append("BASELINE (JSON / parsed pred_label)")
    lines.append(f"rows total:            {len(rows_all)}")
    lines.append(f"metrics rows:          {baseline['n']}")
    lines.append(f"overall accuracy:      {_fmt_pct_float(baseline['acc'])}")
    lines.append(f"overall f1:            {100.0 * baseline['macro_f1']:.1f}%")
    lines.append(f"inconsistent recall:   {_fmt_pct_float(baseline['det'])}")
    lines.append(f"consistent recall:     {_fmt_pct_float(baseline['con'])}")
    lines.append(f"balanced acc (50/50):  {_fmt_pct_float(baseline['bal'])}")
    lines.append(f"macro error recall:    {_fmt_pct_float(baseline['macro_err'])}")
    lines.append(f"balanced-balanced acc: {_fmt_pct_float(baseline['bal_bal'])}")
    lines.append("")
    lines.append("PROB SCORE ANALYSIS")
    lines.append(
        f"json vs prob-argmax disagree: {disagree} "
        f"({_fmt_pct(disagree, len(rows))})"
    )
    if auroc == auroc:
        lines.append(f"AUC-ROC (prob_inconsistent): {auroc:.4f}")
    else:
        lines.append("AUC-ROC (prob_inconsistent): nan")
    if auprc == auprc:
        lines.append(f"AUC-PR (prob_inconsistent):  {auprc:.4f}")
    else:
        lines.append("AUC-PR (prob_inconsistent):  nan")
    lines.append(f"ECE (prob_inconsistent):     {ece:.4f}" if ece == ece else "ECE: nan")
    lines.append(
        f"best balanced-balanced threshold: t={best_bal_t:.6g} → "
        f"acc={_fmt_pct_float(best_bal_metrics['acc'])} "
        f"det={_fmt_pct_float(best_bal_metrics['det'])} "
        f"con={_fmt_pct_float(best_bal_metrics['con'])} "
        f"bal_bal={_fmt_pct_float(best_bal_metrics['bal_bal'])}"
    )
    lines.append(f"best inconsistent-F1 threshold: t={best_f1_t:.6g}")
    lines.append("")
    lines.append("THRESHOLD SWEEP (predict inconsistent if prob_inconsistent ≥ t)")
    lines.append(f"{'threshold':>10} {'acc':>8} {'det':>8} {'con':>8} {'bal':>8} {'bal_bal':>8}")
    for threshold in thresholds:
        m = _metrics_at_threshold(rows, threshold)
        lines.append(
            f"{threshold:10.4g} "
            f"{100.0 * m['acc']:7.1f}% "
            f"{100.0 * m['det']:7.1f}% "
            f"{100.0 * m['con']:7.1f}% "
            f"{100.0 * m['bal']:7.1f}% "
            f"{100.0 * m['bal_bal']:7.1f}%"
        )

    lines.append("")
    lines.append("SELECTIVE PREDICTION (subset with margin ≥ cutoff; JSON label kept)")
    lines.append(f"{'margin':>10} {'coverage':>10} {'acc':>8} {'det':>8} {'con':>8} {'bal':>8}")
    n_total = len(rows)
    for cutoff in margin_cutoffs:
        kept = _rows_passing_filter(rows, lambda r, c=cutoff: _row_margin(r) >= c)
        coverage = 100.0 * len(kept) / max(n_total, 1)
        if not kept:
            lines.append(f"{cutoff:10.4g} {coverage:9.1f}%      — no rows")
            continue
        m = _metrics_for_pred_fn(
            kept,
            lambda r: _safe_int(r.get("pred_label", ""), default=0),
        )
        lines.append(
            f"{cutoff:10.4g} {coverage:9.1f}% "
            f"{100.0 * m['acc']:7.1f}% "
            f"{100.0 * m['det']:7.1f}% "
            f"{100.0 * m['con']:7.1f}% "
            f"{100.0 * m['bal']:7.1f}%"
        )

    if has_confidence:
        conf_rows = [r for r in rows if _row_confidence(r) >= CONFIDENCE_SCORE_MIN]
        lines.append("")
        lines.append("CONFIDENCE ANALYSIS")
        lines.append(
            f"{'confidence':>10} {'N':>8} {'accuracy':>10} {'bal_bal':>10} "
            f"{'recall':>10} {'tnr':>10} {'fpr':>10}"
        )
        for conf in range(CONFIDENCE_SCORE_MIN, CONFIDENCE_SCORE_MAX + 1):
            stats = conf_stats.get(conf)
            if stats is None or int(stats.get("n", 0)) == 0:
                lines.append(f"{conf:10d} {0:8d}        —        —        —        —        —")
            else:
                lines.append(
                    f"{conf:10d} {int(stats['n']):8d} "
                    f"{100.0 * stats['acc']:9.1f}% "
                    f"{100.0 * stats['bal_bal']:9.1f}% "
                    f"{100.0 * stats['det']:9.1f}% "
                    f"{100.0 * stats['con']:9.1f}% "
                    f"{100.0 * stats['fpr']:9.1f}%"
                )

        lines.append("")
        lines.append(
            "CONFIDENCE THRESHOLD (keep rows with confidence ≥ τ; JSON label kept)"
        )
        lines.append(
            f"{'tau':>6} {'N':>8} {'coverage':>10} {'acc':>8} {'recall':>8} "
            f"{'tnr':>8} {'fpr':>8} {'bal_bal':>8}"
        )
        for point in _confidence_threshold_metrics(conf_rows):
            if int(point["n"]) == 0:
                lines.append(
                    f"{int(point['tau']):6d} {0:8d} "
                    f"{point['coverage']:9.1f}%      — no rows"
                )
                continue
            lines.append(
                f"{int(point['tau']):6d} {int(point['n']):8d} "
                f"{point['coverage']:9.1f}% "
                f"{100.0 * point['acc']:7.1f}% "
                f"{100.0 * point['det']:7.1f}% "
                f"{100.0 * point['con']:7.1f}% "
                f"{100.0 * point['fpr']:7.1f}% "
                f"{100.0 * point['bal_bal']:7.1f}%"
            )

        error_types, by_type_conf = _confidence_by_error_type(conf_rows)
        if error_types:
            lines.append("")
            lines.append("CONFIDENCE BY ERROR TYPE (accuracy within bin)")
            header = f"{'error_type':>18}"
            for conf in range(CONFIDENCE_SCORE_MIN, CONFIDENCE_SCORE_MAX + 1):
                header += f" {conf:>8}"
            lines.append(header)
            for error_type in error_types:
                row_parts = [f"{error_type:>18}"]
                for conf in range(CONFIDENCE_SCORE_MIN, CONFIDENCE_SCORE_MAX + 1):
                    cell = by_type_conf.get(error_type, {}).get(conf)
                    if cell is None or int(cell.get("n", 0)) == 0:
                        row_parts.append(f"{'—':>8}")
                    else:
                        row_parts.append(
                            f"{100.0 * cell['acc']:6.1f}%"
                        )
                lines.append("".join(row_parts))

    lines.append("")
    lines.append("CALIBRATION BINS (prob_inconsistent)")
    lines.append(f"{'bin_lo':>8} {'bin_hi':>8} {'N':>8} {'mean_pred':>10} {'actual_rate':>12}")
    for b in _calibration_bins(rows):
        lines.append(
            f"{b['lo']:8.2f} {b['hi']:8.2f} {int(b['n']):8d} "
            f"{b['mean_pred']:10.4f} {b['actual_rate']:12.4f}"
        )

    lines.append("")
    lines.append("OPERATING POINT COMPARISON")
    lines.append(
        f"{'point':<22} {'N':>8} {'acc':>8} {'det':>8} {'con':>8} {'bal_bal':>8}"
    )
    points = [
        ("default (JSON)", baseline, baseline["n"]),
        (f"best bal (t={best_bal_t:.4g})", best_bal_metrics, baseline["n"]),
    ]
    if high_margin_metrics is not None:
        points.append(
            (f"margin≥{high_margin_cutoff}", high_margin_metrics, high_margin_metrics["n"])
        )
    for name, m, n in points:
        lines.append(
            f"{name:<22} {int(n):8d} "
            f"{100.0 * m['acc']:7.1f}% "
            f"{100.0 * m['det']:7.1f}% "
            f"{100.0 * m['con']:7.1f}% "
            f"{100.0 * m['bal_bal']:7.1f}%"
        )

    _append_by_error_type_section(
        lines,
        rows,
        section_title="BY ERROR TYPE @ default (JSON label)",
        pred_fn=lambda r: _safe_int(r.get("pred_label", ""), default=0),
    )
    _append_by_error_type_section(
        lines,
        rows,
        section_title=f"BY ERROR TYPE @ best bal threshold (t={best_bal_t:.4g})",
        pred_fn=lambda r, t=best_bal_t: _prob_pred_label(r, threshold=t),
    )
    if high_margin_rows:
        _append_by_error_type_section(
            lines,
            high_margin_rows,
            section_title=f"BY ERROR TYPE @ margin≥{high_margin_cutoff}",
            pred_fn=lambda r: _safe_int(r.get("pred_label", ""), default=0),
        )

    lines.append("")
    lines.append("NOTES")
    lines.append("- prob_inconsistent / prob_consistent come from true/false token logprobs in the JSON response.")
    lines.append("- prob-argmax at t=0.5 may match JSON pred_label exactly when logprobs align with parsed boolean.")
    lines.append("- Selective prediction keeps JSON labels but restricts metrics to high-margin subsets.")
    lines.append("- Positive detection score = prob_inconsistent; inconsistent class is true_label=0.")

    out_text = "\n".join(lines) + "\n"
    Path(output_report).write_text(out_text, encoding="utf-8")
    print(out_text)
    print(f"Saved report to: {output_report}")

    return ProbEvalResult(
        predictions_csv=predictions_csv,
        run_name=run_name,
        n_total=len(rows_all),
        n_metric=len(rows),
        has_confidence=has_confidence,
        json_prob_disagree=disagree,
        auroc=auroc,
        auprc=auprc,
        best_balanced_threshold=best_bal_t,
        best_f1_threshold=best_f1_t,
        summary_path=output_report,
    )


def write_prob_eval_artifacts(
    predictions_csv: str,
    *,
    run_name: str,
    eval_mode: str = "hard",
    summary_path: Optional[str] = None,
    errors_csv_path: Optional[str] = None,
    plots_dir: Optional[Path] = None,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    margin_cutoffs: Sequence[float] = DEFAULT_MARGIN_CUTOFFS,
    prompt: Optional[str] = None,
    write_plots: bool = True,
    write_errors_csv: bool = True,
) -> ProbEvalResult:
    out_dir = Path(summary_path).resolve().parent if summary_path else Path(predictions_csv).resolve().parent
    plot_root = plots_dir or out_dir.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_root.mkdir(parents=True, exist_ok=True)

    summary_out = summary_path or str(out_dir / "prob_confidence_eval.txt")
    result = write_prob_eval_report(
        predictions_csv,
        summary_out,
        run_name=run_name,
        eval_mode=eval_mode,
        thresholds=thresholds,
        margin_cutoffs=margin_cutoffs,
        prompt=prompt,
    )

    rows = _metric_rows(_load_predictions(predictions_csv))
    best_bal_t, _, _ = _search_best_threshold(rows)

    if write_errors_csv:
        errors_out = errors_csv_path or str(out_dir / "prob_confidence_errors.csv")
        forensics = _collect_forensics_rows(rows)
        _write_forensics_csv(forensics, errors_out)
        result.errors_csv_path = errors_out

    if write_plots:
        conf_stats = _confidence_metrics(rows) if _has_confidence_column(rows) else {}
        plot_paths = [
            str(plot_root / "roc_prob_inconsistent.png"),
            str(plot_root / "pr_prob_inconsistent.png"),
            str(plot_root / "roc_prob_consistent.png"),
            str(plot_root / "pr_prob_consistent.png"),
            str(plot_root / "metrics_vs_threshold.png"),
            str(plot_root / "metrics_vs_threshold_consistent.png"),
            str(plot_root / "selective_margin_curve.png"),
            str(plot_root / "calibration_reliability.png"),
            str(plot_root / "calibration_reliability_consistent.png"),
            str(plot_root / "calibration_reliability_dual.png"),
            str(plot_root / "margin_by_outcome.png"),
            str(plot_root / "confidence_accuracy.png"),
            str(plot_root / "confidence_bal_bal_acc.png"),
            str(plot_root / "confidence_fpr.png"),
            str(plot_root / "confidence_recall.png"),
            str(plot_root / "confidence_tnr.png"),
            str(plot_root / "confidence_metrics.png"),
            str(plot_root / "confidence_detection_metrics.png"),
            str(plot_root / "confidence_threshold_curve.png"),
            str(plot_root / "confidence_by_error_type.png"),
            str(plot_root / "confusion_at_thresholds.png"),
        ]
        plot_roc_curve(rows, plot_paths[0])
        plot_pr_curve(rows, plot_paths[1])
        plot_roc_curve_consistent(rows, plot_paths[2])
        plot_pr_curve_consistent(rows, plot_paths[3])
        plot_metrics_vs_threshold(rows, plot_paths[4], thresholds=thresholds, score="inconsistent")
        plot_metrics_vs_threshold(rows, plot_paths[5], thresholds=thresholds, score="consistent")
        plot_selective_margin_curve(rows, plot_paths[6], margin_cutoffs=margin_cutoffs)
        plot_calibration_reliability(rows, plot_paths[7])
        plot_calibration_reliability_consistent(rows, plot_paths[8])
        plot_calibration_reliability_dual(rows, plot_paths[9])
        plot_margin_by_outcome(rows, plot_paths[10])
        plot_confidence_accuracy(conf_stats, plot_paths[11])
        plot_confidence_bal_bal_acc(conf_stats, plot_paths[12])
        plot_confidence_fpr(conf_stats, plot_paths[13])
        plot_confidence_recall(conf_stats, plot_paths[14])
        plot_confidence_tnr(conf_stats, plot_paths[15])
        plot_confidence_metrics(conf_stats, plot_paths[16])
        plot_confidence_detection_metrics(conf_stats, plot_paths[17])
        plot_confidence_threshold_curve(rows, plot_paths[18])
        plot_confidence_by_error_type(rows, plot_paths[19])
        plot_confusion_at_thresholds(
            rows,
            plot_paths[20],
            thresholds=(0.5, best_bal_t, 0.99),
            labels=("Default (t=0.5)", f"Best bal (t={best_bal_t:.4g})", "High t=0.99"),
        )
        result.plot_paths = plot_paths

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prob/confidence evaluation from LLM predictions.csv."
    )
    parser.add_argument("--predictions_csv", type=str, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        default="hard",
        choices=["all", "lat", "site", "severity", "hard", "cross_pair_full"],
        help="Eval mode label recorded in the report (metrics are from full CSV).",
    )
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument(
        "--output_run_dir",
        type=str,
        default=None,
        help="Run root for results/, plots/, logs/. Default: infer from predictions_csv.",
    )
    parser.add_argument(
        "--output_report",
        type=str,
        default=None,
        help="Defaults to prob_confidence_eval.txt under results/.",
    )
    parser.add_argument(
        "--errors_csv",
        type=str,
        default=None,
        help="Defaults to prob_confidence_errors.csv under results/.",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=RUN_OUTPUT_LOG,
        help="Log filename under logs/ (default: output.log).",
    )
    parser.add_argument(
        "--command_stem",
        type=str,
        default="command",
        help="Basename for command.txt / command_argv.txt in the run root.",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip writing plots under plots/.",
    )
    parser.add_argument(
        "--no_errors_csv",
        action="store_true",
        help="Skip writing prob_confidence_errors.csv.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt variant (used only to detect multiclass; binary expected).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_run_dir = args.output_run_dir
    if not output_run_dir:
        inferred_root = infer_run_root_from_predictions(args.predictions_csv)
        output_run_dir = str(
            inferred_root or Path(args.predictions_csv).resolve().parent
        )
    run_layout = setup_run(
        SCRIPT_ID,
        tags=("report", "prob", args.mode),
        run_name=args.run_name,
        run_category=RUN_CATEGORY,
        output_model_dir=output_run_dir,
        command_stem=args.command_stem,
        argv=current_argv(),
    )
    args.output_report = default_result_path(
        run_layout, "prob_confidence_eval.txt", args.output_report
    )
    args.errors_csv = default_result_path(
        run_layout, "prob_confidence_errors.csv", args.errors_csv
    )
    report_run_name = args.run_name or run_layout.root.name
    results_dir = Path(args.output_report).resolve().parent
    plots_dir = run_layout.plots

    with tee_run_output(run_layout.log_path(args.log_file)):
        print(f"Run outputs: {run_layout.root}")
        write_prob_eval_artifacts(
            predictions_csv=args.predictions_csv,
            run_name=report_run_name,
            eval_mode=args.mode,
            summary_path=args.output_report,
            errors_csv_path=args.errors_csv,
            plots_dir=plots_dir,
            prompt=args.prompt,
            write_plots=not args.no_plots,
            write_errors_csv=not args.no_errors_csv,
        )


if __name__ == "__main__":
    main()
