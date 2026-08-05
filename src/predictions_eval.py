"""Unified evaluation from row-level predictions CSV (BERT + LLM, binary + multiclass)."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


SCRIPT_ID = "predictions-eval"
RUN_CATEGORY = "llm_baseline"

try:
    from src.llm_eval_prompts import (
        MULTICLASS_LABEL_NAMES,
        MULTICLASS_N_CLASSES,
        is_multiclass_prompt,
        label_from_error_type,
    )
    from src.nf_swap_eval import (
        format_nf_swap_direction_lines,
        summarize_nf_swap_by_direction,
    )
except ImportError:
    import sys

    _repo_root = Path(__file__).resolve().parents[1]
    if str(_repo_root) not in sys.path:
        sys.path.append(str(_repo_root))
    from src.llm_eval_prompts import (
        MULTICLASS_LABEL_NAMES,
        MULTICLASS_N_CLASSES,
        is_multiclass_prompt,
        label_from_error_type,
    )
    from src.nf_swap_eval import (  # type: ignore
        format_nf_swap_direction_lines,
        summarize_nf_swap_by_direction,
    )


SourceKind = Literal["auto", "llm", "bert"]
TaskKind = Literal["binary", "multiclass"]


@dataclass
class EvalResult:
    predictions_csv: str
    source: str
    task: TaskKind
    run_name: str
    eval_mode: str
    eval_csv: Optional[str]
    overall: Dict[str, int]
    by_error_type: Dict[str, Dict[str, object]]
    multiclass_metrics: Optional[Dict[str, object]] = None
    nf_swap_by_direction: Optional[Dict[str, Dict[str, float]]] = None
    summary_path: Optional[str] = None
    incidence_reports: Dict[str, str] = field(default_factory=dict)


def _resolve_source(rows: List[Dict], source: SourceKind) -> str:
    if source != "auto":
        return source
    if _has_parse_ok_column(rows):
        return "llm"
    return "bert"


def _nf_swap_from_rows(
    rows: List[Dict],
    *,
    task: TaskKind,
    eval_csv: Optional[str],
) -> Optional[Dict[str, Dict[str, float]]]:
    valid = _metric_rows(rows)
    if not any((r.get("error_type") or "").strip() == "NF_v_Dis_swap" for r in valid):
        return None
    preds = [_safe_int(r.get("pred_label", ""), default=0) for r in valid]
    labels = [_safe_int(r.get("true_label", ""), default=0) for r in valid]
    return summarize_nf_swap_by_direction(
        valid,
        preds,
        labels,
        task=task,
        eval_csv=eval_csv,
    )


def compute_eval_result(
    predictions_csv: str,
    *,
    source: SourceKind = "auto",
    eval_mode: str = "hard",
    eval_csv: Optional[str] = None,
    run_name: str = "",
    prompt: Optional[str] = None,
) -> EvalResult:
    rows = _load_predictions(predictions_csv)
    resolved_source = _resolve_source(rows, source)
    multiclass = _resolve_multiclass(rows, prompt)
    task: TaskKind = "multiclass" if multiclass else "binary"
    overall = _overall_counts(rows, multiclass=multiclass)
    by_type = summarize_by_error_type(rows, multiclass=multiclass)
    mc_metrics = _compute_multiclass_metrics(rows) if multiclass else None
    nf_swap = _nf_swap_from_rows(rows, task=task, eval_csv=eval_csv)
    return EvalResult(
        predictions_csv=predictions_csv,
        source=resolved_source,
        task=task,
        run_name=run_name,
        eval_mode=eval_mode,
        eval_csv=eval_csv,
        overall=overall,
        by_error_type=by_type,
        multiclass_metrics=mc_metrics,
        nf_swap_by_direction=nf_swap,
    )


def eval_from_predictions(
    predictions_csv: str,
    *,
    source: SourceKind = "auto",
    eval_mode: str = "hard",
    eval_csv: Optional[str] = None,
    run_name: str = "",
    prompt: Optional[str] = None,
) -> EvalResult:
    """Compute metrics dict from a predictions CSV (BERT or LLM)."""
    return compute_eval_result(
        predictions_csv,
        source=source,
        eval_mode=eval_mode,
        eval_csv=eval_csv,
        run_name=run_name,
        prompt=prompt,
    )


def eval_from_llm(
    predictions_csv: str,
    *,
    eval_mode: str = "hard",
    eval_csv: Optional[str] = None,
    run_name: str = "",
    prompt: Optional[str] = None,
) -> EvalResult:
    return eval_from_predictions(
        predictions_csv,
        source="llm",
        eval_mode=eval_mode,
        eval_csv=eval_csv,
        run_name=run_name,
        prompt=prompt,
    )


def eval_from_bert(
    predictions_csv: str,
    *,
    eval_mode: str = "hard",
    eval_csv: Optional[str] = None,
    run_name: str = "",
) -> EvalResult:
    return eval_from_predictions(
        predictions_csv,
        source="bert",
        eval_mode=eval_mode,
        eval_csv=eval_csv,
        run_name=run_name,
    )


def write_eval_artifacts(
    predictions_csv: str,
    *,
    run_name: str,
    eval_mode: str = "hard",
    eval_csv: Optional[str] = None,
    source: SourceKind = "auto",
    prompt: Optional[str] = None,
    summary_path: Optional[str] = None,
    results_dir: Optional[Path] = None,
    plots_dir: Optional[Path] = None,
    write_incidence: bool = True,
    write_confusion: bool = True,
    model_name: Optional[str] = None,
    confusion_plot_path: Optional[str] = None,
) -> EvalResult:
    """Write eval_summary.txt plus optional incidence reports and confusion matrix."""
    result = compute_eval_result(
        predictions_csv,
        source=source,
        eval_mode=eval_mode,
        eval_csv=eval_csv,
        run_name=run_name,
        prompt=prompt,
    )
    out_dir = results_dir or Path(predictions_csv).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_out = summary_path or str(out_dir / "eval_summary.txt")
    write_eval_summary_report(
        predictions_csv=predictions_csv,
        output_report=summary_out,
        run_name=run_name,
        mode=eval_mode,
        eval_csv=eval_csv,
        source=result.source,
        prompt=prompt,
    )
    result.summary_path = summary_out

    if write_incidence:
        plot_root = plots_dir or out_dir.parent / "plots"
        result.incidence_reports = write_incidence_reports_for_eval_mode(
            predictions_csv=predictions_csv,
            run_name=run_name,
            eval_mode=eval_mode,
            results_dir=out_dir,
            plots_dir=plot_root,
            prompt=prompt,
        )

    if write_confusion:
        cm_path = confusion_plot_path or str(
            (plots_dir or out_dir.parent / "plots") / "confusion_matrix.png"
        )
        plot_confusion_matrix_from_predictions(
            predictions_csv=predictions_csv,
            output_path=cm_path,
            mode=eval_mode,
            model_name=model_name,
            prompt=prompt,
        )

    return result


def _swap_incidence_for_example(
    conclusion: str,
    error_type: str,
    *,
    original_conclusion: Optional[str] = None,
    incidence_profile: Optional[str] = None,
) -> str:
    """Lazy import to avoid circular import with bert_inconsistency_detector."""
    try:
        from src.bert_inconsistency_detector import (
            swap_incidence_for_example,
        )
    except ImportError:
        import sys

        _repo_root = Path(__file__).resolve().parents[1]
        if str(_repo_root) not in sys.path:
            sys.path.append(str(_repo_root))
        from src.bert_inconsistency_detector import (
            swap_incidence_for_example,
        )
    return swap_incidence_for_example(
        conclusion,
        error_type,
        original_conclusion=original_conclusion,
        incidence_profile=incidence_profile,
    )


_MULTICLASS_ERROR_NAMES = {
    name for name in MULTICLASS_LABEL_NAMES.values() if name != "consistent"
}

# Binary eval CSVs: 0 = inconsistent (detection-positive), 1 = consistent.
# Consistent recall = TP/(TP+FN) on true_label=1 rows (not confusion-matrix "tn0").
CONSISTENT_LABEL = 1
INCONSISTENT_LABEL = 0


def _safe_int(value: str, default: int = 0) -> int:
    try:
        return int((value or "").strip())
    except Exception:
        return default


def _safe_bool_int(value: str) -> int:
    v = (value or "").strip().lower()
    if v in {"1", "true", "yes"}:
        return 1
    if v in {"0", "false", "no"}:
        return 0
    return _safe_int(value, default=0)


def _has_parse_ok_column(rows: List[Dict]) -> bool:
    return any("parse_ok" in r for r in rows)


def _metric_rows(rows: List[Dict]) -> List[Dict]:
    """Rows used for accuracy/recall/F1; excludes parse failures when parse_ok exists."""
    if not _has_parse_ok_column(rows):
        return rows
    return [r for r in rows if _safe_bool_int(r.get("parse_ok", "0"))]


def _pct(num: int, den: int) -> float:
    if den == 0:
        return float("nan")
    return 100.0 * num / den


def _fmt_pct(num: int, den: int) -> str:
    p = _pct(num, den)
    if p != p:
        return "nan"
    return f"{p:.1f}%"


def _fmt_pct_float(rate: float) -> str:
    if rate != rate:
        return "nan"
    return f"{100.0 * rate:.1f}%"


def _binary_class_recall_rates(overall: Dict[str, int]) -> Tuple[float, float]:
    """Return (inconsistent_recall, consistent_recall) as fractions in [0, 1]."""
    inc_recall = overall["tp0"] / max(overall["n_inconsistent"], 1)
    con_recall = overall["consistent_recall_tp"] / max(overall["n_consistent"], 1)
    return inc_recall, con_recall


def _weighted_binary_class_accuracy(
    overall: Dict[str, int],
    *,
    weight_inconsistent: float,
    weight_consistent: float,
) -> float:
    """Class-weighted accuracy: w_err * recall(errors) + w_ok * recall(no errors)."""
    total_w = weight_inconsistent + weight_consistent
    if total_w <= 0:
        return float("nan")
    inc_recall, con_recall = _binary_class_recall_rates(overall)
    w_inc = weight_inconsistent / total_w
    w_con = weight_consistent / total_w
    return w_inc * inc_recall + w_con * con_recall


def _macro_error_recall_from_by_type(
    by_type: Dict[str, Dict[str, object]],
) -> float:
    """Unweighted mean per-error-type recall (excludes consistent controls).

    Binary: fraction detected inconsistent per error_type bucket.
    Multiclass: exact type-match rate per error_type bucket.
    """
    recalls: List[float] = []
    for error_type, stats in by_type.items():
        if error_type == "consistent":
            continue
        n = int(stats.get("n", 0))
        if n <= 0:
            continue
        detected = int(stats.get("detected", 0))
        recalls.append(detected / n)
    if not recalls:
        return float("nan")
    return sum(recalls) / len(recalls)


def _balanced_balanced_accuracy(
    overall: Dict[str, int],
    by_type: Dict[str, Dict[str, object]],
) -> Tuple[float, float]:
    """Return (macro_error_recall, 0.5*consistent + 0.5*macro_error_recall)."""
    macro_err = _macro_error_recall_from_by_type(by_type)
    _, con_recall = _binary_class_recall_rates(overall)
    if macro_err != macro_err:
        return macro_err, float("nan")
    return macro_err, 0.5 * con_recall + 0.5 * macro_err


def _append_weighted_accuracy_lines(
    lines: List[str],
    overall: Dict[str, int],
    *,
    by_type: Optional[Dict[str, Dict[str, object]]] = None,
) -> None:
    balanced = _weighted_binary_class_accuracy(overall, weight_inconsistent=0.5, weight_consistent=0.5)
    prevalence = _weighted_binary_class_accuracy(
        overall, weight_inconsistent=0.05, weight_consistent=0.95
    )
    lines.append(
        f"weighted accuracy (50/50):  {_fmt_pct_float(balanced)}"
    )
    lines.append(
        f"weighted accuracy (95/5):   {_fmt_pct_float(prevalence)}"
    )
    if by_type is not None:
        macro_err, bal_bal = _balanced_balanced_accuracy(overall, by_type)
        lines.append(f"macro error recall:      {_fmt_pct_float(macro_err)}")
        lines.append(f"balanced-balanced acc:   {_fmt_pct_float(bal_bal)}")


def _class_f1(tp: int, fp: int, fn: int) -> float:
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    return 2 * prec * rec / (prec + rec + 1e-8)


def _binary_macro_f1(tp0: int, fp0: int, fn0: int, tn0: int) -> float:
    """Macro-averaged F1 over inconsistent (0) and consistent (1) classes."""
    f1_inconsistent = _class_f1(tp0, fp0, fn0)
    f1_consistent = _class_f1(tn0, fn0, fp0)
    return 0.5 * (f1_inconsistent + f1_consistent)


def _resolve_multiclass(rows: List[Dict], prompt: Optional[str] = None) -> bool:
    if prompt is not None:
        return is_multiclass_prompt(prompt)
    return _predictions_are_multiclass(rows)


def _compute_multiclass_metrics(rows: List[Dict]) -> Dict[str, object]:
    """Multiclass metrics over parse_ok rows (mirrors eval_llm_baseline)."""
    valid = _metric_rows(rows)
    empty_per_class = {
        c: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0, "tp": 0, "fp": 0, "fn": 0}
        for c in range(MULTICLASS_N_CLASSES)
    }
    if not valid:
        return {
            "acc": 0.0,
            "binary_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "errors_only_type_acc": float("nan"),
            "per_class": empty_per_class,
        }

    preds = [_safe_int(r.get("pred_label", ""), default=0) for r in valid]
    labels = [_safe_int(r.get("true_label", ""), default=0) for r in valid]
    acc = sum(p == l for p, l in zip(preds, labels)) / len(valid)

    binary_preds = [1 if p != 0 else 0 for p in preds]
    binary_labels = [1 if l != 0 else 0 for l in labels]
    tp_bin = sum(1 for bp, bl in zip(binary_preds, binary_labels) if bp == 1 and bl == 1)
    fp_bin = sum(1 for bp, bl in zip(binary_preds, binary_labels) if bp == 1 and bl == 0)
    fn_bin = sum(1 for bp, bl in zip(binary_preds, binary_labels) if bp == 0 and bl == 1)
    bin_prec = tp_bin / (tp_bin + fp_bin + 1e-8)
    bin_rec = tp_bin / (tp_bin + fn_bin + 1e-8)
    bin_f1 = 2 * bin_prec * bin_rec / (bin_prec + bin_rec + 1e-8)

    per_class: Dict[int, Dict[str, object]] = {}
    for c in range(MULTICLASS_N_CLASSES):
        tp = sum(1 for p, l in zip(preds, labels) if p == c and l == c)
        fp = sum(1 for p, l in zip(preds, labels) if p == c and l != c)
        fn = sum(1 for p, l in zip(preds, labels) if p != c and l == c)
        support = sum(1 for l in labels if l == c)
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        per_class[c] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    active = [c for c in range(MULTICLASS_N_CLASSES) if per_class[c]["support"] > 0]
    n_active = max(len(active), 1)
    macro_precision = sum(float(per_class[c]["precision"]) for c in active) / n_active
    macro_recall = sum(float(per_class[c]["recall"]) for c in active) / n_active
    macro_f1 = sum(float(per_class[c]["f1"]) for c in active) / n_active
    total_sup = sum(int(per_class[c]["support"]) for c in active)
    weighted_f1 = (
        sum(float(per_class[c]["f1"]) * int(per_class[c]["support"]) for c in active)
        / max(total_sup, 1)
    )

    error_rows = [r for r in valid if _safe_int(r.get("true_label", ""), default=0) != 0]
    if error_rows:
        errors_only_type_acc = (
            sum(_safe_int(r.get("is_correct", "0"), default=0) for r in error_rows)
            / len(error_rows)
        )
    else:
        errors_only_type_acc = float("nan")

    return {
        "acc": acc,
        "binary_f1": bin_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "errors_only_type_acc": errors_only_type_acc,
        "per_class": per_class,
    }


def summarize_predictions_csv(
    predictions_csv: str,
    mode: str = "all",
    *,
    prompt: Optional[str] = None,
) -> Dict[str, object]:
    """Return paper-friendly summary metrics for one predictions.csv."""
    rows = _load_predictions(predictions_csv)
    multiclass = _resolve_multiclass(rows, prompt)
    overall = _overall_counts(rows, multiclass=multiclass)
    n_inc = overall["n_inconsistent"]
    n_con = overall["n_consistent"]
    has_parse = any("parse_ok" in r for r in rows)
    parse_denom = overall["n_total"] if has_parse else 0

    summary: Dict[str, object] = {
        "predictions_csv": predictions_csv,
        "mode": mode,
        "multiclass": multiclass,
        "n_total": overall["n_total"],
        "accuracy_pct": _pct(overall["correct"], overall["n_metric"]),
        "detection_rate_pct": _pct(overall["tp0"], n_inc),
        "inconsistent_recall_pct": _pct(overall["tp0"], n_inc),
        "consistent_recall_pct": _pct(overall["consistent_recall_tp"], n_con),
        "false_alarm_rate_pct": _pct(overall["fp0"], n_con),
        "n_inconsistent": n_inc,
        "n_consistent": n_con,
        "tp0": overall["tp0"],
        "fp0": overall["fp0"],
        "fn0": overall["fn0"],
        "tn0": overall["tn0"],
        "consistent_recall_tp": overall["consistent_recall_tp"],
        "weighted_accuracy_50_50_pct": 100.0
        * _weighted_binary_class_accuracy(overall, weight_inconsistent=0.5, weight_consistent=0.5),
        "weighted_accuracy_95_5_pct": 100.0
        * _weighted_binary_class_accuracy(overall, weight_inconsistent=0.05, weight_consistent=0.95),
    }
    by_type = summarize_by_error_type(rows, multiclass=multiclass)
    macro_err, bal_bal = _balanced_balanced_accuracy(overall, by_type)
    summary["macro_error_recall_pct"] = 100.0 * macro_err if macro_err == macro_err else float("nan")
    summary["balanced_balanced_accuracy_pct"] = (
        100.0 * bal_bal if bal_bal == bal_bal else float("nan")
    )
    if has_parse:
        summary["parse_ok_pct"] = _pct(overall["parse_ok"], parse_denom)
        summary["parse_ok_n"] = overall["parse_ok"]
    else:
        summary["parse_ok_pct"] = float("nan")
        summary["parse_ok_n"] = 0

    if multiclass:
        mc = _compute_multiclass_metrics(rows)
        summary["macro_precision"] = mc["macro_precision"]
        summary["macro_recall"] = mc["macro_recall"]
        summary["macro_f1"] = mc["macro_f1"]
        summary["weighted_f1"] = mc["weighted_f1"]
        summary["binary_detection_f1"] = mc["binary_f1"]
        summary["errors_only_type_acc"] = mc["errors_only_type_acc"]

    if mode == "lat":
        lat_buckets = _error_swap_incidence_counts(
            rows, "laterality_swap", multiclass=multiclass, incidence_mode="lat"
        )
        n1 = lat_buckets.get(1, {}).get("n", 0)
        d1 = lat_buckets.get(1, {}).get("tp", 0)
        n_ge2, d_ge2 = _sum_subset(lat_buckets, 2)
        summary["lat_swap_n"] = sum(b["n"] for b in lat_buckets.values())
        summary["lat_detection_inc1_pct"] = _pct(d1, n1)
        summary["lat_detection_inc_ge2_pct"] = _pct(d_ge2, n_ge2)
    elif mode == "severity":
        sev_buckets = _error_swap_incidence_counts(
            rows, "severity_flip", multiclass=multiclass, incidence_mode="severity"
        )
        n1 = sev_buckets.get(1, {}).get("n", 0)
        d1 = sev_buckets.get(1, {}).get("tp", 0)
        n_ge2, d_ge2 = _sum_subset(sev_buckets, 2)
        summary["severity_flip_n"] = sum(b["n"] for b in sev_buckets.values())
        summary["severity_detection_inc1_pct"] = _pct(d1, n1)
        summary["severity_detection_inc_ge2_pct"] = _pct(d_ge2, n_ge2)

    summary["by_error_type"] = by_type
    return summary


def summarize_by_error_type(
    rows: List[Dict],
    *,
    multiclass: bool = False,
) -> Dict[str, Dict[str, object]]:
    """Per error_type metrics for paper tables (no incidence buckets)."""
    grouped_all: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        grouped_all[(row.get("error_type") or "unknown").strip()].append(row)

    metric_rows = _metric_rows(rows)
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in metric_rows:
        grouped[(row.get("error_type") or "unknown").strip()].append(row)

    out: Dict[str, Dict[str, object]] = {}
    for error_type in sorted(grouped_all):
        subset = grouped.get(error_type, [])
        n_total = len(grouped_all[error_type])
        n = len(subset)
        correct = sum(
            1
            for r in subset
            if _safe_int(r.get("true_label", ""), default=0)
            == _safe_int(r.get("pred_label", ""), default=0)
        )
        entry: Dict[str, object] = {
            "n_total": n_total,
            "n": n,
            "accuracy_pct": _pct(correct, n),
            "class_recall_pct": _pct(correct, n),
        }
        if error_type == "consistent":
            if multiclass:
                consistent_recall = sum(
                    1
                    for r in subset
                    if _safe_int(r.get("true_label", ""), default=0) == 0
                    and _safe_int(r.get("pred_label", ""), default=1) == 0
                )
                false_alarm = sum(
                    1
                    for r in subset
                    if _safe_int(r.get("true_label", ""), default=0) == 0
                    and _safe_int(r.get("pred_label", ""), default=0) != 0
                )
            else:
                consistent_recall = sum(
                    1
                    for r in subset
                    if _safe_int(r.get("true_label", ""), default=CONSISTENT_LABEL)
                    == CONSISTENT_LABEL
                    and _safe_int(r.get("pred_label", ""), default=INCONSISTENT_LABEL)
                    == CONSISTENT_LABEL
                )
                false_alarm = sum(
                    1
                    for r in subset
                    if _safe_int(r.get("true_label", ""), default=CONSISTENT_LABEL)
                    == CONSISTENT_LABEL
                    and _safe_int(r.get("pred_label", ""), default=INCONSISTENT_LABEL)
                    == INCONSISTENT_LABEL
                )
            entry["consistent_recall"] = consistent_recall
            entry["consistent_recall_pct"] = _pct(consistent_recall, n)
            entry["false_alarm"] = false_alarm
            entry["false_alarm_rate_pct"] = _pct(false_alarm, n)
        elif multiclass:
            expected = label_from_error_type(error_type)
            detected = sum(
                1
                for r in subset
                if _safe_int(r.get("true_label", ""), default=expected) == expected
                and _safe_int(r.get("pred_label", ""), default=-1) == expected
            )
            detected_any = sum(
                1 for r in subset if _safe_int(r.get("pred_label", ""), default=0) != 0
            )
            entry["detected"] = detected
            entry["detected_any_error"] = detected_any
            entry["detection_rate_pct"] = _pct(detected, n)
            entry["any_error_detection_rate_pct"] = _pct(detected_any, n)
        else:
            detected = sum(
                1
                for r in subset
                if _safe_int(r.get("true_label", ""), default=0) == 0
                and _safe_int(r.get("pred_label", ""), default=1) == 0
            )
            entry["detected"] = detected
            entry["detection_rate_pct"] = _pct(detected, n)
        out[error_type] = entry
    return out


def write_eval_summary_report(
    predictions_csv: str,
    output_report: str,
    run_name: str,
    mode: str,
    *,
    eval_csv: Optional[str] = None,
    source: SourceKind = "auto",
    prompt: Optional[str] = None,
) -> None:
    """Overall + per-error_type summary (no incidence buckets)."""
    rows = _load_predictions(predictions_csv)
    resolved_source = _resolve_source(rows, source)
    multiclass = _resolve_multiclass(rows, prompt)
    task: TaskKind = "multiclass" if multiclass else "binary"
    overall = _overall_counts(rows, multiclass=multiclass)
    by_type = summarize_by_error_type(rows, multiclass=multiclass)
    nf_swap = _nf_swap_from_rows(rows, task=task, eval_csv=eval_csv)

    lines: List[str] = []
    lines.append("EVAL SUMMARY")
    lines.append(f"Run: {run_name}")
    lines.append(f"Source: {resolved_source}")
    lines.append(f"Task: {task}")
    if eval_csv:
        lines.append(f"Eval CSV: {eval_csv}")
    lines.append(f"Predictions: {predictions_csv}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Eval mode: {mode}")
    lines.append("")
    lines.append("OVERALL")
    lines.append(f"rows total:            {overall['n_total']}")
    if multiclass:
        mc = _compute_multiclass_metrics(rows)
        macro_precision = float(mc["macro_precision"])
        macro_recall = float(mc["macro_recall"])
        lines.append(
            f"multiclass accuracy:   {_fmt_pct(overall['correct'], overall['n_metric'])}"
        )
        if macro_precision == macro_precision:
            lines.append(f"macro precision:       {100.0 * macro_precision:.1f}%")
        else:
            lines.append("macro precision:       nan")
        if macro_recall == macro_recall:
            lines.append(f"macro recall:          {100.0 * macro_recall:.1f}%")
        else:
            lines.append("macro recall:          nan")
        lines.append(f"macro F1:              {100.0 * float(mc['macro_f1']):.1f}%")
        lines.append(f"weighted F1:           {100.0 * float(mc['weighted_f1']):.1f}%")
        lines.append(f"binary detection F1:   {100.0 * float(mc['binary_f1']):.1f}%")
        err_acc = float(mc["errors_only_type_acc"])
        if err_acc == err_acc:
            lines.append(f"errors-only type acc:  {100.0 * err_acc:.1f}%")
        else:
            lines.append("errors-only type acc:  nan")
    else:
        lines.append(
            f"overall accuracy:      {_fmt_pct(overall['correct'], overall['n_metric'])}"
        )
        macro_f1 = _binary_macro_f1(
            overall["tp0"], overall["fp0"], overall["fn0"], overall["tn0"]
        )
        lines.append(f"overall f1:            {100.0 * macro_f1:.1f}%")
    if _has_parse_ok_column(rows) and overall["n_metric"] < overall["n_total"]:
        excluded = overall["n_total"] - overall["n_metric"]
        lines.append(
            f"metrics rows:          {overall['n_metric']} "
            f"(parse_ok only; {excluded} parse failures excluded)"
        )
    lines.append(
        f"inconsistent rows:     {overall['n_inconsistent']}"
    )
    lines.append(
        f"inconsistent recall:   {overall['tp0']} "
        f"({_fmt_pct(overall['tp0'], overall['n_inconsistent'])})"
    )
    lines.append(
        f"consistent rows:       {overall['n_consistent']}"
    )
    lines.append(
        f"consistent recall:     {overall['consistent_recall_tp']} "
        f"({_fmt_pct(overall['consistent_recall_tp'], overall['n_consistent'])})"
    )
    lines.append(
        f"false alarms:          {overall['fp0']} "
        f"({_fmt_pct(overall['fp0'], overall['n_consistent'])})"
    )
    _append_weighted_accuracy_lines(lines, overall, by_type=by_type)
    has_parse = any("parse_ok" in r for r in rows)
    if has_parse:
        lines.append(
            f"parse success:         {overall['parse_ok']} "
            f"({_fmt_pct(overall['parse_ok'], overall['n_total'])})"
        )
        if overall["parse_ok"] == 0:
            lines.append(
                "WARNING: 0% parse success — detection and accuracy above are not meaningful."
            )

    if multiclass:
        mc = _compute_multiclass_metrics(rows)
        sep = "-" * 62
        header = f"  {'class':>5}  {'label':<22}  {'n':>5}  {'P':>6}  {'R':>6}  {'F1':>6}"
        lines.append("")
        lines.append("PER CLASS")
        lines.append(sep)
        lines.append(header)
        lines.append(sep)
        for c in range(MULTICLASS_N_CLASSES):
            m = mc["per_class"][c]
            lines.append(
                f"  {c:>5}  {MULTICLASS_LABEL_NAMES[c]:<22}  {int(m['support']):>5}  "
                f"{float(m['precision']):>6.4f}  {float(m['recall']):>6.4f}  "
                f"{float(m['f1']):>6.4f}"
            )
        lines.append(sep)
        lines.append(
            f"  {'':>5}  {'macro avg':<22}  {'':>5}  "
            f"{float(mc['macro_precision']):>6.4f}  "
            f"{float(mc['macro_recall']):>6.4f}  "
            f"{float(mc['macro_f1']):>6.4f}"
        )
        lines.append(
            f"  {'':>5}  {'weighted avg':<22}  {'':>5}  {'':>6}  {'':>6}  "
            f"{float(mc['weighted_f1']):>6.4f}"
        )
        lines.append(sep)

    lines.append("")
    lines.append("BY ERROR TYPE")
    if multiclass:
        lines.append(
            f"{'error_type':>18} {'OG_N':>6} {'pass_N':>6} {'type_match':>10} "
            f"{'rate':>10} {'any_error':>10} {'class_recall':>14}"
        )
    else:
        lines.append(
            f"{'error_type':>18} {'OG_N':>6} {'pass_N':>6} {'metric':>10} "
            f"{'rate':>10} {'class_recall':>14}"
        )
    for error_type, stats in by_type.items():
        n_total = int(stats.get("n_total", stats["n"]))
        n_passed = int(stats["n"])
        if error_type == "consistent":
            metric_val = int(stats.get("consistent_recall", 0))
            rate = _fmt_pct(metric_val, n_passed)
            any_error_str = ""
        elif multiclass:
            metric_val = int(stats.get("detected", 0))
            rate = _fmt_pct(metric_val, n_passed)
            any_error_str = _fmt_pct(
                int(stats.get("detected_any_error", 0)), n_passed
            )
        else:
            metric_val = int(stats.get("detected", 0))
            rate = _fmt_pct(metric_val, n_passed)
            any_error_str = ""
        class_recall_pct = float(
            stats.get("class_recall_pct", stats.get("accuracy_pct", float("nan")))
        )
        class_recall_str = (
            "nan" if class_recall_pct != class_recall_pct else f"{class_recall_pct:.1f}%"
        )
        if multiclass and error_type != "consistent":
            lines.append(
                f"{error_type:>18} {n_total:>6} {n_passed:>6} {metric_val:>10} "
                f"{rate:>10} {any_error_str:>10} {class_recall_str:>14}"
            )
        else:
            lines.append(
                f"{error_type:>18} {n_total:>6} {n_passed:>6} {metric_val:>10} "
                f"{rate:>10} {class_recall_str:>14}"
            )

    if nf_swap:
        lines.extend(format_nf_swap_direction_lines(nf_swap, task=task))

    out_text = "\n".join(lines) + "\n"
    Path(output_report).write_text(out_text, encoding="utf-8")
    print(out_text)
    print(f"Saved report to: {output_report}")


def _confusion_matrix_title(mode: str, model_name: Optional[str] = None) -> str:
    titles = {
        "lat": "Confusion Matrix (Laterality Benchmark)",
        "site": "Confusion Matrix (Site Benchmark)",
        "severity": "Confusion Matrix (Severity Benchmark)",
        "hard": "Confusion Matrix (Hard Benchmark)",
        "cross_pair_full": "Confusion Matrix (Cross-Pair Benchmark)",
        "all": "Confusion Matrix",
    }
    title = titles.get(mode, "Confusion Matrix")
    if model_name:
        return f"{title}\n{model_name}"
    return title


def _predictions_are_multiclass(rows: List[Dict]) -> bool:
    """Heuristic when --prompt is not passed (e.g. regenerating from CSV alone)."""
    for r in rows:
        true_label = _safe_int(r.get("true_label", ""), default=0)
        pred_label = _safe_int(r.get("pred_label", ""), default=0)
        if true_label > 1 or pred_label > 1:
            return True
    # Binary exports (BERT + LLM) use inconsistent/consistent label names.
    # error_type is corruption metadata on those rows, not a multiclass label.
    for r in rows:
        for field in ("true_label_name", "pred_label_name"):
            name = (r.get(field) or "").strip().lower()
            if name in {"inconsistent", "consistent"}:
                return False
    for r in rows:
        for field in ("true_label_name", "pred_label_name"):
            name = (r.get(field) or "").strip()
            if name in _MULTICLASS_ERROR_NAMES:
                return True
    return False


def plot_confusion_matrix_multiclass_from_predictions(
    predictions_csv: str,
    output_path: str,
    model_name: Optional[str] = None,
    *,
    parse_ok_only: bool = True,
) -> None:
    """6×6 confusion matrix matching bert_inconsistency_multiclass.plot_confusion_matrix_multiclass."""
    if plt is None:
        raise ImportError(
            "matplotlib is required for confusion matrix plots. "
            "Install with: pip install matplotlib"
        )

    rows = _load_predictions(predictions_csv)
    if parse_ok_only:
        rows = [r for r in rows if _safe_bool_int(r.get("parse_ok", "1"))]

    matrix = [[0] * MULTICLASS_N_CLASSES for _ in range(MULTICLASS_N_CLASSES)]
    for r in rows:
        true_label = _safe_int(r.get("true_label", ""), default=0)
        pred_label = _safe_int(r.get("pred_label", ""), default=0)
        if 0 <= true_label < MULTICLASS_N_CLASSES and 0 <= pred_label < MULTICLASS_N_CLASSES:
            matrix[true_label][pred_label] += 1

    class_names = [MULTICLASS_LABEL_NAMES[c] for c in range(MULTICLASS_N_CLASSES)]
    title = "Multiclass Confusion Matrix"
    if model_name:
        title = f"{title}\n{model_name}"

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(MULTICLASS_N_CLASSES), labels=class_names, rotation=45, ha="right")
    ax.set_yticks(range(MULTICLASS_N_CLASSES), labels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(MULTICLASS_N_CLASSES):
        for j in range(MULTICLASS_N_CLASSES):
            ax.text(
                j,
                i,
                str(matrix[i][j]),
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved confusion matrix plot to: {output_path}")


def plot_confusion_matrix_inconsistent_positive(
    overall: Dict[str, int],
    output_path: str,
    title: str = "Confusion Matrix",
) -> None:
    """Match BERT plot layout: inconsistent (0) is the positive class."""
    if plt is None:
        raise ImportError(
            "matplotlib is required for confusion matrix plots. "
            "Install with: pip install matplotlib"
        )

    matrix = [
        [int(overall["tp0"]), int(overall["fn0"])],
        [int(overall["fp0"]), int(overall["tn0"])],
    ]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks([0, 1], labels=["Pred: Inconsistent", "Pred: Consistent"])
    ax.set_yticks([0, 1], labels=["True: Inconsistent", "True: Consistent"])
    ax.set_title(title)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center", color="black")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved confusion matrix plot to: {output_path}")


def plot_confusion_matrix_from_predictions(
    predictions_csv: str,
    output_path: str,
    mode: str = "hard",
    model_name: Optional[str] = None,
    *,
    multiclass: Optional[bool] = None,
    prompt: Optional[str] = None,
) -> None:
    if multiclass is None and prompt is not None:
        multiclass = is_multiclass_prompt(prompt)
    if multiclass is None:
        rows = _load_predictions(predictions_csv)
        multiclass = _predictions_are_multiclass(rows)

    if multiclass:
        plot_confusion_matrix_multiclass_from_predictions(
            predictions_csv=predictions_csv,
            output_path=output_path,
            model_name=model_name,
        )
        return

    rows = _load_predictions(predictions_csv)
    overall = _overall_counts(rows)
    plot_confusion_matrix_inconsistent_positive(
        overall=overall,
        output_path=output_path,
        title=_confusion_matrix_title(mode, model_name=model_name),
    )


def _load_predictions(csv_path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        required = {"true_label", "pred_label", "error_type"}
        missing = required - fields
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        for row in reader:
            rows.append(row)
    return rows


def _overall_counts(rows: List[Dict], *, multiclass: bool = False) -> Dict[str, int]:
    metric_rows = _metric_rows(rows)
    out = {
        "n_total": len(rows),
        "n_metric": len(metric_rows),
        "n_inconsistent": 0,
        "n_consistent": 0,
        "tp0": 0,
        "fp0": 0,
        "fn0": 0,
        "tn0": 0,
        "consistent_recall_tp": 0,
        "correct": 0,
        "parse_ok": 0,
    }
    for r in metric_rows:
        default_label = 0 if multiclass else INCONSISTENT_LABEL
        t = _safe_int(r.get("true_label", ""), default=default_label)
        p = _safe_int(r.get("pred_label", ""), default=default_label)
        out["correct"] += int(t == p)
        if multiclass:
            true_error = t != 0
            pred_error = p != 0
            out["n_inconsistent"] += int(true_error)
            out["n_consistent"] += int(not true_error)
            out["tp0"] += int(true_error and pred_error)
            out["fp0"] += int(not true_error and pred_error)
            out["fn0"] += int(true_error and not pred_error)
            out["tn0"] += int(not true_error and not pred_error)
            out["consistent_recall_tp"] += int(not true_error and not pred_error)
        else:
            out["n_inconsistent"] += int(t == INCONSISTENT_LABEL)
            out["n_consistent"] += int(t == CONSISTENT_LABEL)
            out["tp0"] += int(t == INCONSISTENT_LABEL and p == INCONSISTENT_LABEL)
            out["fp0"] += int(t == CONSISTENT_LABEL and p == INCONSISTENT_LABEL)
            out["fn0"] += int(t == INCONSISTENT_LABEL and p == CONSISTENT_LABEL)
            out["tn0"] += int(t == CONSISTENT_LABEL and p == CONSISTENT_LABEL)
            out["consistent_recall_tp"] += int(
                t == CONSISTENT_LABEL and p == CONSISTENT_LABEL
            )
    if _has_parse_ok_column(rows):
        out["parse_ok"] = sum(_safe_bool_int(r.get("parse_ok", "0")) for r in rows)
    return out


def _build_original_conclusion_by_source_idx(rows: List[Dict]) -> Dict[int, str]:
    """Uncorrupted conclusion per source_idx from consistent control rows."""
    out: Dict[int, str] = {}
    for r in rows:
        et = (r.get("error_type") or "").strip()
        if et not in {"consistent", "positive"}:
            continue
        raw = (r.get("source_idx") or "").strip()
        if not raw:
            continue
        try:
            idx = int(raw)
        except ValueError:
            continue
        conclusion = (r.get("conclusion") or "").strip()
        if conclusion:
            out[idx] = conclusion
    return out


def _row_incidence_for_mode(
    row: Dict,
    original_by_idx: Dict[int, str],
    incidence_mode: str,
) -> int:
    """Keyword incidence for a predictions row under lat/site/severity profile."""
    error_type = (row.get("error_type") or "").strip()
    conclusion = (row.get("conclusion") or "").strip()
    original_conclusion: Optional[str] = None
    raw = (row.get("source_idx") or "").strip()
    if raw:
        try:
            original_conclusion = original_by_idx.get(int(raw))
        except ValueError:
            original_conclusion = None
    raw_inc = _swap_incidence_for_example(
        conclusion,
        error_type,
        original_conclusion=original_conclusion,
        incidence_profile=incidence_mode,
    )
    if not raw_inc:
        return -1
    return _safe_int(raw_inc, default=-1)


def _resolve_row_incidence(
    row: Dict,
    *,
    original_by_idx: Dict[int, str],
    incidence_mode: Optional[str],
) -> int:
    if incidence_mode in INCIDENCE_EVAL_MODES:
        return _row_incidence_for_mode(row, original_by_idx, incidence_mode)
    return _safe_int(row.get("incidence", ""), default=-1)


def _error_swap_incidence_counts(
    rows: List[Dict],
    error_type: str,
    *,
    multiclass: bool = False,
    incidence_mode: Optional[str] = None,
) -> Dict[int, Dict[str, int]]:
    original_by_idx = (
        _build_original_conclusion_by_source_idx(rows)
        if incidence_mode in INCIDENCE_EVAL_MODES
        else {}
    )
    buckets: Dict[int, Dict[str, int]] = defaultdict(lambda: {"n": 0, "tp": 0, "fn": 0})
    for r in _metric_rows(rows):
        if (r.get("error_type") or "").strip() != error_type:
            continue
        incidence = _resolve_row_incidence(
            r, original_by_idx=original_by_idx, incidence_mode=incidence_mode
        )
        if incidence < 0:
            continue
        true_label = _safe_int(r.get("true_label", ""), default=0)
        pred_label = _safe_int(r.get("pred_label", ""), default=1)
        if multiclass:
            b = buckets[incidence]
            b["n"] += 1
            if pred_label != 0:
                b["tp"] += 1
            else:
                b["fn"] += 1
        else:
            if true_label != 0:
                continue
            b = buckets[incidence]
            b["n"] += 1
            if pred_label == 0:
                b["tp"] += 1
            else:
                b["fn"] += 1
    return dict(sorted(buckets.items(), key=lambda kv: kv[0]))


def _lat_incidence_counts(rows: List[Dict]) -> Dict[int, Dict[str, int]]:
    return _error_swap_incidence_counts(rows, "laterality_swap")


def _severity_incidence_counts(rows: List[Dict]) -> Dict[int, Dict[str, int]]:
    return _error_swap_incidence_counts(rows, "severity_flip")


def _consistent_incidence_counts(
    rows: List[Dict],
    *,
    multiclass: bool = False,
    incidence_mode: Optional[str] = None,
) -> Dict[int, Dict[str, int]]:
    original_by_idx = (
        _build_original_conclusion_by_source_idx(rows)
        if incidence_mode in INCIDENCE_EVAL_MODES
        else {}
    )
    buckets: Dict[int, Dict[str, int]] = defaultdict(lambda: {"n": 0, "fp": 0})
    for r in _metric_rows(rows):
        if (r.get("error_type") or "").strip() != "consistent":
            continue
        incidence = _resolve_row_incidence(
            r, original_by_idx=original_by_idx, incidence_mode=incidence_mode
        )
        if incidence < 0:
            continue
        true_label = _safe_int(r.get("true_label", ""), default=1)
        pred_label = _safe_int(r.get("pred_label", ""), default=1)
        if multiclass:
            if true_label != 0:
                continue
        elif true_label != 1:
            continue
        b = buckets[incidence]
        b["n"] += 1
        if multiclass:
            if pred_label != 0:
                b["fp"] += 1
        elif pred_label == 0:
            b["fp"] += 1
    return dict(sorted(buckets.items(), key=lambda kv: kv[0]))


def _sum_subset(buckets: Dict[int, Dict[str, int]], min_incidence: int) -> Tuple[int, int]:
    n = 0
    tp = 0
    for incidence, stats in buckets.items():
        if incidence >= min_incidence:
            n += stats["n"]
            tp += stats.get("tp", 0)
    return n, tp


INCIDENCE_EVAL_MODES = frozenset({"lat", "site", "severity"})
HARD_SUITE_EVAL_MODES = frozenset({"hard", "all", "cross_pair_full"})

_INCIDENCE_REPORT_CONFIG = {
    "lat": {
        "swap_error_type": "laterality_swap",
        "swap_section": "LATERALITY_SWAP — recall by incidence",
        "consistent_section": "False positive rate by lateral incidence",
        "x_label": "Lateral incidence",
    },
    "site": {
        "swap_error_type": "site_swap",
        "swap_section": "SITE_SWAP — recall by incidence",
        "consistent_section": "False positive rate by site incidence",
        "x_label": "Site incidence",
    },
    "severity": {
        "swap_error_type": "severity_flip",
        "swap_section": "SEVERITY_FLIP — recall by incidence",
        "consistent_section": "False positive rate by severity incidence",
        "x_label": "Severity incidence",
    },
}


def default_incidence_plot_paths(predictions_csv: str, mode: str) -> Tuple[str, str]:
    """Default plot paths under <run>/plots/ next to results/."""
    if mode not in INCIDENCE_EVAL_MODES:
        raise ValueError(f"Incidence plots require mode in {sorted(INCIDENCE_EVAL_MODES)}, got {mode!r}")
    plots_dir = Path(predictions_csv).resolve().parent.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return (
        str(plots_dir / f"{mode}_recall_by_incidence.png"),
        str(plots_dir / f"{mode}_fp_rate_by_incidence.png"),
    )


def _marker_sizes_from_counts(
    counts: List[int],
    *,
    min_size: float = 40.0,
    max_size: float = 450.0,
) -> List[float]:
    """Scatter marker areas (points^2) scaled by sqrt(N) within each plot."""
    if not counts:
        return []
    max_n = max(counts)
    if max_n <= 0:
        return [min_size] * len(counts)
    span = max_size - min_size
    return [
        min_size + span * ((n / max_n) ** 0.5) if n > 0 else min_size
        for n in counts
    ]


def _plot_incidence_line(
    incidences: List[int],
    values_pct: List[float],
    *,
    title: str,
    x_label: str,
    y_label: str,
    output_path: str,
    color: str,
    counts: Optional[List[int]] = None,
    label_below: bool = False,
) -> None:
    if not incidences:
        print(f"Skipping empty incidence plot: {output_path}")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(incidences, values_pct, color=color, linewidth=2.5, zorder=2)
    sizes = _marker_sizes_from_counts(counts) if counts is not None else [80.0] * len(incidences)
    ax.scatter(
        incidences,
        values_pct,
        s=sizes,
        color=color,
        zorder=3,
        edgecolors="white",
        linewidths=0.8,
    )
    for x, y, size in zip(incidences, values_pct, sizes):
        if y == y:
            # Lift/drop labels a bit more for larger markers.
            offset = 10 + max(0.0, (size ** 0.5) - 6.0) * 0.35
            if label_below:
                offset = -offset
            ax.annotate(
                f"{y:.1f}%",
                (x, y),
                textcoords="offset points",
                xytext=(0, offset),
                ha="center",
                va="top" if label_below else "bottom",
                fontsize=13,
            )
    ax.set_title(title, fontsize=18, pad=12)
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    ax.tick_params(axis="both", labelsize=13)
    ax.set_ylim(0, 100)
    ax.set_xticks(incidences)
    ax.grid(True, alpha=0.3)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved incidence plot to: {output_path}")


def plot_swap_recall_by_incidence(
    predictions_csv: str,
    output_path: str,
    mode: str,
    *,
    prompt: Optional[str] = None,
) -> None:
    if mode not in _INCIDENCE_REPORT_CONFIG:
        raise ValueError(
            f"Incidence recall plot requires mode in {sorted(INCIDENCE_EVAL_MODES)}, got {mode!r}"
        )
    cfg = _INCIDENCE_REPORT_CONFIG[mode]
    rows = _load_predictions(predictions_csv)
    multiclass = _resolve_multiclass(rows, prompt)
    buckets = _error_swap_incidence_counts(
        rows, cfg["swap_error_type"], multiclass=multiclass, incidence_mode=mode
    )
    incidences: List[int] = []
    values_pct: List[float] = []
    counts: List[int] = []
    for incidence, stats in buckets.items():
        n = stats["n"]
        if n <= 0:
            continue
        incidences.append(incidence)
        values_pct.append(_pct(stats["tp"], n))
        counts.append(n)
    _plot_incidence_line(
        incidences,
        values_pct,
        title=cfg["swap_section"],
        x_label=cfg["x_label"],
        y_label="Recall (%)",
        output_path=output_path,
        color="#1f77b4",
        counts=counts,
        label_below=True,
    )


def plot_consistent_fp_rate_by_incidence(
    predictions_csv: str,
    output_path: str,
    mode: str,
    *,
    prompt: Optional[str] = None,
) -> None:
    if mode not in _INCIDENCE_REPORT_CONFIG:
        raise ValueError(
            f"Incidence FP plot requires mode in {sorted(INCIDENCE_EVAL_MODES)}, got {mode!r}"
        )
    cfg = _INCIDENCE_REPORT_CONFIG[mode]
    rows = _load_predictions(predictions_csv)
    multiclass = _resolve_multiclass(rows, prompt)
    buckets = _consistent_incidence_counts(
        rows, multiclass=multiclass, incidence_mode=mode
    )
    incidences: List[int] = []
    values_pct: List[float] = []
    counts: List[int] = []
    for incidence, stats in buckets.items():
        n = stats["n"]
        if n <= 0:
            continue
        incidences.append(incidence)
        values_pct.append(_pct(stats["fp"], n))
        counts.append(n)
    _plot_incidence_line(
        incidences,
        values_pct,
        title=cfg["consistent_section"],
        x_label=cfg["x_label"],
        y_label="False positive rate (%)",
        output_path=output_path,
        color="#d62728",
        counts=counts,
    )


def write_incidence_plots(
    predictions_csv: str,
    mode: str,
    *,
    recall_plot_path: Optional[str] = None,
    fp_plot_path: Optional[str] = None,
    prompt: Optional[str] = None,
) -> None:
    if mode not in INCIDENCE_EVAL_MODES:
        return
    recall_path, fp_path = default_incidence_plot_paths(predictions_csv, mode)
    plot_swap_recall_by_incidence(
        predictions_csv,
        recall_plot_path or recall_path,
        mode,
        prompt=prompt,
    )
    plot_consistent_fp_rate_by_incidence(
        predictions_csv,
        fp_plot_path or fp_path,
        mode,
        prompt=prompt,
    )


def write_incidence_eval_artifacts(
    predictions_csv: str,
    output_report: str,
    run_name: str,
    mode: str,
    *,
    recall_plot_path: Optional[str] = None,
    fp_plot_path: Optional[str] = None,
    prompt: Optional[str] = None,
) -> None:
    """Write incidence_detection_report.txt and recall/FP incidence line plots."""
    write_report(predictions_csv, output_report, run_name, mode, prompt=prompt)
    write_incidence_plots(
        predictions_csv,
        mode,
        recall_plot_path=recall_plot_path,
        fp_plot_path=fp_plot_path,
        prompt=prompt,
    )


def incidence_modes_for_eval_mode(eval_mode: str) -> List[str]:
    """Incidence report modes to emit for a given eval --mode."""
    if eval_mode in HARD_SUITE_EVAL_MODES:
        return ["lat", "site", "severity"]
    if eval_mode in INCIDENCE_EVAL_MODES:
        return [eval_mode]
    return []


def incidence_report_filename(incidence_mode: str, eval_mode: str) -> str:
    if eval_mode in INCIDENCE_EVAL_MODES:
        return "incidence_detection_report.txt"
    return f"incidence_detection_report_{incidence_mode}.txt"


def write_incidence_reports_for_eval_mode(
    predictions_csv: str,
    run_name: str,
    eval_mode: str,
    *,
    results_dir: Optional[Path] = None,
    plots_dir: Optional[Path] = None,
    prompt: Optional[str] = None,
) -> Dict[str, str]:
    """Write incidence text reports and plots for all modes implied by eval_mode."""
    out_dir = results_dir or Path(predictions_csv).resolve().parent
    plot_root = plots_dir or out_dir.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_root.mkdir(parents=True, exist_ok=True)

    written: Dict[str, str] = {}
    for incidence_mode in incidence_modes_for_eval_mode(eval_mode):
        report_path = str(out_dir / incidence_report_filename(incidence_mode, eval_mode))
        write_incidence_eval_artifacts(
            predictions_csv=predictions_csv,
            output_report=report_path,
            run_name=run_name,
            mode=incidence_mode,
            recall_plot_path=str(plot_root / f"{incidence_mode}_recall_by_incidence.png"),
            fp_plot_path=str(plot_root / f"{incidence_mode}_fp_rate_by_incidence.png"),
            prompt=prompt,
        )
        written[incidence_mode] = report_path
    return written


def _append_swap_incidence_sections(
    lines: List[str],
    swap_buckets: Dict[int, Dict[str, int]],
    swap_section_title: str,
) -> None:
    lines.append("")
    lines.append("========================================================================")
    lines.append(swap_section_title)
    lines.append("========================================================================")
    lines.append(f"{'incidence':>10} {'N':>8} {'tp':>10} {'recall':>10} {'fn':>8}")
    for incidence, stats in swap_buckets.items():
        n = stats["n"]
        tp = stats["tp"]
        fn = stats["fn"]
        lines.append(
            f"{incidence:>10} {n:>8} {tp:>10} {_fmt_pct(tp, n):>10} {fn:>8}"
        )

    n1 = swap_buckets.get(1, {}).get("n", 0)
    d1 = swap_buckets.get(1, {}).get("tp", 0)
    n2 = swap_buckets.get(2, {}).get("n", 0)
    d2 = swap_buckets.get(2, {}).get("tp", 0)
    n_ge2, d_ge2 = _sum_subset(swap_buckets, 2)
    n_ge3, d_ge3 = _sum_subset(swap_buckets, 3)
    lines.append("")
    lines.append("Combined buckets:")
    lines.append(f"  incidence = 1    n={n1:4d}  tp={d1:4d}  recall={_fmt_pct(d1, n1)}")
    lines.append(f"  incidence = 2    n={n2:4d}  tp={d2:4d}  recall={_fmt_pct(d2, n2)}")
    lines.append(f"  incidence >= 2   n={n_ge2:4d}  tp={d_ge2:4d}  recall={_fmt_pct(d_ge2, n_ge2)}")
    lines.append(f"  incidence >= 3   n={n_ge3:4d}  tp={d_ge3:4d}  recall={_fmt_pct(d_ge3, n_ge3)}")


def write_report(
    predictions_csv: str,
    output_report: str,
    run_name: str,
    mode: str,
    *,
    prompt: Optional[str] = None,
) -> None:
    if mode not in _INCIDENCE_REPORT_CONFIG:
        raise ValueError(
            f"Incidence report requires mode in {sorted(INCIDENCE_EVAL_MODES)}, got {mode!r}"
        )
    cfg = _INCIDENCE_REPORT_CONFIG[mode]
    rows = _load_predictions(predictions_csv)
    multiclass = _resolve_multiclass(rows, prompt)
    overall = _overall_counts(rows, multiclass=multiclass)
    swap_buckets = _error_swap_incidence_counts(
        rows, cfg["swap_error_type"], multiclass=multiclass, incidence_mode=mode
    )
    cons_buckets = _consistent_incidence_counts(
        rows, multiclass=multiclass, incidence_mode=mode
    )

    lines: List[str] = []
    lines.append("Incidence vs recall report")
    lines.append(f"Run: {run_name}")
    lines.append(f"Predictions: {predictions_csv}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(f"Eval mode: {mode}")
    lines.append("========================================================================")
    lines.append("OVERALL")
    lines.append("========================================================================")
    lines.append(f"rows total:            {overall['n_total']}")
    lines.append(
        f"overall accuracy:      {_fmt_pct(overall['correct'], overall['n_metric'])}"
    )
    if _has_parse_ok_column(rows) and overall["n_metric"] < overall["n_total"]:
        excluded = overall["n_total"] - overall["n_metric"]
        lines.append(
            f"metrics rows:          {overall['n_metric']} "
            f"(parse_ok only; {excluded} parse failures excluded)"
        )
    lines.append(
        f"inconsistent rows:     {overall['n_inconsistent']}"
    )
    lines.append(
        f"recall (inconsistent): {overall['tp0']} ({_fmt_pct(overall['tp0'], overall['n_inconsistent'])})"
    )
    lines.append(
        f"consistent rows:       {overall['n_consistent']}"
    )
    lines.append(
        f"false positives:       {overall['fp0']} ({_fmt_pct(overall['fp0'], overall['n_consistent'])})"
    )
    has_parse = _has_parse_ok_column(rows)
    if has_parse:
        lines.append(
            f"parse success:         {overall['parse_ok']} ({_fmt_pct(overall['parse_ok'], overall['n_total'])})"
        )
        if overall["parse_ok"] == 0:
            lines.append(
                "WARNING: 0% parse success — detection and accuracy above are not meaningful."
            )

    _append_swap_incidence_sections(lines, swap_buckets, cfg["swap_section"])

    lines.append("")
    lines.append("========================================================================")
    lines.append(cfg["consistent_section"])
    lines.append("========================================================================")
    lines.append(f"{'incidence':>10} {'N':>8} {'fp':>10} {'fp_rate':>10}")
    for incidence, stats in cons_buckets.items():
        n = stats["n"]
        fp = stats["fp"]
        lines.append(f"{incidence:>10} {n:>8} {fp:>10} {_fmt_pct(fp, n):>10}")

    lines.append("")
    lines.append("========================================================================")
    lines.append("NOTES")
    lines.append("========================================================================")
    lines.append("- Recall (inconsistent) = TP / (TP + FN); positive class is inconsistent (true_label=0).")
    lines.append("- False positive = pred_label=0 on true consistent (true_label=1) rows; FP rate = FP / N_consistent.")
    lines.append(
        "- Incidence recomputed from conclusions using the "
        f"{mode} profile ({cfg['swap_error_type']} term family for swap rows; "
        f"matching term family for consistent controls)."
    )
    if has_parse and overall["parse_ok"] > 0:
        lines.append("- parse_ok comes from eval_llm_baseline.py retry/parse handling.")
    if has_parse and overall["n_metric"] < overall["n_total"]:
        lines.append(
            "- Accuracy/recall/incidence metrics exclude rows with parse_ok=0."
        )

    out_text = "\n".join(lines) + "\n"
    Path(output_report).write_text(out_text, encoding="utf-8")
    print(out_text)
    print(f"Saved report to: {output_report}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build summary report from LLM predictions.csv."
    )
    parser.add_argument("--predictions_csv", type=str, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        default="hard",
        choices=["all", "lat", "site", "severity", "hard", "cross_pair_full"],
    )
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument(
        "--output_run_dir",
        type=str,
        default=None,
        help=(
            "Run root for results/, plots/, logs/. "
            "Default: infer from --predictions_csv (parent run when CSV is under results/)."
        ),
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
        "--report_style",
        type=str,
        default="summary",
        choices=["summary", "incidence"],
        help=(
            "summary = overall + per error_type plus incidence reports for lat/site/severity "
            "when mode is hard/all; incidence = incidence reports only."
        ),
    )
    parser.add_argument(
        "--no_incidence_reports",
        action="store_true",
        help="Skip writing incidence_detection_report*.txt and incidence plots.",
    )
    parser.add_argument(
        "--output_report",
        type=str,
        default=None,
        help="Defaults to eval_summary.txt or incidence_detection_report.txt",
    )
    parser.add_argument(
        "--confusion_plot_path",
        type=str,
        default=None,
        help="Optional path to save confusion_matrix.png",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Optional model label shown in the confusion matrix title.",
    )
    parser.add_argument(
        "--no_confusion_plot",
        action="store_true",
        help="Skip writing plots/confusion_matrix.png (default: write when output_run_dir is set).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt variant used for eval (4 = multiclass 6×6 confusion matrix).",
    )
    parser.add_argument(
        "--eval_csv",
        type=str,
        default=None,
        help="Optional eval dataset path (for NF swap direction breakdown).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="auto",
        choices=["auto", "llm", "bert"],
        help="Prediction source (auto-detect from parse_ok column when omitted).",
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
        tags=("report", args.mode, args.report_style),
        run_name=args.run_name,
        run_category=RUN_CATEGORY,
        output_model_dir=output_run_dir,
        command_stem=args.command_stem,
        argv=current_argv(),
    )
    default_name = (
        "incidence_detection_report.txt"
        if args.report_style == "incidence"
        else "eval_summary.txt"
    )
    args.output_report = default_result_path(run_layout, default_name, args.output_report)
    args.confusion_plot_path = default_plot_path(
        run_layout, "confusion_matrix.png", args.confusion_plot_path
    )
    report_run_name = args.run_name or run_layout.root.name
    results_dir = Path(args.output_report).resolve().parent
    plots_dir = Path(args.confusion_plot_path).resolve().parent
    with tee_run_output(run_layout.log_path(args.log_file)):
        print(f"Run outputs: {run_layout.root}")
        if args.report_style == "incidence":
            if args.mode in INCIDENCE_EVAL_MODES:
                write_incidence_eval_artifacts(
                    predictions_csv=args.predictions_csv,
                    output_report=args.output_report,
                    run_name=report_run_name,
                    mode=args.mode,
                    recall_plot_path=default_plot_path(
                        run_layout, f"{args.mode}_recall_by_incidence.png", None
                    ),
                    fp_plot_path=default_plot_path(
                        run_layout, f"{args.mode}_fp_rate_by_incidence.png", None
                    ),
                )
            elif not args.no_incidence_reports:
                write_incidence_reports_for_eval_mode(
                    predictions_csv=args.predictions_csv,
                    run_name=report_run_name,
                    eval_mode=args.mode,
                    results_dir=results_dir,
                    plots_dir=plots_dir,
                )
        else:
            write_eval_artifacts(
                predictions_csv=args.predictions_csv,
                run_name=report_run_name,
                eval_mode=args.mode,
                eval_csv=args.eval_csv,
                source=args.source,
                prompt=args.prompt,
                summary_path=args.output_report,
                results_dir=results_dir,
                plots_dir=plots_dir,
                write_incidence=not args.no_incidence_reports,
                write_confusion=False,
                model_name=args.model_name,
            )
        if not args.no_confusion_plot and args.report_style != "incidence":
            plot_confusion_matrix_from_predictions(
                predictions_csv=args.predictions_csv,
                output_path=args.confusion_plot_path,
                mode=args.mode,
                model_name=args.model_name,
                prompt=args.prompt,
            )


if __name__ == "__main__":
    main()
