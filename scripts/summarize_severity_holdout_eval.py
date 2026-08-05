#!/usr/bin/env python3
"""Slice v6 severity_flip rows by swap kind/term and report binary detection recall."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.severity_flip import (  # noqa: E402
    SEVERITY_FLIP_TERMS,
    SEVERITY_SCALAR_RULES,
    SEVERITY_TIERS,
)


def _scalar_terms() -> Set[str]:
    out: Set[str] = set()
    for left, right, _ in SEVERITY_SCALAR_RULES:
        out.add(left.lower())
        out.add(right.lower())
    return out


def _tier_terms() -> Set[str]:
    out: Set[str] = set()
    for bag in SEVERITY_TIERS:
        out.update(t.lower() for t in bag)
    return out


SCALAR_TERMS = _scalar_terms()
TIER_TERMS = _tier_terms()


def _token_pairs(orig: str, corr: str) -> List[Tuple[str, str]]:
    o_tokens = [m.group(0) for m in re.finditer(r"\b\w+\b", orig)]
    c_tokens = [m.group(0) for m in re.finditer(r"\b\w+\b", corr)]
    swaps: List[Tuple[str, str]] = []
    for o, c in zip(o_tokens, c_tokens):
        if o.lower() != c.lower():
            swaps.append((o, c))
    return swaps


def swapped_severity_token(orig: str, corr: str) -> Optional[str]:
    severity = {t.lower() for t in SEVERITY_FLIP_TERMS}
    for o, c in _token_pairs(orig, corr):
        if o.lower() in severity:
            return o.lower()
        if c.lower() in severity:
            return c.lower()
    return None


def classify_swap(term: Optional[str]) -> str:
    if term is None:
        return "unknown"
    if term in SCALAR_TERMS:
        return "scalar"
    if term in TIER_TERMS:
        return "tier"
    return "other"


def load_eval_rows(eval_csv: str) -> List[dict]:
    with open(eval_csv, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_predictions(predictions_csv: str) -> Dict[Tuple[str, str, str], dict]:
    keyed: Dict[Tuple[str, str, str], dict] = {}
    with open(predictions_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (
                (row.get("findings") or "").strip(),
                (row.get("conclusion") or "").strip(),
                (row.get("error_type") or "").strip(),
            )
            keyed[key] = row
    return keyed


def binary_detected(row: dict) -> bool:
    pred = row.get("pred_label", row.get("prediction", ""))
    try:
        return int(pred) == 0
    except (TypeError, ValueError):
        return str(pred).strip() in {"0", "inconsistent", "False", "false"}


def recall(rows: Sequence[dict]) -> Tuple[int, int, float]:
    if not rows:
        return 0, 0, float("nan")
    hits = sum(1 for r in rows if binary_detected(r))
    return hits, len(rows), 100.0 * hits / len(rows)


def summarize_group(
    title: str,
    rows: Sequence[dict],
    *,
    min_count: int = 1,
) -> None:
    if not rows:
        print(f"\n{title}: (no rows)")
        return
    hits, n, pct = recall(rows)
    print(f"\n{title}")
    print(f"  n={n}  detected={hits}  recall={pct:.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Binary recall on severity_flip slices of a v6 eval run."
    )
    parser.add_argument("--predictions_csv", required=True)
    parser.add_argument("--eval_csv", required=True)
    parser.add_argument(
        "--holdout_scalar",
        action="store_true",
        help="Highlight scalar swap rows (held out when training with --severity_exclude_scalar).",
    )
    parser.add_argument(
        "--holdout_terms",
        type=str,
        default=None,
        help="Comma-separated terms held out during training (e.g. mild,mildly,severe,severely).",
    )
    args = parser.parse_args()

    eval_rows = load_eval_rows(args.eval_csv)
    preds = load_predictions(args.predictions_csv)
    consistent = {
        int(r["source_idx"]): (r.get("conclusion") or "").strip()
        for r in eval_rows
        if (r.get("error_type") or "").strip() == "consistent"
    }

    holdout_terms = {
        t.strip().lower() for t in (args.holdout_terms or "").split(",") if t.strip()
    }
    if not holdout_terms and args.holdout_terms is None:
        holdout_terms = set()

    tagged: List[dict] = []
    kind_counter: Counter[str] = Counter()
    term_counter: Counter[str] = Counter()

    for row in eval_rows:
        if (row.get("error_type") or "").strip() != "severity_flip":
            continue
        source_idx = int(row["source_idx"])
        orig = consistent.get(source_idx, "")
        corr = (row.get("conclusion") or "").strip()
        term = swapped_severity_token(orig, corr)
        kind = classify_swap(term)
        kind_counter[kind] += 1
        if term:
            term_counter[term] += 1
        key = (
            (row.get("findings") or "").strip(),
            corr,
            "severity_flip",
        )
        pred_row = preds.get(key, {})
        tagged.append(
            {
                "source_idx": source_idx,
                "swap_kind": kind,
                "swapped_from": term or "",
                "pred_label": pred_row.get("pred_label", pred_row.get("prediction", "")),
            }
        )

    all_rows = tagged
    scalar_rows = [r for r in tagged if r["swap_kind"] == "scalar"]
    tier_rows = [r for r in tagged if r["swap_kind"] == "tier"]
    holdout_rows = [r for r in tagged if r["swapped_from"] in holdout_terms] if holdout_terms else []
    seen_rows = [r for r in tagged if r["swapped_from"] and r["swapped_from"] not in holdout_terms] if holdout_terms else []

    print(f"Eval CSV: {args.eval_csv}")
    print(f"Predictions: {args.predictions_csv}")
    print(f"severity_flip rows: {len(tagged)}")
    print(f"  kind counts: {dict(kind_counter)}")

    summarize_group("ALL severity_flip", all_rows)
    summarize_group("scalar swaps (small/large, low-grade/high-grade)", scalar_rows)
    summarize_group("tier qualitative swaps", tier_rows)

    if holdout_terms:
        summarize_group(
            f"HELD-OUT terms {sorted(holdout_terms)}",
            holdout_rows,
        )
        summarize_group(
            f"SEEN tier terms (excluding holdout)",
            seen_rows,
        )

    if args.holdout_scalar:
        summarize_group(">>> HELD-OUT scalar slice <<<", scalar_rows)
        summarize_group(">>> SEEN tier slice (scalar held out) <<<", tier_rows)

    print("\nTop swapped-from terms:")
    for term, count in term_counter.most_common(15):
        subset = [r for r in tagged if r["swapped_from"] == term]
        _, n, pct = recall(subset)
        flag = "  [holdout]" if term in holdout_terms else ""
        flag = flag or ("  [scalar]" if term in SCALAR_TERMS else "")
        print(f"  {term:12s} n={n:4d}  recall={pct:5.1f}%{flag}")


if __name__ == "__main__":
    main()
