import argparse
import csv
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple


try:
    from paths import NO_FINDINGS_DIR, data_path, eval_data_path
    from report_split_io import ReportRecord, load_report_records
except ImportError:
    BERT_MODEL_ROOT = Path(__file__).resolve().parents[1]
    if str(BERT_MODEL_ROOT) not in sys.path:
        sys.path.insert(0, str(BERT_MODEL_ROOT))
    from report_split_io import ReportRecord, load_report_records

    def data_path(name: str) -> str:
        return str(BERT_MODEL_ROOT / "data" / name)

    def eval_data_path(name: str) -> str:
        return str(BERT_MODEL_ROOT / "data" / "eval_datasets" / name)

    NO_FINDINGS_DIR = BERT_MODEL_ROOT / "no_findings_analysis"


SWAP_INSTANCES_CHOICES = ("all", "random", "first", "single")
SINGLE_SWAP_INSTANCES = frozenset({"random", "first", "single"})
MODE_CHOICES = ("cross_pair_only", "hard_v3", "hard_v4", "hard_v5")
RULE_SELECTION_CHOICES = ("first", "random")

LATERALITY_SWAP_RULES: List[Tuple[str, str, str]] = [
    ("left", "right", "laterality_swap"),
]

SITE_SWAP_RULES: List[Tuple[str, str, str]] = [
    ("upper", "lower", "site_swap"),
    ("top", "bottom", "site_swap"),
    ("thoracic", "abdominal", "site_swap"),
    ("apical", "basal", "site_swap"),
    ("proximal", "distal", "site_swap"),
    ("anterior", "posterior", "site_swap"),
    ("lung", "liver", "site_swap"),
    ("liver", "bone", "site_swap"),
    ("pleural", "parenchymal", "site_swap"),
]

try:
    from src.severity_flip import (
        SEVERITY_FLIP_RULES,
        SEVERITY_FLIP_TERMS,
        apply_severity_flip,
    )
except ImportError:
    _src = Path(__file__).resolve().parents[1] / "src"
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
    from severity_flip import (  # type: ignore[import-not-found]
        SEVERITY_FLIP_RULES,
        SEVERITY_FLIP_TERMS,
        apply_severity_flip,
    )


@dataclass
class EvalExample:
    patient_id: str
    source_idx: int
    findings: str
    conclusion: str
    label: int
    error_type: str
    incidence: str = ""


@dataclass(frozen=True)
class HumanNfReviewRow:
    source_row: str
    conclusion: str
    is_nf: bool
    patient_id: str = ""


def _rule_terms(rules: Sequence[Tuple[str, str, str]]) -> List[str]:
    terms: List[str] = []
    seen: set[str] = set()
    for left, right, _ in rules:
        for term in (left, right):
            key = term.lower()
            if key not in seen:
                seen.add(key)
                terms.append(term)
    return terms


LATERALITY_SWAP_TERMS = _rule_terms(LATERALITY_SWAP_RULES)
SITE_SWAP_TERMS = _rule_terms(SITE_SWAP_RULES)
ERROR_TYPE_INCIDENCE_TERMS: Dict[str, List[str]] = {
    "laterality_swap": LATERALITY_SWAP_TERMS,
    "site_swap": SITE_SWAP_TERMS,
    "severity_flip": SEVERITY_FLIP_TERMS,
    "consistent": LATERALITY_SWAP_TERMS,
    "positive": LATERALITY_SWAP_TERMS,
}


def _is_no_findings(conclusion: str) -> bool:
    c_lower = conclusion.strip().lower()
    if not c_lower.startswith("no "):
        return False
    words = c_lower.split()
    if len(words) > 1 and words[1] in {"other", "new", "further", "additional", "remaining"}:
        return False
    if "fdg" not in c_lower:
        return False
    return bool(re.search(r"\b(?:disease|recurrence|residual)\b", c_lower))


def parse_human_nf_label(val: str) -> Optional[int]:
    v = str(val).strip().lower()
    if v in ("0", "1"):
        return int(v)
    return None


def load_nf_labels_from_csv(csv_path: str, split: str) -> Dict[str, bool]:
    labels: Dict[str, bool] = {}
    try:
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if row["split"] == split:
                    labels[row["conclusion"]] = row["nf_label"] == "1"
        print(f"Loaded {len(labels):,} NF labels for split='{split}' from {csv_path}")
    except FileNotFoundError:
        print(f"Warning: NF labels CSV not found at {csv_path}; falling back to regex.")
    return labels


def load_human_nf_labels_from_csv(
    csv_path: str,
    split: Optional[str] = None,
) -> Dict[str, bool]:
    """Load definitive human_nf labels keyed by conclusion (0/1 only; excludes x)."""
    return {
        row.conclusion: row.is_nf
        for row in load_human_nf_review_rows(csv_path, split=split)
    }


def load_human_nf_review_rows(
    csv_path: str,
    split: Optional[str] = None,
) -> List[HumanNfReviewRow]:
    """Load human-reviewed rows with definitive human_nf (0/1; excludes x)."""
    rows: List[HumanNfReviewRow] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if split is not None and row.get("split") != split:
                continue
            parsed = parse_human_nf_label(row.get("human_nf", ""))
            if parsed is None:
                continue
            rows.append(
                HumanNfReviewRow(
                    source_row=str(row.get("source_row", "")).strip(),
                    conclusion=str(row.get("conclusion", "")).strip(),
                    is_nf=parsed == 1,
                    patient_id=str(row.get("patient_id", "")).strip(),
                )
            )
    print(
        f"Loaded {len(rows):,} definitive human_nf review rows"
        + (f" for split='{split}'" if split else "")
        + f" from {csv_path}"
    )
    return rows


def _is_nf_labeled(conclusion: str, labels: Optional[Dict[str, bool]]) -> bool:
    if labels is not None:
        return labels.get(conclusion, False)
    return _is_no_findings(conclusion)


def count_swap_keyword_incidence(text: str, terms: Sequence[str]) -> int:
    if not text or not terms:
        return 0
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b",
        re.IGNORECASE,
    )
    return len(pattern.findall(text))


def swap_incidence_for_example(
    conclusion: str,
    error_type: str,
    original_conclusion: Optional[str] = None,
) -> str:
    et = "consistent" if error_type == "positive" else error_type
    terms = ERROR_TYPE_INCIDENCE_TERMS.get(et)
    if terms is None:
        return ""
    source = original_conclusion if original_conclusion is not None else conclusion
    return str(count_swap_keyword_incidence(source, terms))


def _replacement_for_token(token: str, a: str, b: str) -> str:
    token_lower = token.lower()
    replacement = b if token_lower == a.lower() else a
    if token.isupper():
        return replacement.upper()
    if token[0].isupper():
        return replacement.capitalize()
    return replacement


def _swap_term(
    text: str,
    a: str,
    b: str,
    swap_instances: str,
    rng: random.Random,
) -> str:
    if swap_instances not in SWAP_INSTANCES_CHOICES:
        raise ValueError(
            f"swap_instances must be one of {SWAP_INSTANCES_CHOICES}, got {swap_instances!r}"
        )
    pattern = re.compile(rf"\b({re.escape(a)}|{re.escape(b)})\b", re.IGNORECASE)
    matches = list(pattern.finditer(text))
    if not matches:
        return text

    if swap_instances == "all":

        def repl(match: re.Match) -> str:
            return _replacement_for_token(match.group(0), a, b)

        return pattern.sub(repl, text)

    if swap_instances == "first":
        chosen = 0
    elif swap_instances in SINGLE_SWAP_INSTANCES:
        chosen = rng.randrange(len(matches))
    else:
        raise ValueError(
            f"swap_instances must be one of {SWAP_INSTANCES_CHOICES}, got {swap_instances!r}"
        )
    match = matches[chosen]
    replacement = _replacement_for_token(match.group(0), a, b)
    return text[: match.start()] + replacement + text[match.end() :]


def apply_first_matching_rule(
    text: str,
    rules: Sequence[Tuple[str, str, str]],
    swap_instances: str,
    rng: random.Random,
) -> Tuple[Optional[str], Optional[str]]:
    for left, right, error_type in rules:
        pattern = re.compile(rf"\b({re.escape(left)}|{re.escape(right)})\b", re.IGNORECASE)
        if pattern.search(text):
            return _swap_term(text, left, right, swap_instances=swap_instances, rng=rng), error_type
    return None, None


def apply_random_matching_rule(
    text: str,
    rules: Sequence[Tuple[str, str, str]],
    swap_instances: str,
    rng: random.Random,
) -> Tuple[Optional[str], Optional[str]]:
    matching: List[Tuple[str, str, str]] = []
    for left, right, error_type in rules:
        pattern = re.compile(
            rf"\b({re.escape(left)}|{re.escape(right)})\b", re.IGNORECASE
        )
        if pattern.search(text):
            matching.append((left, right, error_type))
    if not matching:
        return None, None
    left, right, error_type = rng.choice(matching)
    return _swap_term(text, left, right, swap_instances=swap_instances, rng=rng), error_type


def _apply_matching_rule(
    text: str,
    rules: Sequence[Tuple[str, str, str]],
    swap_instances: str,
    rng: random.Random,
    rule_selection: str,
) -> Tuple[Optional[str], Optional[str]]:
    if rule_selection == "random":
        return apply_random_matching_rule(text, rules, swap_instances, rng)
    if rule_selection == "first":
        return apply_first_matching_rule(text, rules, swap_instances, rng)
    raise ValueError(
        f"rule_selection must be one of {RULE_SELECTION_CHOICES}, got {rule_selection!r}"
    )


def apply_laterality_swap(
    text: str,
    swap_instances: str,
    rng: random.Random,
    rule_selection: str = "first",
) -> Optional[str]:
    out, _ = _apply_matching_rule(
        text, LATERALITY_SWAP_RULES, swap_instances, rng, rule_selection
    )
    return out


def apply_site_swap(
    text: str,
    swap_instances: str,
    rng: random.Random,
    rule_selection: str = "first",
) -> Optional[str]:
    out, _ = _apply_matching_rule(
        text, SITE_SWAP_RULES, swap_instances, rng, rule_selection
    )
    return out


def _cross_pair_partner_index(idx: int, n: int, rng: random.Random) -> Optional[int]:
    if n <= 1:
        return None
    j = rng.randrange(n)
    while j == idx:
        j = rng.randrange(n)
    return j


def _eval_version_tag(mode: str, human_nf: bool) -> str:
    if human_nf:
        return "v6"
    if mode == "hard_v5":
        return "v5"
    if mode == "hard_v4":
        return "v4"
    return "v3"


def _default_output_for_mode(
    mode: str,
    swap_instances: str,
    human_nf: bool = False,
) -> str:
    if mode == "cross_pair_only":
        return eval_data_path("cross_pair_full_eval.csv")
    version = _eval_version_tag(mode, human_nf)
    if swap_instances in SINGLE_SWAP_INSTANCES:
        return eval_data_path(f"hard_negative_eval_{version}_single_swap.csv")
    return eval_data_path(f"hard_negative_eval_{version}.csv")


def build_cross_pair_only_eval(
    records: Sequence[ReportRecord],
    rng: random.Random,
) -> List[EvalExample]:
    rows: List[EvalExample] = []
    for idx, rec in enumerate(records):
        rows.append(
            EvalExample(
                patient_id=rec.patient_id,
                source_idx=idx,
                findings=rec.findings,
                conclusion=rec.conclusion,
                label=1,
                error_type="consistent",
            )
        )
        partner_idx = _cross_pair_partner_index(idx, len(records), rng)
        if partner_idx is None:
            continue
        rows.append(
            EvalExample(
                patient_id=rec.patient_id,
                source_idx=idx,
                findings=rec.findings,
                conclusion=records[partner_idx].conclusion,
                label=0,
                error_type="cross_pair_mismatch",
            )
        )
    return rows


def build_hard_eval(
    records: Sequence[ReportRecord],
    mode: str,
    swap_instances: str,
    nf_labels: Optional[Dict[str, bool]],
    rng: random.Random,
    human_nf_review_rows: Optional[Sequence[HumanNfReviewRow]] = None,
) -> List[EvalExample]:
    include_cross_pair = mode in ("hard_v4", "hard_v5")
    rule_selection = "random" if mode == "hard_v5" else "first"
    rows: List[EvalExample] = []
    hard_generators: List[Tuple[str, Callable[[str], Optional[str]]]] = [
        (
            "laterality_swap",
            partial(
                apply_laterality_swap,
                swap_instances=swap_instances,
                rng=rng,
                rule_selection=rule_selection,
            ),
        ),
        (
            "site_swap",
            partial(
                apply_site_swap,
                swap_instances=swap_instances,
                rng=rng,
                rule_selection=rule_selection,
            ),
        ),
        (
            "severity_flip",
            partial(
                apply_severity_flip,
                swap_instances=swap_instances,
                rng=rng,
                rule_selection=rule_selection,
            ),
        ),
    ]

    human_nf_by_source_row: Dict[str, bool] = {}
    if human_nf_review_rows is not None:
        nf_pool = [hr.conclusion for hr in human_nf_review_rows if hr.is_nf]
        disease_pool = [hr.conclusion for hr in human_nf_review_rows if not hr.is_nf]
        human_nf_by_source_row = {hr.source_row: hr.is_nf for hr in human_nf_review_rows}
        label_source = "human_nf"
    else:
        conclusions = [r.conclusion for r in records]
        nf_pool = [c for c in conclusions if _is_nf_labeled(c, nf_labels)]
        disease_pool = [c for c in conclusions if not _is_nf_labeled(c, nf_labels)]
        label_source = "CSV" if nf_labels else "regex"
    print(
        f"build_hard_eval ({mode}): NF pool={len(nf_pool)}, disease pool={len(disease_pool)} "
        f"(labels: {label_source}), swap_instances={swap_instances}, "
        f"rule_selection={rule_selection}, include_cross_pair={include_cross_pair}"
    )

    human_nf_swap_count = 0
    for idx, rec in enumerate(records):
        base = {
            "patient_id": rec.patient_id,
            "source_idx": idx,
            "findings": rec.findings,
            "conclusion": rec.conclusion,
        }
        rows.append(
            EvalExample(
                **base,
                label=1,
                error_type="consistent",
                incidence=swap_incidence_for_example(rec.conclusion, "consistent"),
            )
        )

        if include_cross_pair:
            partner_idx = _cross_pair_partner_index(idx, len(records), rng)
            if partner_idx is not None:
                rows.append(
                    EvalExample(
                        patient_id=rec.patient_id,
                        source_idx=idx,
                        findings=rec.findings,
                        conclusion=records[partner_idx].conclusion,
                        label=0,
                        error_type="cross_pair_mismatch",
                    )
                )

        for error_type, fn in hard_generators:
            transformed = fn(rec.conclusion)
            if not transformed or transformed == rec.conclusion:
                continue
            rows.append(
                EvalExample(
                    patient_id=rec.patient_id,
                    source_idx=idx,
                    findings=rec.findings,
                    conclusion=transformed,
                    label=0,
                    error_type=error_type,
                    incidence=swap_incidence_for_example(
                        rec.conclusion, error_type, original_conclusion=rec.conclusion
                    ),
                )
            )

        if human_nf_review_rows is not None:
            if rec.source_row not in human_nf_by_source_row:
                continue
            swap_pool = (
                disease_pool
                if human_nf_by_source_row[rec.source_row]
                else nf_pool
            )
        else:
            swap_pool = (
                disease_pool
                if _is_nf_labeled(rec.conclusion, nf_labels)
                else nf_pool
            )
        swap_candidates = [c for c in swap_pool if c != rec.conclusion]
        if swap_candidates:
            rows.append(
                EvalExample(
                    patient_id=rec.patient_id,
                    source_idx=idx,
                    findings=rec.findings,
                    conclusion=rng.choice(swap_candidates),
                    label=0,
                    error_type="NF_v_Dis_swap",
                )
            )
            if human_nf_review_rows is not None:
                human_nf_swap_count += 1

    if human_nf_review_rows is not None:
        print(
            f"human_nf NF_v_Dis_swap rows: {human_nf_swap_count}/"
            f"{len(human_nf_by_source_row)} reviewed source_rows"
        )

    return rows


def write_csv(rows: Sequence[EvalExample], output_path: str) -> None:
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "patient_id",
                "source_idx",
                "findings",
                "conclusion",
                "label",
                "error_type",
                "incidence",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "patient_id": r.patient_id,
                    "source_idx": r.source_idx,
                    "findings": r.findings,
                    "conclusion": r.conclusion,
                    "label": r.label,
                    "error_type": r.error_type,
                    "incidence": r.incidence,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build evaluation datasets for findings/conclusion consistency."
    )
    parser.add_argument(
        "--input",
        default=data_path("test_text_path"),
        help="Input report split file (e.g., test_text_path).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default inferred from --mode and --swap_instances).",
    )
    parser.add_argument(
        "--mode",
        choices=MODE_CHOICES,
        default="hard_v4",
        help=(
            "cross_pair_only: positives + cross-pair mismatches only. "
            "hard_v3: consistent + hard errors (laterality/site/severity + NF_v_Dis_swap). "
            "hard_v4: hard_v3 + cross_pair_mismatch rows (first matching swap rule). "
            "hard_v5: same as hard_v4 but random choice among all matching swap rules."
        ),
    )
    parser.add_argument(
        "--swap_instances",
        choices=SWAP_INSTANCES_CHOICES,
        default="all",
        help=(
            "Keyword swap scope for laterality/site/severity: "
            "all=every match; single/random=one trigger word (random position); "
            "first=first match only."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for mismatch pairing and random swap behavior.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Optional cap on source records before generation.",
    )
    parser.add_argument(
        "--nf_labels_csv",
        default=str(NO_FINDINGS_DIR / "no_findings_labels.csv"),
        help="Path to no_findings_labels.csv used for NF_v_Dis_swap pool assignment.",
    )
    parser.add_argument(
        "--nf_labels_split",
        default="test",
        choices=["train", "val", "test"],
        help="Split key used when loading --nf_labels_csv or --human_nf_csv.",
    )
    parser.add_argument(
        "--human_nf",
        action="store_true",
        help=(
            "Use definitive human_nf labels from --human_nf_csv for NF_v_Dis_swap "
            "(only rows with human_nf 0/1; excludes x). Default output uses v6 naming."
        ),
    )
    parser.add_argument(
        "--human_nf_csv",
        default=str(NO_FINDINGS_DIR / "no_findings_test_set_human_review.csv"),
        help="Human review CSV with human_nf column (used with --human_nf).",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    records = load_report_records(args.input)
    if not records:
        raise ValueError(f"No reports found in source file: {args.input}")

    if args.max_examples is not None:
        records = list(records[: args.max_examples])

    if args.output is None:
        args.output = _default_output_for_mode(
            args.mode, args.swap_instances, human_nf=args.human_nf
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {len(records)} reports from {args.input}")
    print(
        f"Building mode={args.mode}, swap_instances={args.swap_instances}, "
        f"seed={args.seed}, human_nf={args.human_nf}"
    )

    if args.mode == "cross_pair_only":
        rows = build_cross_pair_only_eval(records=records, rng=rng)
    else:
        human_nf_review_rows: Optional[List[HumanNfReviewRow]] = None
        if args.human_nf:
            human_nf_review_rows = load_human_nf_review_rows(
                args.human_nf_csv, split=args.nf_labels_split
            )
            if not human_nf_review_rows:
                raise ValueError(
                    f"No definitive human_nf labels found in {args.human_nf_csv} "
                    f"for split='{args.nf_labels_split}'"
                )
            nf_labels = None
        else:
            nf_labels = load_nf_labels_from_csv(args.nf_labels_csv, split=args.nf_labels_split)
        rows = build_hard_eval(
            records=records,
            mode=args.mode,
            swap_instances=args.swap_instances,
            nf_labels=nf_labels,
            rng=rng,
            human_nf_review_rows=human_nf_review_rows,
        )

    write_csv(rows, args.output)
    type_counts = Counter(r.error_type for r in rows)
    print(f"Wrote eval dataset to: {args.output}")
    print(f"rows={len(rows)} error_type_counts={dict(sorted(type_counts.items()))}")


if __name__ == "__main__":
    main()
