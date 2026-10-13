import argparse
import csv
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers.processors import TemplateProcessing
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertForSequenceClassification, BertModel
import wandb
from sklearn.metrics import roc_auc_score as _sklearn_roc_auc_score

def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUROC with graceful fallback for single-class input."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(_sklearn_roc_auc_score(y_true, y_score))

try:
    from paths import (
        DEFAULT_TEST_CSV,
        DEFAULT_TOKENIZER_PATH,
        DEFAULT_TRAIN_CSV,
        DEFAULT_VAL_CSV,
        NO_FINDINGS_DIR,
        RunLayout,
        checkpoint_path,
        create_run,
        current_argv,
        data_path,
        default_plot_path,
        default_result_path,
        eval_data_path,
        DEFAULT_SEQ_LENGTH,
        init_cls_checkpoint_path,
        init_model_checkpoint_path,
        resolve_checkpoint,
        SUPPORTED_SEQ_LENGTHS,
        setup_run,
        tee_run_output,
        RUN_OUTPUT_LOG,
    )
    from report_split_io import (
        load_findings_conclusion_pairs,
        load_report_records,
    )
    from src.predictions_eval import (
        INCIDENCE_EVAL_MODES,
        write_eval_artifacts,
    )
    from src.severity_flip import (
        SEVERITY_FLIP_RULES,
        SEVERITY_FLIP_TERMS,
        SEVERITY_SCALAR_RULES,
        SEVERITY_TIERS,
        SeverityFlipOptions,
        apply_severity_flip as _apply_severity_flip_impl,
        apply_severity_jitter as _apply_severity_jitter_impl,
        parse_excluded_severity_terms,
    )
except ImportError:
    _repo_root = Path(__file__).resolve().parents[1]
    import sys

    if str(_repo_root) not in sys.path:
        sys.path.append(str(_repo_root))
    from paths import (
        DEFAULT_TEST_CSV,
        DEFAULT_TOKENIZER_PATH,
        DEFAULT_TRAIN_CSV,
        DEFAULT_VAL_CSV,
        NO_FINDINGS_DIR,
        RunLayout,
        checkpoint_path,
        create_run,
        current_argv,
        data_path,
        default_plot_path,
        default_result_path,
        eval_data_path,
        DEFAULT_SEQ_LENGTH,
        init_cls_checkpoint_path,
        init_model_checkpoint_path,
        resolve_checkpoint,
        SUPPORTED_SEQ_LENGTHS,
        setup_run,
        tee_run_output,
        RUN_OUTPUT_LOG,
    )
    from report_split_io import (
        load_findings_conclusion_pairs,
        load_report_records,
    )
    from src.predictions_eval import (
        INCIDENCE_EVAL_MODES,
        write_eval_artifacts,
    )
    from src.severity_flip import (
        SEVERITY_FLIP_RULES,
        SEVERITY_FLIP_TERMS,
        SEVERITY_SCALAR_RULES,
        SEVERITY_TIERS,
        SeverityFlipOptions,
        apply_severity_flip as _apply_severity_flip_impl,
        apply_severity_jitter as _apply_severity_jitter_impl,
        parse_excluded_severity_terms,
    )

# -------------------------
# CONFIG
# -------------------------
#MODEL_PATH = checkpoint_path("bert-contrastive_bs128_090526")
MODEL_PATH = init_cls_checkpoint_path(DEFAULT_SEQ_LENGTH)
TOKENIZER_PATH = DEFAULT_TOKENIZER_PATH
TRAIN_FILE = str(DEFAULT_TRAIN_CSV)
VAL_FILE = str(DEFAULT_VAL_CSV)
TEST_FILE = str(DEFAULT_TEST_CSV)
NF_LABELS_CSV = str(NO_FINDINGS_DIR / "no_findings_labels.csv")
from src.nf_v_dis_human_eval import (  # noqa: E402
    NF_HUMAN_REVIEW_CSV,
    build_nf_v_dis_human_eval_examples as _build_human_eval_examples,
)
from src.nf_swap_eval import (  # noqa: E402
    print_nf_swap_direction_table,
    summarize_nf_swap_by_direction,
)
MAX_LENGTH = DEFAULT_SEQ_LENGTH
MODEL_SIZE = "small"
SCRATCH_INIT = False
BATCH_SIZE = 128
EPOCHS = 20
LR = 2e-5
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_CHECKPOINT = "bert-inconsistency-detector-hard-neg-v3"
DEFAULT_HARD_EVAL_CSV = eval_data_path("hard_negative_eval_v3.csv")
SWAP_INSTANCES_CHOICES = ("all", "random", "random_n", "single")
# Modes that swap exactly one keyword match (per laterality/site/severity corruption).
SINGLE_SWAP_INSTANCES = frozenset({"random", "single"})
DEFAULT_CROSS_PAIR_EVAL_CSV = eval_data_path("cross_pair_full_eval.csv")
DEFAULT_VAL_HARD_EVAL_CSV = eval_data_path("val_hard_negative_eval.csv")
SCRIPT_ID = "bert-inconsistency"
RUN_CATEGORY = "bert_binary"

TRAINING_METRICS_FIELDS = [
    "epoch",
    "train_loss",
    "val_loss",
    "val_auroc",
    "val_recall_inconsistent",
    "val_fp_rate",
    "val_acc",
    "val_f1_inconsistent",
    "val_confusion_tp0",
    "val_confusion_fp0",
    "val_confusion_fn0",
    "val_confusion_tn0",
]


def _append_training_log(log_dir: Path, line: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "training_log.txt", "a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def default_model_dir() -> str:
    return checkpoint_path(DEFAULT_CHECKPOINT)
CLS_HEAD_FILE = "cls_classifier.pt"

FINDINGS_SPECIAL = "[Findings]"
CONCLUSION_SPECIAL = "[Conclusion]"
tokenizer = None


def format_joint_report(findings: str, conclusion: str) -> str:
    return (
        f"{tokenizer.cls_token} {FINDINGS_SPECIAL} {findings} "
        f"{CONCLUSION_SPECIAL} {conclusion} {tokenizer.sep_token}"
    )


def normalize_error_type_name(error_type: str) -> str:
    et = (error_type or "unknown").strip()
    if et == "positive":
        return "consistent"
    return et


def row_included_in_eval_mode(error_type: str, mode: str) -> bool:
    """Match CsvEvalDataset row filtering (all/hard/lat/severity/cross_pair_full)."""
    et = normalize_error_type_name(error_type)
    if mode == "cross_pair_full":
        return et in {"consistent", "cross_pair_mismatch"}
    if mode == "lat":
        return et in {"consistent", "laterality_swap"}
    if mode == "site":
        return et in {"consistent", "site_swap"}
    if mode == "severity":
        return et in {"consistent", "severity_flip"}
    return True


def configure_seq_length(
    seq_length: int,
    *,
    wo_cls_init: bool = False,
    scratch_init: bool = False,
    init_checkpoint: Optional[str] = None,
    model_size: str = "small",
    batch_size: Optional[int] = None,
) -> None:
    """Set module max token length and matching init checkpoint."""
    global MAX_LENGTH, MODEL_PATH, BATCH_SIZE, MODEL_SIZE, SCRATCH_INIT
    if seq_length not in SUPPORTED_SEQ_LENGTHS:
        raise ValueError(
            f"seq_length must be one of {SUPPORTED_SEQ_LENGTHS}, got {seq_length!r}"
        )
    MAX_LENGTH = seq_length
    MODEL_SIZE = model_size
    SCRATCH_INIT = scratch_init
    if scratch_init:
        MODEL_PATH = f"<scratch:{model_size}>"
        if batch_size is not None:
            BATCH_SIZE = batch_size
        return
    if init_checkpoint:
        init_path = Path(init_checkpoint)
        if init_path.is_dir() and (init_path / "config.json").exists():
            MODEL_PATH = str(init_path.resolve())
        else:
            MODEL_PATH = checkpoint_path(init_checkpoint)
    else:
        MODEL_PATH = init_model_checkpoint_path(
            seq_length, wo_cls_init=wo_cls_init, model_size=model_size
        )
    if batch_size is not None:
        BATCH_SIZE = batch_size


def init_tokenizer(tokenizer_path: str):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    tokenizer.add_special_tokens({"cls_token": "[CLS]"})
    extra = []
    vocab = tokenizer.get_vocab()
    for tok in (FINDINGS_SPECIAL, CONCLUSION_SPECIAL):
        if tok not in vocab:
            extra.append(tok)
    if extra:
        tokenizer.add_special_tokens({"additional_special_tokens": extra})
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single="$A",
        pair="$A $B",
        special_tokens=[],
    )
    return tokenizer


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_findings_conclusion(file_path: str) -> List[Tuple[str, str]]:
    return load_findings_conclusion_pairs(file_path)


def _is_no_findings(conclusion: str) -> bool:
    """Return True if the conclusion represents a fully normal (no-disease) scan.

    A no-findings conclusion in PET/CT reports typically starts with "No" and
    references FDG-avid disease/recurrence/residual without implying that
    disease was found elsewhere.  Qualifiers such as "other", "new", "further",
    "additional", and "remaining" after "No" are excluded because phrases like
    "No other FDG-avid disease" imply that primary disease WAS found.
    """
    c_lower = conclusion.strip().lower()
    if not c_lower.startswith("no "):
        return False
    words = c_lower.split()
    if len(words) > 1 and words[1] in {"other", "new", "further", "additional", "remaining"}:
        return False
    if "fdg" not in c_lower:
        return False
    return bool(re.search(r"\b(?:disease|recurrence|residual)\b", c_lower))


def load_nf_labels_from_csv(csv_path: str, split: str) -> Dict[str, bool]:
    """Load per-split no-findings labels from no_findings_labels.csv.

    Returns a dict mapping conclusion text -> is_no_findings (bool).
    Falls back to an empty dict (callers then use regex) if the file is
    missing or the split has no rows.
    """
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


def _is_nf_labeled(conclusion: str, labels: Optional[Dict[str, bool]]) -> bool:
    """Classify conclusion as NF using the CSV labels dict.

    If labels is provided but the conclusion is not found, returns False (treat
    as disease) rather than falling back to regex.  A missing entry is more
    likely a data-version mismatch than a genuine NF conclusion, and an
    incorrect NF classification would corrupt the NF/disease swap pools.

    Falls back to the regex _is_no_findings() only when no labels dict is
    supplied at all (i.e. the CSV was not loaded).
    """
    if labels is not None:
        return labels.get(conclusion, False)
    return _is_no_findings(conclusion)


def _apply_error_conclusion_only(
    findings: str, conclusion: str, fn: Callable[[str], Optional[str]]
) -> Optional[Tuple[str, str]]:
    """Apply generator to the conclusion only; findings are left unchanged.

    Returns (findings, new_conclusion) if the generator fires on the conclusion,
    otherwise None.  This guarantees that every negative training example has a
    verifiable inconsistency: the findings remain the ground-truth reference and
    the corrupted conclusion contradicts them.
    """
    new_conclusion = fn(conclusion)
    if new_conclusion and new_conclusion != conclusion:
        return findings, new_conclusion
    return None


def _apply_error_random_side(
    findings: str, conclusion: str, fn: Callable[[str], Optional[str]]
) -> Optional[Tuple[str, str]]:
    corrupt_findings_first = random.random() < 0.5
    first_text = findings if corrupt_findings_first else conclusion
    second_text = conclusion if corrupt_findings_first else findings

    first_out = fn(first_text)
    if first_out and first_out != first_text:
        if corrupt_findings_first:
            return first_out, conclusion
        return findings, first_out

    second_out = fn(second_text)
    if second_out and second_out != second_text:
        if corrupt_findings_first:
            return findings, second_out
        return second_out, conclusion
    return None


@dataclass
class PairExample:
    findings: str
    conclusion: str
    label: int  # 1 = consistent, 0 = inconsistent
    error_type: str = "unknown"
    patient_id: str = ""
    source_idx: Optional[int] = None


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


def _metadata_from_eval_csv_row(row: dict) -> Tuple[str, Optional[int]]:
    patient_id = (row.get("patient_id") or row.get("PatientID") or "").strip()
    return patient_id, _optional_int(row.get("source_idx"))


class InconsistencyDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], negatives_per_positive: int = 1):
        self.examples: List[PairExample] = []

        # Positives
        for findings, conclusion in pairs:
            self.examples.append(PairExample(findings, conclusion, 1, "consistent"))

        # Negatives (randomly mismatched conclusions)
        conclusions = [c for _, c in pairs]
        n = len(pairs)
        for i, (findings, true_conclusion) in enumerate(pairs):
            used = set()
            for _ in range(negatives_per_positive):
                j = random.randrange(n)
                while j == i or conclusions[j] == true_conclusion or j in used:
                    j = random.randrange(n)
                used.add(j)
                self.examples.append(PairExample(findings, conclusions[j], 0, "random_mismatch"))

        random.shuffle(self.examples)
        print(f"Built dataset with {len(self.examples)} examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


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

# Human-readable groups for coverage plots / notebooks (synced with swap rules above).
TRIGGER_WORD_GROUPS: Dict[str, List[str]] = {
    "Laterality": LATERALITY_SWAP_TERMS,
    "Site": SITE_SWAP_TERMS,
    "Severity": SEVERITY_FLIP_TERMS,
}

ERROR_TYPE_INCIDENCE_TERMS: Dict[str, List[str]] = {
    "laterality_swap": LATERALITY_SWAP_TERMS,
    "site_swap": SITE_SWAP_TERMS,
    "severity_flip": SEVERITY_FLIP_TERMS,
    "consistent": LATERALITY_SWAP_TERMS,
    "positive": LATERALITY_SWAP_TERMS,
}


def count_swap_keyword_incidence(text: str, terms: Sequence[str]) -> int:
    """Count word-boundary matches for swap-eligible terms (all instances)."""
    if not text or not terms:
        return 0
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b",
        re.IGNORECASE,
    )
    return len(pattern.findall(text))


def incidence_terms_for_example(
    error_type: str,
    incidence_profile: Optional[str] = None,
) -> Optional[Sequence[str]]:
    """Terms used to count keyword incidence for predictions / reports."""
    et = "consistent" if error_type == "positive" else error_type
    if incidence_profile == "severity":
        if et in {"consistent", "severity_flip"}:
            return SEVERITY_FLIP_TERMS
    if incidence_profile == "site":
        if et in {"consistent", "site_swap"}:
            return SITE_SWAP_TERMS
    return ERROR_TYPE_INCIDENCE_TERMS.get(et)


def swap_incidence_for_example(
    conclusion: str,
    error_type: str,
    original_conclusion: Optional[str] = None,
    incidence_profile: Optional[str] = None,
) -> str:
    """Incidence of swappable keywords before corruption (1 = single-instance swap)."""
    terms = incidence_terms_for_example(error_type, incidence_profile=incidence_profile)
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


def _swap_term(text: str, a: str, b: str, swap_instances: str = "all") -> str:
    """Swap term pair in text.

    swap_instances:
      - all: swap every match
      - single / random: swap exactly one match (random position)
      - random_n: swap a random number of matches (uniform from 1..N)
    """
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

    if swap_instances == "random_n":
        chosen_count = random.randint(1, len(matches))
        chosen_indices = set(random.sample(range(len(matches)), k=chosen_count))
        out_parts: List[str] = []
        cursor = 0
        for i, match in enumerate(matches):
            out_parts.append(text[cursor : match.start()])
            token = match.group(0)
            if i in chosen_indices:
                out_parts.append(_replacement_for_token(token, a, b))
            else:
                out_parts.append(token)
            cursor = match.end()
        out_parts.append(text[cursor:])
        return "".join(out_parts)

    if swap_instances in SINGLE_SWAP_INSTANCES:
        match = matches[random.randrange(len(matches))]
        replacement = _replacement_for_token(match.group(0), a, b)
        return text[: match.start()] + replacement + text[match.end() :]

    raise ValueError(
        f"swap_instances must be one of {SWAP_INSTANCES_CHOICES}, got {swap_instances!r}"
    )


def apply_first_matching_rule(
    text: str,
    rules: Sequence[Tuple[str, str, str]],
    swap_instances: str = "all",
) -> Tuple[Optional[str], Optional[str]]:
    for left, right, error_type in rules:
        pattern = re.compile(rf"\b({re.escape(left)}|{re.escape(right)})\b", re.IGNORECASE)
        if pattern.search(text):
            return _swap_term(text, left, right, swap_instances=swap_instances), error_type
    return None, None


def apply_random_matching_rule(
    text: str,
    rules: Sequence[Tuple[str, str, str]],
    swap_instances: str = "all",
) -> Tuple[Optional[str], Optional[str]]:
    """Pick uniformly among all rules that match the text, then apply swap_instances."""
    matching: List[Tuple[str, str, str]] = []
    for left, right, error_type in rules:
        pattern = re.compile(
            rf"\b({re.escape(left)}|{re.escape(right)})\b", re.IGNORECASE
        )
        if pattern.search(text):
            matching.append((left, right, error_type))
    if not matching:
        return None, None
    left, right, error_type = random.choice(matching)
    return _swap_term(text, left, right, swap_instances=swap_instances), error_type


def flip_negation(text: str) -> Optional[str]:
    """Flip negative phrasing to positive (removes or replaces negative markers).

    Trigger words (negative → positive):
      no evidence of → evidence of
      without evidence of → with evidence of
      no → (removed)
      absent → present
      negative for → positive for
      not seen → seen
      none → present
    """
    substitutions: List[Tuple[str, str]] = [
        (r"\bno evidence of\b", "evidence of"),
        (r"\bwithout evidence of\b", "with evidence of"),
        (r"\bno\b", ""),
        (r"\babsent\b", "present"),
        (r"\bnegative for\b", "positive for"),
        (r"\bnot seen\b", "seen"),
        (r"\bnone\b", "present"),
    ]
    for pattern, repl in substitutions:
        regex = re.compile(pattern, re.IGNORECASE)
        if regex.search(text):
            out = regex.sub(repl, text, count=1)
            out = re.sub(r"\s{2,}", " ", out).strip()
            return out
    return None


def flip_negation_reverse(text: str) -> Optional[str]:
    """Flip positive phrasing to negative (reverse direction of flip_negation).

    Complements flip_negation by inserting negations into positively-phrased
    text, creating a "false-negative" error (conclusion claims no disease when
    findings describe disease).

    Trigger words (positive → negative):
      with evidence of → without evidence of
      positive for → negative for
      seen → not seen
      present → absent
    """
    substitutions: List[Tuple[str, str]] = [
        (r"\bno evidence of\b", "evidence of"),
        (r"\bno fdg avid disease\b", "fdg avid disease"),
        (r"\bwith evidence of\b", "without evidence of"),
        (r"\bpositive for\b", "negative for"),
        (r"\bseen\b", "not seen"),
        (r"\bpresent\b", "absent"),
    ]
    for pattern, repl in substitutions:
        regex = re.compile(pattern, re.IGNORECASE)
        if regex.search(text):
            out = regex.sub(repl, text, count=1)
            out = re.sub(r"\s{2,}", " ", out).strip()
            return out
    return None


def apply_laterality_swap(text: str, swap_instances: str = "all") -> Optional[str]:
    out, _ = apply_random_matching_rule(
        text, LATERALITY_SWAP_RULES, swap_instances=swap_instances
    )
    return out


def apply_site_swap(text: str, swap_instances: str = "all") -> Optional[str]:
    out, _ = apply_random_matching_rule(
        text, SITE_SWAP_RULES, swap_instances=swap_instances
    )
    return out


def apply_severity_flip(
    text: str,
    swap_instances: str = "all",
    options: Optional[SeverityFlipOptions] = None,
) -> Optional[str]:
    return _apply_severity_flip_impl(
        text, swap_instances=swap_instances, options=options
    )


def build_severity_flip_options(
    exclude_scalar: bool = False,
    excluded_terms: Optional[str] = None,
) -> SeverityFlipOptions:
    return SeverityFlipOptions(
        exclude_scalar=exclude_scalar,
        excluded_terms=parse_excluded_severity_terms(excluded_terms),
    )


class HardNegativeTrainingDataset(Dataset):
    """On-the-fly hard-negative training dataset.

    Each call to __getitem__ either returns the original positive pair or a
    corrupted negative generated on the fly.  All five error types
    (laterality_swap, site_swap, severity_flip, NF_v_Dis_swap,
    cross_pair_mismatch) are shuffled uniformly; the first one that succeeds
    is used.  cross_pair_mismatch always succeeds, so it also serves as the
    guaranteed fallback if all other generators fail.

    NF_v_Dis_swap replaces the NF/disease role of flip_negation and
    flip_negation_reverse: rather than synthetically corrupting text, it swaps
    in a real conclusion from the opposite pool (no-findings ↔ disease).  Use
    no_findings_analysis/label_no_findings.py to generate improved NF pool
    labels before training.
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        corruption_prob: float = 0.5,
        nf_labels: Optional[Dict[str, bool]] = None,
        error_type_weights: Optional[Dict[str, float]] = None,
        only_error_types: Optional[List[str]] = None,
        swap_instances: str = "all",
        severity_jitter_prob: float = 0.0,
        nf_swap_nf_to_disease_frac: Optional[float] = 0.5,
        severity_flip_options: Optional[SeverityFlipOptions] = None,
    ):
        self.pairs = pairs
        self.corruption_prob = corruption_prob
        self.severity_jitter_prob = max(0.0, min(1.0, severity_jitter_prob))
        self._severity_flip_options = severity_flip_options or SeverityFlipOptions()
        self._nf_labels = nf_labels  # conclusion -> is_nf; None → regex fallback
        if swap_instances not in SWAP_INSTANCES_CHOICES:
            raise ValueError(
                f"swap_instances must be one of {SWAP_INSTANCES_CHOICES}, got {swap_instances!r}"
            )
        self.swap_instances = swap_instances

        # Pre-build NF_v_Dis_swap pools using CSV labels when available, else regex.
        conclusions = [c for _, c in pairs]
        self._nf_pool: List[str] = [c for c in conclusions if _is_nf_labeled(c, nf_labels)]
        self._disease_pool: List[str] = [c for c in conclusions if not _is_nf_labeled(c, nf_labels)]
        self._nf_pair_indices: List[int] = [
            i for i, (_, c) in enumerate(pairs) if _is_nf_labeled(c, nf_labels)
        ]
        self._disease_pair_indices: List[int] = [
            i for i, (_, c) in enumerate(pairs) if not _is_nf_labeled(c, nf_labels)
        ]
        if nf_swap_nf_to_disease_frac is not None:
            frac = float(nf_swap_nf_to_disease_frac)
            if not 0.0 <= frac <= 1.0:
                raise ValueError(
                    "nf_swap_nf_to_disease_frac must be in [0, 1] or None (disabled)"
                )
        self._nf_swap_nf_to_disease_frac = nf_swap_nf_to_disease_frac

        # All generators participate in weighted sampling without replacement.
        # cross_pair_mismatch is included in the pool (so it competes at ~1/5
        # weight by default) but is NOT added as a separate guaranteed fallback
        # at the end — that would double-count it and cause it to dominate.
        # The only remaining fallback is for the degenerate case where the
        # dataset has a single pair and cross_pair literally cannot fire.
        self._hard_generators: List[
            Tuple[str, Optional[Callable[[str], Optional[str]]]]
        ] = [
            (
                "laterality_swap",
                lambda text: apply_laterality_swap(text, swap_instances=self.swap_instances),
            ),
            (
                "site_swap",
                lambda text: apply_site_swap(text, swap_instances=self.swap_instances),
            ),
            (
                "severity_flip",
                lambda text: apply_severity_flip(
                    text,
                    swap_instances=self.swap_instances,
                    options=self._severity_flip_options,
                ),
            ),
            ("NF_v_Dis_swap", None),       # handled via _apply_NF_v_Dis_swap
            ("cross_pair_mismatch", None), # always succeeds when reached
        ]
        if only_error_types is not None:
            allowed = set(only_error_types)
            self._hard_generators = [
                g for g in self._hard_generators if g[0] in allowed
            ]
            if not self._hard_generators:
                raise ValueError(
                    f"only_error_types {only_error_types!r} matched no hard generators"
                )
        # Precompute per-generator weights (default uniform = 1.0).
        self._hard_generator_weights: List[float] = [
            float((error_type_weights or {}).get(et, 1.0))
            for et, _ in self._hard_generators
        ]

        label_source = "CSV" if nf_labels else "regex"
        weights_str = (
            str({et: w for (et, _), w in zip(self._hard_generators, self._hard_generator_weights)})
            if error_type_weights else "uniform"
        )
        print(
            f"Initialized on-the-fly hard-negative dataset: {len(self.pairs)} pairs "
            f"(corruption_prob={self.corruption_prob:.2f}, "
            f"swap_instances={self.swap_instances}, "
            f"NF labels: {label_source}, "
            f"error_type_weights: {weights_str}) "
            f"| NF pool: {len(self._nf_pool)} | disease pool: {len(self._disease_pool)}"
            + (
                f" | nf_swap_nf_to_disease_frac: {self._nf_swap_nf_to_disease_frac:.2f}"
                if self._nf_swap_nf_to_disease_frac is not None
                else " | nf_swap_direction_balance: off"
            )
            + (
                f" | severity_jitter_prob: {self.severity_jitter_prob:.2f}"
                if self.severity_jitter_prob > 0
                else ""
            )
            + (
                f" | severity_flip_options: exclude_scalar={self._severity_flip_options.exclude_scalar}"
                f" excluded_terms={sorted(self._severity_flip_options.excluded_terms)}"
                if (
                    self._severity_flip_options.exclude_scalar
                    or self._severity_flip_options.excluded_terms
                )
                else ""
            )
        )

    def _maybe_jitter_positive_conclusion(self, conclusion: str) -> str:
        if self.severity_jitter_prob <= 0 or random.random() >= self.severity_jitter_prob:
            return conclusion
        jittered = _apply_severity_jitter_impl(
            conclusion,
            swap_instances=self.swap_instances,
            options=self._severity_flip_options,
        )
        return jittered if jittered else conclusion

    def _apply_NF_v_Dis_swap(self, conclusion: str) -> Optional[str]:
        """Swap conclusion from the opposite NF/disease pool, or return None."""
        pool = self._disease_pool if _is_nf_labeled(conclusion, self._nf_labels) else self._nf_pool
        candidates = [c for c in pool if c != conclusion]
        return random.choice(candidates) if candidates else None

    def _apply_NF_v_Dis_swap_directed(
        self,
        idx: int,
        findings: str,
        conclusion: str,
        want_nf_to_disease: bool,
    ) -> Optional[Tuple[str, str]]:
        """Apply NF_v_Dis_swap, re-sourcing the base report when needed for direction."""
        is_nf = _is_nf_labeled(conclusion, self._nf_labels)
        if want_nf_to_disease:
            if not is_nf:
                if not self._nf_pair_indices:
                    return None
                swap_idx = random.choice(self._nf_pair_indices)
                findings, conclusion = self.pairs[swap_idx]
        else:
            if is_nf:
                if not self._disease_pair_indices:
                    return None
                swap_idx = random.choice(self._disease_pair_indices)
                findings, conclusion = self.pairs[swap_idx]
        swapped = self._apply_NF_v_Dis_swap(conclusion)
        if swapped is None:
            return None
        return findings, swapped

    def _sample_corruption(
        self, idx: int, findings: str, conclusion: str
    ) -> Tuple[str, Optional[Tuple[str, str]]]:
        # Weighted sampling without replacement over all hard generators.
        # cross_pair_mismatch is included in the pool at its configured weight
        # (default 1.0, same as the others → ~1/5 nominal probability).
        # Because it always succeeds, it will fire whenever it is drawn from
        # the pool — but it is NOT repeated as an explicit fallback, so it
        # cannot accumulate extra probability from the failures of other types.
        remaining = list(range(len(self._hard_generators)))
        remaining_weights = list(self._hard_generator_weights)

        while remaining:
            total_w = sum(remaining_weights[k] for k in remaining)
            r = random.random() * total_w
            cumulative = 0.0
            chosen = remaining[0]
            for k in remaining:
                cumulative += remaining_weights[k]
                if r <= cumulative:
                    chosen = k
                    break
            remaining.remove(chosen)

            error_type, fn = self._hard_generators[chosen]

            if error_type == "NF_v_Dis_swap":
                if self._nf_swap_nf_to_disease_frac is None:
                    swapped = self._apply_NF_v_Dis_swap(conclusion)
                    if swapped is not None:
                        return error_type, (findings, swapped)
                else:
                    want_nf_to_disease = (
                        random.random() < self._nf_swap_nf_to_disease_frac
                    )
                    directed = self._apply_NF_v_Dis_swap_directed(
                        idx, findings, conclusion, want_nf_to_disease
                    )
                    if directed is not None:
                        return error_type, directed
                continue

            if error_type == "cross_pair_mismatch":
                if len(self.pairs) > 1:
                    j = random.randrange(len(self.pairs))
                    while j == idx:
                        j = random.randrange(len(self.pairs))
                    return error_type, (findings, self.pairs[j][1])
                continue

            result = _apply_error_conclusion_only(findings, conclusion, fn)
            if result is not None:
                return error_type, result

        # True fallback: only reached if the dataset has a single pair and
        # cross_pair_mismatch could never fire (degenerate edge case).
        return "none", None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        findings, conclusion = self.pairs[idx]

        # On-the-fly corruption: new sampled variant each time item is fetched.
        if random.random() >= self.corruption_prob:
            return PairExample(
                findings,
                self._maybe_jitter_positive_conclusion(conclusion),
                1,
                "consistent",
            )

        error_type, transformed = self._sample_corruption(idx, findings, conclusion)
        if transformed is None:
            return PairExample(
                findings,
                self._maybe_jitter_positive_conclusion(conclusion),
                1,
                "consistent",
            )

        neg_findings, neg_conclusion = transformed
        return PairExample(neg_findings, neg_conclusion, 0, error_type)


def collate_fn(batch):
    texts = [format_joint_report(x.findings, x.conclusion) for x in batch]
    labels = torch.tensor([x.label for x in batch], dtype=torch.long)

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
        add_special_tokens=False,
    )
    encoded["labels"] = labels
    return encoded


class CsvEvalDataset(Dataset):
    def __init__(self, csv_path: str, mode: str = "all"):
        self.examples: List[PairExample] = []
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            required = {"findings", "conclusion", "label", "error_type"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

            for row in reader:
                findings = (row.get("findings") or "").strip()
                conclusion = (row.get("conclusion") or "").strip()
                label_raw = (row.get("label") or "").strip()
                error_type = normalize_error_type_name(row.get("error_type") or "unknown")
                if not findings or not conclusion:
                    continue

                if not row_included_in_eval_mode(error_type, mode):
                    continue

                label = int(label_raw)
                patient_id, source_idx = _metadata_from_eval_csv_row(row)
                self.examples.append(
                    PairExample(
                        findings,
                        conclusion,
                        label,
                        error_type,
                        patient_id=patient_id,
                        source_idx=source_idx,
                    )
                )

        print(f"Loaded {len(self.examples)} eval rows from {csv_path} (mode={mode})")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_eval_csv(batch):
    texts = [format_joint_report(x.findings, x.conclusion) for x in batch]
    labels = torch.tensor([x.label for x in batch], dtype=torch.long)
    error_types = [x.error_type for x in batch]

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
        add_special_tokens=False,
    )
    encoded["labels"] = labels
    return encoded, error_types


def binary_metrics(logits, labels):
    # --- Previous (BertForSequenceClassification, 2-logit CE) ---
    # preds = torch.argmax(logits, dim=1)
    # --- New (1-logit BCE) ---
    preds = (torch.sigmoid(logits) >= 0.5).long()
    acc = (preds == labels).float().mean().item()

    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # Treat inconsistent (label 0) as the positive class too.
    tp0 = ((preds == 0) & (labels == 0)).sum().item()
    fp0 = ((preds == 0) & (labels == 1)).sum().item()
    fn0 = ((preds == 1) & (labels == 0)).sum().item()
    tn0 = ((preds == 1) & (labels == 1)).sum().item()

    inconsistent_precision = tp0 / (tp0 + fp0 + 1e-8)
    inconsistent_recall = tp0 / (tp0 + fn0 + 1e-8)
    inconsistent_f1 = (
        2 * inconsistent_precision * inconsistent_recall
        / (inconsistent_precision + inconsistent_recall + 1e-8)
    )

    return {
        "acc": acc,
        # consistent (label 1)
        "precision": precision,
        "recall": recall,
        "f1": f1,
        # inconsistent (label 0)
        "inconsistent_precision": inconsistent_precision,
        "inconsistent_recall": inconsistent_recall,
        "inconsistent_f1": inconsistent_f1,
        # confusion matrix wrt inconsistent as positive class
        "tp0": tp0,
        "fp0": fp0,
        "fn0": fn0,
        "tn0": tn0,
    }


def normalize_error_type_for_reporting(error_type: str) -> str:
    # Merge location-specific variants into a single error family for reporting.
    for suffix in ("_in_findings", "_in_conclusion"):
        if error_type.endswith(suffix):
            return error_type[: -len(suffix)]
    return error_type


def run_eval(model, dataloader):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out = model(**batch)
            total_loss += out.loss.item()
            all_logits.append(out.logits.cpu())
            all_labels.append(batch["labels"].cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    metrics = binary_metrics(logits, labels)
    metrics["loss"] = total_loss / len(dataloader)
    metrics["auroc"] = _sklearn_roc_auc_score(
        (labels == 0).numpy(), (1.0 - torch.sigmoid(logits)).numpy()
    )
    return metrics


def run_eval_csv(
    model,
    dataloader,
    examples: Optional[List[PairExample]] = None,
    eval_csv: Optional[str] = None,
):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []
    all_error_types: List[str] = []

    with torch.no_grad():
        for batch, error_types in dataloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out = model(**batch)
            total_loss += out.loss.item()
            all_logits.append(out.logits.cpu())
            all_labels.append(batch["labels"].cpu())
            all_error_types.extend(error_types)

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    # --- New (1-logit BCE) ---
    preds = (torch.sigmoid(logits) >= 0.5).long()
    # prob_inconsistent: higher score → model is more confident the pair is an error.
    prob_inconsistent = (1.0 - torch.sigmoid(logits)).numpy()
    error_labels_binary = (labels == 0).numpy()  # 1 = error, 0 = consistent

    overall = binary_metrics(logits, labels)
    overall["loss"] = total_loss / len(dataloader)
    overall["auroc"] = _sklearn_roc_auc_score(error_labels_binary, prob_inconsistent)

    # "consistent" is the canonical name for the no-error class; "positive" is
    # the legacy name from old CSVs — treat them identically.
    non_consistent_mask = torch.tensor(
        [et not in {"consistent", "positive"} for et in all_error_types], dtype=torch.bool
    )
    if int(non_consistent_mask.sum().item()) > 0:
        err_labels = labels[non_consistent_mask]
        err_preds = preds[non_consistent_mask]
        overall["errors_only_acc"] = (err_preds == err_labels).float().mean().item()
        overall["errors_only_n"] = int(non_consistent_mask.sum().item())
    else:
        overall["errors_only_acc"] = float("nan")
        overall["errors_only_n"] = 0

    per_error_type: Dict[str, Dict[str, float]] = {}
    reporting_error_types = [
        normalize_error_type_for_reporting(et) for et in all_error_types
    ]
    reporting_error_types_np = np.array(reporting_error_types)
    consistent_mask_np = np.isin(reporting_error_types_np, ["consistent", "positive"])
    unique_types = sorted(set(reporting_error_types))

    for et in unique_types:
        mask = torch.tensor([x == et for x in reporting_error_types], dtype=torch.bool)
        et_logits = logits[mask]
        et_labels = labels[mask]
        et_preds = preds[mask]

        metrics = binary_metrics(et_logits, et_labels)
        metrics["count"] = int(mask.sum().item())
        metrics["acc"] = (et_preds == et_labels).float().mean().item()

        # Per-type AUROC: this error type vs all consistent examples.
        if et not in {"consistent", "positive"}:
            et_mask_np = reporting_error_types_np == et
            combined_np = consistent_mask_np | et_mask_np
            combined_true = error_labels_binary[combined_np]   # 1=error, 0=consistent
            combined_scores = prob_inconsistent[combined_np]
            metrics["auroc"] =_sklearn_roc_auc_score(combined_true, combined_scores)
        else:
            metrics["auroc"] = float("nan")

        per_error_type[et] = metrics

    if examples is not None and len(examples) == len(preds):
        overall["nf_swap_by_direction"] = summarize_nf_swap_by_direction(
            examples,
            preds.numpy(),
            labels.numpy(),
            task="binary",
            eval_csv=eval_csv,
        )

    return overall, per_error_type


def build_nf_v_dis_human_eval_examples(
    human_review_csv: str,
    test_file: str = TEST_FILE,
    seed: int = SEED,
) -> List[PairExample]:
    """Build NF_v_Dis_swap eval pairs using human_nf labels on the reviewed test subset."""
    return [
        PairExample(
            ex.findings,
            ex.conclusion,
            ex.label,
            ex.error_type,
            patient_id=ex.patient_id,
            source_idx=ex.source_idx,
        )
        for ex in _build_human_eval_examples(human_review_csv, test_file, seed)
    ]


class _ListEvalDataset(Dataset):
    def __init__(self, examples: List[PairExample]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def eval_nf_v_dis_human_subset(
    model,
    human_review_csv: str = NF_HUMAN_REVIEW_CSV,
    test_file: str = TEST_FILE,
) -> Optional[Dict[str, float]]:
    """NF_v_Dis_swap detection recall on human-labeled test subset."""
    examples = build_nf_v_dis_human_eval_examples(human_review_csv, test_file)
    swap_n = sum(1 for ex in examples if ex.error_type == "NF_v_Dis_swap")
    if swap_n == 0:
        return None
    dataloader = DataLoader(
        _ListEvalDataset(examples),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_eval_csv,
    )
    _, per_error_type = run_eval_csv(model, dataloader)
    nf_metrics = per_error_type.get("NF_v_Dis_swap")
    if nf_metrics is None:
        return None
    return {
        "n": int(nf_metrics["count"]),
        "inconsistent_recall": float(nf_metrics["inconsistent_recall"]),
        "auroc": float(nf_metrics.get("auroc", float("nan"))),
        "n_reviewed": sum(
            1
            for ex in examples
            if ex.error_type == "consistent"
        ),
    }


def save_eval_predictions_csv(
    model,
    dataset: CsvEvalDataset,
    output_csv_path: str,
    incidence_profile: Optional[str] = None,
):
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_eval_csv,
    )

    original_conclusion_by_idx: Dict[int, str] = {}
    for ex in dataset.examples:
        if ex.error_type in {"consistent", "positive"} and ex.source_idx is not None:
            original_conclusion_by_idx[ex.source_idx] = ex.conclusion

    rows = []
    offset = 0
    model.eval()
    with torch.no_grad():
        for batch, _ in dataloader:
            current_batch_size = batch["labels"].shape[0]
            batch_gpu = {k: v.to(DEVICE) for k, v in batch.items()}
            out = model(**batch_gpu)
            # --- Previous (2-logit CE) ---
            # probs = torch.softmax(out.logits, dim=1).cpu()
            # preds = torch.argmax(probs, dim=1).cpu()
            # --- New (1-logit BCE): probs is P(consistent) ---
            probs = torch.sigmoid(out.logits).cpu()
            preds = (probs >= 0.5).long()
            labels = batch["labels"].cpu()

            for i in range(current_batch_size):
                ex = dataset.examples[offset + i]
                pred_label = int(preds[i].item())
                true_label = int(labels[i].item())
                #--- Previous (2-logit CE softmax) ---
                #prob_inconsistent = float(probs[i, 0].item()),
                #prob_consistent = float(probs[i, 1].item()),
                #--- New (1-logit BCE sigmoid) ---
                prob_consistent = float(probs[i].item())
                prob_inconsistent = 1.0 - prob_consistent
                original_conclusion = (
                    original_conclusion_by_idx.get(ex.source_idx)
                    if ex.source_idx is not None
                    else None
                )
                rows.append(
                    {
                        "patient_id": ex.patient_id,
                        "source_idx": ex.source_idx if ex.source_idx is not None else "",
                        "findings": ex.findings,
                        "conclusion": ex.conclusion,
                        "error_type": ex.error_type,
                        "incidence": swap_incidence_for_example(
                            ex.conclusion,
                            ex.error_type,
                            original_conclusion=original_conclusion,
                            incidence_profile=incidence_profile,
                        ),
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "true_label_name": "inconsistent" if true_label == 0 else "consistent",
                        "pred_label_name": "inconsistent" if pred_label == 0 else "consistent",
                        "prob_inconsistent": prob_inconsistent,
                        "prob_consistent": prob_consistent,
                        "is_correct": int(pred_label == true_label),
                        "is_tp0": int(true_label == 0 and pred_label == 0),
                        "is_fp0": int(true_label == 1 and pred_label == 0),
                        "is_fn0": int(true_label == 0 and pred_label == 1),
                        "is_tn0": int(true_label == 1 and pred_label == 1),
                    }
                )
            offset += current_batch_size

    with open(output_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved row-level predictions to: {output_csv_path}")


def load_classifier_head(
    bert: BertModel,
    *,
    init_cls_head: bool = False,
    init_model_path: Optional[str] = None,
) -> nn.Linear:
    """Build the binary head; optionally load weights from CLS pretraining."""
    classifier_head = nn.Linear(bert.config.hidden_size, 1)
    if not init_cls_head:
        return classifier_head
    if not init_model_path:
        raise ValueError("init_cls_head requires an init checkpoint path")
    head_path = Path(init_model_path) / CLS_HEAD_FILE
    if not head_path.is_file():
        raise FileNotFoundError(
            f"--init_cls_head set but {head_path} not found. "
            "Only CLS-pretrained checkpoints (bert_cls_classification_*) include this file."
        )
    classifier_head.load_state_dict(torch.load(head_path, map_location="cpu"))
    print(f"Loaded pretrained CLS classifier head from: {head_path}")
    return classifier_head


# Mirrors bert_cls_classification.py: BertModel + nn.Linear(hidden, 1) + BCE.
# Wrapper exposes a 1-D `.logits` tensor of shape [B] (one logit per example)
# and a BCE-with-logits loss. Downstream metric / eval / inference helpers
# treat `prob_consistent = sigmoid(logits)` and pred = (prob >= 0.5).
class ConsistencyClassifier(nn.Module):
    def __init__(self, bert: BertModel, classifier_head: nn.Linear):
        super().__init__()
        self.bert = bert
        self.classifier = classifier_head

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        output_attentions=False,
        **kwargs,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
        )
        cls_repr = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_repr).squeeze(-1)  # [B]

        loss = None
        if labels is not None:
            labels = labels.float()
            loss = F.binary_cross_entropy_with_logits(logits, labels)

        return type(
            "ConsistencyOutput",
            (),
            {
                "logits": logits,
                "loss": loss,
                "attentions": getattr(outputs, "attentions", None),
            },
        )()


def train(
    train_mode: str = "hard",
    corruption_prob: float = 0.5,
    negatives_per_positive: int = 1,
    epochs: int = EPOCHS,
    nf_labels_csv: str = NF_LABELS_CSV,
    val_hard_eval_csv: Optional[str] = None,
    grad_clip: float = 1.0,
    error_type_weights: Optional[Dict[str, float]] = None,
    swap_instances: str = "all",
    severity_jitter_prob: float = 0.0,
    nf_swap_nf_to_disease_frac: Optional[float] = 0.5,
    severity_flip_options: Optional[SeverityFlipOptions] = None,
    use_wandb: bool = False,
    wandb_project: str = "bert-inconsistency",
    wandb_run_name: Optional[str] = None,
    output_model_dir: Optional[str] = None,
    log_dir: Optional[str] = None,
    init_cls_head: bool = False,
) -> str:
    if not output_model_dir:
        raise ValueError("output_model_dir is required for training")
    set_seed(SEED)
    if swap_instances not in SWAP_INSTANCES_CHOICES:
        raise ValueError(
            f"swap_instances must be in {SWAP_INSTANCES_CHOICES}, got {swap_instances!r}"
        )
    valid_hard_error_types = {
        "laterality_swap",
        "site_swap",
        "severity_flip",
        "NF_v_Dis_swap",
        "cross_pair_mismatch",
    }
    if error_type_weights is not None:
        unknown_error_types = set(error_type_weights.keys()) - valid_hard_error_types
        if unknown_error_types:
            raise ValueError(
                "Unknown keys in --error_type_weights: "
                f"{sorted(unknown_error_types)}. Valid error types: "
                f"{sorted(valid_hard_error_types)}"
            )
        for error_type, weight in error_type_weights.items():
            if float(weight) < 0.0:
                raise ValueError(
                    f"--error_type_weights[{error_type!r}] must be >= 0, got {weight}"
                )
        if all(float(weight) == 0.0 for weight in error_type_weights.values()):
            raise ValueError("At least one --error_type_weights value must be > 0")

    train_pairs = parse_findings_conclusion(TRAIN_FILE)
    val_pairs = parse_findings_conclusion(VAL_FILE)
    print(f"Train pairs: {len(train_pairs)} | Val pairs: {len(val_pairs)}")

    # Load BERT-quality NF labels for the train split (keeps pools split-safe).
    nf_labels_train = load_nf_labels_from_csv(nf_labels_csv, split="train")

    if train_mode == "original":
        train_dataset = InconsistencyDataset(
            train_pairs, negatives_per_positive=negatives_per_positive
        )
    else:
        only_error_types = None
        if train_mode == "lat":
            if error_type_weights:
                # Keep "lat" behavior anchored on laterality while allowing
                # weighted balancing across additional selected hard errors.
                only_error_types = [
                    et for et, w in error_type_weights.items() if float(w) > 0.0
                ]
                if "laterality_swap" not in only_error_types:
                    only_error_types.append("laterality_swap")
                print(
                    "train_mode=lat + error_type_weights: allowing weighted errors "
                    f"{sorted(only_error_types)} (laterality_swap always included)"
                )
            else:
                only_error_types = ["laterality_swap"]
        elif train_mode == "severity":
            if error_type_weights:
                # Keep "severity" behavior anchored on severity flips while allowing
                # weighted balancing across additional selected hard errors.
                only_error_types = [
                    et for et, w in error_type_weights.items() if float(w) > 0.0
                ]
                if "severity_flip" not in only_error_types:
                    only_error_types.append("severity_flip")
                print(
                    "train_mode=severity + error_type_weights: allowing weighted errors "
                    f"{sorted(only_error_types)} (severity_flip always included)"
                )
            else:
                only_error_types = ["severity_flip"]

        train_dataset = HardNegativeTrainingDataset(
            train_pairs,
            corruption_prob=corruption_prob,
            nf_labels=nf_labels_train,
            error_type_weights=error_type_weights,
            only_error_types=only_error_types,
            swap_instances=swap_instances,
            severity_jitter_prob=severity_jitter_prob,
            nf_swap_nf_to_disease_frac=nf_swap_nf_to_disease_frac,
            severity_flip_options=severity_flip_options,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Build the val eval loader. If a pre-built hard eval CSV exists for the val
    # split, use it (run_eval_csv) so per-epoch metrics match the hard benchmark.
    # Otherwise fall back to random val negatives (run_eval).
    val_eval_mode = "lat" if train_mode == "lat" else "severity" if train_mode == "severity" else "hard"
    if val_hard_eval_csv and Path(val_hard_eval_csv).exists():
        val_eval_dataset = CsvEvalDataset(val_hard_eval_csv, mode=val_eval_mode)
        val_loader = DataLoader(
            val_eval_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_eval_csv,
        )
        use_hard_val_eval = True
        print(
            f"Val eval: CSV '{val_hard_eval_csv}' mode={val_eval_mode} "
            f"({len(val_eval_dataset)} rows)"
        )
    else:
        if val_hard_eval_csv:
            print(
                f"Warning: val_hard_eval_csv '{val_hard_eval_csv}' not found; "
                "falling back to random val negatives."
            )
        val_dataset = InconsistencyDataset(val_pairs, negatives_per_positive=1)
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
        )
        use_hard_val_eval = False

    training_config_line = (
        f"Training mode: {train_mode} | init_model: {MODEL_PATH} "
        f"| init_cls_head: {init_cls_head} "
        f"| corruption_prob: {corruption_prob:.2f} "
        f"| negatives_per_positive: {negatives_per_positive} "
        f"| swap_instances: {swap_instances}"
    )
    print(training_config_line)

    log_path: Optional[Path] = Path(log_dir) if log_dir else None
    metrics_csv = None
    metrics_writer = None
    if log_path is not None:
        log_path.mkdir(parents=True, exist_ok=True)
        _append_training_log(
            log_path,
            f"Train pairs: {len(train_pairs)} | Val pairs: {len(val_pairs)}",
        )
        _append_training_log(log_path, training_config_line)
        if use_hard_val_eval:
            _append_training_log(
                log_path,
                f"Val eval: CSV '{val_hard_eval_csv}' mode={val_eval_mode} "
                f"({len(val_eval_dataset)} rows)",
            )
        else:
            _append_training_log(log_path, "Val eval: random mismatched negatives")
        metrics_csv = open(log_path / "training_metrics.csv", "w", newline="", encoding="utf-8")
        metrics_writer = csv.DictWriter(metrics_csv, fieldnames=TRAINING_METRICS_FIELDS)
        metrics_writer.writeheader()

    wandb_enabled = bool(use_wandb)
    if wandb_enabled:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "train_mode": train_mode,
                "train_model_path": MODEL_PATH,
                "corruption_prob": corruption_prob,
                "negatives_per_positive": negatives_per_positive,
                "batch_size": BATCH_SIZE,
                "epochs": epochs,
                "lr": LR,
                "max_length": MAX_LENGTH,
                "device": DEVICE,
                "grad_clip": grad_clip,
                "val_hard_eval_csv": val_hard_eval_csv or "random_negatives",
                "error_type_weights": error_type_weights or "uniform",
                "init_cls_head": init_cls_head,
                "severity_flip_options": {
                    "exclude_scalar": (severity_flip_options or SeverityFlipOptions()).exclude_scalar,
                    "excluded_terms": sorted(
                        (severity_flip_options or SeverityFlipOptions()).excluded_terms
                    ),
                },
            },
        )

    # --- Previous model (BertForSequenceClassification, num_labels=2, CE loss) ---
    # model = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=2)
    # #to match cls pretraining could load in classification head of previous instead of this one. would need to change a few bits later on.
    # model.resize_token_embeddings(len(tokenizer))
    # model.to(DEVICE)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    # --- New model (BertModel + Linear(hidden, 1), BCE loss; mirrors bert_cls_classification.py) ---
    # add_pooling_layer=False: we use last_hidden_state[:, 0, :] directly and don't need the
    # pooler (Linear + Tanh). Skipping it avoids random-init warnings since the MLM checkpoint
    # has no pooler weights, and drops a few unused parameters.
    if SCRATCH_INIT:
        try:
            from bert_architectures import bert_mlm_config
        except ImportError:
            from bert_architectures import bert_mlm_config  # type: ignore[no-redef]
        bert_config = bert_mlm_config(len(tokenizer), MAX_LENGTH, MODEL_SIZE)
        bert = BertModel(bert_config, add_pooling_layer=False)
    else:
        bert = BertModel.from_pretrained(MODEL_PATH, add_pooling_layer=False)
    bert.resize_token_embeddings(len(tokenizer))
    classifier_head = load_classifier_head(
        bert,
        init_cls_head=init_cls_head,
        init_model_path=None if SCRATCH_INIT else MODEL_PATH,
    )
    model = ConsistencyClassifier(bert, classifier_head).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        if use_hard_val_eval:
            metrics, _ = run_eval_csv(model, val_loader)
        else:
            metrics = run_eval(model, val_loader)
        auroc_val = metrics.get("auroc", float("nan"))
        auroc_str = f"{auroc_val:.4f}" if not math.isnan(auroc_val) else "  n/a "
        epoch_line = (
            f"Epoch {epoch + 1}/{epochs} "
            f"| train_loss={avg_train_loss:.4f} "
            f"| val_loss={metrics['loss']:.4f} "
            f"| val_auroc={auroc_str} "
            f"| val_recall={metrics['inconsistent_recall']:.4f} "
            f"| val_fp_rate={1.0 - metrics['recall']:.4f}"
        )
        print(epoch_line)
        if metrics_writer is not None:
            metrics_writer.writerow(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": metrics["loss"],
                    "val_auroc": auroc_val,
                    "val_recall_inconsistent": metrics["inconsistent_recall"],
                    "val_fp_rate": 1.0 - metrics["recall"],
                    "val_acc": metrics["acc"],
                    "val_f1_inconsistent": metrics["inconsistent_f1"],
                    "val_confusion_tp0": metrics["tp0"],
                    "val_confusion_fp0": metrics["fp0"],
                    "val_confusion_fn0": metrics["fn0"],
                    "val_confusion_tn0": metrics["tn0"],
                }
            )
            metrics_csv.flush()
            _append_training_log(log_path, epoch_line)
        if wandb_enabled:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": avg_train_loss,
                    "val/loss": metrics["loss"],
                    "val/auroc": auroc_val,
                    "val/recall_inconsistent": metrics["inconsistent_recall"],
                    "val/fp_rate": 1.0 - metrics["recall"],
                    "val/acc": metrics["acc"],
                    "val/f1_inconsistent": metrics["inconsistent_f1"],
                    "val/confusion_tp0": metrics["tp0"],
                    "val/confusion_fp0": metrics["fp0"],
                    "val/confusion_fn0": metrics["fn0"],
                    "val/confusion_tn0": metrics["tn0"],
                }
            )

    # --- Previous save (BertForSequenceClassification) ---
    # model.save_pretrained(OUTPUT_MODEL_DIR)
    # tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
    # --- New save: bert backbone + classifier head + tokenizer ---
    out_dir = Path(output_model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bert.save_pretrained(out_dir)
    torch.save(classifier_head.state_dict(), out_dir / CLS_HEAD_FILE)
    tokenizer.save_pretrained(out_dir)
    print(f"Saved inconsistency detector to: {out_dir}")
    if metrics_csv is not None:
        metrics_csv.close()
        print(f"Saved training metrics to: {log_path / 'training_metrics.csv'}")
        print(f"Saved training log to: {log_path / 'training_log.txt'}")
    if wandb_enabled:
        wandb.finish()
    return str(out_dir)


def _confusion_matrix_title(mode: str) -> str:
    titles = {
        "lat": "Confusion Matrix (Laterality Benchmark)",
        "site": "Confusion Matrix (Site Benchmark)",
        "severity": "Confusion Matrix (Severity Benchmark)",
        "hard": "Confusion Matrix (Hard Benchmark)",
        "cross_pair_full": "Confusion Matrix (Cross-Pair Benchmark)",
    }
    return titles.get(mode, "Confusion Matrix")


def _default_confusion_plot_filename(mode: str) -> str:
    names = {
        "lat": "lat_confusion_matrix.png",
        "site": "site_confusion_matrix.png",
        "severity": "severity_confusion_matrix.png",
        "hard": "hard_confusion_matrix.png",
    }
    return names.get(mode, "confusion_matrix.png")


def _default_attention_attribution_dir(predictions_csv: str, mode: str) -> str:
    plots_dir = Path(predictions_csv).resolve().parent.parent / "plots"
    out = plots_dir / f"attention_attribution_{mode}"
    out.mkdir(parents=True, exist_ok=True)
    return str(out)


def maybe_run_attention_attribution(
    predictions_csv: str,
    model_dir: str,
    mode: str,
    *,
    output_dir: Optional[str] = None,
    enabled: bool = True,
) -> None:
    """TP/TN/FP/FN attribution PNGs; row filter follows eval mode (hard → all types)."""
    if not enabled or not predictions_csv:
        return
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "visualize_bert_attention.py"
    )
    spec = importlib.util.spec_from_file_location("bert_attention_viz", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load attribution script: {script_path}")
    viz = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(viz)
    viz.run_attention_attribution(
        predictions_csv,
        model_dir,
        mode=mode,
        output_dir=output_dir or _default_attention_attribution_dir(predictions_csv, mode),
    )


def eval_from_csv(
    eval_csv: str,
    model_dir: str,
    mode: str = "all",
    save_predictions_csv: Optional[str] = None,
    confusion_plot_path: Optional[str] = None,
    run_attention_attribution: bool = True,
    attention_attribution_dir: Optional[str] = None,
    incidence_report_path: Optional[str] = None,
    incidence_recall_plot_path: Optional[str] = None,
    incidence_fp_plot_path: Optional[str] = None,
):
    # --- Previous model load (BertForSequenceClassification) ---
    # model = BertForSequenceClassification.from_pretrained(model_dir, num_labels=2)
    # if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
    #     model.resize_token_embeddings(len(tokenizer))
    # model.to(DEVICE)
    # --- New model load (BertModel + classifier head) ---
    bert = BertModel.from_pretrained(model_dir, add_pooling_layer=False)
    if bert.get_input_embeddings().weight.shape[0] != len(tokenizer):
        bert.resize_token_embeddings(len(tokenizer))
    classifier_head = nn.Linear(bert.config.hidden_size, 1)
    head_path = Path(model_dir) / CLS_HEAD_FILE
    if not head_path.exists():
        raise FileNotFoundError(f"Missing classifier head: {head_path}")
    classifier_head.load_state_dict(torch.load(head_path, map_location="cpu"))
    model = ConsistencyClassifier(bert, classifier_head).to(DEVICE)
    dataset = CsvEvalDataset(eval_csv, mode=mode)
    if len(dataset) == 0:
        raise ValueError(f"No rows available after applying mode='{mode}' to {eval_csv}")
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_eval_csv,
    )

    overall, per_error_type = run_eval_csv(
        model, dataloader, examples=dataset.examples, eval_csv=eval_csv
    )
    print(
        f"[overall] loss={overall['loss']:.4f} acc={overall['acc']:.4f} "
        f"precision_consistent={overall['precision']:.4f} recall_consistent={overall['recall']:.4f} "
        f"f1_consistent={overall['f1']:.4f} "
        f"precision_inconsistent={overall['inconsistent_precision']:.4f} "
        f"recall_inconsistent={overall['inconsistent_recall']:.4f} "
        f"f1_inconsistent={overall['inconsistent_f1']:.4f}"
    )
    print(
        f"[overall_confusion_inconsistent_positive] "
        f"tp0={int(overall['tp0'])} fp0={int(overall['fp0'])} "
        f"fn0={int(overall['fn0'])} tn0={int(overall['tn0'])}"
    )
    print(
        f"[errors_only_non_positive] n={int(overall['errors_only_n'])} "
        f"acc={overall['errors_only_acc']:.4f}"
    )
    print("\nPer error_type metrics:")
    for et, m in per_error_type.items():
        print(
            f"- {et}: n={int(m['count'])} acc={m['acc']:.4f} "
            f"precision_consistent={m['precision']:.4f} "
            f"recall_consistent={m['recall']:.4f} f1_consistent={m['f1']:.4f} "
            f"precision_inconsistent={m['inconsistent_precision']:.4f} "
            f"recall_inconsistent={m['inconsistent_recall']:.4f} "
            f"f1_inconsistent={m['inconsistent_f1']:.4f}"
        )

    # --- Clean summary table ---------------------------------------------------
    _SEP = "─" * 64
    print(f"\n{_SEP}")
    print("=== Summary (threshold=0.50) ===")
    print(f"{'Error type':<28} {'N':>6}  {'Recall':>14}  {'AUROC':>6}")
    print(_SEP)
    total_errors = 0
    for et, m in sorted(per_error_type.items()):
        if et in {"consistent", "positive"}:
            continue
        n = int(m["count"])
        total_errors += n
        det = m["inconsistent_recall"]
        auroc_val = m.get("auroc", float("nan"))
        auroc_str = f"{auroc_val:.3f}" if not math.isnan(auroc_val) else "  -  "
        print(f"  {et:<26} {n:>6}  {det:>13.1%}  {auroc_str:>6}")
    print(_SEP)
    auroc_overall = overall.get("auroc", float("nan"))
    auroc_str = f"{auroc_overall:.3f}" if not math.isnan(auroc_overall) else "  -  "
    print(f"  {'Any error':<26} {total_errors:>6}  {overall['inconsistent_recall']:>13.1%}  {auroc_str:>6}")
    print(_SEP)
    if "consistent" in per_error_type:
        n_cons = int(per_error_type["consistent"]["count"])
        fp_rate = 1.0 - per_error_type["consistent"]["recall"]
        print(f"  {'Consistent (FP rate)':<26} {n_cons:>6}  {fp_rate:>13.1%}  {'  -  ':>6}")
        print(_SEP)

    if overall.get("nf_swap_by_direction"):
        print_nf_swap_direction_table(overall["nf_swap_by_direction"], task="binary")

    if save_predictions_csv:
        save_eval_predictions_csv(
            model=model,
            dataset=dataset,
            output_csv_path=save_predictions_csv,
            incidence_profile=mode if mode in INCIDENCE_EVAL_MODES else None,
        )
    if confusion_plot_path:
        plot_confusion_matrix_inconsistent_positive(
            metrics=overall,
            output_path=confusion_plot_path,
            title=_confusion_matrix_title(mode),
        )
    if run_attention_attribution and save_predictions_csv:
        maybe_run_attention_attribution(
            save_predictions_csv,
            model_dir,
            mode,
            output_dir=attention_attribution_dir,
            enabled=True,
        )
    if save_predictions_csv:
        run_name = Path(save_predictions_csv).resolve().parent.parent.name
        write_eval_artifacts(
            predictions_csv=save_predictions_csv,
            run_name=run_name,
            eval_mode=mode,
            eval_csv=eval_csv,
            source="bert",
            confusion_plot_path=confusion_plot_path,
            write_confusion=bool(confusion_plot_path),
        )
    return overall, per_error_type


def plot_confusion_matrix_inconsistent_positive(
    metrics: Dict[str, float],
    output_path: str,
    title: str = "Confusion Matrix (Cross-Pair Benchmark)",
):
    # Rows: actual [inconsistent(0), consistent(1)]
    # Cols: predicted [inconsistent(0), consistent(1)]
    matrix = [
        [int(metrics["tp0"]), int(metrics["fn0"])],
        [int(metrics["fp0"]), int(metrics["tn0"])],
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

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved confusion matrix plot to: {output_path}")


def eval_benchmarks(
    model_dir: str,
    eval_csv_hard: Optional[str] = None,
    eval_csv_cross_pair: Optional[str] = None,
    hard_eval_mode: str = "hard",
    hard_confusion_path: Optional[str] = None,
    cross_pair_confusion_path: Optional[str] = None,
    hard_predictions_csv: Optional[str] = None,
    cross_pair_predictions_csv: Optional[str] = None,
    hard_incidence_report_path: Optional[str] = None,
    run_attention_attribution: bool = True,
    hard_attention_attribution_dir: Optional[str] = None,
    cross_pair_attention_attribution_dir: Optional[str] = None,
):
    ran_any = False

    if eval_csv_hard:
        ran_any = True
        print(f"\n=== Hard Mismatch Benchmark (mode={hard_eval_mode}) ===")
        eval_from_csv(
            eval_csv_hard,
            model_dir,
            mode=hard_eval_mode,
            save_predictions_csv=hard_predictions_csv,
            confusion_plot_path=hard_confusion_path,
            run_attention_attribution=run_attention_attribution,
            attention_attribution_dir=hard_attention_attribution_dir,
            incidence_report_path=hard_incidence_report_path,
        )

    if eval_csv_cross_pair:
        ran_any = True
        print("\n=== Cross-Pair Benchmark ===")
        cross_overall, _ = eval_from_csv(
            eval_csv_cross_pair,
            model_dir,
            mode="cross_pair_full",
            save_predictions_csv=cross_pair_predictions_csv,
            run_attention_attribution=run_attention_attribution,
            attention_attribution_dir=cross_pair_attention_attribution_dir,
        )
        if cross_pair_confusion_path:
            plot_confusion_matrix_inconsistent_positive(
                metrics=cross_overall,
                output_path=cross_pair_confusion_path,
            )

    if not ran_any:
        raise ValueError("No benchmark CSV provided. Set --eval_csv_hard and/or --eval_csv_cross_pair.")


def load_consistency_classifier(model_dir: str) -> ConsistencyClassifier:
    """Load BertModel + cls head for inference / attention visualization."""
    bert = BertModel.from_pretrained(
        model_dir, add_pooling_layer=False, attn_implementation="eager"
    )
    if bert.get_input_embeddings().weight.shape[0] != len(tokenizer):
        bert.resize_token_embeddings(len(tokenizer))
    classifier_head = nn.Linear(bert.config.hidden_size, 1)
    head_path = Path(model_dir) / CLS_HEAD_FILE
    if not head_path.exists():
        raise FileNotFoundError(f"Missing classifier head: {head_path}")
    classifier_head.load_state_dict(torch.load(head_path, map_location="cpu"))
    model = ConsistencyClassifier(bert, classifier_head).to(DEVICE)
    model.eval()
    return model


def section_char_ranges(findings: str, conclusion: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Character spans of findings / conclusion inside format_joint_report text."""
    prefix = f"{tokenizer.cls_token} {FINDINGS_SPECIAL} "
    mid = f" {CONCLUSION_SPECIAL} "
    f_start = len(prefix)
    f_end = f_start + len(findings)
    c_start = f_end + len(mid)
    c_end = c_start + len(conclusion)
    return (f_start, f_end), (c_start, c_end)


def encode_joint_report(findings: str, conclusion: str) -> Dict[str, object]:
    """Tokenise a joint report with offset mappings aligned to the source string."""
    text = format_joint_report(findings, conclusion)
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    return {
        "text": text,
        "input_ids": encoded["input_ids"].to(DEVICE),
        "attention_mask": encoded["attention_mask"].to(DEVICE),
        "offsets": encoded["offset_mapping"][0].tolist(),
        "findings_span": section_char_ranges(findings, conclusion)[0],
        "conclusion_span": section_char_ranges(findings, conclusion)[1],
    }


def _cls_token_scores(model: ConsistencyClassifier, encoded: Dict[str, object]) -> List[float]:
    with torch.no_grad():
        out = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            output_attentions=True,
        )
    attn = out.attentions[-1].mean(dim=1)[0, 0]
    return [float(s) for s in attn.cpu()]


def _integrated_gradients_token_scores(
    model: ConsistencyClassifier,
    encoded: Dict[str, object],
    pred_label: int,
    steps: int = 50,
) -> List[float]:
    """IG attributions summed over embedding dims (positive class = predicted label)."""
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    emb_layer = model.bert.embeddings
    input_embeds = emb_layer(input_ids=input_ids)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0
    baseline_ids = torch.full_like(input_ids, pad_id)
    baseline_embeds = emb_layer(input_ids=baseline_ids)
    diff = input_embeds - baseline_embeds

    integrated_grad = torch.zeros_like(input_embeds)
    for step in range(1, steps + 1):
        alpha = float(step) / float(steps)
        scaled = baseline_embeds + alpha * diff
        scaled = scaled.detach().requires_grad_(True)
        outputs = model.bert(
            inputs_embeds=scaled,
            attention_mask=attention_mask,
        )
        cls_repr = outputs.last_hidden_state[:, 0, :]
        logits = model.classifier(cls_repr).squeeze(-1)
        target = logits if pred_label == 1 else -logits
        model.zero_grad(set_to_none=True)
        target.sum().backward()
        integrated_grad += scaled.grad.detach()

    attr = (diff * integrated_grad).sum(dim=-1).squeeze(0)
    mask = attention_mask.squeeze(0).float()
    attr = attr.abs() * mask
    return [float(s) for s in attr.detach().cpu()]


def explain_consistency(
    model: ConsistencyClassifier,
    findings: str,
    conclusion: str,
    attribution: str = "cls",
    ig_steps: int = 50,
    ig_target_label: Optional[int] = None,
) -> Dict[str, object]:
    """Run forward pass and return prediction + per-token attribution scores.

    attribution:
      cls — final-layer CLS→token attention (mean heads)
      ig  — Integrated Gradients w.r.t. the predicted class logit

    ig_target_label: if set (0=inconsistent, 1=consistent), IG is computed toward
    that class instead of the model prediction. Useful for aggregate summaries.
    """
    if attribution not in {"cls", "ig"}:
        raise ValueError(f"attribution must be 'cls' or 'ig', got {attribution!r}")

    encoded = encode_joint_report(findings, conclusion)
    with torch.no_grad():
        out = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )
    consistency_prob = torch.sigmoid(out.logits).item()
    inconsistency_prob = 1.0 - consistency_prob
    pred = int(consistency_prob >= 0.5)

    if attribution == "cls":
        scores = _cls_token_scores(model, encoded)
    else:
        ig_label = pred if ig_target_label is None else int(ig_target_label)
        scores = _integrated_gradients_token_scores(
            model, encoded, pred_label=ig_label, steps=ig_steps
        )

    input_ids = encoded["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    token_scores = [(t, float(s)) for t, s in zip(tokens, scores)]

    return {
        "prediction": "consistent" if pred == 1 else "inconsistent",
        "inconsistency_prob": inconsistency_prob,
        "consistency_prob": consistency_prob,
        "attribution": attribution,
        "token_scores": token_scores,
        "offsets": encoded["offsets"],
        "findings_span": encoded["findings_span"],
        "conclusion_span": encoded["conclusion_span"],
    }


def predict_inconsistency(findings: str, conclusion: str, model_dir: str | None = None):
    model_dir = model_dir or default_model_dir()
    model = load_consistency_classifier(model_dir)
    return explain_consistency(model, findings, conclusion)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_csv",
        type=str,
        default=None,
        help="If set, skip training and evaluate this CSV with columns findings, conclusion, label, error_type.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Model path for --eval_csv mode (default: latest bert-inconsistency run).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "hard", "lat", "site", "severity", "cross_pair_full"],
        default="all",
        help=(
            "Eval mode for --eval_csv: "
            "all=use all rows, "
            "hard=full hard-negative v4 suite (all error types in the CSV, "
            "including cross_pair_mismatch), "
            "lat=only consistent + laterality_swap rows, "
            "severity=only consistent + severity_flip rows, "
            "cross_pair_full=only consistent + cross_pair_mismatch rows."
        ),
    )
    parser.add_argument(
        "--eval_csv_hard",
        type=str,
        default=None,
        help="Optional hard-mismatch eval CSV. If provided, runs hard benchmark eval.",
    )
    parser.add_argument(
        "--eval_csv_cross_pair",
        type=str,
        default=None,
        help="Optional cross-pair eval CSV. If provided, runs cross-pair benchmark eval.",
    )
    parser.add_argument(
        "--cross_pair_confusion_path",
        type=str,
        default=None,
        help="Output path for cross-pair confusion matrix plot in dual benchmark mode.",
    )
    parser.add_argument(
        "--confusion_plot_path",
        type=str,
        default=None,
        help=(
            "Output path for confusion matrix plot in single --eval_csv mode. "
            "When --mode lat/severity (or --train_mode lat/severity with default mode), "
            "defaults to plots/lat_confusion_matrix.png or "
            "plots/severity_confusion_matrix.png under the run directory."
        ),
    )
    parser.add_argument(
        "--output_model_dir",
        type=str,
        default=None,
        help="Override model output directory (default: new timestamped run under trained_models/runs/bert_binary/).",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional run label; saved as YYYYMMDD_<name> under trained_models/runs/bert_binary/ (default: full timestamp + script id + tags).",
    )
    parser.add_argument(
        "--save_predictions_csv",
        type=str,
        default=None,
        help="Optional row-level predictions CSV path for single --eval_csv mode.",
    )
    parser.add_argument(
        "--hard_predictions_csv",
        type=str,
        default=None,
        help="Optional row-level predictions CSV path for hard benchmark in dual mode.",
    )
    parser.add_argument(
        "--cross_pair_predictions_csv",
        type=str,
        default=None,
        help="Optional row-level predictions CSV path for cross-pair benchmark in dual mode.",
    )
    parser.add_argument(
        "--train_mode",
        type=str,
        choices=["original", "hard", "lat", "severity"],
        default="hard",
        help=(
            "Training mode when not running eval: "
            "original=random cross-pair negatives, "
            "hard=corruption with harder error types, "
            "lat=hard corruption with laterality-focused negatives "
            "(only laterality by default; when --error_type_weights is provided, "
            "positive-weighted error types are also enabled), "
            "severity=hard corruption with severity-focused negatives "
            "(only severity by default; when --error_type_weights is provided, "
            "positive-weighted error types are also enabled)."
        ),
    )
    parser.add_argument(
        "--corruption_prob",
        type=float,
        default=0.5,
        help="Hard/lat/severity mode: probability of corrupting each training sample.",
    )
    parser.add_argument(
        "--negatives_per_positive",
        type=int,
        default=1,
        help="Original mode only: number of random mismatched negatives per positive pair.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Number of training epochs (default: {EPOCHS}).",
    )
    parser.add_argument(
        "--swap_instances",
        type=str,
        choices=SWAP_INSTANCES_CHOICES,
        default="all",
        help=(
            "Hard/lat/severity mode keyword swap scope for laterality/site/severity errors: "
            "all=swap every match; single=swap exactly one trigger word (random position); "
            "random=same as single; random_n=swap a random count in [1..N]."
        ),
    )
    parser.add_argument(
        "--severity_jitter_prob",
        type=float,
        default=None,
        help=(
            "On uncorrupted (consistent) training rows, probability of within-tier severity "
            "synonym jitter in the conclusion. Default: 0.2 for train_mode=severity, else 0."
        ),
    )
    parser.add_argument(
        "--severity_exclude_scalar",
        action="store_true",
        help=(
            "Hard training only: never apply scalar severity flips (small/large, "
            "low-grade/high-grade) when sampling severity_flip negatives or jitter. "
            "Eval CSV generation is unchanged."
        ),
    )
    parser.add_argument(
        "--severity_exclude_terms",
        type=str,
        default=None,
        help=(
            "Hard training only: comma-separated qualitative severity trigger words to "
            "withhold from severity_flip and jitter (e.g. mild,mildly,severe,severely). "
            "Remaining tier terms stay active."
        ),
    )

    parser.add_argument(
        "--nf_labels_csv",
        type=str,
        default=NF_LABELS_CSV,
        help=(
            "Path to no_findings_analysis/no_findings_labels.csv produced by "
            "no_findings_analysis/label_no_findings.py. "
            "Used to classify conclusions into NF / disease pools for NF_v_Dis_swap. "
            f"Default: {NF_LABELS_CSV}"
        ),
    )
    parser.add_argument(
        "--nf_swap_nf_to_disease_frac",
        type=float,
        default=0.5,
        help=(
            "Hard training only: when NF_v_Dis_swap is sampled, target this fraction "
            "of NF→disease swaps (re-sources NF-original reports when needed). "
            "Set to -1 to disable direction balancing (legacy behavior)."
        ),
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="If set, skip training and run only evaluation.",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Log per-epoch training/eval metrics to Weights & Biases.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="bert-inconsistency",
        help="wandb project name (used when --use_wandb is set).",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Optional wandb run name.",
    )
    parser.add_argument(
        "--val_hard_eval_csv",
        type=str,
        default=None,
        help=(
            "Path to a pre-built hard eval CSV for the val split. "
            "When provided (and the file exists), per-epoch validation uses "
            "run_eval_csv on this CSV instead of random val negatives, so the "
            "live training metrics match the hard benchmark."
        ),
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Gradient clipping max norm (0.0 = disabled). Default: 1.0.",
    )
    parser.add_argument(
        "--error_type_weights",
        type=str,
        default=None,
        help=(
            "Optional JSON dict of relative sampling weights for hard generators. "
            "Example: '{\"laterality_swap\": 3.0, \"site_swap\": 2.0}'. "
            "Unspecified types default to weight 1.0. "
            "cross_pair_mismatch is always the fallback and cannot be weighted."
        ),
    )
    parser.add_argument(
        "--skip_attention_attribution",
        action="store_true",
        help=(
            "Skip TP/TN/FP/FN attribution PNGs under plots/attention_attribution_<mode>/ "
            "(default: run after saving predictions)."
        ),
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        choices=SUPPORTED_SEQ_LENGTHS,
        default=DEFAULT_SEQ_LENGTH,
        help=(
            "Max token sequence length for training/eval "
            f"(default: {DEFAULT_SEQ_LENGTH}). "
            "Init checkpoint is bert_cls_classification_mxln{seq_length}[_big|_8L|_12L] "
            "unless --wo_cls_init is set."
        ),
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="small",
        help="One of small/big/8L/12L (or 4/6/8/12). Used to resolve init checkpoint.",
    )
    parser.add_argument(
        "--wo_cls_init",
        action="store_true",
        help=(
            "Init from bert-pretrained_mxlen_{seq_length} (MLM pretrain only) instead "
            "of bert_cls_classification_mxln{seq_length}. Ignored when --init_checkpoint "
            "is set."
        ),
    )
    parser.add_argument(
        "--init_checkpoint",
        type=str,
        default=None,
        help=(
            "Optional init checkpoint name (e.g. bert_cls_classification_mxln600_8L) "
            "or path to a model directory with config.json + weights. Overrides the "
            "default from --seq_length / --model_size / --wo_cls_init."
        ),
    )
    parser.add_argument(
        "--scratch_init",
        action="store_true",
        help=(
            "Random-init BERT backbone (no MLM/CLS checkpoint). "
            "Mutually exclusive with --wo_cls_init and --init_checkpoint."
        ),
    )
    parser.add_argument(
        "--init_cls_head",
        action="store_true",
        help=(
            "Load cls_classifier.pt from the init checkpoint directory instead of "
            "random-init Linear(hidden, 1). Requires a CLS-pretrained init "
            "(default bert_cls_classification_* or --init_checkpoint with head file). "
            "Incompatible with --scratch_init and --wo_cls_init."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Training/eval batch size (default: 128 for small/big/8L, 64 for 12L).",
    )
    args = parser.parse_args()
    try:
        from bert_architectures import default_cls_batch_size
    except ImportError:
        from bert_architectures import default_cls_batch_size  # type: ignore[no-redef]
    train_batch_size = args.batch_size or default_cls_batch_size(args.model_size)
    if args.scratch_init and (args.wo_cls_init or args.init_checkpoint):
        parser.error("--scratch_init cannot be combined with --wo_cls_init or --init_checkpoint")
    if args.init_cls_head and (args.scratch_init or args.wo_cls_init):
        parser.error(
            "--init_cls_head requires a CLS checkpoint; incompatible with "
            "--scratch_init and --wo_cls_init"
        )
    configure_seq_length(
        args.seq_length,
        wo_cls_init=args.wo_cls_init,
        scratch_init=args.scratch_init,
        init_checkpoint=args.init_checkpoint,
        model_size=args.model_size,
        batch_size=train_batch_size,
    )
    run_attention_attribution = not args.skip_attention_attribution

    def _run_tags() -> tuple[str, ...]:
        return (args.train_mode,)

    run_hard_eval = bool(args.eval_csv_hard)
    run_cross_eval = bool(args.eval_csv_cross_pair)
    run_benchmark_eval = run_hard_eval or run_cross_eval
    run_single_eval = bool(args.eval_csv)
    hard_eval_mode = args.mode
    if hard_eval_mode == "cross_pair_full":
        print(
            "Warning: --mode cross_pair_full is not valid for --eval_csv_hard; "
            "using --mode hard for hard benchmark eval."
        )
        hard_eval_mode = "hard"
    if args.train_mode == "lat" and hard_eval_mode == "all":
        hard_eval_mode = "lat"
    if args.train_mode == "severity" and hard_eval_mode == "all":
        hard_eval_mode = "severity"
    init_tokenizer(TOKENIZER_PATH)
    if args.eval_only:
        args.model_dir = args.model_dir or default_model_dir()

    if args.eval_only and run_benchmark_eval:
        eval_layout = setup_run(
            SCRIPT_ID,
            tags=("eval", *_run_tags()),
            run_name=args.run_name,
            run_category=RUN_CATEGORY,
            argv=current_argv(),
        )
        with tee_run_output(eval_layout.log_path(RUN_OUTPUT_LOG)):
            print("Evaluating benchmark CSV(s) on model: ", args.model_dir)
            print(f"Run outputs: {eval_layout.root}")
            hard_predictions_csv = default_result_path(
                eval_layout, "hard_predictions.csv", args.hard_predictions_csv
            )
            eval_benchmarks(
                model_dir=args.model_dir,
                eval_csv_hard=args.eval_csv_hard,
                eval_csv_cross_pair=args.eval_csv_cross_pair,
                hard_eval_mode=hard_eval_mode,
                hard_confusion_path=default_plot_path(
                    eval_layout,
                    _default_confusion_plot_filename(hard_eval_mode),
                    args.confusion_plot_path,
                ),
                cross_pair_confusion_path=default_plot_path(
                    eval_layout, "cross_pair_confusion_matrix.png", args.cross_pair_confusion_path
                ),
                hard_predictions_csv=hard_predictions_csv,
                cross_pair_predictions_csv=default_result_path(
                    eval_layout, "cross_pair_predictions.csv", args.cross_pair_predictions_csv
                ),
                hard_incidence_report_path=default_result_path(
                    eval_layout,
                    "incidence_detection_report.txt",
                    None,
                )
                if hard_eval_mode in INCIDENCE_EVAL_MODES
                else None,
                run_attention_attribution=run_attention_attribution,
            )
    elif args.eval_only and run_single_eval:
        eval_layout = setup_run(
            SCRIPT_ID,
            tags=("eval", *_run_tags()),
            run_name=args.run_name,
            run_category=RUN_CATEGORY,
            argv=current_argv(),
        )
        with tee_run_output(eval_layout.log_path(RUN_OUTPUT_LOG)):
            print("Evaluating single benchmark on model: ", args.model_dir)
            print(f"Run outputs: {eval_layout.root}")
            eval_mode = args.mode
            if args.train_mode == "lat" and eval_mode == "all":
                eval_mode = "lat"
            if args.train_mode == "severity" and eval_mode == "all":
                eval_mode = "severity"
            confusion_plot_path = None
            if eval_mode in INCIDENCE_EVAL_MODES or args.confusion_plot_path:
                confusion_plot_path = default_plot_path(
                    eval_layout,
                    _default_confusion_plot_filename(eval_mode),
                    args.confusion_plot_path,
                )
            predictions_csv_path = default_result_path(
                eval_layout, "predictions.csv", args.save_predictions_csv
            )
            eval_from_csv(
                args.eval_csv,
                args.model_dir,
                eval_mode,
                save_predictions_csv=predictions_csv_path,
                confusion_plot_path=confusion_plot_path,
                run_attention_attribution=run_attention_attribution,
                incidence_report_path=default_result_path(
                    eval_layout, "incidence_detection_report.txt", None
                )
                if eval_mode in INCIDENCE_EVAL_MODES
                else None,
                incidence_recall_plot_path=default_plot_path(
                    eval_layout, f"{eval_mode}_recall_by_incidence.png", None
                )
                if eval_mode in INCIDENCE_EVAL_MODES
                else None,
                incidence_fp_plot_path=default_plot_path(
                    eval_layout, f"{eval_mode}_fp_rate_by_incidence.png", None
                )
                if eval_mode in INCIDENCE_EVAL_MODES
                else None,
            )
    else:
        import json as _json
        error_type_weights = (
            _json.loads(args.error_type_weights) if args.error_type_weights else None
        )
        train_layout = setup_run(
            SCRIPT_ID,
            tags=_run_tags(),
            run_name=args.run_name,
            run_category=RUN_CATEGORY,
            output_model_dir=args.output_model_dir,
            argv=current_argv(),
        )
        with tee_run_output(train_layout.log_path(RUN_OUTPUT_LOG)):
            print(f"Run directory: {train_layout.root}")
            jitter_prob = (
                args.severity_jitter_prob
                if args.severity_jitter_prob is not None
                else (0.2 if args.train_mode == "severity" else 0.0)
            )
            nf_swap_frac = (
                None
                if args.nf_swap_nf_to_disease_frac < 0
                else args.nf_swap_nf_to_disease_frac
            )
            severity_flip_options = build_severity_flip_options(
                exclude_scalar=args.severity_exclude_scalar,
                excluded_terms=args.severity_exclude_terms,
            )
            saved_model_dir = train(
                train_mode=args.train_mode,
                corruption_prob=args.corruption_prob,
                negatives_per_positive=args.negatives_per_positive,
                epochs=args.epochs,
                swap_instances=args.swap_instances,
                severity_jitter_prob=jitter_prob,
                nf_labels_csv=args.nf_labels_csv,
                nf_swap_nf_to_disease_frac=nf_swap_frac,
                val_hard_eval_csv=args.val_hard_eval_csv,
                grad_clip=args.grad_clip,
                error_type_weights=error_type_weights,
                severity_flip_options=severity_flip_options,
                use_wandb=args.use_wandb,
                wandb_project=args.wandb_project,
                wandb_run_name=args.wandb_run_name or train_layout.root.name,
                output_model_dir=str(train_layout.model),
                log_dir=str(train_layout.logs),
                init_cls_head=args.init_cls_head,
            )

            # Optional post-training evaluation(s) in the same run.
            if run_benchmark_eval:
                hard_predictions_csv = default_result_path(
                    train_layout, "hard_predictions.csv", args.hard_predictions_csv
                )
                eval_benchmarks(
                    model_dir=saved_model_dir,
                    eval_csv_hard=args.eval_csv_hard,
                    eval_csv_cross_pair=args.eval_csv_cross_pair,
                    hard_eval_mode=hard_eval_mode,
                    hard_confusion_path=default_plot_path(
                        train_layout,
                        _default_confusion_plot_filename(hard_eval_mode),
                        args.confusion_plot_path,
                    ),
                    cross_pair_confusion_path=default_plot_path(
                        train_layout,
                        "cross_pair_confusion_matrix.png",
                        args.cross_pair_confusion_path,
                    ),
                    hard_predictions_csv=hard_predictions_csv,
                    cross_pair_predictions_csv=default_result_path(
                        train_layout,
                        "cross_pair_predictions.csv",
                        args.cross_pair_predictions_csv,
                    ),
                    hard_incidence_report_path=default_result_path(
                        train_layout,
                        "incidence_detection_report.txt",
                        None,
                    )
                    if hard_eval_mode in INCIDENCE_EVAL_MODES
                    else None,
                    run_attention_attribution=run_attention_attribution,
                )
            elif run_single_eval:
                eval_mode = args.mode
                if args.train_mode == "lat" and eval_mode == "all":
                    eval_mode = "lat"
                if args.train_mode == "severity" and eval_mode == "all":
                    eval_mode = "severity"
                confusion_plot_path = None
                if eval_mode in INCIDENCE_EVAL_MODES or args.confusion_plot_path:
                    confusion_plot_path = default_plot_path(
                        train_layout,
                        _default_confusion_plot_filename(eval_mode),
                        args.confusion_plot_path,
                    )
                predictions_csv_path = default_result_path(
                    train_layout, "predictions.csv", args.save_predictions_csv
                )
                eval_from_csv(
                    args.eval_csv,
                    saved_model_dir,
                    eval_mode,
                    save_predictions_csv=predictions_csv_path,
                    confusion_plot_path=confusion_plot_path,
                    run_attention_attribution=run_attention_attribution,
                    incidence_report_path=default_result_path(
                        train_layout, "incidence_detection_report.txt", None
                    )
                    if eval_mode in INCIDENCE_EVAL_MODES
                    else None,
                    incidence_recall_plot_path=default_plot_path(
                        train_layout, f"{eval_mode}_recall_by_incidence.png", None
                    )
                    if eval_mode in INCIDENCE_EVAL_MODES
                    else None,
                    incidence_fp_plot_path=default_plot_path(
                        train_layout, f"{eval_mode}_fp_rate_by_incidence.png", None
                    )
                    if eval_mode in INCIDENCE_EVAL_MODES
                    else None,
                )

