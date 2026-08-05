"""Zero-shot LLM baseline for findings/conclusion inconsistency eval.

Designed for **local-only** inference: point --base_url at a server on this machine
(vLLM, llama.cpp, Ollama, etc.). By default, remote hosts are rejected so report
text is not sent to cloud APIs unless you pass --allow_remote.
"""
import argparse
import csv
import json
import math
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib import error, parse, request

try:
    from paths import (
        RUN_OUTPUT_LOG,
        current_argv,
        default_result_path,
        setup_run,
        tee_run_output,
    )
    from src.llm_eval_prompts import (
        CONFIDENCE_DEFAULT_MAX_TOKENS,
        CONFIDENCE_SCORE_MAX,
        CONFIDENCE_SCORE_MIN,
        MULTICLASS_DEFAULT_MAX_TOKENS,
        MULTICLASS_LABEL_NAMES,
        MULTICLASS_N_CLASSES,
        SYSTEM_PROMPT_TEMPLATES,
        USER_PROMPT_TEMPLATES,
        is_confidence_prompt,
        is_multiclass_prompt,
        label_from_error_type,
        prompt_choices_help,
    )
except ImportError:
    import sys

    _repo_root = Path(__file__).resolve().parents[1]
    if str(_repo_root) not in sys.path:
        sys.path.append(str(_repo_root))
    from paths import (
        RUN_OUTPUT_LOG,
        current_argv,
        default_result_path,
        setup_run,
        tee_run_output,
    )
    from src.llm_eval_prompts import (
        CONFIDENCE_DEFAULT_MAX_TOKENS,
        CONFIDENCE_SCORE_MAX,
        CONFIDENCE_SCORE_MIN,
        MULTICLASS_DEFAULT_MAX_TOKENS,
        MULTICLASS_LABEL_NAMES,
        MULTICLASS_N_CLASSES,
        SYSTEM_PROMPT_TEMPLATES,
        USER_PROMPT_TEMPLATES,
        is_confidence_prompt,
        is_multiclass_prompt,
        label_from_error_type,
        prompt_choices_help,
    )

SCRIPT_ID = "llm-inconsistency-eval"
RUN_CATEGORY = "llm_baseline"
_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_QWEN3_MODEL_RE = re.compile(r"qwen[_-]?3", re.IGNORECASE)
_THINK_BLOCK_RE = re.compile(
    r"<think(?:ing)?>[\s\S]*?</think(?:ing)?>",
    re.IGNORECASE,
)
# vLLM ChatMessage uses "reasoning"; some clients use "reasoning_content".
_MESSAGE_TEXT_KEYS = ("content", "reasoning", "reasoning_content")
_CONSISTENT_JSON_RE = re.compile(
    r'\{\s*"consistent"\s*:\s*(true|false)\s*\}',
    re.IGNORECASE,
)
_CONSISTENT_KV_RE = re.compile(
    r'"consistent"\s*:\s*(true|false)',
    re.IGNORECASE,
)
_ERROR_JSON_RE = re.compile(
    r'\{\s*"error"\s*:\s*(true|false)\s*\}',
    re.IGNORECASE,
)
_ERROR_KV_RE = re.compile(
    r'"error"\s*:\s*(true|false)',
    re.IGNORECASE,
)
_ERROR_TYPE_JSON_RE = re.compile(
    r'\{\s*"error_type"\s*:\s*"([^"]+)"\s*\}',
    re.IGNORECASE,
)
_ERROR_TYPE_KV_RE = re.compile(
    r'"error_type"\s*:\s*"([^"]+)"',
    re.IGNORECASE,
)
DEFAULT_MAX_TOKENS = 32
DEFAULT_ABORT_AFTER_PARSE_FAILURES = 50
DEFAULT_CHECKPOINT_EVERY = 100
TOKEN_LOGPROBS_TOP_K = 5
_BOOL_DECISION_KEY_MARKERS = ('"consistent":', '"error":')
PREDICTION_CSV_FIELDNAMES = [
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
    "parse_ok",
    "raw_response",
    "model_name",
]
CONFIDENCE_CSV_FIELD = "confidence"


def prediction_csv_fieldnames(*, expects_confidence: bool = False) -> List[str]:
    if not expects_confidence:
        return list(PREDICTION_CSV_FIELDNAMES)
    fields = list(PREDICTION_CSV_FIELDNAMES)
    insert_at = fields.index("prob_consistent") + 1
    fields.insert(insert_at, CONFIDENCE_CSV_FIELD)
    return fields


class ParseFailureAbort(RuntimeError):
    """Raised when the first N rows all fail JSON parsing (model/config issue)."""


@dataclass
class EvalRow:
    patient_id: str
    source_idx: Optional[int]
    findings: str
    conclusion: str
    error_type: str
    incidence: str
    true_label: int


def _is_local_base_url(base_url: str) -> bool:
    """True if base_url targets loopback (privacy-safe local inference)."""
    parsed = parse.urlparse(base_url)
    if parsed.scheme not in {"http", "https"}:
        return False
    host = (parsed.hostname or "").lower()
    return host in _LOCAL_HOSTS


def _assert_local_inference(base_url: str, allow_remote: bool) -> None:
    if _is_local_base_url(base_url):
        return
    if allow_remote:
        print(
            "WARNING: --allow_remote set; report text will be sent to "
            f"{base_url!r}. Use only if your data governance permits it."
        )
        return
    raise ValueError(
        f"Refusing non-local base_url {base_url!r}. "
        "For privacy, use a local OpenAI-compatible server (e.g. vLLM on "
        "http://127.0.0.1:8000/v1). To override, pass --allow_remote."
    )


def _assert_api_reachable(base_url: str, timeout_s: float = 5.0) -> None:
    """Fail fast when the local inference server is not running."""
    url = base_url.rstrip("/") + "/models"
    try:
        with request.urlopen(url, timeout=timeout_s) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Unexpected HTTP {resp.status} from {url}")
    except error.URLError as e:
        raise RuntimeError(
            f"Cannot reach inference API at {url} ({e}). "
            "Start your local server first (e.g. vLLM on port 8000), then rerun eval."
        ) from e


def _normalize_error_type(error_type: str) -> str:
    if error_type == "positive":
        return "consistent"
    return error_type


def _parse_error_types_filter(raw: Optional[str]) -> Optional[frozenset[str]]:
    if raw is None or not str(raw).strip():
        return None
    types = frozenset(
        _normalize_error_type(part.strip())
        for part in str(raw).split(",")
        if part.strip()
    )
    if not types:
        raise ValueError("--error_types must list at least one error_type.")
    return types


def _load_eval_rows(
    csv_path: str,
    mode: str,
    *,
    error_types: Optional[frozenset[str]] = None,
) -> List[EvalRow]:
    rows: List[EvalRow] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        required = {"findings", "conclusion", "error_type"}
        missing = required - fields
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        has_true_label = "true_label" in fields
        has_label = "label" in fields
        if not has_true_label and not has_label:
            raise ValueError("CSV must have either 'label' or 'true_label' column.")

        for row in reader:
            findings = (row.get("findings") or "").strip()
            conclusion = (row.get("conclusion") or "").strip()
            if not findings or not conclusion:
                continue

            error_type = _normalize_error_type((row.get("error_type") or "unknown").strip())
            if mode == "lat" and error_type not in {"consistent", "laterality_swap"}:
                continue
            if mode == "site" and error_type not in {"consistent", "site_swap"}:
                continue
            if mode == "severity" and error_type not in {"consistent", "severity_flip"}:
                continue
            if mode == "cross_pair_full" and error_type not in {"consistent", "cross_pair_mismatch"}:
                continue
            if error_types is not None and error_type not in error_types:
                continue

            label_raw = row.get("true_label") if has_true_label else row.get("label")
            true_label = int((label_raw or "").strip())

            source_idx_raw = (row.get("source_idx") or "").strip()
            source_idx = int(source_idx_raw) if source_idx_raw else None
            incidence = (row.get("incidence") or "").strip()

            rows.append(
                EvalRow(
                    patient_id=(row.get("patient_id") or "").strip(),
                    source_idx=source_idx,
                    findings=findings,
                    conclusion=conclusion,
                    error_type=error_type,
                    incidence=incidence,
                    true_label=true_label,
                )
            )
    return rows


def _is_qwen3_model(model: str) -> bool:
    return bool(_QWEN3_MODEL_RE.search(model or ""))


def _strip_thinking(text: str) -> str:
    if not text:
        return ""
    text = _THINK_BLOCK_RE.sub("", text)
    # Qwen3 may emit reasoning in a redacted block when thinking mode is on.
    redacted = "redacted_thinking"
    text = re.sub(
        rf"<\|?{redacted}\|?>[\s\S]*?<\|?/{redacted}\|?>",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip()


def _message_fields(message: dict) -> List[str]:
    parts: List[str] = []
    for key in _MESSAGE_TEXT_KEYS:
        val = message.get(key)
        if val:
            parts.append(str(val))
    return parts


def _message_text(message: dict) -> str:
    """Combine API message fields and drop reasoning/thinking wrappers."""
    return _strip_thinking("\n".join(_message_fields(message)))


def _prepare_user_prompt(
    template: str,
    findings: str,
    conclusion: str,
    *,
    model: str,
    no_think: bool,
) -> str:
    prompt = template.format(findings=findings, conclusion=conclusion)
    if no_think and _is_qwen3_model(model):
        if not prompt.lstrip().lower().startswith("/no_think"):
            prompt = "/no_think\n" + prompt
    return prompt


def _extract_json_object(
    text: str,
    *,
    multiclass: bool = False,
    expects_confidence: bool = False,
) -> Dict:
    if not text:
        raise ValueError("Empty model response.")
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if expects_confidence:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start : end + 1])
            raise
        if multiclass:
            match = _ERROR_TYPE_JSON_RE.search(text)
            if match:
                return {"error_type": match.group(1)}
            kv = _ERROR_TYPE_KV_RE.search(text)
            if kv:
                return {"error_type": kv.group(1)}
        else:
            match = _CONSISTENT_JSON_RE.search(text)
            if match:
                return json.loads(match.group(0))
            kv = _CONSISTENT_KV_RE.search(text)
            if kv:
                return {"consistent": kv.group(1).lower() == "true"}
            match = _ERROR_JSON_RE.search(text)
            if match:
                return json.loads(match.group(0))
            kv = _ERROR_KV_RE.search(text)
            if kv:
                return {"error": kv.group(1).lower() == "true"}
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _bool_label_from_token(token: str) -> Optional[str]:
    label = token.strip().lower()
    if label == "true":
        return "true"
    if label == "false":
        return "false"
    return None


def _logprob_content_items(choice_logprobs: object) -> Optional[List[Dict]]:
    if not choice_logprobs:
        return None
    if isinstance(choice_logprobs, dict):
        content = choice_logprobs.get("content")
        if isinstance(content, list):
            return content
    return None


def _find_boolean_decision_index(content: List[Dict]) -> Optional[int]:
    assembled = ""
    json_started = False
    seen_key = False
    for i, item in enumerate(content):
        token = str(item.get("token", ""))
        assembled += token
        if "{" in assembled:
            json_started = True
        if not json_started:
            continue
        lower = assembled.lower()
        if any(marker in lower for marker in _BOOL_DECISION_KEY_MARKERS):
            seen_key = True
        if seen_key and _bool_label_from_token(token) is not None:
            return i
    return None


def _logprob_to_prob(entry: Dict) -> Optional[float]:
    if "logprob" not in entry:
        return None
    return math.exp(float(entry["logprob"]))


def _alternate_bool_prob_from_top_logprobs(
    top_logprobs: object,
    *,
    alternate_label: str,
) -> Optional[float]:
    if not isinstance(top_logprobs, list):
        return None
    for entry in top_logprobs:
        if not isinstance(entry, dict):
            continue
        if _bool_label_from_token(str(entry.get("token", ""))) != alternate_label:
            continue
        return _logprob_to_prob(entry)
    return None


def _bool_token_probs_from_choice_logprobs(
    choice_logprobs: object,
) -> Tuple[Optional[float], Optional[float]]:
    content = _logprob_content_items(choice_logprobs)
    if not content:
        return None, None
    decision_idx = _find_boolean_decision_index(content)
    if decision_idx is None:
        return None, None
    decision_item = content[decision_idx]
    chosen_label = _bool_label_from_token(str(decision_item.get("token", "")))
    chosen_prob = _logprob_to_prob(decision_item)
    if chosen_label is None or chosen_prob is None:
        return None, None
    if chosen_label == "true":
        return (
            chosen_prob,
            _alternate_bool_prob_from_top_logprobs(
                decision_item.get("top_logprobs"),
                alternate_label="false",
            ),
        )
    return (
        _alternate_bool_prob_from_top_logprobs(
            decision_item.get("top_logprobs"),
            alternate_label="true",
        ),
        chosen_prob,
    )


def _format_prob_for_csv(value: object) -> str:
    if value == "" or value is None:
        return ""
    prob = float(value)
    if math.isnan(prob):
        return ""
    return repr(prob)


def _parse_confidence(value: object) -> int:
    if value is None:
        raise ValueError("JSON did not contain 'confidence'")
    if isinstance(value, bool):
        raise ValueError(
            f"'confidence' must be integer {CONFIDENCE_SCORE_MIN}-{CONFIDENCE_SCORE_MAX}, "
            f"got: {value!r}"
        )
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if not isinstance(value, int):
        raise ValueError(
            f"'confidence' must be integer {CONFIDENCE_SCORE_MIN}-{CONFIDENCE_SCORE_MAX}, "
            f"got: {value!r}"
        )
    if not CONFIDENCE_SCORE_MIN <= value <= CONFIDENCE_SCORE_MAX:
        raise ValueError(
            f"'confidence' must be between {CONFIDENCE_SCORE_MIN} and "
            f"{CONFIDENCE_SCORE_MAX}, got: {value}"
        )
    return value


def _parse_model_response(
    parsed: Dict,
    *,
    multiclass: bool,
    expects_confidence: bool = False,
) -> Dict:
    if multiclass:
        if "error_type" not in parsed:
            raise ValueError(f"JSON did not contain 'error_type': {parsed}")
        raw_error_type = str(parsed["error_type"]).strip()
        if not raw_error_type:
            raise ValueError(f"'error_type' must be non-empty, got: {parsed!r}")
        class_idx = label_from_error_type(raw_error_type)
        error_type_name = MULTICLASS_LABEL_NAMES[class_idx]
        consistent = class_idx == 0
        return {
            "error_type": error_type_name,
            "class_index": class_idx,
            "consistent": consistent,
            "prob_consistent": _bool_to_prob(consistent),
            "prob_inconsistent": _bool_to_prob(not consistent),
        }

    if "consistent" in parsed:
        value = parsed["consistent"]
        if not isinstance(value, bool):
            raise ValueError(f"'consistent' must be boolean, got: {value!r}")
    elif "error" in parsed:
        error_present = parsed["error"]
        if not isinstance(error_present, bool):
            raise ValueError(f"'error' must be boolean, got: {error_present!r}")
        # Prompts using {"error": ...} invert semantics vs {"consistent": ...}.
        value = not error_present
    else:
        raise ValueError(
            f"JSON did not contain 'consistent' or 'error': {parsed}"
        )
    result = {
        "consistent": value,
        "prob_consistent": _bool_to_prob(value),
        "prob_inconsistent": _bool_to_prob(not value),
    }
    if expects_confidence:
        result["confidence"] = _parse_confidence(parsed.get("confidence"))
    return result


def _bool_to_prob(value: bool) -> float:
    return 1.0 if value else 0.0


def _apply_bool_token_logprobs(result: Dict, choice_logprobs: object) -> None:
    p_true, p_false = _bool_token_probs_from_choice_logprobs(choice_logprobs)
    result["prob_consistent"] = (
        p_true if p_true is not None else math.nan
    )
    result["prob_inconsistent"] = (
        p_false if p_false is not None else math.nan
    )


class OpenAICompatClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 32,
        timeout_s: float = 240.0,
        chat_template_kwargs: Optional[Dict[str, object]] = None,
        request_token_logprobs: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.chat_template_kwargs = chat_template_kwargs
        self.request_token_logprobs = request_token_logprobs

    def classify(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        multiclass: bool = False,
        expects_confidence: bool = False,
    ) -> Dict:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.request_token_logprobs and not multiclass:
            payload["logprobs"] = True
            payload["top_logprobs"] = TOKEN_LOGPROBS_TOP_K
        if self.chat_template_kwargs:
            payload["chat_template_kwargs"] = self.chat_template_kwargs
        req = request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as e:
            details = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {e.code} calling chat/completions: {details}") from e
        except error.URLError as e:
            raise RuntimeError(f"Failed to call chat/completions: {e}") from e

        try:
            choice = body["choices"][0]
            message = choice["message"]
        except Exception as e:
            raise RuntimeError(f"Unexpected API response schema: {body}") from e

        content = _message_text(message)
        try:
            parsed = _extract_json_object(
                content,
                multiclass=multiclass,
                expects_confidence=expects_confidence,
            )
            result = _parse_model_response(
                parsed,
                multiclass=multiclass,
                expects_confidence=expects_confidence,
            )
            if self.request_token_logprobs and not multiclass:
                _apply_bool_token_logprobs(result, choice.get("logprobs"))
        except Exception as e:
            preview = content[:400] if content else repr(message)
            raise ValueError(f"{e} | model_output={preview!r}") from e
        result["raw_text"] = content
        return result


def _fmt_pct(num: int, den: int) -> str:
    if den == 0:
        return "nan"
    return f"{(100.0 * num / den):.1f}%"


def _rows_for_metrics(rows: List[Dict]) -> List[Dict]:
    """Rows used for detection metrics; excludes parse failures when parse_ok exists."""
    if not rows or "parse_ok" not in rows[0]:
        return rows
    return [r for r in rows if r["parse_ok"]]


def _quick_console_summary(rows: List[Dict]) -> None:
    total = len(rows)
    valid = _rows_for_metrics(rows)
    passed = len(valid)
    correct = sum(r["is_correct"] for r in valid)
    n_inconsistent = sum(1 for r in valid if r["true_label"] == 0)
    n_consistent = sum(1 for r in valid if r["true_label"] == 1)
    tp0 = sum(r["is_tp0"] for r in valid)
    fp0 = sum(r["is_fp0"] for r in valid)
    parse_ok = sum(r["parse_ok"] for r in rows)
    print("=" * 72)
    print("LLM eval summary (binary)")
    print("=" * 72)
    print(f"rows (OG):       {total}")
    print(f"rows (passed):   {passed}")
    print(f"accuracy:        {_fmt_pct(correct, passed)}")
    print(f"parse success:   {_fmt_pct(parse_ok, total)} ({parse_ok}/{total})")
    print(
        f"detection rate:  {_fmt_pct(tp0, n_inconsistent)} "
        f"({tp0}/{n_inconsistent}, passed rows only)"
    )
    print(
        f"false alarm:     {_fmt_pct(fp0, n_consistent)} "
        f"({fp0}/{n_consistent}, passed rows only)"
    )

    grouped_all: Dict[str, List[Dict]] = defaultdict(list)
    grouped_valid: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        grouped_all[(row.get("error_type") or "unknown").strip()].append(row)
    for row in valid:
        grouped_valid[(row.get("error_type") or "unknown").strip()].append(row)

    if grouped_all:
        sep = "─" * 72
        print(f"\n{sep}")
        print(
            f"{'error_type':>18} {'OG_N':>6} {'pass_N':>6} {'detected':>10} "
            f"{'rate':>10} {'accuracy':>10}"
        )
        print(sep)
        for error_type in sorted(grouped_all):
            og_n = len(grouped_all[error_type])
            subset = grouped_valid.get(error_type, [])
            pass_n = len(subset)
            if error_type == "consistent":
                metric_val = sum(
                    1
                    for r in subset
                    if r["true_label"] == 1 and r["pred_label"] == 1
                )
            else:
                metric_val = sum(
                    1
                    for r in subset
                    if r["true_label"] == 0 and r["pred_label"] == 0
                )
            acc = sum(r["is_correct"] for r in subset)
            print(
                f"{error_type:>18} {og_n:>6} {pass_n:>6} {metric_val:>10} "
                f"{_fmt_pct(metric_val, pass_n):>10} {_fmt_pct(acc, pass_n):>10}"
            )
        print(sep)


def _compute_multiclass_metrics(rows: List[Dict]) -> Dict:
    """Multiclass metrics over parse_ok rows (mirrors bert_inconsistency_multiclass)."""
    valid = [r for r in rows if r["parse_ok"]]
    if not valid:
        return {
            "acc": 0.0,
            "binary_acc": 0.0,
            "binary_precision": 0.0,
            "binary_recall": 0.0,
            "binary_f1": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "errors_only_type_acc": float("nan"),
            "per_class": {
                c: {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "support": 0,
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                }
                for c in range(MULTICLASS_N_CLASSES)
            },
        }

    preds = [int(r["pred_label"]) for r in valid]
    labels = [int(r["true_label"]) for r in valid]
    acc = sum(p == l for p, l in zip(preds, labels)) / len(valid)

    binary_preds = [1 if p != 0 else 0 for p in preds]
    binary_labels = [1 if l != 0 else 0 for l in labels]
    binary_acc = sum(bp == bl for bp, bl in zip(binary_preds, binary_labels)) / len(valid)
    tp_bin = sum(1 for bp, bl in zip(binary_preds, binary_labels) if bp == 1 and bl == 1)
    fp_bin = sum(1 for bp, bl in zip(binary_preds, binary_labels) if bp == 1 and bl == 0)
    fn_bin = sum(1 for bp, bl in zip(binary_preds, binary_labels) if bp == 0 and bl == 1)
    bin_prec = tp_bin / (tp_bin + fp_bin + 1e-8)
    bin_rec = tp_bin / (tp_bin + fn_bin + 1e-8)
    bin_f1 = 2 * bin_prec * bin_rec / (bin_prec + bin_rec + 1e-8)

    per_class: Dict[int, Dict] = {}
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
    macro_f1 = sum(per_class[c]["f1"] for c in active) / max(len(active), 1)
    total_sup = sum(per_class[c]["support"] for c in active)
    weighted_f1 = (
        sum(per_class[c]["f1"] * per_class[c]["support"] for c in active)
        / max(total_sup, 1)
    )

    error_rows = [r for r in valid if int(r["true_label"]) != 0]
    if error_rows:
        errors_only_type_acc = (
            sum(int(r["is_correct"]) for r in error_rows) / len(error_rows)
        )
    else:
        errors_only_type_acc = float("nan")

    return {
        "acc": acc,
        "binary_acc": binary_acc,
        "binary_precision": bin_prec,
        "binary_recall": bin_rec,
        "binary_f1": bin_f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "errors_only_type_acc": errors_only_type_acc,
        "per_class": per_class,
    }


def _print_multiclass_summary(rows: List[Dict]) -> None:
    total = len(rows)
    parse_ok = sum(r["parse_ok"] for r in rows)
    metrics = _compute_multiclass_metrics(rows)
    valid_n = sum(r["parse_ok"] for r in rows)
    correct = sum(r["is_correct"] for r in rows if r["parse_ok"])
    og_support = [sum(1 for r in rows if int(r["true_label"]) == c) for c in range(MULTICLASS_N_CLASSES)]

    print("=" * 72)
    print("LLM eval summary (multiclass)")
    print("=" * 72)
    print(f"rows (OG):            {total}")
    print(f"rows (passed):        {valid_n}")
    print(f"parse success:        {_fmt_pct(parse_ok, total)} ({parse_ok}/{total})")
    print(f"multiclass accuracy:  {_fmt_pct(correct, valid_n)}")
    print(f"macro F1:             {metrics['macro_f1']:.4f}")
    print(f"weighted F1:          {metrics['weighted_f1']:.4f}")
    print(f"binary detection F1:  {metrics['binary_f1']:.4f}")
    err_acc = metrics["errors_only_type_acc"]
    n_errors = sum(1 for r in rows if r["parse_ok"] and int(r["true_label"]) != 0)
    if n_errors and err_acc == err_acc:
        print(f"errors-only type acc: {_fmt_pct(int(err_acc * n_errors), n_errors)}")
    else:
        print("errors-only type acc: nan")

    sep = "─" * 72
    header = (
        f"  {'class':>5}  {'label':<22}  {'OG_N':>5}  {'pass_N':>6}  "
        f"{'P':>6}  {'R':>6}  {'F1':>6}"
    )
    print(f"\n{sep}")
    print(header)
    print(sep)
    for c in range(MULTICLASS_N_CLASSES):
        m = metrics["per_class"][c]
        print(
            f"  {c:>5}  {MULTICLASS_LABEL_NAMES[c]:<22}  {og_support[c]:>5}  "
            f"{m['support']:>6}  {m['precision']:>6.4f}  {m['recall']:>6.4f}  "
            f"{m['f1']:>6.4f}"
        )
    print(sep)
    print(f"  {'':>5}  {'macro avg':<22}  {'':>5}  {'':>6}  {'':>6}  {metrics['macro_f1']:>6.4f}")
    print(
        f"  {'':>5}  {'weighted avg':<22}  {'':>5}  {'':>6}  {'':>6}  "
        f"{metrics['weighted_f1']:>6.4f}"
    )
    print(sep)


def _binary_label_from_class(class_idx: int) -> int:
    """Map multiclass index to binary CSV label (1=consistent, 0=inconsistent)."""
    return 1 if class_idx == 0 else 0


def _append_prediction_row(
    *,
    row: EvalRow,
    parse_ok: int,
    raw_response: str,
    error_msg: str,
    model_name: str,
    multiclass: bool,
    pred_label: int,
    prob_consistent: float,
    prob_inconsistent: float,
    confidence: Optional[int] = None,
    expects_confidence: bool = False,
) -> Dict:
    if multiclass:
        true_label = label_from_error_type(row.error_type)
        true_label_name = MULTICLASS_LABEL_NAMES[true_label]
        pred_label_name = (
            MULTICLASS_LABEL_NAMES[pred_label]
            if parse_ok
            else "parse_failure"
        )
        true_binary = _binary_label_from_class(true_label)
        pred_binary = _binary_label_from_class(pred_label) if parse_ok else 0
        is_correct = int(parse_ok and pred_label == true_label)
    else:
        true_label = int(row.true_label)
        true_label_name = "inconsistent" if true_label == 0 else "consistent"
        pred_label_name = "inconsistent" if pred_label == 0 else "consistent"
        true_binary = true_label
        pred_binary = pred_label
        is_correct = int(pred_label == true_label)

    out = {
        "patient_id": row.patient_id,
        "source_idx": row.source_idx if row.source_idx is not None else "",
        "findings": row.findings,
        "conclusion": row.conclusion,
        "error_type": row.error_type,
        "incidence": row.incidence,
        "true_label": true_label,
        "pred_label": pred_label,
        "true_label_name": true_label_name,
        "pred_label_name": pred_label_name,
        "prob_inconsistent": prob_inconsistent,
        "prob_consistent": prob_consistent,
        "is_correct": is_correct,
        "is_tp0": int(true_binary == 0 and pred_binary == 0),
        "is_fp0": int(true_binary == 1 and pred_binary == 0),
        "is_fn0": int(true_binary == 0 and pred_binary == 1),
        "is_tn0": int(true_binary == 1 and pred_binary == 1),
        "parse_ok": parse_ok,
        "raw_response": raw_response if not error_msg else f"ERROR: {error_msg}",
        "model_name": model_name,
    }
    if expects_confidence:
        out[CONFIDENCE_CSV_FIELD] = confidence if confidence is not None else ""
    return out


def _checkpoint_csv_path(output_csv: str) -> Path:
    path = Path(output_csv)
    return path.with_name(f"{path.stem}.checkpoint{path.suffix}")


def _coerce_prediction_row(row: Dict) -> Dict:
    out = dict(row)
    for key in (
        "source_idx",
        "true_label",
        "pred_label",
        "parse_ok",
        "is_correct",
        "is_tp0",
        "is_fp0",
        "is_fn0",
        "is_tn0",
    ):
        raw = out.get(key, "")
        if raw == "":
            continue
        out[key] = int(raw)
    for key in ("prob_inconsistent", "prob_consistent"):
        raw = out.get(key, "")
        if raw == "":
            out[key] = math.nan
        else:
            out[key] = float(raw)
    raw_confidence = out.get(CONFIDENCE_CSV_FIELD, "")
    if raw_confidence != "":
        out[CONFIDENCE_CSV_FIELD] = int(raw_confidence)
    return out


def _load_predictions_csv(
    csv_path: str,
    *,
    fieldnames: Optional[List[str]] = None,
) -> List[Dict]:
    expected = fieldnames or list(PREDICTION_CSV_FIELDNAMES)
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return []
        missing = set(expected) - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Predictions CSV {csv_path!r} missing columns: {sorted(missing)}"
            )
        return [_coerce_prediction_row(row) for row in reader]


def _write_predictions_csv(
    output_path: str,
    rows: Iterable[Dict],
    *,
    fieldnames: Optional[List[str]] = None,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = fieldnames or list(PREDICTION_CSV_FIELDNAMES)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key in ("prob_inconsistent", "prob_consistent"):
                if key in out:
                    out[key] = _format_prob_for_csv(out.get(key))
            writer.writerow(out)


def _classify_eval_row(
    client: "OpenAICompatClient",
    row: EvalRow,
    *,
    system_prompt: str,
    user_prompt_template: str,
    model: str,
    no_think: bool,
    multiclass: bool,
    expects_confidence: bool,
    max_retries: int,
    retry_sleep_s: float,
) -> Dict:
    prompt = _prepare_user_prompt(
        user_prompt_template,
        row.findings,
        row.conclusion,
        model=model,
        no_think=no_think,
    )
    parse_ok = 1
    raw_response = ""
    pred_label = 0
    prob_consistent = 0.0
    prob_inconsistent = 1.0
    confidence: Optional[int] = None
    error_msg = ""
    parse_failure_class = label_from_error_type("cross_pair_mismatch")

    for attempt in range(1, max_retries + 2):
        try:
            out = client.classify(
                system_prompt,
                prompt,
                multiclass=multiclass,
                expects_confidence=expects_confidence,
            )
            raw_response = out["raw_text"]
            prob_consistent = float(out["prob_consistent"])
            prob_inconsistent = float(out["prob_inconsistent"])
            if expects_confidence:
                confidence = int(out["confidence"])
            if multiclass:
                pred_label = int(out["class_index"])
            else:
                pred_label = 1 if out["consistent"] else 0
            parse_ok = 1
            error_msg = ""
            break
        except Exception as e:
            parse_ok = 0
            pred_label = parse_failure_class if multiclass else 0
            prob_consistent = 0.0
            prob_inconsistent = 1.0
            confidence = None
            error_msg = str(e)
            raw_response = str(e)
            if attempt <= max_retries:
                time.sleep(retry_sleep_s)
            else:
                break

    return _append_prediction_row(
        row=row,
        parse_ok=parse_ok,
        raw_response=raw_response,
        error_msg=error_msg,
        model_name=client.model,
        multiclass=multiclass,
        pred_label=pred_label,
        prob_consistent=prob_consistent,
        prob_inconsistent=prob_inconsistent,
        confidence=confidence,
        expects_confidence=expects_confidence,
    )


def _maybe_write_checkpoint(
    output_csv: str,
    rows: List[Dict],
    *,
    checkpoint_every: int,
    row_idx: int,
    fieldnames: Optional[List[str]] = None,
    force: bool = False,
) -> None:
    if checkpoint_every <= 0 or not rows:
        return
    if not force and row_idx % checkpoint_every != 0:
        return
    checkpoint_path = _checkpoint_csv_path(output_csv)
    _write_predictions_csv(str(checkpoint_path), rows, fieldnames=fieldnames)
    print(f"Checkpoint: {len(rows)} rows -> {checkpoint_path}")


def run_llm_eval(args: argparse.Namespace) -> None:
    eval_rows = _load_eval_rows(
        args.eval_csv,
        mode=args.mode,
        error_types=args.error_types,
    )
    if args.limit is not None and args.limit > 0:
        eval_rows = eval_rows[: args.limit]
    if not eval_rows:
        raise ValueError(f"No eval rows after filtering mode={args.mode!r} from {args.eval_csv}")

    _assert_api_reachable(args.base_url)

    client = OpenAICompatClient(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_s=args.timeout_s,
        chat_template_kwargs=args.chat_template_kwargs,
        request_token_logprobs=args.token_logprobs and not args.multiclass,
    )

    csv_fieldnames = args.prediction_csv_fieldnames
    pred_rows: List[Dict] = []
    start_idx = 0
    checkpoint_path = _checkpoint_csv_path(args.output_csv)
    if args.resume:
        resume_path = args.output_csv
        if not Path(resume_path).is_file() and checkpoint_path.is_file():
            resume_path = str(checkpoint_path)
        if Path(resume_path).is_file():
            pred_rows = _load_predictions_csv(resume_path, fieldnames=csv_fieldnames)
            start_idx = len(pred_rows)
            if start_idx > len(eval_rows):
                raise ValueError(
                    f"Resume CSV has {start_idx} rows but eval set only has {len(eval_rows)} "
                    f"after mode={args.mode!r} filtering"
                )
            if start_idx == len(eval_rows):
                print(
                    f"Resume: {resume_path} already has all {start_idx} rows; "
                    f"writing final predictions to {args.output_csv}"
                )
                _write_predictions_csv(
                    args.output_csv,
                    pred_rows,
                    fieldnames=csv_fieldnames,
                )
                _finish_llm_eval(client, args, pred_rows, aborted=False)
                return
            print(
                f"Resume: loaded {start_idx} rows from {resume_path}; "
                f"continuing at {start_idx + 1}/{len(eval_rows)}"
            )
        else:
            print(f"Resume requested but no checkpoint found at {checkpoint_path}")

    started = time.time()
    aborted = False
    try:
        for idx, row in enumerate(eval_rows[start_idx:], start=start_idx + 1):
            pred_rows.append(
                _classify_eval_row(
                    client,
                    row,
                    system_prompt=args.system_prompt,
                    user_prompt_template=args.user_prompt_template,
                    model=args.model,
                    no_think=args.no_think,
                    multiclass=args.multiclass,
                    expects_confidence=args.expects_confidence,
                    max_retries=args.max_retries,
                    retry_sleep_s=args.retry_sleep_s,
                )
            )

            abort_n = args.abort_after_parse_failures
            if abort_n and idx >= abort_n:
                head = pred_rows[:abort_n]
                if head and all(r["parse_ok"] == 0 for r in head):
                    print(
                        f"ABORT: first {abort_n} rows failed to parse JSON "
                        f"(0/{abort_n} parse success). Check model, /no_think, and max_tokens."
                    )
                    aborted = True
                    break

            if idx % args.progress_every == 0 or idx == len(eval_rows):
                elapsed = max(time.time() - started, 1e-9)
                done_this_run = idx - start_idx
                eps = done_this_run / elapsed if done_this_run else 0.0
                eta_s = (len(eval_rows) - idx) / max(eps, 1e-9)
                parse_ok_so_far = sum(int(r["parse_ok"]) for r in pred_rows)
                print(
                    f"[{idx}/{len(eval_rows)}] "
                    f"{eps:.2f} rows/s | parse_ok={parse_ok_so_far}/{idx} | "
                    f"eta={eta_s/60.0:.1f} min"
                )

            _maybe_write_checkpoint(
                args.output_csv,
                pred_rows,
                checkpoint_every=args.checkpoint_every,
                row_idx=idx,
                fieldnames=csv_fieldnames,
            )
    finally:
        _maybe_write_checkpoint(
            args.output_csv,
            pred_rows,
            checkpoint_every=args.checkpoint_every,
            row_idx=0,
            fieldnames=csv_fieldnames,
            force=bool(pred_rows),
        )

    _write_predictions_csv(
        args.output_csv,
        pred_rows,
        fieldnames=csv_fieldnames,
    )
    print(f"Saved row-level predictions to: {args.output_csv}")
    _finish_llm_eval(client, args, pred_rows, aborted=aborted)


def _finish_llm_eval(
    client: "OpenAICompatClient",
    args: argparse.Namespace,
    pred_rows: List[Dict],
    *,
    aborted: bool,
) -> None:
    if args.multiclass:
        _print_multiclass_summary(pred_rows)
    else:
        _quick_console_summary(pred_rows)

    if aborted:
        abort_n = args.abort_after_parse_failures
        raise ParseFailureAbort(
            f"Stopped after {abort_n} rows with 0% parse success (see {args.output_csv})"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a locally hosted OpenAI-compatible LLM on inconsistency CSV "
            "datasets. Default base_url is loopback only (no cloud APIs)."
        )
    )
    parser.add_argument("--eval_csv", type=str, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "lat", "site", "severity", "hard", "cross_pair_full"],
        help=(
            "Row filter: all=no filter; hard=full hard-negative v4 suite (all error types "
            "in the CSV, including cross_pair_mismatch); lat=consistent+laterality_swap only; "
            "severity=consistent+severity_flip only; "
            "cross_pair_full=consistent+cross_pair_mismatch only. Default: all."
        ),
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible API on this machine (vLLM, Ollama, etc.).",
    )
    parser.add_argument(
        "--allow_remote",
        action="store_true",
        help="Allow non-loopback base_url (cloud APIs). Off by default for privacy.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="EMPTY",
        help="API key if your local server requires one (often unused for vLLM).",
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--token_logprobs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Request top-5 token logprobs at the boolean JSON decision. "
            "prob_consistent/prob_inconsistent are set from the chosen token's "
            "P(true)/P(false); the alternate column is filled only if that token "
            "appears in top-k. Binary prompts only."
        ),
    )
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--timeout_s", type=float, default=240.0)
    parser.add_argument("--max_retries", type=int, default=1)
    parser.add_argument("--retry_sleep_s", type=float, default=1.0)
    parser.add_argument("--progress_every", type=int, default=25)
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY,
        help=(
            "Rewrite predictions.checkpoint.csv every N completed rows (0 disables). "
            f"Default: {DEFAULT_CHECKPOINT_EVERY}."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue from predictions.csv or predictions.checkpoint.csv in the output "
            "run directory (same eval_csv/mode/limit as the original run)."
        ),
    )
    parser.add_argument(
        "--abort_after_parse_failures",
        type=int,
        default=DEFAULT_ABORT_AFTER_PARSE_FAILURES,
        help=(
            "Stop early if the first N rows all fail JSON parsing (0 disables). "
            f"Default: {DEFAULT_ABORT_AFTER_PARSE_FAILURES}."
        ),
    )
    parser.add_argument(
        "--no_think",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For Qwen3 models, prepend /no_think to disable reasoning mode (default: on). "
            "Keeps short JSON answers within max_tokens."
        ),
    )
    parser.add_argument(
        "--chat_template_kwargs",
        type=str,
        default=None,
        help=(
            "JSON object merged into vLLM chat_template_kwargs (e.g. "
            '\'{"enable_thinking": false}\').'
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N rows after mode/error_type filtering (smoke tests).",
    )
    parser.add_argument(
        "--error_types",
        type=str,
        default=None,
        help=(
            "Comma-separated error_type filter applied after --mode filtering. "
            "Example for v6 human NF vs disease swaps only: "
            "--error_types NF_v_Dis_swap (512 rows in hard_negative_eval_v6_single_swap.csv)."
        ),
    )
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument(
        "--output_run_dir",
        type=str,
        default=None,
        help=(
            "Use this directory as the run root (logs/, command.txt) instead of "
            "creating a new dated folder under runs/llm_baseline/."
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
        "--output_csv",
        type=str,
        default=None,
        help="Explicit output CSV. Defaults to run results/predictions.csv",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        choices=sorted(USER_PROMPT_TEMPLATES),
        default="1",
        help=prompt_choices_help(),
    )
    parser.add_argument(
        "--user_prompt_template",
        type=str,
        default=None,
        help=(
            "Custom Python format template with {findings} and {conclusion} placeholders "
            "(overrides --prompt). Use single braces for placeholders; escape literal "
            "braces as {{ and }}."
        ),
    )
    return parser


def _resolve_prompts(args: argparse.Namespace) -> None:
    args.system_prompt = SYSTEM_PROMPT_TEMPLATES[args.prompt]
    args.custom_user_prompt = args.user_prompt_template is not None
    if not args.custom_user_prompt:
        args.user_prompt_template = USER_PROMPT_TEMPLATES[args.prompt]
    args.multiclass = is_multiclass_prompt(args.prompt)
    args.expects_confidence = is_confidence_prompt(args.prompt)
    args.prediction_csv_fieldnames = prediction_csv_fieldnames(
        expects_confidence=args.expects_confidence
    )
    if args.multiclass and args.max_tokens == DEFAULT_MAX_TOKENS:
        args.max_tokens = MULTICLASS_DEFAULT_MAX_TOKENS
    elif args.expects_confidence and args.max_tokens == DEFAULT_MAX_TOKENS:
        args.max_tokens = CONFIDENCE_DEFAULT_MAX_TOKENS


def _resolve_error_types_arg(args: argparse.Namespace) -> None:
    args.error_types = _parse_error_types_filter(getattr(args, "error_types", None))


def _resolve_chat_template_kwargs_arg(args: argparse.Namespace) -> None:
    raw = getattr(args, "chat_template_kwargs", None)
    if raw is None or raw == "":
        args.chat_template_kwargs = None
        return
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("--chat_template_kwargs must be a JSON object")
    args.chat_template_kwargs = parsed


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _resolve_prompts(args)
    _resolve_error_types_arg(args)
    _resolve_chat_template_kwargs_arg(args)
    _assert_local_inference(args.base_url, args.allow_remote)
    run_layout = setup_run(
        SCRIPT_ID,
        tags=("eval", args.mode),
        run_name=args.run_name,
        run_category=RUN_CATEGORY,
        output_model_dir=args.output_run_dir,
        command_stem=args.command_stem,
        argv=current_argv(),
    )
    args.output_csv = default_result_path(run_layout, "predictions.csv", args.output_csv)
    with tee_run_output(run_layout.log_path(args.log_file)):
        print(f"Run outputs: {run_layout.root}")
        print(f"Eval CSV: {args.eval_csv}")
        print(f"Mode: {args.mode}")
        if args.error_types:
            print(f"Error types: {', '.join(sorted(args.error_types))}")
        print(f"Model: {args.model}")
        print(f"Prompt variant: {args.prompt} (system + user)")
        if args.multiclass:
            print("Eval mode: multiclass (6-class error_type)")
        if args.custom_user_prompt:
            print("User prompt: custom (--user_prompt_template)")
        else:
            print("User prompt: built-in template for variant")
        print(f"API (local): {args.base_url}")
        if args.no_think and _is_qwen3_model(args.model):
            print("Qwen3: /no_think enabled (non-reasoning mode)")
        if args.token_logprobs and not args.multiclass:
            print(
                "Token logprobs: enabled "
                f"(top {TOKEN_LOGPROBS_TOP_K}; chosen-token prob always written; "
                "alternate only if in top-k)"
            )
        try:
            run_llm_eval(args)
        except ParseFailureAbort as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
