"""Prompt variants for LLM findings/conclusion inconsistency eval.

Add new variants by appending a PromptVariant to PROMPT_VARIANTS. Each variant
pairs a system prompt with a user template that must include {findings} and
{conclusion} placeholders (escape literal braces as {{ and }}).

Binary prompts may ask for either {"consistent": true/false} or
{"error": true/false} (inverse semantics); eval_llm_baseline accepts both.
"""
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class PromptVariant:
    id: str
    description: str
    system_prompt: str
    user_prompt_template: str
    multiclass: bool = False
    expects_confidence: bool = False


# Mirrors bert_inconsistency_multiclass.LABEL_MAP / LABEL_NAMES (keep in sync).
MULTICLASS_LABEL_MAP: Dict[str, int] = {
    "positive": 0,
    "consistent": 0,
    "laterality_swap": 1,
    "site_swap": 2,
    "severity_flip": 3,
    "NF_v_Dis_swap": 4,
    "cross_pair_mismatch": 5,
    "random_mismatch": 5,
}
MULTICLASS_LABEL_NAMES: Dict[int, str] = {
    0: "consistent",
    1: "laterality_swap",
    2: "site_swap",
    3: "severity_flip",
    4: "NF_v_Dis_swap",
    5: "cross_pair_mismatch",
}
MULTICLASS_N_CLASSES = 6
MULTICLASS_DEFAULT_MAX_TOKENS = 48
CONFIDENCE_DEFAULT_MAX_TOKENS = 48
CONFIDENCE_SCORE_MIN = 1
CONFIDENCE_SCORE_MAX = 5


def normalize_multiclass_error_type(error_type: str) -> str:
    """Canonical error_type string for multiclass label lookup."""
    error_type = (error_type or "unknown").strip()
    if error_type == "positive":
        return "consistent"
    if error_type == "random_mismatch":
        return "cross_pair_mismatch"
    return error_type


def label_from_error_type(error_type: str) -> int:
    """Multiclass index for an error_type string (unknown → cross_pair_mismatch)."""
    return MULTICLASS_LABEL_MAP.get(
        normalize_multiclass_error_type(error_type),
        MULTICLASS_LABEL_MAP["cross_pair_mismatch"],
    )


PROMPT_VARIANTS: Dict[str, PromptVariant] = {
    "0": PromptVariant(
        id="0",
        description="classify-style",
        system_prompt=(
            "You are a radiologist checking an oncology FDG PET/CT report for errors by comparing the FINDINGS and CONCLUSION."
        ),
        user_prompt_template="""FINDINGS:
{findings}

CONCLUSION:
{conclusion}

Question: Is the conclusion consistent with the findings?
Answer with exactly one JSON object and no additional text:
{{"consistent": true}}  if the conclusion accurately reflects the findings and doesn't contain an error
{{"consistent": false}} if the conclusion contains an error.
""",
    ),
    "16": PromptVariant(
        id="16",
        description="p0 really no information",
        system_prompt=(
            "You are checking a report for errors by comparing the FINDINGS and CONCLUSION."
        ),
        user_prompt_template="""FINDINGS:
{findings}

CONCLUSION:
{conclusion}

Question: Is the conclusion consistent with the findings?
Answer with exactly one JSON object and no additional text:
{{"consistent": true}}  if the conclusion accurately reflects the findings and doesn't contain an error
{{"consistent": false}} if the conclusion contains an error.
""",
    ),
    "17": PromptVariant(
        id="17",
        description="p16 order swap",
        system_prompt=(
            "You are checking a report for errors by comparing the FINDINGS and CONCLUSION."
        ),
        user_prompt_template="""FINDINGS:
{findings}

CONCLUSION:
{conclusion}

Question: Is the conclusion consistent with the findings?
Answer with exactly one JSON object and no additional text:
{{"consistent": false}} if the conclusion contains an error
{{"consistent": true}}  if the conclusion accurately reflects the findings and doesn't contain an error.

""",
    ),
    "1": PromptVariant(
        id="1",
        description="question-style (default)",
        system_prompt=(
            "You are reviewing a PET/CT report for errors."
        ),
        user_prompt_template="""FINDINGS:
{findings}

CONCLUSION:
{conclusion}

Question: Does this report contain any errors?
Answer with exactly one JSON object and no additional text:
{{"error": false}} if no errors.
{{"error": true}} if any errors are present.
""",
    ),
    "2": PromptVariant(
        id="2",
        description="classify-style",
        system_prompt=(
            "You are a radiologist checking an oncology FDG PET/CT report for errors by comparing the FINDINGS and CONCLUSION."
        ),
        user_prompt_template="""FINDINGS:
{findings}

CONCLUSION:
{conclusion}

Question: Is the conclusion consistent with the findings?
Answer with exactly one JSON object and no additional text:
{{"consistent": true}}  if the conclusion accurately reflects the findings and doesn't contain an error
{{"consistent": false}} if the conclusion contains an error, like contradicting the findings, misstating laterality, site, or disease severity.
""",
    ),
    # Incremental ladder for paper figures (render prompt 12 once, color by layer):
    #   Base (prompt 2): system role + sections + question + JSON rubric (no examples).
    #   +Layer A (prompt 3): parenthetical examples on the false branch only.
    #   +Layer B (prompt 12): review-protocol block between CONCLUSION and Question.
    "3": PromptVariant(
        id="3",
        description="classify-style",
        system_prompt=(
            "You are a radiologist checking an oncology FDG PET/CT report for errors by comparing the FINDINGS and CONCLUSION."
        ),
        user_prompt_template="""FINDINGS:
{findings}

CONCLUSION:
{conclusion}

Question: Is the conclusion consistent with the findings?
Answer with exactly one JSON object and no additional text:
{{"consistent": true}}  if the conclusion accurately reflects the findings and doesn't contain an error
{{"consistent": false}} if the conclusion contains an error, like contradicting the findings, misstating laterality (left/right), site (e.g. upper/lower, lung cancer instead of bone), or disease severity (e.g. faint uptake instead of extreme, normal instead of cancer).
""",
    ),
    "12": PromptVariant(
        id="12",
        description="classify-style + review protocol (extends prompt 3)",
        system_prompt=(
            "You are a radiologist checking an oncology FDG PET/CT report for errors by comparing the FINDINGS and CONCLUSION."
        ),
        user_prompt_template="""FINDINGS:
{findings}

CONCLUSION:
{conclusion}

The FINDINGS section is the detailed interpretation; the CONCLUSION is the summary impression.
Compare them on:
1. Disease presence — FDG-avid disease vs no significant disease
2. Laterality — left vs right
3. Site — anatomical location (e.g. lung vs liver, upper vs lower)
4. Severity — disease burden / uptake intensity (e.g. faint vs intense, mild vs severe, markedly vs mildly avid)
5. Conclusion mismatch — does the conclusion address the same key findings as the findings section (not a different disease, site, or overall assessment)?

Question: Is the conclusion consistent with the findings?
Answer with exactly one JSON object and no additional text:
{{"consistent": true}}  if the conclusion accurately reflects the findings and doesn't contain an error
{{"consistent": false}} if the conclusion contains an error, like contradicting the findings, misstating laterality (left/right), site (e.g. upper/lower, lung cancer instead of bone), or disease severity (e.g. faint uptake instead of extreme, normal instead of cancer).
""",
    ),
     "13": PromptVariant(
        id="13",
        description="classify-style + review protocol (extends prompt 3)",
        system_prompt=(
            "You are a radiologist checking an oncology FDG PET/CT report for errors by comparing the FINDINGS and CONCLUSION."
        ),
        user_prompt_template="""FINDINGS:
{findings}

CONCLUSION:
{conclusion}

The FINDINGS section is the detailed interpretation; the CONCLUSION is the summary impression.
Compare them on:
1. Disease or significant findings presence — FDG-avid disease or cancer presence vs no significant disease or findings. 
2. Laterality — left vs right
3. Site — spatial or anatomical location (e.g. lung vs liver, upper vs lower, pleural vs parenchymal, etc.)
4. Severity — disease burden / uptake intensity (e.g. faint vs intense, mild vs severe, markedly instead of mildly avid, small vs large tumour)
5. Conclusion mismatch — does the conclusion address the same findings as the findings section (not a different disease, site, or overall assessment, i.e. the conclusion has not been swapped)?

Question: Is the conclusion consistent with the findings?
Answer with exactly one JSON object and no additional text:
{{"consistent": true}}  if the conclusion accurately reflects the findings and doesn't contain an error
{{"consistent": false}} if the conclusion contains an error, like contradicting the findings, misstating laterality (left/right), site (e.g. upper/lower, lung cancer instead of bone), or disease severity (e.g. faint uptake instead of extreme, normal instead of cancer).
""",
    ),
    
    "4": PromptVariant(
        id="4",
        description="multiclass (6 error types, matches bert_inconsistency_multiclass)",
        multiclass=True,
        system_prompt=(
            "You are a radiologist checking an oncology FDG PET/CT report for errors by "
            "comparing the FINDINGS and CONCLUSION. Classify the error type using the "
            "standard multiclass taxonomy."
        ),
        user_prompt_template="""FINDINGS:
{findings}

CONCLUSION:
{conclusion}

Task: Classify the relationship between the FINDINGS and CONCLUSION. Choose exactly one error type.

Error types (use these exact strings):
- consistent: findings and conclusion agree; no clinically meaningful mismatch or error.
- laterality_swap: conclusion misstates left/right laterality relative to findings
- site_swap: conclusion misstates spatial or anatomical site (e.g. lung vs liver, upper vs lower, pleural vs parenchymal)
- severity_flip: conclusion misstates disease severity or uptake intensity (e.g. faint vs intense, mild vs severe, small vs large tumour)
- NF_v_Dis_swap: conclusion contradicts whether findings show disease (no-findings vs disease conclusion swapped)
- cross_pair_mismatch: conclusion contradicts findings but the error is not laterality_swap, site_swap, severity_flip, or NF_v_Dis_swap (ie the whole conclusion has been swapped)

Answer with exactly one JSON object and no additional text:
{{"error_type": "<one of: consistent, laterality_swap, site_swap, severity_flip, NF_v_Dis_swap, cross_pair_mismatch>"}}
""",
    ),
    "5": PromptVariant(
        id="5",
        description="single-classify-style",
        system_prompt=(
            "You are a radiologist checking an oncology FDG PET/CT report for one type of error by comparing the FINDINGS and CONCLUSION."
        ),
        user_prompt_template="""FINDINGS:
{findings}

CONCLUSION:
{conclusion}

Question: Is the conclusion consistent with the findings?
Answer with exactly one JSON object and no additional text:
{{"consistent": false}} if the conclusion contains an error of disease severity (e.g. faint uptake vs intense, mild vs severe).
{{"consistent": true}}  otherwise (if the conclusion accurately reflects the findings and doesn't contain an error of disease severity or if the evidence is ambiguous).
""",
    ),
    "6": PromptVariant(
        id="6",
        description="single-classify-style",
        system_prompt=(
            "You are a radiologist checking an oncology FDG PET/CT report for one type of error by comparing the FINDINGS and CONCLUSION."
        ),
        user_prompt_template="""FINDINGS:
{findings}

CONCLUSION:
{conclusion}

Question: Is the conclusion consistent with the findings?
Answer with exactly one JSON object and no additional text:
{{"consistent": false}} if the conclusion contains an error of laterality (e.g. the conclusion states left instead of right).
{{"consistent": true}}  otherwise (if the conclusion accurately reflects the findings and doesn't contain an error or if the evidence is ambiguous).
""",
    ),
    "7": PromptVariant(
        id="7",
        description="classify-style",
        system_prompt=(
            "You are a radiologist checking an oncology FDG PET/CT report for errors by comparing the FINDINGS and CONCLUSION."
        ),
        user_prompt_template="""FINDINGS:
{findings}

CONCLUSION:
{conclusion}

Question: Is the conclusion consistent with the findings?
Answer with exactly one JSON object and no additional text:
{{"consistent": false}} if the conclusion contains an error, like contradicting the findings, misstating laterality (left/right), site (e.g. upper/lower, lung cancer instead of bone), or disease severity (e.g. faint uptake instead of extreme, normal instead of cancer).
{{"consistent": true}}  otherwise (if the conclusion accurately reflects the findings and doesn't contain an error or if the evidence is ambiguous). 
The priority is high precision. Only flag errors that are strongly supported by the report text.
Do not consider stylistic issues, missing details, or subjective differences in interpretation to be errors.
""",
    ),
    "8": PromptVariant(
        id="8",
        description="3_plus_7_precision_prompt",
        system_prompt=(
            "You are a radiologist checking an oncology FDG PET/CT report for errors by comparing the FINDINGS and CONCLUSION."
        ),
        user_prompt_template="""FINDINGS:
{findings}

CONCLUSION:
{conclusion}

Question: Is the conclusion consistent with the findings?
Answer with exactly one JSON object and no additional text:
{{"consistent": true}}  if the conclusion accurately reflects the findings and doesn't contain an error
{{"consistent": false}} if the conclusion contains an error, like contradicting the findings, misstating laterality (left/right), site (e.g. upper/lower, lung cancer instead of bone), or disease severity (e.g. faint uptake instead of extreme, normal instead of cancer).
Minimize false alarms: only output {{\"consistent\": false}} when an error is clearly and explicitly supported by the text. When uncertain, output {{\"consistent\": true}}.
""",
    ),
    "9": PromptVariant(
        id="9",
        description="3 but switching consistent to error",
        system_prompt=(
            "You are a radiologist checking an oncology FDG PET/CT report for errors by comparing the FINDINGS and CONCLUSION."
        ),
        user_prompt_template="""FINDINGS:
{findings}

CONCLUSION:
{conclusion}

Question: Is the conclusion consistent with the findings?
Answer with exactly one JSON object and no additional text:
{{"error": false}}  if the conclusion accurately reflects the findings and doesn't contain an error
{{"error": true}} if the conclusion contains an error, like contradicting the findings, misstating laterality (left/right), site (e.g. upper/lower, lung cancer instead of bone), or disease severity (e.g. faint uptake instead of extreme, normal instead of cancer).
""",
    ),
    "10": PromptVariant(
        id="10",
        description="structured checklist review (error key)",
        system_prompt=(
            "You are a radiologist performing a structured quality review of an oncology "
            "FDG PET/CT report. Compare the FINDINGS and CONCLUSION for factual consistency only."
        ),
        user_prompt_template="""FINDINGS:
{findings}

CONCLUSION:
{conclusion}

Review the report using this checklist. For each item, decide whether the CONCLUSION matches the FINDINGS:
1. Disease presence: Does the conclusion correctly reflect whether findings show FDG-avid disease vs no significant disease?
2. Laterality: Are left/right assignments consistent?
3. Severity: Is disease burden / uptake intensity consistent (e.g. faint vs intense, mild vs severe)?
4. Conclusion mismatch: Does the conclusion address the same findings and impression as the findings section (not a different disease, site, or overall assessment)?
5. Other: Any other error such as are anatomical sites consistent (e.g. lung vs liver, upper vs lower, pleural vs parenchymal)?

Count an error only if one checklist item is clearly contradicted by the report text.
Do NOT count as errors: stylistic differences, omitted minor details, hedging language, or reasonable summarization.

If any checklist item is clearly failed then there is an error present.
If all checklist items pass, or the evidence is ambiguous then there is no error.

Answer with exactly one JSON object and no additional text:
{{"error": false}} if no checklist item is clearly failed.
{{"error": true}}  if any checklist item is clearly failed.

Minimize false alarms: output {{\"error\": true}} only when a checklist failure is explicit in the text. When uncertain, output {{\"error\": false}}.
""",
    ),
    "11": PromptVariant(
        id="11",
        description="p3 + confidence 1-5",
        expects_confidence=True,
        system_prompt=(
            "You are a radiologist checking an oncology FDG PET/CT report for errors by comparing the FINDINGS and CONCLUSION."
        ),
        user_prompt_template="""FINDINGS:
{findings}

CONCLUSION:
{conclusion}

Question: Is the conclusion consistent with the findings?
Answer with exactly one JSON object and no additional text:
{{"consistent": true, "confidence": <integer 1-5>}}  if the conclusion accurately reflects the findings and doesn't contain an error
{{"consistent": false, "confidence": <integer 1-5>}} if the conclusion contains an error, like contradicting the findings, misstating laterality (left/right), site (e.g. upper/lower, lung cancer instead of bone), or disease severity (e.g. faint uptake instead of extreme, normal instead of cancer).
"confidence" is how strongly the error judgment is supported by the evidence (1 = not well supported, 5 = very well supported). When evidence is ambiguous, use confidence 1–2. When there is explicit evidence for your decision then use confidence 4-5.
""",
    ),
    "14": PromptVariant(
        id="14",
        description="p3 + confidence 1-5 minus radiologist",
        expects_confidence=True,
        system_prompt=(
            "You are checking an oncology FDG PET/CT report for errors by comparing the FINDINGS and CONCLUSION."
        ),
        user_prompt_template="""FINDINGS:
{findings}

CONCLUSION:
{conclusion}

Question: Is the conclusion consistent with the findings?
Answer with exactly one JSON object and no additional text:
{{"consistent": true, "confidence": <integer 1-5>}}  if the conclusion accurately reflects the findings and doesn't contain an error
{{"consistent": false, "confidence": <integer 1-5>}} if the conclusion contains an error, like contradicting the findings, misstating laterality (left/right), site (e.g. upper/lower, lung cancer instead of bone), or disease severity (e.g. faint uptake instead of extreme, normal instead of cancer).
"confidence" is how strongly the error judgment is supported by the evidence (1 = not well supported, 5 = very well supported). When evidence is ambiguous, use confidence 1–2. When there is explicit evidence for your decision then use confidence 4-5.
""",
    ),
    "15": PromptVariant(
        id="15",
        description="p3 ish def error, prob error, unsure, prob not error, def not error",
        expects_confidence=True,
        system_prompt=(
            "You are a radiologist checking an oncology FDG PET/CT report for errors by comparing the FINDINGS and CONCLUSION."
        ),
        user_prompt_template="""FINDINGS:
{findings}

CONCLUSION:
{conclusion}

Question: Is the conclusion consistent with the findings?
Answer with exactly one JSON object and no additional text:
{{"consistent": "<one of: definite error, probable error, unsure, probable not error, definite not error>"}} 
Errors include contradicting the findings, misstating laterality (left/right), site (e.g. upper/lower, lung cancer instead of bone), or disease severity (e.g. faint uptake instead of extreme, normal instead of cancer).
""",
    ),
    "18": PromptVariant(
        id="18",
        description="p13 plus fp form p8",
        system_prompt=(
            "You are a radiologist checking an oncology FDG PET/CT report for errors by comparing the FINDINGS and CONCLUSION."
        ),
        user_prompt_template="""FINDINGS:
{findings}

CONCLUSION:
{conclusion}

The FINDINGS section is the detailed interpretation; the CONCLUSION is the summary impression.
Compare them on:
1. Disease or significant findings presence — FDG-avid disease or cancer presence vs no significant disease or findings. 
2. Laterality — left vs right
3. Site — spatial or anatomical location (e.g. lung vs liver, upper vs lower, pleural vs parenchymal, etc.)
4. Severity — disease burden / uptake intensity (e.g. faint vs intense, mild vs severe, markedly instead of mildly avid, small vs large tumour)
5. Conclusion mismatch — does the conclusion address the same findings as the findings section (not a different disease, site, or overall assessment, i.e. the conclusion has not been swapped)?

Question: Is the conclusion consistent with the findings?
Answer with exactly one JSON object and no additional text:
{{"consistent": true}}  if the conclusion accurately reflects the findings and doesn't contain an error
{{"consistent": false}} if the conclusion contains an error, like contradicting the findings, misstating laterality (left/right), site (e.g. upper/lower, lung cancer instead of bone), or disease severity (e.g. faint uptake instead of extreme, normal instead of cancer).
Minimize false alarms: only output {{\"consistent\": false}} when an error is clearly and explicitly supported by the text. When uncertain, output {{\"consistent\": true}}.
""",
    ),
}

USER_PROMPT_TEMPLATES: Dict[str, str] = {
    vid: variant.user_prompt_template for vid, variant in PROMPT_VARIANTS.items()
}
SYSTEM_PROMPT_TEMPLATES: Dict[str, str] = {
    vid: variant.system_prompt for vid, variant in PROMPT_VARIANTS.items()
}

# Backward-compatible alias (prompt 1).
DEFAULT_USER_PROMPT_TEMPLATE = USER_PROMPT_TEMPLATES["1"]


def is_multiclass_prompt(prompt_id: str) -> bool:
    variant: Optional[PromptVariant] = PROMPT_VARIANTS.get(prompt_id)
    return bool(variant and variant.multiclass)


def is_confidence_prompt(prompt_id: str) -> bool:
    variant: Optional[PromptVariant] = PROMPT_VARIANTS.get(prompt_id)
    return bool(variant and variant.expects_confidence)


def prompt_choices_help() -> str:
    """Argparse help text listing built-in --prompt variants."""
    lines = [
        "Built-in prompt variant (system + user). User template ignored if "
        "--user_prompt_template is set; system prompt still follows this variant.",
        "Variants:",
    ]
    for vid in sorted(PROMPT_VARIANTS):
        desc = PROMPT_VARIANTS[vid].description
        lines.append(f"  {vid}={desc}")
    return " ".join(lines)
