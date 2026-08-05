"""Severity-flip corruption: scalar pairs + tiered qualitative bags (morphology-aware)."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

# Directional size / grade swaps (Option A — scalar).
SEVERITY_SCALAR_RULES: List[Tuple[str, str, str]] = [
    ("small", "large", "severity_flip"),
    ("low-grade", "high-grade", "severity_flip"),
]

# Qualitative tiers; cross-tier replacement draws from opposite bag (Option A + C).
SEVERITY_TIERS: Tuple[frozenset[str], ...] = (
    frozenset({"mild", "mildly", "faint", "faintly", "minimal", "minimally"}),
    frozenset({"moderate", "moderately"}),
    frozenset(
        {
            "severe",
            "severely",
            "marked",
            "markedly",
            "extreme",
            "extremely",
            "intense",
            "intensely",
        }
    ),
)

# Backward-compatible alias (scalar rules only).
SEVERITY_FLIP_RULES = SEVERITY_SCALAR_RULES

SWAP_INSTANCES_CHOICES_TRAINING = ("all", "random", "random_n", "single")
SINGLE_SWAP_INSTANCES = frozenset({"random", "single"})

_ARTICLE_BEFORE_TERM = re.compile(r"\b(a|an)\s+$", re.IGNORECASE)


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


def severity_flip_terms() -> List[str]:
    """All trigger terms (scalar + tier bags) for incidence / coverage."""
    terms = _rule_terms(SEVERITY_SCALAR_RULES)
    seen = {t.lower() for t in terms}
    for bag in SEVERITY_TIERS:
        for term in sorted(bag, key=len, reverse=True):
            key = term.lower()
            if key not in seen:
                seen.add(key)
                terms.append(term)
    return terms


SEVERITY_FLIP_TERMS = severity_flip_terms()

# Common holdout preset: drop mild↔severe anchor terms (adjective + adverb forms).
MILD_SEVERE_EXCLUDED_TERMS = frozenset({"mild", "mildly", "severe", "severely"})


@dataclass(frozen=True)
class SeverityFlipOptions:
    """Training-only filters; eval dataset generation leaves options at default."""

    exclude_scalar: bool = False
    excluded_terms: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        normalized = frozenset(t.strip().lower() for t in self.excluded_terms if t.strip())
        object.__setattr__(self, "excluded_terms", normalized)


def parse_excluded_severity_terms(text: Optional[str]) -> frozenset[str]:
    if not text or not str(text).strip():
        return frozenset()
    return frozenset(
        t.strip().lower() for t in str(text).split(",") if t.strip()
    )


def _tier_terms_allowed(options: Optional[SeverityFlipOptions]) -> List[str]:
    excluded = (options.excluded_terms if options else frozenset())
    terms: List[str] = []
    seen: set[str] = set()
    for bag in SEVERITY_TIERS:
        for term in sorted(bag, key=len, reverse=True):
            key = term.lower()
            if key not in seen and key not in excluded:
                seen.add(key)
                terms.append(term)
    return terms


def _tier_pattern_for(options: Optional[SeverityFlipOptions]) -> Optional[re.Pattern[str]]:
    terms = _tier_terms_allowed(options)
    if not terms:
        return None
    return re.compile(
        r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b",
        re.IGNORECASE,
    )


def _filtered_tier_bag(bag: frozenset[str], options: Optional[SeverityFlipOptions]) -> frozenset[str]:
    excluded = options.excluded_terms if options else frozenset()
    filtered = frozenset(t for t in bag if t.lower() not in excluded)
    return filtered


def _severity_tier_index(term: str) -> Optional[int]:
    key = term.lower()
    for i, bag in enumerate(SEVERITY_TIERS):
        if key in bag:
            return i
    return None


def _is_adverb_form(term: str) -> bool:
    return term.lower().endswith("ly")


def _morphology_pool(
    target_bag: frozenset[str],
    matched: str,
    options: Optional[SeverityFlipOptions] = None,
) -> List[str]:
    """Synonyms in target tier with same morphology class (adverb vs adjective)."""
    excluded = options.excluded_terms if options else frozenset()
    matched_lower = matched.lower()
    if _is_adverb_form(matched):
        pool = [
            t
            for t in target_bag
            if t.endswith("ly") and t != matched_lower and t not in excluded
        ]
    else:
        pool = [
            t
            for t in target_bag
            if not t.endswith("ly") and t != matched_lower and t not in excluded
        ]
    if not pool:
        pool = [t for t in target_bag if t != matched_lower and t not in excluded]
    return pool


def _opposite_tier_indices(tier_idx: int, n_tiers: int) -> List[int]:
    if tier_idx == 0:
        return [n_tiers - 1]
    if tier_idx == n_tiers - 1:
        return [0]
    return [0, n_tiers - 1]


def _uses_an_article(word: str) -> bool:
    """Heuristic a/an choice from the first letter of the following word."""
    w = word.strip().lower()
    return bool(w) and w[0] in "aeiou"


def _correct_article(article: str, following_word: str) -> str:
    want_an = _uses_an_article(following_word)
    if (article.lower() == "an") == want_an:
        return article
    new_art = "an" if want_an else "a"
    if article.isupper():
        return new_art.upper()
    if article[0].isupper():
        return new_art.capitalize()
    return new_art


def _apply_casing_to_replacement(source_token: str, replacement: str) -> str:
    if source_token.isupper():
        return replacement.upper()
    if source_token[0].isupper():
        return replacement.capitalize()
    return replacement


def _replacement_for_token(token: str, a: str, b: str) -> str:
    token_lower = token.lower()
    replacement = b if token_lower == a.lower() else a
    return _apply_casing_to_replacement(token, replacement)


def _replace_token_span(
    text: str,
    match: re.Match[str],
    replacement_word: str,
) -> str:
    """Replace a matched token, fixing a/an immediately before it when present."""
    start = match.start()
    prefix = text[:start]
    article_match = _ARTICLE_BEFORE_TERM.search(prefix)
    if article_match:
        article = article_match.group(1)
        new_article = _correct_article(article, replacement_word)
        return (
            text[: article_match.start()]
            + new_article
            + " "
            + replacement_word
            + text[match.end() :]
        )
    return text[:start] + replacement_word + text[match.end() :]


def _tier_pattern() -> re.Pattern[str]:
    terms = sorted(
        (t for bag in SEVERITY_TIERS for t in bag),
        key=len,
        reverse=True,
    )
    return re.compile(
        r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b",
        re.IGNORECASE,
    )


_TIER_PATTERN = _tier_pattern()


def _pick_tier_replacement(
    matched_token: str,
    target_bag: frozenset[str],
    rng: random.Random,
    options: Optional[SeverityFlipOptions] = None,
) -> Optional[str]:
    pool = _morphology_pool(target_bag, matched_token, options=options)
    if not pool:
        return None
    base = rng.choice(pool)
    return _apply_casing_to_replacement(matched_token, base)


def _swap_scalar_term(
    text: str,
    a: str,
    b: str,
    swap_instances: str,
    rng: random.Random,
) -> str:
    pattern = re.compile(rf"\b({re.escape(a)}|{re.escape(b)})\b", re.IGNORECASE)
    matches = list(pattern.finditer(text))
    if not matches:
        return text

    if swap_instances == "all":
        out = text
        for _ in range(len(matches)):
            fresh = list(pattern.finditer(out))
            if not fresh:
                break
            m = fresh[-1]
            repl = _replacement_for_token(m.group(0), a, b)
            out = _replace_token_span(out, m, repl)
        return out

    if swap_instances == "random_n":
        chosen_count = rng.randint(1, len(matches))
        chosen_indices = set(rng.sample(range(len(matches)), k=chosen_count))
        out = text
        for i in sorted(chosen_indices, reverse=True):
            fresh = list(pattern.finditer(out))
            if i >= len(fresh):
                continue
            m = fresh[i]
            repl = _replacement_for_token(m.group(0), a, b)
            out = _replace_token_span(out, m, repl)
        return out

    if swap_instances in SINGLE_SWAP_INSTANCES:
        match = matches[rng.randrange(len(matches))]
        repl = _replacement_for_token(match.group(0), a, b)
        return _replace_token_span(text, match, repl)

    raise ValueError(
        f"swap_instances must be one of {SWAP_INSTANCES_CHOICES_TRAINING}, got {swap_instances!r}"
    )


def _swap_tier_bag(
    text: str,
    swap_instances: str,
    rng: random.Random,
    *,
    cross_tier: bool = True,
    options: Optional[SeverityFlipOptions] = None,
) -> Optional[str]:
    tier_pattern = _tier_pattern_for(options)
    if tier_pattern is None:
        return None
    matches = list(tier_pattern.finditer(text))
    if not matches:
        return None

    n_tiers = len(SEVERITY_TIERS)

    def replacement_for_match(token: str) -> Optional[str]:
        tier_idx = _severity_tier_index(token)
        if tier_idx is None:
            return None
        if cross_tier:
            opp_choices = _opposite_tier_indices(tier_idx, n_tiers)
            target_idx = rng.choice(opp_choices)
            target_bag = _filtered_tier_bag(SEVERITY_TIERS[target_idx], options)
        else:
            target_bag = _filtered_tier_bag(SEVERITY_TIERS[tier_idx], options)
        if not target_bag:
            return None
        return _pick_tier_replacement(token, target_bag, rng, options=options)

    def _apply_one_tier_swap(out: str, match: re.Match[str]) -> Optional[str]:
        token = match.group(0)
        repl = replacement_for_match(token)
        if repl is None:
            return None
        new_out = _replace_token_span(out, match, repl)
        return new_out if new_out != out else None

    if swap_instances == "all":
        out = text
        for _ in range(len(matches)):
            fresh = list(tier_pattern.finditer(out))
            if not fresh:
                break
            updated = _apply_one_tier_swap(out, fresh[-1])
            if updated is None:
                break
            out = updated
        return out if out != text else None

    if swap_instances == "random_n":
        chosen_indices = set(
            rng.sample(range(len(matches)), k=rng.randint(1, len(matches)))
        )
        out = text
        for i in sorted(chosen_indices, reverse=True):
            fresh = list(tier_pattern.finditer(out))
            if i >= len(fresh):
                continue
            updated = _apply_one_tier_swap(out, fresh[i])
            if updated is not None:
                out = updated
        return out if out != text else None

    if swap_instances in SINGLE_SWAP_INSTANCES:
        eligible = list(matches)
        rng.shuffle(eligible)
        for match in eligible:
            updated = _apply_one_tier_swap(text, match)
            if updated is not None:
                return updated
        return None

    raise ValueError(
        f"swap_instances must be one of {SWAP_INSTANCES_CHOICES_TRAINING}, got {swap_instances!r}"
    )


def _scalar_matching_rules(
    text: str,
    options: Optional[SeverityFlipOptions] = None,
) -> List[Tuple[str, str, str]]:
    if options and options.exclude_scalar:
        return []
    excluded = options.excluded_terms if options else frozenset()
    matching: List[Tuple[str, str, str]] = []
    for left, right, error_type in SEVERITY_SCALAR_RULES:
        if left.lower() in excluded or right.lower() in excluded:
            continue
        pattern = re.compile(
            rf"\b({re.escape(left)}|{re.escape(right)})\b", re.IGNORECASE
        )
        if pattern.search(text):
            matching.append((left, right, error_type))
    return matching


def _apply_scalar_rule(
    text: str,
    left: str,
    right: str,
    swap_instances: str,
    rng: random.Random,
) -> str:
    out = _swap_scalar_term(text, left, right, swap_instances=swap_instances, rng=rng)
    return out if out != text else text


def apply_severity_flip(
    text: str,
    swap_instances: str = "all",
    rng: Optional[random.Random] = None,
    rule_selection: str = "random",
    options: Optional[SeverityFlipOptions] = None,
) -> Optional[str]:
    """Flip severity in text via scalar pair or tier-bag corruption.

    rule_selection:
      - random: uniform over matching scalar rules and tier-bag path (training default)
      - first: first matching scalar rule in list order, else first tier match

  options:
      Training-only filters (exclude_scalar, excluded_terms). Eval generation uses default.
    """
    rng = rng or random
    scalar_rules = _scalar_matching_rules(text, options=options)
    tier_pattern = _tier_pattern_for(options)
    has_tier = tier_pattern.search(text) is not None if tier_pattern else False

    if rule_selection == "first":
        if scalar_rules:
            left, right, _ = scalar_rules[0]
            out = _apply_scalar_rule(text, left, right, swap_instances, rng)
            return out if out != text else None
        if has_tier:
            out = _swap_tier_bag(
                text, swap_instances, rng, cross_tier=True, options=options
            )
            return out if out and out != text else None
        return None

    if rule_selection != "random":
        raise ValueError(f"rule_selection must be 'random' or 'first', got {rule_selection!r}")

    actions: List[Tuple[str, str, str]] = []
    for left, right, _ in scalar_rules:
        actions.append(("scalar", left, right))
    if has_tier:
        actions.append(("tier", "", ""))
    if not actions:
        return None

    kind, left, right = rng.choice(actions)
    if kind == "scalar":
        out = _apply_scalar_rule(text, left, right, swap_instances, rng)
    else:
        out = _swap_tier_bag(
            text, swap_instances, rng, cross_tier=True, options=options
        )
    if out is None or out == text:
        return None
    return out


def apply_severity_jitter(
    text: str,
    swap_instances: str = "all",
    rng: Optional[random.Random] = None,
    options: Optional[SeverityFlipOptions] = None,
) -> Optional[str]:
    """Within-tier synonym swap for consistent (positive) training examples."""
    rng = rng or random
    return _swap_tier_bag(
        text, swap_instances, rng, cross_tier=False, options=options
    )
