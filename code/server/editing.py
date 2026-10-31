"""Conservative editorial suggestions and transparent surface-level meaning checks."""
from __future__ import annotations

import re
from collections import Counter


REPLACEMENTS = (
    (r"\bin order to\b", "to", "wordiness"),
    (r"\bdue to the fact that\b", "because", "wordiness"),
    (r"\bat this point in time\b", "now", "wordiness"),
    (r"\bthe\s+the\b", "the", "repetition"),
    (r"\bvery\s+very\b", "very", "repetition"),
)
VAGUE = re.compile(r"\b(very important|many things|a lot of things|good results|bad results)\b", re.I)
LOGIC = re.compile(r"\b(therefore|thus|consequently|clearly)\b", re.I)
TOKEN = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?|[\u4e00-\u9fff]")
NUMBER = re.compile(
    r"(?<![A-Za-z0-9_])[-+−]?(?:\d+(?:[,.]\d+)*|\.\d+)"
    r"(?:[eE][+-]?\d+)?(?:[ \t]*[%％‰])?(?![A-Za-z0-9_])"
)
NEGATION = re.compile(r"\b(?:no|not|never|without|cannot|can't|won't|doesn't|don't|isn't|aren't)\b|[不无未非没]", re.I)
QUOTED = re.compile(r"[\"“]([^\"”]+)[\"”]")
ACRONYM = re.compile(r"\b[A-Z]{2,}\b")


def _tokens(text: str) -> Counter[str]:
    return Counter(token.lower() for token in TOKEN.findall(text))


def _numbers(text: str) -> Counter[str]:
    # Whitespace before a unit and equivalent Unicode signs are formatting.
    # Keep signs, decimal points, exponents and units in the compared value.
    return Counter(re.sub(r"[ \t]", "", value).replace("−", "-").replace("％", "%")
                   for value in NUMBER.findall(text))


def meaning_check(original: str, edited: str) -> dict:
    """Guard obvious drift; passing this check never proves semantic equivalence."""
    before, after = _tokens(original), _tokens(edited)
    shared = sum((before & after).values())
    total = sum((before | after).values())
    overlap = shared / total if total else 1.0
    checks = {
        "numbers_preserved": _numbers(original) == _numbers(edited),
        "negations_preserved": Counter(x.lower() for x in NEGATION.findall(original)) == Counter(
            x.lower() for x in NEGATION.findall(edited)
        ),
        "quoted_text_preserved": Counter(QUOTED.findall(original)) == Counter(QUOTED.findall(edited)),
        "acronyms_preserved": Counter(ACRONYM.findall(original)) == Counter(ACRONYM.findall(edited)),
        "lexical_overlap_at_least_0_65": overlap >= 0.65,
    }
    passed = all(checks.values())
    return {
        "status": "surface_checks_passed_human_review_required" if passed else "blocked",
        "checks": checks,
        "lexical_overlap": round(overlap, 3),
        "limitation": "These are surface checks, not proof that the meaning or factual claims are unchanged.",
    }


def edit_text(text: str) -> dict:
    if len(text) > 10_000:
        raise ValueError("Text exceeds the 10,000-character editing limit.")
    if not text.strip():
        raise ValueError("Text is empty.")

    candidate = text
    edits = []
    for pattern, replacement, category in REPLACEMENTS:
        matches = list(re.finditer(pattern, candidate, re.I))
        if not matches:
            continue
        edits.extend(
            {"category": category, "original": match.group(), "replacement": replacement,
             "reason": "Removes a repeated or unnecessarily long expression."}
            for match in matches
        )
        candidate = re.sub(
            pattern,
            lambda match: replacement.capitalize() if match.group()[0].isupper() else replacement,
            candidate,
            flags=re.I,
        )

    check = meaning_check(text, candidate)
    rejected_edits = []
    if check["status"] == "blocked":
        rejected_edits = edits
        candidate = text
        edits = []

    suggestions = []
    for match in VAGUE.finditer(text):
        suggestions.append({
            "category": "specificity", "excerpt": match.group(),
            "advice": "Replace this phrase with a concrete observation, quantity, or example that you can verify.",
            "auto_applied": False,
        })
    for match in LOGIC.finditer(text):
        suggestions.append({
            "category": "reasoning", "excerpt": match.group(),
            "advice": "Check that the preceding evidence actually supports this conclusion.",
            "auto_applied": False,
        })
    sentences = [s.strip() for s in re.split(r"(?<=[.!?。！？])\s+", text) if s.strip()]
    seen = set()
    for sentence in sentences:
        key = re.sub(r"\s+", " ", sentence.casefold())
        if len(key) >= 20 and key in seen:
            suggestions.append({
                "category": "repetition", "excerpt": sentence[:120],
                "advice": "This sentence repeats an earlier sentence; check whether both are needed.",
                "auto_applied": False,
            })
        seen.add(key)

    return {
        "original_text": text,
        "edited_text": candidate,
        "applied_edits": edits,
        "rejected_edits": rejected_edits,
        "suggestions": suggestions,
        "meaning_check": check,
        "editor_type": "deterministic_rule_based",
        "note": "This tool improves clarity. It does not certify authorship or remove an AI-detection signal.",
    }
