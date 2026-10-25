"""A local, auditable workflow that invokes detection and editing tools."""
from __future__ import annotations

import time
from typing import Callable


def run_agent(
    *,
    text: str | None,
    image_bytes: bytes | None,
    request_edit: bool,
    detect_text: Callable[[str], dict],
    detect_image: Callable[[bytes], dict],
    edit_text: Callable[[str], dict],
    request_llm_preview: bool = False,
    suggest_llm: Callable[[str], dict] | None = None,
) -> dict:
    calls: list[dict] = []
    findings: dict[str, dict] = {}
    editing_result = None
    llm_preview = None

    def invoke(name: str, summary: str, action: Callable[[], dict]) -> dict:
        start = time.perf_counter()
        try:
            result = action()
            calls.append({"tool": name, "status": "ok", "input_summary": summary,
                          "duration_ms": round((time.perf_counter() - start) * 1000, 1)})
            return result
        except Exception as exc:
            calls.append({"tool": name, "status": "unavailable", "input_summary": summary,
                          "duration_ms": round((time.perf_counter() - start) * 1000, 1),
                          "error_type": type(exc).__name__})
            return {"decision": "uncertain", "risk_score": None,
                    "evidence": ["The tool could not return a usable result."],
                    "model_version": None, "applicability": "unavailable"}

    if image_bytes is not None:
        findings["image"] = invoke("image_detection", f"{len(image_bytes)} image bytes", lambda: detect_image(image_bytes))
    if text is not None and text.strip():
        findings["text"] = invoke("text_detection", f"{len(text)} text characters", lambda: detect_text(text))
        if request_edit:
            editing_result = invoke("clarity_edit", f"{len(text)} text characters", lambda: edit_text(text))
        if request_llm_preview and suggest_llm is not None:
            llm_preview = invoke("llm_edit_preview", f"{len(text)} text characters", lambda: suggest_llm(text))

    signals = [item["decision"] for item in findings.values()]
    if not findings:
        decision = "no_input"
        explanation = "No image or nonempty text was supplied."
    elif any(signal == "uncertain" for signal in signals):
        decision = "insufficient_evidence"
        explanation = "At least one detection tool abstained or was unavailable; manual review is needed."
    elif len(set(signals)) > 1:
        decision = "mixed_signals"
        explanation = "The image and text tools gave different results; there is no validated cross-modal fusion model."
    else:
        decision = "review_recommended"
        explanation = "Model outputs are screening signals only and cannot establish who authored the content."

    return {
        "workflow": "deterministic_local_agent_v1",
        "decision": decision,
        "explanation": explanation,
        "findings": findings,
        "editing_result": editing_result,
        "llm_preview": llm_preview,
        "tool_calls": calls,
        "limitations": [
            "Image and text datasets are unpaired; their scores are not combined.",
            "A detection score does not establish authorship or provenance.",
            "Editing meaning checks are surface safeguards and require human review.",
        ],
    }
