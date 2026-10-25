"""Read C2PA Content Credentials without turning claims into model probabilities.

The uploaded bytes stay in memory. Remote manifests and OCSP lookups are disabled,
and the API never returns the full manifest, signer identity, thumbnails, or
arbitrary user-authored assertion values.
"""

from __future__ import annotations

import io
import json
from typing import Any


SOURCE_TYPES = {
    "digitalCapture": "数字设备采集",
    "digitalCreation": "数字创作",
    "algorithmicMedia": "算法生成或处理",
    "trainedAlgorithmicMedia": "训练模型生成",
    "compositeWithTrainedAlgorithmicMedia": "包含训练模型生成内容的合成作品",
}
ACTION_TYPES = {
    "c2pa.created": "创建",
    "c2pa.opened": "打开",
    "c2pa.edited": "编辑",
    "c2pa.placed": "放置素材",
    "c2pa.deleted": "删除内容",
    "c2pa.converted": "格式转换",
}


def _safe_claims(manifest: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return only recognized vocabulary, not arbitrary signed metadata values."""
    actions: list[str] = []
    sources: list[str] = []
    for assertion in manifest.get("assertions", []):
        if not isinstance(assertion, dict) or assertion.get("label") not in {"c2pa.actions", "c2pa.actions.v2"}:
            continue
        payload = assertion.get("data")
        if not isinstance(payload, dict):
            continue
        for item in payload.get("actions", [])[:30]:
            if not isinstance(item, dict):
                continue
            action = item.get("action")
            if action in ACTION_TYPES and ACTION_TYPES[action] not in actions:
                actions.append(ACTION_TYPES[action])
            source = item.get("digitalSourceType")
            if isinstance(source, str) and source.startswith((
                "http://cv.iptc.org/newscodes/digitalsourcetype/",
                "https://cv.iptc.org/newscodes/digitalsourcetype/",
            )):
                name = source.rsplit("/", 1)[-1]
                if name in SOURCE_TYPES and SOURCE_TYPES[name] not in sources:
                    sources.append(SOURCE_TYPES[name])
    return actions, sources


def _failure_codes(results: Any) -> list[str]:
    if not isinstance(results, dict):
        return []
    active = results.get("activeManifest")
    if not isinstance(active, dict):
        return []
    codes = []
    for entry in active.get("failure", [])[:20]:
        code = entry.get("code") if isinstance(entry, dict) else None
        if isinstance(code, str) and len(code) <= 80 and code.replace(".", "").replace("_", "").isalnum():
            codes.append(code)
    return codes


def inspect_provenance(data: bytes, image_format: str | None) -> dict[str, Any]:
    """Inspect original JPG/PNG bytes with the official C2PA verifier, locally."""
    mime = {"JPEG": "image/jpeg", "PNG": "image/png"}.get((image_format or "").upper())
    if mime is None:
        return {"status": "not_checked", "explanation": "当前只核验 JPG/PNG 的内容凭证。"}
    try:
        import c2pa
    except ImportError:
        return {"status": "unavailable", "explanation": "未安装 C2PA 核验组件。"}

    try:
        # Do not fetch remote manifests or query third-party OCSP endpoints with user uploads.
        with c2pa.Context.from_dict({"verify": {"remote_manifest_fetch": False, "ocsp_fetch": False}}) as context:
            with c2pa.Reader(mime, io.BytesIO(data), context=context) as reader:
                state = (reader.get_validation_state() or "").lower()
                store = json.loads(reader.json())
                active_id = store.get("active_manifest")
                active = store.get("manifests", {}).get(active_id, {})
                if not isinstance(active, dict):
                    active = {}
                actions, sources = _safe_claims(active)
                failures = _failure_codes(reader.get_validation_results())
    except c2pa.C2paError.ManifestNotFound:
        return {
            "status": "credential_absent",
            "explanation": "未发现嵌入式内容凭证；这不能说明图片是 AI 生成的。",
        }
    except (c2pa.C2paError, ValueError, TypeError, KeyError):
        return {
            "status": "invalid_or_unreadable",
            "explanation": "内容凭证无法可靠核验；不能据此判断来源。",
        }

    status = {"trusted": "trusted", "valid": "valid_but_untrusted", "invalid": "invalid"}.get(
        state, "unverified"
    )
    result: dict[str, Any] = {
        "status": status,
        "validation_state": state or None,
        "local_only": True,
        "revocation_checked_online": False,
        "failure_codes": failures,
        "explanation": {
            "trusted": "本地核验显示凭证受信任；声明仍需结合签发者和实际用途审查，未在线查询撤销状态。",
            "valid_but_untrusted": "签名和内容绑定通过本地核验，但签发者未建立信任；声明不能视为来源证明。",
            "invalid": "凭证核验失败；其中的来源声明不可采信。",
            "unverified": "凭证状态不明确；不能据此判断来源。",
        }[status],
    }
    if status in {"trusted", "valid_but_untrusted"}:
        result["declared_actions"] = actions
        result["declared_source_types"] = sources
    return result
