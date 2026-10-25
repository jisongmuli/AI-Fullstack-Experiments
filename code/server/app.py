"""FastAPI backend for the AIGC course project.

Run from the project root:
    python -m uvicorn app:app --app-dir code/server --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import importlib.util
import inspect
import io
import json
import logging
import math
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from agent import run_agent
from editing import edit_text
from llm_preview import LLMPreviewService, LLMPreviewUnavailable
from provenance import inspect_provenance
from documents import (DocumentError, MAX_DOCUMENT_BYTES, MAX_DOCUMENT_CHARS,
                       analyze_document, extract_document, extract_pasted_text)
from document_reports import build_report
from report_store import ReportStore
from runtime import ProcessingLimit, ProcessingState


ROOT = Path(__file__).resolve().parents[2]
MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_REQUEST_BYTES = 35 * 1024 * 1024
MAX_TEXT_CHARS = 10_000
Image.MAX_IMAGE_PIXELS = 25_000_000
LOG = logging.getLogger(__name__)


class ModelUnavailable(RuntimeError):
    pass


class RequestSizeLimit:
    def __init__(self, app: Any, max_bytes: int = MAX_REQUEST_BYTES):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope["method"] != "POST" or not scope["path"].startswith("/api/"):
            await self.app(scope, receive, send)
            return

        async def reject() -> None:
            response = JSONResponse(
                {"detail": "Request body exceeds the 35 MB limit."}, status_code=413
            )
            await response(scope, receive, send)

        headers = dict(scope.get("headers", []))
        declared = headers.get(b"content-length")
        if declared is not None:
            try:
                if int(declared) > self.max_bytes:
                    await reject()
                    return
            except ValueError:
                await JSONResponse(
                    {"detail": "Invalid Content-Length header."}, status_code=400
                )(scope, receive, send)
                return

        chunks: list[bytes] = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > self.max_bytes:
                await reject()
                return
            chunks.append(chunk)
            if not message.get("more_body", False):
                break

        body = b"".join(chunks)
        chunks.clear()
        first = True

        async def replay_receive() -> dict:
            nonlocal first
            if first:
                first = False
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay_receive, send)


class ModelService:
    def __init__(self, root: Path = ROOT):
        self.root = root
        self._bundle: Any | None = None
        self._lock = threading.Lock()
        self._fingerprint_lock = threading.Lock()
        self._fingerprints: dict[str, str] | None = None

    def get_bundle(self) -> Any:
        if self._bundle is not None:
            return self._bundle
        with self._lock:
            if self._bundle is not None:
                return self._bundle
            module_path = self.root / "code" / "models" / "inference.py"
            if not module_path.is_file():
                raise ModelUnavailable("Trained inference module is not available yet.")
            try:
                models_dir = str(module_path.parent)
                if models_dir not in sys.path:
                    sys.path.insert(0, models_dir)
                spec = importlib.util.spec_from_file_location("aigc_local_inference", module_path)
                if spec is None or spec.loader is None:
                    raise ImportError("Cannot load inference module.")
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
                if hasattr(module, "load_models"):
                    self._bundle = module.load_models(self.root)
                else:
                    self._bundle = module.DetectorService(self.root)
            except Exception as exc:
                LOG.exception("Could not load local models")
                raise ModelUnavailable(f"Model load failed: {type(exc).__name__}.") from exc
            return self._bundle

    def artifact_fingerprints(self) -> dict[str, str]:
        # Model artifacts are fixed for this service instance. In particular,
        # long documents must not re-read every large image model per paragraph.
        with self._fingerprint_lock:
            if self._fingerprints is None:
                self._fingerprints = self._compute_fingerprints()
            return dict(self._fingerprints)

    def _compute_fingerprints(self) -> dict[str, str]:
        paths = {
            "image": self.root / "models" / "image_detector.ts",
            "highres_image": self.root / "models" / "highres_image" / "model.safetensors",
            "highres_image_vit": self.root / "models" / "highres_nyuad" / "model.safetensors",
            "text": self.root / "models" / "text_detector.joblib",
            "chinese_text": self.root / "models" / "text_detector_zh.joblib",
        }
        result = {}
        for modality, path in paths.items():
            if path.is_file():
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                result[modality] = digest.hexdigest()[:16]
        return result


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)


class AgentRequest(BaseModel):
    text: str | None = Field(default=None, max_length=MAX_TEXT_CHARS)
    image_base64: str | None = None
    edit: bool = True
    llm_preview: bool = False


def _uncertain(reason: str, applicability: str = "outside_training_scope", modality: str | None = None) -> dict:
    return {
        "risk_score": None,
        "decision": "uncertain",
        "evidence": [reason],
        "model_version": None,
        "applicability": applicability,
        "modality": modality,
        "threshold": None,
        "limitations": "输入超出当前可靠评估范围，因此不提供模型分数。",
        "evidence_note": "弃判不代表内容由人或 AI 创作。",
        "disclaimer": "该结果不能证明内容来源或作者身份。",
    }


def _normalize_prediction(raw: dict, modality: str, fingerprint: str | None = None) -> dict:
    if not isinstance(raw, dict):
        raise RuntimeError("Model returned a non-object result.")
    if raw.get("out_of_scope"):
        result = _uncertain(raw.get("scope_reason") or "The model marked this input outside its evaluated range.",
                            applicability=raw.get("applicability") or "outside_training_scope", modality=modality)
        if modality == "text" and "segments" in raw:
            result["segments"] = raw["segments"]
            result["segment_summary"] = raw["segment_summary"]
            result["limitations"] = raw.get("limitations") or result["limitations"]
            result["model_version"] = str(raw.get("model_version") or
                                          (f"sha256:{fingerprint}" if fingerprint else "local-model"))
        return result
    score = raw.get("risk_score")
    decision = raw.get("decision")
    if not isinstance(score, (float, int)) or isinstance(score, bool) or not math.isfinite(score) or not 0 <= score <= 1:
        raise RuntimeError("Model returned an invalid risk score.")
    decision = {"ai": "likely_ai", "human": "likely_human"}.get(decision, decision)
    if decision not in {"likely_ai", "likely_human", "uncertain"}:
        raise RuntimeError("Model returned an invalid decision.")
    evidence = raw.get("evidence", raw.get("signals", []))
    if not isinstance(evidence, list) or not all(isinstance(item, str) for item in evidence):
        raise RuntimeError("Model returned invalid evidence.")
    result = {
        "risk_score": round(float(score), 4),
        "decision": decision,
        "evidence": evidence,
        "model_version": str(raw.get("model_version") or raw.get("version") or
                             (f"sha256:{fingerprint}" if fingerprint else "local-model")),
        "applicability": raw.get("applicability") or "limited_same_domain",
        "modality": modality,
        "threshold": raw.get("threshold"),
        "limitations": raw.get("limitations") or
                       "This score is calibrated on validation data and may not transfer to other sources.",
        "evidence_note": "所列线索是统计特征，不是来源的因果证据。",
        "disclaimer": "筛查分数不能证明内容来源或作者身份。",
    }
    if modality == "text" and "segments" in raw:
        result["segments"] = raw["segments"]
        result["segment_summary"] = raw["segment_summary"]
    if modality == "image" and "model_scores" in raw:
        scores = raw["model_scores"]
        if not isinstance(scores, list) or not scores or any(
            not isinstance(item, dict) or not isinstance(item.get("model"), str)
            or not isinstance(item.get("score"), (int, float)) or isinstance(item.get("score"), bool)
            or not math.isfinite(item["score"]) or not 0 <= item["score"] <= 1
            for item in scores
        ):
            raise RuntimeError("Model returned invalid component scores.")
        result["model_scores"] = scores
        disagreement = raw.get("model_disagreement")
        if not isinstance(disagreement, (int, float)) or isinstance(disagreement, bool) or not math.isfinite(disagreement) or not 0 <= disagreement <= 1:
            raise RuntimeError("Model returned invalid disagreement.")
        result["model_disagreement"] = round(float(disagreement), 4)
        hints = raw.get("metadata_hints", [])
        if not isinstance(hints, list) or not all(isinstance(item, str) for item in hints):
            raise RuntimeError("Model returned invalid metadata hints.")
        result["metadata_hints"] = hints
        result["input_size"] = raw.get("input_size")
        result["model_input_size"] = raw.get("model_input_size")
    if modality == "image" and "local_evidence" in raw:
        evidence_map = raw["local_evidence"]
        if not isinstance(evidence_map, dict):
            raise RuntimeError("Model returned invalid local evidence.")
        rows, columns = evidence_map.get("grid_rows"), evidence_map.get("grid_cols")
        values = evidence_map.get("values")
        deltas = evidence_map.get("raw_deltas")
        if (
            not isinstance(rows, int) or isinstance(rows, bool) or not 1 <= rows <= 8
            or not isinstance(columns, int) or isinstance(columns, bool) or not 1 <= columns <= 8
            or not isinstance(values, list) or len(values) != rows * columns
            or any(not isinstance(value, (int, float)) or isinstance(value, bool)
                   or not math.isfinite(value) or not 0 <= value <= 1 for value in values)
            or not isinstance(deltas, list) or len(deltas) != rows * columns
            or any(not isinstance(delta, (int, float)) or isinstance(delta, bool)
                   or not math.isfinite(delta) or not -1 <= delta <= 1 for delta in deltas)
            or evidence_map.get("method") != "blur_occlusion_3x3"
            or not isinstance(evidence_map.get("meaning"), str)
        ):
            raise RuntimeError("Model returned invalid local evidence.")
        result["local_evidence"] = {
            "grid_rows": rows,
            "grid_cols": columns,
            "values": [round(float(value), 4) for value in values],
            "raw_deltas": [round(float(delta), 6) for delta in deltas],
            "method": evidence_map["method"],
            "meaning": evidence_map["meaning"][:500],
        }
        analysis_size = evidence_map.get("analysis_size")
        if analysis_size is not None:
            if (not isinstance(analysis_size, list) or len(analysis_size) != 2
                    or any(not isinstance(side, int) or isinstance(side, bool)
                           or not 1 <= side <= 512 for side in analysis_size)):
                raise RuntimeError("Model returned invalid local evidence size.")
            result["local_evidence"]["analysis_size"] = analysis_size
        analysis_baseline = evidence_map.get("analysis_baseline_score")
        if analysis_baseline is not None:
            if (not isinstance(analysis_baseline, (int, float)) or isinstance(analysis_baseline, bool)
                    or not math.isfinite(analysis_baseline) or not 0 <= analysis_baseline <= 1):
                raise RuntimeError("Model returned invalid local evidence baseline.")
            result["local_evidence"]["analysis_baseline_score"] = round(float(analysis_baseline), 6)
    return result


def _decode_image(data: bytes) -> Image.Image:
    if not data:
        raise HTTPException(status_code=400, detail="Image is empty.")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 25 MB limit.")
    try:
        image = Image.open(io.BytesIO(data))
        if image.width * image.height > Image.MAX_IMAGE_PIXELS:
            raise HTTPException(status_code=413, detail="Image dimensions are too large.")
        image.load()
        return image
    except Image.DecompressionBombError as exc:
        raise HTTPException(status_code=413, detail="Image dimensions are too large.") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Unsupported or corrupt image.") from exc


def _decode_base64_image(encoded: str) -> bytes:
    if encoded.startswith("data:"):
        header, sep, payload = encoded.partition(",")
        if not sep or ";base64" not in header or not header.startswith("data:image/"):
            raise HTTPException(status_code=400, detail="Invalid image data URL.")
        encoded = payload
    if len(encoded) > (MAX_IMAGE_BYTES * 4 // 3) + 8:
        raise HTTPException(status_code=413, detail="Image exceeds the 25 MB limit.")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 image.") from exc
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 25 MB limit.")
    return data


def create_app(model_service: Any | None = None, llm_service: Any | None = None,
               *, max_active_requests: int = 2) -> FastAPI:
    report_store = ReportStore()
    processing = ProcessingState(max_active_requests)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        try:
            yield
        finally:
            report_store.close()

    api = FastAPI(
        title="AIGC Content Screening and Trustworthy Editing",
        version="0.1.0",
        description="Local course prototype. Results are screening signals, not authorship determinations.",
        lifespan=lifespan,
    )
    api.add_middleware(RequestSizeLimit)
    api.add_middleware(ProcessingLimit, state=processing)
    api.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])
    api.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )
    web_dir = ROOT / "web"
    if web_dir.is_dir():
        api.mount("/static", StaticFiles(directory=web_dir), name="static")

        @api.get("/", include_in_schema=False)
        def home() -> FileResponse:
            return FileResponse(web_dir / "index.html", headers={"Cache-Control": "no-store, max-age=0"})
    service = model_service or ModelService()
    llm = llm_service or LLMPreviewService(ROOT)
    api.state.model_service = service
    api.state.llm_service = llm
    api.state.report_store = report_store
    api.state.processing = processing

    def get_bundle() -> Any:
        try:
            return service.get_bundle()
        except ModelUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    def fingerprint(modality: str) -> str | None:
        if hasattr(service, "artifact_fingerprints"):
            return service.artifact_fingerprints().get(modality)
        return None

    def detect_text(text: str) -> dict:
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text is empty.")
        if len(text) > MAX_TEXT_CHARS:
            raise HTTPException(status_code=413, detail="Text exceeds the 10,000-character limit.")
        han = sum("\u4e00" <= char <= "\u9fff" for char in text)
        visible = sum(not char.isspace() for char in text)
        chinese = han >= 20 and han / max(visible, 1) >= 0.25
        letters = [char for char in text if char.isalpha()]
        ascii_letters = sum(char.isascii() for char in letters)
        try:
            bundle = get_bundle()
            if hasattr(bundle, "predict_text_segments"):
                raw = bundle.predict_text_segments(text)
                return _normalize_prediction(raw, "text", fingerprint("chinese_text" if chinese else "text"))
            if chinese:
                if len(text) < 80 or len(text) > 4000 or han < 40:
                    return _uncertain("中文检测当前只接受 80 至 4000 字符、至少 40 个汉字的文本。", modality="text")
            elif len(text) < 23 or len(text) > 587 or len(letters) < 10 or ascii_letters / max(len(letters), 1) < 0.8:
                return _uncertain("The text is outside the short English essay range evaluated in training.", modality="text")
            if chinese:
                if not hasattr(bundle, "predict_text_chinese"):
                    raise ModelUnavailable("Chinese text model is not installed.")
                raw = bundle.predict_text_chinese(text)
            else:
                raw = bundle.predict_text(text)
            return _normalize_prediction(raw, "text", fingerprint("chinese_text" if chinese else "text"))
        except HTTPException:
            raise
        except Exception as exc:
            LOG.exception("Text prediction failed")
            raise HTTPException(status_code=503, detail=f"Text model unavailable: {type(exc).__name__}.") from exc

    def detect_image(data: bytes) -> dict:
        image = _decode_image(data)
        if getattr(image, "n_frames", 1) != 1:
            return _uncertain("Animated images were not evaluated in training.", modality="image")
        if image.size == (32, 32) and image.mode != "RGB":
            return _uncertain("The image model was evaluated on RGB images, not this color mode.", modality="image")
        try:
            predictor = get_bundle().predict_image
            supports_local_evidence = "include_local_evidence" in inspect.signature(predictor).parameters
            raw = predictor(image, include_local_evidence=True) if supports_local_evidence else predictor(image)
            result = _normalize_prediction(raw, "image",
                                           fingerprint("image" if image.size == (32, 32) else "highres_image"))
            result["provenance"] = inspect_provenance(data, image.format)
            return result
        except HTTPException:
            raise
        except Exception as exc:
            LOG.exception("Image prediction failed")
            raise HTTPException(status_code=503, detail=f"Image model unavailable: {type(exc).__name__}.") from exc

    def detect_document(source: dict) -> dict:
        try:
            result = analyze_document(source, detect_text)
            return report_store.put(result)
        except DocumentError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            # Do not include the uploaded document, filename or extracted text in logs.
            LOG.error("Document analysis failed (%s)", type(exc).__name__)
            raise HTTPException(status_code=503, detail="文档分析未完成，请检查模型状态后重试。") from exc

    async def read_document(file: UploadFile) -> dict:
        data = await file.read(MAX_DOCUMENT_BYTES + 1)
        try:
            return await run_in_threadpool(extract_document, data, file.filename or "")
        except DocumentError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    def retrieve_report(report_id: str) -> dict:
        report = report_store.get(report_id)
        if report is None:
            raise HTTPException(status_code=404, detail="报告已过期、已删除或服务已重启，请重新检测。")
        return report

    @api.get("/health")
    async def health() -> dict:
        return {"status": "ok", "service": "aigc-local-api"}

    @api.get("/api/runtime")
    async def runtime() -> Response:
        return JSONResponse({"status": "ok", "processing": processing.snapshot(),
                             "reports": report_store.stats()},
                            headers={"Cache-Control": "no-store"})

    @api.get("/api/models")
    def models() -> dict:
        manifest_path = ROOT / "models" / "manifest.json"
        if not manifest_path.is_file():
            manifest_path = ROOT / "models" / "model_config.json"
        manifest = None
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                manifest = None
        try:
            service.get_bundle()
            available, reason = True, None
        except Exception as exc:
            available, reason = False, str(exc)
        return {
            "available": available,
            "reason": reason,
            "manifest": manifest,
            "artifact_fingerprints": service.artifact_fingerprints() if hasattr(service, "artifact_fingerprints") else {},
            "scope": {
                "image": "32×32 CIFAKE 教学模型；其他短边至少 200 像素的静态图片采用 SigLIP 与 ViT 交叉检查，显示未经平台校准的候选分数、分歧和九宫格遮挡敏感度。短边 200–511 像素尤其属于低分辨率实验性输出，低于 200 像素弃判；颜色不是局部 AI 来源真值，普通图的来源结论仍未校准。",
                "local_evidence": "九宫格逐格模糊后比较双模型均分下降；无明显正向下降时不绘制热图。敏感度不是区域 AI 率或局部篡改真值。",
                "provenance": "JPG/PNG 原始字节可进行本地 C2PA 内容凭证核验；无凭证不表示 AI 生成，有效但不受信任的声明也不能当作来源证明。",
                "text": "最多 10,000 字符的中文或英文文本可逐段筛查；整篇超出已评估长度或段落信号明显冲突时弃判。高亮比例不是 AI 撰写比例，真实平台迁移效果尚未验证。",
                "document": "DOCX、含文本层的 PDF、TXT 和 Markdown 可提取并检测，最多 20 MB、100,000 字符，PDF 最多 200 页。不执行 OCR。下划线表示高风险模型段落，AI 特征覆盖率是高亮字符覆盖率，不是真实 AI 撰写比例。报告保存在本地内存中，最多 30 分钟；可导出 Word、HTML 和 JSON。",
                "editing": "基于规则的清晰度建议和原意表面检查。",
                "llm_preview": "可选的英文 LoRA 改写预览；未通过表面检查时拒绝候选。",
            },
            "llm_preview_available": llm.available(),
        }

    @api.post("/api/image/detect")
    async def image_detect(file: UploadFile = File(...)) -> dict:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=415, detail="Upload an image file.")
        data = await file.read(MAX_IMAGE_BYTES + 1)
        return await run_in_threadpool(detect_image, data)

    @api.post("/api/text/detect")
    def text_detect(body: TextRequest) -> dict:
        return detect_text(body.text)

    @api.post("/api/document/detect")
    async def document_detect(file: UploadFile | None = File(None), text: str = Form("")) -> Response:
        if file is not None and text.strip():
            raise HTTPException(status_code=400, detail="请选择上传文件或粘贴文本中的一种方式。")
        if file is not None:
            source = await read_document(file)
        else:
            try:
                source = extract_pasted_text(text)
            except DocumentError as exc:
                raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        report = await run_in_threadpool(detect_document, source)
        return JSONResponse(report, headers={"Cache-Control": "no-store"})

    @api.get("/api/document/reports")
    def document_report_list() -> Response:
        return JSONResponse({"reports": report_store.list_reports()},
                            headers={"Cache-Control": "no-store"})

    @api.get("/api/document/reports/{report_id}")
    def document_report(report_id: str) -> Response:
        return JSONResponse(retrieve_report(report_id), headers={"Cache-Control": "no-store"})

    @api.delete("/api/document/reports/{report_id}")
    def delete_document_report(report_id: str) -> Response:
        return JSONResponse({"deleted": report_store.delete(report_id)}, headers={"Cache-Control": "no-store"})

    @api.get("/api/document/reports/{report_id}/{report_format}")
    def export_document_report(report_id: str, report_format: str) -> Response:
        if report_format not in {"docx", "html", "json"}:
            raise HTTPException(status_code=400, detail="报告格式支持 docx、html 或 json。HTML 可在浏览器中打印为 PDF。")
        report = retrieve_report(report_id)
        try:
            content, media_type, filename = build_report(report, report_format)
        except Exception as exc:
            LOG.error("Document export failed (%s)", type(exc).__name__)
            raise HTTPException(status_code=503, detail="报告导出失败，请检查文档导出依赖。") from exc
        return Response(content, media_type=media_type, headers={
            "Content-Disposition": "attachment; filename=\"aigc-report." + report_format + "\"; filename*=UTF-8''" + quote(filename, safe=""),
            "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'",
        })

    @api.post("/api/text/edit")
    def text_edit(body: TextRequest) -> dict:
        try:
            return edit_text(body.text)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.post("/api/text/edit/llm-preview")
    def llm_text_edit(body: TextRequest) -> dict:
        try:
            return llm.suggest(body.text)
        except LLMPreviewUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            LOG.exception("LLM preview failed")
            raise HTTPException(status_code=503, detail=f"LLM preview unavailable: {type(exc).__name__}.") from exc

    @api.post("/api/agent/run")
    def agent_run(body: AgentRequest) -> dict:
        if (body.text is None or not body.text.strip()) and body.image_base64 is None:
            raise HTTPException(status_code=400, detail="Supply text or image_base64.")
        image_data = _decode_base64_image(body.image_base64) if body.image_base64 is not None else None
        if image_data is not None:
            _decode_image(image_data)
        return run_agent(
            text=body.text,
            image_bytes=image_data,
            request_edit=body.edit,
            detect_text=detect_text,
            detect_image=detect_image,
            edit_text=edit_text,
            request_llm_preview=body.llm_preview,
            suggest_llm=llm.suggest if body.llm_preview else None,
        )

    @api.post("/api/agent/upload")
    async def agent_upload(
        text: str = Form(""),
        file: UploadFile | None = File(None),
        document: UploadFile | None = File(None),
        edit: bool = Form(False),
        llm_preview: bool = Form(False),
    ) -> dict:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        if len(text) > MAX_TEXT_CHARS:
            raise HTTPException(status_code=413, detail="Text exceeds the 10,000-character limit.")
        if text.strip() and document is not None:
            raise HTTPException(status_code=400, detail="文档和粘贴文本请分别提交；文档可与图片一起分析。")
        if not text.strip() and file is None and document is None:
            raise HTTPException(status_code=400, detail="Supply text, an image file, or a document.")
        image_data = None
        if file is not None:
            if file.content_type not in {"image/jpeg", "image/png"}:
                raise HTTPException(status_code=415, detail="Upload a JPG or PNG image.")
            image_data = await file.read(MAX_IMAGE_BYTES + 1)
            await run_in_threadpool(_decode_image, image_data)
        source = await read_document(document) if document is not None else None
        result = await run_in_threadpool(run_agent,
            text=text or None,
            image_bytes=image_data,
            request_edit=edit,
            detect_text=detect_text,
            detect_image=detect_image,
            edit_text=edit_text,
            request_llm_preview=llm_preview,
            suggest_llm=llm.suggest if llm_preview else None,
        )
        if source is not None:
            started = time.perf_counter()
            report = await run_in_threadpool(detect_document, source)
            result["findings"]["document"] = report
            result["tool_calls"].append({"tool": "document_detection", "status": "ok",
                                          "input_summary": f"{len(source['text'])} extracted characters",
                                          "duration_ms": round((time.perf_counter() - started) * 1000, 1)})
            result["decision"] = "screening_complete"
            result["explanation"] = "文档和图像分别完成检测；文档提供模型分数、覆盖率及下划线报告，不合成来源概率。"
            if edit or llm_preview:
                result["limitations"].append("文档未执行整篇改写；请在可信编辑页选择具体段落。")
        return JSONResponse(result, headers={"Cache-Control": "no-store"}) if document is not None else result

    return api


app = create_app()
