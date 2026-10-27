"""LLM-selected, allowlisted API tools. Uploaded content remains in this request."""
from __future__ import annotations
import json
import threading
import time
from pathlib import Path

import httpx
from starlette.concurrency import run_in_threadpool


class AgentError(Exception):
    def __init__(self, message, status_code=503):
        super().__init__(message)
        self.status_code = status_code


class CourseAgent:
    def __init__(self, root, model=None):
        import sys
        directory = str(Path(root) / "code/llm")
        if directory not in sys.path:
            sys.path.insert(0, directory)
        from agent_model import LocalAgentModel
        self.model = model or LocalAgentModel(root)
        self.admission = threading.BoundedSemaphore(1)
        self.active = threading.Event()

    async def run(self, application, instruction, *, text="", image=None, document=None, history=(), profile="advanced"):
        if not self.admission.acquire(blocking=False):
            raise AgentError("助手正在处理上一项请求，请稍后再试。")
        self.active.set()
        try:
            status = self.model.status()
            if not status["available"]:
                raise AgentError("助手模型尚未准备好，请先完成模型下载与路由微调。")
            available = {"text": bool(text.strip()), "image": image is not None, "document": document is not None}
            try:
                plan = await run_in_threadpool(self.model.plan, instruction, available, history)
            except ValueError as exc:
                raise AgentError(str(exc), 400) from exc
            except Exception as exc:
                raise AgentError("本地助手模型未完成工具选择，请检查模型文件后重试。") from exc
            allowed = {"model_info", "course_overview", "text_detection", "image_detection", "document_detection", "clarity_edit"}
            tools = plan.get("tools")
            if not isinstance(tools, list) or len(tools) > 2 or len(set(tools)) != len(tools) or any(tool not in allowed for tool in tools):
                raise AgentError("助手返回了无效的工具计划，请重试。")
            needs = {"text_detection": "text", "clarity_edit": "text", "image_detection": "image", "document_detection": "document"}
            if any(tool in needs and not available[needs[tool]] for tool in tools):
                raise AgentError("工具所需的正文或附件尚未提供。", 400)
            calls = []
            # ASGITransport issues real requests through the same routing,
            # decoding, limits and report store as the browser's API calls.
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url="http://localhost", timeout=180) as client:
                for tool in tools:
                    started = time.perf_counter()
                    prefix = "/api/course" if profile == "course" else "/api"
                    if tool in {"model_info", "course_overview"}:
                        path = "/api/models" if tool == "model_info" else "/api/course"
                        response = await client.get(path)
                    elif tool == "text_detection":
                        path = prefix + "/text/detect"
                        response = await client.post(path, json={"text": text})
                    elif tool == "image_detection":
                        path = prefix + "/image/detect"
                        response = await client.post(path, files={"file": image})
                    elif tool == "document_detection":
                        path = "/api/document/detect"
                        response = await client.post(path, files={"file": document})
                    else:
                        path = "/api/text/edit"
                        response = await client.post(path, json={"text": text})
                    result = response.json()
                    if tool == "document_detection" and response.is_success:
                        result = {key: result.get(key) for key in ("report_id", "filename", "created_at", "expires_at", "document_summary", "model_versions")}
                    summary = summarize_result(tool, result, response.is_success)
                    calls.append({"tool": tool, "path": path, "http_status": response.status_code,
                                  "status": "ok" if response.is_success else "failed", "duration_ms": round((time.perf_counter()-started)*1000, 2),
                                  "summary": summary, "result": result})
            if not calls:
                answer, source = "请在正文框提供待分析文本，或上传图片、文档，并说明希望完成的操作。", "missing_input"
            else:
                try:
                    answer = await run_in_threadpool(self.model.summarize, instruction, calls)
                    if not isinstance(answer, str) or not answer.strip():
                        raise ValueError("Empty explanation")
                    source = "local_llm"
                except Exception:
                    answer, source = "处理结果已列在工具记录中，可以查看各项结果或打开文档报告。", "tool_results"
            return {"version": "2026-10-27", "planner": plan, "profile": profile, "tool_calls": calls,
                    "answer": answer, "answer_source": source, "model": status["model"]}
        finally:
            self.active.clear()
            self.admission.release()


def summarize_result(tool, result, success):
    if not success:
        return str(result.get("detail", "工具未完成，请检查输入后重试。"))[:300]
    if tool in {"text_detection", "image_detection"}:
        score = result.get("risk_score")
        if isinstance(score, (int, float)):
            return f"模型分数 {score*100:.1f}/100；状态 {result.get('decision', 'uncertain')}；模型 {result.get('model_version', '')}"
        return str(result.get("scope_reason") or "当前内容未评分，请查看段落或输入范围。")
    if tool == "document_detection":
        summary = result.get("document_summary") or {}
        return f"文档报告已生成；可评分覆盖率 {summary.get('scored_character_percent', '—')}%；报告可导出 Word、HTML、JSON。"
    if tool == "clarity_edit":
        return "编辑建议已生成，请查看修改后的英文与原意检查。"
    if tool == "course_overview":
        return "；".join(row["name"] + ("已就绪" if row["ready"] else "待准备") for row in result.get("stages", []))
    scope = result.get("scope") or {}
    return "图像：" + str(scope.get("image", ""))[:160] + "；文本：" + str(scope.get("text", ""))[:160]
