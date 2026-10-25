"""Service responsiveness, admission control and report lifecycle regressions."""
from __future__ import annotations

import asyncio
import io
import json
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import MAX_REQUEST_BYTES, ModelUnavailable, create_app  # noqa: E402
from report_store import ReportStore  # noqa: E402
from runtime import ProcessingLimit, ProcessingState  # noqa: E402


def png_bytes():
    stream = io.BytesIO()
    Image.new("RGB", (32, 32)).save(stream, "PNG")
    return stream.getvalue()


class ControlledModel:
    def __init__(self):
        self.started = threading.Event()
        self.finish = threading.Event()

    def get_bundle(self):
        return self

    def predict_image(self, image):
        self.started.set()
        if not self.finish.wait(5):
            raise TimeoutError("Test worker was not released")
        return {"risk_score": .3, "decision": "uncertain", "model_version": "test"}


class RuntimeConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.model = ControlledModel()
        self.app = create_app(self.model, max_active_requests=1)
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://localhost")
        self.pending = []

    async def asyncTearDown(self):
        self.model.finish.set()
        if self.pending:
            await asyncio.wait_for(asyncio.gather(*self.pending, return_exceptions=True), 6)
        await self.client.aclose()
        self.app.state.report_store.close()

    async def start_image(self):
        task = asyncio.create_task(self.client.post(
            "/api/image/detect", files={"file": ("sample.png", png_bytes(), "image/png")}))
        self.pending.append(task)
        self.assertTrue(await asyncio.to_thread(self.model.started.wait, 2))
        self.assertFalse(task.done(), "Inference must still be running during the probe")
        return task

    async def test_health_and_runtime_remain_responsive_during_image_inference(self):
        task = await self.start_image()
        health = await asyncio.wait_for(self.client.get("/health"), .8)
        runtime = await asyncio.wait_for(self.client.get("/api/runtime"), .8)
        self.assertEqual(health.json()["status"], "ok")
        self.assertEqual(runtime.json()["processing"]["active"], 1)
        self.assertIn("no-store", runtime.headers["cache-control"])
        self.model.finish.set()
        self.assertEqual((await task).status_code, 200)
        final = (await self.client.get("/api/runtime")).json()["processing"]
        self.assertEqual((final["active"], final["completed"]), (0, 1))

    async def test_busy_request_is_rejected_then_capacity_recovers(self):
        task = await self.start_image()
        busy = await asyncio.wait_for(self.client.post(
            "/api/text/detect", json={"text": "A second analysis."}), .8)
        self.assertEqual(busy.status_code, 503)
        self.assertEqual(busy.headers["retry-after"], "2")
        self.model.finish.set()
        self.assertEqual((await task).status_code, 200)
        response = await self.client.post("/api/text/edit", json={
            "text": "In order to verify the result, the team checked 20 samples."})
        self.assertEqual(response.status_code, 200)
        stats = self.app.state.processing.snapshot()
        self.assertEqual((stats["active"], stats["completed"], stats["rejected"]), (0, 2, 1))

    async def test_busy_gate_does_not_read_upload_body(self):
        state = ProcessingState(1)
        state.acquire()
        received = []
        sent = []

        async def downstream(scope, receive, send):
            self.fail("Busy requests must not enter the application")

        async def receive():
            received.append(True)
            raise AssertionError("Upload body was read while busy")

        async def send(message):
            sent.append(message)

        await ProcessingLimit(downstream, state)(
            {"type": "http", "method": "POST", "path": "/api/agent/upload"}, receive, send)
        self.assertFalse(received)
        self.assertEqual(sent[0]["status"], 503)
        self.assertEqual(state.snapshot()["active"], 1)
        state.release()

    async def test_unexpected_exception_releases_admission_slot(self):
        state = ProcessingState(1)

        async def broken(scope, receive, send):
            raise RuntimeError("Worker failed")

        with self.assertRaises(RuntimeError):
            await ProcessingLimit(broken, state)(
                {"type": "http", "method": "POST", "path": "/api/text/edit"}, None, None)
        self.assertEqual(state.snapshot()["active"], 0)
        self.assertTrue(state.acquire())
        state.release()


class RuntimeLifecycleTests(unittest.TestCase):
    def test_rejected_inputs_and_model_failure_release_slots(self):
        class Missing:
            def get_bundle(self):
                raise ModelUnavailable("Model unavailable in test")

        with TestClient(create_app(Missing(), max_active_requests=1), base_url="http://localhost") as client:
            cases = [
                ({"content": b"x", "headers": {"Content-Length": str(MAX_REQUEST_BYTES + 1)}}, 413),
                ({"json": {}}, 422),
                ({"json": {"text": " "}}, 400),
                ({"json": {"text": "This text is long enough to reach the model."}}, 503),
            ]
            for arguments, expected in cases:
                response = client.post("/api/text/detect", **arguments)
                self.assertEqual(response.status_code, expected)
                self.assertEqual(client.get("/api/runtime").json()["processing"]["active"], 0)
            self.assertEqual(client.post("/api/text/edit", json={
                "text": "The team reviewed all 20 results in order to check consistency."}).status_code, 200)

    def test_runtime_is_aggregate_only_and_shutdown_clears_reports(self):
        application = create_app()
        store = application.state.report_store
        with TestClient(application, base_url="http://localhost") as client:
            report = store.put({"filename": "private-name.txt", "text": "private-body"})
            timer = store._expiry_timer
            response = client.get("/api/runtime")
            self.assertEqual(response.json()["reports"]["count"], 1)
            for private_value in (report["report_id"], "private-name.txt", "private-body"):
                self.assertNotIn(private_value, response.text)
        self.assertEqual(store.stats()["count"], 0)
        self.assertIsNone(store._expiry_timer)
        self.assertTrue(timer.finished.is_set())

    def test_report_stats_prune_expired_entries_and_account_utf8_bytes(self):
        now = [1000.0]
        store = ReportStore(ttl_seconds=10, clock=lambda: now[0])
        try:
            report = store.put({"text": "测试中文内容"})
            self.assertEqual(store.stats()["bytes"], len(json.dumps(
                report, ensure_ascii=False, allow_nan=False).encode("utf-8")))
            now[0] += 10
            self.assertEqual(store.stats()["count"], 0)
            self.assertEqual(store.stats()["bytes"], 0)
        finally:
            store.close()

    def test_agent_validates_document_before_image_inference(self):
        with TestClient(create_app(), base_url="http://localhost") as client, patch("app.run_agent") as agent:
            response = client.post("/api/agent/upload", files={
                "file": ("image.png", png_bytes(), "image/png"),
                "document": ("broken.docx", b"broken archive", "application/octet-stream"),
            })
            self.assertEqual(response.status_code, 422)
            agent.assert_not_called()
            self.assertEqual(client.get("/api/runtime").json()["processing"]["active"], 0)


if __name__ == "__main__":
    unittest.main()
