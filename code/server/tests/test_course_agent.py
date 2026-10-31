"""Actual ASGI tool calls, bounded orchestration, and course model deployment."""
import io
import json
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from fastapi.responses import PlainTextResponse
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import create_app
from course_agent import CourseAgent


def png():
    output = io.BytesIO()
    Image.new("RGB", (32, 32)).save(output, "PNG")
    return output.getvalue()


class Tools:
    def get_bundle(self): return self
    def predict_image(self, image): return {"risk_score": .7, "decision": "uncertain", "signals": [], "model_version": "fixture-image"}
    def predict_text(self, text): return {"risk_score": .3, "decision": "human", "signals": [], "model_version": "fixture-text"}


class Planner:
    def __init__(self, tools): self.tools = tools
    def status(self): return {"available": True, "loaded": True, "model": "test-language-model"}
    def plan(self, message, available, history):
        self.available, self.history = available, history
        return {"tools": self.tools, "method": "test-controlled-planner"}
    def summarize(self, message, findings): return "工具结果已返回。"


class AgentApiTests(unittest.TestCase):
    def test_model_selected_two_tools_hit_real_routes_and_release_slots(self):
        planner = Planner(["image_detection", "text_detection"])
        with TestClient(create_app(Tools(), agent_model=planner), base_url="http://localhost") as client:
            value = client.post("/api/agent/run", data={"instruction": "check image and text", "text": "The team compared each record and reviewed the final results."},
                files={"image": ("image.png", png(), "image/png")})
            self.assertEqual(value.status_code, 200, value.text)
            self.assertEqual([row["path"] for row in value.json()["tool_calls"]], ["/api/image/detect", "/api/text/detect"])
            self.assertTrue(all(row["http_status"] == 200 for row in value.json()["tool_calls"]))
            self.assertEqual(client.get("/api/runtime").json()["processing"]["active"], 0)
            self.assertFalse(client.get("/api/runtime").json()["agent"]["active"])
            self.assertEqual(value.headers["cache-control"], "no-store")

    def test_missing_input_and_unknown_tools_are_rejected_without_calls(self):
        for tools, expected in [(["image_detection"], 400), (["external_shell"], 503)]:
            with TestClient(create_app(Tools(), agent_model=Planner(tools)), base_url="http://localhost") as client:
                response = client.post("/api/agent/run", data={"instruction": "test"})
                self.assertEqual(response.status_code, expected)
                self.assertEqual(client.get("/api/runtime").json()["processing"]["completed"], 0)

    def test_invalid_history_and_corrupt_upload_are_rejected_before_planning(self):
        planner = Planner([])
        with patch.object(planner, "plan", side_effect=AssertionError("Should not plan")), \
             TestClient(create_app(Tools(), agent_model=planner), base_url="http://localhost") as client:
            self.assertEqual(client.post("/api/agent/run", data={"instruction": "test", "history": "{}"}).status_code, 400)
            self.assertEqual(client.post("/api/agent/run", data={"instruction": "test"}, files={"image": ("bad.png", b"invalid", "image/png")}).status_code, 400)
            self.assertEqual(client.post("/api/agent/run", data={"instruction": "   "}).status_code, 400)

    def test_invalid_document_keeps_its_ingestion_error_and_allows_retry(self):
        planner = Planner([])
        with TestClient(create_app(Tools(), agent_model=planner), base_url="http://localhost", raise_server_exceptions=False) as client:
            for filename, content, expected in [("old.doc", b"old document", 415), ("broken.docx", b"not a zip", 422),
                                                 ("empty.txt", b"", 400), ("blank.txt", b"   ", 422)]:
                with self.subTest(filename=filename), patch.object(planner, "plan") as planned:
                    response = client.post("/api/agent/run", data={"instruction": "check document"},
                        files={"document": (filename, content, "application/octet-stream")})
                    self.assertEqual(response.status_code, expected, response.text)
                    self.assertIsInstance(response.json()["detail"], str)
                    planned.assert_not_called()
            self.assertEqual(client.post("/api/agent/run", data={"instruction": "test"}).status_code, 200)
            self.assertFalse(client.get("/api/runtime").json()["agent"]["active"])

    def test_nested_history_roles_return_a_validation_message(self):
        planner = Planner([])
        with TestClient(create_app(Tools(), agent_model=planner), base_url="http://localhost", raise_server_exceptions=False) as client:
            for role in ([], {}, None, 1):
                with self.subTest(role=role), patch.object(planner, "plan") as planned:
                    response = client.post("/api/agent/run", data={"instruction": "test", "history": json.dumps([{"role": role, "content": "hello"}])})
                    self.assertEqual(response.status_code, 400, response.text)
                    planned.assert_not_called()

    def test_malformed_model_plan_fails_cleanly_and_releases_admission(self):
        planner = Planner([])
        with TestClient(create_app(Tools(), agent_model=planner), base_url="http://localhost", raise_server_exceptions=False) as client:
            for plan in (None, [], {"tools": [{}]}, {"tools": [["text_detection"]]}, {"tools": ["model_info", "model_info"]}):
                with self.subTest(plan=plan), patch.object(planner, "plan", return_value=plan):
                    response = client.post("/api/agent/run", data={"instruction": "test"})
                    self.assertEqual(response.status_code, 503, response.text)
                    self.assertIsInstance(response.json()["detail"], str)
                    self.assertFalse(client.get("/api/runtime").json()["agent"]["active"])
            self.assertEqual(client.post("/api/agent/run", data={"instruction": "test"}).status_code, 200)

    def test_non_json_or_broken_tool_keeps_other_tool_results(self):
        for failure in ("text", "exception", "json-list"):
            application = create_app(Tools(), agent_model=Planner(["model_info", "course_overview"]))
            def broken_tool():
                if failure == "exception":
                    raise RuntimeError("private diagnostic must not be returned")
                if failure == "json-list":
                    return ["unexpected"]
                return PlainTextResponse("temporarily unavailable", status_code=503)
            for route in application.routes:
                if getattr(route, "path", None) == "/api/models":
                    route.dependant.call = broken_tool
            with self.subTest(failure=failure), TestClient(application, base_url="http://localhost", raise_server_exceptions=False) as client:
                response = client.post("/api/agent/run", data={"instruction": "models and course"})
                self.assertEqual(response.status_code, 200, response.text)
                calls = response.json()["tool_calls"]
                self.assertEqual([row["status"] for row in calls], ["failed", "ok"])
                self.assertIsInstance(calls[0]["result"]["detail"], str)
                self.assertEqual(calls[1]["http_status"], 200)
                self.assertNotIn("private diagnostic", response.text)
                self.assertFalse(client.get("/api/runtime").json()["agent"]["active"])

    def test_completed_document_can_be_opened_and_exported(self):
        with TestClient(create_app(Tools(), agent_model=Planner(["document_detection"])), base_url="http://localhost") as client:
            result = client.post("/api/agent/run", data={"instruction": "check document"},
                files={"document": ("example.txt", b"The team compared each record and reviewed the final results.", "text/plain")})
            self.assertEqual(result.status_code, 200, result.text)
            report = result.json()["tool_calls"][0]["result"]
            self.assertNotIn("extracted_text", report)
            self.assertEqual(client.get(f"/api/document/reports/{report['report_id']}/docx").status_code, 200)

    def test_summary_failure_keeps_successful_tools(self):
        planner = Planner(["model_info"])
        with patch.object(planner, "summarize", side_effect=RuntimeError("fixture")), \
             TestClient(create_app(Tools(), agent_model=planner), base_url="http://localhost") as client:
            result = client.post("/api/agent/run", data={"instruction": "models"}).json()
            self.assertEqual(result["answer_source"], "tool_results")
            self.assertEqual(result["tool_calls"][0]["http_status"], 200)

    def test_exclusive_agent_admission_does_not_consume_detector_slots(self):
        application = create_app(Tools(), agent_model=Planner([]))
        agent = application.state.course_agent
        agent.admission.acquire()
        try:
            with TestClient(application, base_url="http://localhost") as client:
                self.assertEqual(client.post("/api/agent/run", data={"instruction": "test"}).status_code, 503)
                self.assertEqual(client.get("/api/runtime").json()["processing"]["active"], 0)
                self.assertEqual(client.get("/health").status_code, 200)
        finally:
            agent.admission.release()


class CourseDeploymentTests(unittest.TestCase):
    def test_trained_and_optimized_weights_are_both_served(self):
        with TestClient(create_app(), base_url="http://localhost") as client:
            overview = client.get("/api/course").json()
            self.assertEqual(len(overview["stages"]), 5)
            self.assertTrue(all(stage["ready"] for stage in overview["stages"]))
            text = client.get("/api/course/example/text").json()["text"]
            for stage in ("trained", "optimized"):
                image = client.post(f"/api/course/image/detect?stage={stage}", files={"file": ("image.png", png(), "image/png")})
                self.assertEqual(image.status_code, 200, image.text)
                self.assertIn(stage, image.json()["model_version"])
                detected = client.post("/api/course/text/detect", json={"text": text, "stage": stage})
                self.assertEqual(detected.status_code, 200, detected.text)
                self.assertIn(stage, detected.json()["model_version"])
            self.assertEqual(client.post("/api/course/text/detect", json={"text": "short", "stage": "invalid"}).status_code, 422)


if __name__ == "__main__": unittest.main()
