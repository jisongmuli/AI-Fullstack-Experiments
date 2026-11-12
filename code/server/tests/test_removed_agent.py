"""Six course workflows and rejection of the obsolete Agent protocol."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image
import io

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import create_app


class CourseNavigationTests(unittest.TestCase):
    def test_legacy_endpoints_are_absent_from_routes_and_openapi(self):
        with TestClient(create_app(), base_url="http://localhost") as client:
            paths = client.get("/openapi.json").json()["paths"]
            for endpoint in ("/api/agent/upload",):
                self.assertNotIn(endpoint, paths)
                self.assertEqual(client.post(endpoint, json={"text": "example"}).status_code, 404)
            self.assertIn("/api/agent/run", paths)
            self.assertEqual(client.post("/api/agent/run", json={"text": "example"}).status_code, 422)
            self.assertEqual(client.get("/api/runtime").json()["processing"]["active"], 0)

    def test_home_keeps_four_direct_workflows_without_agent_controls(self):
        with TestClient(create_app(), base_url="http://localhost") as client:
            html = client.get("/").text
            for name in ("image", "text", "document", "edit", "course", "agent"):
                self.assertIn(f'data-view="{name}"', html)
            self.assertIn('id="view-agent"', html)
            self.assertNotIn("agent.css", html)
            self.assertIn("/static/text.css", html)


class AnyResolutionBundle:
    def get_bundle(self):
        return self

    def predict_image(self, image):
        return {"risk_score": .7, "decision": "uncertain", "signals": [],
                "model_version": "resolution-contract-fixture", "out_of_scope": False,
                "model_scores": [{"model": "one modern detector", "score": .7}],
                "input_size": list(image.size), "model_input_size": [256, 256]}


class AnyResolutionApiTests(unittest.TestCase):
    def test_any_positive_shape_through_4k_reaches_the_image_model(self):
        with TestClient(create_app(AnyResolutionBundle()), base_url="http://localhost") as client:
            for width, height, mode in [(1, 1, "RGB"), (1, 4096, "L"), (4096, 1, "RGBA"),
                                        (17, 29, "RGB"), (199, 199, "RGB"), (32, 32, "L"),
                                        (3840, 2160, "RGB"), (2160, 3840, "RGB"), (4096, 2160, "RGB")]:
                with self.subTest(size=(width, height), mode=mode):
                    encoded = io.BytesIO()
                    with Image.new(mode, (width, height)) as image:
                        image.save(encoded, "PNG")
                    response = client.post("/api/image/detect", files={"file": ("fixture.png", encoded.getvalue(), "image/png")})
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(response.json()["input_size"], [width, height])
                    self.assertEqual(response.json()["risk_score"], .7)
                    self.assertNotIn("model_disagreement", response.json())


if __name__ == "__main__":
    unittest.main()
