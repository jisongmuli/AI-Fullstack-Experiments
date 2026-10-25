"""Background document processing, cooperative cancellation and capacity."""
from __future__ import annotations

import json
import sys
import threading
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import create_app  # noqa: E402
from document_jobs import DocumentJobs  # noqa: E402
from runtime import ProcessingLease, ProcessingState  # noqa: E402
from documents import analyze_document, extract_pasted_text  # noqa: E402


TEXT = "The team reviewed the source files and recorded all missing values before preparing the data."


class ControlledText:
    def __init__(self):
        self.entered = threading.Event()
        self.finish = threading.Event()
        self.finish.set()

    def get_bundle(self):
        return self

    def predict_text_segments(self, text):
        self.entered.set()
        if not self.finish.wait(5):
            raise TimeoutError("Test worker was not released")
        return {"risk_score": .2, "decision": "likely_human", "model_version": "job-test",
                "segments": [{"start": 0, "end": len(text), "text": text, "risk_score": .2,
                              "evidence": [], "decision": "likely_human"}], "segment_summary": {}}


class DocumentJobApiTests(unittest.TestCase):
    def setUp(self):
        self.model = ControlledText()
        self.app = create_app(self.model, max_active_requests=1)
        self.client = TestClient(self.app, base_url="http://localhost")
        self.client.__enter__()

    def tearDown(self):
        self.model.finish.set()
        self.client.__exit__(None, None, None)

    def submit(self, text=TEXT):
        response = self.client.post("/api/document/jobs", data={"text": text})
        self.assertEqual(response.status_code, 202, response.text)
        return response.json()

    def wait_done(self, job_id):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            result = self.client.get(f"/api/document/jobs/{job_id}").json()
            if result["status"] in {"completed", "failed", "cancelled"} and self.app.state.processing.snapshot()["active"] == 0:
                return result
            time.sleep(.005)
        self.fail("Job did not finish")

    def test_acceptance_does_not_wait_and_slot_stays_held_until_worker_finishes(self):
        self.model.finish.clear()
        job = self.submit()
        self.assertTrue(self.model.entered.wait(1))
        self.assertEqual(self.client.get("/api/runtime").json()["processing"]["active"], 1)
        self.assertEqual(self.client.get("/health").status_code, 200)
        busy = self.client.post("/api/text/edit", json={"text": TEXT})
        self.assertEqual(busy.status_code, 503)
        self.model.finish.set()
        done = self.wait_done(job["job_id"])
        self.assertEqual(done["status"], "completed")
        self.assertEqual(done["progress"], 100)
        self.assertEqual(self.client.get(f"/api/document/reports/{done['report_id']}").json()["extracted_text"], TEXT)
        self.assertEqual(self.client.post("/api/text/edit", json={"text": TEXT}).status_code, 200)

    def test_cancellation_waits_for_current_call_and_produces_no_report(self):
        self.model.finish.clear()
        job = self.submit()
        self.assertTrue(self.model.entered.wait(1))
        cancelled = self.client.post(f"/api/document/jobs/{job['job_id']}/cancel")
        self.assertEqual(cancelled.json()["status"], "cancel_requested")
        self.assertEqual(self.app.state.processing.snapshot()["active"], 1)
        self.model.finish.set()
        done = self.wait_done(job["job_id"])
        self.assertEqual(done["status"], "cancelled")
        self.assertIsNone(done["report_id"])
        self.assertEqual(self.app.state.report_store.stats()["count"], 0)

    def test_job_survives_a_new_browser_client_and_metadata_has_no_source(self):
        body = "PRIVATE_BODY_ABC " + TEXT
        job = self.submit(body)
        done = self.wait_done(job["job_id"])
        browser = TestClient(self.app, base_url="http://localhost")
        listing = browser.get("/api/document/jobs")
        self.assertEqual(listing.headers["cache-control"], "no-store")
        self.assertEqual(listing.json()["jobs"][0]["job_id"], job["job_id"])
        self.assertNotIn("PRIVATE_BODY_ABC", listing.text)
        self.assertNotIn("extracted_text", listing.text)
        self.assertEqual(browser.get(f"/api/document/jobs/{job['job_id']}").json()["report_id"], done["report_id"])

    def test_invalid_document_fails_cleanly_and_releases_capacity(self):
        response = self.client.post("/api/document/jobs", files={"file": ("broken.docx", b"broken", "application/octet-stream")})
        self.assertEqual(response.status_code, 202)
        done = self.wait_done(response.json()["job_id"])
        self.assertEqual(done["status"], "failed")
        self.assertEqual(done["error"]["status_code"], 422)
        self.assertIsNone(done["report_id"])
        self.assertEqual(self.client.post("/api/text/edit", json={"text": TEXT}).status_code, 200)

    def test_rejected_submission_and_unknown_ids(self):
        self.assertEqual(self.client.post("/api/document/jobs").status_code, 400)
        self.assertEqual(self.client.post("/api/document/jobs", data={"text": TEXT}, files={"file": ("x.txt", b"x")}).status_code, 400)
        self.assertEqual(self.client.post("/api/document/jobs", data={"text": "x" * 100001}).status_code, 413)
        self.assertEqual(self.app.state.processing.snapshot()["active"], 0)
        self.assertEqual(self.client.get("/api/document/jobs/missing").status_code, 404)
        self.assertEqual(self.client.post("/api/document/jobs/missing/cancel").status_code, 404)

    def test_terminal_cancel_is_idempotent_and_expired_metadata_is_removed(self):
        now = [1000.0]
        jobs = self.app.state.document_jobs
        jobs.clock = lambda: now[0]
        jobs.ttl_seconds = 10
        job = self.submit()
        done = self.wait_done(job["job_id"])
        self.assertEqual(self.client.post(f"/api/document/jobs/{job['job_id']}/cancel").json()["status"], "completed")
        self.assertEqual(self.client.get(f"/api/document/reports/{done['report_id']}").status_code, 200)
        now[0] += 10
        self.assertEqual(self.client.get("/api/document/jobs").json(), {"jobs": []})

    def test_retained_job_metadata_is_bounded_and_finished_jobs_are_evicted(self):
        self.app.state.document_jobs.max_jobs = 2
        ids = []
        for _ in range(3):
            job = self.submit()
            self.wait_done(job["job_id"])
            ids.append(job["job_id"])
        retained = self.client.get("/api/document/jobs").json()["jobs"]
        self.assertEqual([job["job_id"] for job in retained], ids[:0:-1])
        self.assertEqual(self.client.get(f"/api/document/jobs/{ids[0]}").status_code, 404)

    def test_long_filename_keeps_its_format_during_extraction(self):
        response = self.client.post("/api/document/jobs", files={
            "file": ("a" * 245 + ".txt", TEXT.encode(), "text/plain")})
        self.assertEqual(response.status_code, 202)
        done = self.wait_done(response.json()["job_id"])
        self.assertEqual(done["status"], "completed")
        report = self.client.get(f"/api/document/reports/{done['report_id']}").json()
        self.assertEqual(report["format"], "txt")


class JobWorkerTests(unittest.TestCase):
    def test_retention_preserves_an_older_running_job(self):
        state = ProcessingState(2)
        jobs = DocumentJobs(workers=2, max_jobs=2, delete_report=lambda _: True)
        finish = threading.Event()
        self.addCleanup(jobs.close)
        self.addCleanup(finish.set)

        def submit(process):
            self.assertTrue(state.acquire())
            return jobs.submit("fixture.txt", process, ProcessingLease(state))

        def slow(progress, check_cancel):
            finish.wait(3)
            return "slow-report"

        active = submit(slow)
        for _ in range(3):
            submit(lambda progress, check_cancel: "fast-report")
            deadline = time.monotonic() + 1
            while state.snapshot()["active"] != 1 and time.monotonic() < deadline:
                time.sleep(.002)
            self.assertEqual(state.snapshot()["active"], 1)
        self.assertIsNotNone(jobs.get(active["job_id"]))
        self.assertEqual(len(jobs.list_jobs()), 2)

    def test_shutdown_cancels_workers_and_releases_detached_lease_once(self):
        state = ProcessingState(1)
        state.acquire()
        lease = ProcessingLease(state)
        jobs = DocumentJobs(workers=1, delete_report=lambda _: True)
        started = threading.Event()

        def process(progress, check_cancel):
            progress("extracting")
            started.set()
            while True:
                check_cancel()
                time.sleep(.002)

        jobs.submit("example.txt", process, lease)
        self.assertTrue(started.wait(1))
        lease.release_if_attached()
        self.assertEqual(state.snapshot()["active"], 1)
        jobs.close()
        lease.release()
        self.assertEqual(state.snapshot()["active"], 0)
        self.assertEqual(state.snapshot()["completed"], 1)
        self.assertEqual(jobs.list_jobs(), [])

    def test_cancel_at_report_commit_removes_the_just_created_report(self):
        state = ProcessingState(1)
        state.acquire()
        lease = ProcessingLease(state)
        removed = []
        jobs = DocumentJobs(workers=1, delete_report=lambda report_id: removed.append(report_id))
        self.addCleanup(jobs.close)
        started, finish = threading.Event(), threading.Event()

        def process(progress, check_cancel):
            progress("saving")
            started.set()
            finish.wait(2)
            return "fresh-report"

        job = jobs.submit("example.txt", process, lease)
        self.assertTrue(started.wait(1))
        jobs.cancel(job["job_id"])
        finish.set()
        jobs.close()
        self.assertEqual(removed, ["fresh-report"])
        self.assertEqual(state.snapshot()["active"], 0)

    def test_long_document_reports_progress_and_stops_between_calls(self):
        source = extract_pasted_text((TEXT + "\n\n") * 150)
        calls, updates = [], []
        cancelled = [False]

        def detect(text):
            calls.append(text)
            return {"risk_score": .2, "decision": "likely_human", "evidence": []}

        def progress(done, total):
            updates.append((done, total))
            if done == 1:
                cancelled[0] = True

        def check_cancel():
            if cancelled[0]:
                raise RuntimeError("cancel fixture")

        with self.assertRaisesRegex(RuntimeError, "cancel fixture"):
            analyze_document(source, detect, progress=progress, check_cancel=check_cancel)
        self.assertEqual(len(calls), 1)
        self.assertEqual(updates[0][0], 0)
        self.assertEqual(updates[1][0], 1)
        self.assertGreater(updates[0][1], 1)


if __name__ == "__main__":
    unittest.main()
