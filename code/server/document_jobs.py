"""Bounded document workers. Inputs live only in worker memory."""
from __future__ import annotations

import secrets
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Callable

from runtime import ProcessingLease


TERMINAL = {"completed", "failed", "cancelled"}


def timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


class JobCancelled(Exception):
    pass


class JobError(Exception):
    def __init__(self, message: str, status_code: int = 503):
        super().__init__(message)
        self.status_code = status_code


class DocumentJobs:
    def __init__(self, *, workers: int = 2, max_jobs: int = 20,
                 ttl_seconds: int = 1800, delete_report: Callable[[str], bool], clock=time.time):
        self.max_jobs = max_jobs
        self.ttl_seconds = ttl_seconds
        self.delete_report = delete_report
        self.clock = clock
        self._jobs: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()
        self._closing = False
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="document")

    def _prune(self) -> None:
        now = self.clock()
        for job_id, job in list(self._jobs.items()):
            if job["finished"] is not None and job["finished"] + self.ttl_seconds <= now:
                del self._jobs[job_id]

    def _snapshot(self, job: dict) -> dict:
        return {key: job[key] for key in ("job_id", "filename", "status", "stage", "completed",
                "total", "progress", "created_at", "updated_at", "expires_at", "report_id")} | {
                    "error": dict(job["error"]) if job["error"] else None}

    def submit(self, filename: str, processor: Callable, lease: ProcessingLease) -> dict:
        now = self.clock()
        with self._lock:
            if self._closing:
                raise JobError("服务正在退出，请稍后重试。")
            self._prune()
            while len(self._jobs) >= self.max_jobs:
                oldest = next((key for key, value in self._jobs.items() if value["status"] in TERMINAL), None)
                if oldest is None:
                    raise JobError("当前文档任务已满，请稍后重试。")
                del self._jobs[oldest]
            job_id = secrets.token_urlsafe(24)
            job = {"job_id": job_id, "filename": filename, "status": "pending", "stage": "pending",
                   "completed": 0, "total": 0, "progress": 0, "created_at": timestamp(now),
                   "updated_at": timestamp(now), "expires_at": None, "report_id": None,
                   "error": None, "finished": None, "cancel": threading.Event()}
            self._jobs[job_id] = job
            try:
                self._executor.submit(self._run, job, processor, lease)
            except Exception:
                del self._jobs[job_id]
                raise JobError("任务启动失败，请稍后重试。") from None
            lease.detach()
            return self._snapshot(job)

    def _run(self, job: dict, processor: Callable, lease: ProcessingLease) -> None:
        def check_cancel() -> None:
            if job["cancel"].is_set():
                raise JobCancelled()

        def progress(stage: str, completed: int = 0, total: int = 0) -> None:
            check_cancel()
            if stage not in {"extracting", "analyzing", "saving"} or not 0 <= completed <= total:
                raise ValueError("Invalid task progress")
            with self._lock:
                check_cancel()
                job.update(status="running", stage=stage, completed=completed, total=total,
                           updated_at=timestamp(self.clock()), progress=(5 if stage == "extracting" else
                               95 if stage == "saving" else round(10 + 85 * completed / max(total, 1))))

        report_id = None
        try:
            check_cancel()
            report_id = processor(progress, check_cancel)
            if not isinstance(report_id, str) or not report_id:
                raise JobError("任务未生成可用报告，请重新检测。")
            with self._lock:
                check_cancel()
                job.update(status="completed", stage="completed", progress=100, report_id=report_id)
        except Exception as exc:
            with self._lock:
                cancelled = job["cancel"].is_set() or isinstance(exc, JobCancelled)
                job.update(status="cancelled" if cancelled else "failed", stage="cancelled" if cancelled else "failed")
                if not cancelled:
                    job["error"] = {"message": str(exc) if isinstance(exc, JobError) else "文档处理失败，请检查文件或模型后重试。",
                                    "status_code": exc.status_code if isinstance(exc, JobError) else 503}
            if report_id:
                self.delete_report(report_id)
        finally:
            with self._lock:
                now = self.clock()
                job.update(finished=now, updated_at=timestamp(now), expires_at=timestamp(now + self.ttl_seconds))
            lease.release()

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            self._prune()
            job = self._jobs.get(job_id)
            return self._snapshot(job) if job else None

    def list_jobs(self) -> list[dict]:
        with self._lock:
            self._prune()
            return [self._snapshot(job) for job in reversed(self._jobs.values())]

    def cancel(self, job_id: str) -> dict | None:
        with self._lock:
            self._prune()
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job["status"] not in TERMINAL:
                job["cancel"].set()
                job.update(status="cancel_requested", updated_at=timestamp(self.clock()))
            return self._snapshot(job)

    def close(self) -> None:
        with self._lock:
            self._closing = True
            for job in self._jobs.values():
                if job["status"] not in TERMINAL:
                    job["cancel"].set()
        self._executor.shutdown(wait=True)
        with self._lock:
            self._jobs.clear()
