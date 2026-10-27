"""Bound expensive requests before uploads are buffered or workers are occupied."""
from __future__ import annotations

import threading
from typing import Any

from starlette.responses import JSONResponse


PROCESSING_PATHS = frozenset({
    "/api/image/detect", "/api/text/detect", "/api/document/detect",
    "/api/document/jobs",
    "/api/text/edit", "/api/text/edit/llm-preview",
    "/api/course/image/detect", "/api/course/text/detect",
})


class ProcessingState:
    def __init__(self, limit: int = 2):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("Processing limit must be a positive integer.")
        self.limit = limit
        self._active = self._completed = self._rejected = 0
        self._lock = threading.Lock()

    def acquire(self) -> bool:
        with self._lock:
            if self._active >= self.limit:
                self._rejected += 1
                return False
            self._active += 1
            return True

    def release(self) -> None:
        with self._lock:
            self._active -= 1
            self._completed += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {"active": self._active, "limit": self.limit,
                    "completed": self._completed, "rejected": self._rejected}


class ProcessingLimit:
    def __init__(self, app: Any, state: ProcessingState):
        self.app = app
        self.state = state

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        # Starlette also accepts a trailing slash and redirects it to the route.
        if (scope["type"] != "http" or scope["method"] != "POST"
                or scope["path"].rstrip("/") not in PROCESSING_PATHS):
            await self.app(scope, receive, send)
            return
        if not self.state.acquire():
            await JSONResponse(
                {"detail": "当前处理任务已满，请稍后重试。"}, status_code=503,
                headers={"Retry-After": "2", "Cache-Control": "no-store"},
            )(scope, receive, send)
            return
        lease = ProcessingLease(self.state)
        scope["aigc.processing_lease"] = lease
        try:
            await self.app(scope, receive, send)
        finally:
            lease.release_if_attached()


class ProcessingLease:
    """An accepted background task keeps its slot until its worker finishes."""
    def __init__(self, state: ProcessingState):
        self.state = state
        self._lock = threading.Lock()
        self._released = False
        self._detached = False

    def detach(self) -> None:
        with self._lock:
            self._detached = True

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        self.state.release()

    def release_if_attached(self) -> None:
        with self._lock:
            if self._detached or self._released:
                return
            self._released = True
        self.state.release()
