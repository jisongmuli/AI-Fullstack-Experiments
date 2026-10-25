"""Bounded, expiring report storage. Source text is never written to disk."""
from __future__ import annotations

import copy
import json
import secrets
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone


class ReportStore:
    def __init__(self, *, ttl_seconds: int = 1800, max_items: int = 20,
                 max_bytes: int = 32 * 1024 * 1024, clock=time.time):
        self.ttl_seconds = ttl_seconds
        self.max_items = max_items
        self.max_bytes = max_bytes
        self.clock = clock
        self._items: OrderedDict[str, tuple[float, int, dict]] = OrderedDict()
        self._lock = threading.Lock()
        self._expiry_timer: threading.Timer | None = None

    def _prune(self, now: float) -> None:
        for key, (expiry, _, _) in list(self._items.items()):
            if expiry <= now:
                del self._items[key]

    def _schedule_expiry(self) -> None:
        if self._expiry_timer is not None:
            self._expiry_timer.cancel()
            self._expiry_timer = None
        if self._items:
            expiry = min(item[0] for item in self._items.values())
            self._expiry_timer = threading.Timer(max(0.01, expiry - self.clock()), self._expire)
            self._expiry_timer.daemon = True
            self._expiry_timer.start()

    def _expire(self) -> None:
        with self._lock:
            self._expiry_timer = None
            self._prune(self.clock())
            self._schedule_expiry()

    def close(self) -> None:
        with self._lock:
            self._items.clear()
            if self._expiry_timer is not None:
                self._expiry_timer.cancel()
                self._expiry_timer = None

    def put(self, result: dict) -> dict:
        now = self.clock()
        report = copy.deepcopy(result)
        report["report_id"] = secrets.token_urlsafe(24)
        report["expires_at"] = datetime.fromtimestamp(now + self.ttl_seconds, timezone.utc).isoformat()
        report["retention_note"] = "报告仅在本地服务内存中保存最多 30 分钟、20 份；达到内存上限会提前移除旧报告。重启后失效。"
        size = len(json.dumps(report, ensure_ascii=False, allow_nan=False).encode("utf-8"))
        if size > self.max_bytes:
            raise ValueError("报告超过内存缓存上限，请缩短文档。")
        with self._lock:
            self._prune(now)
            while self._items and (len(self._items) >= self.max_items or
                                  sum(item[1] for item in self._items.values()) + size > self.max_bytes):
                self._items.popitem(last=False)
            self._items[report["report_id"]] = (now + self.ttl_seconds, size, report)
            self._schedule_expiry()
        return copy.deepcopy(report)

    def get(self, report_id: str) -> dict | None:
        with self._lock:
            self._prune(self.clock())
            record = self._items.get(report_id)
            return copy.deepcopy(record[2]) if record else None

    def stats(self) -> dict:
        """Return aggregate capacity without document text, names or report IDs."""
        with self._lock:
            self._prune(self.clock())
            return {"count": len(self._items), "limit": self.max_items,
                    "bytes": sum(item[1] for item in self._items.values()),
                    "max_bytes": self.max_bytes, "ttl_seconds": self.ttl_seconds}

    def delete(self, report_id: str) -> bool:
        with self._lock:
            self._prune(self.clock())
            deleted = self._items.pop(report_id, None) is not None
            self._schedule_expiry()
            return deleted
