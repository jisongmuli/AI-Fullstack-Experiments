"""Probe a running local service while two real image analyses are in flight."""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import time
from pathlib import Path

import httpx
from PIL import Image


async def measure(url: str) -> dict:
    image = Image.new("RGB", (512, 512))
    image.putdata([(x % 256, y % 256, (x + y) % 256)
                   for y in range(512) for x in range(512)])
    stream = io.BytesIO()
    image.save(stream, "PNG")
    data = stream.getvalue()
    async with httpx.AsyncClient(base_url=url, timeout=180, trust_env=False) as client:
        before = (await client.get("/api/runtime")).json()
        if before["processing"]["active"]:
            raise RuntimeError("Wait for the service to become idle before this probe.")
        if before["processing"]["limit"] != 2:
            raise RuntimeError("This probe expects the default limit of two requests.")
        started = time.perf_counter()
        jobs = [asyncio.create_task(client.post("/api/image/detect", files={
            "file": ("generated-stability-fixture.png", data, "image/png")})) for _ in range(2)]
        health_times = []
        try:
            deadline = time.perf_counter() + 10
            while True:
                status = (await client.get("/api/runtime")).json()
                if status["processing"]["active"] == 2:
                    break
                if any(job.done() for job in jobs) or time.perf_counter() >= deadline:
                    raise AssertionError("Two image tasks did not overlap during the probe.")
                await asyncio.sleep(.02)
            busy = await client.post("/api/text/detect", json={"text": "Concurrent probe"})
            assert busy.status_code == 503, busy.text
            assert busy.headers.get("retry-after") == "2"
            for _ in range(6):
                tick = time.perf_counter()
                health = await client.get("/health", timeout=2)
                health_times.append(round((time.perf_counter() - tick) * 1000, 2))
                assert health.json()["status"] == "ok"
                await asyncio.sleep(.05)
            responses = await asyncio.gather(*jobs)
            for response in responses:
                response.raise_for_status()
                assert len(response.json()["model_scores"]) >= 1
            after = (await client.get("/api/runtime")).json()
            assert after["processing"]["active"] == 0
            assert after["processing"]["completed"] - before["processing"]["completed"] == 2
            assert after["processing"]["rejected"] - before["processing"]["rejected"] == 1
            return {"status": "passed", "concurrent_real_image_tasks": 2,
                    "excess_request_status": busy.status_code,
                    "retry_after_seconds": 2, "health_latency_ms": health_times,
                    "max_health_latency_ms": max(health_times),
                    "elapsed_seconds": round(time.perf_counter() - started, 2),
                    "processing_after": after["processing"],
                    "accuracy_evaluated": False}
        finally:
            await asyncio.gather(*jobs, return_exceptions=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8002")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = asyncio.run(measure(args.url))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
