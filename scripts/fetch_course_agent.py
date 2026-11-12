"""Fetch pinned Qwen3.5 local Agent assets from the author's repository."""
import argparse
import json
import hashlib
import time
import httpx
from fetch_2026_assets import ROOT, download

REPO = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"


def resumable(client, url, target, digest, size):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == size:
        with target.open("rb") as stream: actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest is None or digest == actual:
            return {"path": target.relative_to(ROOT).as_posix(), "url": url, "bytes": size, "sha256": actual}
    temporary = target.with_name(target.name + ".part")
    for attempt in range(8):
        offset = temporary.stat().st_size if temporary.exists() else 0
        if offset == size:
            break
        if offset > size:
            raise RuntimeError("Partial download exceeds pinned file size")
        try:
            with client.stream("GET", url + f"?course_offset={offset}", headers={"Range": f"bytes={offset}-"}) as response:
                response.raise_for_status()
                if offset and (response.status_code != 206 or not response.headers.get("content-range", "").startswith(f"bytes {offset}-")):
                    raise RuntimeError("Resume range not honored")
                with temporary.open("ab") as stream:
                    for chunk in response.iter_bytes(1024*1024):
                        stream.write(chunk)
                        if stream.tell() > size:
                            raise RuntimeError("Download exceeds pinned length")
            if temporary.stat().st_size == size:
                break
        except (httpx.HTTPError, RuntimeError) as exc:
            print("Retry", target.name, type(exc).__name__, "saved", temporary.stat().st_size if temporary.exists() else 0, flush=True)
            time.sleep(min(attempt+1, 5))
    if not temporary.exists() or temporary.stat().st_size != size:
        raise RuntimeError("Pinned asset download did not finish")
    with temporary.open("rb") as stream: actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest and actual != digest:
        raise RuntimeError("Pinned asset SHA-256 mismatch")
    temporary.replace(target)
    print("Verified", target.name, size, flush=True)
    return {"path": target.relative_to(ROOT).as_posix(), "url": url, "bytes": size, "sha256": actual}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy")
    args = parser.parse_args()
    with httpx.Client(proxy=args.proxy, timeout=180, follow_redirects=True, trust_env=False) as client:
        metadata = client.get(f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true")
        metadata.raise_for_status()
        receipts = []
        for row in metadata.json()["siblings"]:
            name = row["rfilename"]
            if name == ".gitattributes":
                continue
            receipts.append(resumable(client, f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}",
                ROOT / "models/agent_qwen35" / name, row.get("lfs", {}).get("sha256"), row.get("size")))
    output = ROOT / "results/course_20261027/agent-assets.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"repo": REPO, "revision": REVISION, "license": "Apache-2.0", "assets": receipts}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
