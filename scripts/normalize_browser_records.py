"""Normalize CLI result envelopes and fail on a failed browser verification."""
import json
import sys
from pathlib import Path

for name in sys.argv[1:]:
    path = Path(name)
    result = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(result, dict) and "result" in result:
        result = result["result"]
    while isinstance(result, str):
        result = json.loads(result)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(path.name, result.get("status"), len(result.get("checks", [])), result.get("error", ""))
    if result.get("status") != "passed":
        raise SystemExit(1)
