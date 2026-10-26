"""Compare current text inference with a local Git baseline on generated fixtures."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/models"))
from inference import DetectorService


def measure(call, repeats):
    call()
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        timings.append((time.perf_counter() - started) * 1000)
    return {"median_ms": round(statistics.median(timings), 3),
            "min_ms": round(min(timings), 3), "max_ms": round(max(timings), 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="ef605bc")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, default=ROOT / "results/algorithm_20261026/text-benchmark.json")
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("--repeats must be at least 3")
    # Read an explicit local commit; never fetch or execute a remote revision.
    ref = subprocess.check_output(["git", "rev-parse", "--verify", args.baseline_ref + "^{commit}"], cwd=ROOT, text=True).strip()
    source = subprocess.check_output(["git", "show", ref + ":code/models/inference.py"], cwd=ROOT).decode("utf-8")
    baseline_module = types.ModuleType("aigc_text_benchmark_baseline")
    exec(compile(source, "git:" + ref + ":inference.py", "exec"), baseline_module.__dict__)
    current = DetectorService(ROOT)
    baseline = baseline_module.DetectorService.__new__(baseline_module.DetectorService)
    for name in ["config", "text_model", "zh_config", "zh_text_model"]:
        setattr(baseline, name, getattr(current, name))

    english = [
        "The research team compared the source records and checked every label before the final evaluation.",
        "After the rain stopped, I walked home past the station and found the old shop still open.",
        "The sensor was tested at several temperatures, with calibration values recorded for each measurement.",
        "Students discussed the reading in small groups and wrote their own notes about the central argument.",
    ]
    chinese = [
        "研究团队先核对原始样本的来源与时间，再检查缺失字段、重复记录和异常数值。完成清洗后，团队按来源划分训练集与验证集，分别记录评估结果，整理误报样本并进行人工复核。",
        "早上出门时天还没有亮，街角的小店已经开门了。我买了热豆浆，沿着河边走到车站，想起昨天和朋友讨论的旅行计划。等车的人渐渐多起来，远处的钟声传过来，新的一天就这样开始了。",
        "本次实验使用独立的验证流程。采集阶段记录文件格式、字符长度和标签来源，处理阶段检查空白内容与重复文本。测试阶段逐项保存接口响应、分析耗时和输出报告，方便后续复查与比较。",
        "城市公共服务的改进需要持续收集居民意见。社区通过访谈和问卷了解交通、医疗与教育方面的需求，公开项目进度和预算安排，并根据反馈调整实施方案，使服务更加贴近实际生活。",
    ]
    normal = english + chinese + [x + "\n\n" + y for x, y in zip(english, chinese)]
    normal += [x.replace("数据", "样本") + "ＡＢＣ１２３" for x in chinese]
    normal += ["\n\n".join(english * 10), "\n\n".join(chinese * 15), "\n\n".join(normal[:8] * 6)]
    cases = []
    for index, text in enumerate(normal):
        old, new = baseline.predict_text_segments(text), current.predict_text_segments(text)
        for key in ["risk_score", "decision", "out_of_scope", "applicability", "segment_summary", "segments"]:
            assert old.get(key) == new.get(key), (index, key, old.get(key), new.get(key))
        cases.append({"case": index + 1, "characters": len(text), "segments": len(new["segments"]), "same_scores_decisions_offsets_and_evidence": True})
    batches = english * 16 + chinese * 16
    unique_english = [english[i % 4] + f" Record number {i + 1} was reviewed separately." for i in range(40)]
    unique_chinese = [chinese[i % 4] + f"本次记录编号为{i + 1}，已经单独复查。" for i in range(60)]
    unique_documents = ["\n\n".join(unique_english), "\n\n".join(unique_chinese)]
    for text in unique_documents:
        old, new = baseline.predict_text_segments(text), current.predict_text_segments(text)
        assert old == new
        cases.append({"case": len(cases) + 1, "characters": len(text), "segments": len(new["segments"]), "same_scores_decisions_offsets_and_evidence": True})
    batch = current.predict_text_batch(batches)
    for text, result in zip(batches, batch):
        assert result == baseline.predict_text(text)
    oov = []
    for name, text in [("unseen_latin_ngrams", "qzxvjk " * 9), ("unseen_han_ngrams", "龘" * 100)]:
        old, new = baseline.predict_text(text), current.predict_text(text)
        assert old["decision"] == "human" and not old["out_of_scope"]
        assert new["risk_score"] is None and new["applicability"] == "insufficient_model_features"
        oov.append({"fixture": name, "old_score": old["risk_score"], "old_decision": old["decision"],
                    "new_score": new["risk_score"], "new_applicability": new["applicability"]})
    jobs = [
        ("english_40_paragraphs", lambda: baseline.predict_text_segments(normal[-3]), lambda: current.predict_text_segments(normal[-3])),
        ("chinese_60_paragraphs", lambda: baseline.predict_text_segments(normal[-2]), lambda: current.predict_text_segments(normal[-2])),
        ("mixed_48_paragraphs", lambda: baseline.predict_text_segments(normal[-1]), lambda: current.predict_text_segments(normal[-1])),
        ("batch_128_passages", lambda: [baseline.predict_text(t) for t in batches], lambda: current.predict_text_batch(batches)),
        ("english_40_unique_paragraphs", lambda: baseline.predict_text_segments(unique_documents[0]), lambda: current.predict_text_segments(unique_documents[0])),
        ("chinese_60_unique_paragraphs", lambda: baseline.predict_text_segments(unique_documents[1]), lambda: current.predict_text_segments(unique_documents[1])),
    ]
    timings = []
    for name, old, new in jobs:
        before, after = measure(old, args.repeats), measure(new, args.repeats)
        timings.append({"fixture": name, "before": before, "after": after,
                        "speedup": round(before["median_ms"] / after["median_ms"], 2)})
    result = {"status": "passed", "baseline_commit": ref, "runtime": "CPU; existing trained models",
              "fixtures": "Generated prose and synthetic edge cases; no authorship accuracy labels",
              "timing_note": "First four workloads repeat four source passages; final two use distinct paragraphs. Both versions use identical inputs and trained weights.",
              "normal_document_cases": cases, "normal_batch_passages": len(batches),
              "zero_feature_correction": oov, "warm_repeats": args.repeats, "timings": timings,
              "model_sha256": {name: hashlib.sha256((ROOT / "models" / name).read_bytes()).hexdigest()
                               for name in ["text_detector.joblib", "text_detector_zh.joblib"]},
              "accuracy_evaluated": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "normal_cases": len(cases), "batch_passages": len(batches),
                      "zero_feature_correction": oov, "timings": timings}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
