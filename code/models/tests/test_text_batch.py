"""Sparse text batching, input scope, and source-offset regression checks."""
from __future__ import annotations

import sys
import unittest
import unicodedata
from pathlib import Path
from unittest.mock import patch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference import DetectorService, _calibrate


EN = "The team reviewed the dataset and recorded every result before publication."
ZH = "研究团队对样本数据进行了清洗，分别记录数据来源、标签质量、评估指标和交叉验证结果。" * 3


class BatchTextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def model(texts, analyzer, ngram):
            vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram)
            matrix = vectorizer.fit_transform(texts)
            return {"vectorizer": vectorizer, "classifier": LogisticRegression().fit(matrix, [0, 1])}
        cls.english = model([EN, "Artificial intelligence improves productivity across many industries and research fields."], "char_wb", (3, 5))
        cls.chinese = model([ZH, "人工智能技术通过机器学习的方法提高生产效率，为各个行业提供新的发展机会。" * 3], "char", (2, 4))

    def setUp(self):
        self.service = DetectorService.__new__(DetectorService)
        self.service.config = {"abstain_probability": .8, "text_calibration": {"coefficient": 3., "intercept": -.2}}
        self.service.zh_config = {"model_version": "fixture-v1", "calibration": {"coefficient": 2., "intercept": .1}}
        self.service.text_model = self.english
        self.service.zh_text_model = self.chinese
        self.service._english_feature_names = self.english["vectorizer"].get_feature_names_out()

    def test_language_routing_preserves_order_and_original_calibration(self):
        texts = [EN, ZH, EN + " The analysis used separate validation data.", ZH.replace("数据", "样本")]
        results = self.service.predict_text_batch(texts)
        for text, result in zip(texts, results):
            chinese = text.startswith("研究")
            model = self.chinese if chinese else self.english
            calibration = self.service.zh_config["calibration"] if chinese else self.service.config["text_calibration"]
            clean = unicodedata.normalize("NFKC", text) if chinese else text
            raw = model["classifier"].decision_function(model["vectorizer"].transform([clean]))[0]
            self.assertEqual(result["risk_score"], round(_calibrate(raw, calibration["coefficient"], calibration["intercept"]), 6))
            self.assertEqual(result["applicability"], "unverified_platform" if chinese else "english_no_valid_external_benchmark")

    def test_unique_inputs_use_bounded_batches_per_language(self):
        texts = [EN + " " + str(i) for i in range(5)] + [ZH + str(i) for i in range(3)]
        en, zh = self.english["vectorizer"], self.chinese["vectorizer"]
        with patch.object(en, "transform", wraps=en.transform) as e, patch.object(zh, "transform", wraps=zh.transform) as z:
            results = self.service.predict_text_batch(texts, batch_size=2)
        self.assertEqual(len(results), 8)
        self.assertEqual([len(c.args[0]) for c in e.call_args_list], [2, 2, 1])
        self.assertEqual([len(c.args[0]) for c in z.call_args_list], [2, 1])

    def test_identical_inputs_are_reused_without_sharing_mutable_results(self):
        vectorizer = self.english["vectorizer"]
        with patch.object(vectorizer, "transform", wraps=vectorizer.transform) as transform:
            results = self.service.predict_text_batch([EN, EN, EN])
        self.assertEqual(transform.call_args.args[0], [EN])
        results[0]["signals"].clear()
        results[0]["risk_score"] = None
        self.assertTrue(results[1]["signals"])
        self.assertIsNotNone(results[1]["risk_score"])

    def test_no_source_text_is_cached_between_requests(self):
        vectorizer = self.english["vectorizer"]
        with patch.object(vectorizer, "transform", wraps=vectorizer.transform) as transform:
            self.service.predict_text(EN)
            self.service.predict_text(EN)
        self.assertEqual(transform.call_count, 2)

    def test_single_segment_does_not_repeat_whole_document_inference(self):
        vectorizer = self.english["vectorizer"]
        with patch.object(vectorizer, "transform", wraps=vectorizer.transform) as transform:
            result = self.service.predict_text_segments(EN)
        self.assertEqual(transform.call_count, 1)
        self.assertEqual(transform.call_args.args[0], [EN])
        self.assertEqual(result["segments"][0]["risk_score"], result["risk_score"])

    def test_english_zero_feature_input_abstains_despite_classifier_bias(self):
        result = self.service.predict_text("qzxvjk " * 9)
        self.assertEqual(result["applicability"], "insufficient_model_features")
        self.assertIsNone(result["risk_score"])
        self.assertEqual(result["decision"], "uncertain")
        self.assertTrue(result["out_of_scope"])

    def test_chinese_zero_feature_input_abstains_in_both_entry_points(self):
        for predict in [self.service.predict_text, self.service.predict_text_chinese]:
            result = predict("龘" * 100)
            self.assertIsNone(result["risk_score"])
            self.assertEqual(result["applicability"], "insufficient_model_features")
            self.assertEqual(result["model_version"], "fixture-v1")

    def test_oov_segments_do_not_increase_scored_coverage(self):
        result = self.service.predict_text_segments(EN + "\n\n" + "qzxvjk " * 25)
        self.assertIsNone(result["segments"][1]["risk_score"])
        self.assertLess(result["segment_summary"]["scored_character_percent"], 80)
        self.assertEqual(result["applicability"], "partial_segment_coverage")
        self.assertIsNone(result["risk_score"])

    def test_context_blocks_do_not_run_vectorizer(self):
        texts = ["Please write an essay about science.", "> " + EN, "```python\nprint('hello')\n```"]
        vectorizer = self.english["vectorizer"]
        with patch.object(vectorizer, "transform", wraps=vectorizer.transform) as transform:
            results = self.service.predict_text_batch(texts)
        transform.assert_not_called()
        self.assertTrue(all(r["risk_score"] is None for r in results))

    def test_nfkc_normalization_keeps_original_segment_offsets(self):
        text = "  " + ZH + "ＡＢＣ１２３\n\n> " + EN + "  "
        result = self.service.predict_text_segments(text)
        for segment in result["segments"]:
            self.assertEqual(text[segment["start"]:segment["end"]], segment["text"])
        self.assertIsNotNone(result["segments"][0]["risk_score"])
        self.assertIsNone(result["segments"][1]["risk_score"])

    def test_invalid_batch_inputs_are_rejected_before_transform(self):
        for value in [0, -1, 65, True, 1.5]:
            with self.assertRaises(ValueError):
                self.service.predict_text_batch([EN], batch_size=value)
        with self.assertRaises(TypeError):
            self.service.predict_text_batch(EN)
        with self.assertRaises(TypeError):
            self.service.predict_text_batch([EN, None])
        with self.assertRaises(ValueError):
            self.service.predict_text_batch(["a" * 100001])
        self.assertEqual(self.service.predict_text_batch([]), [])


if __name__ == "__main__":
    unittest.main()
