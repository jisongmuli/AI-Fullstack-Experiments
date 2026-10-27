# Current detector and course model interfaces

The 2026-10-27 course chain trains `CourseCNN` and a Chinese character TF-IDF classifier on the new 2026 data splits. `scripts/train_course_2026.py` saves both trained and optimized artifacts to `models/course_20261027`; `course_models.py` loads these exact weights for `/api/course/image/detect` and `/api/course/text/detect`.

The advanced detector in `inference.py` keeps DEAR / MPU as the default. Select the course trained or optimized profile in the image/text pages to demonstrate the experiment-two → experiment-three → experiment-four deployment chain. Measured compression tradeoffs and all five experiments are documented in `docs/课程要求对照与验收_20261027.md`.

## Historical 2026-10-11 training interface

Run `train_and_optimize.py` from this directory or pass `--root` to train on the
experiment-one splits. It writes model artifacts under `models/` and metrics,
predictions, and benchmarks under `results/exp2_exp3/`.

The application entry point is `DetectorService(root)` in `inference.py`:

```python
service = DetectorService(project_root)
service.predict_image(image_bytes)
service.predict_text(text)
service.predict_text_batch([text, another_text])
```

Single-input methods return a JSON-serializable dictionary with `risk_score` (the
source-dataset calibrated AI-class score), `decision` (`human`, `ai`, or
`uncertain`), `threshold`, `signals`, and `limitations`. Scores are not proof of
authorship or provenance.

`predict_text_batch` returns predictions in input order, groups sparse feature
extraction by language, and bounds each batch to 32 inputs by default (maximum
64). Identical passages are reused within the call. The segment path uses this
batch method for paragraphs and the whole document. Inputs with zero vocabulary
features return a null score with `insufficient_model_features`; they are excluded
from scored coverage. See [the inference upgrade measurements](../../docs/文本推理算法升级_20261026.md).

The current runtime also routes Chinese passages to `predict_text_chinese()` and
normal-resolution images to two pretrained candidates. Non-teaching images with a
shortest side below 200 px abstain; 200–511 px images return experimental candidate
scores, and larger normal images return cross-source candidate scores. Normal
images retain an `uncertain` decision. See [the image-input audit](../../docs/普通图片与12GB显存验证.md).

For an acceptance audit, run `evaluate_acceptance.py` against independently
labeled prediction CSVs. It reports abstention coverage, false-positive and
false-negative rates with Wilson 95% intervals. Public-set results are saved in
`results/acceptance_public.json`:

```powershell
python code/models/evaluate_acceptance.py --dataset cifake=results/exp2_exp3/image_test_predictions.csv --dataset text_same=results/exp2_exp3/text_test_predictions.csv --dataset text_external=results/exp2_exp3/text_external_test_predictions.csv --output results/acceptance_public.json
python -m unittest discover -s code/models/tests -v
```

No pass/fail gate is applied without predeclared limits. A target-platform
blind-test review must supply `--max-fpr`, `--max-fnr`, and `--min-coverage`
together; `--min-per-class` defaults to 100. These statistical checks do not
replace data-rights, safety, privacy, or operational acceptance.
Leave `risk_score` empty for a labeled out-of-scope case; the audit counts it as
an abstention in the coverage denominator. Split source, language, and time
cohorts into separate named CSVs to expose subgroup failures.
