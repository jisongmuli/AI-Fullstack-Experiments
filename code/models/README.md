# Experiment 2/3 model interface

Run `train_and_optimize.py` from this directory or pass `--root` to train on the
experiment-one splits. It writes model artifacts under `models/` and metrics,
predictions, and benchmarks under `results/exp2_exp3/`.

The application entry point is `DetectorService(root)` in `inference.py`:

```python
service = DetectorService(project_root)
service.predict_image(image_bytes)
service.predict_text(text)
```

Each method returns a JSON-serializable dictionary with `risk_score` (the
source-dataset calibrated AI-class score), `decision` (`human`, `ai`, or
`uncertain`), `threshold`, `signals`, and `limitations`. Scores are not proof of
authorship or provenance.

The current runtime also routes Chinese passages to `predict_text_chinese()` and
normal-resolution images to a separate pretrained candidate. Images with a shortest
side below 512 px abstain; larger normal images return an unvalidated candidate
score with an `uncertain` decision. See [the 12 GB GPU and image-input audit](../../docs/普通图片与12GB显存验证.md).

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
