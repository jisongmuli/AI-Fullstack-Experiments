# Local detection and editing API

Use Python 3.12 and the project virtual environment. Start from the project root:

```powershell
.venv\Scripts\python.exe run.py --port 8005
```

The four workflows are image detection, text detection, document analysis, and editing. Previous Agent routes have been removed and return 404.

| Endpoint | Purpose |
| --- | --- |
| `GET /health`, `GET /api/runtime` | Health, active requests and bounded report statistics |
| `GET /api/models` | Versions, SHA-256 fingerprints, scope and recent sample measurements |
| `POST /api/image/detect` | JPG/PNG analysis, content credentials and local sensitivity |
| `POST /api/text/detect` | Chinese/English analysis with original-text segment offsets |
| `POST /api/document/detect` | Synchronous document analysis |
| `POST /api/document/jobs` | Start a background document task, returning 202 |
| `GET /api/document/jobs`, `GET /api/document/jobs/{id}` | Task metadata and progress revision |
| `POST /api/document/jobs/{id}/cancel` | Cooperative cancellation |
| `DELETE /api/document/jobs/{id}` | Remove ended task metadata while preserving its report |
| `GET /api/document/reports`, `GET /api/document/reports/{id}` | List and reopen reports |
| `GET /api/document/reports/{id}/{format}` | Export docx, html or json |
| `DELETE /api/document/reports/{id}` | Delete report content |
| `POST /api/text/edit` | English clarity suggestions and surface-content checks |
| `POST /api/text/edit/llm-preview` | Local SmolLM2/LoRA editing preview |

The active policy is `models/algorithm_policy.json`. MPU v3 Short handles Chinese and English; Chinese calibration uses a separate NLPCC 2026 Phase 1 slice. DEAR uses native-scale crops and resolution-specific calibration. TASTE and earlier classifiers are retained for reproducing comparisons.

Images accept arbitrary positive dimensions through 4K, subject to 25 MiB / 25-million-pixel limits. Text input allows 10,000 characters. Documents allow 20 MiB / 100,000 characters / 200 PDF pages. Reports stay in memory for 30 minutes, limited to 20 reports / 32 MiB. Browser history stores summaries only.

Background jobs retain their shared processing slot until cleanup finishes. Identical paragraphs reuse inference within one document only. The source mapping is released when that analysis ends.

Run `python -m unittest discover -s code/server/tests -q`. Actual-model checks and evaluation records are described in `docs/20261026_模型升级与Agent移除.md`.
