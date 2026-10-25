# Experiments 4-5: local API and Agent

From the project root, install the API dependencies and start the server. The opt-in LLM preview also needs the separate LLM requirements:

```powershell
python -m pip install -r code/server/requirements.txt
python -m pip install -r code/llm/requirements.txt
python -m uvicorn app:app --app-dir code/server --host 127.0.0.1 --port 8000
```

The trained inference module and artifacts from experiments 2-3 must be present for scores. If they are missing, detection returns HTTP 503 and `GET /api/models` reports `available: false`. The editor works without models. Interactive OpenAPI documentation is at `http://127.0.0.1:8000/docs`. The requirements pin scikit-learn 1.9.1 because the saved text estimator was created with that version.

| Endpoint | Input | Output |
| --- | --- | --- |
| `GET /health` | none | Liveness status |
| `GET /api/models` | none | Artifact availability, manifest, SHA-256 fingerprints and applicability scope |
| `POST /api/image/detect` | multipart `file` | AI risk score, decision, evidence, applicability |
| `POST /api/text/detect` | JSON `{"text":"..."}` | AI risk score, decision, evidence, applicability |
| `POST /api/document/jobs` | multipart `file` or form `text` | 202 with a document task ID |
| `GET /api/document/jobs` | none | Bounded task metadata, without source text |
| `GET /api/document/jobs/{job_id}` | task ID | Stage, progress, report ID or error |
| `POST /api/document/jobs/{job_id}/cancel` | task ID | Cooperative cancellation status |
| `POST /api/text/edit` | JSON `{"text":"..."}` | Conservative edits, suggestions and meaning checks |
| `POST /api/text/edit/llm-preview` | JSON `{"text":"..."}` | Opt-in LoRA draft or explicit rejection, never automatic replacement |
| `POST /api/agent/run` | JSON `{"text":"...","image_base64":"...","edit":true,"llm_preview":false}` | Separate tool findings, optional LLM preview, call log and cautious conclusion |

The API rejects POST bodies over 35 MB before endpoint parsing, including requests without a `Content-Length` header. The image endpoint accepts static JPG/PNG files up to 25 MiB and 25 million pixels. A 32 x 32 RGB image uses the CIFAKE teaching model; normal images with shortest side at least 512 px use a separate 224 x 224 candidate model. Because external false-positive rates are high, normal-image responses always return `decision: uncertain`; their `risk_score` is an uncalibrated research output, not a platform decision. Smaller thumbnails abstain with a null score. The text endpoint accepts up to 10,000 characters; short English essays (23-587 characters) and Chinese passages (80-4000 characters with at least 40 Han characters) route to separate models. Chinese results are marked `unverified_platform`. These are input guards, not claims that all in-range inputs are reliable. See [the image and GPU note](../../docs/普通图片与12GB显存验证.md) for external evaluation and GPU measurements.

The Agent uses deterministic, allowlisted tool routing. When `llm_preview` is true, it additionally calls the actual local SmolLM2-135M-Instruct + CoEdIT LoRA adapter. The language model generates an editing candidate; it does not autonomously plan which tools to call. The workflow records tool status and elapsed time without copying the input into the log, and never combines unpaired image/text scores into one number. The rule editor applies limited redundant-phrase changes and flags vague claims. The LLM preview accepts English passages of 20-320 characters and rejects unchanged output, explanatory text, abnormal length, changes to numbers, negations, quotations, acronyms or alphanumeric identifiers, or low lexical overlap. Passing these surface checks is **not** proof of semantic equivalence; human review remains necessary. Neither editor is a detector-evasion service.

Run the integration tests from the project root:

```powershell
python -m unittest discover -s code/server/tests -v
```

The suite covers API behavior, Agent orchestration, report retention, document tasks, shared capacity and cancellation with predictable stand-in outputs. Run `scripts/document_jobs_smoke.py` for generated-input checks using actual local models. Tasks retain their processing slot after the 202 response until the worker finishes. Cancellation is checked between extraction and model calls; completed tasks reference the existing report endpoints. See `docs/文档任务升级_20261025.md` for the workflow and verification records. The server binds to localhost and accepts only loopback Host headers.
