# Desic

**English** · [Türkçe](README.tr.md)

> **Experimental research software:** Desic makes no production-safety or end-to-end decision-risk guarantees. The API has no authentication; do not expose the service to the internet unprotected. The student's risk budget does not guarantee the error rate of final answers, including teacher and rule answers. Exact counterfactual state is not yet guaranteed when learning continues after a retraction or after a restart; see the G1–G4 table in [`RESEARCH.md`](RESEARCH.md).

> **Research thesis and evidence status:** [`RESEARCH.md`](RESEARCH.md): immediate adaptation, bounded learning blast radius, explicit decision risk, reversible state evolution; measurements for every stage live under [`benchmarks/`](benchmarks/).

**A self-learning, calibrated typed-decision engine (System One).**
Like Jev and Laya, Desic does not generate free text: it takes a `state` (text or JSON) plus typed questions and returns a **probability distribution** over the allowed answers. The difference: it **learns instantly** from every piece of user feedback, **abstains** when unsure, and when needed consults a "teacher" (System 2) running on your own LLM key, then learns from it too.

```
                     ┌──────────────── Dashboard (real time, WebSocket) ──────────────────┐
                     │ playground · calibration · review queue · feedback log · versions   │
                     └───────────────┬─────────────────────────────────────▲──────────────┘
 your service ─ POST /v1/decide ────►│ state + { question: choice | score | noul }        │
                                     ▼                                                    │
                        fixed rules ──no match──► STUDENT (System 1)                      │
                                                  prior · linear · tree · memory          │
                                                  (+ neural: Laya-style encoder)          │
                                                  → Hedge mixture → temperature calib.    │
                                                  → familiarity → confidence / abstain    │
                                  if abstain ──► TEACHER (System 2): Claude,              │
                                                  an OpenAI-compatible LLM or a           │
                                                  Jev-compatible API (e.g. Laya)          │
                                                  answer is served + taught to student
 your service ─ POST /v1/feedback ──► immutable feedback log ─► student updates instantly
                                      (retract · rebuild · snapshot · rollback)
```

## Quick start

```bash
pip install -e .            # Python 3.10+
desic demo                  # 3 questions: support tickets (text) + loan applications (JSON)
desic serve                 # http://127.0.0.1:8000
```

In the **Playground**, pick the "Support ticket" example and press **Decide**. Then simulate real users from another terminal; the learning curve and reliability diagram on the question page update live:

```bash
python examples/feedback_simulator.py --n 1000 --drift 500   # the loan policy changes after decision 500
```

Teacher (optional): enter a provider and key on the dashboard's **Teacher** page, or:

```bash
ANTHROPIC_API_KEY=sk-ant-... desic serve                       # Claude (default: claude-opus-5-5)
DESIC_TEACHER_PROVIDER=openai DESIC_TEACHER_BASE_URL=https://api.openai.com/v1 \
DESIC_TEACHER_MODEL=<model> DESIC_TEACHER_API_KEY=... desic serve
DESIC_TEACHER_PROVIDER=jev_compatible DESIC_TEACHER_BASE_URL=http://localhost:9000/v1/decide desic serve  # e.g. self-hosted Laya
```

Neural student (optional): `pip install -e '.[neural]'` → **Neural** page on the dashboard.

### No computer needed: GitHub Codespaces (works from a phone too)

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/KamelotWalker/Desic)

On the repo page choose **Code → Codespaces → Create codespace** (or use the badge above). Installation, demo data and the server start automatically via `.devcontainer/devcontainer.json`; port 8000 is forwarded as **private**, so the address only opens with your GitHub account. The address appears in the **Ports** tab (`https://<codespace-name>-8000.app.github.dev`). Stop the codespace when you are not using it (saves free quota).

Docker: `docker build -t desic . && docker run -p 127.0.0.1:8000:8000 -v desic-data:/data desic`

This example publishes the port only on the host's localhost interface. The server inside the container listens on `0.0.0.0`; using `-p 8000:8000` opens the port on every host interface. If you need remote access, put an authenticated access layer in front; keep the Codespaces port **private**.

## API (Jev request format)

```bash
curl -s -X POST localhost:8000/v1/decide -H 'Content-Type: application/json' -d '{
  "state": "I was charged twice on my card, I need an urgent refund",
  "questions": {
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"billing": "payments, refunds", "technical": "bugs", "sales": "pricing"}},
    "severity":   {"type": "score",  "criteria": {"low": "cosmetic", "medium": "workaround exists", "high": "blocking"}},
    "urgent":     {"type": "noul",   "instructions": "The customer needs a response today."}
  },
  "explain": true
}'
```

```jsonc
{
  "id": "dec_…",
  "answers": {
    "department": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.97, "sales": 0.02, "technical": 0.01},
                   "confidence": 0.97, "abstain": false, "source": "student", "explanation": {…}},
    "severity":   {"type": "score", "score": 0.4, "level": "low", "probabilities": {…}, …},
    "urgent":     {"type": "noul", "probability": 0.95, "answer": true, …}
  }
}
```

- A new question name is registered automatically on first use; for a known question, sending `{}` is enough.
- A `choice` question can be taught a **new answer** through feedback (open world); `score` levels are fixed.
- `escalate`: `auto` (the question's teacher setting) · `never` · `always`. `abstain_threshold` can be overridden per request.

Once the real outcome is known:

```bash
curl -s -X POST localhost:8000/v1/feedback -H 'Content-Type: application/json' \
  -d '{"decision_id": "dec_…", "answers": {"department": "billing", "urgent": true, "severity": 2}}'
```

| Endpoint | Description |
|---|---|
| `POST /v1/decide` · `POST /v1/feedback` · `GET /v1/decisions/{id}` | decide · feedback · decision record |
| `GET/POST /v1/questions` · `GET/PATCH/DELETE /v1/questions/{name}` | questions; settings: `abstain_threshold`, `teacher_mode` (`off`/`on_abstain`/`always`), `teacher_weight` |
| `POST …/{name}/learn` | batch learning from labelled examples `{"examples": [{"state": …, "answer": …}]}` |
| `GET …/{name}/decisions?pending=true&uncertain_first=true` | live stream / active-learning queue |
| `PUT …/{name}/rules` | fixed rules (JSON path, e.g. `applicant.credit_score`, or `$text`) |
| `GET …/{name}/feedback` · `POST /v1/feedback/{id}/retract` · `POST …/{name}/rebuild` | immutable log · retract a wrong label · rebuild from the log |
| `GET/POST …/{name}/snapshots` · `POST …/{name}/rollback` | versions and rollback |
| `GET/PUT/DELETE /v1/teacher` · `POST /v1/teacher/test` | teacher settings (key kept in memory only) |
| `POST /v1/datasets` · `POST /v1/datasets/{id}/train` · `POST /v1/datasets/{id}/distill` | upload · train · have the teacher label |
| `POST /v1/generate/design` · `POST /v1/generate/examples` | design a question with an LLM · generate examples (+ train) |
| `GET /v1/neural` · `PATCH /v1/neural/config` · `POST /v1/neural/train` | neural student status · settings · candidate training |
| `POST /v1/neural/checkpoints/{id}/{promote\|reject\|retire\|delete}` | checkpoint lifecycle |
| `GET /v1/jobs/{id}` · `WS /ws` | background jobs · live events |

Interactive reference: `http://127.0.0.1:8000/docs`. Directly from Python: `desic.service.Desic`.

## How the student works (`desic/core`)

Pure Python, no extra ML dependencies. One student per question:

| Part | What it does |
|---|---|
| **Features** (`features.py`) | Text: words, 5-character stems (a light stemmer for agglutinative languages such as Turkish), bigrams; case and ASCII folding (`şikayet` = `sikayet`). JSON: fields such as `customer.plan=pro`, standardised numbers. |
| **linear** expert | Online softmax regression, log loss + AdaGrad (a learning-rate floor keeps it adapting to drift). Also learns from the teacher's *soft* probabilities. |
| **tree** expert | Hoeffding tree (VFDT) over JSON fields + DDM drift detector; thresholds such as `credit_score > 1445`. |
| **memory** expert | Nearest labelled examples (cosine with an inverted index). **Hot layer**: a single correction immediately changes the next answer for the same or a similar state. |
| **prior** expert | Base rates. |
| **Mixture** (`experts.py`) | Hedge on log loss (η=1 → Bayesian mixture), with *sleeping experts* and *fixed share* for fast recovery after drift. Log loss is a strictly proper scoring rule: experts are rewarded for honest probabilities. |
| **Calibration** (`calibration.py`) | Online temperature scaling over the last 500 labels; bounded to T ∈ [0.5, 4] (so a temperature learned on familiar data does not produce overconfidence on never-seen input). |
| **Familiarity** | How much of the evidence in the state was seen in training? Below 50%, the distribution is pulled towards "I don't know" → abstains → goes to the teacher. |
| **Measurement** | Prequential: every human label is scored against the probabilities *shown* to the user before it is learned. Accuracy, NLL, Brier, ECE, RPS (score), reliability diagram, risk–coverage, abstain and teacher rates. Teacher labels do not count towards the metrics. |

## Neural student (Laya-style, `desic/neural`)

The online experts are the **fast layer**: they learn instantly from every piece of feedback. The neural student is the **slow layer**: an encoder + decision head in the Laya architecture, trained periodically from the feedback log.

```
[CLS] choice: Which team? [MASK] billing: payments [MASK] technical: bugs [MASK] sales: pricing [SEP] <state> [SEP]
        │ encoder (ModernBERT-large / mmBERT / any HF encoder / built-in small encoder)
        │ 2-layer decision transformer
        ├─ option scorer on each [MASK] → softmax → distribution over this question's answers
        └─ act/escalate head on [CLS] → P(top answer is correct)
```

- **Data:** all non-retracted labels, one per (question, state); human > dataset > teacher. Teacher probabilities are soft targets (distillation).
- **Loss:** log score (strictly proper) + RPS for ordinal questions + act head; an optional **RLCD-style** stage (Gaussian noise on logits, group sampling, log-score reward, REINFORCE with a group-mean baseline). Separate learning rates for encoder and head.
- **Calibration:** per-question temperature on the validation slice.
- **Gates:** (1) *offline* — on a hash-fixed test slice that no checkpoint is trained on, it must beat base rates and not lose to the active checkpoint; (2) *shadow* — predicts on the next N labelled decisions without being served, and is promoted if its log loss is close to what users were served; (3) *active* — joins each question's mixture as the `neural` expert; Hedge decides per question how much to trust it. It can be retired on demand.
- **Backbone:** `scratch` (no download, seconds on CPU), `answerdotai/ModernBERT-large` (Laya English), `jhu-clsp/mmBERT-base` (100+ languages including Turkish), or any HF encoder id / local path. Pretrained encoders are downloaded from Hugging Face on first use and need a GPU.
- **Safety note:** the built-in (not pretrained) encoder can give 99% confidence even to meaningless text — the classic out-of-distribution overconfidence of neural networks. That is why the "familiarity" guard is only relaxed while a **pretrained** encoder is active.

### Comparison with Jev / Laya

| | Jev | Laya | Desic |
|---|---|---|---|
| Interface | state + choice/score/noul | same | same (Jev request format) |
| Model | closed, hosted | ModernBERT + decision head (421M) | online expert mixture + optional Laya-style neural student |
| Zero-shot knowledge | strong | moderate | none: learns from the teacher + data |
| Learning from user feedback | no (no customer fine-tuning) | offline fine-tune | **on every piece of feedback, instantly** |
| Calibration | RLCD | proper scoring + temperature | log-loss Hedge + online temperature + familiarity |
| Abstain / escalate | threshold in code | act/escalate head | first-class `abstain` + automatic teacher |
| Safety | — | — | immutable log, retract, rebuild, snapshot/rollback |

Desic can use Laya (behind a Jev-compatible server) as its **teacher**: Laya's zero-shot knowledge + Desic's instant learning.

## Benchmark: Banking77

77 banking intents, 10,003 train / 3,080 test (`python examples/benchmark_banking77.py [--neural] [--sorted]`, CPU, no GPU):

| Setup | Accuracy | Log loss | ECE |
|---|---|---|---|
| Online student, 10 examples per class | 57.2% | 1.70 | 0.122 |
| Online student, 50 examples per class | 82.6% | 0.62 | 0.011 |
| Online student, full data (94 s) | **88.5%** | 0.43 | **0.013** |
| Built-in neural student alone (6 epochs, 13.5 min) | 68.4% | 1.15 | 0.015 |
| Online + neural mixture | 88.3% | 0.44 | 0.035 |
| Online student, **stream sorted by class** (stress test) | 15.8% | 3.37 | 0.474 |

Selective prediction (online student, full data): answering only when ≥70% confident, it answers 85.7% of questions with 95.1% accuracy; at ≥90% confidence, 73.3% with 97.8% accuracy.

### Stream scenarios (`desic eval`)

Accuracy alone is not enough: how does the model learn along the stream, and how do wrong labels and drift affect it? The evaluation harness measures this with 7 scenarios × 3 seeds ([stage 02](benchmarks/stage-02-eval-harness/)):

| Scenario | Result |
|---|---|
| 1% / 5% wrong labels | Accuracy 88.5% / 87.9% (robust); ECE 0.085 at 5% |
| Burst of 30 wrong labels | Victim class 77% → 1%, 3–5 points of damage to other classes too; undoing = replaying the whole log (5,000 events) |
| Meaning of 10 classes changes (drift) | Half-recovery takes 3,000 labels; the drift detector does not fire |
| Cold start with a teacher | Teacher dependence 95% → 46% |

**Reversible patch layer** (`desic/core/patches.py`, [stage 03](benchmarks/stage-03-reversible-patches/), [stage 04](benchmarks/stage-04-service-integration/)): every label first becomes a local, reversible patch and moves into the permanent model after a probation period (10% of labels, 20–500). Every question goes through this layer; **Retract** and **Undo last N** on the dashboard take effect instantly. Compared with the base model on the same streams:

| | Base model | Patch layer |
|---|---|---|
| Mixed stream | 88.75% | 89.02% |
| Sorted stream (forgetting index) | 13.7% (0.63) | **80.0% (0.10)** |
| 30 wrong labels: damage to other classes | 3.8 points | **0.6 points** |
| Undoing those labels | 29.6 s, 5,001 events replayed | **0.16 s, 0 events** |
| Drift half-life | 3,000 labels | 3,417 labels (worse) |
| Repeating a wrong label on the same message | 7% | 11% (worse; 23% before [stage 06](benchmarks/stage-06-diagnosis/)) |

The undo measurements belong to the stated experimental streams; they do not guarantee a state identical to a system that never saw the label once learning continues or after a restart ([known architectural gaps](RESEARCH.md)).

**Decision contract** (`desic/core/contract.py`, [stage 07](benchmarks/stage-07-decision-contract/)): instead of a fixed threshold, a question can be given a *risk budget* (target error rate on the student's served answers, e.g. 5%) or *costs* (a wrong answer vs. asking the teacher; a cost matrix for asymmetric errors). In this experiment, budgets of 2% / 5% / 10% gave measured errors of 1.4% / 4.0% / 8.5%; these results do not guarantee an error bound on other streams or on final answers that include the teacher/rules. Every decision is recorded with its rule, threshold, propensity and policy version. There are two modes: *guaranteed* (a mode name; it relies on label evidence for student answers and asks the teacher more often while labels are scarce) and *expected* (the model's calibrated confidence + correction from labels; [stage 08](benchmarks/stage-08-expected-risk-budget/): 95% correct answers in the teacher scenario, escalations 93% → 71%). Because the teacher makes mistakes, more escalation can increase the error rate users see ([stage 11](benchmarks/stage-11-policy-and-served-risk/)).

Measurements, raw logs and JSON results for every stage: [`benchmarks/`](benchmarks/).

Reference: published results (Casanueva et al., 2020) put fine-tuned BERT / ConveRT with full data at about 93%. Desic has no pretrained language knowledge, so this gap is expected. The neural student with a pretrained backbone (mmBERT / ModernBERT) still needs to be measured on a GPU. The collapse on the sorted stream is a known weakness (catastrophic forgetting) and the first target of the next development line.

## Data

- **Your own data:** CSV/TSV/JSON/JSONL (≤ 50 MB). Pick the answer column and the state columns; the state becomes text (columns are joined) or a JSON object. Hold-out evaluation including calibration.
- **Distill:** the teacher labels unlabelled rows and the student learns from its probabilities (System 2 → System 1).
- **Generate:** describe the decision in plain text → the LLM designs the question (type, answers, state format, expert rules) → edit → examples are written in batches of 20 and validated against the schema → optionally trained right away.
- Keys are kept only in server memory (and, if you choose, in your browser); they are never written to the database or logs.

## Development

```bash
pip install -e '.[dev]'
pytest                      # 87 tests: core, API, teacher/generation, neural student (skipped without torch), evaluation
desic eval --seeds 1        # stream scenarios (Banking77)
desic serve --reload
```

```
desic/
  core/        features · experts · calibration · task · contract (decision contract) · patches (reversible layer) · tree · model (adaptive tree) · drift · rules
  eval/        data (Banking77) · metrics (prequential, selective risk, forgetting, half-life) · scenarios · run
  neural/      text (input format) · model (encoder + decision head) · train (losses, RLCD, calibration) · runtime (gates)
  service.py   decisions, feedback, teacher, jobs, snapshot/rebuild
  llm.py       teacher providers (Anthropic SDK, OpenAI-compatible, Jev-compatible)
  generation.py  LLM question design and example generation
  storage.py   SQLite: students, snapshots, decisions, feedback log, datasets
  api.py       FastAPI + WebSocket
  static/      dashboard (dependency-free HTML/CSS/JS)
```

## Roadmap

- A multilingual router for the neural student (per-language checkpoint selection like Laya Router) and fast inference with ONNX/quantization.
- A paired statistical test in shadow testing (currently mean log loss + margin).
- Multi-label and ranking primitives, hierarchical choice (255+ options).
- Speeding up the hot path (feature extraction, expert scoring) with C++/pybind11.
- Authentication and API keys (currently meant for local/trusted networks; `desic serve` listens only on `127.0.0.1` by default).

> Student states are stored with `pickle`; only load data folders you created yourself.

---

MIT License.
