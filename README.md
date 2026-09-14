# Agent Reliability Harness

LLM agents that call tools can change real state: they search, book, and write records. A single successful trace is not evidence that the system is reliable. The same request can fail on a later run, under a paraphrased prompt, or when a tool is slow or returns an error. Unlike a chat hallucination, a bad tool call leaves a durable, incorrect outcome.

This repository is a harness for that measurement. A live OpenAI model uses LangGraph against a simulated airline: each trial gets its own SQLite catalog, with search and book as tools. Tasks are YAML booking constraints. A trial passes when the catalog has a confirmed reservation that matches the task. Reliability is measured by repeating each task *K* times on a baseline run, then under paraphrases, added latency, and injected tool failures, with optional checkpoint/retry (`--full`). Scores are listed below.

## Metrics

| Metric | Meaning |
| --- | --- |
| Consistency | Stability of pass/fail across identical repeats |
| Trajectory mix / order | Similarity of tool bags and call order across repeats |
| Resource stability | Stability of tool-call count across repeats |
| Robustness | Pass-rate under paraphrase, latency, and tool faults relative to baseline |
| Predictability | Calibration of post-run confidence vs actual success (Brier) |
| Early-failure AUROC | Whether a bad first tool call predicts trial failure |
| Bounded severity | Cost of a failure (e.g. missed search vs wrong booking) |

`--full` compares checkpoint/retry on vs off. The difference is largest under injected `book_flight` failures.

## Layout

| Path | Role |
| --- | --- |
| `agent/` | LangGraph ReAct agent + airline tools |
| `store/` | Schema and seed (~200 flights, airports, existing PNR clutter) |
| `tasks/` | Five booking jobs as YAML |
| `perturbation/` | Prompt paraphrases; latency and tool-fault wrappers |
| `mitigation/` | Snapshot search; retry book on failure |
| `harness/` | Trial runner, JSONL/CSV logs |
| `metrics/` | Score table + plot |
| `experiments/` | `python -m experiments.run_suite` |
| `results/` | `logs/`, `csv/`, `plots/` (generated, not committed) |

## Setup

Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # OPENAI_API_KEY
```

Catalog only (no LLM):

```bash
python -m agent.store
```

One live agent turn:

```bash
python -m agent.graph
```

Rebuild the seed with `python store/generate_seed.py`.

## Run the suite

Default: all five tasks, baseline only, `k` from config.

```bash
python -m experiments.run_suite --k 1 --task flight_booking_sfo_jfk
python -m experiments.run_suite --k 2
```

Full matrix (baseline / paraphrase / latency / tool-fault × checkpoint on/off). Start with one task:

```bash
python -m experiments.run_suite --k 1 --full --task flight_booking_sfo_jfk --quiet
```

`--concurrency N` for parallel trials. Traces → `results/logs/`. Scores → `results/csv/`. Pass-rate plot → `results/plots/`.
