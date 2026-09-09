# Agent Reliability Harness

A testing framework that measures how **consistently** an LLM agent succeeds at multi-step tasks under repeats and perturbations — not whether it can succeed once.

Inspired by [Towards a Science of AI Agent Reliability](https://arxiv.org/abs/2602.16666) (Rabanser et al., 2026). This repo is a small harness for a **real OpenAI agent** with simulated airline tools. See [PLAN.md](PLAN.md) for scope, metrics, and stretch work (including τ-bench).

## Layout

| Path | Role |
| --- | --- |
| `agent/` | LangGraph agent, mock booking tools, LLM factory |
| `perturbation/` | Paraphrases; latency and partial tool-failure wrappers |
| `harness/` | Config, trial runner, JSONL/CSV logging |
| `metrics/` | Consistency, robustness, predictability, bounded severity |
| `mitigation/` | Checkpoint / rollback |
| `experiments/` | Suite entrypoint (`run_suite`) |
| `tasks/` | Task YAML (flight booking) |
| `results/` | `logs/`, `csv/`, `plots/` (generated; not committed) |
| `PLAN.md` | Full project plan |

## Setup

Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# put OPENAI_API_KEY in .env
```

## How to run (once the suite exists)

**Real agent (the experiment):**

```bash
python -m experiments.run_suite --k 3
```

Requires `OPENAI_API_KEY`. This is the path you would cite.

**Wiring check only (not a reliability result):**

```bash
python -m experiments.run_suite --mock --k 2
```

Uses a deterministic fake LLM so you can test logging and metrics without API calls.

`run_suite` is not implemented yet (build step 1 is folders only).

## Four core metrics

| Metric | Meaning |
| --- | --- |
| Consistency | Same task, identical repeats: does pass/fail stay stable? |
| Robustness | How much pass rate drops under paraphrase / latency / tool faults |
| Predictability | After the run, “how sure are you?” → Brier (paper-style) |
| Bounded severity | How badly a failure cascades (e.g. wrong booking vs missed search) |

Checkpoint/rollback is compared on vs off, especially under tool failure. Extra paper metrics and τ-bench are stretch; see PLAN.md.
