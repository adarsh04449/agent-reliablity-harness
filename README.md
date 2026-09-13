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
| `tasks/` | Task YAML (route, passenger, constraints) |
| `store/` | SQLite schema + seed (airports, flights, reservations) |
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

## Verify the catalog (no LLM)

From the repo root:

```bash
python -m agent.store
```

Each trial clones `store/seed.sql` into a private in-memory SQLite DB (200 flights, SFO/JFK/LAX, existing reservations). Success is **exactly one** confirmed reservation for the passenger that matches the task (SFO→JFK on 2026-10-15, morning, ≤ $400). `UA100` and other qualifying morning fares (e.g. `DL1197`, `DL220`) pass. Evening or over-budget ids (`UA900`, `B6200`) are `wrong_booking`. A second book for the same passenger is rejected. Regenerate the seed with `python store/generate_seed.py`.

## One real agent run (needs API key)

```bash
python -m agent.graph
```

Prints success, outcome, booked id, and post-run confidence. Uses `gpt-4o-mini` at temperature 0.

## How to run

Real OpenAI trials (needs `OPENAI_API_KEY`). Default is **baseline only**, `k` from config (2).

```bash
python -m experiments.run_suite --k 2
```

Full matrix (4 conditions × checkpoint on/off):

```bash
python -m experiments.run_suite --k 3 --full
```

Writes JSONL under `results/logs/`, trial CSV under `results/csv/`, reliability table + pass-rate plot under `results/csv/` and `results/plots/`.

## Four core metrics

| Metric | Meaning |
| --- | --- |
| Consistency | Same task, identical repeats: does pass/fail stay stable? |
| Robustness | How much pass rate drops under paraphrase / latency / tool faults |
| Predictability | After the run, “how sure are you?” → Brier (paper-style) |
| Bounded severity | How badly a failure cascades (e.g. wrong booking vs missed search) |

Checkpoint/rollback is compared on vs off, especially under tool failure. Extra paper metrics and τ-bench are stretch; see PLAN.md.
