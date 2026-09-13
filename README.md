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
| `tasks/` | Task YAML (flight booking goal + gold id) |
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

Each trial clones `store/seed.sql` into a private in-memory SQLite DB (200 flights, SFO/JFK/LAX, existing reservations). Correct-date SFO→JFK search still includes gold `UA100` (morning $349) plus distractors (`B6200`, `UA900`, `UA150`, `DL220`, and generated ids). Booking `UA100` for Alice Chen is success (reservation row + seat taken). Booking any other flight for her is `wrong_booking`. Regenerate the seed with `python store/generate_seed.py`.

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
