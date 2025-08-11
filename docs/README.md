## Darwinism: GEPA-in-DSPy

A practical, visual reproduction of the GEPA (Genetic Evolution Prompt Algorithm) loop, adapted to Python + DSPy with a lightweight web dashboard. You can evolve prompts for different tasks (currently: a two‑sentence horror microfiction writer) and watch the evolution as it happens.

### What this repo gives you

- GEPA loop: Pareto-based candidate selection, reflective prompt mutation, optional system-aware merge
- Two runnable tasks:
  - `horror` (two‑sentence horror microstories with an LLM judge and optional human ratings)
  - `hotpot` (toy HotpotQA pipeline with Exact Match metric)
- Realtime web UI (http://127.0.0.1:3000): leaderboard, prompts, last mutation, events, baseline outputs, live outputs
- Centralized settings and colorful logs (Loguru)

### Quickstart

```bash
python -m pip install -r requirements.txt

# Horror mode with web UI
python main.py --task horror --optimize --budget 20 --minibatch 3 --pareto 4 --web

# Optional: use paper prompts for Hotpot
python main.py --pdf ./2507.19457v1.pdf --use-paper-prompts --optimize --budget 60 --minibatch 6 --pareto 8 --web
```

Open the dashboard at http://127.0.0.1:3000

### CLI flags (common)

- `--task {hotpot,horror}`: which task to run
- `--optimize`: enable GEPA optimization loop (without it we only run a baseline)
- `--pareto N`: scoreboard size (candidate ranking set)
- `--minibatch N`: items per mutation attempt (child vs parent)
- `--budget N`: mutation attempts
- `--merge-prob p`: try a crossover merge with probability `p` each iteration
- `--web`: start the local web UI at port 3000

See docs/parameters.md for a deeper dive.

### How “good” is measured (horror)

- LLM judge scores each story on: scariness, suspense, originality, clarity, and a 2‑sentence rule.
- Weighted sum normalized 0..1 is the metric.
- Human-in-the-loop ratings (POST /rate) override the LLM judge per exact-story match.

### Web UI sections

- Leaderboard: one row per accepted candidate (full prompt set snapshot); Mean is Pareto average; checkmark marks the current best.
- Last Mutation: minibatch improvement that triggered accept/reject + old/new instruction.
- Selected Candidate Prompts: click a row to view its module prompts.
- Events: recent algorithm events (mutation attempts, accepts, merges).
- Baseline Outputs: all inputs/outputs/scores from the pre‑optimization pass.
- Live Outputs: stream of minibatch and Pareto evaluations during optimization.

### Environment

Configure via `.env` (see `.env.example`): provider API keys, model, logging prefs, MLflow URI, etc. Full list in docs/config.md.

### Next steps

- Add your own dataset/prompts or pre‑labeled human ratings (see docs/horror-mode.md)
- Swap the judge or metric
- Extend to multi‑module systems and use the Merge operator to combine strengths