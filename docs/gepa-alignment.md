## How this implementation aligns with the GEPA paper

This project implements the essential shape of GEPA:

- Reflective prompt evolution: we build a compact batch report from traces and ask a reflector LLM to propose a new instruction. The result replaces the module instruction if it improves minibatch mean.
- Pareto-based selection: we select candidates by frequency on instance-wise max sets and remove dominated ones to maintain diversity.
- System-aware merge: optional crossover that merges module instructions from two candidates and keeps the better child if it improves on a minibatch.
- Lineage and evaluation: each accepted child is evaluated on a fixed Pareto set and logged as a new candidate with a parent link.

### Differences vs the paper (for clarity)

- Scale and tasks: this repo ships with small toy datasets (Hotpot toy, five horror prompts) for affordability; the paper reports large tasks and bigger budgets.
- Feedback richness: our reflector uses a short batch report; the paper emphasizes rich linguistic feedback. You can extend `make_batch_report()` to include judge rationales or curated critiques.
- Metrics: Hotpot uses EM; horror uses an LLM judge (optionally human). Paper results vary per task/metric.
- Visualization and HITL: we add a realtime dashboard and a rating endpoint to anchor the judge—extra tooling beyond the paper.

Overall, algorithmically this is faithful to GEPA’s core: mutate → test → select with Pareto-driven exploration and natural-language reflection for prompt updates.