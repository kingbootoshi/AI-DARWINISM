### Parameters reference

- **Task**: Selects pipeline + metric.
  - `hotpot`: multi-module QA, Exact Match metric.
  - `horror`: single-module writer, LLM judge metric with optional human override.

- **Model**: Backend LLM used by DSPy for modules: writer(s), judge, reflector.

- **Pareto (pareto_set_size)**: Size of fixed scoreboard `D_pareto` for candidate ranking.
  - Larger → more stable/generalizable ranking; more expensive.
  - Small → faster iteration; noisier.

- **Minibatch (minibatch_size)**: Items used per mutation attempt to compare parent vs child.
  - Larger → less noisy accept/reject decisions; higher cost per attempt.

- **Budget (rollout_budget)**: Number of mutation attempts.
  - Cost ≈ budget × (2 × minibatch) calls for parent/child + accepted × pareto for evaluation.

- **Merge Prob (merge_prob)**: Probability per iteration to attempt system-aware crossover.
  - `0` disables; `0.1–0.3` adds exploration and can discover combinations of good modules.

- **Seed**: RNG seed for shuffling and sampling.

- **Web**: Start the real-time dashboard at http://127.0.0.1:3000.