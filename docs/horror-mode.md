## Horror mode (two‑sentence microfiction)

This task evolves a single instruction for a module named `horror_writer` that outputs a two‑sentence story from a short prompt. The metric is an LLM judge with a rubric, optionally overridden by human ratings.

### Dataset

- Default: five seed prompts in `horror_seed_data()`.
- You can replace them in code or extend with your own list.

### Metric

- Judge scores integer 0–5 for: scariness, suspense, originality, clarity, rule_two_sentences.
- Weighted sum normalized to 0..1.
- POST /rate with matching `story` overrides the LLM score using your numbers.

### Web UI helpers

- Baseline Outputs: all prompts → stories → scores before optimization.
- Live Outputs: minibatch parent/child comparisons and Pareto evaluations as the run progresses.

### Collecting human ratings

```
POST http://127.0.0.1:3000/rate
{
  "story": "The mirror smiled a beat too late...",
  "scariness": 4,
  "suspense": 5,
  "originality": 4,
  "clarity": 4,
  "rule_two_sentences": 5
}
```

The app uses your rating for exactly matching `story` strings whenever encountered.