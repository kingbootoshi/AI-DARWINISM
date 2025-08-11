Totally doable—there are a few levels to “seeing everything,” from quick console prints to full clickable traces. Pick the one that matches how deep you want to go.

# 1) Quick: print the last prompts, messages, and outputs

```python
import dspy

# ... run your program once or a few times ...

# Show the last 5 LLM calls across your app:
dspy.inspect_history(n=5)
```

`inspect_history` dumps the exact system/user/assistant messages DSPy sent to the model (including the field markers like `[[ ## answer ## ]]`, and, if present, any “reasoning” content). ([DSPy][1])

For raw access (including token usage & cost), you can also read the LM’s in-memory log:

```python
lm = dspy.LM("openai/gpt-4o-mini")
dspy.settings.configure(lm=lm)

# ... run your program ...

print(lm.history[-1].keys())  # -> dict_keys(['prompt','messages','kwargs','response','outputs','usage','cost'])
```

Each entry includes the prompt/messages, the raw response, parsed outputs, and usage. ([DSPy][2])

# 2) Turn on console logging (more verbose)

```python
import dspy
dspy.enable_logging()            # enable DSPy’s internal event logging
dspy.enable_litellm_logging()    # turn on LiteLLM DEBUG logs for provider calls
```

The first prints DSPy events; the second emits provider-level request/response logs (handy, but noisy). ([DSPy][3])

# 3) See the model’s “reasoning” live (streaming)

Two ways to surface intermediate thoughts:

**A. Ask for a `reasoning` field and stream it**

```python
import dspy

dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini", cache=False))

predict = dspy.Predict("question -> answer, reasoning")
listener = dspy.streaming.StreamListener("reasoning")
stream_predict = dspy.streamify(predict, stream_listeners=[listener])

async for chunk in stream_predict(question="Why did the chicken cross the road?"):
    print(chunk)  # you'll see chunks of the reasoning as they arrive
```

`streamify` + `StreamListener` lets you stream any string output field (like `reasoning`) in real time. ([DSPy][4])

**B. Use `ChainOfThought` (adds a reasoning field for you)**

```python
class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

cot = dspy.ChainOfThought(QA)
pred = cot(question="…")
print(pred.reasoning)  # full reasoning
```

Modules like `ChainOfThought` automatically include a `.reasoning` field you can print or stream. ([DSPy][5])

# 4) Full, clickable traces (best for teams & prod)

If you want a proper UI that shows every step (LM calls, tools, inputs/outputs, timings), wire up **MLflow Tracing**:

```python
import mlflow, dspy

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("DSPy")
mlflow.dspy.autolog()  # turn on DSPy tracing

# ... run your DSPy program ...
```

Open the MLflow UI and look under **Traces** to drill into each call, tool, and prompt. ([MLflow][6], [DSPy][7])

# 5) Optional dashboards

Prefer hosted dashboards? DSPy has cookbook-level integrations with **Langfuse** and **Langtrace** for rich tracing/observability views. ([Langfuse][8], [Langtrace][9])

---

## Handy tips / gotchas

* **Disable cache while debugging:** `dspy.LM(..., cache=False)` to ensure every run actually hits the model and shows up in logs. ([DSPy][10])
* **Reasoning models (OpenAI o-series):** If you use them, DSPy expects `temperature=1.0` and a large `max_tokens` when constructing `dspy.LM(...)`. ([DSPy][10])
* **Usage tracking on predictions:** Add `dspy.settings.configure(track_usage=True)` to attach token usage to each `Prediction` (also aggregated automatically inside modules). ([DSPy][11])
* **Verbose provider logs can include payloads.** Keep secrets redacted when sharing logs. ([DSPy][12])

If you tell me how you’re running the app (notebook, API server, Streamlit, etc.), I can drop in an exact snippet that prints or streams the reasoning where you need it.

[1]: https://dspy.ai/api/utils/inspect_history/ "inspect_history - DSPy"
[2]: https://dspy.ai/learn/programming/language_models/?utm_source=chatgpt.com "Language Models - DSPy"
[3]: https://dspy.ai/api/utils/enable_logging/ "enable_logging - DSPy"
[4]: https://dspy.ai/api/utils/streamify/ "streamify - DSPy"
[5]: https://dspy.ai/learn/programming/signatures/?utm_source=chatgpt.com "Signatures - DSPy"
[6]: https://mlflow.org/docs/2.21.3/tracing/integrations/dspy?utm_source=chatgpt.com "Tracing DSPy | MLflow"
[7]: https://dspy.ai/tutorials/observability/ "Debugging & Observability - DSPy"
[8]: https://langfuse.com/docs/integrations/dspy?utm_source=chatgpt.com "DSPy - Observability & Tracing - Langfuse"
[9]: https://www.langtrace.ai/blog/announcing-dspy-support-in-langtrace?utm_source=chatgpt.com "Monitoring & Tracing DSPy with Langtrace"
[10]: https://dspy.ai/api/models/LM/ "LM - DSPy"
[11]: https://dspy.ai/learn/programming/modules/?utm_source=chatgpt.com "Modules - DSPy"
[12]: https://dspy.ai/api/utils/enable_litellm_logging/ "enable_litellm_logging - DSPy"
