# Intro

* **What evaluation is and why we do it**
* **A pragmatic, step‑by‑step workflow** you can run today
* **How to do evaluation efficiently with LLM‑as‑a‑judge** (LLM judges) without burning time or tokens

The framing below follows Hugging Face’s take: there are **three complementary ways to evaluate LLMs** — **automatic benchmarks**, **human evaluation**, and **LLM‑as‑a‑judge**. You’ll often use all three at different stages. ([Hugging Face][1])

---

# Why do evaluation (“e‑vow”)?

1. **Pick the right model for your job.** Leaderboards/benchmarks tell you rough capability; your own evals tell you if it meets *your* quality bar. ([Hugging Face][2])
2. **De‑risk deployments.** You’ll catch regressions, bias, hallucinations, and safety issues before customers do. (HF’s evaluation blogs repeatedly stress standardization + reproducibility to avoid “benchmark theater.”) ([Hugging Face][3])
3. **Compare fairly and make changes confidently.** Small prompt tweaks or few‑shot order can move scores by non‑trivial amounts; reproducible runs and statistical tests prevent false wins. ([Hugging Face][4])
4. **Scale human judgment.** Human eval is gold‑standard but expensive; **LLM‑as‑a‑judge** approximates it well when set up carefully. ([Hugging Face][5], [arXiv][6])

---

# The step‑by‑step playbook

## 1) Define success before you run anything

* **Use‑case**: e.g., customer‑support QA, coding help, policy Q\&A, RAG answers.
* **Quality rubric**: 3–5 criteria you can explain to a non‑ML colleague (e.g., *correctness, helpfulness, groundedness, safety*).
* **Decision rule**: e.g., “Ship if ≥ 80% win‑rate vs. baseline with 95% confidence.”

> Why: this lets you choose the right mix of automatic metrics, human checks, and LLM judges later. HF’s “how and why we evaluate” post frames this well. ([Hugging Face][1])

## 2) Choose evaluation modes (often a combo)

* **Automatic benchmarks** (fast sanity checks, regression tests): MMLU, HellaSwag, GSM8K, TruthfulQA, etc.
* **Human evaluation** (final quality gates, safety): small but carefully‑designed samples.
* **LLM‑as‑a‑judge** (cheaply scale “human‑like” scoring): your daily driver for open‑ended tasks. ([Hugging Face][1])

## 3) Build or pick datasets

* **Coverage**: include easy, typical, and hard/edge cases.
* **Size**: a few hundred examples goes a long way for offline gating and A/Bs.
* **Cleanliness**: avoid training contamination if possible; keep a private hold‑out set.

## 4) Wire up **automatic benchmarks** quickly with LightEval

LightEval is HF’s **all‑in‑one toolkit** for running many tasks across multiple backends (Transformers, vLLM, TGI, OpenAI API, etc.), with saved, sample‑level results for debugging. ([Hugging Face][7])

**One‑liner example (local GPU/CPU via Accelerate)**

```bash
lighteval accelerate \
  "model_name=openai-community/gpt2" \
  "leaderboard|truthfulqa:mc|0|0"
```

* The `suite|task|few_shot|truncate_flag` syntax selects tasks; run multiple tasks with a comma list or a file.
* Use `--output-dir`, `--save-details`, and `--push-to-hub` to preserve full prompts, predictions, and metrics for reproducibility. ([Hugging Face][8])

## 5) Add **reference metrics** for structured tasks

For classification/extraction, use **accuracy/F1/EM**; for summaries/translations use **ROUGE/BLEU/BERTScore** (not perfect, but still useful); HF’s **Evaluate** library wraps dozens of metrics in one API. (For modern LLM eval, HF recommends LightEval.) ([Hugging Face][9])

## 6) Design your **LLM‑as‑a‑judge** track (the efficient workhorse)

LLM judges can rate answers with high agreement to humans—*if* you set them up right. HF’s cookbook walks through the whole process; key moves below. ([Hugging Face][5])

**6a. Calibrate the judge**

* Curate \~30–50 examples with **human ratings** using your rubric.
* Prompt a candidate judge model to score them; compute agreement/correlation vs. human labels. Iterate your prompt until agreement is solid. (The cookbook shows going from r≈0.56 to ≈0.84 by tightening the rubric, switching to a **small integer scale (1–4 or 1–5)**, and asking for a brief rationale.) ([Hugging Face][5])

**6b. Use good rubrics & parsing**

* Prefer an **additive rubric** (award points per criterion) and a **short justification**; parse output as structured JSON to avoid regex pain. ([Hugging Face][5])

**6c. Mitigate judge biases**

* **Position bias**: flip **A/B → B/A** and aggregate.
* **Verbosity bias**: cap output tokens; ask judge to penalize fluff.
* **Self‑enhancement bias**: don’t let a model judge outputs produced by itself/family when possible; or use multiple, diverse judges.
  Evidence and mitigations are discussed in **MT‑Bench/Chatbot Arena**. ([arXiv][6])

**6d. Prefer pairwise comparisons for ranking**

* For model selection, ask the judge: “Which is better, A or B, given the rubric?” then compute **win‑rates**; these are robust and easy to reason about. (This aligns with the Arena setup and much industry practice.) ([arXiv][6])

**Minimal judge prompt (skeleton)**

```
SYSTEM: You are a careful evaluator. You will choose the better answer strictly by the rubric.

RUBRIC (score 1–4):
1 = incorrect/irrelevant; 2 = partly helpful; 3 = mostly correct/helpful; 4 = fully correct, concise, well‑grounded.

TASK: Given a question and two answers, pick the better and explain briefly.

INPUT
Question: {question}
Answer A: {a}
Answer B: {b}

RESPONSE JSON:
{"winner": "A"|"B", "score": 1-4, "explanation": "<1–2 sentences>"}
```

(HF’s cookbook gives concrete, working examples and shows why smaller integer scales tend to work better.) ([Hugging Face][5])

## 7) Do the stats (just enough to be safe)

* **Confidence** on win‑rates: if A wins **≥ 55%** of \~**385** paired comparisons, that’s typically >50% at **95% confidence** (rule‑of‑thumb near p=0.5). For a **±3%** margin, target \~**1,068** pairs.
* Report **CIs** (Wilson/binomial) and sample counts alongside point estimates.
* For scalar scores, use **bootstrap CIs**; avoid naked p‑values with no effect sizes. (Wilson intervals are a standard way to interval‑estimate binomial proportions.) ([Wikipedia][10], [Qualtrics][11])

## 8) Make it reproducible

* Fix seeds, temperature=0, record **full prompts**, few‑shot examples and their **order**, model + tokenizer versions, and backend settings. LightEval’s `--save-details` dumps prompts/tokens per sample; push runs to the Hub/W\&B for auditability. ([Hugging Face][12])

## 9) Close the loop

* Turn failing cases into **new test items** (grow a private hold‑out).
* Re‑run the exact suite on every model/prompt change.
* Periodically refresh with newer/stronger tasks so you don’t overfit public benchmarks. (This was a big reason behind HF’s Open LLM Leaderboard v2 refresh.) ([Hugging Face][13])

---

# How to do evaluation **efficiently** with LLMs (practical tips)

**Pipeline design**

* **Two‑stage judging**: cheap judge (or heuristics) to filter easy wins/losses → send **disagreements** or “close calls” to a stronger judge.
* **Pairwise before scalar**: use pairwise wins to winnow contenders; run expensive scalar scoring only on finalists.

**Token + time control**

* Keep prompts lean: only the task, rubric, and answers.
* Limit judge output (e.g., 50–80 tokens) and **require JSON**.
* Batch prompts (LightEval/vLLM backends help). ([Hugging Face][8])

**Bias & variance control**

* Always randomize A/B order and flip once.
* Sticky determinism: temperature=0; set/record seeds; don’t change chat templates mid‑experiment. (LightEval exposes backend knobs and logs them.) ([Hugging Face][8])

**Judge quality**

* Calibrate on a **small, human‑rated set** each time you change rubric/model.
* Track **judge↔human correlation** over time; if it slips, revisit the rubric or switch judges. HF’s cookbook shows a concrete calibration flow. ([Hugging Face][5])

**Where LightEval fits**

* Unified CLI across backends, easy task lists, sample‑level details, and built‑in logging to Hub/W\&B/TensorBoard. Great for regression suites and for keeping evals **fast and reproducible**. ([Hugging Face][7])

---

# Quick “do‑this‑today” checklist

* [ ] Write a 3–5‑bullet **rubric** for your use‑case.
* [ ] Assemble **200–500** eval prompts (plus 30–50 with human ratings for judge calibration).
* [ ] Run **LightEval** on 3–6 automatic tasks for fast baselines; save details. ([Hugging Face][8])
* [ ] Stand up an **LLM judge** with the rubric; calibrate it on the human‑rated mini‑set; flip A/B order. ([Hugging Face][5])
* [ ] Compare contenders with **pairwise win‑rates**; target \~385 pairs for a \~±5% decision.
* [ ] Ship if you meet your pre‑declared **decision rule**; otherwise tighten prompts or swap models and re‑run the same suite.

---

## Handy links to keep open

* **Hugging Face LLM Evaluation Guidebook** (table of contents + pointers). ([GitHub][14])
* **LightEval docs** (index & quicktour). ([Hugging Face][7])
* **Saving & reading LightEval results** (reproducibility/debugging). ([Hugging Face][12])
* **LLM‑as‑a‑judge cookbook** (end‑to‑end, with code & prompt patterns). ([Hugging Face][5])
* **MT‑Bench / Chatbot Arena paper** (judge reliability & known biases). ([arXiv][6])
* **HF Evaluate library** (classic metrics, pipelines). ([Hugging Face][9])

---

If you want, tell me your exact use‑case (e.g., “agent for sales emails” or “RAG for policy answers”), and I’ll turn this into a tailored eval suite (tasks, rubric, judge prompt, and the exact LightEval command file) you can run as‑is.

[1]: https://huggingface.co/blog/clefourrier/llm-evaluation?utm_source=chatgpt.com "Let's talk about LLM evaluation - Hugging Face"
[2]: https://huggingface.co/docs/leaderboards/en/index?utm_source=chatgpt.com "Leaderboards and Evaluations - Hugging Face"
[3]: https://huggingface.co/blog/zero-shot-eval-on-the-hub?utm_source=chatgpt.com "Very Large Language Models and How to Evaluate Them"
[4]: https://huggingface.co/posts/clefourrier/174172922896344?utm_source=chatgpt.com "@clefourrier on Hugging Face: \"Fun fact about evaluation! Did you ..."
[5]: https://huggingface.co/learn/cookbook/en/llm_judge "Using LLM-as-a-judge ‍⚖️ for an automated and versatile evaluation - Hugging Face Open-Source AI Cookbook"
[6]: https://arxiv.org/abs/2306.05685?utm_source=chatgpt.com "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"
[7]: https://huggingface.co/docs/lighteval/index "Lighteval"
[8]: https://huggingface.co/docs/lighteval/quicktour "Quicktour"
[9]: https://huggingface.co/docs/evaluate/en/index " Evaluate"
[10]: https://en.wikipedia.org/?redirect=no&title=Wilson_score_interval&utm_source=chatgpt.com "Wilson score interval - Wikipedia"
[11]: https://www.qualtrics.com/experience-management/research/margin-of-error/?utm_source=chatgpt.com "Margin of Error Guide & Calculator - Qualtrics"
[12]: https://huggingface.co/docs/lighteval/saving-and-reading-results "Saving and reading results"
[13]: https://huggingface.co/spaces/open-llm-leaderboard/blog?utm_source=chatgpt.com "Performances are plateauing, let's make the leaderboard steep again"
[14]: https://github.com/huggingface/evaluation-guidebook "GitHub - huggingface/evaluation-guidebook: Sharing both practical insights and theoretical knowledge about LLM evaluation that we gathered while managing the Open LLM Leaderboard and designing lighteval!"
