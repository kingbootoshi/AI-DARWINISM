#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GEPA-in-DSPy: A small, practical reproduction of the GEPA optimizer loop.

What you get:
- Algorithm 1: reflective prompt evolution with minibatches
- Algorithm 2: Pareto-based candidate selection
- Algorithm 4: (optional) Merge crossover for modular candidates
- Prompt extraction from the GEPA PDF (Appendix I) so you can use the paper's exact prompts
- A tiny Hotpot-like demo pipeline in DSPy to play with prompt evolution end-to-end

Setup
-----
pip install -U dspy PyMuPDF numpy
# Optional for real datasets:
# pip install datasets

Environment
-----------
# Configure via .env file or environment variables:
#   OPENROUTER_API_KEY=your_key_here
#   DSPY_MODEL=openai/gpt-4o-mini (or any LiteLLM-compatible model)
#   See config/settings.py for full configuration options

Usage
-----
1) Download the paper PDF locally (from arXiv).
2) Run a quick dry run with the paper prompts:
   python main.py --pdf /path/to/2507.19457.pdf --use-paper-prompts

3) Try GEPA optimization (toy dataset):
   python main.py --pdf /path/to/2507.19457.pdf \
       --optimize --budget 60 --minibatch 6 --pareto 8 --merge-prob 0.2

Notes
-----
- This is a faithful *shape* of GEPA in DSPy + Python. Exact numbers in the paper
  depend on their datasets, budgets, and backend models, so treat this as a
  reproduction scaffolding rather than a guaranteed bit-for-bit replication.
- The HotpotQA pipeline here is minimal (summaries + answer). Replace the toy
  dataset with your task-specific loader and metric for serious runs.

References
----------
GEPA paper (algorithms & prompts): Agrawal et al., 2025 (arXiv:2507.19457).  # cites in the chat message
"""

from __future__ import annotations
import argparse, dataclasses, json, math, os, random, re, sys, time, threading, ast
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from loguru import logger

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

import dspy

# Import the DSPy app configuration modules
from config.settings import get_settings
from config.logging import init_logging
from dspy_app.lm import configure_dspy
from webui.state import state as ui_state

# -------------------------------
# Utilities
# -------------------------------

def set_seed(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)

def normalize_text(s: str) -> str:
    return " ".join(s.strip().lower().split())

def exact_match(pred: str, gold: str) -> float:
    return float(normalize_text(pred) == normalize_text(gold))

def safe_get(d: dict, *keys, default=None):
    out = d
    for k in keys:
        if not isinstance(out, dict) or k not in out:
            return default
        out = out[k]
    return out

# -------------------------------
# Dynamic DSPy Signature builder
# -------------------------------

def build_signature(name: str, instruction: str, inputs: Dict[str, str], outputs: Dict[str, str]):
    """
    Create a DSPy Signature subclass dynamically with the given instruction/docstring,
    input and output fields. Example:
        Sig = build_signature("Summarize1", "Your task is ...", {"question": "str", "passages": "list[str]"}, {"summary": "str"})
    """
    ns = {"__doc__": instruction}
    # dspy.InputField and OutputField accept optional desc=...
    for in_name, desc in inputs.items():
        ns[in_name] = dspy.InputField(desc=desc)
    for out_name, desc in outputs.items():
        ns[out_name] = dspy.OutputField(desc=desc)
    Sig = type(name, (dspy.Signature,), ns)
    return Sig

@dataclass
class ModuleSpec:
    name: str
    inputs: Dict[str, str]
    outputs: Dict[str, str]
    instruction: str

    def make_module(self):
        Sig = build_signature(self.name + "Sig", self.instruction, self.inputs, self.outputs)
        return dspy.Predict(Sig)

# -------------------------------
# A tiny Hotpot-like system to play with
# -------------------------------

@dataclass
class HotpotItem:
    question: str
    # For the toy loop, we'll pretend we "retrieved" these.
    passages1: List[str]
    passages2: List[str]
    answer: str

@dataclass
class SystemIO:
    # traces across modules
    inter: Dict[str, Any]
    # final output
    answer: str

class HotpotSystem:
    """
    Minimal, modular DSPy pipeline:
      summarize1 -> create_query_hop2 -> summarize2 -> final_answer
    You can swap the ModuleSpec.instructions at any time (that's what GEPA updates).
    """
    def __init__(self, lm: dspy.LM, module_specs: Dict[str, ModuleSpec]):
        self.lm = lm
        self.specs = module_specs
        # Rebuild modules from current specs.
        self.modules = {k: v.make_module() for k, v in self.specs.items()}

    def run(self, item: HotpotItem, capture_traces: bool = True) -> SystemIO:
        inter = {}
        # Summarize1
        m = self.modules["summarize1"]
        s1 = m(question=item.question, passages=item.passages1).summary
        inter["summary1"] = s1

        # Create query for hop2  (in a real system you'd retrieve with this; here we just keep toy passages2)
        m = self.modules["create_query_hop2"]
        q2 = m(question=item.question, summary_1=s1).query
        inter["query2"] = q2

        # Summarize2
        m = self.modules["summarize2"]
        s2 = m(question=item.question, context=s1, passages=item.passages2).summary
        inter["summary2"] = s2

        # Final answer
        m = self.modules["final_answer"]
        ans = m(question=item.question, summary_1=s1, summary_2=s2).answer

        return SystemIO(inter=inter, answer=ans)

# -------------------------------
# Horror mode (two-sentence horror microfiction)
# -------------------------------

@dataclass
class HorrorItem:
    """A tiny task item for two-sentence horror generation.

    Attributes
    ----------
    prompt: str
        Seed premise or constraint to guide the microstory.
    """

    prompt: str


class HorrorSystem:
    """Single-module system that writes a two-sentence horror microstory.

    The module is named 'horror_writer' and returns a 'story' field. We reuse
    the SystemIO.answer slot to carry the story so the metric hook remains uniform.
    """

    def __init__(self, lm: dspy.LM, module_specs: Dict[str, ModuleSpec]):
        self.lm = lm
        self.specs = module_specs
        self.modules = {k: v.make_module() for k, v in self.specs.items()}

    def run(self, item: HorrorItem, capture_traces: bool = True) -> SystemIO:
        story = self.modules["horror_writer"](prompt=item.prompt).story
        return SystemIO(inter={"story": story}, answer=story)


BASE_SEED_PROMPTS_HORROR = {
    "horror_writer": (
        "Write a TWO-SENTENCE horror microstory. Requirements:\n"
        "- Exactly two sentences; no ellipses to cheat length.\n"
        "- Vivid concrete imagery; grounded in one scene.\n"
        "- A subtle twist or reveal in sentence two.\n"
        "- No cliches (no 'it was all a dream', 'the call is from inside the house').\n"
        "- Keep it under 70 words total."
    )
}


def make_specs_from_prompts_horror(prompts: Dict[str, str]) -> Dict[str, ModuleSpec]:
    """Create the single horror module spec from instruction prompts."""
    return {
        "horror_writer": ModuleSpec(
            name="horror_writer",
            inputs={"prompt": "seed/premise or constraint"},
            outputs={"story": "two sentences of horror"},
            instruction=prompts["horror_writer"],
        )
    }


def system_factory_from_prompts_horror(lm: dspy.LM, prompts: Dict[str, str]) -> Callable[[Dict[str, str]], HorrorSystem]:
    """Return a factory that instantiates HorrorSystem from updated prompts."""
    def factory(updated_prompts: Dict[str, str]) -> HorrorSystem:
        specs = make_specs_from_prompts_horror(updated_prompts)
        return HorrorSystem(lm=lm, module_specs=specs)

    return factory


def horror_seed_data() -> List[HorrorItem]:
    """Small seed set for quick evolution."""
    seeds = [
        "A baby monitor that starts narrating future sounds.",
        "A calendar where tomorrow’s date is torn out by someone else.",
        "A mirror that returns your smile a beat too late.",
        "A house whose lights flicker in Morse code spelling a name.",
        "Footsteps that stop when you stop—but never resume.",
    ]
    return [HorrorItem(prompt=s) for s in seeds]


def build_horror_judge() -> dspy.Module:
    """Construct the LLM judge module that scores two-sentence horror stories.

    Returns
    -------
    dspy.Module
        A Predict module that takes 'story' and returns JSON in 'json'.
    """
    rubric = (
        "You are a strict microfiction judge. Score a TWO-SENTENCE horror story.\n"
        "Return JSON with integer fields 0–5:\n"
        "{ 'scariness':int, 'suspense':int, 'originality':int, 'clarity':int, 'rule_two_sentences':int, 'explanation': str }\n"
        "Rules:\n"
        "- 'rule_two_sentences'=5 only if exactly 2 sentences (periods ending sentences; abbreviations allowed rarely).\n"
        "- Penalize clichés and incoherence. Stories >70 words lose clarity points.\n"
        "- Consider imagery, escalation, and twist quality.\n"
        "Be concise in explanation (<= 20 words)."
    )
    Sig = build_signature(
        "JudgeHorrorSig",
        rubric,
        inputs={"story": "the two-sentence story"},
        outputs={"json": "judge JSON"},
    )
    return dspy.Predict(Sig)


_horror_judge = None


def _horror_metric_from_human_or_llm(pred_story: str) -> float:
    """Compute a 0..1 score using human ratings when available, else LLM judge.

    Priority: if the FastAPI endpoint has received a rating for this exact story,
    use the human-provided component scores. Otherwise, run the LLM judge.
    """
    # Check for human rating events matching this story
    # Note: simple scan of bounded event buffer keeps complexity low and acceptable
    ratings = None
    for ev in reversed(ui_state.events):  # type: ignore[attr-defined]
        if ev.type == "human_rating" and ev.data.get("story", "") == pred_story:
            ratings = ev.data
            break

    if ratings is not None:
        s = (
            0.35 * int(ratings.get("scariness", 0))
            + 0.25 * int(ratings.get("suspense", 0))
            + 0.20 * int(ratings.get("originality", 0))
            + 0.10 * int(ratings.get("clarity", 0))
            + 0.10 * int(ratings.get("rule_two_sentences", 0))
        )
        return float(s) / 5.0

    # Lazy-init judge and score
    global _horror_judge
    if _horror_judge is None:
        _horror_judge = build_horror_judge()
    out = _horror_judge(story=pred_story)
    # Some models may return single-quoted JSON; parse robustly
    raw = out.json if isinstance(out.json, str) else str(out.json)
    try:
        data = json.loads(raw)
        s = (
            0.35 * data.get("scariness", 0)
            + 0.25 * data.get("suspense", 0)
            + 0.20 * data.get("originality", 0)
            + 0.10 * data.get("clarity", 0)
            + 0.10 * data.get("rule_two_sentences", 0)
        )
        return float(s) / 5.0
    except Exception:
        try:
            data = ast.literal_eval(raw)
            s = (
                0.35 * int(data.get("scariness", 0))
                + 0.25 * int(data.get("suspense", 0))
                + 0.20 * int(data.get("originality", 0))
                + 0.10 * int(data.get("clarity", 0))
                + 0.10 * int(data.get("rule_two_sentences", 0))
            )
            return float(s) / 5.0
        except Exception:
            return 0.0

# -------------------------------
# Paper prompt extraction
# -------------------------------

GEPA_SECTION_HEADER = r"I\.\s*1\s+HOTPOTQA,\s+GPT-4\.1\s+MINI"
MODULE_HEADER_PATTERNS = {
    "create_query_hop2": r"HotpotQA GPT-4\.1 Mini create_query_hop2\.predict",
    "final_answer":      r"HotpotQA GPT-4\.1 Mini final_answer\.predict",
    "summarize1":        r"HotpotQA GPT-4\.1 Mini summarize1\.predict",
    "summarize2":        r"HotpotQA GPT-4\.1 Mini summarize2\.predict",
}
PROMPT_BOX_ANCHOR = r"GEPA Prompt generated by config GEPA:"

def extract_prompts_from_pdf(pdf_path: str) -> Dict[str, str]:
    """
    Reads the GEPA PDF (local file) and extracts the *GEPA* prompts for the HotpotQA / GPT-4.1 Mini
    system modules from Appendix I.

    It looks for:
      "HotpotQA GPT-4.1 Mini <module>.predict"
      then the box titled "GEPA Prompt generated by config GEPA:"
      ... collects text until the next module header.

    Returns dict: {module_name: instruction_string}
    """
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) not installed. `pip install PyMuPDF`")

    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text("text") for page in doc)
    doc.close()

    # Narrow to the HotpotQA / GPT-4.1 Mini subsection.
    m = re.search(GEPA_SECTION_HEADER, text, flags=re.IGNORECASE)
    if not m:
        raise RuntimeError("Couldn't locate 'I.1 HOTPOTQA, GPT-4.1 MINI' in the PDF. Check the file.")
    start = m.start()
    # stop near the next section "I.2 HOTPOTQA, Qwen3 8B"
    m2 = re.search(r"I\.\s*2\s+HOTPOTQA,\s+Qwen3", text[start:], flags=re.IGNORECASE)
    end = start + (m2.start() if m2 else len(text))
    sub = text[start:end]

    def grab_for(module_key: str) -> str:
        head_pat = MODULE_HEADER_PATTERNS[module_key]
        mH = re.search(head_pat, sub, flags=re.IGNORECASE)
        if not mH:
            raise RuntimeError(f"Couldn't find header for {module_key}")
        after = sub[mH.end():]
        mA = re.search(PROMPT_BOX_ANCHOR, after, flags=re.IGNORECASE)
        if not mA:
            raise RuntimeError(f"Couldn't find GEPA prompt anchor for {module_key}")
        after2 = after[mA.end():]
        # Stop at the next module header or the end of subsection
        stops = [re.search(pat, after2, flags=re.IGNORECASE) for pat in MODULE_HEADER_PATTERNS.values()]
        stops = [s for s in stops if s]
        stop_idx = min(s.start() for s in stops) if stops else len(after2)
        prompt = after2[:stop_idx].strip()
        # Heuristic cleanups (PDF may insert hyphen newlines etc.)
        prompt = re.sub(r"\n{2,}", "\n", prompt)
        return prompt.strip()

    prompts = {k: grab_for(k) for k in MODULE_HEADER_PATTERNS.keys()}
    return prompts

# -------------------------------
# Reflective mutation (UPDATEPROMPT)
# -------------------------------

def build_reflector_module(lm: dspy.LM):
    """
    A small 'LLM-as-optimizer' module: given current instruction + a handful of
    failed/passing traces, propose a new instruction that keeps IO schema intact.
    """
    instruction = (
        "You are improving the INSTRUCTION for one module in a modular LLM system. "
        "Goal: propose a strictly better instruction that fixes observed errors without changing the IO schema. "
        "Keep it crisp, declarative, and focused on actionable rules (no few-shot demos).\n\n"
        "Return JSON with fields:\n"
        "{\n"
        '  "rationale": "<brief>",\n'
        '  "new_instruction": "<full instruction replacing the prior one>"\n'
        "}\n"
        "Constraints:\n"
        "- Do not invent new input fields or output fields.\n"
        "- Prefer explicit rules (bullets, steps) over vague advice.\n"
        "- Include task-specific clarifications that directly target recurrent mistakes in the traces."
    )
    Sig = build_signature(
        "ReflectAndProposeSig",
        instruction,
        inputs={
            "module_name": "name of the module",
            "io_schema": "string describing input and output fields",
            "current_instruction": "the instruction text being replaced",
            "batch_report": "compact report of successes/failures with minimal excerpts",
        },
        outputs={"json": "a JSON object with rationale and new_instruction"},
    )
    return dspy.Predict(Sig)

def make_batch_report(module_name: str, batch_traces: List[Dict[str, Any]], max_items: int = 4) -> str:
    """
    Compress traces + feedback into a small human-readable report.
    """
    lines = [f"Module: {module_name}", "Cases:"]
    for t in batch_traces[:max_items]:
        score = t.get("score", None)
        lines.append(f"- score={score} :: issue={t.get('issue','?')}")
        inputs = t.get("inputs", {})
        outs = t.get("outputs", {})
        # Keep short
        for k, v in list(inputs.items())[:3]:
            vv = v if isinstance(v, str) else json.dumps(v)[:220]
            lines.append(f"  in.{k}: {vv[:220]}")
        for k, v in list(outs.items())[:2]:
            vv = v if isinstance(v, str) else json.dumps(v)[:220]
            lines.append(f"  out.{k}: {vv[:220]}")
    return "\n".join(lines)

def update_instruction_via_reflection(
    lm: dspy.LM,
    module_name: str,
    io_schema: str,
    current_instruction: str,
    batch_traces: List[Dict[str, Any]],
) -> str:
    reflector = build_reflector_module(lm)
    report = make_batch_report(module_name, batch_traces)
    out = reflector(
        module_name=module_name,
        io_schema=io_schema,
        current_instruction=current_instruction,
        batch_report=report,
    )
    try:
        data = json.loads(out.json)
        return data.get("new_instruction", current_instruction).strip()
    except Exception:
        # If parsing fails, fall back to the original
        return current_instruction

# -------------------------------
# GEPA core (Algorithms 1 & 2 + Merge)
# -------------------------------

@dataclass
class Candidate:
    prompts: Dict[str, str]  # module_name -> instruction

class GEPA:
    def __init__(
        self,
        system_factory: Callable[[Dict[str, str]], HotpotSystem],
        train_items: List[HotpotItem],
        eval_metric: Callable[[str, str], float],
        minibatch_size: int = 8,
        pareto_set_size: int = 16,
        seed: int = 2025,
        merge_prob: float = 0.0,
    ):
        set_seed(seed)
        self.system_factory = system_factory
        self.train_items = train_items
        self.metric = eval_metric
        self.minibatch_size = minibatch_size
        self.pareto_set_size = pareto_set_size
        self.merge_prob = merge_prob

        # Split D_train into D_feedback and D_pareto (Algorithm 1, line 1)
        idx = list(range(len(train_items)))
        random.shuffle(idx)
        self.D_pareto_idx = idx[:min(self.pareto_set_size, len(idx))]
        self.D_feedback_idx = idx[min(self.pareto_set_size, len(idx)):] or idx  # fall back

        self.candidates: List[Candidate] = []
        self.parents: List[Optional[int]] = []  # parent index
        self.S: List[List[float]] = []          # candidate-by-instance scores on D_pareto

        # bookkeeping
        self._pareto_cache: Dict[int, float] = {}

    # --- helpers ---

    def _evaluate_on(self, cand: Candidate, indices: Sequence[int]) -> Tuple[List[float], List[Dict[str, Any]]]:
        """Run the system for selected items and compute scores + traces.

        This is task-agnostic: it calls `system.run(item)` and scores via
        `self.metric(io.answer, getattr(item, 'answer', ''))`. Traces include
        dataclass inputs (when available) and module outputs.
        """
        sys_inst = self.system_factory(cand.prompts)
        scores, traces = [], []
        for i in indices:
            item = self.train_items[i]
            io = sys_inst.run(item, capture_traces=True)
            gold = getattr(item, "answer", "")
            scr = self.metric(io.answer, gold)
            scores.append(scr)
            # Compact inputs: handle dataclasses or fallback to repr
            if dataclasses.is_dataclass(item):
                try:
                    inputs_view = dataclasses.asdict(item)
                except Exception:
                    inputs_view = {"item": repr(item)}
            else:
                inputs_view = {"item": repr(item)}
            # Compact outputs: include inter and answer
            outs = {"answer": io.answer}
            for k, v in (io.inter or {}).items():
                outs[k] = v
            issue = "ok" if scr >= 1.0 else "needs-improvement"
            traces.append({
                "score": scr,
                "inputs": inputs_view,
                "outputs": outs,
                "issue": issue,
            })
            # Live stream: record every evaluation to outputs_stream (phase decided by caller)
        return scores, traces

    def _initialize(self, seed_prompts: Dict[str, str]):
        base = Candidate(prompts=dict(seed_prompts))
        self.candidates = [base]
        self.parents = [None]
        # Initialize S on D_pareto (Algorithm 1, lines 3-5)
        scores, traces = self._evaluate_on(base, self.D_pareto_idx)
        self.S = [scores]
        # UI: log pareto evals for baseline candidate 0
        for local_idx, global_idx in enumerate(self.D_pareto_idx):
            t = traces[local_idx]
            ui_state.add_output_stream(
                phase="pareto",
                cand_id=0,
                item_index=int(global_idx),
                inputs=t["inputs"],
                outputs=t["outputs"],
                score=float(scores[local_idx]),
                meta={"candidate": 0},
            )

    def _select_candidate_index(self) -> int:
        """
        Algorithm 2: Pareto-based candidate selection.
        Returns the index k of the candidate to evolve next.
        """
        # Build instance-wise maxima
        S_arr = np.array(self.S)  # shape: (K, P)
        # For each instance, find max score across candidates
        s_star = S_arr.max(axis=0)  # (P,)
        # Candidates appearing on instance-wise Pareto sets:
        C = set()
        for j, best in enumerate(s_star):
            winners = np.where(S_arr[:, j] == best)[0].tolist()
            C.update(winners)

        # Remove dominated candidates in C
        C = list(C)
        non_dom = []
        for i in C:
            dominated = False
            for j in C:
                if i == j:
                    continue
                # j dominates i if S_j >= S_i for all instances and strictly better in at least one
                if np.all(S_arr[j] >= S_arr[i]) and np.any(S_arr[j] > S_arr[i]):
                    dominated = True
                    break
            if not dominated:
                non_dom.append(i)

        # Frequency-weighted sampling by how often they are in instance-wise Pareto sets
        freqs = []
        for i in non_dom:
            f = int((S_arr[i] == s_star).sum())
            freqs.append(max(1, f))
        weights = np.array(freqs, dtype=float)
        weights = weights / weights.sum()
        return int(np.random.choice(non_dom, p=weights))

    def _select_module_name(self) -> str:
        return random.choice(list(self.candidates[0].prompts.keys()))

    def _merge(self) -> Optional[Candidate]:
        """System-aware merge (Algorithm 4, simplified).
        We skip ancestry bookkeeping here for brevity; instead we merge by taking, per module,
        the instruction of the better-scoring of two random candidates on the average Pareto score.
        """
        if len(self.candidates) < 2:
            return None
        i, j = random.sample(range(len(self.candidates)), 2)
        # Choose the better overall
        mean_i = float(np.mean(self.S[i]))
        mean_j = float(np.mean(self.S[j]))
        chosen = i if mean_i >= mean_j else j

        parent = self.candidates[chosen]
        other = self.candidates[j if chosen == i else i]
        merged = {}
        for m in parent.prompts.keys():
            merged[m] = parent.prompts[m] if random.random() < 0.5 else other.prompts[m]
        return Candidate(prompts=merged)

    # --- main loop ---

    def optimize(self, seed_prompts: Dict[str, str], rollout_budget: int = 60):
        """
        rollout_budget counts 'minibatch evaluations' (i.e., we decrement by 1 per mutation attempt).
        """
        self._initialize(seed_prompts)
        # UI: register base candidate and initial Pareto scores
        base_mean = float(np.mean(self.S[0])) if self.S and self.S[0] else None
        ui_state.add_candidate(
            cand_id=0,
            parent=None,
            prompts=self.candidates[0].prompts,
            mean_score=base_mean,
            scores_row=self.S[0] if self.S else None,
        )
        if base_mean is not None:
            ui_state.set_best(0)
        b = max(1, self.minibatch_size)

        while rollout_budget > 0:
            rollout_budget -= 1

            # maybe propose a merge
            do_merge = (self.merge_prob > 0 and random.random() < self.merge_prob)
            if do_merge:
                merged = self._merge()
                if merged is not None:
                    # Evaluate merged on minibatch
                    batch = random.sample(self.D_feedback_idx, k=min(b, len(self.D_feedback_idx)))
                    old_sc, old_tr = self._evaluate_on(self.candidates[0], batch)  # baseline for comparison
                    new_sc, new_tr = self._evaluate_on(merged, batch)
                    # UI: stream minibatch comparisons
                    for j, idx_global in enumerate(batch):
                        ui_state.add_output_stream(
                            phase="minibatch_parent",
                            cand_id=0,
                            item_index=int(idx_global),
                            inputs=old_tr[j]["inputs"],
                            outputs=old_tr[j]["outputs"],
                            score=float(old_sc[j]),
                            meta={"merge": True},
                        )
                        ui_state.add_output_stream(
                            phase="minibatch_child",
                            cand_id=-1,  # temporary id for merge-proposed child
                            item_index=int(idx_global),
                            inputs=new_tr[j]["inputs"],
                            outputs=new_tr[j]["outputs"],
                            score=float(new_sc[j]),
                            meta={"merge": True},
                        )
                    if np.mean(new_sc) > np.mean(old_sc):
                        # promote; evaluate on D_pareto
                        self.candidates.append(merged)
                        self.parents.append(None)
                        S_new, tr_new = self._evaluate_on(merged, self.D_pareto_idx)
                        self.S.append(S_new)
                        cand_id = len(self.candidates) - 1
                        mean_new = float(np.mean(S_new)) if S_new else None
                        ui_state.add_candidate(
                            cand_id=cand_id,
                            parent=None,
                            prompts=merged.prompts,
                            mean_score=mean_new,
                            scores_row=S_new,
                        )
                        ui_state.log_event("merge_accept", {"child": cand_id, "mean": mean_new})
                        # Stream pareto evaluations for the new candidate
                        for local_idx, global_idx in enumerate(self.D_pareto_idx):
                            t = tr_new[local_idx]
                            ui_state.add_output_stream(
                                phase="pareto",
                                cand_id=cand_id,
                                item_index=int(global_idx),
                                inputs=t["inputs"],
                                outputs=t["outputs"],
                                score=float(S_new[local_idx]),
                                meta={"candidate": cand_id},
                            )
                        # Update best
                        means = [float(np.mean(s)) for s in self.S]
                        best_idx = int(np.argmax(means))
                        ui_state.set_best(best_idx)
                    continue

            # Algorithm 1: select candidate & module
            k = self._select_candidate_index()
            module_name = self._select_module_name()

            parent = self.candidates[k]
            batch = random.sample(self.D_feedback_idx, k=min(b, len(self.D_feedback_idx)))

            # Evaluate parent on minibatch; collect traces
            parent_scores, parent_traces = self._evaluate_on(parent, batch)
            # Build traces that emphasize the module we're mutating
            batch_traces_for_module = []
            for pt in parent_traces:
                bt = {
                    "score": pt["score"],
                    "issue": pt["issue"],
                    "inputs": pt["inputs"],
                    "outputs": {"answer": pt["outputs"]["answer"]},
                }
                # add module outputs that are relevant
                for k2 in ["summary1", "summary2", "query2"]:
                    if k2 in pt["outputs"]:
                        bt["outputs"][k2] = pt["outputs"][k2]
                batch_traces_for_module.append(bt)
            # Stream parent minibatch outputs
            for j, idx_global in enumerate(batch):
                ui_state.add_output_stream(
                    phase="minibatch_parent",
                    cand_id=k,
                    item_index=int(idx_global),
                    inputs=parent_traces[j]["inputs"],
                    outputs=parent_traces[j]["outputs"],
                    score=float(parent_scores[j]),
                    meta={"module": module_name},
                )

            # propose new instruction
            cur_instruction = parent.prompts[module_name]
            io_schema = f"Module {module_name}: keep IO schema unchanged and improve instruction."
            new_instruction = update_instruction_via_reflection(
                dspy.settings.lm, module_name, io_schema, cur_instruction, batch_traces_for_module
            )
            if new_instruction.strip() == cur_instruction.strip():
                continue  # no change

            child_prompts = dict(parent.prompts)
            child_prompts[module_name] = new_instruction
            child = Candidate(prompts=child_prompts)

            # Evaluate child vs parent on same minibatch
            child_scores, child_traces = self._evaluate_on(child, batch)
            # Stream child minibatch outputs
            for j, idx_global in enumerate(batch):
                ui_state.add_output_stream(
                    phase="minibatch_child",
                    cand_id=-1,  # not accepted yet
                    item_index=int(idx_global),
                    inputs=child_traces[j]["inputs"],
                    outputs=child_traces[j]["outputs"],
                    score=float(child_scores[j]),
                    meta={"module": module_name},
                )

            parent_mean = float(np.mean(parent_scores)) if parent_scores else 0.0
            child_mean = float(np.mean(child_scores)) if child_scores else 0.0

            # UI: record mutation attempt
            ui_state.set_last_mutation({
                "module": module_name,
                "parent_mean": parent_mean,
                "child_mean": child_mean,
                "accepted": child_mean > parent_mean,
                "old_instruction": cur_instruction,
                "new_instruction": new_instruction,
            })
            ui_state.log_event("mutation_attempt", {"module": module_name, "parent_mean": parent_mean, "child_mean": child_mean})

            if child_mean > parent_mean:
                # accept
                self.candidates.append(child)
                self.parents.append(k)
                # Evaluate on D_pareto and save
                S_child, tr_child = self._evaluate_on(child, self.D_pareto_idx)
                self.S.append(S_child)
                cand_id = len(self.candidates) - 1
                mean_child = float(np.mean(S_child)) if S_child else None
                ui_state.add_candidate(
                    cand_id=cand_id,
                    parent=k,
                    prompts=child.prompts,
                    mean_score=mean_child,
                    scores_row=S_child,
                )
                ui_state.log_event("mutation_accept", {"child": cand_id, "module": module_name, "mean": mean_child})
                # Stream pareto evals for accepted child
                for local_idx, global_idx in enumerate(self.D_pareto_idx):
                    t = tr_child[local_idx]
                    ui_state.add_output_stream(
                        phase="pareto",
                        cand_id=cand_id,
                        item_index=int(global_idx),
                        inputs=t["inputs"],
                        outputs=t["outputs"],
                        score=float(S_child[local_idx]),
                        meta={"candidate": cand_id},
                    )
                means = [float(np.mean(s)) for s in self.S]
                best_idx = int(np.argmax(means))
                ui_state.set_best(best_idx)

        # Return best by mean Pareto score
        means = [float(np.mean(s)) for s in self.S]
        best_idx = int(np.argmax(means))
        return self.candidates[best_idx], means[best_idx]

# -------------------------------
# Seed prompts
# -------------------------------

BASE_SEED_PROMPTS = {
    # intentionally tiny base prompts; you’ll usually start here and let GEPA evolve
    "summarize1": (
        "Given fields 'question' and 'passages', produce field 'summary'. "
        "Summarize only facts relevant to the question."
    ),
    "create_query_hop2": (
        "Given 'question' and 'summary_1', produce field 'query' for second-hop retrieval. "
        "Target missing info not covered in the first hop."
    ),
    "summarize2": (
        "Given 'question', 'context' and 'passages', produce field 'summary'. "
        "Integrate context with new passages to support answering the question."
    ),
    "final_answer": (
        "Given 'question', 'summary_1', 'summary_2', produce field 'answer'. "
        "Answer concisely and exactly."
    ),
}

def make_specs_from_prompts(prompts: Dict[str, str]) -> Dict[str, ModuleSpec]:
    return {
        "summarize1": ModuleSpec(
            name="summarize1",
            inputs={"question": "question", "passages": "list of short passages"},
            outputs={"summary": "one short paragraph"},
            instruction=prompts["summarize1"],
        ),
        "create_query_hop2": ModuleSpec(
            name="create_query_hop2",
            inputs={"question": "original question", "summary_1": "summary from hop1"},
            outputs={"query": "a concise second-hop query"},
            instruction=prompts["create_query_hop2"],
        ),
        "summarize2": ModuleSpec(
            name="summarize2",
            inputs={"question": "question", "context": "summary_1", "passages": "list of passages for hop2"},
            outputs={"summary": "structured summary"},
            instruction=prompts["summarize2"],
        ),
        "final_answer": ModuleSpec(
            name="final_answer",
            inputs={"question": "question", "summary_1": "summary from hop1", "summary_2": "summary from hop2"},
            outputs={"answer": "final answer"},
            instruction=prompts["final_answer"],
        ),
    }

def system_factory_from_prompts(lm: dspy.LM, prompts: Dict[str, str]) -> Callable[[Dict[str, str]], HotpotSystem]:
    def factory(updated_prompts: Dict[str, str]) -> HotpotSystem:
        specs = make_specs_from_prompts(updated_prompts)
        return HotpotSystem(lm=lm, module_specs=specs)
    return factory

# -------------------------------
# Tiny demo dataset
# -------------------------------

def toy_hotpot_data() -> List[HotpotItem]:
    # Extremely small, just to exercise the loop.
    return [
        HotpotItem(
            question="Are Macharaenthera and Prumnopitys both plants?",
            passages1=[
                "Macharaenthera is a genus of flowering plants in the daisy family.",
                "Prumnopitys is a genus of coniferous trees in the podocarp family.",
            ],
            passages2=[
                "Both are plant genera: Macharaenthera (daisy family) and Prumnopitys (podocarp family).",
            ],
            answer="Yes",
        ),
        HotpotItem(
            question="Which bank was founded by the great-great-great-grandfather of the second Duke of Florence?",
            passages1=[
                "Giovanni di Bicci de' Medici founded the Medici Bank.",
            ],
            passages2=[
                "Cosimo I de' Medici became Grand Duke of Tuscany; Giovanni di Bicci de' Medici was his ancestor.",
            ],
            answer="The Medici Bank",
        ),
        HotpotItem(
            question="In which Serie does Simone Benedetti's team currently compete?",
            passages1=["Virtus Entella competes in Serie B."],
            passages2=["Simone Benedetti plays for Virtus Entella."],
            answer="Serie B",
        ),
    ]

# -------------------------------
# CLI
# -------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=str, default=None, help="Path to arXiv PDF (2507.19457).")
    parser.add_argument("--use-paper-prompts", action="store_true",
                        help="Load exact GEPA prompts for HotpotQA/GPT-4.1 Mini from the PDF.")
    parser.add_argument("--optimize", action="store_true",
                        help="Run the GEPA optimization loop.")
    parser.add_argument("--budget", type=int, default=60, help="GEPA rollout budget (mutation attempts).")
    parser.add_argument("--minibatch", type=int, default=6, help="Minibatch size for mutation tests.")
    parser.add_argument("--pareto", type=int, default=8, help="Pareto set size.")
    parser.add_argument("--merge-prob", type=float, default=0.0, help="Probability of trying a Merge step per iteration.")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--task", choices=["hotpot", "horror"], default="hotpot", help="Choose task to run.")
    parser.add_argument("--web", action="store_true", help="Launch a local web UI on http://127.0.0.1:3000")
    args = parser.parse_args()

    # Initialize logging and DSPy configuration using the centralized setup
    settings = get_settings()
    init_logging(
        app_name="GEPA-in-DSPy",
        level=settings.log_level,
        log_dir=settings.log_dir,
        colorize_console=settings.colorize_console,
        file_rotation=settings.file_rotation,
        file_retention=settings.file_retention,
    )
    
    logger.info("Starting GEPA-in-DSPy with model: {}", settings.model)
    
    set_seed(args.seed)
    lm = configure_dspy(settings)

    # Start web UI if requested
    if args.web:
        try:
            from webui.server import run as run_server
            t = threading.Thread(target=run_server, kwargs={"host": "127.0.0.1", "port": 3000}, daemon=True)
            t.start()
            logger.info("Web UI running at http://127.0.0.1:3000")
        except Exception as exc:
            logger.warning("Failed to start web UI: {}", exc)

    # Task-specific setup
    if args.task == "horror":
        if args.use_paper_prompts:
            logger.warning("--use-paper-prompts is only for Hotpot. Using horror seed prompts instead.")
        prompts = dict(BASE_SEED_PROMPTS_HORROR)
        system_factory = system_factory_from_prompts_horror(lm, prompts)
        data = horror_seed_data()
        metric = lambda pred, gold: _horror_metric_from_human_or_llm(pred)
    else:
        # Hotpot
        if args.use_paper_prompts:
            if not args.pdf:
                print("ERROR: --use-paper-prompts requires --pdf path to the GEPA paper.", file=sys.stderr)
                sys.exit(1)
            print("Extracting GEPA prompts from PDF ...")
            paper_prompts = extract_prompts_from_pdf(args.pdf)
            prompts = paper_prompts
            print("Loaded modules:", ", ".join(sorted(prompts.keys())))
        else:
            prompts = dict(BASE_SEED_PROMPTS)
        system_factory = system_factory_from_prompts(lm, prompts)
        data = toy_hotpot_data()
        metric = exact_match

    # Initialize UI state header
    ui_state.reset(
        task=args.task,
        model=settings.model,
        pareto_size=args.pareto,
        minibatch_size=args.minibatch,
        budget=args.budget,
        merge_prob=args.merge_prob,
    )

    # Quick eval (no optimization)
    sys_inst = system_factory(prompts)
    print("\n=== Baseline evaluation ===")
    s_list = []
    if args.task == "horror":
        for idx, it in enumerate(data):
            out = sys_inst.run(it)
            score = metric(out.answer, "")
            s_list.append(score)
            ui_state.add_baseline_output(
                item_index=idx,
                inputs={"prompt": it.prompt},
                outputs={"story": out.answer},
                score=float(score),
            )
            print(f"Prompt: {it.prompt}\nStory: {out.answer}\nScore: {score:.3f}\n")
        print(f"Mean score: {np.mean(s_list):.3f}")
    else:
        for idx, it in enumerate(data):
            out = sys_inst.run(it)
            score = metric(out.answer, it.answer)
            s_list.append(score)
            ui_state.add_baseline_output(
                item_index=idx,
                inputs={"question": it.question},
                outputs={"answer": out.answer, "gold": it.answer},
                score=float(score),
            )
            print(f"Q: {it.question}\nA: {out.answer}  | gold: {it.answer}  | EM={score:.0f}\n")
        print(f"Mean EM: {np.mean(s_list):.3f}")

    # GEPA optimization (Algorithms 1 & 2)
    if args.optimize:
        print("\n=== Running GEPA optimization ===")
        gepa = GEPA(
            system_factory=system_factory,
            train_items=data,
            eval_metric=metric,
            minibatch_size=args.minibatch,
            pareto_set_size=args.pareto,
            seed=args.seed,
            merge_prob=args.merge_prob,
        )
        winner, best_mean = gepa.optimize(seed_prompts=prompts, rollout_budget=args.budget)
        print("\nBest Pareto mean after GEPA:", f"{best_mean:.3f}")
        print("\n--- Improved prompts (diff vs start) ---")
        for k in sorted(winner.prompts.keys()):
            if winner.prompts[k].strip() != prompts[k].strip():
                print(f"\n[{k}]")
                print(winner.prompts[k])
        # Final pass with improved prompts
        final_sys = system_factory(winner.prompts)
        s_list2 = []
        if args.task == "horror":
            for it in data:
                out = final_sys.run(it)
                s_list2.append(metric(out.answer, ""))
            print(f"\nFinal Mean score with evolved prompts: {np.mean(s_list2):.3f}")
        else:
            for it in data:
                out = final_sys.run(it)
                s_list2.append(metric(out.answer, it.answer))
            print(f"\nFinal Mean EM with evolved prompts: {np.mean(s_list2):.3f}")

if __name__ == "__main__":
    main()