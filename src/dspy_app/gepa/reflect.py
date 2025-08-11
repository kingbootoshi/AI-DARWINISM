"""
Reflective mutation utilities (UPDATEPROMPT-style) used by GEPA.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
import dspy
import os
import re

from .signatures import build_signature


def build_reflector_module(lm_model_id: Optional[str] = None, temperature: float = 0.7):
    """Construct a small LLM-as-optimizer module for proposing instructions.

    Why this design
    ---------------
    Passing an `lm` instance directly to `dspy.Predict(..., lm=hot_lm)` can leak
    the LM object into downstream LiteLLM/OpenAI kwargs in some DSPy versions,
    which then triggers serialization errors ("Object of type LM is not JSON
    serializable"). To avoid that, we return a predictor without an attached
    LM and perform a temporary `dspy.settings.configure(lm=hot_lm)` swap at
    call time in `update_instruction_via_reflection`.
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
    model_id = lm_model_id or os.getenv("DSPY_MODEL")
    # Build a hotter LM but do NOT attach it to the module here; see docstring.
    try:
        from config.settings import get_settings
        settings = get_settings()
        hot_lm = dspy.LM(
            model_id,
            temperature=temperature,
            api_base=settings.api_base,
            api_key=settings.api_key,
        ) if model_id else None
    except Exception:
        hot_lm = dspy.LM(model_id, temperature=temperature) if model_id else None

    predictor = dspy.Predict(Sig)
    # Return both the predictor and the hotter LM for the caller to use in a
    # temporary configuration context.
    return predictor, hot_lm


def make_batch_report(module_name: str, batch_traces: List[Dict[str, Any]], max_items: int = 4) -> str:
    """Compress traces + feedback into a small human-readable report.

    Includes judge subscores and short explanation when available so the
    reflector gets actionable signals (the "why").
    """
    lines = [f"Module: {module_name}", "Cases:"]
    for t in batch_traces[:max_items]:
        score = t.get("score", None)
        lines.append(f"- score={score} :: issue={t.get('issue','?')}")
        j = t.get("judge", {})
        if j:
            lines.append(
                "  subs: "
                f"S={j.get('scariness')} Su={j.get('suspense')} "
                f"O={j.get('originality')} C={j.get('clarity')} "
                f"R2={j.get('rule_two_sentences')} ; note: {j.get('explanation','')}"
            )
        story = t.get("outputs", {}).get("answer", "")
        if story:
            lines.append(f"  story: {story[:180]}")
    return "\n".join(lines)


def update_instruction_via_reflection(
    lm: dspy.LM,
    module_name: str,
    io_schema: str,
    current_instruction: str,
    batch_traces: List[Dict[str, Any]],
    *,
    k: int = 1,
    reflect_temp: float = 0.7,
    min_diff_chars: int = 24,
) -> List[str]:
    """Propose up to K diverse new instructions; return unique list of strings."""
    proposals: List[str] = []
    for _ in range(max(1, k)):
        # Build reflector module and a higher-temperature LM. We will temporarily
        # swap `dspy.settings.lm` to avoid passing an LM object through kwargs.
        reflector, hot_lm = build_reflector_module(temperature=reflect_temp)
        report = make_batch_report(module_name, batch_traces)
        prev_lm = dspy.settings.lm
        try:
            if hot_lm is not None:
                dspy.settings.configure(lm=hot_lm)
            out = reflector(
                module_name=module_name,
                io_schema=io_schema,
                current_instruction=current_instruction,
                batch_report=report,
            )
        finally:
            # Always restore the previous LM configuration
            dspy.settings.configure(lm=prev_lm)
        try:
            ni = json.loads(out.json).get("new_instruction", "").strip()
        except Exception:
            ni = current_instruction
        if not ni or ni == current_instruction:
            continue
        # Skip trivial edits by rough char-diff
        if len(set(ni) ^ set(current_instruction)) < min_diff_chars:
            continue
        proposals.append(ni)

    # De-duplicate by normalized whitespace
    uniq: List[str] = []
    seen: set[str] = set()
    for p in proposals:
        key = re.sub(r"\s+", " ", p.strip().lower())
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


