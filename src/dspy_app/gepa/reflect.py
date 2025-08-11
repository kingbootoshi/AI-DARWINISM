"""
Reflective mutation utilities (UPDATEPROMPT-style) used by GEPA.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
import dspy

from .signatures import build_signature


def build_reflector_module(lm: dspy.LM):
    """Return a small LLM-as-optimizer module that proposes new instructions."""
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
    """Compress traces + feedback into a small human-readable report."""
    lines = [f"Module: {module_name}", "Cases:"]
    for t in batch_traces[:max_items]:
        score = t.get("score", None)
        lines.append(f"- score={score} :: issue={t.get('issue','?')}")
        inputs = t.get("inputs", {})
        outs = t.get("outputs", {})
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
    """Call the reflector to propose a new instruction and parse its JSON output."""
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
    except Exception:  # pragma: no cover - defensive parsing fallback
        return current_instruction


