"""
Horror microfiction task: seed prompt, system factory, seed data, and judge.
"""

from __future__ import annotations

from typing import Callable, Dict, List
import json
import ast
import dspy

from ..gepa.types import ModuleSpec, HorrorItem
from ..gepa.signatures import build_signature
from ..gepa.systems import HorrorSystem
from webui.state import state as ui_state


BASE_SEED_PROMPTS_HORROR: Dict[str, str] = {
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
    return {
        "horror_writer": ModuleSpec(
            name="horror_writer",
            inputs={"prompt": "seed/premise or constraint"},
            outputs={"story": "two sentences of horror"},
            instruction=prompts["horror_writer"],
        )
    }


def system_factory_from_prompts_horror(lm: dspy.LM, prompts: Dict[str, str]) -> Callable[[Dict[str, str]], HorrorSystem]:
    def factory(updated_prompts: Dict[str, str]) -> HorrorSystem:
        specs = make_specs_from_prompts_horror(updated_prompts)
        return HorrorSystem(lm=lm, module_specs=specs)

    return factory


def build_horror_judge() -> dspy.Module:
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
    # Prefer human ratings if available in recent UI events
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

    global _horror_judge
    if _horror_judge is None:
        _horror_judge = build_horror_judge()
    out = _horror_judge(story=pred_story)
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


def horror_metric(pred: str, gold_unused: str) -> float:  # noqa: ARG001
    return _horror_metric_from_human_or_llm(pred)


def horror_seed_data() -> List[HorrorItem]:
    seeds = [
        "A baby monitor that starts narrating future sounds.",
        "A calendar where tomorrow’s date is torn out by someone else.",
        "A mirror that returns your smile a beat too late.",
        "A house whose lights flicker in Morse code spelling a name.",
        "Footsteps that stop when you stop—but never resume.",
    ]
    return [HorrorItem(prompt=s) for s in seeds]


