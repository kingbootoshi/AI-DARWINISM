"""
Horror microfiction task: seed prompt, system factory, seed data, and judge.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Any
import json
import ast
import re
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
    # Carry provider kwargs so hot_lm inherits API base/key
    provider_kwargs: Dict[str, Any] = {}
    if getattr(lm, "api_base", None):  # type: ignore[attr-defined]
        provider_kwargs["api_base"] = lm.api_base  # type: ignore[attr-defined]
    if getattr(lm, "api_key", None):  # type: ignore[attr-defined]
        provider_kwargs["api_key"] = lm.api_key  # type: ignore[attr-defined]

    def factory(updated_prompts: Dict[str, str]) -> HorrorSystem:
        specs = make_specs_from_prompts_horror(updated_prompts)
        return HorrorSystem(lm=lm, module_specs=specs, provider_kwargs=provider_kwargs)

    return factory


def build_horror_judge(judge_model_id: str | None = None) -> dspy.Module:
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
    # Use a possibly different LM for judging (temperature 0 for consistency)
    lm = None
    if judge_model_id:
        lm = dspy.LM(judge_model_id, temperature=0.0)
    return dspy.Predict(Sig, lm=lm) if lm else dspy.Predict(Sig)


_horror_judge = None
_horror_judge_model_id: str | None = None


def _two_sentence_gate(text: str) -> bool:
    sents = [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s]
    return len(sents) == 2


def _horror_metric_from_human_or_llm(pred_story: str) -> float:
    # Hard enforce two-sentence rule
    if not _two_sentence_gate(pred_story):
        return 0.0
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
    global _horror_judge_model_id
    if _horror_judge is None:
        _horror_judge = build_horror_judge(_horror_judge_model_id)
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


def horror_feedback_extractor(item: HorrorItem, io) -> Dict[str, Any]:  # type: ignore[override]
    """Compute judge subscores + reason for the produced story for reflector.

    Returns a dict with keys: 'judge' {...} and 'issue' (comma-joined labels).
    """
    global _horror_judge, _horror_judge_model_id
    story = getattr(io, "answer", "")
    if not _two_sentence_gate(story):
        return {
            "judge": {
                "scariness": 0,
                "suspense": 0,
                "originality": 0,
                "clarity": 0,
                "rule_two_sentences": 0,
                "explanation": "not two sentences",
            },
            "issue": "not_two_sentences",
        }
    if _horror_judge is None:
        _horror_judge = build_horror_judge(_horror_judge_model_id)
    out = _horror_judge(story=story)
    raw = out.json if isinstance(out.json, str) else str(out.json)
    try:
        data = json.loads(raw)
    except Exception:
        try:
            data = ast.literal_eval(raw)
        except Exception:
            data = {}
    issues = []
    if int(data.get("rule_two_sentences", 0)) < 5:
        issues.append("not_two_sentences")
    if int(data.get("scariness", 0)) < 3:
        issues.append("not_scary")
    if int(data.get("originality", 0)) < 3:
        issues.append("cliche")
    if int(data.get("clarity", 0)) < 3:
        issues.append("unclear")
    return {
        "judge": {
            "scariness": int(data.get("scariness", 0)),
            "suspense": int(data.get("suspense", 0)),
            "originality": int(data.get("originality", 0)),
            "clarity": int(data.get("clarity", 0)),
            "rule_two_sentences": int(data.get("rule_two_sentences", 0)),
            "explanation": data.get("explanation", ""),
        },
        "issue": ",".join(issues) or "ok",
    }


def horror_seed_data() -> List[HorrorItem]:
    # Reddit best-of two-sentence horror prompts as seeds (user-provided)
    seeds = [
        "All my life, my parents have told me not to open the basement door, but I got curious and disobeyed them. What is that glowing ball in the sky and why does it hurt my eyes?",
        "When the kidnapper made me guess where he kept my daughter, I went for the basement and he said 'Correct!' allowing me to see her. But when I found her severed head in there, I learned that every other choice would have been correct as well.",
        "Whenever I considered killing myself to escape my parents' abuse, I'd just recite my mantra 'you don't deserve to die'. Ironically, now that they are old, hungry, covered in bedsores, and begging to be put out of their misery, I still have the same mantra.",
        "...she said last time, we're stuck in a time loop. Which really pisses me off because that's what...",
        "I framed the first letter I got as a police officer, from a woman thanking me after I'd supported her through her daughter's suicide. I passed it in my hallway every day for nearly eight years before realising the handwriting was the same as on the girl's suicide note.",
        "'Now be careful, that line of rock salt is the only thing keeping them out,' the man said, welcoming my group into his refuge. 'Sea salt,' I clarified, 'sea salt keeps us out.'",
        "My husband has been very upset with me since my failed suicide attempt. He’s crying nonstop and he won’t acknowledge me.",
        "I was born blind but was lucky enough to have a loving mother who took care of my every need. Imagine the betrayal I felt when a stitch slipped and a ray of light hit my eye for the first time.",
        "As I slit her throat, I looked in her unblinking eyes and realised too late that she wanted to live. I knew it to be true because mirrors don't lie.",
        "Please, take me instead! I scream, grabbing at the two men who took my child. 'Sorry ma’am, children only' they said, as they continue loading up the last lifeboat on the ship.",
        # Extra seeds to improve diversity/learning signal
        "A doorbell camera that records people who never existed.",
        "Your phone’s 'live photo' shows three extra frames.",
        "Teeth marks on your spoon don’t match your own.",
        "Caption under your childhood photo changes every day.",
        "Your cat sits and stares at the new crack in the mirror.",
        "The voicemail transcription uses your nickname—one no one knows.",
        "Your reflection keeps the bruise you swore had healed.",
        "A second birthday candle reappears every time you blow them out.",
    ]
    return [HorrorItem(prompt=s) for s in seeds]


def set_horror_judge_model(model_id: str | None) -> None:
    """Optionally swap judge model to reduce self-bias during scoring."""
    global _horror_judge, _horror_judge_model_id
    _horror_judge = None
    _horror_judge_model_id = model_id

    


