"""
Hotpot-like toy task: seed prompts, factories, tiny dataset, and metric.
"""

from __future__ import annotations

from typing import Callable, Dict, List
import dspy

from ..gepa.types import ModuleSpec, HotpotItem
from ..gepa.signatures import build_signature
from ..gepa.systems import HotpotSystem


def normalize_text(s: str) -> str:
    return " ".join(s.strip().lower().split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_text(pred) == normalize_text(gold))


BASE_SEED_PROMPTS: Dict[str, str] = {
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


def toy_hotpot_data() -> List[HotpotItem]:
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



