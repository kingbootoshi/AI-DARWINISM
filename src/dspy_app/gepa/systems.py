"""
Small runtime systems that wire ModuleSpec-defined modules into an executable
pipeline for each task.
"""

from __future__ import annotations

from typing import Any, Dict
import dspy

from .types import ModuleSpec, SystemIO, HotpotItem, HorrorItem


class HotpotSystem:
    """Minimal, modular DSPy pipeline with four modules.

    summarize1 -> create_query_hop2 -> summarize2 -> final_answer
    """

    def __init__(self, lm: dspy.LM, module_specs: Dict[str, ModuleSpec]):
        self.lm = lm
        self.specs = module_specs
        self.modules = {k: v.make_module() for k, v in self.specs.items()}

    def run(self, item: HotpotItem, capture_traces: bool = True) -> SystemIO:  # noqa: ARG002
        inter: Dict[str, Any] = {}
        m = self.modules["summarize1"]
        s1 = m(question=item.question, passages=item.passages1).summary
        inter["summary1"] = s1

        m = self.modules["create_query_hop2"]
        q2 = m(question=item.question, summary_1=s1).query
        inter["query2"] = q2

        m = self.modules["summarize2"]
        s2 = m(question=item.question, context=s1, passages=item.passages2).summary
        inter["summary2"] = s2

        m = self.modules["final_answer"]
        ans = m(question=item.question, summary_1=s1, summary_2=s2).answer
        return SystemIO(inter=inter, answer=ans)


class HorrorSystem:
    """Single-module horror microfiction writer. Output is reused as answer."""

    def __init__(self, lm: dspy.LM, module_specs: Dict[str, ModuleSpec]):
        self.lm = lm
        self.specs = module_specs
        self.modules = {k: v.make_module() for k, v in self.specs.items()}

    def run(self, item: HorrorItem, capture_traces: bool = True) -> SystemIO:  # noqa: ARG002
        story = self.modules["horror_writer"](prompt=item.prompt).story
        return SystemIO(inter={"story": story}, answer=story)


