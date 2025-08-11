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
    """Single-module horror microfiction writer. Output is reused as answer.

    For horror generation, we default the module's LM to temperature=1 for
    diverse outputs unless the provided `lm` already specifies otherwise.
    """

    def __init__(self, lm: dspy.LM, module_specs: Dict[str, ModuleSpec], provider_kwargs: Dict[str, Any] | None = None):
        self.lm = lm
        self.specs = module_specs
        # Always source credentials from central settings to ensure OpenRouter base is used
        try:
            from config.settings import get_settings
            s = get_settings()
            self.provider_kwargs = {
                **(provider_kwargs or {}),
                "api_base": s.api_base,
                "api_key": s.api_key,
            }
        except Exception:
            self.provider_kwargs = dict(provider_kwargs or {})
        self.hot_lm = dspy.LM(lm.model, temperature=1.0, **self.provider_kwargs)
        # Build modules with default LM; we'll temporarily swap settings on run
        self.modules = {k: v.make_module() for k, v in self.specs.items()}

    def run(self, item: HorrorItem, capture_traces: bool = True) -> SystemIO:  # noqa: ARG002
        prev_lm = dspy.settings.lm
        try:
            dspy.settings.configure(lm=self.hot_lm)
            story = self.modules["horror_writer"](prompt=item.prompt).story
        finally:
            # Restore original LM configuration
            dspy.settings.configure(lm=prev_lm)
        return SystemIO(inter={"story": story}, answer=story)


