"""
Core dataclasses and simple type wrappers used across GEPA modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ModuleSpec:
    """Declarative module recipe for building a `dspy.Predict` module.

    Fields
    ------
    name: str
        Human-readable module name and class stem for the signature.
    inputs: Dict[str, str]
        Mapping of input field name to short description.
    outputs: Dict[str, str]
        Mapping of output field name to short description.
    instruction: str
        Instruction text used as the signature docstring.
    """

    name: str
    inputs: Dict[str, str]
    outputs: Dict[str, str]
    instruction: str

    def make_module(self):  # lazy import to avoid circulars
        import dspy
        from .signatures import build_signature

        Sig = build_signature(self.name + "Sig", self.instruction, self.inputs, self.outputs)
        return dspy.Predict(Sig)


@dataclass
class SystemIO:
    """Container for intermediate traces and final answer/story."""

    inter: Dict[str, Any]
    answer: str


@dataclass
class HotpotItem:
    """Toy Hotpot-like item used for demo runs and tests."""

    question: str
    passages1: List[str]
    passages2: List[str]
    answer: str


@dataclass
class HorrorItem:
    """Simple item for two-sentence horror generation."""

    prompt: str


@dataclass
class Candidate:
    """A set of module instructions that define a system variant."""

    prompts: Dict[str, str]


