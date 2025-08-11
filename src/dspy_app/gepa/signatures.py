"""
Utilities for building DSPy signatures dynamically.
"""

from __future__ import annotations

import dspy


def build_signature(name: str, instruction: str, inputs: dict[str, str], outputs: dict[str, str]):
    """Create a `dspy.Signature` subclass with given IO schema and instruction.

    This mirrors the behavior from the original monolithic file, but keeps the
    responsibility in a dedicated module for reusability.
    """
    ns: dict[str, object] = {"__doc__": instruction}
    for in_name, desc in inputs.items():
        ns[in_name] = dspy.InputField(desc=desc)
    for out_name, desc in outputs.items():
        ns[out_name] = dspy.OutputField(desc=desc)
    Sig = type(name, (dspy.Signature,), ns)
    return Sig


