"""
Custom DSPy callbacks to surface agent reasoning and tool actions.

These callbacks complement `dspy.inspect_history()` by logging intermediate
module and tool steps to the console and files via Loguru.
"""

from __future__ import annotations

from loguru import logger
from dspy.utils.callback import BaseCallback


class AgentLoggingCallback(BaseCallback):
    """Emit readable logs for reasoning vs. acting steps.

    Logs with different levels to make the chain-of-thought and tool usage
    visually distinct while running.
    """

    def on_module_start(self, call_id, inputs):  # noqa: D401
        logger.info("<b>Module start</b> id={} inputs={} ", call_id, inputs)

    def on_module_end(self, call_id, outputs, exception):  # noqa: D401
        if exception:
            logger.error("<red>Module error</red> id={} exc={}", call_id, exception)
            return
        # Heuristic: surface reasoning fields prominently when present
        step = "Reasoning" if any(k.startswith("Thought") or k == "reasoning" for k in outputs.keys()) else "Acting"
        logger.success("== {} Step == id={} outputs={} ", step, call_id, outputs)

    def on_tool_start(self, call_id, inputs):  # noqa: D401
        tool_name = inputs.get("tool_name", "tool") if isinstance(inputs, dict) else "tool"
        logger.debug("<cyan>Tool start</cyan> {} id={} inputs={}", tool_name, call_id, inputs)

    def on_tool_end(self, call_id, outputs, exception):  # noqa: D401
        if exception:
            logger.warning("<yellow>Tool error</yellow> id={} exc={}", call_id, exception)
        else:
            logger.info("<cyan>Tool end</cyan> id={} outputs={}", call_id, outputs)

    def on_lm_start(self, call_id, inputs):  # noqa: D401
        logger.debug("LM call start id={} keys={}", call_id, list(inputs.keys()) if isinstance(inputs, dict) else type(inputs))

    def on_lm_end(self, call_id, outputs, exception):  # noqa: D401
        if exception:
            logger.error("LM call error id={} exc={}", call_id, exception)
        else:
            logger.debug("LM call end id={} keys={}", call_id, list(outputs.keys()) if isinstance(outputs, dict) else type(outputs))


