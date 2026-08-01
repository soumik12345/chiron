"""Optional terminal tool.

Termination semantics: the diorama loop ends a turn when the model returns **no
tool calls** — there is no *required* `final_answer` tool. :class:`FinalAnswerTool`
is provided for callers who want an explicit "I'm done" affordance, but the loop
does not depend on it.
"""

from __future__ import annotations

from typing import Any

import weave

from chiron.core.tool import Tool, ToolParameter


class FinalAnswerTool(Tool):
    """Passthrough tool: returns its `answer` argument unchanged."""

    tool_name: str = "final_answer"
    description: str = (
        "Provide a final answer to the task. Optional — you may also finish by "
        "replying in plain text with no tool call."
    )
    parameters: list[ToolParameter] = [
        ToolParameter(
            param_name="answer",
            tool_type="any",
            description="The final answer to the task.",
        )
    ]

    @weave.op
    async def forward(self, answer: Any) -> Any:
        """Return the `answer` argument unchanged.

        Args:
            answer (Any): The final answer value to pass through.

        Returns:
            Any: The `answer` argument, unmodified.
        """
        return answer
