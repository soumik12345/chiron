"""Small, dependency-free demo tools.

These exist so the ReAct loop is runnable and testable out of the box without any
external services or API keys. Replace/extend them with chiron-domain tools as the
project grows.
"""

from __future__ import annotations

import ast
import operator
from datetime import datetime, timezone
from typing import Any

import weave

from chiron.core.tool import Tool, ToolParameter

# Binary/unary operators allowed in the calculator's safe expression evaluator.
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluate a parsed arithmetic AST, rejecting anything unsafe.

    Args:
        node (ast.AST): A node from `ast.parse(expr, mode="eval")`.

    Returns:
        float: The numeric result of the (sub)expression.

    Raises:
        ValueError: If the expression contains a construct that is not a number or a
            supported arithmetic operation.
    """
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported or unsafe expression")


class CalculatorTool(Tool):
    """Evaluate a basic arithmetic expression (`+ - * / // % **` and parentheses)."""

    tool_name: str = "calculator"
    description: str = (
        "Evaluate a basic arithmetic expression and return the numeric result. "
        "Supports + - * / // % ** and parentheses, e.g. '2 * (3 + 4) ** 2'."
    )
    parameters: list[ToolParameter] = [
        ToolParameter(
            param_name="expression",
            tool_type="string",
            description="The arithmetic expression to evaluate, e.g. '2 + 2 * 3'.",
        )
    ]

    @weave.op
    async def forward(self, expression: str) -> Any:
        """Safely evaluate `expression` and return the numeric result.

        Args:
            expression (str): An arithmetic expression.

        Returns:
            float | int: The evaluated result.

        Raises:
            ValueError: If the expression cannot be parsed or is unsafe.
        """
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as e:
            raise ValueError(f"Could not parse expression: {e}") from e
        return _safe_eval(tree)


class CurrentTimeTool(Tool):
    """Return the current date and time (UTC by default) as an ISO-8601 string."""

    tool_name: str = "current_time"
    description: str = (
        "Return the current date and time as an ISO-8601 string. Returns UTC unless a "
        "fixed UTC offset in hours is provided."
    )
    parameters: list[ToolParameter] = [
        ToolParameter(
            param_name="utc_offset_hours",
            tool_type="number",
            description="Optional fixed UTC offset in hours (e.g. 5.5). Defaults to 0 (UTC).",
            required=False,
            nullable=True,
        )
    ]

    @weave.op
    async def forward(self, utc_offset_hours: float | None = None) -> Any:
        """Return the current time at the given UTC offset.

        Args:
            utc_offset_hours (float | None): A fixed offset from UTC in hours. When
                None, UTC is used.

        Returns:
            str: The current time as an ISO-8601 string.
        """
        from datetime import timedelta

        tz = timezone.utc
        if utc_offset_hours is not None:
            tz = timezone(timedelta(hours=float(utc_offset_hours)))
        return datetime.now(tz).isoformat()
