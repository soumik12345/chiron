"""Tool base classes for the chiron ReAct agent (adapted from chiron).

A :class:`Tool` declares its name, description, and parameters, exposes an OpenAI
function-calling schema via :meth:`to_json_schema`, and implements an **async**
:meth:`forward`. The one production affordance kept from chiron's tool model is
`requires_approval` — when True the agent loop pauses for human confirmation
before executing the tool (unless approval is auto-granted).

A tool's `forward` may optionally declare `tool_call_id` and/or `signal`
parameters; the :class:`~chiron.core.router.ToolRouter` injects them when present.
`signal` is the run's :class:`~chiron.core.cancellation.CancellationToken`, which a
long-running tool should poll so `ReactAgent.cancel()` can stop it promptly.
"""

from __future__ import annotations

from typing import Any, Literal

import weave
from pydantic import BaseModel

ToolType = Literal["string", "number", "boolean", "object", "array", "any"]


class ToolParameter(BaseModel):
    """Schema for a single tool parameter.

    Attributes:
        param_name (str): The parameter name as it appears in the function signature.
        description (str): Human-readable description surfaced to the LLM.
        tool_type (ToolType): JSON Schema primitive type for the parameter.
        required (bool): Whether the parameter must be supplied by the model.
            Defaults to True.
        nullable (bool): When True, the generated schema allows `null` in addition
            to the declared type. Defaults to False.
    """

    param_name: str
    description: str
    tool_type: ToolType
    required: bool = True
    nullable: bool = False


class Tool(BaseModel):
    """Base class for all tools callable by the agent.

    Subclasses set the class attributes and implement `async forward(...)`.

    Attributes:
        tool_name (str): Unique name used by the LLM to invoke this tool.
        description (str): Human-readable description surfaced to the LLM.
        parameters (list[ToolParameter]): Ordered list of parameter definitions used
            to build the OpenAI function-calling schema.
        requires_approval (bool): When True, the agent loop pauses for human
            confirmation before executing (unless approval is auto-granted).
            Defaults to False.
        parameters_schema (dict[str, Any] | None): When set, replaces the
            auto-generated parameters schema in :meth:`to_json_schema`. Useful for
            tools with complex nested schemas. Defaults to None.
    """

    tool_name: str
    description: str
    parameters: list[ToolParameter] = []
    requires_approval: bool = False
    parameters_schema: dict[str, Any] | None = None

    def _parameter_schema(self, parameter: ToolParameter) -> dict[str, Any]:
        """Build the JSON Schema fragment for a single parameter.

        Args:
            parameter (ToolParameter): The parameter definition to convert.

        Returns:
            dict[str, Any]: A JSON Schema-compatible dict with a `description` and,
                unless the tool type is `"any"`, a `type` key. Nullable parameters
                use a two-element type list (e.g. `["string", "null"]`).
        """
        schema: dict[str, Any] = {"description": parameter.description}
        if parameter.tool_type == "any":
            return schema
        if parameter.nullable:
            schema["type"] = [parameter.tool_type, "null"]
        else:
            schema["type"] = parameter.tool_type
        return schema

    def to_json_schema(self) -> dict[str, Any]:
        """Return the OpenAI function-calling schema for this tool.

        When :attr:`parameters_schema` is set it is used verbatim; otherwise the
        schema is generated from :attr:`parameters`.

        Returns:
            dict[str, Any]: A `{"type": "function", "function": {...}}` dict suitable
                for the `tools` list in an OpenAI chat completion request.
        """
        if self.parameters_schema is not None:
            params = self.parameters_schema
        else:
            params = {
                "type": "object",
                "properties": {
                    p.param_name: self._parameter_schema(p) for p in self.parameters
                },
                "required": [p.param_name for p in self.parameters if p.required],
                "additionalProperties": False,
            }
        return {
            "type": "function",
            "function": {
                "name": self.tool_name,
                "description": self.description,
                "parameters": params,
            },
        }

    @weave.op
    async def forward(self, **kwargs: Any) -> Any:
        """Execute the tool. Must be overridden by subclasses.

        The router injects `tool_call_id` and `signal` into `kwargs` when the
        overriding method declares those parameters in its signature. All other keys
        come directly from the model's parsed `arguments` dict.

        Args:
            **kwargs: Tool-specific keyword arguments parsed from the model's tool
                call, plus an optional `tool_call_id` and `signal` injected by the
                router.

        Returns:
            Any: JSON-serialisable result that the router stringifies and returns to
                the agent loop as a `role: tool` message.

        Raises:
            NotImplementedError: Always, unless overridden by a subclass.
        """
        raise NotImplementedError("Subclasses must implement async forward(...)")
