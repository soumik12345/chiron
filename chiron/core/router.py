"""Tool registry + dispatcher.

Holds the available tools, exports their OpenAI schemas for the LLM, and routes a
parsed tool call to the right `async forward`. It introspects each tool's
`forward` signature and injects `tool_call_id`, `signal`, and `on_update`
when the tool declares them, so most tools stay pure while a few can learn which
call they belong to, poll the run's cancellation token, or report progress.

Tools can be **deferred**: registered but hidden from the model until something
activates them (typically a :class:`~diorama.core.results.ToolResult` carrying
`added_tool_names`). This keeps a large toolset out of the prompt until a
discovery step establishes it is relevant.

Tool execution errors are caught and returned as a failed
:class:`~diorama.core.results.ToolResult` rather than crashing the loop — this is
what lets the model observe a failure and adapt on the next turn.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

from chiron.core.results import ToolResult, stringify
from chiron.core.tool import Tool

ProgressCallback = Callable[[Any], None]


class ToolRouter:
    """Registry and async dispatcher for :class:`Tool` instances.

    Attributes:
        tools (dict[str, Tool]): Mapping from tool name to tool instance.
        active (set[str]): Names currently exposed to the model. Deferred tools are
            registered but absent from this set until :meth:`activate` is called.
    """

    def __init__(
        self,
        tools: list[Tool] | None = None,
        deferred_tools: list[Tool] | None = None,
    ) -> None:
        """Initialise the router.

        Args:
            tools (list[Tool] | None): Tools exposed to the model immediately.
            deferred_tools (list[Tool] | None): Tools registered but hidden until
                activated.
        """
        self.tools: dict[str, Tool] = {}
        self.active: set[str] = set()
        for tool in tools or []:
            self.register(tool)
        for tool in deferred_tools or []:
            self.register(tool, active=False)

    def register(self, tool: Tool, *, active: bool = True) -> None:
        """Add a tool to the registry, keyed by its `tool_name`.

        Args:
            tool (Tool): The tool to register. If a tool with the same name already
                exists it is silently replaced.
            active (bool): Whether the tool is immediately visible to the model.
        """
        self.tools[tool.tool_name] = tool
        if active:
            self.active.add(tool.tool_name)
        else:
            self.active.discard(tool.tool_name)

    def activate(self, *names: str) -> list[str]:
        """Expose deferred tools to the model.

        Unknown names are ignored — a tool asking for something that does not exist
        should not break the run.

        Args:
            *names (str): Tool names to activate.

        Returns:
            list[str]: The names that were actually newly activated.
        """
        activated = []
        for name in names:
            if name in self.tools and name not in self.active:
                self.active.add(name)
                activated.append(name)
        return activated

    def get(self, name: str) -> Tool | None:
        """Look up a tool by name, whether active or deferred."""
        return self.tools.get(name)

    def get_tool_specs_for_llm(self) -> list[dict[str, Any]]:
        """Return the *active* tool schemas in OpenAI function-calling format."""
        return [
            tool.to_json_schema()
            for name, tool in self.tools.items()
            if name in self.active
        ]

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        tool_call_id: str | None = None,
        signal: Any = None,
        on_update: ProgressCallback | None = None,
    ) -> ToolResult:
        """Execute `tool_name` and return its normalised result.

        A tool may return a :class:`~chiron.core.results.ToolResult` or any plain
        value (which is stringified). Deferred tools can be invoked once activated;
        calling an unregistered — or still-deferred — tool yields a failed result
        rather than raising.

        Args:
            tool_name (str): The name of the tool to invoke.
            arguments (dict[str, Any]): Parsed arguments from the model's tool call.
            tool_call_id (str | None): The id of the originating tool call, injected
                into `forward` when declared.
            signal (CancellationToken | None): The run's cancellation token, injected
                into `forward` when declared so long-running tools can bail out.
            on_update (ProgressCallback | None): Progress sink, injected into
                `forward` when declared. Each call reports a partial result.

        Returns:
            ToolResult: The tool's output, with `is_error` set when it failed.
        """
        tool = self.tools.get(tool_name)
        if tool is None or tool_name not in self.active:
            # A deferred tool that nothing has activated is not callable, and answering
            # "unknown" rather than "not yet available" is both true from the model's
            # side (it was never in the schemas) and keeps the hidden set hidden.
            return ToolResult.error(stringify({"error": f"Unknown tool: {tool_name}"}))

        # Inject the optional context parameters the tool actually declares.
        try:
            params = inspect.signature(tool.forward).parameters
        except (TypeError, ValueError):
            params = {}

        call_kwargs = dict(arguments)
        if "tool_call_id" in params:
            call_kwargs["tool_call_id"] = tool_call_id
        if "signal" in params:
            call_kwargs["signal"] = signal
        if "on_update" in params:
            call_kwargs["on_update"] = on_update or (lambda _partial: None)

        try:
            return ToolResult.coerce(await tool.forward(**call_kwargs))
        except Exception as e:  # noqa: BLE001 — surfaced to the agent, not fatal
            return ToolResult.error(stringify({"error": str(e)}))
