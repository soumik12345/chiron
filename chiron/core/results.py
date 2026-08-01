"""Rich tool results.

A tool's return value used to be flattened to a string, which threw away three
things the agent and the UI both want: whether the call *failed*, structured data a
renderer could format, and non-text output such as images.

:class:`ToolResult` carries all of it. Tools may keep returning plain values —
:meth:`ToolResult.coerce` wraps them — or return a `ToolResult` when they need the
extra channels:

* `content` — ordered text/image blocks. Only the text reaches the `role: tool`
  message; images are re-attached as a follow-up user message (see
  :func:`image_followup_message`), because the Chat Completions tool-result slot is
  text-only.
* `details` — arbitrary structured data that never reaches the model. It rides on
  the tool-execution event for renderers and logs.
* `is_error` — surfaced on events and preserved in durable history.
* `added_tool_names` — tools to expose to the model from now on, letting a tool
  unlock further tools (a discovery step that reveals a domain toolset).
* `terminate` — end the run after this turn, for tools that *are* the answer.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

MAX_INLINE_IMAGE_BYTES = 5 * 1024 * 1024


class TextBlock(BaseModel):
    """A run of text produced by a tool."""

    type: Literal["text"] = "text"
    text: str


class ImageBlock(BaseModel):
    """An image produced by a tool, carried as base64.

    Attributes:
        data (str): Base64-encoded image bytes (no data-URI prefix).
        mime_type (str): The image's media type, e.g. `image/png`.
    """

    type: Literal["image"] = "image"
    data: str
    mime_type: str = "image/png"

    def to_data_uri(self) -> str:
        """Return the block as a `data:` URI suitable for an `image_url` part."""
        return f"data:{self.mime_type};base64,{self.data}"


ContentBlock = TextBlock | ImageBlock


def stringify(value: Any) -> str:
    """Coerce an arbitrary tool return value into text.

    Strings pass through unchanged. Anything else is JSON-serialised with
    `default=str` so non-serialisable objects fall back to their `str()`
    representation; if JSON serialisation itself raises, the raw `str()` is used.
    """
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


class ToolResult(BaseModel):
    """The full outcome of one tool call.

    Attributes:
        content (list[ContentBlock]): Ordered text/image blocks.
        details (Any): Structured data for renderers and logs. Never sent to the model.
        is_error (bool): Whether the call failed. Surfaced on events and kept in
            durable history.
        added_tool_names (list[str] | None): Deferred tools to expose from now on.
        terminate (bool): End the run after the current turn.
    """

    content: list[ContentBlock] = Field(default_factory=list)
    details: Any = None
    is_error: bool = False
    added_tool_names: list[str] | None = None
    terminate: bool = False

    @property
    def text(self) -> str:
        """The concatenated text blocks — what the model actually sees."""
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))

    @property
    def images(self) -> list[ImageBlock]:
        """The image blocks, in order."""
        return [b for b in self.content if isinstance(b, ImageBlock)]

    @classmethod
    def from_text(
        cls, text: str, *, is_error: bool = False, **kwargs: Any
    ) -> ToolResult:
        """Build a text-only result."""
        return cls(content=[TextBlock(text=text)], is_error=is_error, **kwargs)

    @classmethod
    def error(cls, message: str, **kwargs: Any) -> ToolResult:
        """Build a failed result carrying `message`."""
        return cls.from_text(message, is_error=True, **kwargs)

    @classmethod
    def coerce(cls, value: Any) -> ToolResult:
        """Wrap whatever a tool returned into a :class:`ToolResult`.

        Existing tools that return plain values keep working unchanged.

        Args:
            value (Any): A `ToolResult`, a content block, a list of blocks, or any
                other value (which is stringified).

        Returns:
            ToolResult: The normalised result.
        """
        if isinstance(value, ToolResult):
            return value
        if isinstance(value, (TextBlock, ImageBlock)):
            return cls(content=[value])
        if (
            isinstance(value, list)
            and value
            and all(isinstance(v, (TextBlock, ImageBlock)) for v in value)
        ):
            return cls(content=list(value))
        return cls.from_text(stringify(value))

    def model_text(self, *, placeholder: str = "[image]") -> str:
        """Return the text the `role: tool` message should carry.

        Image-only results still need *something* in the tool message, since the
        provider requires a non-empty result for every tool call.
        """
        text = self.text
        if text:
            return text
        if self.images:
            return " ".join(placeholder for _ in self.images)
        return ""


def image_followup_message(result: ToolResult, tool_name: str) -> dict[str, Any] | None:
    """Build the user message that carries a tool's images to the model.

    The Chat Completions tool-result slot accepts text only, so images are attached
    as a follow-up user message with `image_url` parts — the shape litellm maps to
    each provider's native image input.

    Args:
        result (ToolResult): The tool's result.
        tool_name (str): Name of the tool, mentioned in the message text.

    Returns:
        dict | None: The user message, or None when the result carried no images.
    """
    images = result.images
    if not images:
        return None
    parts: list[dict[str, Any]] = [
        {"type": "text", "text": f"Image output from the '{tool_name}' tool:"}
    ]
    for image in images:
        parts.append({"type": "image_url", "image_url": {"url": image.to_data_uri()}})
    return {"role": "user", "content": parts}


__all__ = [
    "ContentBlock",
    "ImageBlock",
    "MAX_INLINE_IMAGE_BYTES",
    "TextBlock",
    "ToolResult",
    "image_followup_message",
    "stringify",
]
