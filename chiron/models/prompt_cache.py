"""Explicit prompt-cache breakpoints for Anthropic models.

OpenRouter forwards provider-native prompt caching: Anthropic models need
explicit `cache_control: {type: ephemeral}` markers, while OpenAI-family models
cache eligible prefixes automatically and ignore markers. So this module is a
**no-op for everything except Anthropic**.

Two breakpoints are placed (Anthropic allows up to 4):

* the **last tool** spec: caches the whole, stable tool-schema block;
* one **rolling** breakpoint on the closest `system`/`user` text message
  *before* the newest one: which, by Anthropic's prefix rule, also caches the
  system prompt and every earlier turn. The newest turn stays dynamic, so each
  turn reads the prior cache and writes only its extension.

Crucially, the live conversation/tool objects are never mutated; only shallow
copies are marked for the outgoing request, so persisted history and the tool
router are untouched.
"""

from __future__ import annotations

from typing import Any

_CACHE_CONTROL = {"type": "ephemeral"}
_CACHEABLE_ROLES = {"system", "user"}


def model_supports_explicit_cache(model_id: str) -> bool:
    """True for Anthropic/Claude models, where explicit markers do real work."""
    mid = (model_id or "").lower()
    return "anthropic/" in mid or "claude" in mid


# -- message helpers ----------------------------------------------------------
def _has_cacheable_text(content: Any) -> bool:
    """Return True if the message content contains at least one non-empty text block.

    Args:
        content (Any): A message `content` value, either a plain string or a
            list of content blocks in OpenAI block format.

    Returns:
        bool: True if a non-empty text string or text block is present.
    """
    if isinstance(content, str):
        return bool(content)
    if not isinstance(content, list):
        return False
    return any(
        isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
        and bool(block.get("text"))
        for block in content
    )


def _cache_target_index(messages: list[dict[str, Any]]) -> int | None:
    """Index of the rolling breakpoint: the last cacheable system/user message
    *before* the newest message (which is left dynamic)."""
    if len(messages) < 2:
        return None
    for idx in range(len(messages) - 2, -1, -1):
        message = messages[idx]
        if message.get("role") not in _CACHEABLE_ROLES:
            continue
        if _has_cacheable_text(message.get("content")):
            return idx
    return None


def _content_with_cache_control(content: Any) -> list[dict[str, Any]]:
    """Return block-form content with cache_control on its last text block.

    Converts a plain string to a single-element block list and marks the last
    text block with an Anthropic `cache_control` marker. Input is never mutated.

    Args:
        content (Any): A message `content` value, either a plain string or a
            list of content blocks.

    Returns:
        list[dict[str, Any]]: Content in block-list form with the cache marker applied.
    """
    if isinstance(content, str):
        return [
            {"type": "text", "text": content, "cache_control": dict(_CACHE_CONTROL)}
        ]

    blocks = [dict(block) if isinstance(block, dict) else block for block in content]
    for idx in range(len(blocks) - 1, -1, -1):
        block = blocks[idx]
        if (
            isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
            and bool(block.get("text"))
        ):
            cached = dict(block)
            cached["cache_control"] = dict(_CACHE_CONTROL)
            blocks[idx] = cached
            break
    return blocks


def _tools_with_cache_control(
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    """Return a copy of the tool list with a cache_control marker on the last entry.

    A no-op (returns the original) when the list is empty or None. Only the last
    tool dict is shallow-copied; the list itself is always a new object.

    Args:
        tools (list[dict[str, Any]] | None): Tool schemas in OpenAI function-calling
            format, or None.

    Returns:
        list[dict[str, Any]] | None: A new list with the cache marker on the last tool,
            or the original value when there are no tools.
    """
    if not tools:
        return tools
    cached_tools = list(tools)
    last_tool = dict(cached_tools[-1])
    last_tool["cache_control"] = dict(_CACHE_CONTROL)
    cached_tools[-1] = last_tool
    return cached_tools


def apply_prompt_caching(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    model_id: str,
    *,
    enabled: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """Return (messages, tools) with cache breakpoints for Anthropic models.

    A no-op (returns the inputs unchanged) when disabled or for non-Anthropic
    models. Inputs are never mutated, only the marked message/tool are copied.
    """
    if not (enabled and model_supports_explicit_cache(model_id)):
        return messages, tools

    cached_tools = _tools_with_cache_control(tools)
    idx = _cache_target_index(messages)
    if idx is None:
        return messages, cached_tools

    cached_message = dict(messages[idx])
    cached_message["content"] = _content_with_cache_control(
        cached_message.get("content")
    )
    cached_messages = list(messages)
    cached_messages[idx] = cached_message
    return cached_messages, cached_tools


# -- usage / observability ----------------------------------------------------
def _get(obj: Any, key: str) -> Any:
    """Extract a field from an object or dict, returning None if absent.

    Args:
        obj (Any): A dict, an object with attributes, or None.
        key (str): The key or attribute name to retrieve.

    Returns:
        Any: The field value, or None if obj is None or the key is missing.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def extract_cache_tokens(usage: Any) -> tuple[int, int]:
    """Return (cache_read_tokens, cache_write_tokens) from a usage object/dict.

    Handles both the Anthropic shape (`cache_read_input_tokens` /
    `cache_creation_input_tokens`) and the OpenAI/litellm-normalised shape
    (`prompt_tokens_details.cached_tokens`). Missing fields read as 0.
    """
    if usage is None:
        return 0, 0
    read = _get(usage, "cache_read_input_tokens")
    if not read:
        details = _get(usage, "prompt_tokens_details")
        read = _get(details, "cached_tokens")
    write = _get(usage, "cache_creation_input_tokens")
    return int(read or 0), int(write or 0)
