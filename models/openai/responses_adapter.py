# encoding:utf-8

"""
OpenAI Responses API <-> Chat Completions adapter.

Some routes (GPT-5 family on the official OpenAI API, anyrouter and similar
proxies) only expose ``POST /v1/responses`` and reject ``/v1/chat/completions``.
This module hides the difference: callers keep building chat-completion-shape
requests and consuming chat-completion-shape stream chunks; the adapter
rewrites the request before sending and rewrites events after receiving.

Three entry points:
- :func:`build_responses_request`  — chat-completion payload -> responses payload
- :func:`translate_stream`         — generator of responses SSE chunks ->
                                     generator of chat-completion stream chunks
- :func:`translate_sync`           — responses sync response dict ->
                                     chat-completion sync response dict

The chat-completion shape we emit matches what
``agent.protocol.agent_stream._call_llm_stream`` already parses, so no changes
are required upstream.
"""

import json
from typing import Any, Dict, Generator, Iterable, List, Optional


def build_responses_request(
    chat_payload: Dict[str, Any],
    *,
    reasoning_effort: Optional[str] = None,
) -> Dict[str, Any]:
    """Translate a chat-completions request payload into a responses payload.

    Mapping:
      - ``messages`` (system + user/assistant + tool)        -> ``input`` array
        with ``role`` / ``content`` items, plus separate ``function_call`` and
        ``function_call_output`` items for tool-use chains.
      - First leading system message                          -> ``instructions``
      - ``tools: [{type:function, function:{name,...}}]``     -> ``tools:
        [{type:function, name, description, parameters}]`` (flattened)
      - ``max_tokens``                                        -> ``max_output_tokens``
      - ``temperature`` / ``top_p`` / ``stream`` / ``model``  -> kept as-is
      - ``stop`` / ``frequency_penalty`` / ``presence_penalty`` are dropped —
        the Responses API rejects them.
      - ``reasoning_effort`` parameter -> ``reasoning: {effort: ...}``
    """
    payload: Dict[str, Any] = {}

    # Required / pass-through fields
    if "model" in chat_payload:
        payload["model"] = chat_payload["model"]
    if "stream" in chat_payload:
        payload["stream"] = chat_payload["stream"]
    # NOTE: GPT-5 family rejects ``temperature`` / ``top_p`` /
    # ``frequency_penalty`` / ``presence_penalty`` with
    # ``Unsupported parameter`` errors. The Responses API in general also
    # doesn't accept ``frequency_penalty`` / ``presence_penalty``. Default to
    # dropping them; callers who need temperature on a non-GPT-5 model that
    # somehow goes through the Responses endpoint can rebuild the payload
    # themselves.
    if "max_tokens" in chat_payload and chat_payload["max_tokens"] is not None:
        payload["max_output_tokens"] = chat_payload["max_tokens"]

    # Reasoning effort (gpt-5 family)
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}

    # Tools: chat-completions wraps each function inside {"type":"function",
    # "function": {...}}; Responses wants the function fields at the top level.
    if chat_payload.get("tools"):
        payload["tools"] = _flatten_tools(chat_payload["tools"])
        if chat_payload.get("tool_choice") is not None:
            payload["tool_choice"] = chat_payload["tool_choice"]

    # Messages -> input + instructions
    instructions, input_items = _convert_messages_to_input(
        chat_payload.get("messages") or []
    )
    if instructions:
        payload["instructions"] = instructions
    payload["input"] = input_items

    return payload


def _flatten_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for t in tools:
        if t.get("type") == "function" and isinstance(t.get("function"), dict):
            fn = t["function"]
            out.append({
                "type": "function",
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {"type": "object"}),
            })
        else:
            # Already flattened or non-function tool — pass through.
            out.append(t)
    return out


def _convert_messages_to_input(
    messages: List[Dict[str, Any]],
) -> (Optional[str], List[Dict[str, Any]]):
    """Convert chat-completion messages into a Responses ``input`` array.

    Returns (instructions_or_None, input_items).
    """
    instructions: Optional[str] = None
    items: List[Dict[str, Any]] = []
    msgs = list(messages)

    # Pull a leading system message into ``instructions``. (Mid-conversation
    # system messages, if any, are flattened into the input as a system role.)
    if msgs and msgs[0].get("role") == "system":
        instructions = _coerce_text(msgs[0].get("content"))
        msgs = msgs[1:]

    for msg in msgs:
        role = msg.get("role")

        if role == "tool":
            # OpenAI chat-completion tool result -> function_call_output item
            items.append({
                "type": "function_call_output",
                "call_id": msg.get("tool_call_id", ""),
                "output": _coerce_text(msg.get("content", "")),
            })
            continue

        if role == "assistant":
            text = _coerce_text(msg.get("content"))
            if text:
                items.append({
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                })
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {})
                items.append({
                    "type": "function_call",
                    "call_id": tc.get("id", ""),
                    "name": fn.get("name", ""),
                    "arguments": fn.get("arguments", "") or "",
                })
            continue

        # user / system / other: emit a message item with input_text content
        text = _coerce_text(msg.get("content"))
        items.append({
            "type": "message",
            "role": role or "user",
            "content": [{"type": "input_text", "text": text}],
        })

    return instructions, items


def _coerce_text(content: Any) -> str:
    """Flatten message ``content`` (which may be str or list of blocks) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            # OpenAI image blocks, etc. — only keep text-bearing blocks
            if block.get("type") in ("text", "input_text", "output_text"):
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


# ---------------------------------------------------------------------------
# Streaming translation
# ---------------------------------------------------------------------------


def translate_stream(
    responses_chunks: Iterable[Dict[str, Any]],
) -> Generator[Dict[str, Any], None, None]:
    """Translate a Responses-API SSE chunk stream into chat-completion chunks.

    The shape we emit is what ``agent_stream._call_llm_stream`` consumes:

        {"choices":[{"delta":{"content"|"tool_calls":...},"finish_reason":...}]}

    We emit:
      - ``response.output_text.delta``                -> delta.content
      - ``response.function_call_arguments.delta``    -> delta.tool_calls (one entry per output_index)
      - ``response.output_item.added`` (function_call) -> seeds tool_calls[i].id + .name
      - ``response.completed``                         -> finish_reason
      - any error chunk                                -> passed through verbatim
    """
    # Map output_index -> (call_id, name, accumulated_arguments)
    fn_state: Dict[int, Dict[str, str]] = {}

    for chunk in responses_chunks:
        # Pre-existing error chunks from openai_http_client._stream_chat:
        # they already carry {"error": True/dict, "status_code": ...} — pass through.
        if not isinstance(chunk, dict):
            continue
        if chunk.get("error"):
            yield chunk
            continue

        ev_type = chunk.get("type") or ""

        if ev_type == "response.output_text.delta":
            delta_text = chunk.get("delta", "")
            if delta_text:
                yield _make_text_chunk(delta_text)

        elif ev_type == "response.output_item.added":
            item = chunk.get("item") or {}
            if item.get("type") == "function_call":
                idx = chunk.get("output_index", 0)
                fn_state[idx] = {
                    "id": item.get("call_id", ""),
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments", "") or "",
                }
                # Emit a chunk with tool_call seed so the downstream aggregator
                # registers id+name before deltas start arriving.
                yield _make_tool_call_chunk(
                    index=idx,
                    call_id=fn_state[idx]["id"],
                    name=fn_state[idx]["name"],
                    arguments_delta=fn_state[idx]["arguments"],
                )

        elif ev_type == "response.function_call_arguments.delta":
            idx = chunk.get("output_index", 0)
            delta = chunk.get("delta", "")
            if not delta:
                continue
            state = fn_state.setdefault(idx, {"id": "", "name": "", "arguments": ""})
            state["arguments"] += delta
            yield _make_tool_call_chunk(
                index=idx,
                call_id=state.get("id", ""),
                name=state.get("name", ""),
                arguments_delta=delta,
            )

        elif ev_type == "response.completed":
            response_obj = chunk.get("response") or {}
            # Choose finish_reason: any function_call output -> "tool_calls",
            # else "stop". This mirrors chat-completion semantics.
            finish_reason = "stop"
            for item in response_obj.get("output") or []:
                if isinstance(item, dict) and item.get("type") == "function_call":
                    finish_reason = "tool_calls"
                    break
            usage = response_obj.get("usage") or {}
            yield {
                "choices": [{"delta": {}, "finish_reason": finish_reason}],
                "usage": _translate_usage(usage),
            }

        elif ev_type == "response.failed" or ev_type == "error":
            err = (chunk.get("response") or {}).get("error") or chunk.get("error") or {}
            msg = err.get("message") if isinstance(err, dict) else str(err)
            yield {
                "error": {
                    "message": msg or "Responses API error",
                    "code": (err.get("code") if isinstance(err, dict) else "") or "",
                    "type": (err.get("type") if isinstance(err, dict) else "") or "",
                },
                "message": msg or "Responses API error",
                "status_code": 500,
            }

        # Other event types (response.created, response.in_progress,
        # response.output_item.done, response.content_part.added/done, etc.)
        # carry no content for the chat-completion shape — drop silently.


def _make_text_chunk(text: str) -> Dict[str, Any]:
    return {"choices": [{"delta": {"content": text}, "finish_reason": None}]}


def _make_tool_call_chunk(
    *, index: int, call_id: str, name: str, arguments_delta: str,
) -> Dict[str, Any]:
    tc: Dict[str, Any] = {"index": index}
    if call_id:
        tc["id"] = call_id
    fn: Dict[str, str] = {}
    if name:
        fn["name"] = name
    if arguments_delta:
        fn["arguments"] = arguments_delta
    if fn:
        tc["function"] = fn
    return {"choices": [{"delta": {"tool_calls": [tc]}, "finish_reason": None}]}


def _translate_usage(usage: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get(
            "total_tokens",
            (usage.get("input_tokens", 0) or 0) + (usage.get("output_tokens", 0) or 0),
        ),
    }


# ---------------------------------------------------------------------------
# Sync translation
# ---------------------------------------------------------------------------


def translate_sync(response: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a non-streaming Responses dict into a chat-completion dict.

    Used by paths like title generation (``ChatGPTBot.reply_text``) that read
    ``response["choices"][0]["message"]["content"]`` and ``["usage"]``.
    """
    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []

    for item in response.get("output") or []:
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        if itype == "message":
            for block in item.get("content") or []:
                if isinstance(block, dict) and block.get("type") == "output_text":
                    text_parts.append(block.get("text", ""))
        elif itype == "function_call":
            tool_calls.append({
                "id": item.get("call_id", ""),
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments", "") or "",
                },
            })

    message: Dict[str, Any] = {
        "role": "assistant",
        "content": "".join(text_parts) if text_parts else "",
    }
    finish_reason = "stop"
    if tool_calls:
        message["tool_calls"] = tool_calls
        finish_reason = "tool_calls"

    usage = response.get("usage") or {}
    return {
        "id": response.get("id", ""),
        "object": "chat.completion",
        "created": response.get("created_at", 0),
        "model": response.get("model", ""),
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": _translate_usage(usage),
    }
