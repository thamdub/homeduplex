"""Chat through `POST /chat/completions` (streaming, tools) on any OpenAI-compatible server: llama.cpp, vLLM, LM
Studio, LocalAI, Ollama's /v1, hosted APIs."""

import asyncio
import contextlib
import json
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from typing import Any

from homeduplex.backends.base import BackendError, ChatMessage, ModelRequest, TextDelta, ToolCall
from homeduplex.backends.openai.http import check_status, http_errors, make_client, open_stream
from homeduplex.config.schema import OpenAIChat
from homeduplex.session.config import Tool
from homeduplex.transport.events import new_id


class OpenAIChatModel:
    def __init__(self, settings: OpenAIChat) -> None:
        self._settings = settings
        self._name = f"language model {settings.url}"
        self._timeout = max(settings.first_token_timeout_s, settings.idle_timeout_s)
        self._http = make_client(settings.url, settings.api_key, self._timeout)

    async def stream(self, request: ModelRequest) -> AsyncGenerator[TextDelta | ToolCall]:
        body: dict[str, Any] = {
            "model": self._settings.model,
            "messages": [_message(m) for m in request.messages],
            "stream": True,
            **self._settings.extra_body,
        }
        if request.tools:
            body["tools"] = tools_wire(request.tools)
        calls: dict[int, dict[str, str]] = {}
        with http_errors(self._name, self._timeout):
            async with open_stream(self._http, "POST", "chat/completions", json=body) as response:
                await check_status(response, self._name)
                settings = self._settings
                lines = timed_lines(
                    response.aiter_lines(), settings.first_token_timeout_s, settings.idle_timeout_s, self._name
                )
                async with contextlib.aclosing(lines):
                    async for line in lines:
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        chunk = _json(data, self._name)
                        if "error" in chunk:
                            raise BackendError(f"{self._name}: {_error_text(chunk['error'])}")
                        for choice in chunk.get("choices") or []:
                            delta = choice.get("delta") or {}
                            if text := delta.get("content"):
                                yield TextDelta(text)
                            for call in delta.get("tool_calls") or []:
                                _accumulate(calls, call)
        for call in calls.values():
            if call["name"]:
                yield ToolCall(call["id"] or new_id("call"), call["name"], call["arguments"] or "{}")

    async def aclose(self) -> None:
        await self._http.aclose()


def tools_wire(tools: Sequence[Tool]) -> list[dict[str, Any]]:
    """Function tools in the chat-completions shape (Ollama's native API uses the same)."""
    return [
        {
            "type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": dict(t.parameters)},
        }
        for t in tools
    ]


async def timed_lines(lines: AsyncIterator[str], first: float, idle: float, name: str) -> AsyncGenerator[str]:
    """Lines of a streaming answer, with a deadline for the first one (the server reads the prompt first) and
    another between the next ones."""
    within = first
    iterator = aiter(lines)
    while True:
        try:
            line = await asyncio.wait_for(anext(iterator), within)
        except StopAsyncIteration:
            return
        except TimeoutError:
            raise BackendError(f"{name}: no answer within {within:g} s") from None
        yield line
        within = idle


def _message(message: ChatMessage) -> dict[str, Any]:
    if message.role == "tool":
        return {"role": "tool", "tool_call_id": message.tool_call_id, "content": message.content}
    out: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        out["content"] = message.content or None
        out["tool_calls"] = [
            {"id": c.call_id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
            for c in message.tool_calls
        ]
    return out


def _accumulate(calls: dict[int, dict[str, str]], delta: dict[str, Any]) -> None:
    """Tool calls stream in pieces keyed by index: the id and name first, the arguments in fragments."""
    index = delta.get("index", len(calls))
    call = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
    if delta.get("id"):
        call["id"] = delta["id"]
    function = delta.get("function") or {}
    if function.get("name"):
        call["name"] = function["name"]
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        call["arguments"] += arguments
    elif isinstance(arguments, dict):
        call["arguments"] = json.dumps(arguments)


def _json(data: str, name: str) -> dict[str, Any]:
    try:
        value = json.loads(data)
    except ValueError:
        raise BackendError(f"{name}: unreadable answer {data[:200]!r}") from None
    if not isinstance(value, dict):
        raise BackendError(f"{name}: unexpected answer {data[:200]!r}")
    return value


def _error_text(error: Any) -> str:
    if isinstance(error, dict):
        return str(error.get("message") or error)
    return str(error)
