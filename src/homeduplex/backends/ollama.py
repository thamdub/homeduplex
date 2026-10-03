"""Ollama's native `/api/chat`: a dialect of the chat adapter, kept because Ollama's OpenAI-compatible endpoint
ignores `options` (context size), `keep_alive` and `think`, and a request whose options differ from the other users
of the same Ollama server makes it reload the model (docs/design.md)."""

import contextlib
import json
from collections.abc import AsyncGenerator, Sequence
from typing import Any

from homeduplex.backends.base import BackendError, ChatMessage, ModelRequest, TextDelta, ToolCall
from homeduplex.backends.openai.chat import timed_lines, tools_wire
from homeduplex.backends.openai.http import check_status, http_errors, make_client, open_stream
from homeduplex.config.schema import OllamaChat
from homeduplex.transport.events import new_id


class OllamaChatModel:
    def __init__(self, settings: OllamaChat) -> None:
        self._settings = settings
        self._name = f"language model {settings.url}"
        self._timeout = max(settings.first_token_timeout_s, settings.idle_timeout_s)
        self._http = make_client(settings.url, None, self._timeout)

    async def stream(self, request: ModelRequest) -> AsyncGenerator[TextDelta | ToolCall]:
        body: dict[str, Any] = {
            "model": self._settings.model,
            "messages": _messages(request.messages),
            "stream": True,
        }
        if request.tools:
            body["tools"] = tools_wire(request.tools)
        if self._settings.options:
            body["options"] = self._settings.options
        if self._settings.keep_alive is not None:
            body["keep_alive"] = self._settings.keep_alive
        if self._settings.think is not None:
            body["think"] = self._settings.think
        with http_errors(self._name, self._timeout):
            async with open_stream(self._http, "POST", "api/chat", json=body) as response:
                await check_status(response, self._name)
                settings = self._settings
                lines = timed_lines(
                    response.aiter_lines(), settings.first_token_timeout_s, settings.idle_timeout_s, self._name
                )
                async with contextlib.aclosing(lines):
                    async for line in lines:
                        if not line.strip():
                            continue
                        try:
                            chunk = json.loads(line)
                        except ValueError:
                            raise BackendError(f"{self._name}: unreadable answer {line[:200]!r}") from None
                        if chunk.get("error"):
                            raise BackendError(f"{self._name}: {chunk['error']}")
                        message = chunk.get("message") or {}
                        if text := message.get("content"):
                            yield TextDelta(text)
                        for call in message.get("tool_calls") or []:
                            function = call.get("function") or {}
                            if name := function.get("name"):
                                arguments = function.get("arguments")
                                text_args = arguments if isinstance(arguments, str) else json.dumps(arguments or {})
                                yield ToolCall(call.get("id") or new_id("call"), name, text_args)
                        if chunk.get("done"):
                            break

    async def aclose(self) -> None:
        await self._http.aclose()


def _messages(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool":
            out.append({"role": "tool", "content": m.content, "tool_name": m.name or ""})
            continue
        message: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.tool_calls:
            message["tool_calls"] = [
                {"function": {"name": c.name, "arguments": _arguments(c.arguments)}} for c in m.tool_calls
            ]
        out.append(message)
    return out


def _arguments(text: str) -> Any:
    try:
        value = json.loads(text or "{}")
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}
