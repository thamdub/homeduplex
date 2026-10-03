"""Turns a conversation into a model request, laid out for prompt caching.

Model servers reuse the computation for the part of a prompt that matches the previous request; anything that
changes early makes them re-read everything after it. So:

- The system message is the configured preamble (the same for everyone), then the client's instructions (the same
  for a client), then the context text (the same for a room all day).
- Then the conversation, which only grows at the end, except when old turns are trimmed (Conversation.window).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from homeduplex.backends.base import ChatMessage, ModelRequest, ToolCall
from homeduplex.config.schema import PromptSettings
from homeduplex.prompt import template
from homeduplex.session.config import SessionConfig
from homeduplex.session.conversation import FunctionCall, FunctionCallOutput, Item, Message
from homeduplex.transport.room import Room


@dataclass(frozen=True)
class PromptBuilder:
    settings: PromptSettings
    room: Room

    def now(self) -> datetime:
        if self.settings.timezone:
            return datetime.now(ZoneInfo(self.settings.timezone))
        return datetime.now().astimezone()

    def system(self, config: SessionConfig, now: datetime) -> str:
        context = self._render(self.settings.context, now)
        parts = (self.settings.preamble, config.instructions, context)
        return "\n\n".join(p.strip() for p in parts if p.strip())

    def request(self, config: SessionConfig, items: Sequence[Item], now: datetime | None = None) -> ModelRequest:
        now = now or self.now()
        system = self.system(config, now)
        messages = [ChatMessage("system", system)] if system else []
        messages += history(items)
        return ModelRequest(messages, config.tools)

    def _render(self, text: str, now: datetime) -> str:
        return template.render(text, self.room.id, self.room.fields, now) if text else ""


def history(items: Sequence[Item]) -> list[ChatMessage]:
    """Conversation items as chat messages. Consecutive function calls become one assistant message (attached to
    the answer text before them, if any); a tool result whose call isn't in the window is dropped, since model
    servers reject it; and ignored user speech (echo, too quiet) is left out."""
    out: list[ChatMessage] = []
    names: dict[str, str] = {}
    for item in items:
        match item:
            case Message(role="user") if item.text and not item.ignored:
                out.append(ChatMessage("user", item.text))
            case Message(role="assistant") if item.text:
                out.append(ChatMessage("assistant", item.text))
            case Message(role="system") if item.text:
                out.append(ChatMessage("system", item.text))
            case FunctionCall():
                names[item.call_id] = item.name
                call = ToolCall(item.call_id, item.name, item.arguments)
                last = out[-1] if out else None
                if last is not None and last.role == "assistant":
                    out[-1] = ChatMessage("assistant", last.content, (*last.tool_calls, call))
                else:
                    out.append(ChatMessage("assistant", "", (call,)))
            case FunctionCallOutput() if item.call_id in names:
                out.append(ChatMessage("tool", item.output, tool_call_id=item.call_id, name=names[item.call_id]))
    return out
