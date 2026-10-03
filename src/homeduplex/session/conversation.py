"""The conversation as the client sees it: items in order, with ids the client refers to (delete, truncate)."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from homeduplex.transport.events import new_id


@dataclass(frozen=True)
class Segment:
    """A sentence of an assistant answer and where its audio sits in what was sent to the client."""

    start_ms: float
    end_ms: float
    text: str


@dataclass
class Message:
    id: str
    role: Literal["user", "assistant", "system"]
    text: str = ""
    # User speech whose transcript hasn't arrived yet.
    transcribing: bool = False
    # Assistant answers: the spoken sentences, for truncation.
    segments: list[Segment] = field(default_factory=list)
    # Typed text rather than speech (affects only the wire shape).
    typed: bool = False
    # User speech the model must not see, and why: "echo" (the assistant's own voice, text/echo.py) or "quiet"
    # (below vad.min_speech_level). The client still sees it.
    ignored: Literal["echo", "quiet"] | None = None


@dataclass
class FunctionCall:
    id: str
    call_id: str
    name: str
    arguments: str


@dataclass
class FunctionCallOutput:
    id: str
    call_id: str
    output: str


Item = Message | FunctionCall | FunctionCallOutput


class InvalidItem(ValueError):
    pass


class Conversation:
    def __init__(self) -> None:
        self.items: list[Item] = []
        # Last item the model no longer sees (see `window`); None: it sees everything.
        self._hidden_until: Item | None = None

    def add(self, item: Item) -> str | None:
        """Append; returns the id of the item before it (Realtime events carry it)."""
        previous = self.items[-1].id if self.items else None
        self.items.append(item)
        return previous

    def get(self, item_id: str) -> Item | None:
        return next((i for i in self.items if i.id == item_id), None)

    def delete(self, item_id: str) -> bool:
        index = next((n for n, i in enumerate(self.items) if i.id == item_id), None)
        if index is None:
            return False
        removed = self.items.pop(index)
        if removed is self._hidden_until:
            self._hidden_until = self.items[index - 1] if index > 0 else None
        return True

    def truncate(self, item_id: str, audio_end_ms: int) -> bool:
        """The client played only `audio_end_ms` of this answer: keep the sentences that had started by then."""
        item = self.get(item_id)
        if not isinstance(item, Message) or item.role != "assistant":
            return False
        kept = [s for s in item.segments if s.start_ms < audio_end_ms]
        item.segments = kept
        item.text = " ".join(s.text for s in kept)
        return True

    def window(self, max_turns: int) -> Sequence[Item]:
        """The items the model sees: at most `max_turns` user turns (replayed sessions pile up history).

        The start moves in steps, not turn by turn: past `max_turns` it jumps forward to keep half of them, then
        stays put until the limit is reached again. Model servers reuse their work for the part of a prompt that
        hasn't changed; dropping the oldest turn every time would change the prompt right after the system message
        and make them re-read the whole conversation on every turn.
        """
        start = self._start_index()
        users = [n for n in range(start, len(self.items)) if _is_user(self.items[n])]
        if len(users) > max_turns:
            start = users[-max(1, max_turns // 2)]
            self._hidden_until = self.items[start - 1]
        return self.items[start:]

    def _start_index(self) -> int:
        if self._hidden_until is None:
            return 0
        return next((n + 1 for n, i in enumerate(self.items) if i is self._hidden_until), 0)


def _is_user(item: Item) -> bool:
    return isinstance(item, Message) and item.role == "user"


def item_from_client(raw: Mapping[str, Any]) -> Item:
    """An item from `conversation.item.create`: tool results, typed messages and replayed history."""
    raw_id = raw.get("id")
    item_id = raw_id if isinstance(raw_id, str) else new_id("item")
    kind = raw.get("type", "message")
    if kind == "function_call_output":
        call_id = raw.get("call_id")
        if not isinstance(call_id, str):
            raise InvalidItem("function_call_output needs a call_id")
        output = raw.get("output")
        return FunctionCallOutput(item_id, call_id, output if isinstance(output, str) else _json(output))
    if kind == "function_call":
        call_id, name = raw.get("call_id"), raw.get("name")
        if not isinstance(call_id, str) or not isinstance(name, str):
            raise InvalidItem("function_call needs a call_id and a name")
        arguments = raw.get("arguments")
        return FunctionCall(item_id, call_id, name, arguments if isinstance(arguments, str) else _json(arguments))
    if kind == "message":
        role = raw.get("role", "user")
        if role not in ("user", "assistant", "system"):
            raise InvalidItem(f"unknown role {role!r}")
        return Message(item_id, role, flatten_content(raw.get("content")), typed=True)
    raise InvalidItem(f"unknown item type {kind!r}")


def flatten_content(content: Any) -> str:
    """Realtime content is a list of parts (`{type: input_text|text|output_text, text}` or
    `{type: input_audio|audio|output_audio, transcript}`); models want a string. Replayed history (Kiosk Satellite
    2026.10.3+) arrives this way: passing the list through made model servers answer HTTP 400."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    parts = content if isinstance(content, list) else [content]
    texts: list[str] = []
    for part in parts:
        if isinstance(part, str):
            texts.append(part)
        elif isinstance(part, Mapping):
            text = part.get("text") or part.get("transcript")
            if isinstance(text, str):
                texts.append(text)
    return " ".join(t.strip() for t in texts if t.strip())


def to_wire(item: Item) -> dict[str, Any]:
    base: dict[str, Any] = {"id": item.id, "object": "realtime.item", "status": "completed"}
    match item:
        case Message(role="user", typed=False):
            content = [{"type": "input_audio", "transcript": None if item.transcribing else item.text}]
        case Message(role="assistant"):
            content = [{"type": "output_audio", "transcript": item.text}]
        case Message():
            content = [{"type": "input_text", "text": item.text}]
        case FunctionCall():
            call = {"call_id": item.call_id, "name": item.name, "arguments": item.arguments}
            return {**base, "type": "function_call", **call}
        case FunctionCallOutput():
            return {**base, "type": "function_call_output", "call_id": item.call_id, "output": item.output}
    return {**base, "type": "message", "role": item.role, "content": content}


def _json(value: Any) -> str:
    return json.dumps(value if value is not None else {})
