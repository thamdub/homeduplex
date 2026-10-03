"""The interfaces session code talks to. Adapters implement them for a protocol (Wyoming, the OpenAI HTTP API, ...).

Contract, checked by tests/backends for every adapter:
- Every call has a timeout from the settings; a backend that stops answering raises BackendError, never hangs.
- Failures raise BackendError with a message fit for a log line and for the client (no secrets).
- Streams are async generators; closing one early (`aclose()`, or cancelling the task consuming it) closes the
  connection to the backend, so the backend stops working on it. Consumers wrap them in `contextlib.aclosing`.
"""

from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from homeduplex.session.config import Tool


class BackendError(Exception):
    """A backend failed or timed out. `str()` is safe to show to the client."""


@dataclass(frozen=True)
class Audio:
    """PCM16 mono."""

    pcm: bytes
    rate: int


class SpeechToText(Protocol):
    async def transcribe(self, audio: Audio) -> str: ...

    async def aclose(self) -> None: ...


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: str  # JSON object, as text


@dataclass(frozen=True)
class ChatMessage:
    """Model-neutral chat history. Adapters translate it to their wire format."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    # For role "tool": which call this answers, and its function name (some servers want one, some the other).
    tool_call_id: str | None = None
    name: str | None = None


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ModelRequest:
    messages: Sequence[ChatMessage]
    tools: Sequence[Tool] = field(default=())


class LanguageModel(Protocol):
    def stream(self, request: ModelRequest) -> AsyncGenerator[TextDelta | ToolCall]:
        """Text as it is written; each tool call once it is complete."""
        ...

    async def aclose(self) -> None: ...


class TextToSpeech(Protocol):
    def synthesize(self, text: str, voice: str | None) -> AsyncGenerator[Audio]:
        """Audio for one sentence, chunk by chunk, at the engine's own rate."""
        ...

    async def aclose(self) -> None: ...
