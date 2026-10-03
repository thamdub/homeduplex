"""Backends seen through a fair queue: what a session uses, so every call waits its turn for the shared backend.

Waiting for a slot doesn't count against the backend's own timeouts: those measure the backend, not the queue.
"""

import contextlib
from collections.abc import AsyncGenerator

from homeduplex.backends.base import Audio, LanguageModel, ModelRequest, SpeechToText, TextDelta, TextToSpeech, ToolCall
from homeduplex.scheduler.fair import FairQueue


class ScheduledSpeechToText:
    def __init__(self, inner: SpeechToText, queue: FairQueue, owner: str) -> None:
        self._inner, self._queue, self._owner = inner, queue, owner

    async def transcribe(self, audio: Audio) -> str:
        async with self._queue.slot(self._owner):
            return await self._inner.transcribe(audio)

    async def aclose(self) -> None:
        pass  # the shared backend is closed by its owner


class ScheduledLanguageModel:
    def __init__(self, inner: LanguageModel, queue: FairQueue, owner: str) -> None:
        self._inner, self._queue, self._owner = inner, queue, owner

    async def stream(self, request: ModelRequest) -> AsyncGenerator[TextDelta | ToolCall]:
        async with self._queue.slot(self._owner), contextlib.aclosing(self._inner.stream(request)) as stream:
            async for piece in stream:
                yield piece

    async def aclose(self) -> None:
        pass


class ScheduledTextToSpeech:
    """One slot per sentence: rooms' answers interleave sentence by sentence."""

    def __init__(self, inner: TextToSpeech, queue: FairQueue, owner: str) -> None:
        self._inner, self._queue, self._owner = inner, queue, owner

    async def synthesize(self, text: str, voice: str | None) -> AsyncGenerator[Audio]:
        async with self._queue.slot(self._owner), contextlib.aclosing(self._inner.synthesize(text, voice)) as stream:
            async for chunk in stream:
                yield chunk

    async def aclose(self) -> None:
        pass
