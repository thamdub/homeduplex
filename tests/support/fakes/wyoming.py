"""Stand-in Wyoming servers on a real localhost socket, scripted by the test."""

import asyncio
import contextlib
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import async_read_event, async_write_event
from wyoming.tts import Synthesize

# answer: normally. hang: never answer. close: hang up without answering. stall (TTS): start, then stop mid-sentence.
Behaviour = Literal["answer", "hang", "close", "stall"]


@dataclass
class STTRequest:
    language: str | None = None
    rate: int | None = None
    audio: bytearray = field(default_factory=bytearray)


@dataclass
class FakeWyoming:
    """Speech-to-text: answers each request with the next of `transcripts`. Text-to-speech: answers with
    `tts_chunks` chunks of `tts_rate` audio per sentence. `behaviour` changes how it answers."""

    transcripts: list[str] = field(default_factory=lambda: ["hello there"])
    behaviour: Behaviour = "answer"
    delay: float = 0.0
    tts_rate: int = 22050
    tts_chunks: int = 3
    tts_chunk_ms: int = 100
    stt_requests: list[STTRequest] = field(default_factory=list)
    sentences: list[str] = field(default_factory=list)
    voices: list[str | None] = field(default_factory=list)
    connections: int = 0
    disconnects: int = 0
    port: int = 0

    @property
    def uri(self) -> str:
        return f"tcp://127.0.0.1:{self.port}"

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.connections += 1
        try:
            await self._serve(reader, writer)
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            self.disconnects += 1
            writer.close()

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        request = STTRequest()
        while (event := await async_read_event(reader)) is not None:
            if Transcribe.is_type(event.type):
                request.language = Transcribe.from_event(event).language
            elif AudioStart.is_type(event.type):
                request.rate = AudioStart.from_event(event).rate
            elif AudioChunk.is_type(event.type):
                request.audio += AudioChunk.from_event(event).audio
            elif AudioStop.is_type(event.type):
                self.stt_requests.append(request)
                if await self._misbehave(reader):
                    return
                text = self.transcripts.pop(0) if self.transcripts else ""
                await async_write_event(Transcript(text=text).event(), writer)
            elif Synthesize.is_type(event.type):
                synth = Synthesize.from_event(event)
                self.sentences.append(synth.text)
                self.voices.append(synth.voice.name if synth.voice else None)
                if await self._misbehave(reader):
                    return
                await self._speak(reader, writer)
                return

    async def _misbehave(self, reader: asyncio.StreamReader) -> bool:
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.behaviour == "hang":
            await reader.read()  # until the client goes away
            return True
        return self.behaviour == "close"

    async def _speak(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        rate = self.tts_rate
        await async_write_event(AudioStart(rate=rate, width=2, channels=1).event(), writer)
        samples = rate * self.tts_chunk_ms // 1000
        for n in range(self.tts_chunks):
            if n == 1 and self.behaviour == "stall":
                await reader.read()
                return
            # Distinct non-zero content per chunk, so tests can tell audio from silence.
            pcm = (1000 + n).to_bytes(2, "little", signed=True) * samples
            await async_write_event(AudioChunk(rate=rate, width=2, channels=1, audio=pcm).event(), writer)
        await async_write_event(AudioStop().event(), writer)


@contextlib.asynccontextmanager
async def fake_wyoming(**kwargs: object) -> AsyncIterator[FakeWyoming]:
    fake = FakeWyoming(**kwargs)  # type: ignore[arg-type]
    server = await asyncio.start_server(fake._handle, "127.0.0.1", 0)
    fake.port = server.sockets[0].getsockname()[1]
    try:
        yield fake
    finally:
        server.close()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(server.wait_closed(), 1)
