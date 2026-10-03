"""Text-to-speech through any Wyoming TTS server (Piper, Kokoro, ...)."""

from collections.abc import AsyncGenerator

from wyoming.audio import AudioChunk, AudioStop
from wyoming.tts import Synthesize, SynthesizeVoice

from homeduplex.backends.base import Audio
from homeduplex.backends.wyoming.client import connect
from homeduplex.config.schema import WyomingTTS


class WyomingTextToSpeech:
    def __init__(self, settings: WyomingTTS) -> None:
        self._settings = settings
        self._name = f"text-to-speech {settings.uri}"

    async def synthesize(self, text: str, voice: str | None) -> AsyncGenerator[Audio]:
        within = self._settings.chunk_timeout_s
        async with connect(self._settings.uri, self._name, within) as conn:
            await conn.write(Synthesize(text=text, voice=SynthesizeVoice(name=voice) if voice else None).event())
            while True:
                # A server that stops mid-sentence must not hang the conversation: some TTS servers stall for
                # everyone when a client hangs up mid-sentence, so every read has a deadline.
                event = await conn.read(within)
                if AudioStop.is_type(event.type):
                    return
                if AudioChunk.is_type(event.type):
                    chunk = AudioChunk.from_event(event)
                    yield Audio(chunk.audio, chunk.rate)

    async def aclose(self) -> None:
        pass
