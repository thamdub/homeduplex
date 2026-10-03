"""Speech-to-text through any Wyoming ASR server (faster-whisper, Parakeet, Vosk, ...)."""

import asyncio
import time

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop

from homeduplex.audio.pcm import SAMPLE_WIDTH
from homeduplex.audio.resample import resample
from homeduplex.backends.base import Audio, BackendError
from homeduplex.backends.wyoming.client import connect
from homeduplex.config.schema import WyomingSTT

CHUNK_MS = 100


class WyomingSpeechToText:
    def __init__(self, settings: WyomingSTT) -> None:
        self._settings = settings
        self._name = f"speech-to-text {settings.uri}"

    async def transcribe(self, audio: Audio) -> str:
        rate = self._settings.sample_rate
        pcm = await asyncio.to_thread(resample, audio.pcm, audio.rate, rate)
        deadline = time.monotonic() + self._settings.timeout_s
        async with connect(self._settings.uri, self._name, self._settings.timeout_s) as conn:
            await conn.write(Transcribe(language=self._settings.language).event())
            await conn.write(AudioStart(rate=rate, width=SAMPLE_WIDTH, channels=1).event())
            step = rate * CHUNK_MS // 1000 * SAMPLE_WIDTH
            for i in range(0, len(pcm), step):
                chunk = AudioChunk(rate=rate, width=SAMPLE_WIDTH, channels=1, audio=pcm[i : i + step])
                await conn.write(chunk.event())
            await conn.write(AudioStop().event())
            while True:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise BackendError(f"{self._name}: no transcript within {self._settings.timeout_s:g} s")
                event = await conn.read(left)
                if Transcript.is_type(event.type):
                    return Transcript.from_event(event).text.strip()

    async def aclose(self) -> None:
        pass
