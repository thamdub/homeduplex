"""Text-to-speech through `POST /audio/speech` (raw PCM output) on any OpenAI-compatible server."""

import asyncio
from collections.abc import AsyncGenerator

from homeduplex.audio.pcm import SAMPLE_WIDTH
from homeduplex.backends.base import Audio, BackendError
from homeduplex.backends.openai.http import check_status, http_errors, make_client, open_stream
from homeduplex.config.schema import OpenAITTS


class OpenAITextToSpeech:
    def __init__(self, settings: OpenAITTS) -> None:
        self._settings = settings
        self._name = f"text-to-speech {settings.url}"
        self._http = make_client(settings.url, settings.api_key, settings.chunk_timeout_s)

    async def synthesize(self, text: str, voice: str | None) -> AsyncGenerator[Audio]:
        body = {"model": self._settings.model, "input": text, "response_format": "pcm"}
        if voice:
            body["voice"] = voice
        within = self._settings.chunk_timeout_s
        rate = self._settings.sample_rate
        with http_errors(self._name, within):
            async with open_stream(self._http, "POST", "audio/speech", json=body) as response:
                await check_status(response, self._name)
                chunks = aiter(response.aiter_bytes())
                odd = b""  # chunks can split a sample
                while True:
                    try:
                        data = await asyncio.wait_for(anext(chunks), within)
                    except StopAsyncIteration:
                        return
                    except TimeoutError:
                        raise BackendError(f"{self._name}: no audio within {within:g} s") from None
                    data, odd = odd + data, b""
                    if len(data) % SAMPLE_WIDTH:
                        data, odd = data[:-1], data[-1:]
                    if data:
                        yield Audio(data, rate)

    async def aclose(self) -> None:
        await self._http.aclose()
