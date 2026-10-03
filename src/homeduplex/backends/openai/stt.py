"""Speech-to-text through `POST /audio/transcriptions` on any OpenAI-compatible server."""

import asyncio
import io
import wave

from homeduplex.audio.pcm import SAMPLE_WIDTH
from homeduplex.backends.base import Audio, BackendError
from homeduplex.backends.openai.http import check_status, http_errors, make_client, open_stream
from homeduplex.config.schema import OpenAISTT


class OpenAISpeechToText:
    def __init__(self, settings: OpenAISTT) -> None:
        self._settings = settings
        self._name = f"speech-to-text {settings.url}"
        self._http = make_client(settings.url, settings.api_key, settings.timeout_s)

    async def transcribe(self, audio: Audio) -> str:
        data = {"model": self._settings.model, "response_format": "json"}
        if self._settings.language:
            data["language"] = self._settings.language
        files = {"file": ("speech.wav", to_wav(audio), "audio/wav")}
        timeout = self._settings.timeout_s
        try:
            # httpx bounds each read; this bounds the whole request.
            async with asyncio.timeout(timeout):
                with http_errors(self._name, timeout):
                    async with open_stream(
                        self._http, "POST", "audio/transcriptions", data=data, files=files
                    ) as response:
                        await check_status(response, self._name)
                        await response.aread()
        except TimeoutError:
            raise BackendError(f"{self._name}: no answer within {timeout:g} s") from None
        try:
            text = response.json()["text"]
        except (ValueError, KeyError, TypeError):
            raise BackendError(f"{self._name}: unexpected answer {response.text[:200]!r}") from None
        return str(text).strip()

    async def aclose(self) -> None:
        await self._http.aclose()


def to_wav(audio: Audio) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(SAMPLE_WIDTH)
        w.setframerate(audio.rate)
        w.writeframes(audio.pcm)
    return buf.getvalue()
