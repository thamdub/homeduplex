"""Check that the configured backends answer, without using them: run at start-up (results go to the log) and by
`homeduplex check-config --connect`. Validating a session with a client never touches the backends, so a wrong
address otherwise only shows when the first conversation fails."""

import asyncio
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx
from pydantic import SecretStr
from wyoming.info import Describe, Info

from homeduplex.backends.base import BackendError
from homeduplex.backends.wyoming.client import connect
from homeduplex.config import Settings
from homeduplex.config.schema import OllamaChat, OpenAIChat, OpenAISTT, OpenAITTS, WyomingSTT, WyomingTTS

TIMEOUT_S = 3.0
_LOOPBACK = {"127.0.0.1", "localhost", "::1"}


@dataclass(frozen=True)
class ProbeResult:
    role: str
    where: str
    ok: bool
    detail: str

    def line(self) -> str:
        return f"{self.role} {self.where}: {'ok' if self.ok else 'PROBLEM'}, {self.detail}"


async def probe_all(settings: Settings, within: float = TIMEOUT_S) -> list[ProbeResult]:
    return list(
        await asyncio.gather(
            _probe_stt(settings.stt, within), _probe_tts(settings.tts, within), _probe_llm(settings.llm, within)
        )
    )


async def _probe_stt(settings: WyomingSTT | OpenAISTT, within: float) -> ProbeResult:
    if isinstance(settings, WyomingSTT):
        return await _wyoming("speech-to-text", settings.uri, within, want="asr")
    return await _http("speech-to-text", settings.url, "models", within, settings.api_key)


async def _probe_tts(settings: WyomingTTS | OpenAITTS, within: float) -> ProbeResult:
    if isinstance(settings, WyomingTTS):
        return await _wyoming("text-to-speech", settings.uri, within, want="tts", voice=settings.voice)
    return await _http("text-to-speech", settings.url, "models", within, settings.api_key)


async def _probe_llm(settings: OpenAIChat | OllamaChat, within: float) -> ProbeResult:
    if isinstance(settings, OllamaChat):
        return await _ollama(settings, within)
    return await _http("language model", settings.url, "models", within, settings.api_key)


async def _wyoming(role: str, uri: str, within: float, want: str, voice: str | None = None) -> ProbeResult:
    try:
        async with connect(uri, role, within) as conn:
            await conn.write(Describe().event())
            while not Info.is_type((event := await conn.read(within)).type):
                pass
    except BackendError as e:
        return _failed(role, uri, str(e).removeprefix(f"{role}: "))
    info = Info.from_event(event)
    programs = info.asr if want == "asr" else info.tts
    if not programs:
        return ProbeResult(role, uri, False, f"answers, but offers no {'transcription' if want == 'asr' else 'voices'}")
    names = ", ".join(p.name for p in programs)
    if voice is not None and want == "tts":
        voices = {v.name for p in info.tts for v in p.voices}
        if voices and voice not in voices:
            return ProbeResult(role, uri, False, f"{names} answers, but has no voice {voice!r}")
    return ProbeResult(role, uri, True, f"{names} answers")


async def _http(role: str, url: str, path: str, within: float, api_key: SecretStr | None) -> ProbeResult:
    headers = {"Authorization": f"Bearer {api_key.get_secret_value()}"} if api_key else {}
    try:
        async with httpx.AsyncClient(timeout=within, headers=headers) as http:
            response = await http.get(f"{url}/{path}")
    except httpx.HTTPError as e:
        return _failed(role, url, _http_reason(e))
    if response.status_code in (401, 403):
        return ProbeResult(role, url, False, f"answers, but refused the API key (HTTP {response.status_code})")
    if response.status_code >= 500:
        return ProbeResult(role, url, False, f"answers with an error (HTTP {response.status_code})")
    return ProbeResult(role, url, True, "answers")


async def _ollama(settings: OllamaChat, within: float) -> ProbeResult:
    role = "language model"
    try:
        async with httpx.AsyncClient(timeout=within) as http:
            response = await http.get(f"{settings.url}/api/tags")
            response.raise_for_status()
            models = {m.get("name", "") for m in response.json().get("models", [])}
    except httpx.HTTPError as e:
        return _failed(role, settings.url, _http_reason(e))
    except (ValueError, AttributeError):
        return ProbeResult(role, settings.url, False, "answers, but not like Ollama (try type: openai)")
    model = settings.model
    if model not in models and f"{model}:latest" not in models:
        return ProbeResult(role, settings.url, False, f"Ollama answers, but has no model {model!r} (ollama pull?)")
    return ProbeResult(role, settings.url, True, f"Ollama answers, model {model} installed")


def _http_reason(error: httpx.HTTPError) -> str:
    if isinstance(error, httpx.TimeoutException):
        return "no answer"
    if isinstance(error, httpx.HTTPStatusError):
        return f"HTTP {error.response.status_code}"
    return f"cannot connect ({type(error).__name__})"


def _failed(role: str, where: str, reason: str) -> ProbeResult:
    host = urlsplit(where).hostname or ""
    if host in _LOOPBACK:
        reason += f"; inside a container {host} is the container itself: use the host's address"
    return ProbeResult(role, where, False, reason)
