"""Contract tests every speech-to-text adapter passes (backends/base.py), plus protocol details per adapter."""

import asyncio
import contextlib
import io
import socket
import time
import wave
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

import pytest
from support.audio import speech
from support.fakes.openai import FakeOpenAI, fake_openai
from support.fakes.wyoming import FakeWyoming, fake_wyoming
from support.wait import wait_for

from homeduplex.backends.base import Audio, BackendError, SpeechToText
from homeduplex.backends.openai.stt import OpenAISpeechToText
from homeduplex.backends.wyoming.stt import WyomingSpeechToText
from homeduplex.config.schema import OpenAISTT, WyomingSTT

TIMEOUT = 0.5
AUDIO = Audio(speech(1000), 24000)


@dataclass
class Case:
    stt: SpeechToText
    fake: FakeWyoming | FakeOpenAI
    # Server-side count of requests the client abandoned.
    abandoned: Callable[[], int]
    requests: Callable[[], int]


@contextlib.asynccontextmanager
async def wyoming_case() -> AsyncIterator[Case]:
    async with fake_wyoming(transcripts=["  turn on the light "]) as fake:
        stt = WyomingSpeechToText(WyomingSTT(type="wyoming", uri=fake.uri, language="en", timeout_s=TIMEOUT))
        yield Case(stt, fake, lambda: fake.disconnects, lambda: len(fake.stt_requests))


@contextlib.asynccontextmanager
async def openai_case() -> AsyncIterator[Case]:
    async with fake_openai(transcripts=["  turn on the light "]) as fake:
        settings = OpenAISTT(type="openai", url=fake.url, model="whisper-1", language="en", timeout_s=TIMEOUT)
        stt = OpenAISpeechToText(settings)
        try:
            yield Case(stt, fake, lambda: fake.cancelled, lambda: len(fake.transcriptions))
        finally:
            await stt.aclose()


@pytest.fixture(params=["wyoming", "openai"])
async def case(request: pytest.FixtureRequest) -> AsyncIterator[Case]:
    factory = wyoming_case if request.param == "wyoming" else openai_case
    async with factory() as c:
        yield c


async def test_transcribes(case: Case) -> None:
    assert await case.stt.transcribe(AUDIO) == "turn on the light"


async def test_times_out(case: Case) -> None:
    case.fake.behaviour = "hang"
    started = time.monotonic()
    with pytest.raises(BackendError, match="speech-to-text"):
        await case.stt.transcribe(AUDIO)
    assert time.monotonic() - started < TIMEOUT + 0.5


async def test_server_failure(case: Case) -> None:
    case.fake.behaviour = "close" if isinstance(case.fake, FakeWyoming) else "error"
    with pytest.raises(BackendError, match="speech-to-text"):
        await case.stt.transcribe(AUDIO)


async def test_cancelling_closes_the_request(case: Case) -> None:
    case.fake.behaviour = "hang"
    task = asyncio.create_task(case.stt.transcribe(AUDIO))
    await wait_for(lambda: case.requests() == 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await wait_for(lambda: case.abandoned() == 1)


@pytest.mark.parametrize("kind", ["wyoming", "openai"])
async def test_unreachable_server(kind: str) -> None:
    port = free_port()
    stt: SpeechToText
    if kind == "wyoming":
        stt = WyomingSpeechToText(WyomingSTT(type="wyoming", uri=f"tcp://127.0.0.1:{port}", timeout_s=TIMEOUT))
    else:
        stt = OpenAISpeechToText(OpenAISTT(type="openai", url=f"http://127.0.0.1:{port}", model="m", timeout_s=TIMEOUT))
    with pytest.raises(BackendError, match="speech-to-text"):
        await stt.transcribe(AUDIO)
    await stt.aclose()


async def test_wyoming_sends_resampled_audio_and_language() -> None:
    async with wyoming_case() as case:
        await case.stt.transcribe(AUDIO)
        assert isinstance(case.fake, FakeWyoming)
        [request] = case.fake.stt_requests
        assert request.rate == 16000
        assert request.language == "en"
        assert len(request.audio) == 16000 * 2


async def test_openai_sends_wav_model_language_and_key() -> None:
    async with fake_openai() as fake:
        settings = OpenAISTT.model_validate(
            {"type": "openai", "url": fake.url, "model": "whisper-1", "language": "fr", "api_key": "k"}
        )
        stt = OpenAISpeechToText(settings)
        await stt.transcribe(AUDIO)
        await stt.aclose()
    [request] = fake.transcriptions
    assert request["model"] == "whisper-1"
    assert request["language"] == "fr"
    with wave.open(io.BytesIO(request["wav"])) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()) == (24000, 1, 2, 24000)
    assert fake.authorization == ["Bearer k"]


async def test_openai_unexpected_answer() -> None:
    async with fake_openai(transcripts=[]) as fake:
        fake.transcripts = []
        stt = OpenAISpeechToText(OpenAISTT(type="openai", url=fake.url, model="m"))
        assert await stt.transcribe(AUDIO) == ""
        await stt.aclose()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
