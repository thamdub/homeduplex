"""Contract tests every text-to-speech adapter passes (backends/base.py), plus protocol details per adapter."""

import contextlib
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest
from support.fakes.openai import FakeOpenAI, fake_openai
from support.fakes.wyoming import FakeWyoming, fake_wyoming
from support.wait import wait_for

from homeduplex.backends import text_to_speech
from homeduplex.backends.base import Audio, BackendError, TextToSpeech
from homeduplex.config.schema import OpenAITTS, WyomingTTS

TIMEOUT = 0.5


@dataclass
class Case:
    tts: TextToSpeech
    fake: FakeWyoming | FakeOpenAI
    rate: int
    sentences: Any
    abandoned: Any


@contextlib.asynccontextmanager
async def wyoming_case() -> AsyncIterator[Case]:
    async with fake_wyoming(tts_rate=22050) as fake:
        tts = text_to_speech(WyomingTTS(type="wyoming", uri=fake.uri, chunk_timeout_s=TIMEOUT))
        yield Case(tts, fake, 22050, lambda: fake.sentences, lambda: fake.disconnects)


@contextlib.asynccontextmanager
async def openai_case() -> AsyncIterator[Case]:
    async with fake_openai() as fake:
        tts = text_to_speech(OpenAITTS(type="openai", url=fake.url, model="tts-1", chunk_timeout_s=TIMEOUT))
        try:
            yield Case(tts, fake, 24000, lambda: [s["input"] for s in fake.speeches], lambda: fake.cancelled)
        finally:
            await tts.aclose()


@pytest.fixture(params=["wyoming", "openai"])
async def case(request: pytest.FixtureRequest) -> AsyncIterator[Case]:
    async with wyoming_case() if request.param == "wyoming" else openai_case() as c:
        yield c


async def collect(tts: TextToSpeech, text: str = "Hello there.", voice: str | None = None) -> list[Audio]:
    async with contextlib.aclosing(tts.synthesize(text, voice)) as stream:
        return [chunk async for chunk in stream]


async def test_synthesizes_in_chunks_at_the_engine_rate(case: Case) -> None:
    chunks = await collect(case.tts)
    assert len(chunks) == 3
    assert {c.rate for c in chunks} == {case.rate}
    assert sum(len(c.pcm) for c in chunks) == 3 * case.rate // 10 * 2  # 3 chunks of 100 ms
    assert all(len(c.pcm) % 2 == 0 for c in chunks)
    assert case.sentences() == ["Hello there."]


async def test_silent_server_times_out(case: Case) -> None:
    case.fake.behaviour = "hang"
    started = time.monotonic()
    with pytest.raises(BackendError, match="text-to-speech"):
        await collect(case.tts)
    assert time.monotonic() - started < TIMEOUT + 0.5


async def test_server_stopping_mid_sentence_times_out(case: Case) -> None:
    """The prototype's worst bug: a TTS server stuck mid-sentence hung the conversation."""
    case.fake.behaviour = "stall"
    received: list[Audio] = []
    with pytest.raises(BackendError, match="text-to-speech"):
        async with contextlib.aclosing(case.tts.synthesize("Hello.", None)) as stream:
            async for chunk in stream:
                received.append(chunk)
    assert len(received) == 1


async def test_server_failure(case: Case) -> None:
    case.fake.behaviour = "close" if isinstance(case.fake, FakeWyoming) else "error"
    with pytest.raises(BackendError, match="text-to-speech"):
        await collect(case.tts)


async def test_closing_early_abandons_the_request(case: Case) -> None:
    case.fake.behaviour = "stall"
    stream = case.tts.synthesize("Hello.", None)
    await anext(stream)
    await stream.aclose()
    await wait_for(lambda: case.abandoned() == 1)


async def test_voice_is_passed(case: Case) -> None:
    await collect(case.tts, voice="af_heart")
    if isinstance(case.fake, FakeWyoming):
        assert case.fake.voices == ["af_heart"]
    else:
        assert case.fake.speeches[0]["voice"] == "af_heart"
        assert case.fake.speeches[0]["response_format"] == "pcm"


async def test_no_voice_means_engine_default(case: Case) -> None:
    await collect(case.tts)
    if isinstance(case.fake, FakeWyoming):
        assert case.fake.voices == [None]
    else:
        assert "voice" not in case.fake.speeches[0]


@pytest.mark.parametrize("kind", ["wyoming", "openai"])
async def test_unreachable(kind: str) -> None:
    tts = text_to_speech(
        WyomingTTS(type="wyoming", uri="tcp://127.0.0.1:9", chunk_timeout_s=TIMEOUT)
        if kind == "wyoming"
        else OpenAITTS(type="openai", url="http://127.0.0.1:9", model="m", chunk_timeout_s=TIMEOUT)
    )
    with pytest.raises(BackendError, match="text-to-speech"):
        await collect(tts)
    await tts.aclose()
