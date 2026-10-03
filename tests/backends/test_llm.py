"""Contract tests every language model adapter passes (backends/base.py), plus each dialect's wire format."""

import asyncio
import contextlib
import gc
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest
from support.fakes.openai import FakeOpenAI, fake_openai
from support.wait import wait_for

from homeduplex.backends import language_model
from homeduplex.backends.base import BackendError, ChatMessage, LanguageModel, ModelRequest, TextDelta, ToolCall
from homeduplex.config.schema import LLMSettings, OllamaChat, OpenAIChat
from homeduplex.session.config import Tool

TIMEOUT = 0.5
TOOL = Tool("HassTurnOn", "Turn something on", {"type": "object", "properties": {"name": {"type": "string"}}})
REQUEST = ModelRequest([ChatMessage("system", "Be brief."), ChatMessage("user", "Hi")], [TOOL])


@dataclass
class Case:
    llm: LanguageModel
    fake: FakeOpenAI
    kind: str


def settings_for(kind: str, fake: FakeOpenAI, **extra: Any) -> LLMSettings:
    common = {"model": "m", "first_token_timeout_s": TIMEOUT, "idle_timeout_s": TIMEOUT, **extra}
    if kind == "openai":
        return OpenAIChat.model_validate({"type": "openai", "url": fake.url, **common})
    return OllamaChat.model_validate({"type": "ollama", "url": fake.ollama_url, **common})


@contextlib.asynccontextmanager
async def make_case(kind: str, **extra: Any) -> AsyncIterator[Case]:
    async with fake_openai() as fake:
        llm = language_model(settings_for(kind, fake, **extra))
        try:
            yield Case(llm, fake, kind)
        finally:
            await llm.aclose()


@pytest.fixture(params=["openai", "ollama"])
async def case(request: pytest.FixtureRequest) -> AsyncIterator[Case]:
    async with make_case(request.param) as c:
        yield c


async def collect(llm: LanguageModel, request: ModelRequest = REQUEST) -> list[TextDelta | ToolCall]:
    async with contextlib.aclosing(llm.stream(request)) as stream:
        return [piece async for piece in stream]


async def test_streams_text(case: Case) -> None:
    pieces = await collect(case.llm)
    assert pieces == [TextDelta("Hello"), TextDelta(" there.")]


async def test_tool_calls(case: Case) -> None:
    case.fake.replies = [["Checking.", {"tool": "HassTurnOn", "arguments": {"name": "Office Light"}}]]
    text, call = await collect(case.llm)
    assert text == TextDelta("Checking.")
    assert isinstance(call, ToolCall)
    assert call.name == "HassTurnOn"
    assert json.loads(call.arguments) == {"name": "Office Light"}
    assert call.call_id


async def test_several_tool_calls(case: Case) -> None:
    case.fake.replies = [[{"tool": "A", "arguments": {}}, {"tool": "B", "arguments": {"x": 1}}]]
    calls = await collect(case.llm)
    assert [c.name for c in calls if isinstance(c, ToolCall)] == ["A", "B"]
    assert len({c.call_id for c in calls if isinstance(c, ToolCall)}) == 2


async def test_first_token_timeout(case: Case) -> None:
    case.fake.behaviour = "hang"
    started = time.monotonic()
    with pytest.raises(BackendError, match="no answer within"):
        await collect(case.llm)
    assert time.monotonic() - started < TIMEOUT + 0.5


async def test_stall_mid_answer(case: Case) -> None:
    case.fake.behaviour = "stall"
    with pytest.raises(BackendError, match="no answer within"):
        await collect(case.llm)


async def test_http_error(case: Case) -> None:
    case.fake.behaviour, case.fake.error_status = "error", 400
    with pytest.raises(BackendError, match=r"HTTP 400: .*scripted failure"):
        await collect(case.llm)


async def test_error_inside_stream(case: Case) -> None:
    case.fake.behaviour = "stream_error"
    with pytest.raises(BackendError, match="scripted stream failure"):
        await collect(case.llm)


async def test_closing_the_stream_abandons_the_request(case: Case) -> None:
    case.fake.default_reply = ["One.", " Two.", " Three.", " Four."]
    case.fake.piece_delay = 0.2
    stream = case.llm.stream(REQUEST)
    assert await anext(stream) == TextDelta("One.")
    await stream.aclose()
    await wait_for(lambda: case.fake.cancelled == 1)


async def test_unreachable(case: Case) -> None:
    settings = settings_for(case.kind, case.fake)
    llm = language_model(settings.model_copy(update={"url": "http://127.0.0.1:9"}))
    with pytest.raises(BackendError, match="language model"):
        await collect(llm)
    await llm.aclose()


HISTORY = ModelRequest(
    [
        ChatMessage("system", "S"),
        ChatMessage("user", "Is the light on?"),
        ChatMessage("assistant", "", (ToolCall("call_1", "GetLiveContext", '{"area": "Office"}'),)),
        ChatMessage("tool", "on", tool_call_id="call_1", name="GetLiveContext"),
        ChatMessage("assistant", "It is on."),
    ],
    [TOOL],
)


async def test_openai_wire_format() -> None:
    async with make_case("openai", extra_body={"temperature": 0.2}, api_key="k") as case:
        await collect(case.llm, HISTORY)
        [body] = case.fake.chats
    assert body["model"] == "m" and body["stream"] is True and body["temperature"] == 0.2
    assert body["messages"] == [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "Is the light on?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "GetLiveContext", "arguments": '{"area": "Office"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "on"},
        {"role": "assistant", "content": "It is on."},
    ]
    function = {"name": "HassTurnOn", "description": "Turn something on", "parameters": dict(TOOL.parameters)}
    assert body["tools"] == [{"type": "function", "function": function}]
    assert case.fake.authorization == ["Bearer k"]


async def test_ollama_wire_format() -> None:
    async with make_case("ollama", options={"num_ctx": 16384}, keep_alive=-1, think=False) as case:
        await collect(case.llm, HISTORY)
        [body] = case.fake.chats
    assert body["options"] == {"num_ctx": 16384}
    assert body["keep_alive"] == -1
    assert body["think"] is False
    assert body["messages"][2] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "GetLiveContext", "arguments": {"area": "Office"}}}],
    }
    assert body["messages"][3] == {"role": "tool", "content": "on", "tool_name": "GetLiveContext"}
    assert body["tools"][0]["function"]["name"] == "HassTurnOn"


async def test_ollama_omits_unset_options() -> None:
    async with make_case("ollama") as case:
        await collect(case.llm, ModelRequest([ChatMessage("user", "x")]))
        [body] = case.fake.chats
    assert "options" not in body and "keep_alive" not in body and "think" not in body and "tools" not in body


async def test_cancelling_at_any_moment_leaks_nothing(case: Case) -> None:
    """httpcore loses the socket if cancelled while connecting; open_stream works around it. Leaks show up as
    ResourceWarnings at garbage collection, which the test suite turns into errors."""
    case.fake.piece_delay = 0.2
    for i in range(100):
        llm = language_model(settings_for(case.kind, case.fake))
        task = asyncio.create_task(collect(llm))
        await asyncio.sleep((i % 25) * 0.0001)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await llm.aclose()
        gc.collect()
