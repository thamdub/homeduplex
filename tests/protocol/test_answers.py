"""Answers: model → sentences → speech → client, with tools, cancellation, truncation and failures."""

import asyncio
import base64
import contextlib
import itertools
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest
from support.audio import chunks, silence, speech
from support.client import KIOSK_SESSION, RealtimeClient, open_client
from support.fakes.openai import FakeOpenAI, Reply, fake_openai
from support.fakes.wyoming import FakeWyoming, fake_wyoming
from support.server import running_server
from support.settings import make_settings
from support.wait import wait_for

from homeduplex.audio.pcm import encode_b64


@dataclass
class Stack:
    client: RealtimeClient
    stt: FakeWyoming
    tts: FakeWyoming
    llm: FakeOpenAI

    def last_messages(self) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = self.llm.chats[-1]["messages"]
        return messages


@contextlib.asynccontextmanager
async def stack(
    query: str = "model=gpt-realtime",
    session: dict[str, Any] | None = None,
    llm_kind: str = "openai",
    **sections: Any,
) -> AsyncIterator[Stack]:
    """A validated Kiosk-like client on a real server, with every backend faked on its own socket."""
    async with fake_wyoming() as stt, fake_wyoming(tts_rate=22050) as tts, fake_openai() as llm:
        llm_settings = (
            {"type": "openai", "url": llm.url, "model": "m"}
            if llm_kind == "openai"
            else {"type": "ollama", "url": llm.ollama_url, "model": "m"}
        )
        settings = make_settings(
            stt={"type": "wyoming", "uri": stt.uri, "timeout_s": 2},
            tts={"type": "wyoming", "uri": tts.uri, "chunk_timeout_s": 0.5, **sections.pop("tts", {})},
            llm={**llm_settings, "first_token_timeout_s": 2, "idle_timeout_s": 2},
            **sections,
        )
        async with running_server(settings) as srv, open_client(srv.url(query)) as client:
            await client.expect("session.created")
            await client.update_session(session or KIOSK_SESSION)
            yield Stack(client, stt, tts, llm)


async def say(
    client: RealtimeClient, stt: FakeWyoming, text: str, wait_for_transcript: bool = True, amplitude: int = 6000
) -> str:
    """Speak an utterance the fake speech-to-text will hear as `text`; returns its item id once transcribed (or
    once speech stopped)."""
    stt.transcripts = [text]
    for chunk in chunks(silence(300) + speech(800, amplitude=amplitude) + silence(900)):
        await client.send("input_audio_buffer.append", audio=encode_b64(chunk))
    stopped = await client.expect("input_audio_buffer.speech_stopped", skip={"input_audio_buffer.speech_started"})
    if wait_for_transcript:
        await client.expect(
            "conversation.item.done",
            skip={
                "input_audio_buffer.committed",
                "conversation.item.added",
                "conversation.item.input_audio_transcription.completed",
            },
        )
    return str(stopped["item_id"])


def of_type(events: list[dict[str, Any]], type_: str) -> list[dict[str, Any]]:
    return [e for e in events if e["type"] == type_]


def audio_ms(events: list[dict[str, Any]]) -> float:
    total = sum(len(base64.b64decode(e["delta"])) for e in of_type(events, "response.output_audio.delta"))
    return total / 2 / 24000 * 1000


async def test_a_spoken_turn() -> None:
    async with stack() as s:
        await say(s.client, s.stt, "Is the office light on?")
        await s.client.send("response.create")
        events = await s.client.collect_until("response.done")
        types = [e["type"] for e in events]
        assert types[0] == "response.created"
        done = events[-1]["response"]
        assert done["status"] == "completed"
        assert [e["transcript"] for e in of_type(events, "response.output_audio_transcript.done")] == ["Hello there."]
        # One sentence, 3 chunks of 100 ms of speech at 22.05 kHz, resampled to 24 kHz for the client.
        assert audio_ms(events) == pytest.approx(300, abs=5)
        deltas = of_type(events, "response.output_audio.delta")
        assert {d["item_id"] for d in deltas} == {done["output"][0]["id"]}
        assert done["output"][0]["content"] == [{"type": "output_audio", "transcript": "Hello there."}]
        assert types.index("response.output_item.added") < types.index("response.output_audio.delta")
        assert types.index("response.output_audio.done") < types.index("response.done")
        # The model saw the client's instructions and what the user said.
        messages = s.last_messages()
        assert messages[0] == {"role": "system", "content": KIOSK_SESSION["instructions"]}
        assert messages[-1] == {"role": "user", "content": "Is the office light on?"}
        assert s.tts.sentences == ["Hello there."]


async def test_answer_waits_for_the_transcript() -> None:
    async with stack() as s:
        s.stt.delay = 0.4
        await say(s.client, s.stt, "Slow words", wait_for_transcript=False)
        await s.client.send("response.create")  # right after speech_stopped, before the transcript
        events = await s.client.collect_until("response.done")
        assert events[-1]["response"]["status"] == "completed"
        assert s.last_messages()[-1]["content"] == "Slow words"


async def test_sentences_are_spoken_as_the_model_writes() -> None:
    async with stack() as s:
        s.llm.default_reply = ["First sentence.", " Second", " sentence.\n- a list item"]
        await s.client.send("response.create")
        events = await s.client.collect_until("response.done")
        assert s.tts.sentences == ["First sentence.", "Second sentence.", "a list item"]
        deltas = [e["delta"] for e in of_type(events, "response.output_audio_transcript.delta")]
        assert deltas == ["First sentence. ", "Second sentence. ", "a list item "]
        assert audio_ms(events) == pytest.approx(900, abs=10)


async def test_tool_round_trip() -> None:
    async with stack() as s:
        s.llm.replies = [
            [{"tool": "GetLiveContext", "arguments": {}}],
            ["The office light is on."],
        ]
        await say(s.client, s.stt, "Is the office light on?")
        await s.client.send("response.create")
        events = await s.client.collect_until("response.done")
        [call] = of_type(events, "response.function_call_arguments.done")
        assert call["name"] == "GetLiveContext" and call["arguments"] == "{}"
        assert events[-1]["response"]["status"] == "completed"
        assert events[-1]["response"]["output"][0]["type"] == "function_call"
        assert of_type(events, "response.output_audio.delta") == []
        # The client runs the tool and asks for the answer, as Kiosk Satellite does.
        await s.client.send(
            "conversation.item.create",
            item={"type": "function_call_output", "call_id": call["call_id"], "output": "Office Light: on"},
        )
        await s.client.send("response.create")
        events = await s.client.collect_until("response.done")
        assert of_type(events, "response.output_audio_transcript.done")[0]["transcript"] == "The office light is on."
        messages = s.last_messages()
        assert messages[-2]["tool_calls"][0]["function"]["name"] == "GetLiveContext"
        assert messages[-1] == {"role": "tool", "tool_call_id": call["call_id"], "content": "Office Light: on"}


async def test_cancel_mid_answer_stops_everything() -> None:
    async with stack() as s:
        s.llm.default_reply = ["One.", " Two.", " Three.", " Four.", " Five."]
        s.llm.piece_delay = 0.3
        await s.client.send("response.create")
        await s.client.expect("response.output_audio.delta", skip=STREAMING)
        await s.client.send("response.cancel")
        events = await s.client.collect_until("response.done")
        assert events[-1]["response"]["status"] == "cancelled"
        await s.client.nothing(0.5)  # nothing after done
        await wait_for(lambda: s.llm.cancelled == 1)  # the model stopped writing
        assert len(s.tts.sentences) < 5


async def test_new_response_replaces_the_active_one() -> None:
    async with stack() as s:
        s.llm.default_reply = ["One.", " Two.", " Three."]
        s.llm.piece_delay = 0.3
        await s.client.send("response.create")
        await s.client.expect("response.created")
        await s.client.send("response.create")
        first = await s.client.collect_until("response.done")
        assert first[-1]["response"]["status"] == "cancelled"
        second = await s.client.collect_until("response.done")
        assert second[-1]["response"]["status"] == "completed"
        assert second[0]["response"]["id"] != first[-1]["response"]["id"]


async def test_cancel_with_nothing_active_is_a_benign_error() -> None:
    async with stack() as s:
        await s.client.send("response.cancel")
        assert (await s.client.expect("error"))["error"]["code"] == "response_cancel_not_active"


async def test_truncate_keeps_what_was_played() -> None:
    async with stack() as s:
        s.llm.replies = [["One.", " Two.", " Three."], ["OK."]]
        await say(s.client, s.stt, "Count")
        await s.client.send("response.create")
        events = await s.client.collect_until("response.done")
        item_id = events[-1]["response"]["output"][0]["id"]
        # Each sentence is 300 ms of audio: the user interrupted during the second one.
        await s.client.send("conversation.item.truncate", item_id=item_id, content_index=0, audio_end_ms=450)
        assert (await s.client.expect("conversation.item.truncated"))["audio_end_ms"] == 450
        await say(s.client, s.stt, "Stop")
        await s.client.send("response.create")
        await s.client.collect_until("response.done")
        assert {"role": "assistant", "content": "One. Two."} in s.last_messages()


async def test_model_failure_is_reported_not_hidden() -> None:
    """The prototype turned a model error into an empty "completed"; the client then hung on "thinking"."""
    async with stack() as s:
        s.llm.behaviour, s.llm.error_status = "error", 400
        await s.client.send("response.create")
        events = await s.client.collect_until("response.done")
        response = events[-1]["response"]
        assert response["status"] == "failed"
        assert "HTTP 400" in response["status_details"]["error"]["message"]


async def test_empty_answer_fails() -> None:
    async with stack() as s:
        s.llm.default_reply = ["   "]
        await s.client.send("response.create")
        events = await s.client.collect_until("response.done")
        assert events[-1]["response"]["status"] == "failed"
        assert "empty answer" in events[-1]["response"]["status_details"]["error"]["message"]


async def test_stuck_speech_server_costs_one_sentence() -> None:
    async with stack() as s:
        s.llm.default_reply = ["One.", " Two."]
        s.tts.behaviour = "stall"
        await s.client.send("response.create")
        events = await s.client.collect_until("response.done", within=5)
        assert events[-1]["response"]["status"] == "completed"
        # Each sentence got its first chunk, then the server stalled and was given up on.
        assert audio_ms(events) == pytest.approx(200, abs=10)
        assert of_type(events, "response.output_audio_transcript.done")[0]["transcript"] == "One. Two."


async def test_disconnect_mid_answer_abandons_backend_work() -> None:
    async with fake_wyoming() as stt, fake_wyoming() as tts, fake_openai() as llm:
        llm.default_reply = ["One.", " Two.", " Three."]
        llm.piece_delay = 0.5
        settings = make_settings(
            stt={"type": "wyoming", "uri": stt.uri},
            tts={"type": "wyoming", "uri": tts.uri},
            llm={"type": "openai", "url": llm.url, "model": "m"},
        )
        async with running_server(settings) as srv:
            async with open_client(srv.url()) as client:
                await client.expect("session.created")
                await client.send("response.create")
                await client.expect("response.created")
                await wait_for(lambda: len(llm.chats) == 1)
            await wait_for(lambda: llm.cancelled == 1)


async def test_preamble_and_context_prompt() -> None:
    prompt = {"preamble": "Server rules.", "context": "You are in the {area}. Year {now:%Y}."}
    async with stack(query="room=office", prompt=prompt) as s:
        await say(s.client, s.stt, "Hi")
        await s.client.send("response.create")
        await s.client.collect_until("response.done")
        messages = s.last_messages()
        system = f"Server rules.\n\n{KIOSK_SESSION['instructions']}\n\nYou are in the Office. Year 20"
        assert messages[0]["content"].startswith(system)
        assert messages[1] == {"role": "user", "content": "Hi"}
        # The next prompt only grows at the end.
        first_prompt = messages
        await say(s.client, s.stt, "Again")
        await s.client.send("response.create")
        await s.client.collect_until("response.done")
        assert s.last_messages()[: len(first_prompt)] == first_prompt


async def test_voice_mapping() -> None:
    async with stack(tts={"voice": "default_voice", "voice_map": {"marin": "af_heart"}}) as s:
        await s.client.send("response.create")
        await s.client.collect_until("response.done")
        assert s.tts.voices == ["af_heart"]  # Kiosk Satellite asks for "marin"
        await s.client.update_session({"audio": {"output": {"voice": "cedar"}}})
        await s.client.send("response.create")
        await s.client.collect_until("response.done")
        assert s.tts.voices[-1] == "default_voice"  # unmapped: the configured voice


async def test_replayed_history_reaches_the_model_as_text() -> None:
    """Kiosk Satellite 2026.10.3+ replays earlier turns as content-part lists; models want strings."""
    async with stack() as s:
        for role, part in (
            ("user", {"type": "input_text", "text": "My name is Sam."}),
            ("assistant", {"type": "output_audio", "transcript": "Nice to meet you, Sam."}),
        ):
            await s.client.send("conversation.item.create", item={"type": "message", "role": role, "content": [part]})
            await s.client.expect("conversation.item.done", skip={"conversation.item.added"})
        await say(s.client, s.stt, "What is my name?")
        await s.client.send("response.create")
        await s.client.collect_until("response.done")
        assert s.last_messages()[1:] == [
            {"role": "user", "content": "My name is Sam."},
            {"role": "assistant", "content": "Nice to meet you, Sam."},
            {"role": "user", "content": "What is my name?"},
        ]


async def test_ollama_dialect_end_to_end() -> None:
    async with stack(llm_kind="ollama") as s:
        s.llm.replies = [[{"tool": "HassTurnOn", "arguments": {"name": "Office Light"}}]]
        await s.client.send("response.create")
        events = await s.client.collect_until("response.done")
        [call] = of_type(events, "response.function_call_arguments.done")
        assert json.loads(call["arguments"]) == {"name": "Office Light"}


STREAMING = {
    "response.created",
    "response.output_item.added",
    "conversation.item.added",
    "response.output_audio_transcript.delta",
}


ECHO_REPLY: Reply = ["The office light is on, and the kitchen light is off."]


async def test_echo_of_the_answer_is_not_answered() -> None:
    """The assistant's voice leaking back as "user" speech must not reach the model as the user's words."""
    async with stack() as s:
        s.llm.replies = [ECHO_REPLY]
        await s.client.send("response.create")
        await s.client.collect_until("response.done")
        # While the answer is still playing on the client, the microphone hears it and the client accepts it.
        heard = "the office light is on and the kitchen"
        await say(s.client, s.stt, heard)
        await s.client.send("response.create")
        events = await s.client.collect_until("response.done")
        assert events[-1]["response"]["status"] == "cancelled"
        assert events[-1]["response"]["status_details"]["reason"] == "echo"
        assert len(s.llm.chats) == 1


async def test_same_words_later_are_the_user() -> None:
    async with stack() as s:
        s.llm.replies = [ECHO_REPLY, ["OK."]]
        await s.client.send("response.create")
        await s.client.collect_until("response.done")
        await asyncio.sleep(1.5)  # 300 ms of audio played long ago
        await say(s.client, s.stt, "the office light is on and the kitchen")
        await s.client.send("response.create")
        events = await s.client.collect_until("response.done")
        assert events[-1]["response"]["status"] == "completed"
        assert s.last_messages()[-1]["content"] == "the office light is on and the kitchen"


async def test_echo_filter_can_be_turned_off() -> None:
    async with stack(conversation={"drop_echo_transcripts": False}) as s:
        s.llm.replies = [ECHO_REPLY, ["OK."]]
        await s.client.send("response.create")
        await s.client.collect_until("response.done")
        await say(s.client, s.stt, "the office light is on and the kitchen")
        await s.client.send("response.create")
        assert (await s.client.collect_until("response.done"))[-1]["response"]["status"] == "completed"


async def test_cancelling_an_answer_keeps_the_pending_transcript() -> None:
    """An answer waiting for a transcript is cancelled: the transcription itself must carry on."""
    async with stack() as s:
        s.stt.delay = 0.5
        await say(s.client, s.stt, "Turn on the light", wait_for_transcript=False)
        await s.client.send("response.create")
        await s.client.expect("response.created", skip=AFTER_COMMIT)
        await s.client.send("response.cancel")
        assert (await s.client.collect_until("response.done"))[-1]["response"]["status"] == "cancelled"
        completed = await s.client.expect("conversation.item.input_audio_transcription.completed", skip=AFTER_COMMIT)
        assert completed["transcript"] == "Turn on the light"
        await s.client.send("response.create")
        await s.client.collect_until("response.done")
        assert s.last_messages()[-1]["content"] == "Turn on the light"


AFTER_COMMIT = {"input_audio_buffer.committed", "conversation.item.added"}


async def test_speech_below_the_level_floor_is_not_transcribed_or_answered() -> None:
    """Near-silence makes speech-to-text invent words ("Thank you."); the model then answers nonsense."""
    async with stack(vad={"min_speech_level": 3000}) as s:
        await say(s.client, s.stt, "Thank you.", amplitude=1500)  # loudest frame ~950
        assert s.stt.stt_requests == []
        transcripts = of_type(s.client.received, "conversation.item.input_audio_transcription.completed")
        assert transcripts[-1]["transcript"] == ""
        await s.client.send("response.create")
        events = await s.client.collect_until("response.done")
        assert events[-1]["response"]["status"] == "cancelled"
        assert events[-1]["response"]["status_details"]["reason"] == "quiet"
        assert s.llm.chats == []
        # Speech above the floor is the user, as usual.
        await say(s.client, s.stt, "Turn on the light")  # loudest frame ~3800
        await s.client.send("response.create")
        assert (await s.client.collect_until("response.done"))[-1]["response"]["status"] == "completed"
        assert [m["content"] for m in s.last_messages() if m["role"] == "user"] == ["Turn on the light"]


async def test_conversation_text_is_logged_only_at_debug(caplog: pytest.LogCaptureFixture) -> None:
    async with stack() as s:
        s.llm.replies = [["Checking.", {"tool": "GetLiveContext", "arguments": {}}]]
        with caplog.at_level(logging.INFO, logger="homeduplex"):
            await say(s.client, s.stt, "Is the light on?")
            await s.client.send("response.create")
            await s.client.collect_until("response.done")
        assert "Is the light on" not in caplog.text and "Checking" not in caplog.text
        caplog.clear()
        s.llm.replies = [["It is on."]]
        with caplog.at_level(logging.DEBUG, logger="homeduplex"):
            await say(s.client, s.stt, "And now?")
            await s.client.send(
                "conversation.item.create", item={"type": "function_call_output", "call_id": "c", "output": "on"}
            )
            await s.client.send("response.create")
            await s.client.collect_until("response.done")
        assert "user said 'And now?'" in caplog.text
        assert "FunctionCallOutput 'on'" in caplog.text
        assert "said 'It is on.'" in caplog.text


async def test_long_conversations_keep_the_prompt_prefix_between_trims() -> None:
    """Dropping the oldest turn on every turn made each prompt differ right after the system message, so the model
    server re-read the whole conversation every time (first word 2 s → 5-7 s after ten turns on a real house)."""
    async with stack(conversation={"max_turns": 4}) as s:
        s.llm.replies = [[f"Answer {n}."] for n in range(10)]
        for n in range(10):
            await say(s.client, s.stt, f"Question {n}")
            await s.client.send("response.create")
            await s.client.collect_until("response.done")
        prompts = [chat["messages"] for chat in s.llm.chats]
        extends = [new[: len(old)] == old for old, new in itertools.pairwise(prompts)]
        # Turns 5 and 8 trim back to 2 turns; every other prompt starts with the previous one, unchanged.
        assert [n + 1 for n, ok in enumerate(extends) if not ok] == [4, 7]
        users = [m["content"] for m in prompts[-1] if m["role"] == "user"]
        assert users == ["Question 6", "Question 7", "Question 8", "Question 9"]
