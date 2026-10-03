"""Microphone audio in: turn detection events, transcription, and the client's say over what was the user."""

import contextlib
from collections.abc import AsyncIterator
from typing import Any

import pytest
from support.audio import chunks, silence, speech
from support.client import KIOSK_SESSION, RealtimeClient, open_client
from support.fakes.wyoming import FakeWyoming, fake_wyoming
from support.server import running_server
from support.settings import make_settings
from support.wait import wait_for

from homeduplex.audio.pcm import encode_b64


@contextlib.asynccontextmanager
async def kiosk(**fake: Any) -> AsyncIterator[tuple[RealtimeClient, FakeWyoming]]:
    """A connected, validated Kiosk-like client, with a fake Wyoming speech-to-text behind the server."""
    async with fake_wyoming(**fake) as stt:
        settings = make_settings(stt={"type": "wyoming", "uri": stt.uri, "timeout_s": 2})
        async with running_server(settings) as srv, open_client(srv.url()) as client:
            await client.expect("session.created")
            await client.update_session(KIOSK_SESSION)
            yield client, stt


async def say(client: RealtimeClient, pcm: bytes) -> None:
    for chunk in chunks(pcm):
        await client.send("input_audio_buffer.append", audio=encode_b64(chunk))


UTTERANCE = silence(500) + speech(1200) + silence(1000)


async def test_utterance_events_in_order() -> None:
    async with kiosk(transcripts=["what time is it"]) as (client, stt):
        await say(client, UTTERANCE)
        started = await client.expect("input_audio_buffer.speech_started")
        item_id = started["item_id"]
        assert 100 <= started["audio_start_ms"] <= 500
        stopped = await client.expect("input_audio_buffer.speech_stopped")
        assert stopped["item_id"] == item_id
        assert stopped["audio_end_ms"] > started["audio_start_ms"] + 1200
        committed = await client.expect("input_audio_buffer.committed")
        assert committed["item_id"] == item_id and committed["previous_item_id"] is None
        added = await client.expect("conversation.item.added")
        assert added["item"]["id"] == item_id
        assert added["item"]["content"] == [{"type": "input_audio", "transcript": None}]
        completed = await client.expect("conversation.item.input_audio_transcription.completed")
        assert completed["item_id"] == item_id and completed["transcript"] == "what time is it"
        done = await client.expect("conversation.item.done")
        assert done["item"]["content"][0]["transcript"] == "what time is it"
        # The utterance reached speech-to-text whole (300 ms lead-in + speech + 800 ms of silence), at its rate.
        [request] = stt.stt_requests
        assert request.rate == 16000
        assert len(request.audio) / 2 / 16000 == pytest.approx(0.3 + 1.2 + 0.8, abs=0.1)


async def test_two_utterances_chain_items() -> None:
    async with kiosk(transcripts=["one", "two"]) as (client, _):
        await say(client, UTTERANCE + UTTERANCE)
        first = (await client.expect("input_audio_buffer.committed", skip=SPEECH))["item_id"]
        second = await client.expect("input_audio_buffer.committed", skip=SPEECH | AFTER_COMMIT)
        assert second["previous_item_id"] == first


async def test_client_deletes_speech_that_was_not_the_user() -> None:
    """Kiosk Satellite judges each utterance; if it was echo or noise it deletes the item."""
    async with kiosk(delay=0.5) as (client, stt):
        await say(client, UTTERANCE)
        item_id = (await client.expect("input_audio_buffer.committed", skip=SPEECH))["item_id"]
        await client.expect("conversation.item.added")
        await wait_for(lambda: len(stt.stt_requests) == 1)  # in flight at the speech-to-text server
        await client.send("conversation.item.delete", item_id=item_id)
        assert (await client.expect("conversation.item.deleted"))["item_id"] == item_id
        await client.nothing(0.8)  # no transcription for a deleted item
        assert stt.disconnects == 1  # the transcription was abandoned


async def test_transcription_failure_is_reported() -> None:
    async with kiosk(behaviour="close") as (client, _):
        await say(client, UTTERANCE)
        failed = await client.expect("conversation.item.input_audio_transcription.failed", skip=SPEECH | AFTER_COMMIT)
        assert failed["error"]["code"] == "transcription_failed"
        assert "speech-to-text" in failed["error"]["message"]


async def test_clear_drops_speech_in_progress() -> None:
    async with kiosk() as (client, stt):
        await say(client, silence(500) + speech(600))
        await client.expect("input_audio_buffer.speech_started")
        await client.send("input_audio_buffer.clear")
        await client.expect("input_audio_buffer.cleared")
        await say(client, silence(1500))
        await client.nothing()
        assert stt.stt_requests == []


async def test_silence_produces_nothing() -> None:
    async with kiosk() as (client, _):
        await say(client, silence(3000))
        await client.nothing()


async def test_bad_audio() -> None:
    async with kiosk() as (client, _):
        await client.send("input_audio_buffer.append", audio="%%%")
        assert (await client.expect("error"))["error"]["code"] == "invalid_value"
        await client.send("input_audio_buffer.append")
        assert (await client.expect("error"))["error"]["code"] == "invalid_value"


async def test_history_replay_and_tool_output_items() -> None:
    async with kiosk() as (client, _):
        await client.send(
            "conversation.item.create",
            item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hi"}]},
        )
        added = await client.expect("conversation.item.added")
        assert added["item"]["content"] == [{"type": "input_text", "text": "Hi"}]
        await client.expect("conversation.item.done")
        await client.send(
            "conversation.item.create",
            item={"type": "message", "role": "assistant", "content": [{"type": "output_audio", "transcript": "Hey"}]},
        )
        assert (await client.expect("conversation.item.added"))["previous_item_id"] == added["item"]["id"]
        await client.expect("conversation.item.done")
        await client.send(
            "conversation.item.create", item={"type": "function_call_output", "call_id": "c1", "output": "ok"}
        )
        assert (await client.expect("conversation.item.added"))["item"]["type"] == "function_call_output"


async def test_item_errors() -> None:
    async with kiosk() as (client, _):
        await client.send("conversation.item.create", item={"type": "image"})
        assert (await client.expect("error"))["error"]["code"] == "invalid_value"
        await client.send("conversation.item.create", item={"id": "x", "type": "message", "content": "a"})
        await client.expect("conversation.item.added")
        await client.expect("conversation.item.done")
        await client.send("conversation.item.create", item={"id": "x", "type": "message", "content": "b"})
        assert (await client.expect("error"))["error"]["code"] == "duplicate_item_id"
        await client.send("conversation.item.delete", item_id="nope")
        assert (await client.expect("error"))["error"]["code"] == "item_not_found"
        await client.send("conversation.item.truncate", item_id="x", content_index=0, audio_end_ms=10)
        assert (await client.expect("error"))["error"]["code"] == "item_not_found"  # not an assistant item


SPEECH = {"input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped"}
AFTER_COMMIT = {
    "input_audio_buffer.committed",
    "conversation.item.added",
    "conversation.item.input_audio_transcription.completed",
    "conversation.item.done",
}
