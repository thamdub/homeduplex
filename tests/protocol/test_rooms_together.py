"""Several rooms talking at once share the backends fairly."""

import asyncio
import contextlib
from collections.abc import AsyncIterator

import pytest
from support.client import KIOSK_SESSION, RealtimeClient, open_client
from support.fakes.openai import FakeOpenAI, fake_openai
from support.fakes.wyoming import fake_wyoming
from support.server import running_server
from support.settings import make_settings


@contextlib.asynccontextmanager
async def two_rooms(llm_slots: int) -> AsyncIterator[tuple[RealtimeClient, RealtimeClient, FakeOpenAI]]:
    async with fake_wyoming() as stt, fake_wyoming() as tts, fake_openai() as llm:
        llm.default_reply = ["One.", " Two."]
        llm.piece_delay = 0.1
        settings = make_settings(
            stt={"type": "wyoming", "uri": stt.uri},
            tts={"type": "wyoming", "uri": tts.uri},
            llm={"type": "openai", "url": llm.url, "model": "m", "slots": llm_slots},
        )
        async with (
            running_server(settings) as srv,
            open_client(srv.url("room=office")) as office,
            open_client(srv.url("room=kitchen")) as kitchen,
        ):
            for client in (office, kitchen):
                await client.expect("session.created")
                await client.update_session(KIOSK_SESSION)
            yield office, kitchen, llm


@pytest.mark.parametrize("slots", [1, 2])
async def test_model_slots_are_respected(slots: int) -> None:
    async with two_rooms(slots) as (office, kitchen, llm):
        await asyncio.gather(office.send("response.create"), kitchen.send("response.create"))
        done = await asyncio.gather(office.collect_until("response.done"), kitchen.collect_until("response.done"))
        assert [d[-1]["response"]["status"] for d in done] == ["completed", "completed"]
        assert llm.max_active_chats == slots


async def test_cancelling_a_waiting_answer_frees_its_place() -> None:
    async with two_rooms(1) as (office, kitchen, llm):
        llm.piece_delay = 0.3
        await office.send("response.create")
        await office.expect("response.created")
        await kitchen.send("response.create")
        await kitchen.expect("response.created")
        await kitchen.send("response.cancel")  # still waiting for the model
        assert (await kitchen.collect_until("response.done"))[-1]["response"]["status"] == "cancelled"
        assert (await office.collect_until("response.done"))[-1]["response"]["status"] == "completed"
        assert len(llm.chats) == 1  # the kitchen's request never reached the model
