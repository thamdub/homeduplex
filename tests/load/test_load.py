"""The concurrency check from docs/design.md: ~10 clients streaming microphone audio in real time, with full
answer cycles, against stand-in backends. Measures the relay's own cost: time spent handling each client event and
how late the event loop runs. Not part of the default run: `uv run pytest -m load -s`."""

import asyncio
import contextlib
import os
import statistics
import time
from typing import Any, ClassVar

import pytest
from support.audio import chunks, silence, speech
from support.client import KIOSK_SESSION, open_client
from support.fakes.openai import fake_openai
from support.fakes.wyoming import fake_wyoming
from support.server import running_server
from support.settings import make_settings
from support.wait import wait_for
from websockets.exceptions import ConnectionClosed

from homeduplex.app import build_services
from homeduplex.audio.pcm import encode_b64
from homeduplex.session.session import Session
from homeduplex.transport.events import Send
from homeduplex.transport.room import Room

pytestmark = pytest.mark.load

CLIENTS = int(os.environ.get("LOAD_CLIENTS", "10"))
SECONDS = float(os.environ.get("LOAD_SECONDS", "10"))
CHUNK_MS = 85
# One utterance and a pause, repeated: speech_stopped every 2.5 s per client.
CYCLE = silence(500) + speech(1000) + silence(1000)


class TimedSession(Session):
    handle_times: ClassVar[list[float]] = []
    instances: ClassVar[list["TimedSession"]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        TimedSession.instances.append(self)

    async def handle(self, event: Any) -> None:
        started = time.perf_counter()
        await super().handle(event)
        if event.get("type") == "input_audio_buffer.append":
            TimedSession.handle_times.append(time.perf_counter() - started)


async def client_loop(url: str, start: asyncio.Event, until: list[float]) -> int:
    """One kiosk: streams its microphone in real time, asks for an answer after each utterance."""
    answers = 0
    async with open_client(url) as client:
        await client.expect("session.created")
        await client.update_session(KIOSK_SESSION)

        async def listen() -> None:
            nonlocal answers
            with contextlib.suppress(ConnectionClosed):
                while True:
                    event = await client.recv(within=30)
                    if event["type"] == "input_audio_buffer.speech_stopped":
                        await client.send("response.create")
                    elif event["type"] == "response.done":
                        answers += 1

        listener = asyncio.create_task(listen())
        await start.wait()
        pieces = chunks(CYCLE, CHUNK_MS)
        n = 0
        next_send = time.monotonic()
        while time.monotonic() < until[0]:
            await client.send("input_audio_buffer.append", audio=encode_b64(pieces[n % len(pieces)]))
            n += 1
            next_send += CHUNK_MS / 1000
            await asyncio.sleep(max(0, next_send - time.monotonic()))
        # Keep reading while closing: a client that stops reading can't complete the close handshake.
        await client.ws.close()
        await listener
    return answers


async def loop_lag(until: float, lags: list[float]) -> None:
    while time.monotonic() < until:
        expected = time.monotonic() + 0.01
        await asyncio.sleep(0.01)
        lags.append(max(0.0, time.monotonic() - expected))


async def test_ten_clients_in_real_time() -> None:
    TimedSession.handle_times, TimedSession.instances = [], []
    async with fake_wyoming() as stt, fake_wyoming(tts_rate=22050, tts_chunks=10) as tts, fake_openai() as llm:
        llm.default_reply = ["Sure.", " The light is on.", " Anything else?"]
        settings = make_settings(
            stt={"type": "wyoming", "uri": stt.uri, "slots": 4},
            tts={"type": "wyoming", "uri": tts.uri, "slots": 4},
            llm={"type": "openai", "url": llm.url, "model": "m", "slots": 4},
        )
        services = build_services(settings)

        def new_session(send: Send, room: Room, model: str) -> Session:
            return TimedSession(send, settings, services, room, model)

        stt.transcripts = ["turn on the light"] * 10_000
        async with running_server(settings, new_session=new_session) as srv:
            start, until = asyncio.Event(), [0.0]
            runs = [asyncio.create_task(client_loop(srv.url(), start, until)) for _ in range(CLIENTS)]
            await wait_for(lambda: len(TimedSession.instances) == CLIENTS)
            await asyncio.sleep(0.5)  # sessions validated
            cpu_start, wall_start = time.process_time(), time.monotonic()
            until[0] = wall_start + SECONDS
            lags: list[float] = []
            lag = asyncio.create_task(loop_lag(until[0], lags))
            start.set()
            answers = sum(await asyncio.gather(*runs))
            await lag
            cpu = time.process_time() - cpu_start
            wall = time.monotonic() - wall_start

    handle = sorted(TimedSession.handle_times)
    p99 = handle[int(len(handle) * 0.99)] * 1000
    report = (
        f"{CLIENTS} clients, {wall:.1f} s: {len(handle)} audio events, {answers} answers; "
        f"handling an audio event: median {statistics.median(handle) * 1000:.2f} ms, p99 {p99:.2f} ms, "
        f"max {handle[-1] * 1000:.2f} ms; event loop lag p99 {sorted(lags)[int(len(lags) * 0.99)] * 1000:.1f} ms, "
        f"max {max(lags) * 1000:.1f} ms; CPU {cpu / wall * 100:.0f}% of one core (clients and fakes included)"
    )
    print("\n" + report)
    assert answers >= CLIENTS * int(SECONDS / 2.5) - CLIENTS  # every client got its answers
    assert p99 < 5, report  # "a few milliseconds per frame at that load" (docs/design.md)
