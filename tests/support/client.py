"""A Realtime client for tests, sending the event shapes real clients send (docs/protocol.md)."""

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Iterable
from typing import Any

from websockets.asyncio.client import ClientConnection, connect

# Kiosk Satellite 2026.10's session.update (OpenAI GA shape), with one Home Assistant tool.
KIOSK_SESSION: dict[str, Any] = {
    "type": "realtime",
    "instructions": "You are a voice assistant on a wall tablet. Keep answers short.",
    "output_modalities": ["audio"],
    "tools": [
        {
            "type": "function",
            "name": "GetLiveContext",
            "description": "Provides real-time information about the CURRENT state, value, or mode of devices.",
            "parameters": {"type": "object", "properties": {}},
        }
    ],
    "tool_choice": "auto",
    "audio": {
        "input": {
            "format": {"type": "audio/pcm", "rate": 24000},
            "turn_detection": {
                "type": "server_vad",
                "silence_duration_ms": 500,
                "prefix_padding_ms": 300,
                "create_response": False,
                "interrupt_response": False,
            },
        },
        "output": {"format": {"type": "audio/pcm", "rate": 24000}, "voice": "marin"},
    },
}

# The flat shape (OpenAI beta, xAI).
FLAT_SESSION: dict[str, Any] = {
    "instructions": "Be brief.",
    "voice": "Ara",
    "input_audio_format": "pcm16",
    "output_audio_format": "pcm16",
    "turn_detection": {"type": "server_vad", "silence_duration_ms": 900, "create_response": False},
    "tools": [{"type": "function", "name": "HassTurnOn", "description": "Turn on", "parameters": {}}],
}


class RealtimeClient:
    def __init__(self, ws: ClientConnection) -> None:
        self.ws = ws
        self.received: list[dict[str, Any]] = []

    async def send(self, type_: str, **fields: Any) -> None:
        await self.ws.send(json.dumps({"type": type_, **fields}))

    async def recv(self, within: float = 2) -> dict[str, Any]:
        raw = await asyncio.wait_for(self.ws.recv(), within)
        event: dict[str, Any] = json.loads(raw)
        self.received.append(event)
        return event

    async def expect(self, type_: str, within: float = 2, skip: Iterable[str] = ()) -> dict[str, Any]:
        """The next event, which must be `type_`; events whose type is in `skip` are passed over."""
        skipped = set(skip)
        async with asyncio.timeout(within):
            while True:
                event = await self.recv(within)
                if event["type"] == type_:
                    return event
                if event["type"] not in skipped:
                    raise AssertionError(f"expected {type_}, got {event}")

    async def collect_until(self, type_: str, within: float = 5) -> list[dict[str, Any]]:
        """Every event up to and including the next `type_`."""
        out: list[dict[str, Any]] = []
        async with asyncio.timeout(within):
            while True:
                event = await self.recv(within)
                out.append(event)
                if event["type"] == type_:
                    return out

    async def nothing(self, within: float = 0.2) -> None:
        """Assert that the server sends nothing for a while."""
        try:
            event = await self.recv(within)
        except TimeoutError:
            return
        raise AssertionError(f"expected no event, got {event}")

    async def update_session(self, session: dict[str, Any]) -> dict[str, Any]:
        await self.send("session.update", session=session)
        return await self.expect("session.updated")


@contextlib.asynccontextmanager
async def open_client(url: str, headers: dict[str, str] | None = None) -> AsyncIterator[RealtimeClient]:
    async with connect(url, additional_headers=headers) as ws:
        yield RealtimeClient(ws)
