"""A Wyoming TCP connection with timeouts, closed however the caller leaves (including cancellation)."""

import asyncio
import contextlib
from collections.abc import AsyncIterator
from urllib.parse import urlsplit

from wyoming.event import Event, async_read_event, async_write_event

from homeduplex.backends.base import BackendError


class WyomingConnection:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, name: str) -> None:
        self._reader = reader
        self._writer = writer
        self._name = name

    async def write(self, event: Event) -> None:
        try:
            await async_write_event(event, self._writer)
        except OSError as e:
            raise BackendError(f"{self._name}: connection lost ({e})") from None

    async def read(self, within: float) -> Event:
        """The next event; BackendError if none comes within `within` seconds or the server hangs up."""
        try:
            event = await asyncio.wait_for(async_read_event(self._reader), within)
        except TimeoutError:
            raise BackendError(f"{self._name}: no answer within {within:g} s") from None
        except OSError as e:
            raise BackendError(f"{self._name}: connection lost ({e})") from None
        if event is None:
            raise BackendError(f"{self._name}: the server closed the connection")
        return event


@contextlib.asynccontextmanager
async def connect(uri: str, name: str, within: float) -> AsyncIterator[WyomingConnection]:
    parts = urlsplit(uri)
    host, port = parts.hostname, parts.port
    if host is None or port is None:
        raise BackendError(f"{name}: bad address {uri}")
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), within)
    except TimeoutError:
        raise BackendError(f"{name}: cannot connect to {uri} within {within:g} s") from None
    except OSError as e:
        raise BackendError(f"{name}: cannot connect to {uri} ({e.strerror or e})") from None
    try:
        yield WyomingConnection(reader, writer, name)
    finally:
        # Closing promptly is what tells the server to stop: never wait on a server that may be stuck.
        writer.close()
        with contextlib.suppress(OSError, TimeoutError):
            await asyncio.wait_for(writer.wait_closed(), 1)
