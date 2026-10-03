"""The WebSocket server: accepts Realtime connections on any path, checks the key, picks the room, and pumps JSON
events between the socket and a Session. `GET /healthz` answers plain HTTP for monitoring."""

import functools
import hmac
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Protocol
from urllib.parse import urlsplit

from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

from homeduplex.config import Settings
from homeduplex.transport import events
from homeduplex.transport.events import Event, Send
from homeduplex.transport.room import Room, UnknownRoom, query_param, resolve

log = logging.getLogger(__name__)

MAX_MESSAGE_BYTES = 8 * 1024 * 1024  # replayed history can be large
DEFAULT_MODEL = "homeduplex"


class SessionLike(Protocol):
    async def start(self) -> None: ...
    async def handle(self, event: Mapping[str, Any]) -> None: ...
    async def close(self) -> None: ...


SessionFactory = Callable[[Send, Room, str], SessionLike]


@dataclass(frozen=True)
class Listener:
    host: str
    port: int
    # Room for clients connecting here without `?room=`.
    room: str | None = None


def listeners(settings: Settings) -> list[Listener]:
    host = settings.server.host
    return [Listener(host, settings.server.port)] + [
        Listener(host, port, room) for port, room in settings.server.extra_ports.items()
    ]


class RealtimeServer:
    def __init__(self, settings: Settings, new_session: SessionFactory) -> None:
        self._settings = settings
        self._new_session = new_session
        self._servers: list[Server] = []

    async def start(self, where: list[Listener]) -> None:
        for listener in where:
            server = await serve(
                functools.partial(self._connection, port_room=listener.room),
                listener.host,
                listener.port,
                process_request=functools.partial(self._check_request, port_room=listener.room),
                max_size=MAX_MESSAGE_BYTES,
            )
            self._servers.append(server)
            log.info("listening on %s:%d%s", listener.host, self.port(server), _room_note(listener.room))

    @property
    def ports(self) -> list[int]:
        """Bound ports, in the order of the listeners (useful when they were 0)."""
        return [self.port(s) for s in self._servers]

    @staticmethod
    def port(server: Server) -> int:
        return int(server.sockets[0].getsockname()[1])

    async def close(self) -> None:
        for server in self._servers:
            server.close()
        for server in self._servers:
            await server.wait_closed()
        self._servers.clear()

    def _check_request(
        self, connection: ServerConnection, request: Request, *, port_room: str | None
    ) -> Response | None:
        if urlsplit(request.path).path == "/healthz":
            return connection.respond(HTTPStatus.OK, "ok\n")
        key = self._settings.server.api_key
        if key is not None:
            sent = request.headers.get("Authorization", "")
            expected = f"Bearer {key.get_secret_value()}"
            if not hmac.compare_digest(sent.encode(), expected.encode()):
                log.warning("refused %s: bad or missing API key", connection.remote_address)
                return connection.respond(HTTPStatus.UNAUTHORIZED, "missing or wrong API key\n")
        try:
            self._room(request.path, port_room)
        except UnknownRoom as e:
            log.warning("refused %s: %s", connection.remote_address, e)
            return connection.respond(HTTPStatus.NOT_FOUND, f"{e}\n")
        return None

    def _room(self, path: str, port_room: str | None) -> Room:
        return resolve(path, port_room, self._settings.rooms, self._settings.default_room)

    async def _connection(self, ws: ServerConnection, *, port_room: str | None) -> None:
        path = ws.request.path if ws.request else "/"
        room = self._room(path, port_room)
        model = query_param(path, "model") or DEFAULT_MODEL

        async def send(event: Event) -> None:
            try:
                await ws.send(json.dumps(event))
            except ConnectionClosed:
                pass

        session = self._new_session(send, room, model)
        log.info("connected %s, room %s", ws.remote_address, room.id or "-")
        try:
            await session.start()
            async for raw in ws:
                event = _parse(raw)
                if event is None:
                    await send(events.error("invalid_json", "expected a JSON object per message"))
                    continue
                await session.handle(event)
        except ConnectionClosed:
            pass
        finally:
            await session.close()
            log.info("disconnected %s, room %s", ws.remote_address, room.id or "-")


def _parse(raw: str | bytes) -> dict[str, Any] | None:
    try:
        event = json.loads(raw)
    except ValueError:
        return None
    return event if isinstance(event, dict) else None


def _room_note(room: str | None) -> str:
    return f" (room {room})" if room else ""
