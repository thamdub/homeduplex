import contextlib
from collections.abc import AsyncIterator
from dataclasses import dataclass

from homeduplex.app import build_server, build_services, close_services
from homeduplex.config import Settings
from homeduplex.transport.server import Listener, RealtimeServer, SessionFactory


@dataclass
class RunningServer:
    server: RealtimeServer
    ports: list[int]

    def url(self, query: str = "model=gpt-realtime", port_index: int = 0, path: str = "/v1/realtime") -> str:
        return f"ws://127.0.0.1:{self.ports[port_index]}{path}" + (f"?{query}" if query else "")


@contextlib.asynccontextmanager
async def running_server(
    settings: Settings, port_rooms: tuple[str | None, ...] = (None,), new_session: SessionFactory | None = None
) -> AsyncIterator[RunningServer]:
    """The real server on ephemeral localhost ports, one listener per entry of `port_rooms`."""
    services = build_services(settings)
    server = RealtimeServer(settings, new_session) if new_session else build_server(settings, services)
    await server.start([Listener("127.0.0.1", 0, room) for room in port_rooms])
    try:
        yield RunningServer(server, server.ports)
    finally:
        await server.close()
        await close_services(services)
