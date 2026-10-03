"""Which room a connection belongs to."""

import pytest
from support.client import open_client
from support.server import running_server
from support.settings import make_settings
from websockets.exceptions import InvalidStatus

from homeduplex.app import build_services
from homeduplex.session.session import Session
from homeduplex.transport.events import Send
from homeduplex.transport.room import Room, UnknownRoom, resolve

ROOMS = {"office": {"area": "Office"}, "kitchen": {"area": "Kitchen"}}


@pytest.mark.parametrize(
    ("path", "port_room", "default", "expected"),
    [
        ("/v1/realtime?model=m&room=kitchen", None, None, "kitchen"),
        ("/v1/realtime?room=kitchen", "office", None, "kitchen"),  # query beats port
        ("/v1/realtime?model=m", "office", "kitchen", "office"),  # port beats default
        ("/v1/realtime", None, "kitchen", "kitchen"),
        ("/", None, None, None),
        ("/v1/realtime?room=", None, None, None),
    ],
)
def test_resolve(path: str, port_room: str | None, default: str | None, expected: str | None) -> None:
    room = resolve(path, port_room, ROOMS, default)
    assert room.id == expected
    assert room.fields == (ROOMS[expected] if expected else {})


def test_unknown_room() -> None:
    with pytest.raises(UnknownRoom, match=r"'attic'.*kitchen, office"):
        resolve("/?room=attic", None, ROOMS, None)


async def test_connection_gets_its_room() -> None:
    settings = make_settings()
    rooms: list[Room] = []

    services = build_services(settings)

    def new_session(send: Send, room: Room, model: str) -> Session:
        rooms.append(room)
        return Session(send, settings, services, room, model)

    async with running_server(settings, port_rooms=(None, "office"), new_session=new_session) as srv:
        for url in (srv.url("room=kitchen"), srv.url(port_index=1), srv.url()):
            async with open_client(url) as client:
                await client.expect("session.created")
    assert [r.id for r in rooms] == ["kitchen", "office", None]
    assert rooms[0].fields == {"area": "Kitchen"}


async def test_unknown_room_is_refused_at_connect() -> None:
    async with running_server(make_settings()) as srv:
        with pytest.raises(InvalidStatus) as info:
            async with open_client(srv.url("room=attic")):
                pass
        assert info.value.response.status_code == 404
