"""Which room a connection talks from: `?room=<id>` on the URL, else the port it connected to, else the default."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urlsplit


@dataclass(frozen=True)
class Room:
    id: str | None
    fields: Mapping[str, str] = field(default_factory=dict)


class UnknownRoom(ValueError):
    pass


def resolve(path: str, port_room: str | None, rooms: Mapping[str, Mapping[str, str]], default: str | None) -> Room:
    room_id = query_param(path, "room") or port_room or default
    if room_id is None:
        return Room(None)
    if room_id not in rooms:
        known = ", ".join(sorted(rooms)) or "none configured"
        raise UnknownRoom(f"unknown room {room_id!r} (known: {known})")
    return Room(room_id, dict(rooms[room_id]))


def query_param(path: str, name: str) -> str | None:
    values = parse_qs(urlsplit(path).query).get(name)
    return values[0] if values else None
