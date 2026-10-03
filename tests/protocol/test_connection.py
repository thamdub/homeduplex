"""Connecting, validating a session (Kiosk Satellite's "Save & Validate") and session settings."""

import asyncio
import json
import urllib.request

import pytest
from pydantic import SecretStr
from support.client import FLAT_SESSION, KIOSK_SESSION, open_client
from support.server import running_server
from support.settings import make_settings
from websockets.exceptions import InvalidStatus


async def test_validation_handshake() -> None:
    """The client is ready on session.updated after its session.update."""
    async with running_server(make_settings()) as srv, open_client(srv.url()) as client:
        created = await client.expect("session.created")
        assert created["session"]["id"].startswith("sess_")
        assert created["session"]["model"] == "gpt-realtime"  # echoes the model the client asked for
        updated = await client.update_session(KIOSK_SESSION)
        session = updated["session"]
        assert session["id"] == created["session"]["id"]
        assert session["instructions"] == KIOSK_SESSION["instructions"]
        assert [t["name"] for t in session["tools"]] == ["GetLiveContext"]
        assert session["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
        assert session["audio"]["output"]["voice"] == "marin"
        assert all(e["event_id"].startswith("event_") for e in client.received)


async def test_silence_is_raised_to_the_configured_minimum() -> None:
    async with running_server(make_settings()) as srv, open_client(srv.url()) as client:
        await client.expect("session.created")
        session = (await client.update_session(KIOSK_SESSION))["session"]
        detection = session["audio"]["input"]["turn_detection"]
        assert detection["silence_duration_ms"] == 800  # client asked for 500
        assert detection["create_response"] is False


async def test_flat_session_shape() -> None:
    async with running_server(make_settings()) as srv, open_client(srv.url()) as client:
        await client.expect("session.created")
        session = (await client.update_session(FLAT_SESSION))["session"]
        assert session["instructions"] == "Be brief."
        assert session["audio"]["output"]["voice"] == "Ara"
        assert session["audio"]["input"]["turn_detection"]["silence_duration_ms"] == 900
        assert session["tools"][0]["parameters"] == {}


async def test_partial_update_keeps_earlier_settings() -> None:
    async with running_server(make_settings()) as srv, open_client(srv.url()) as client:
        await client.expect("session.created")
        await client.update_session(KIOSK_SESSION)
        session = (await client.update_session({"instructions": "New."}))["session"]
        assert session["instructions"] == "New."
        assert [t["name"] for t in session["tools"]] == ["GetLiveContext"]


@pytest.mark.parametrize(
    ("turn_detection", "code"),
    [
        ({"type": "server_vad"}, "unsupported_turn_detection"),  # create_response defaults to true
        ({"type": "server_vad", "create_response": True}, "unsupported_turn_detection"),
        (None, "unsupported_turn_detection"),
    ],
)
async def test_server_driven_turns_are_refused_clearly(turn_detection: object, code: str) -> None:
    async with running_server(make_settings()) as srv, open_client(srv.url()) as client:
        await client.expect("session.created")
        await client.send(
            "session.update", event_id="evt_1", session={"instructions": "x", "turn_detection": turn_detection}
        )
        error = (await client.expect("error"))["error"]
        assert error["code"] == code
        assert error["event_id"] == "evt_1"
        # Nothing was applied.
        session = (await client.update_session({}))["session"]
        assert session["instructions"] == ""
        assert session["audio"]["input"]["turn_detection"]["create_response"] is False


async def test_unsupported_audio_format() -> None:
    async with running_server(make_settings()) as srv, open_client(srv.url()) as client:
        await client.expect("session.created")
        await client.send("session.update", session={"audio": {"input": {"format": {"type": "audio/pcmu"}}}})
        error = (await client.expect("error"))["error"]
        assert error["code"] == "unsupported_audio_format"


async def test_malformed_messages() -> None:
    async with running_server(make_settings()) as srv, open_client(srv.url()) as client:
        await client.expect("session.created")
        await client.ws.send("not json")
        assert (await client.expect("error"))["error"]["code"] == "invalid_json"
        await client.ws.send(json.dumps([1, 2]))
        assert (await client.expect("error"))["error"]["code"] == "invalid_json"
        await client.send("session.update", session="nope")
        assert (await client.expect("error"))["error"]["code"] == "invalid_value"


async def test_unknown_events_are_ignored() -> None:
    async with running_server(make_settings()) as srv, open_client(srv.url()) as client:
        await client.expect("session.created")
        await client.send("transcription_session.update", session={})
        await client.send("output_audio_buffer.clear")
        await client.nothing()
        await client.update_session({})  # still alive


async def test_api_key() -> None:
    settings = make_settings(server={"api_key": "s3cret"})
    async with running_server(settings) as srv:
        with pytest.raises(InvalidStatus) as info:
            async with open_client(srv.url()):
                pass
        assert info.value.response.status_code == 401
        with pytest.raises(InvalidStatus):
            async with open_client(srv.url(), {"Authorization": "Bearer wrong"}):
                pass
        async with open_client(srv.url(), {"Authorization": "Bearer s3cret"}) as client:
            await client.expect("session.created")


async def test_healthz() -> None:
    async with running_server(make_settings()) as srv:
        url = f"http://127.0.0.1:{srv.ports[0]}/healthz"
        body = await asyncio.to_thread(lambda: urllib.request.urlopen(url, timeout=2).read())
        assert body == b"ok\n"


async def test_model_defaults_when_client_names_none() -> None:
    async with running_server(make_settings()) as srv, open_client(srv.url(query="")) as client:
        assert (await client.expect("session.created"))["session"]["model"] == "homeduplex"


def test_secret_type() -> None:
    assert isinstance(make_settings(server={"api_key": "k"}).server.api_key, SecretStr)
