import pytest

from homeduplex.session.config import SessionConfig, Tool, UnsupportedSetting, apply_update


def test_defaults_are_client_driven() -> None:
    config = SessionConfig()
    assert config.turn_detection is not None and not config.turn_detection.create_response


def test_tools_are_filtered_and_normalised() -> None:
    config = apply_update(
        SessionConfig(),
        {
            "tools": [
                {"type": "function", "name": "A", "description": "a", "parameters": {"type": "object"}},
                {"name": "B"},
                {"type": "mcp", "server_label": "x"},
                {"type": "function", "name": ""},
                "junk",
            ]
        },
        800,
    )
    assert [t.name for t in config.tools] == ["A", "B"]
    assert config.tools[1] == Tool("B", "", {"type": "object", "properties": {}})


def test_turn_detection_values() -> None:
    config = apply_update(
        SessionConfig(),
        {"turn_detection": {"silence_duration_ms": 1200, "prefix_padding_ms": 200, "create_response": False}},
        800,
    )
    assert config.turn_detection is not None
    assert config.turn_detection.silence_duration_ms == 1200
    assert config.turn_detection.prefix_padding_ms == 200
    assert config.turn_detection.interrupt_response  # OpenAI's default


@pytest.mark.parametrize("value", ["800", True, None])
def test_bad_numbers_fall_back(value: object) -> None:
    config = apply_update(
        SessionConfig(), {"turn_detection": {"silence_duration_ms": value, "create_response": False}}, 100
    )
    assert config.turn_detection is not None and config.turn_detection.silence_duration_ms == 500


@pytest.mark.parametrize(
    "session",
    [
        {"input_audio_format": "g711_ulaw"},
        {"audio": {"output": {"format": {"type": "audio/pcm", "rate": 16000}}}},
        {"audio": {"input": {"format": {"type": "audio/pcma"}}}},
    ],
)
def test_unsupported_formats(session: dict[str, object]) -> None:
    with pytest.raises(UnsupportedSetting) as info:
        apply_update(SessionConfig(), session, 800)
    assert info.value.code == "unsupported_audio_format"


def test_supported_formats() -> None:
    for session in (
        {"input_audio_format": "pcm16", "output_audio_format": "pcm16"},
        {"audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000}}}},
        {"audio": {"input": {"format": {"type": "audio/pcm"}}}},
    ):
        apply_update(SessionConfig(), session, 800)


def test_wire_shape() -> None:
    wire = SessionConfig(instructions="x", voice="v").to_wire("sess_1", "gpt-realtime")
    assert wire["id"] == "sess_1"
    assert wire["audio"]["input"]["turn_detection"]["type"] == "server_vad"
    assert wire["audio"]["output"] == {"format": {"type": "audio/pcm", "rate": 24000}, "voice": "v"}
