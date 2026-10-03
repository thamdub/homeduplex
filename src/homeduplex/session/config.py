"""What a client asks for in `session.update`, and what the server reports back in `session.updated`.

Clients send two shapes. The OpenAI GA shape nests audio settings:
`{audio: {input: {format, turn_detection}, output: {format, voice}}}`. The older (beta, also xAI) shape is flat:
`{turn_detection, voice, input_audio_format, output_audio_format}`. Both are read; the reply uses the GA shape.

Parsing is lenient: unknown fields are ignored and a field that is absent keeps its previous value, as OpenAI does.
"""

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from homeduplex.audio.pcm import CLIENT_RATE


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: Mapping[str, Any]

    def to_wire(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class TurnDetection:
    """Server voice detection. `create_response=False` means the client decides when to answer (see
    docs/design.md, Turn-taking)."""

    silence_duration_ms: int = 500
    prefix_padding_ms: int = 300
    create_response: bool = True
    interrupt_response: bool = True

    def to_wire(self) -> dict[str, Any]:
        return {"type": "server_vad", **dataclasses.asdict(self)}


@dataclass(frozen=True)
class SessionConfig:
    instructions: str = ""
    tools: tuple[Tool, ...] = ()
    # None: the client turned voice detection off and commits audio itself.
    turn_detection: TurnDetection | None = field(default_factory=lambda: TurnDetection(create_response=False))
    voice: str | None = None

    def to_wire(self, session_id: str, model: str) -> dict[str, Any]:
        audio_format = {"type": "audio/pcm", "rate": CLIENT_RATE}
        turn_detection = self.turn_detection.to_wire() if self.turn_detection else None
        return {
            "id": session_id,
            "object": "realtime.session",
            "type": "realtime",
            "model": model,
            "output_modalities": ["audio"],
            "instructions": self.instructions,
            "tools": [t.to_wire() for t in self.tools],
            "tool_choice": "auto",
            "audio": {
                "input": {"format": audio_format, "turn_detection": turn_detection},
                "output": {"format": audio_format, "voice": self.voice},
            },
        }


class UnsupportedSetting(ValueError):
    """The client asked for something this server can't do; reported as an `error` event."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


_UNSET: Any = object()

# What a user can do about a refused turn-taking mode: they rarely control the protocol, only the client's options.
_CLIENT_TURNS_HINT = (
    "Use a client mode where the client decides the turns (server_vad with create_response: false). "
    "With Kiosk Satellite, choose the OpenAI provider, not xAI Grok."
)


def apply_update(config: SessionConfig, session: Mapping[str, Any], min_silence_ms: int) -> SessionConfig:
    """The config after a `session.update`. Raises UnsupportedSetting for audio formats other than 24 kHz PCM16
    and for turn-taking modes the server doesn't implement yet; nothing is applied in that case."""
    audio = _mapping(session.get("audio"))
    audio_in = _mapping(audio.get("input"))
    audio_out = _mapping(audio.get("output"))

    for fmt in (
        audio_in.get("format", session.get("input_audio_format")),
        audio_out.get("format", session.get("output_audio_format")),
    ):
        _check_format(fmt)

    changes: dict[str, Any] = {}
    if isinstance(instructions := session.get("instructions"), str):
        changes["instructions"] = instructions
    if isinstance(tools := session.get("tools"), list):
        changes["tools"] = tuple(t for t in map(_tool, tools) if t is not None)
    voice = audio_out.get("voice", session.get("voice"))
    if isinstance(voice, str):
        changes["voice"] = voice

    raw_td = audio_in.get("turn_detection", session.get("turn_detection", _UNSET))
    if raw_td is not _UNSET:
        changes["turn_detection"] = _turn_detection(raw_td, min_silence_ms)

    return dataclasses.replace(config, **changes)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _check_format(fmt: Any) -> None:
    if fmt is None or fmt == "pcm16":
        return
    if isinstance(fmt, Mapping):
        kind, rate = fmt.get("type", "audio/pcm"), fmt.get("rate", CLIENT_RATE)
        if kind == "audio/pcm" and rate == CLIENT_RATE:
            return
        fmt = f"{kind} at {rate} Hz"
    raise UnsupportedSetting("unsupported_audio_format", f"audio format {fmt} is not supported; use 24 kHz PCM16")


def _tool(raw: Any) -> Tool | None:
    if not isinstance(raw, Mapping) or raw.get("type", "function") != "function":
        return None
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        return None
    parameters = raw.get("parameters")
    return Tool(
        name=name,
        description=str(raw.get("description") or ""),
        parameters=parameters if isinstance(parameters, Mapping) else {"type": "object", "properties": {}},
    )


def _turn_detection(raw: Any, min_silence_ms: int) -> TurnDetection:
    if raw is None:
        raise UnsupportedSetting(
            "unsupported_turn_detection",
            "this client sends turn_detection: null (it commits audio itself), which homeduplex does not support yet. "
            f"{_CLIENT_TURNS_HINT}",
        )
    td = _mapping(raw)
    detection = TurnDetection(
        silence_duration_ms=max(min_silence_ms, _int(td.get("silence_duration_ms"), 500)),
        prefix_padding_ms=_int(td.get("prefix_padding_ms"), 300),
        create_response=td.get("create_response", True) is not False,
        interrupt_response=td.get("interrupt_response", True) is not False,
    )
    if detection.create_response:
        raise UnsupportedSetting(
            "unsupported_turn_detection",
            "this client wants the server to decide when the user has finished speaking (create_response: true, "
            f"the OpenAI default), which homeduplex does not support yet. {_CLIENT_TURNS_HINT}",
        )
    return detection


def _int(value: Any, default: int) -> int:
    return int(value) if isinstance(value, int | float) and not isinstance(value, bool) else default
