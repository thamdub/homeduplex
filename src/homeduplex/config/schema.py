"""Settings, validated when the server starts so that mistakes show up before the first conversation.

Backends are chosen by protocol (`type: wyoming`, `type: openai`), not by product: any engine that speaks the
protocol works. `ollama` is the one exception, a dialect of the chat API kept for options that Ollama's
OpenAI-compatible endpoint ignores (see docs/design.md).
"""

from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, SecretStr, model_validator

from homeduplex.prompt import template


def _wyoming_uri(value: str) -> str:
    parts = urlsplit(value)
    if parts.scheme != "tcp" or not parts.hostname or parts.port is None:
        raise ValueError("expected tcp://host:port")
    return value


def _http_url(value: str) -> str:
    parts = urlsplit(value)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("expected an http:// or https:// URL")
    return value.rstrip("/")


def _timezone(value: str | None) -> str | None:
    if value is not None:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError):
            raise ValueError(f"unknown time zone {value!r} (expected a name such as Europe/Paris)") from None
    return value


WyomingUri = Annotated[str, AfterValidator(_wyoming_uri)]
HttpUrl = Annotated[str, AfterValidator(_http_url)]
Port = Annotated[int, Field(ge=1, le=65535)]
Seconds = Annotated[float, Field(gt=0)]
Slots = Annotated[int, Field(ge=1)]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ServerSettings(Model):
    host: str = "0.0.0.0"
    port: Port = 8770
    # Clients that can't add `?room=` to their endpoint can be told apart by the port they connect to.
    extra_ports: dict[Port, str] = {}
    # When set, clients must send `Authorization: Bearer <key>`.
    api_key: SecretStr | None = None


class WyomingSTT(Model):
    type: Literal["wyoming"]
    uri: WyomingUri
    language: str | None = None
    sample_rate: int = 16000
    timeout_s: Seconds = 15
    slots: Slots = 1


class OpenAISTT(Model):
    """`POST /audio/transcriptions` on any OpenAI-compatible server."""

    type: Literal["openai"]
    url: HttpUrl
    model: str
    api_key: SecretStr | None = None
    language: str | None = None
    timeout_s: Seconds = 15
    slots: Slots = 1


class TTSCommon(Model):
    # The engine's own voice name; None uses the engine's default.
    voice: str | None = None
    # Voice names clients send (they often only know OpenAI's, such as "marin") to the engine's names.
    voice_map: dict[str, str] = {}
    # A TTS server that stops answering must cost one sentence, not the conversation.
    chunk_timeout_s: Seconds = 10
    slots: Slots = 1


class WyomingTTS(TTSCommon):
    type: Literal["wyoming"]
    uri: WyomingUri


class OpenAITTS(TTSCommon):
    """`POST /audio/speech` with raw PCM output on any OpenAI-compatible server."""

    type: Literal["openai"]
    url: HttpUrl
    model: str
    api_key: SecretStr | None = None
    # Rate of the PCM the server returns (24000 for OpenAI and most compatible servers).
    sample_rate: int = 24000


class LLMCommon(Model):
    model: str
    slots: Slots = 1
    first_token_timeout_s: Seconds = 30
    # Between two pieces of a streaming answer.
    idle_timeout_s: Seconds = 30


class OpenAIChat(LLMCommon):
    """`POST /chat/completions` (streaming, tools) on any OpenAI-compatible server."""

    type: Literal["openai"]
    url: HttpUrl
    api_key: SecretStr | None = None
    # Added to every request as is, for server-specific parameters.
    extra_body: dict[str, Any] = {}


class OllamaChat(LLMCommon):
    """Ollama's native `/api/chat`. Its options must match every other user of the same Ollama server, or it
    reloads the model and everyone's prompt cache is lost."""

    type: Literal["ollama"]
    url: HttpUrl
    options: dict[str, Any] = {}
    keep_alive: int | str | None = None
    think: bool | None = None


STTSettings = Annotated[WyomingSTT | OpenAISTT, Field(discriminator="type")]
TTSSettings = Annotated[WyomingTTS | OpenAITTS, Field(discriminator="type")]
LLMSettings = Annotated[OpenAIChat | OllamaChat, Field(discriminator="type")]


class ConversationSettings(Model):
    # History sent to the model, in user turns. Replayed sessions pile up otherwise.
    max_turns: Annotated[int, Field(ge=1)] = 10
    # End-of-speech silence never goes below this, whatever the client asks: 500 ms cut sentences in two.
    min_silence_ms: Annotated[int, Field(ge=100)] = 800
    # Drop a "user" transcript that repeats what the assistant just said (speaker echo the client let through).
    drop_echo_transcripts: bool = True


class VADSettings(Model):
    type: Literal["energy"] = "energy"
    # Utterances whose loudest 20 ms frame stays below this mean absolute level (0-32767, the "level max" in the
    # logs) are not transcribed or answered. 0: off. Speech-to-text tends to hear words in near-silence.
    min_speech_level: Annotated[int, Field(ge=0, le=32767)] = 0


class PromptSettings(Model):
    """The system message is `preamble`, then the client's instructions, then `context`: from the most shared to the
    least, so model servers can reuse their work on the unchanged start of the prompt.

    `preamble` is plain text, the same for everyone. `context` is a template rendered for every answer, for what
    depends on the room or the day: `{room}`, the room's fields and `{now:<strftime>}`. Keep the time of day out of
    it (it would change the prompt every minute): give the model a time tool instead.
    """

    preamble: str = ""
    preamble_file: Path | None = None
    context: str = ""
    # IANA name such as "Europe/Paris"; None uses the system time zone.
    timezone: Annotated[str | None, AfterValidator(_timezone)] = None

    @model_validator(mode="after")
    def _one_preamble(self) -> "PromptSettings":
        if self.preamble and self.preamble_file:
            raise ValueError("set preamble or preamble_file, not both")
        return self


class Settings(Model):
    server: ServerSettings = ServerSettings()
    stt: STTSettings
    tts: TTSSettings
    llm: LLMSettings
    conversation: ConversationSettings = ConversationSettings()
    vad: VADSettings = VADSettings()
    prompt: PromptSettings = PromptSettings()
    # Room id → free-form fields usable in prompt templates, e.g. {office: {area: Office}}.
    rooms: dict[str, dict[str, str]] = {}
    default_room: str | None = None

    @model_validator(mode="after")
    def _rooms_exist(self) -> "Settings":
        if self.default_room is not None and self.default_room not in self.rooms:
            raise ValueError(f"default_room {self.default_room!r} is not in rooms")
        for port, room in self.server.extra_ports.items():
            if room not in self.rooms:
                raise ValueError(f"server.extra_ports: port {port} names room {room!r}, which is not in rooms")
            if port == self.server.port:
                raise ValueError(f"server.extra_ports: port {port} is already server.port")
        return self

    @model_validator(mode="after")
    def _templates_resolve(self) -> "Settings":
        try:
            template.check(self.prompt.context, self.template_variables())
        except template.TemplateError as e:
            raise ValueError(f"prompt.context: {e}") from None
        return self

    def warnings(self) -> list[str]:
        """Valid but probably unintended settings."""
        out: list[str] = []
        if codes := template.time_of_day_codes(self.prompt.context):
            out.append(
                f"prompt.context uses the time of day ({', '.join(codes)}): the prompt then changes every minute and "
                "the model server re-reads all of it on every answer. Give the model a time tool instead (Home "
                "Assistant's MCP server has GetDateTime)."
            )
        return out

    def template_variables(self) -> set[str]:
        """Built-in names, plus room fields that every room defines (any room can be the one talking)."""
        rooms = list(self.rooms.values())
        shared = set(rooms[0]).intersection(*rooms[1:]) if rooms else set()
        return set(template.BUILTIN_VARIABLES) | shared
