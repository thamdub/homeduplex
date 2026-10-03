"""One client connection: its settings, its conversation, and the work in progress (transcriptions, answers).

The session never touches the network itself: the transport gives it a `send` function and feeds it parsed client
events, so tests can drive it directly. Long work runs in tasks so that client events (a cancel, more audio) are
handled while it runs; `close()` cancels all of it.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from homeduplex.audio.pcm import CLIENT_RATE, decode_b64
from homeduplex.audio.vad import EnergyDetector, Segmenter, SpeechStarted, SpeechStopped
from homeduplex.backends.base import Audio, BackendError, LanguageModel, ModelRequest, SpeechToText, TextToSpeech
from homeduplex.config import Settings
from homeduplex.observability import SessionLog
from homeduplex.prompt.builder import PromptBuilder
from homeduplex.scheduler.fair import FairQueue
from homeduplex.scheduler.scheduled import ScheduledLanguageModel, ScheduledSpeechToText, ScheduledTextToSpeech
from homeduplex.session import conversation
from homeduplex.session.config import SessionConfig, UnsupportedSetting, apply_update
from homeduplex.session.conversation import Conversation, FunctionCallOutput, InvalidItem, Item, Message
from homeduplex.session.response import NoAnswer, Response
from homeduplex.session.turns import ClientTurns, TurnPolicy
from homeduplex.text.echo import is_echo
from homeduplex.transport import events
from homeduplex.transport.events import Send
from homeduplex.transport.room import Room

log = logging.getLogger(__name__)

# Speech that starts this long after the assistant's audio should have finished playing is not its echo.
ECHO_MARGIN_S = 1.0


@dataclass(frozen=True)
class Services:
    """Backends shared by all sessions."""

    stt: SpeechToText
    llm: LanguageModel
    tts: TextToSpeech
    stt_queue: FairQueue
    llm_queue: FairQueue
    tts_queue: FairQueue


Handler = Callable[[Mapping[str, Any]], Awaitable[None]]


class Session:
    def __init__(self, send: Send, settings: Settings, services: Services, room: Room, model: str) -> None:
        self.id = events.new_id("sess")
        self.room = room
        self.model = model
        self.config = SessionConfig()
        self.conversation = Conversation()
        self._send = send
        self._settings = settings
        # Every call to a shared backend waits its turn in that backend's fair queue.
        self._stt = ScheduledSpeechToText(services.stt, services.stt_queue, self.id)
        self._llm = ScheduledLanguageModel(services.llm, services.llm_queue, self.id)
        self._tts = ScheduledTextToSpeech(services.tts, services.tts_queue, self.id)
        self._log = SessionLog(log, self.id, room.id)
        self._turns: TurnPolicy = ClientTurns()
        self._segmenter = self._new_segmenter()
        self._speech_item: str | None = None
        self._transcriptions: dict[str, asyncio.Task[None]] = {}
        self._prompt = PromptBuilder(settings.prompt, room)
        self._response: asyncio.Task[None] | None = None
        self._last_response: Response | None = None
        self._speech_started_at: dict[str, float] = {}
        self._handlers: dict[str, Handler] = {
            "session.update": self._on_session_update,
            "input_audio_buffer.append": self._on_audio_append,
            "input_audio_buffer.clear": self._on_audio_clear,
            "conversation.item.create": self._on_item_create,
            "conversation.item.delete": self._on_item_delete,
            "conversation.item.truncate": self._on_item_truncate,
            "response.create": self._on_response_create,
            "response.cancel": self._on_response_cancel,
        }

    async def start(self) -> None:
        await self._send(events.session_created(self._wire_config()))

    async def handle(self, event: Mapping[str, Any]) -> None:
        kind = event.get("type")
        handler = self._handlers.get(kind) if isinstance(kind, str) else None
        if handler is None:
            self._log.debug("ignored client event %r", kind)
            return
        await handler(event)

    async def close(self) -> None:
        """The client is gone: stop everything this session started."""
        await self._cancel_response()
        tasks = list(self._transcriptions.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._transcriptions.clear()

    async def transcripts_ready(self) -> None:
        """Wait for pending transcriptions: an answer must not start before the user's words are in. (`wait`, not
        `gather`: cancelling the answer must not cancel the transcriptions it was waiting for.)"""
        pending = list(self._transcriptions.values())
        if pending:
            await asyncio.wait(pending)

    # Session

    async def _on_session_update(self, event: Mapping[str, Any]) -> None:
        session = event.get("session")
        if not isinstance(session, Mapping):
            await self._error("invalid_value", "session.update needs a session object", event)
            return
        try:
            self.config = apply_update(self.config, session, self._settings.conversation.min_silence_ms)
        except UnsupportedSetting as e:
            self._log.warning("session.update refused: %s", e)
            await self._error(e.code, str(e), event)
            return
        if self.config.turn_detection:
            self._segmenter.silence_ms = self.config.turn_detection.silence_duration_ms
        self._log.info(
            "session.update: %d chars of instructions, end of speech after %s ms, tools: %s",
            len(self.config.instructions),
            self._segmenter.silence_ms,
            ", ".join(t.name for t in self.config.tools) or "none",
        )
        await self._send(events.session_updated(self._wire_config()))

    # Input audio

    def _new_segmenter(self) -> Segmenter:
        detection = self.config.turn_detection
        assert detection is not None  # other modes are refused in apply_update
        return Segmenter(
            EnergyDetector(),
            CLIENT_RATE,
            silence_ms=max(detection.silence_duration_ms, self._settings.conversation.min_silence_ms),
            prefix_ms=detection.prefix_padding_ms,
        )

    async def _on_audio_append(self, event: Mapping[str, Any]) -> None:
        audio = event.get("audio")
        try:
            pcm = decode_b64(audio) if isinstance(audio, str) else None
        except ValueError:
            pcm = None
        if pcm is None:
            await self._error("invalid_value", "input_audio_buffer.append needs base64 audio", event)
            return
        for segment in self._segmenter.feed(pcm):
            if isinstance(segment, SpeechStarted):
                await self._speech_started(segment)
            else:
                await self._speech_stopped(segment)

    async def _on_audio_clear(self, event: Mapping[str, Any]) -> None:
        self._segmenter.clear()
        if self._speech_item is not None:
            self._speech_started_at.pop(self._speech_item, None)
        self._speech_item = None
        await self._send(events.cleared())

    async def _speech_started(self, segment: SpeechStarted) -> None:
        self._speech_item = item_id = events.new_id("item")
        self._speech_started_at[item_id] = asyncio.get_running_loop().time()
        self._log.info("speech started")
        await self._send(events.speech_started(item_id, segment.audio_start_ms))
        await self._turns.on_speech_started(item_id)

    async def _speech_stopped(self, segment: SpeechStopped) -> None:
        item_id = self._speech_item or events.new_id("item")
        self._speech_item = None
        levels = sorted(segment.levels, reverse=True)
        loudest = levels[0] if levels else 0
        self._log.info(
            "speech stopped: %.1f s, level max %.0f, 3rd loudest %.0f, frames >= 1200: %d",
            len(segment.audio) / 2 / CLIENT_RATE,
            loudest,
            levels[2] if len(levels) > 2 else 0,
            sum(v >= 1200 for v in levels),
        )
        await self._send(events.speech_stopped(item_id, segment.audio_end_ms))
        floor = self._settings.vad.min_speech_level
        too_quiet = loudest < floor
        item = Message(item_id, "user", transcribing=not too_quiet, ignored="quiet" if too_quiet else None)
        previous = self.conversation.add(item)
        await self._send(events.committed(item_id, previous))
        await self._send(events.item_added(conversation.to_wire(item), previous))
        if too_quiet:
            # Near-silence makes speech-to-text invent words; the client still gets a (empty) transcript.
            self._log.info("too quiet to be the user (level max %.0f < %d): not transcribed", loudest, floor)
            self._speech_started_at.pop(item_id, None)
            await self._send(events.transcription_completed(item_id, ""))
            await self._send(events.item_done(conversation.to_wire(item)))
        else:
            self._transcriptions[item_id] = asyncio.create_task(
                self._transcribe(item, Audio(segment.audio, CLIENT_RATE)), name=f"stt {item_id}"
            )
        await self._turns.on_speech_stopped(item_id)

    async def _transcribe(self, item: Message, audio: Audio) -> None:
        loop = asyncio.get_running_loop()
        started = loop.time()
        try:
            text = await self._stt.transcribe(audio)
        except BackendError as e:
            self._log.warning("transcription failed: %s", e)
            item.transcribing = False
            await self._send(events.transcription_failed(item.id, str(e)))
            return
        finally:
            self._transcriptions.pop(item.id, None)
            started_at = self._speech_started_at.pop(item.id, None)
        item.text, item.transcribing = text, False
        if self._sounds_like_echo(started_at, text):
            # The client still gets the transcript; the model never sees it (prompt/builder.py).
            self._log.info("dropped a transcript that repeats the answer being played (echo)")
            item.ignored = "echo"
        seconds = len(audio.pcm) / 2 / audio.rate
        self._log.info("transcribed %.1f s of speech in %.2f s", seconds, loop.time() - started)
        self._log.debug("user said %r", text)
        await self._send(events.transcription_completed(item.id, text))
        await self._send(events.item_done(conversation.to_wire(item)))

    # Conversation

    async def _on_item_create(self, event: Mapping[str, Any]) -> None:
        raw = event.get("item")
        try:
            if not isinstance(raw, Mapping):
                raise InvalidItem("conversation.item.create needs an item object")
            item = conversation.item_from_client(raw)
        except InvalidItem as e:
            await self._error("invalid_value", str(e), event)
            return
        if self.conversation.get(item.id) is not None:
            await self._error("duplicate_item_id", f"item {item.id} already exists", event)
            return
        previous = self.conversation.add(item)
        self._log.debug("client added %s", _describe(item))
        wire = conversation.to_wire(item)
        await self._send(events.item_added(wire, previous))
        await self._send(events.item_done(wire))

    async def _on_item_delete(self, event: Mapping[str, Any]) -> None:
        item_id = event.get("item_id")
        if not isinstance(item_id, str) or not self.conversation.delete(item_id):
            await self._error("item_not_found", f"no item {item_id!r} to delete", event)
            return
        task = self._transcriptions.pop(item_id, None)
        if task is not None:
            task.cancel()
        self._speech_started_at.pop(item_id, None)
        self._log.info("client deleted %s (judged not the user)", item_id)
        await self._send(events.item_deleted(item_id))

    async def _on_item_truncate(self, event: Mapping[str, Any]) -> None:
        item_id, end_ms = event.get("item_id"), event.get("audio_end_ms")
        if not isinstance(item_id, str) or not isinstance(end_ms, int):
            await self._error("invalid_value", "truncate needs item_id and audio_end_ms", event)
            return
        if not self.conversation.truncate(item_id, end_ms):
            await self._error("item_not_found", f"no assistant item {item_id!r} to truncate", event)
            return
        await self._send(events.item_truncated(item_id, end_ms))

    # Answers

    async def _on_response_create(self, event: Mapping[str, Any]) -> None:
        # Kiosk Satellite cancels before creating; a client that doesn't gets the same: the new answer wins.
        await self._cancel_response()
        response = Response(self._send, self._llm, self._tts, self.conversation, self._voice(), self._log)
        self._last_response = response
        self._response = asyncio.create_task(response.run(self._model_request), name=f"answer {response.id}")

    async def _on_response_cancel(self, event: Mapping[str, Any]) -> None:
        if not await self._cancel_response():
            # Benign: clients cancel defensively, and ignore this error once ready (docs/protocol.md).
            await self._error("response_cancel_not_active", "no answer in progress", event)

    async def _cancel_response(self) -> bool:
        task, self._response = self._response, None
        if task is None or task.done():
            return False
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return True

    async def _model_request(self) -> ModelRequest:
        await self.transcripts_ready()
        last = self.conversation.items[-1] if self.conversation.items else None
        if isinstance(last, Message) and last.ignored:
            raise NoAnswer(last.ignored)
        items = self.conversation.window(self._settings.conversation.max_turns)
        return self._prompt.request(self.config, items)

    def _sounds_like_echo(self, started: float | None, text: str) -> bool:
        previous = self._last_response
        if not self._settings.conversation.drop_echo_transcripts or started is None or previous is None:
            return False
        until = previous.playing_until()
        if until is None or started > until + ECHO_MARGIN_S:
            return False
        return is_echo(text, previous.spoken_text)

    def _voice(self) -> str | None:
        """The engine voice: the client's choice when the settings map it, else the configured one."""
        tts = self._settings.tts
        if self.config.voice and self.config.voice in tts.voice_map:
            return tts.voice_map[self.config.voice]
        return tts.voice

    # Helpers

    def _wire_config(self) -> dict[str, Any]:
        return self.config.to_wire(self.id, self.model)

    async def _error(self, code: str, message: str, event: Mapping[str, Any] | None = None) -> None:
        client_event_id = event.get("event_id") if event else None
        await self._send(
            events.error(code, message, client_event_id=client_event_id if isinstance(client_event_id, str) else None)
        )


def _describe(item: Item) -> str:
    """For DEBUG logs (they contain conversation text, so they are off by default)."""
    text = item.output if isinstance(item, FunctionCallOutput) else getattr(item, "text", "")
    short = text if len(text) <= 300 else text[:300] + "..."
    return f"{type(item).__name__} {getattr(item, 'role', '')} {short!r}".replace("  ", " ")
