"""One answer: the model writes, finished sentences go to speech while it keeps writing, audio goes to the client.

Cancelling the task running `Response.run` stops everything end to end: the model's stream and the speech request
are closed (so those servers stop working on it), and the client gets one `response.done` with status `cancelled`.
What was already sent stays in the conversation, so `truncate` can cut it to what was actually played.
"""

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from homeduplex.audio.pcm import CLIENT_RATE, duration_ms, encode_b64
from homeduplex.audio.resample import StreamResampler
from homeduplex.backends.base import BackendError, LanguageModel, ModelRequest, TextToSpeech, ToolCall
from homeduplex.session import conversation
from homeduplex.session.conversation import Conversation, FunctionCall, Message, Segment
from homeduplex.text.sentences import SentenceSplitter
from homeduplex.text.speech import speakable
from homeduplex.transport import events
from homeduplex.transport.events import Send

# Audio is sent in pieces of at most this length, so playback can start early and truncation is precise.
MAX_DELTA_MS = 200


class NoAnswer(Exception):
    """Preparing found nothing to answer; the response ends as cancelled with this reason."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class Response:
    def __init__(
        self,
        send: Send,
        llm: LanguageModel,
        tts: TextToSpeech,
        conversation: Conversation,
        voice: str | None,
        log: logging.LoggerAdapter[logging.Logger],
    ) -> None:
        self.id = events.new_id("resp")
        self._send = send
        self._llm = llm
        self._tts = tts
        self._conversation = conversation
        self._voice = voice
        self._log = log
        self._output: list[dict[str, Any]] = []
        self._message: Message | None = None
        self._message_index = 0
        self._played_ms = 0.0
        self._timings: dict[str, float] = {}
        self._spoken: Message | None = None
        self._audio_started_at: float | None = None

    @property
    def spoken_text(self) -> str:
        """What this answer has said so far."""
        return self._spoken.text if self._spoken else ""

    def playing_until(self) -> float | None:
        """When the client will have played everything sent so far (loop time), assuming it plays as it receives;
        None if no audio was sent."""
        if self._audio_started_at is None:
            return None
        return self._audio_started_at + self._played_ms / 1000

    async def run(self, prepare: Callable[[], Awaitable[ModelRequest]]) -> None:
        """`prepare` builds the model request once the answer has started (it may wait for a transcript)."""
        loop = asyncio.get_running_loop()
        started = loop.time()
        await self._send(events.response_created(events.response(self.id, "in_progress", [], None)))
        try:
            request = await prepare()
            self._timings["ready"] = loop.time() - started
            calls = await self._generate(request, started)
            await self._finish_message()
            for call in calls:
                await self._function_call(call)
            if not self._output:
                raise BackendError("the language model returned an empty answer")
        except asyncio.CancelledError:
            await self._finish_message()
            await self._done("cancelled", {"type": "cancelled", "reason": "client_cancelled"})
            raise
        except NoAnswer as e:
            self._log.info("not answering: %s", e.reason)
            await self._done("cancelled", {"type": "cancelled", "reason": e.reason})
        except BackendError as e:
            self._log.warning("answer failed: %s", e)
            await self._finish_message()
            error = {"type": "server_error", "code": "backend_error", "message": str(e)}
            await self._done("failed", {"type": "failed", "error": error})
        else:
            await self._done("completed", None)

    async def _generate(self, request: ModelRequest, started: float) -> list[ToolCall]:
        loop = asyncio.get_running_loop()
        splitter = SentenceSplitter()
        sentences: asyncio.Queue[str | None] = asyncio.Queue()
        speaker = asyncio.create_task(self._speak(sentences, started), name=f"speak {self.id}")
        calls: list[ToolCall] = []
        try:
            async with contextlib.aclosing(self._llm.stream(request)) as stream:
                async for piece in stream:
                    self._timings.setdefault("first_token", loop.time() - started)
                    if isinstance(piece, ToolCall):
                        calls.append(piece)
                        continue
                    for sentence in splitter.feed(piece.text):
                        await self._queue_sentence(sentences, sentence)
            await self._queue_sentence(sentences, splitter.flush())
            sentences.put_nowait(None)
            await speaker
        finally:
            if not speaker.done():
                speaker.cancel()
                await asyncio.gather(speaker, return_exceptions=True)
        return calls

    async def _queue_sentence(self, queue: asyncio.Queue[str | None], sentence: str) -> None:
        text = speakable(sentence)
        if not text:
            return
        if self._message is None:
            await self._start_message()
        queue.put_nowait(text)

    async def _start_message(self) -> None:
        self._message = self._spoken = Message(events.new_id("item"), "assistant")
        self._message_index = len(self._output)
        previous = self._conversation.add(self._message)
        self._output.append({})  # its final shape is filled in when it is done
        wire = conversation.to_wire(self._message)
        await self._send(events.output_item_added(self.id, self._message_index, {**wire, "status": "in_progress"}))
        await self._send(events.item_added(wire, previous))

    async def _speak(self, sentences: asyncio.Queue[str | None], started: float) -> None:
        loop = asyncio.get_running_loop()
        while (text := await sentences.get()) is not None:
            message = self._message
            assert message is not None
            start = self._played_ms
            await self._send(events.transcript_delta(self.id, message.id, self._message_index, text + " "))
            resampler: StreamResampler | None = None
            try:
                try:
                    async with contextlib.aclosing(self._tts.synthesize(text, self._voice)) as audio:
                        async for chunk in audio:
                            if resampler is None or resampler.from_rate != chunk.rate:
                                resampler = StreamResampler(chunk.rate, CLIENT_RATE)
                            self._timings.setdefault("first_audio", loop.time() - started)
                            await self._audio(message, resampler.feed(chunk.pcm))
                except BackendError as e:
                    # Skip the rest of the sentence, keep the conversation going: a stuck speech server must cost
                    # one sentence, not the answer.
                    self._log.warning("speech failed, sentence skipped: %s", e)
                if resampler is not None:
                    await self._audio(message, resampler.flush())
            finally:
                # Also when cancelled mid-sentence: its start was sent, so truncation must know about it.
                message.segments.append(Segment(start, self._played_ms, text))
                message.text = " ".join(s.text for s in message.segments)

    async def _audio(self, message: Message, pcm: bytes) -> None:
        step = CLIENT_RATE * MAX_DELTA_MS // 1000 * 2
        for i in range(0, len(pcm), step):
            piece = pcm[i : i + step]
            if self._audio_started_at is None:
                self._audio_started_at = asyncio.get_running_loop().time()
            self._played_ms += duration_ms(piece)
            await self._send(events.audio_delta(self.id, message.id, self._message_index, encode_b64(piece)))

    async def _finish_message(self) -> None:
        message, self._message = self._message, None
        if message is None:
            return
        self._log.debug("said %r", message.text)
        wire = conversation.to_wire(message)
        self._output[self._message_index] = wire
        index = self._message_index
        await self._send(events.transcript_done(self.id, message.id, index, message.text))
        await self._send(events.audio_done(self.id, message.id, index))
        await self._send(events.output_item_done(self.id, index, wire))
        await self._send(events.item_done(wire))

    async def _function_call(self, call: ToolCall) -> None:
        self._log.debug("tool call %s(%s)", call.name, call.arguments)
        item = FunctionCall(events.new_id("item"), call.call_id, call.name, call.arguments)
        index = len(self._output)
        previous = self._conversation.add(item)
        wire = conversation.to_wire(item)
        self._output.append(wire)
        await self._send(events.output_item_added(self.id, index, {**wire, "status": "in_progress"}))
        await self._send(events.item_added(wire, previous))
        await self._send(
            events.function_call_arguments_done(self.id, item.id, index, call.call_id, call.name, call.arguments)
        )
        await self._send(events.output_item_done(self.id, index, wire))
        await self._send(events.item_done(wire))

    async def _done(self, status: str, details: dict[str, Any] | None) -> None:
        timings = ", ".join(f"{name} {seconds:.2f} s" for name, seconds in self._timings.items())
        self._log.info("answer %s: %s, %d output items", status, timings or "no timings", len(self._output))
        await self._send(events.response_done(events.response(self.id, status, self._output, details)))
