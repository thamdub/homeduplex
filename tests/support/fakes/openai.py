"""A stand-in OpenAI-compatible server (aiohttp, real localhost socket), scripted by the test. It also answers
Ollama's native `/api/chat`, which the `ollama` dialect uses."""

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

from aiohttp import web

# answer: normally. hang: never start answering. error: HTTP error. stall: start, then stop mid-answer.
# stream_error: start, then report an error inside the stream.
Behaviour = Literal["answer", "hang", "error", "stall", "stream_error"]

# One scripted model answer: text pieces, and tool calls as {"tool": name, "arguments": {...}}.
Reply = list[str | dict[str, Any]]


@dataclass
class FakeOpenAI:
    transcripts: list[str] = field(default_factory=lambda: ["hello there"])
    replies: list[Reply] = field(default_factory=list)
    default_reply: Reply = field(default_factory=lambda: ["Hello", " there."])
    behaviour: Behaviour = "answer"
    delay: float = 0.0
    piece_delay: float = 0.0
    error_status: int = 500
    speech_rate: int = 24000
    # Models Ollama's /api/tags lists; a key the server requires (None: any).
    ollama_models: list[str] = field(default_factory=lambda: ["m:latest"])
    required_key: str | None = None
    speech_chunks: int = 3
    speech_chunk_ms: int = 100
    # Recorded requests.
    transcriptions: list[dict[str, Any]] = field(default_factory=list)
    chats: list[dict[str, Any]] = field(default_factory=list)
    speeches: list[dict[str, Any]] = field(default_factory=list)
    authorization: list[str | None] = field(default_factory=list)
    # Requests the client abandoned before the answer was complete.
    cancelled: int = 0
    # Chat requests being answered now, and the most at once.
    active_chats: int = 0
    max_active_chats: int = 0
    port: int = 0

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    @property
    def ollama_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def _misbehave(self) -> web.Response | None:
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.behaviour == "hang":
            await asyncio.sleep(3600)
        if self.behaviour == "error":
            return web.Response(status=self.error_status, text='{"error": "scripted failure"}')
        return None

    @contextlib.asynccontextmanager
    async def _tracked(self) -> AsyncIterator[None]:
        """Counts requests the client gave up on (aiohttp cancels the handler, or writing fails)."""
        try:
            yield
        except (asyncio.CancelledError, ConnectionResetError):
            self.cancelled += 1
            raise

    async def models(self, request: web.Request) -> web.StreamResponse:
        if self.required_key and request.headers.get("Authorization") != f"Bearer {self.required_key}":
            return web.json_response({"error": "bad key"}, status=401)
        if self.behaviour == "error":
            return web.Response(status=self.error_status)
        return web.json_response({"object": "list", "data": [{"id": "m", "object": "model"}]})

    async def tags(self, request: web.Request) -> web.StreamResponse:
        return web.json_response({"models": [{"name": n} for n in self.ollama_models]})

    async def transcribe(self, request: web.Request) -> web.StreamResponse:
        self.authorization.append(request.headers.get("Authorization"))
        form = await request.post()
        upload = form.get("file")
        audio = b""
        if isinstance(upload, web.FileField):
            with upload.file:
                audio = upload.file.read()
        self.transcriptions.append({"model": form.get("model"), "language": form.get("language"), "wav": audio})
        async with self._tracked():
            if (failure := await self._misbehave()) is not None:
                return failure
        text = self.transcripts.pop(0) if self.transcripts else ""
        return web.json_response({"text": text})

    async def chat(self, request: web.Request) -> web.StreamResponse:
        return await self._chat(request, ollama=False)

    async def ollama_chat(self, request: web.Request) -> web.StreamResponse:
        return await self._chat(request, ollama=True)

    async def _chat(self, request: web.Request, ollama: bool) -> web.StreamResponse:
        self.active_chats += 1
        self.max_active_chats = max(self.max_active_chats, self.active_chats)
        try:
            return await self._answer_chat(request, ollama)
        finally:
            self.active_chats -= 1

    async def _answer_chat(self, request: web.Request, ollama: bool) -> web.StreamResponse:
        self.authorization.append(request.headers.get("Authorization"))
        body = await request.json()
        self.chats.append(body)
        reply = self.replies.pop(0) if self.replies else self.default_reply
        async with self._tracked():
            if (failure := await self._misbehave()) is not None:
                return failure
            response = web.StreamResponse(
                headers={"Content-Type": "application/x-ndjson" if ollama else "text/event-stream"}
            )
            await response.prepare(request)
            write = _ollama_writer(response) if ollama else _sse_writer(response)
            for n, piece in enumerate(reply):
                if self.piece_delay:
                    await asyncio.sleep(self.piece_delay)
                if n == 1 and self.behaviour == "stall":
                    await asyncio.sleep(3600)
                if n == 1 and self.behaviour == "stream_error":
                    await write({"error": {"message": "scripted stream failure"}})
                    return response
                await write(_piece(piece, n, ollama))
            await write(None)
            return response

    async def speech(self, request: web.Request) -> web.StreamResponse:
        self.authorization.append(request.headers.get("Authorization"))
        self.speeches.append(await request.json())
        async with self._tracked():
            if (failure := await self._misbehave()) is not None:
                return failure
            response = web.StreamResponse(headers={"Content-Type": "audio/pcm"})
            await response.prepare(request)
            samples = self.speech_rate * self.speech_chunk_ms // 1000
            for n in range(self.speech_chunks):
                if self.piece_delay:
                    await asyncio.sleep(self.piece_delay)
                if n == 1 and self.behaviour == "stall":
                    await asyncio.sleep(3600)
                await response.write((1000 + n).to_bytes(2, "little", signed=True) * samples)
            await response.write_eof()
            return response


def _piece(piece: str | dict[str, Any], index: int, ollama: bool) -> dict[str, Any]:
    if isinstance(piece, str):
        if ollama:
            return {"message": {"role": "assistant", "content": piece}, "done": False}
        return {"choices": [{"index": 0, "delta": {"content": piece}}]}
    if ollama:
        call = {"function": {"name": piece["tool"], "arguments": piece.get("arguments", {})}}
        return {"message": {"role": "assistant", "content": "", "tool_calls": [call]}, "done": False}
    # OpenAI servers stream a call in fragments: id and name first, then the arguments in two pieces.
    arguments = json.dumps(piece.get("arguments", {}))
    half = len(arguments) // 2
    deltas = [
        {"index": index, "id": f"call_{index}", "function": {"name": piece["tool"], "arguments": arguments[:half]}},
        {"index": index, "function": {"arguments": arguments[half:]}},
    ]
    return {"choices": [{"index": 0, "delta": {"tool_calls": [d]}} for d in deltas], "_split": True}


def _sse_writer(response: web.StreamResponse) -> Any:
    async def write(chunk: dict[str, Any] | None) -> None:
        if chunk is None:
            await response.write(b'data: {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}\n\n')
            await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
            return
        if chunk.pop("_split", False):
            for choice in chunk["choices"]:
                await response.write(f"data: {json.dumps({'choices': [choice]})}\n\n".encode())
            return
        await response.write(f"data: {json.dumps(chunk)}\n\n".encode())

    return write


def _ollama_writer(response: web.StreamResponse) -> Any:
    async def write(chunk: dict[str, Any] | None) -> None:
        if chunk is None:
            done = {"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop"}
            await response.write((json.dumps(done) + "\n").encode())
            await response.write_eof()
            return
        if "error" in chunk:
            chunk = {"error": chunk["error"]["message"]}
        await response.write((json.dumps(chunk) + "\n").encode())

    return write


@contextlib.asynccontextmanager
async def fake_openai(**kwargs: Any) -> AsyncIterator[FakeOpenAI]:
    fake = FakeOpenAI(**kwargs)
    app = web.Application()
    app.router.add_post("/v1/audio/transcriptions", fake.transcribe)
    app.router.add_post("/v1/chat/completions", fake.chat)
    app.router.add_post("/v1/audio/speech", fake.speech)
    app.router.add_post("/api/chat", fake.ollama_chat)
    app.router.add_get("/v1/models", fake.models)
    app.router.add_get("/api/tags", fake.tags)
    runner = web.AppRunner(app, handler_cancellation=True)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    fake.port = runner.addresses[0][1]
    try:
        yield fake
    finally:
        await runner.cleanup()
