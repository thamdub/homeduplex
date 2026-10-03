"""Talk to a running homeduplex (or any Realtime server) the way Kiosk Satellite does, with synthetic speech.

Speaks each question (synthesised by a Wyoming text-to-speech server) in real-time 85 ms chunks, asks for an answer
when the server reports the end of speech, answers tool calls with a canned result, and prints what was heard and
said with timings. Nothing is played aloud.

    uv run python scripts/fake_kiosk.py "ws://127.0.0.1:8770/v1/realtime?room=office" \\
        --tts tcp://127.0.0.1:10200 --voice en_US-lessac-medium "What time is it?" "Is the office light on?"
"""

import argparse
import asyncio
import base64
import contextlib
import json
import time
from typing import Any

from websockets.asyncio.client import connect

from homeduplex.audio.pcm import CLIENT_RATE
from homeduplex.audio.resample import resample
from homeduplex.backends.wyoming.tts import WyomingTextToSpeech
from homeduplex.config.schema import WyomingTTS

CHUNK_MS = 85
TOOL = {
    "type": "function",
    "name": "GetLiveContext",
    "description": "Provides real-time information about the CURRENT state, value, or mode of devices.",
    "parameters": {"type": "object", "properties": {}},
}


async def speech(text: str, uri: str, voice: str | None) -> bytes:
    tts = WyomingTextToSpeech(WyomingTTS(type="wyoming", uri=uri))
    parts: list[bytes] = []
    async with contextlib.aclosing(tts.synthesize(text, voice)) as stream:
        async for chunk in stream:
            parts.append(resample(chunk.pcm, chunk.rate, CLIENT_RATE))
    return b"".join(parts) + bytes(2 * CLIENT_RATE)  # then a second of silence


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url")
    parser.add_argument("questions", nargs="+")
    parser.add_argument("--tts", required=True, help="Wyoming TTS server for the questions, tcp://host:port")
    parser.add_argument("--voice")
    parser.add_argument("--tool-output", default="(no devices: this is a test client)")
    parser.add_argument("--key", help="API key, if the server wants one")
    args = parser.parse_args()

    headers = {"Authorization": f"Bearer {args.key}"} if args.key else None
    async with connect(args.url, additional_headers=headers, max_size=None) as ws:

        async def send(type_: str, **fields: Any) -> None:
            await ws.send(json.dumps({"type": type_, **fields}))

        async def recv() -> dict[str, Any]:
            event: dict[str, Any] = json.loads(await asyncio.wait_for(ws.recv(), 60))
            return event

        await send(
            "session.update",
            session={
                "type": "realtime",
                "instructions": "You are a voice assistant on a wall tablet. Keep answers short.",
                "tools": [TOOL],
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": CLIENT_RATE},
                        "turn_detection": {"type": "server_vad", "silence_duration_ms": 500, "create_response": False},
                    },
                    "output": {"format": {"type": "audio/pcm", "rate": CLIENT_RATE}, "voice": "marin"},
                },
            },
        )
        while (event := await recv())["type"] != "session.updated":
            if event["type"] == "error":
                raise SystemExit(f"error: {event['error']}")

        for question in args.questions:
            print(f"\n>>> {question}")
            pcm = await speech(question, args.tts, args.voice)

            async def talk(pcm: bytes = pcm) -> None:
                step = CLIENT_RATE * CHUNK_MS // 1000 * 2
                for i in range(0, len(pcm), step):
                    await send("input_audio_buffer.append", audio=base64.b64encode(pcm[i : i + step]).decode())
                    await asyncio.sleep(CHUNK_MS / 1000)

            talker = asyncio.create_task(talk())
            stopped = first_audio = 0.0
            audio_bytes = 0
            pending_tool = False
            while True:
                event = await recv()
                kind = event["type"]
                if kind == "input_audio_buffer.speech_stopped":
                    stopped = time.monotonic()
                    await send("response.create")
                elif kind == "conversation.item.input_audio_transcription.completed":
                    print(f"    heard: {event['transcript']!r} (+{time.monotonic() - stopped:.2f} s)")
                elif kind == "response.output_audio.delta":
                    if not first_audio:
                        first_audio = time.monotonic()
                        print(f"    first audio +{first_audio - stopped:.2f} s after the end of speech")
                    audio_bytes += len(base64.b64decode(event["delta"]))
                elif kind == "response.function_call_arguments.done":
                    print(f"    tool call {event['name']}({event['arguments']}) +{time.monotonic() - stopped:.2f} s")
                    item = {"type": "function_call_output", "call_id": event["call_id"], "output": args.tool_output}
                    await send("conversation.item.create", item=item)
                    pending_tool = True
                elif kind == "response.output_audio_transcript.done":
                    seconds = audio_bytes / 2 / CLIENT_RATE
                    print(f"    said: {event['transcript']!r} ({seconds:.1f} s of audio)")
                elif kind == "response.done":
                    status = event["response"]["status"]
                    if status != "completed":
                        print(f"    answer {status}: {event['response'].get('status_details')}")
                    if pending_tool and status == "completed":
                        pending_tool = False
                        await send("response.create")
                        continue
                    break
                elif kind == "error":
                    print(f"    error: {event['error']}")
            await talker


if __name__ == "__main__":
    asyncio.run(main())
