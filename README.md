# homeduplex

A self-hosted voice server for full-duplex conversations at home. It speaks the OpenAI Realtime protocol, built
from your own speech-to-text, language model and text-to-speech: anything that speaks Wyoming or an
OpenAI-compatible API (Whisper or Parakeet, Piper or Kokoro, Ollama, llama.cpp, vLLM, ...).

> **Status: early development, not released.** The server works end to end against test backends; it has not yet
> served a real house. Client-driven turns only (Kiosk Satellite's mode); see [`docs/protocol.md`](docs/protocol.md)
> for what is supported. The design, and the reasons behind it, are in [`docs/design.md`](docs/design.md).

## Why

Realtime voice clients, such as [Kiosk Satellite](https://github.com/jxlarrea/kiosk-satellite) on wall tablets, can
hold a natural conversation: you talk over the answer to interrupt it, and it keeps the context between turns.
Today they only do that with cloud providers. homeduplex lets them point at a server on your own network instead,
running models you already have.

## How it works

```
client (24 kHz PCM16 over WebSocket, OpenAI Realtime events)
   │
   ▼
homeduplex ── voice activity detection ──► speech-to-text (Wyoming / OpenAI API)
   │                                           │ transcript
   │◄──────────── function calls ◄──── language model (Ollama / OpenAI-compatible)
   │                                           │ text, sentence by sentence
   ◄──── audio deltas ◄──────────────── text-to-speech (Wyoming / OpenAI API)
```

- The client runs the tools (for example Home Assistant through its MCP server); homeduplex only returns function
  calls and reads their results back.
- Answers are spoken sentence by sentence while the model is still writing.
- Interruptions cancel the answer end to end: playback, the model's generation and speech synthesis.
- Several clients can talk at once; shared backends (one model, one TTS engine) are scheduled fairly.

## Goals

- Work with real clients first: Kiosk Satellite 2026.10+ is the reference client.
- Run on modest home hardware next to the models (a Mac mini, a small server).
- Be honest about limits: latency is set by your models, and interrupting by voice depends on the client device's
  echo cancellation.

## Quick start

You need a speech-to-text server, a language model server and a text-to-speech server that homeduplex can reach,
for example the Wyoming services and Ollama many Home Assistant users already run.

```sh
git clone https://github.com/thamdub/homeduplex.git && cd homeduplex
uv sync
cp examples/wyoming-ollama.yaml homeduplex.yaml     # then edit addresses, model, rooms
uv run homeduplex check-config                       # validates and prints what it will use
uv run homeduplex serve                              # listens on 0.0.0.0:8770
```

Then point the client at it. In Kiosk Satellite, set the realtime endpoint to
`ws://<server>:8770/v1/realtime?room=office` (the `room` must be one from your settings; leave it out if you have
no rooms). Any path works; an API key is checked only if `server.api_key` is set.

## Configuration

One YAML file: [`examples/homeduplex.example.yaml`](examples/homeduplex.example.yaml) lists every option, and the
other files in [`examples/`](examples/) show common setups. Every setting can also come from the environment,
`HOMEDUPLEX_<SECTION>__<KEY>` (for example `HOMEDUPLEX_LLM__API_KEY`), and `HOMEDUPLEX_CONFIG` names the file.

Backends are chosen by protocol: `wyoming` or `openai` for speech-to-text and text-to-speech, `openai` (any
OpenAI-compatible chat server) or `ollama` (its native API, which honours context size and keep-alive) for the
model. A `context` template adds what depends on the room or the day, such as
`Today is {now:%A, %B %d}. You are in the {area}.`, with per-room fields. Clients that already tell the model where
they are (Kiosk Satellite 2026.10.3+ sends the device's name and Home Assistant area) only need it for what they
don't say, such as which media player is "here". Leave the time of day to a tool (Home Assistant's GetDateTime).

If your model server is shared (with Home Assistant's own voice assistant, say), use the same model options as
the other users, or the server may reload the model on every switch.

## Running it as a service

- **Docker:** `docker build -t homeduplex .`, then
  `docker run -p 8770:8770 -v ./homeduplex.yaml:/config/homeduplex.yaml:ro homeduplex`.
- **macOS:** a LaunchAgent example is in [`examples/macos/`](examples/macos/).
- `GET /healthz` answers `ok` for monitoring.

## Development

Python 3.12+, managed with [uv](https://docs.astral.sh/uv/).

```sh
uv sync                                         # create the environment
uv run pytest                                   # tests (protocol, backend contracts, units)
uv run ruff check && uv run ruff format --check # lint
uv run mypy src tests                           # types
uv run pytest -m load -s                        # load test: 10 real-time clients, prints timings
```

Tests run the real server against stand-in backends on local sockets; no model or speech server is needed.

## License

[Apache License 2.0](LICENSE). Third-party licenses: [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
