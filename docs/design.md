# Design

Decisions made so far, and why. Change them here first when they change.

## Language: Python 3.12, asyncio

homeduplex is an orchestrator. Per conversation it moves about 48 KB/s of audio each way, runs voice detection on
20 ms frames, and forwards text and audio between services. The heavy computing (transcription, the language
model, speech synthesis) happens in other processes. asyncio handles hundreds of concurrent WebSocket sessions; a
home has a handful.

Python also matches the ecosystem: the Wyoming protocol's reference library and Home Assistant are Python, so are
the likely contributors, and the audio/ML libraries (numpy, onnxruntime for Silero VAD) are mature.

CPU work (voice detection, resampling) must not block the event loop: run it in a thread pool (onnxruntime releases
the GIL) or keep it vectorised.

**Checked:** a load test (`tests/load`, `uv run pytest -m load -s`) runs 10 simulated clients streaming microphone
audio in real time, with full answer cycles, against stand-in backends. On an Apple M4: handling an audio
event takes 0.1 ms (median) and 0.5 ms (p99), the event loop runs at most ~12 ms late, and the whole test (clients
and fakes included) uses about 11% of one core. Python is not the bottleneck; the threshold to revisit the language
was "a few milliseconds per frame".

## Where concurrency really breaks

Not in the relay. In the shared backends:

- **The language model.** A single-slot server (for example Ollama with a hybrid model) serves one generation at a
  time, and every other conversation's prompt evicts its cache, so whoever speaks second can wait for a full prompt
  re-read (tens of seconds on a 4B model with a large tool list).
- **Text-to-speech.** Engines often synthesise one request at a time.

So these are first-class features, not afterthoughts:

1. **A scheduler for shared backends:** a fair queue per backend with the backend's number of slots (`slots` in the
   settings). A freed slot goes to the next conversation in round-robin order; text-to-speech takes a slot per
   sentence, so rooms' answers interleave sentence by sentence. Waiting doesn't count against a backend's timeouts;
   a conversation cancelled while waiting leaves the queue. Priorities later.
2. **End-to-end cancellation:** an interruption stops the model's generation and the synthesis, not just the
   audio sent to the client. Closing the HTTP stream or Wyoming socket is what tells the server to stop. httpcore
   loses the socket when cancelled while it sets up a connection (the server then keeps working until garbage
   collection), so HTTP requests run in their own task and a cancellation during connection setup waits for the
   setup (milliseconds) before cancelling (`backends/openai/http.py`, regression-tested).
3. **Prompt-cache-aware prompts:** a stable prefix shared by all conversations, per-room and per-day text as late as
   possible, nothing that changes every turn. Support multi-slot model servers (llama.cpp server, vLLM) where single-slot ones are the limit.
4. **Later:** streaming speech-to-text with partial transcripts.

## Architecture

```
cli.py, app.py  command line; builds backends and the scheduler from the settings
config/         YAML settings file + environment variables, validated at start-up
transport/      WebSocket server, OpenAI Realtime event (de)serialisation, auth, room resolution
session/        per-connection state: config, conversation items, turn policy, the answer pipeline
audio/          PCM utils, resampling, VAD (energy default; Silero optional)
text/           sentence splitting, markdown stripping, echo-transcript filter
backends/
  base.py       SpeechToText / LanguageModel / TextToSpeech interfaces
  wyoming/      Wyoming STT and TTS (any engine: faster-whisper, Parakeet, Piper, Kokoro, ...)
  openai/       OpenAI-compatible chat, transcription and speech (any server that speaks the API)
  ollama.py     Ollama's native chat API (see below)
scheduler/      fair queues for shared backends, cancellation propagation
prompt/         system prompt assembly from configured slots
```

Backends are interfaces so each can be swapped without touching the session logic.

### Backends target protocols, not products

An adapter speaks a protocol (Wyoming, the OpenAI HTTP API), so any engine behind it works: choosing Whisper or
Parakeet, Piper or Kokoro, llama.cpp or vLLM is configuration. Product names belong in docs and examples only.

The one product-specific adapter is Ollama's native `/api/chat`, a thin dialect of the chat adapter. It exists
because Ollama's OpenAI-compatible endpoint ignores `options.num_ctx`, `keep_alive` and `think`, and a request
whose options differ from the other users of the same Ollama server makes it reload the model and drop everyone's
prompt cache. Add another dialect only for a reason of that weight, and write the reason here.

### Turn-taking

The client chooses who decides that the user has finished speaking, with `create_response` in `session.update`:

- **Client-driven (`create_response: false`)**, as Kiosk Satellite does. The server reports `speech_started` /
  `speech_stopped` and waits: the client sends `response.create` when it judges the speech was the user,
  `conversation.item.delete` when it wasn't, and `response.cancel` to interrupt.
- **Server-driven (`create_response: true`)**, OpenAI's default. The server answers on its own after
  `speech_stopped` and, with `interrupt_response`, cancels its answer when the user talks over it.

Both run on the same session and answer pipeline; a turn policy object decides what happens on speech start and
stop. Client-driven comes first. Until server-driven exists, a client asking for it gets an
`unsupported_turn_detection` error event, never silence.

## Configuration

A YAML settings file (Home Assistant users write YAML daily), validated against a strict schema at start-up, plus
environment variables on top (`HOMEDUPLEX_<SECTION>__<KEY>`, for secrets and containers). No addresses or
site-specific text in code. `homeduplex check-config` validates a file; `examples/` has annotated ones.

- Backends: `type` names the protocol (`wyoming`, `openai`, or the `ollama` dialect), then its endpoint, model,
  voice and pass-through options.
- Rooms: picked by a query parameter on the endpoint URL (`ws://host:8770/v1/realtime?room=office`; the client
  keeps extra query parameters), or by a dedicated port for clients that can't set one. Each room has free-form
  fields (`area`, `media_player`, ...) that prompt templates can use.
- Prompt: the server builds nothing house-specific in. The system message is `preamble` (plain text, the same for
  everyone) → the client's instructions → `context`, ordered from most to least shared so model servers can reuse
  their work on the unchanged start. `context` is a template rendered for every answer, with `{room}`, the room's
  fields and `{now:<strftime>}`; unknown names are rejected when the settings are loaded. It is meant for what
  changes by room or by day: the date changes the prompt once a day, which costs one re-read. The time of day does
  not belong in the prompt (it would change it every minute; settings warn about it): the model asks a tool, as
  Home Assistant does (its agents skip the time line when a `GetDateTime` tool is present). An earlier design
  stamped each user turn with the time to keep the prompt stable; the repeated stamps read as noise to the model.
  Kiosk Satellite 2026.10.3+ appends the device's name and Home Assistant area to its own instructions, so with it
  `context` only needs what the client doesn't say (the room's media player, the date). Anything richer (a
  calendar, holidays, house rules) is the user's text, or later a context provider or a server-side tool.
- History: at most `conversation.max_turns` user turns go to the model. Past that, the oldest are dropped in one
  step down to half, and the start then stays put until the limit is reached again. Dropping one turn per turn
  changed the prompt right after the system message every time; on a real house that took the first word from
  about 2 s to 5-7 s once a conversation passed ten turns.

## Answers

The model's text is cut into sentences as it streams (`text/sentences.py`: punctuation followed by a space, or a
line break), cleaned of markdown and emoji, and each sentence is synthesised and sent while the model keeps
writing. Audio goes out in pieces of at most 200 ms, resampled to the client's 24 kHz. Each sentence's position in
the sent audio is recorded, so `conversation.item.truncate` keeps exactly the sentences that had started playing.

- A failed or stalled speech request costs that sentence (after `chunk_timeout_s`), not the answer.
- A model failure, or an empty answer, ends as `failed` with the reason: never an empty `completed`.
- Function calls are returned to the client, which runs them and sends the results with a new `response.create`.
- **Echo:** a transcript is dropped from the model's view when it started while the previous answer was still
  playing (or within a second after) *and* at least 80% of its words (three or more) appear, in order, in what the
  answer said. The client still sees it. If it was the only new input, the answer ends as `cancelled` (reason
  `echo`). The timing condition is what keeps a user repeating the assistant's words later from being dropped.
  `conversation.drop_echo_transcripts: false` turns it off.

## Voice activity detection

Energy-based with an adaptive noise floor as the default (no model, works well behind a client-side echo
canceller); Silero VAD later, as an option for noisy rooms. The floor follows quiet frames quickly and creeps up
slowly even through "voice" (a new steady noise such as a fan is learnt in about 5 s instead of counting as endless
speech; continuous speech stays voiced for well over 10 s). End-of-speech silence: at least 800 ms regardless of
what the client asks (500 ms cut sentences in two in practice). Utterances are capped at 30 s.

`vad.min_speech_level` (off by default) ignores utterances whose loudest frame stays below a level: speech-to-text
invents words ("Thank you.") in near-silence, and the model then answers them. They are not transcribed, the client
gets an empty transcript, and an answer requested with nothing else new ends as `cancelled` (reason `quiet`). Every
utterance's level is logged, which is how to pick the value for a room.

## Testing

- **Protocol test suite (`tests/protocol`):** a fake client that sends the exact event shapes real clients send
  (Kiosk Satellite's `session.update`, ~85 ms audio chunks, history replay), against the real server and stand-in
  backends on real sockets. Every client bug found in the field becomes a case.
- **Backend contract tests (`tests/backends`):** one suite every adapter passes against its fake server
  (`tests/support/fakes`): streaming, sample rate, timeouts (silent and stalled servers), cancellation closing the
  connection, errors, unreachable servers. A new adapter is done when it passes.
- Unit tests: settings, prompt templates and assembly, sentence splitting, speech cleanup, echo matching, content
  flattening, VAD and resampling on generated signals, the fair queue.
- Warnings are errors, and garbage is collected after each test, so leaked sockets fail the test that leaked them.
- Load test: the concurrency check above.
- CI on GitHub Actions (Python 3.12 and 3.13); no secrets needed.

## Privacy

At the default INFO level, logs carry timings, levels and events, never conversation text. DEBUG adds what was
heard, what was said, tool calls and client-added items, for diagnosing answers.

## Packaging

- A Docker image on GitHub Container Registry (`ghcr.io/thamdub/homeduplex`, amd64 and arm64), running as a fixed
  non-root UID with a read-only root filesystem.
- A generic Helm chart in `charts/homeduplex`, published as an OCI chart next to the image. Settings go in a free-form
  `config:` value; secrets come from an existing Secret as environment variables. Nothing about any particular
  cluster or house belongs in it: a deployment's own values live with that deployment.
- Both are published by `.github/workflows/release.yml` when a `v<version>` tag matching `__version__` is pushed,
  on GitHub's runners only (no self-hosted runner touches this public repository). CI builds the image and checks
  the chart on every push.
- A macOS LaunchAgent example. A Home Assistant add-on is a follow-up.

## License: Apache-2.0

Permissive with an explicit patent grant, and the same license as Home Assistant, so code can move between
homeduplex and Home Assistant add-ons or integrations. All runtime dependencies are compatible; soxr is
LGPL-2.1-or-later, used unmodified as a separate library (see `THIRD_PARTY_NOTICES.md`; a published image must
keep that notice).

## Open decisions

- Repository visibility at creation: planned public.
