# Protocol subset

homeduplex implements the part of the OpenAI Realtime WebSocket protocol that real clients use. The reference client
is Kiosk Satellite 2026.10.x (`app/lib/managers/voice/realtime/openai_realtime_backend.dart` and
`realtime_session.dart` in jxlarrea/kiosk-satellite). Keep this file in step with what clients actually send.

## Connection

- URL: `ws[s]://host:port/<any path>?model=<name>[&room=<id>]`. The client adds `model` unless the endpoint already
  has one and keeps other query parameters.
- Header `Authorization: Bearer <key>` when the user set a key; may be ignored or checked.
- With a custom endpoint, Kiosk Satellite does not fetch model or voice lists; it uses fixed names (`gpt-realtime`,
  `marin`, ...). Map or ignore them.
- Validation ("Save & Validate"): the client connects and sends `session.update`; it is ready on `session.updated`,
  or 2 s after `session.created` if nothing else comes.

## Audio

PCM16 mono, 24 kHz, base64 in JSON, both directions.

## Client → server

| Event | Meaning |
|---|---|
| `session.update` | Instructions, tools (`{type: "function", name, description, parameters}`), audio formats, turn detection. OpenAI shape nests `audio.input.turn_detection` with `create_response: false` and `interrupt_response: false`; the xAI shape is flatter. |
| `input_audio_buffer.append` | Microphone audio. |
| `input_audio_buffer.clear` | Drop pending audio. |
| `response.create` | Start an answer. With `create_response: false` the **client** decides the turn: it sends this after `speech_stopped` when it judges the speech was the user. |
| `response.cancel` | Stop the current answer (interruption, or before a new `response.create`). |
| `conversation.item.create` | `function_call_output {call_id, output}` after running a tool; also **replayed history** (Kiosk Satellite 2026.10.3+) as OpenAI-shaped messages whose `content` is a list of parts: `{type: input_text|text, text}` or `{type: output_audio|audio|output_text, transcript|text}`. |
| `conversation.item.delete` | The client judged the last speech not to be the user (echo, noise): forget that item. |
| `conversation.item.truncate` | `{item_id, content_index, audio_end_ms}`: the user interrupted; keep only what was played. |

## Server → client

| Event | Used for |
|---|---|
| `session.created`, `session.updated` | Readiness. |
| `error` | `{error: {code, message}}`. Benign ones (cancel with nothing active, truncate past the end) are ignored by the client once ready. |
| `input_audio_buffer.speech_started` / `speech_stopped` | Turn-taking; `speech_stopped` carries `item_id` (the client may delete it). |
| `conversation.item.input_audio_transcription.delta` / `.completed` | User transcript (shown on screen). |
| `response.created` | An answer started. |
| `response.output_audio.delta` | Audio chunks `{item_id, delta}`; `item_id` is what `truncate` refers to. |
| `response.output_audio_transcript.delta` / `.done` | Answer text. |
| `response.function_call_arguments.done` | `{call_id, name, arguments}`: the client runs the tool. |
| `response.done` | `{response: {status: completed|cancelled|failed, status_details}}`. After tool calls, the client sends the outputs and a new `response.create`. A failed answer must say why, not complete empty. |

homeduplex also sends the GA bookkeeping events that well-behaved clients may track and others ignore:
`input_audio_buffer.committed` and `.cleared`, `conversation.item.added` / `.done` (GA names; the beta name was
`conversation.item.created`), `conversation.item.deleted`, `conversation.item.truncated`,
`conversation.item.input_audio_transcription.failed`, `response.output_item.added` / `.done`,
`response.output_audio.done`.

Not supported yet, and refused with an `error` event rather than ignored: server-driven turns
(`create_response: true`, the OpenAI default when a client sends `turn_detection` without it), `turn_detection: null`,
and audio formats other than 24 kHz PCM16. Unknown client events are ignored.

Kiosk Satellite's **xAI Grok** provider asks for server-driven turns: it sends `turn_detection` without
`create_response`, since xAI's server always keeps the turns (and Kiosk mutes its microphone while an answer plays).
Use its **OpenAI** provider with homeduplex until server-driven turns exist.

"Save & Validate" in Kiosk Satellite opens a session, waits for `session.updated` and disconnects: a connect and an
immediate disconnect in the log are a successful validation. It never reaches the backends; `homeduplex check-config
--connect`, and the start-up log, check those.

## Client behaviour worth knowing

- Interruption is decided on the client: speech over an answer counts as the user only if three ~85 ms microphone
  chunks average at least 1200 (of 32768), or ten times the room noise. A quiet raw microphone can make interruption
  impossible; that is a client/device issue, not a server one.
- Answers are played from a buffer; the server finishes sending long before playback ends, so "answer in progress"
  on the server is not "playing" on the client.
