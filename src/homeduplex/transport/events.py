"""Server → client events, built in one place so their shapes are checked by one set of tests.

Shapes follow the OpenAI Realtime GA API, restricted to the fields clients read (docs/protocol.md).
"""

import uuid
from collections.abc import Awaitable, Callable
from typing import Any

Event = dict[str, Any]
Send = Callable[[Event], Awaitable[None]]


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


def _event(type_: str, **fields: Any) -> Event:
    return {"type": type_, "event_id": new_id("event"), **fields}


# Session


def session_created(session: dict[str, Any]) -> Event:
    return _event("session.created", session=session)


def session_updated(session: dict[str, Any]) -> Event:
    return _event("session.updated", session=session)


def error(code: str, message: str, *, client_event_id: str | None = None, kind: str = "invalid_request_error") -> Event:
    return _event(
        "error",
        error={"type": kind, "code": code, "message": message, "param": None, "event_id": client_event_id},
    )


# Input audio


def speech_started(item_id: str, audio_start_ms: int) -> Event:
    return _event("input_audio_buffer.speech_started", item_id=item_id, audio_start_ms=audio_start_ms)


def speech_stopped(item_id: str, audio_end_ms: int) -> Event:
    return _event("input_audio_buffer.speech_stopped", item_id=item_id, audio_end_ms=audio_end_ms)


def committed(item_id: str, previous_item_id: str | None) -> Event:
    return _event("input_audio_buffer.committed", item_id=item_id, previous_item_id=previous_item_id)


def cleared() -> Event:
    return _event("input_audio_buffer.cleared")


def transcription_completed(item_id: str, transcript: str) -> Event:
    return _event(
        "conversation.item.input_audio_transcription.completed",
        item_id=item_id,
        content_index=0,
        transcript=transcript,
    )


def transcription_failed(item_id: str, message: str) -> Event:
    return _event(
        "conversation.item.input_audio_transcription.failed",
        item_id=item_id,
        content_index=0,
        error={"type": "transcription_error", "code": "transcription_failed", "message": message},
    )


# Conversation


def item_added(item: dict[str, Any], previous_item_id: str | None) -> Event:
    return _event("conversation.item.added", item=item, previous_item_id=previous_item_id)


def item_done(item: dict[str, Any]) -> Event:
    return _event("conversation.item.done", item=item)


def item_deleted(item_id: str) -> Event:
    return _event("conversation.item.deleted", item_id=item_id)


def item_truncated(item_id: str, audio_end_ms: int) -> Event:
    return _event("conversation.item.truncated", item_id=item_id, content_index=0, audio_end_ms=audio_end_ms)


# Responses


def response(
    response_id: str, status: str, output: list[dict[str, Any]], details: dict[str, Any] | None
) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "realtime.response",
        "status": status,
        "status_details": details,
        "output": output,
        "output_modalities": ["audio"],
    }


def response_created(body: dict[str, Any]) -> Event:
    return _event("response.created", response=body)


def response_done(body: dict[str, Any]) -> Event:
    return _event("response.done", response=body)


def output_item_added(response_id: str, output_index: int, item: dict[str, Any]) -> Event:
    return _event("response.output_item.added", response_id=response_id, output_index=output_index, item=item)


def output_item_done(response_id: str, output_index: int, item: dict[str, Any]) -> Event:
    return _event("response.output_item.done", response_id=response_id, output_index=output_index, item=item)


def audio_delta(response_id: str, item_id: str, output_index: int, delta: str) -> Event:
    return _event(
        "response.output_audio.delta",
        response_id=response_id,
        item_id=item_id,
        output_index=output_index,
        content_index=0,
        delta=delta,
    )


def audio_done(response_id: str, item_id: str, output_index: int) -> Event:
    return _event(
        "response.output_audio.done",
        response_id=response_id,
        item_id=item_id,
        output_index=output_index,
        content_index=0,
    )


def transcript_delta(response_id: str, item_id: str, output_index: int, delta: str) -> Event:
    return _event(
        "response.output_audio_transcript.delta",
        response_id=response_id,
        item_id=item_id,
        output_index=output_index,
        content_index=0,
        delta=delta,
    )


def transcript_done(response_id: str, item_id: str, output_index: int, transcript: str) -> Event:
    return _event(
        "response.output_audio_transcript.done",
        response_id=response_id,
        item_id=item_id,
        output_index=output_index,
        content_index=0,
        transcript=transcript,
    )


def function_call_arguments_done(
    response_id: str, item_id: str, output_index: int, call_id: str, name: str, arguments: str
) -> Event:
    return _event(
        "response.function_call_arguments.done",
        response_id=response_id,
        item_id=item_id,
        output_index=output_index,
        call_id=call_id,
        name=name,
        arguments=arguments,
    )
