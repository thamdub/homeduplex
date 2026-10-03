import pytest

from homeduplex.session.conversation import (
    Conversation,
    FunctionCall,
    FunctionCallOutput,
    InvalidItem,
    Message,
    Segment,
    flatten_content,
    item_from_client,
    to_wire,
)


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (None, ""),
        ("  plain ", "plain"),
        ([{"type": "input_text", "text": "Turn on"}, {"type": "input_text", "text": "the light"}], "Turn on the light"),
        ([{"type": "output_audio", "transcript": "Done."}], "Done."),
        ([{"type": "audio", "transcript": None}, {"type": "text", "text": "x"}], "x"),
        ([{"type": "output_text", "text": "a"}, "b", 3], "a b"),
        ({"type": "input_text", "text": "single"}, "single"),
    ],
)
def test_flatten_content(content: object, expected: str) -> None:
    assert flatten_content(content) == expected


def test_replayed_history_items() -> None:
    """Kiosk Satellite 2026.10.3+ replays earlier turns as OpenAI-shaped messages with content parts."""
    user = item_from_client(
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Is it raining?"}]}
    )
    assistant = item_from_client(
        {"type": "message", "role": "assistant", "content": [{"type": "output_audio", "transcript": "No."}]}
    )
    assert isinstance(user, Message) and user.text == "Is it raining?" and user.role == "user"
    assert isinstance(assistant, Message) and assistant.text == "No."
    assert user.id.startswith("item_")


def test_tool_items() -> None:
    output = item_from_client({"type": "function_call_output", "call_id": "c1", "output": "on"})
    assert output == FunctionCallOutput(output.id, "c1", "on")
    structured = item_from_client({"type": "function_call_output", "call_id": "c1", "output": {"state": "on"}})
    assert isinstance(structured, FunctionCallOutput) and structured.output == '{"state": "on"}'
    call = item_from_client({"type": "function_call", "call_id": "c2", "name": "HassTurnOn", "arguments": {"a": 1}})
    assert isinstance(call, FunctionCall) and call.arguments == '{"a": 1}'
    assert item_from_client({"id": "mine", "type": "message", "content": "x"}).id == "mine"


@pytest.mark.parametrize(
    "raw",
    [
        {"type": "function_call_output", "output": "x"},
        {"type": "function_call", "call_id": "c"},
        {"type": "message", "role": "robot"},
        {"type": "image"},
    ],
)
def test_invalid_items(raw: dict[str, object]) -> None:
    with pytest.raises(InvalidItem):
        item_from_client(raw)


def test_add_delete_get() -> None:
    conv = Conversation()
    assert conv.add(Message("a", "user")) is None
    assert conv.add(Message("b", "assistant")) == "a"
    assert conv.get("b") is not None
    assert conv.delete("a")
    assert not conv.delete("a")
    assert [i.id for i in conv.items] == ["b"]


def test_truncate_keeps_sentences_that_started() -> None:
    conv = Conversation()
    answer = Message("a", "assistant", "One. Two. Three.")
    answer.segments = [Segment(0, 900, "One."), Segment(900, 2000, "Two."), Segment(2000, 3000, "Three.")]
    conv.add(answer)
    assert conv.truncate("a", 1200)
    assert answer.text == "One. Two."
    assert not conv.truncate("missing", 10)
    conv.add(Message("u", "user", "hi"))
    assert not conv.truncate("u", 10)


def turn(conv: Conversation, n: int) -> None:
    conv.add(Message(f"u{n}", "user", f"q{n}"))
    conv.add(Message(f"a{n}", "assistant", f"r{n}"))


def first_user(conv: Conversation, max_turns: int) -> str:
    return conv.window(max_turns)[0].id


def test_window_moves_in_steps_so_the_prompt_start_stays_stable() -> None:
    conv = Conversation()
    starts = []
    for n in range(14):
        turn(conv, n)
        starts.append(first_user(conv, 4))
    # Up to 4 turns: from the start. At 5, jump to keep 2; stable until the window holds 4 again; and so on.
    # Up to 4 turns: from the start. At the 5th, jump to keep the last 2; that start then holds for 3 turns.
    assert starts == ["u0", "u0", "u0", "u0", "u3", "u3", "u3", "u6", "u6", "u6", "u9", "u9", "u9", "u12"]


def test_window_with_room_to_spare() -> None:
    conv = Conversation()
    for n in range(5):
        turn(conv, n)
    assert len(conv.window(10)) == 10


def test_window_survives_deleting_its_first_item() -> None:
    conv = Conversation()
    for n in range(5):
        turn(conv, n)
    assert first_user(conv, 4) == "u3"
    conv.delete("u3")  # the first visible item
    assert [i.id for i in conv.window(4)] == ["a3", "u4", "a4"]
    conv.delete("a2")  # the last hidden one
    assert [i.id for i in conv.window(4)] == ["a3", "u4", "a4"]
    for item_id in ("a3", "u4", "a4"):
        conv.delete(item_id)
    assert conv.window(4) == []  # hidden turns stay hidden
    turn(conv, 5)
    assert [i.id for i in conv.window(4)] == ["u5", "a5"]


def test_wire_shapes() -> None:
    spoken = Message("u", "user", transcribing=True)
    assert to_wire(spoken)["content"] == [{"type": "input_audio", "transcript": None}]
    spoken.transcribing, spoken.text = False, "hi"
    assert to_wire(spoken)["content"] == [{"type": "input_audio", "transcript": "hi"}]
    assert to_wire(Message("t", "user", "typed", typed=True))["content"] == [{"type": "input_text", "text": "typed"}]
    assert to_wire(Message("a", "assistant", "ok"))["content"] == [{"type": "output_audio", "transcript": "ok"}]
    call = to_wire(FunctionCall("f", "c", "GetLiveContext", "{}"))
    assert call == {
        "id": "f",
        "object": "realtime.item",
        "status": "completed",
        "type": "function_call",
        "call_id": "c",
        "name": "GetLiveContext",
        "arguments": "{}",
    }
    assert to_wire(FunctionCallOutput("o", "c", "x"))["type"] == "function_call_output"
