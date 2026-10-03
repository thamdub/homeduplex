from datetime import datetime

from homeduplex.backends.base import ChatMessage, ToolCall
from homeduplex.config.schema import PromptSettings
from homeduplex.prompt.builder import PromptBuilder, history
from homeduplex.session.config import SessionConfig, Tool
from homeduplex.session.conversation import FunctionCall, FunctionCallOutput, Item, Message
from homeduplex.transport.room import Room

NOW = datetime(2026, 10, 2, 18, 30)
PROMPT = PromptSettings(preamble="House rules.", context="You are in the {area} ({room}). Today is {now:%A}.")
OFFICE = Room("office", {"area": "Office"})
KITCHEN = Room("kitchen", {"area": "Kitchen"})
CONFIG = SessionConfig(instructions="Client instructions.", tools=(Tool("T", "", {}),))


def test_system_message_order() -> None:
    system = PromptBuilder(PROMPT, OFFICE).system(CONFIG, NOW)
    assert system == "House rules.\n\nClient instructions.\n\nYou are in the Office (office). Today is Friday."


def test_rooms_share_the_prefix_and_differ_only_at_the_end() -> None:
    office = PromptBuilder(PROMPT, OFFICE).system(CONFIG, NOW)
    kitchen = PromptBuilder(PROMPT, KITCHEN).system(CONFIG, NOW)
    shared = "House rules.\n\nClient instructions.\n\n"
    assert office.startswith(shared) and kitchen.startswith(shared)


def test_system_message_changes_only_with_the_day() -> None:
    builder = PromptBuilder(PROMPT, OFFICE)
    assert builder.system(CONFIG, NOW) == builder.system(CONFIG, datetime(2026, 10, 2, 23, 59))
    assert builder.system(CONFIG, NOW) != builder.system(CONFIG, datetime(2026, 10, 3, 0, 0))


def test_empty_slots_are_left_out() -> None:
    builder = PromptBuilder(PromptSettings(), Room(None))
    assert builder.system(SessionConfig(), NOW) == ""
    request = builder.request(SessionConfig(), [Message("u", "user", "hi")], NOW)
    assert request.messages == [ChatMessage("user", "hi")]


def test_user_messages_carry_only_what_the_user_said() -> None:
    request = PromptBuilder(PROMPT, OFFICE).request(CONFIG, [Message("u", "user", "What day is it?")], NOW)
    assert request.messages[-1] == ChatMessage("user", "What day is it?")
    assert request.tools == CONFIG.tools


def test_history_groups_tool_calls_and_drops_orphans() -> None:
    items: list[Item] = [
        Message("u1", "user", "Is the light on?"),
        Message("a1", "assistant", "Let me check."),
        FunctionCall("f1", "c1", "GetLiveContext", "{}"),
        FunctionCall("f2", "c2", "GetDateTime", "{}"),
        FunctionCallOutput("o1", "c1", "on"),
        FunctionCallOutput("o2", "c2", "18:30"),
        FunctionCallOutput("o3", "unknown", "orphan"),
        Message("a2", "assistant", "It is on."),
        Message("u2", "user", ""),  # transcription failed or pending
        Message("u3", "user", "Thanks"),
    ]
    assert history(items) == [
        ChatMessage("user", "Is the light on?"),
        ChatMessage(
            "assistant",
            "Let me check.",
            (ToolCall("c1", "GetLiveContext", "{}"), ToolCall("c2", "GetDateTime", "{}")),
        ),
        ChatMessage("tool", "on", tool_call_id="c1", name="GetLiveContext"),
        ChatMessage("tool", "18:30", tool_call_id="c2", name="GetDateTime"),
        ChatMessage("assistant", "It is on."),
        ChatMessage("user", "Thanks"),
    ]


def test_tool_call_without_text_starts_an_assistant_message() -> None:
    items: list[Item] = [Message("u", "user", "x"), FunctionCall("f", "c", "T", "{}")]
    assert history(items)[-1] == ChatMessage("assistant", "", (ToolCall("c", "T", "{}"),))


def test_timezone() -> None:
    builder = PromptBuilder(PromptSettings(timezone="Asia/Tokyo"), Room(None))
    assert builder.now().utcoffset() is not None
    assert str(builder.now().tzinfo) == "Asia/Tokyo"


def test_echo_is_hidden_from_the_model() -> None:
    items: list[Item] = [
        Message("u1", "user", "Is the light on?"),
        Message("a1", "assistant", "The light is on."),
        Message("u2", "user", "the light is on", ignored="echo"),
    ]
    assert [m.content for m in history(items)] == ["Is the light on?", "The light is on."]
