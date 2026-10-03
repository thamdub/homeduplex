import pytest

from homeduplex.text.echo import is_echo
from homeduplex.text.sentences import SentenceSplitter
from homeduplex.text.speech import speakable


def split_stream(pieces: list[str]) -> list[str]:
    splitter = SentenceSplitter()
    out = [s for piece in pieces for s in splitter.feed(piece)]
    if rest := splitter.flush():
        out.append(rest)
    return out


def test_sentences_as_they_complete() -> None:
    splitter = SentenceSplitter()
    assert splitter.feed("The office light") == []
    assert splitter.feed(" is on. The kitch") == ["The office light is on."]
    assert splitter.feed("en is off!") == []  # no space yet: maybe "off!!" or "off!)"
    assert splitter.feed(" Anything else?") == ["The kitchen is off!"]
    assert splitter.flush() == "Anything else?"
    assert splitter.flush() == ""


@pytest.mark.parametrize(
    ("pieces", "expected"),
    [
        (["It is 21.6 degrees. Nice."], ["It is 21.6 degrees.", "Nice."]),
        (["Version 2.0.1 is out."], ["Version 2.0.1 is out."]),
        (['He said "hi." Then left.'], ['He said "hi."', "Then left."]),
        (["Wait... what?"], ["Wait...", "what?"]),
        (["Shopping:\n- milk\n- eggs"], ["Shopping:", "- milk", "- eggs"]),
        (["Steps:\n1. Open it.\n2. Close it."], ["Steps:", "1. Open it.", "2. Close it."]),
        (["One", ".", " ", "Two", "."], ["One.", "Two."]),
        (["\n\n  "], []),
    ],
)
def test_split(pieces: list[str], expected: list[str]) -> None:
    assert split_stream(pieces) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("**Bold** and _italic_ and `code`", "Bold and italic and code"),
        ("## Heading", "Heading"),
        ("- item one", "item one"),
        ("See [the docs](https://example.lan/x).", "See the docs."),
        ("Done! 🎉👍🏽", "Done!"),
        ("It's 21.6 °C, 50% humidity", "It's 21.6 °C, 50% humidity"),
        ("a  \n  b", "a b"),
    ],
)
def test_speakable(text: str, expected: str) -> None:
    assert speakable(text) == expected


@pytest.mark.parametrize(
    ("heard", "said", "expected"),
    [
        ("the office light is on", "The office light is on, and the kitchen light is off.", True),
        ("office light is on and the", "The office light is on, and the kitchen light is off.", True),
        ("The kitchen light's off.", "The kitchen light's off. Anything else?", True),
        ("turn off the kitchen light", "The office light is on, and the kitchen light is off.", False),
        ("stop", "Stop the music? Sure, stopping it.", False),  # short: a real interruption
        ("what's the weather", "", False),
        ("set a timer for ten minutes", "Do you want a timer for ten minutes?", True),  # timing decides (session)
    ],
)
def test_is_echo(heard: str, said: str, expected: bool) -> None:
    assert is_echo(heard, said) is expected
