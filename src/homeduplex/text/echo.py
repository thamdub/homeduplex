"""Recognise the assistant's own voice coming back as "user" speech.

When a client's echo canceller lets the speaker through and the client accepts it as the user, the model is told
its own words as if the user said them, and small models then start apologising in a loop.
"""

import re
from difflib import SequenceMatcher

_WORD = re.compile(r"[\w']+")


def is_echo(transcript: str, assistant_text: str, min_words: int = 3, threshold: float = 0.8) -> bool:
    """Most of the transcript's words appear, in order, in what the assistant said. Short transcripts never
    count: "yes" or "stop" over an answer are real interruptions."""
    heard = _words(transcript)
    said = _words(assistant_text)
    if len(heard) < min_words or not said:
        return False
    matcher = SequenceMatcher(a=heard, b=said, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return matched / len(heard) >= threshold


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())
