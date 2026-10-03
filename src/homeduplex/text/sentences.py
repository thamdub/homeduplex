"""Cut a streaming answer into sentences, so speech can start while the model is still writing."""

import re

# A sentence ends at . ! ? or … followed by whitespace (so "21.6" and "e.g.x" don't split), or at a line break
# (so a list is spoken item by item instead of waiting for the whole answer).
_END = re.compile(r"[.!?…]+[\"')\]]*\s+|\n+")
# "1. " starting a numbered list item is not the end of a sentence.
_LIST_MARKER = re.compile(r"^\s*\d+[.)]$")


class SentenceSplitter:
    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, text: str) -> list[str]:
        """Complete sentences found so far; the rest waits for more text."""
        self._buffer += text
        out: list[str] = []
        start = 0
        for match in _END.finditer(self._buffer):
            candidate = self._buffer[start : match.end()]
            if _LIST_MARKER.match(candidate.strip()) and "\n" not in match.group():
                continue
            if sentence := candidate.strip():
                out.append(sentence)
            start = match.end()
        self._buffer = self._buffer[start:]
        return out

    def flush(self) -> str:
        """Whatever is left at the end of the answer."""
        rest, self._buffer = self._buffer.strip(), ""
        return rest
