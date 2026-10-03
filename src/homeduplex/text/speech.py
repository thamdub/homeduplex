"""Make model output speakable: speech engines read markdown and emoji literally ("asterisk asterisk")."""

import re

_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MARKUP = re.compile(r"[*_`#>|~]+")
_BULLET = re.compile(r"^\s*[-•+]\s+", re.MULTILINE)
_SPACES = re.compile(r"\s+")
# Emoji and pictographs (with skin tones, variation selectors and joiners), but not symbols speech engines read
# well, such as ° % € ©.
_PICTOGRAPHS = re.compile("[\U0001f000-\U0001faff\u2600-\u27bf\u2b00-\u2bff\ufe0f\u200d]+")


def speakable(text: str) -> str:
    text = _LINK.sub(r"\1", text)
    text = _BULLET.sub("", text)
    text = _MARKUP.sub("", text)
    text = _PICTOGRAPHS.sub("", text)
    return _SPACES.sub(" ", text).strip()
