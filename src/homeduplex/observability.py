"""Logging setup. Conversation text (transcripts, answers) is only logged at DEBUG: it is private."""

import logging
from collections.abc import MutableMapping
from typing import Any

FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=level.upper(), format=FORMAT)
    for noisy in ("websockets", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class SessionLog(logging.LoggerAdapter[logging.Logger]):
    """Prefixes each line with the room and a short session id, so interleaved conversations can be told apart."""

    def __init__(self, logger: logging.Logger, session_id: str, room: str | None) -> None:
        super().__init__(logger, {"session": session_id, "room": room})
        self._prefix = f"[{room or '-'} {session_id[-6:]}]"

    def process(self, msg: Any, kwargs: MutableMapping[str, Any]) -> tuple[Any, MutableMapping[str, Any]]:
        return f"{self._prefix} {msg}", kwargs
