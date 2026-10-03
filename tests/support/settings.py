from typing import Any

from homeduplex.config import Settings

BASE: dict[str, Any] = {
    "stt": {"type": "wyoming", "uri": "tcp://stt.example.lan:10300"},
    "tts": {"type": "wyoming", "uri": "tcp://tts.example.lan:10200"},
    "llm": {"type": "openai", "url": "http://llm.example.lan:8080/v1", "model": "test-model"},
    "rooms": {"office": {"area": "Office"}, "kitchen": {"area": "Kitchen"}},
}


def make_settings(**sections: Any) -> Settings:
    """Valid settings with stand-in endpoints; keyword arguments replace whole top-level sections."""
    return Settings.model_validate({**BASE, **sections})
