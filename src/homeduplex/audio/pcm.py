"""PCM16 mono audio helpers. Clients send and receive 24 kHz; backends use whatever rate they report."""

import base64
import binascii

CLIENT_RATE = 24000
SAMPLE_WIDTH = 2


def decode_b64(data: str) -> bytes:
    """Base64 audio from a client event; an odd trailing byte is dropped (half a sample is noise)."""
    try:
        pcm = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("audio is not valid base64") from None
    return pcm[: len(pcm) - len(pcm) % SAMPLE_WIDTH]


def encode_b64(pcm: bytes) -> str:
    return base64.b64encode(pcm).decode("ascii")


def duration_ms(pcm: bytes, rate: int = CLIENT_RATE) -> float:
    return len(pcm) / SAMPLE_WIDTH / rate * 1000
