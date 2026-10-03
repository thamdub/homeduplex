"""Synthetic microphone audio at the client rate: loud tones stand in for speech, faint noise for a quiet room."""

import numpy as np

RATE = 24000


def speech(ms: int, amplitude: int = 6000, freq: float = 220.0) -> bytes:
    t = np.arange(RATE * ms // 1000) / RATE
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.int16).tobytes()


def silence(ms: int, amplitude: int = 30, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    return rng.integers(-amplitude, amplitude + 1, RATE * ms // 1000, dtype=np.int16).tobytes()


def chunks(pcm: bytes, ms: int = 85) -> list[bytes]:
    """Split like a client's microphone callback (Kiosk Satellite sends ~85 ms chunks)."""
    size = RATE * ms // 1000 * 2
    return [pcm[i : i + size] for i in range(0, len(pcm), size)]
