"""PCM16 mono resampling with soxr (good quality, fast, releases the GIL)."""

import numpy as np
import soxr


def resample(pcm: bytes, from_rate: int, to_rate: int) -> bytes:
    """Whole buffer at once, e.g. an utterance for speech-to-text. Can take a few milliseconds per second of audio:
    call it through `asyncio.to_thread` for long buffers."""
    if from_rate == to_rate or not pcm:
        return pcm
    samples = np.frombuffer(pcm, dtype=np.int16)
    out = soxr.resample(samples, from_rate, to_rate)
    return np.asarray(out, dtype=np.int16).tobytes()


class StreamResampler:
    """Chunk by chunk, for speech synthesis arriving in pieces; keeps the filter state between chunks."""

    def __init__(self, from_rate: int, to_rate: int) -> None:
        self.from_rate = from_rate
        self.to_rate = to_rate
        self._stream = soxr.ResampleStream(from_rate, to_rate, 1, dtype="int16") if from_rate != to_rate else None

    def feed(self, pcm: bytes, last: bool = False) -> bytes:
        if self._stream is None:
            return pcm
        out = self._stream.resample_chunk(np.frombuffer(pcm, dtype=np.int16), last=last)
        return np.asarray(out, dtype=np.int16).tobytes()

    def flush(self) -> bytes:
        return self.feed(b"", last=True)
