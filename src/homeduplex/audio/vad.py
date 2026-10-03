"""Voice activity detection: a per-frame detector (is this 20 ms frame voice?) and a segmenter that turns frames
into utterances (speech started, speech stopped with its audio).

The energy detector is the default: no model, and it works well behind a client's echo canceller. Another detector
(Silero) only has to answer `is_voiced`.
"""

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from homeduplex.audio.pcm import SAMPLE_WIDTH

FRAME_MS = 20


class VoiceDetector(Protocol):
    def is_voiced(self, frame: np.ndarray) -> bool:
        """`frame` is FRAME_MS of int16 samples, in stream order."""
        ...

    def reset(self) -> None: ...


class EnergyDetector:
    """Voice is a frame louder than the room's noise floor by `margin_db` (and above `floor_db`). The noise floor
    follows quiet frames, so a humming fridge raises it and a quiet room lowers it."""

    def __init__(self, margin_db: float = 12, floor_db: float = -50, initial_noise_db: float = -60) -> None:
        self.margin_db = margin_db
        self.floor_db = floor_db
        self._initial = initial_noise_db
        self.noise_db = initial_noise_db

    def is_voiced(self, frame: np.ndarray) -> bool:
        db = max(level_db(frame), -75.0)
        voiced = db > max(self.noise_db + self.margin_db, self.floor_db)
        if db < self.noise_db:
            rate = 0.1  # the room got quieter: follow quickly
        elif not voiced:
            rate = 0.05
        else:
            # Creep up even through "voice", so a steady new noise (a fan) is learnt in a few seconds instead of
            # counting as endless speech; far too slow to swallow a sentence.
            rate = 0.002
        self.noise_db += rate * (db - self.noise_db)
        return voiced

    def reset(self) -> None:
        self.noise_db = self._initial


def level_db(frame: np.ndarray) -> float:
    """RMS level in dBFS."""
    x = frame.astype(np.float32)
    rms = math.sqrt(float(np.mean(x * x))) if len(x) else 0.0
    return 20 * math.log10(rms / 32768 + 1e-9)


@dataclass(frozen=True)
class SpeechStarted:
    audio_start_ms: int


@dataclass(frozen=True)
class SpeechStopped:
    audio_end_ms: int
    audio: bytes
    # Mean absolute sample value of each voiced-segment frame; clients judge interruptions on this scale.
    levels: list[float] = field(repr=False)


SegmentEvent = SpeechStarted | SpeechStopped


class Segmenter:
    """Feeds frames to a detector and cuts utterances: speech starts after `start_ms` of consecutive voice (the
    utterance keeps `prefix_ms` before that) and stops after `silence_ms` without voice."""

    def __init__(
        self,
        detector: VoiceDetector,
        rate: int,
        silence_ms: int,
        prefix_ms: int = 300,
        start_ms: int = 80,
        max_utterance_ms: int = 30_000,
    ) -> None:
        self.detector = detector
        self.rate = rate
        self.silence_ms = silence_ms
        self.prefix_ms = prefix_ms
        self.start_frames = max(1, start_ms // FRAME_MS)
        self.max_utterance_ms = max_utterance_ms
        self._frame_bytes = rate * FRAME_MS // 1000 * SAMPLE_WIDTH
        self._pending = bytearray()
        self._prefix: deque[bytes] = deque(maxlen=max(1, prefix_ms // FRAME_MS))
        self._speech = bytearray()
        self._levels: list[float] = []
        self._in_speech = False
        self._voiced_run = 0
        self._quiet_ms = 0
        self.audio_ms = 0  # position in the client's input stream

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def feed(self, pcm: bytes) -> list[SegmentEvent]:
        out: list[SegmentEvent] = []
        self._pending += pcm
        size = self._frame_bytes
        while len(self._pending) >= size:
            frame = bytes(self._pending[:size])
            del self._pending[:size]
            event = self._frame(frame)
            if event is not None:
                out.append(event)
        return out

    def clear(self) -> None:
        """Drop buffered audio (input_audio_buffer.clear). The noise floor and stream position are kept."""
        self._pending.clear()
        self._prefix.clear()
        self._end_segment()

    def _frame(self, frame: bytes) -> SegmentEvent | None:
        self.audio_ms += FRAME_MS
        samples = np.frombuffer(frame, dtype=np.int16)
        voiced = self.detector.is_voiced(samples)
        if not self._in_speech:
            self._prefix.append(frame)
            self._voiced_run = self._voiced_run + 1 if voiced else 0
            if self._voiced_run < self.start_frames:
                return None
            self._in_speech = True
            self._quiet_ms = 0
            self._speech = bytearray(b"".join(self._prefix))
            self._levels = []
            self._prefix.clear()
            start = self.audio_ms - len(self._speech) // self._frame_bytes * FRAME_MS
            return SpeechStarted(audio_start_ms=max(0, start))
        self._speech += frame
        self._levels.append(float(np.mean(np.abs(samples.astype(np.int32)))))
        self._quiet_ms = 0 if voiced else self._quiet_ms + FRAME_MS
        too_long = len(self._speech) // self._frame_bytes * FRAME_MS >= self.max_utterance_ms
        if self._quiet_ms < self.silence_ms and not too_long:
            return None
        stopped = SpeechStopped(audio_end_ms=self.audio_ms, audio=bytes(self._speech), levels=self._levels)
        self._end_segment()
        return stopped

    def _end_segment(self) -> None:
        self._in_speech = False
        self._speech = bytearray()
        self._levels = []
        self._voiced_run = 0
        self._quiet_ms = 0
