import base64

import numpy as np
import pytest
from support.audio import silence, speech

from homeduplex.audio.pcm import decode_b64, duration_ms, encode_b64
from homeduplex.audio.resample import StreamResampler, resample
from homeduplex.audio.vad import EnergyDetector, Segmenter, SegmentEvent, SpeechStarted, SpeechStopped, level_db

RATE = 24000


def segmenter(silence_ms: int = 800, prefix_ms: int = 300) -> Segmenter:
    return Segmenter(EnergyDetector(), RATE, silence_ms=silence_ms, prefix_ms=prefix_ms)


def test_one_utterance() -> None:
    seg = segmenter()
    events = seg.feed(silence(1000) + speech(1500) + silence(1000))
    assert len(events) == 2
    started, stopped = events
    assert isinstance(started, SpeechStarted) and isinstance(stopped, SpeechStopped)
    # Detection needs 80 ms of voice; the utterance keeps 300 ms of audio before that.
    assert 600 <= started.audio_start_ms <= 800
    # Ends 800 ms into the silence.
    assert stopped.audio_end_ms == pytest.approx(1000 + 1500 + 800, abs=40)
    assert duration_ms(stopped.audio) == pytest.approx(stopped.audio_end_ms - started.audio_start_ms, abs=40)
    assert stopped.levels and max(stopped.levels) > 3000


def test_chunking_does_not_matter() -> None:
    pcm = silence(500) + speech(700) + silence(900)
    whole = segmenter().feed(pcm)
    pieces: list[SegmentEvent] = []
    seg = segmenter()
    for i in range(0, len(pcm), 4082):  # odd sizes, splitting frames and samples
        pieces += seg.feed(pcm[i : i + 4082])
    assert [type(e) for e in pieces] == [type(e) for e in whole]
    assert isinstance(pieces[1], SpeechStopped) and isinstance(whole[1], SpeechStopped)
    assert pieces[1].audio_end_ms == whole[1].audio_end_ms


def test_short_pause_does_not_end_speech() -> None:
    """500 ms cut "play ... some music" in two in the prototype; the configured minimum is 800."""
    events = segmenter().feed(silence(500) + speech(600) + silence(600) + speech(600) + silence(1000))
    assert [type(e) for e in events] == [SpeechStarted, SpeechStopped]


def test_click_is_not_speech() -> None:
    assert segmenter().feed(silence(500) + speech(40) + silence(1000)) == []


def frame(amplitude: int) -> np.ndarray:
    return np.frombuffer(speech(20, amplitude=amplitude), dtype=np.int16)


def test_noise_floor_learns_a_new_steady_noise() -> None:
    """A fan turning on counts as voice at first, is learnt within seconds, and speech above it still counts."""
    detector = EnergyDetector()
    for _ in range(100):
        detector.is_voiced(frame(20))  # quiet room
    assert detector.is_voiced(frame(200))
    voiced = [detector.is_voiced(frame(200)) for _ in range(500)]  # 10 s of hum
    assert all(voiced[:150])  # not learnt in the first 3 s
    assert not any(voiced[-50:])
    assert detector.is_voiced(frame(6000))


def test_noise_floor_does_not_swallow_a_long_sentence() -> None:
    detector = EnergyDetector()
    for _ in range(100):
        detector.is_voiced(frame(20))
    assert all(detector.is_voiced(frame(3000)) for _ in range(500))  # 10 s of continuous speech


def test_noise_floor_follows_a_quieter_room_quickly() -> None:
    detector = EnergyDetector()
    for _ in range(500):
        detector.is_voiced(frame(200))
    for _ in range(50):  # 1 s
        detector.is_voiced(frame(20))
    assert detector.is_voiced(frame(150))


def test_max_utterance_length() -> None:
    seg = Segmenter(EnergyDetector(), RATE, silence_ms=800, max_utterance_ms=2000)
    events = seg.feed(silence(300) + speech(5000))
    assert isinstance(events[1], SpeechStopped)
    assert duration_ms(events[1].audio) <= 2000


def test_clear_drops_speech_in_progress() -> None:
    seg = segmenter()
    assert len(seg.feed(silence(300) + speech(500))) == 1
    assert seg.in_speech
    seg.clear()
    assert not seg.in_speech
    assert seg.feed(silence(1000)) == []


def test_level_db() -> None:
    full = np.full(480, 32767, dtype=np.int16)
    assert level_db(full) == pytest.approx(0, abs=0.01)
    assert level_db(np.zeros(480, dtype=np.int16)) < -150


def test_resample_lengths() -> None:
    pcm = speech(1000)
    assert len(resample(pcm, 24000, 16000)) == 16000 * 2
    assert resample(pcm, 24000, 24000) is pcm
    assert resample(b"", 24000, 16000) == b""


def test_stream_resampler_matches_whole_length() -> None:
    pcm = speech(1000, freq=440)
    stream = StreamResampler(22050, 24000)
    out = b"".join(stream.feed(pcm[i : i + 2000]) for i in range(0, len(pcm), 2000)) + stream.flush()
    expected = len(pcm) // 2 * 24000 / 22050
    assert len(out) // 2 == pytest.approx(expected, abs=2)
    assert StreamResampler(24000, 24000).feed(b"ab") == b"ab"


def test_base64() -> None:
    assert decode_b64(encode_b64(b"\x01\x02\x03\x04")) == b"\x01\x02\x03\x04"
    assert decode_b64(base64.b64encode(b"\x01\x02\x03").decode()) == b"\x01\x02"  # half sample dropped
    with pytest.raises(ValueError):
        decode_b64("not base64!")
