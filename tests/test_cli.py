from pathlib import Path

import pytest

from homeduplex.cli import main

CONFIG = """
server: {extra_ports: {8771: office}}
stt: {type: wyoming, uri: "tcp://stt.example.lan:10300"}
tts: {type: openai, url: "http://tts.example.lan:8880/v1", model: kokoro, voice: af_heart}
llm: {type: ollama, url: "http://llm.example.lan:11434", model: m}
rooms: {office: {area: Office}, kitchen: {area: Kitchen}}
default_room: office
"""


def test_check_config_prints_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOMEDUPLEX_SERVER__API_KEY", "do-not-print")
    path = tmp_path / "c.yaml"
    path.write_text(CONFIG)
    assert main(["check-config", "-c", str(path)]) == 0
    out = capsys.readouterr().out
    assert "listen   0.0.0.0:8770 (+ 8771 → office)" in out
    assert "auth     API key required" in out
    assert "stt      wyoming tcp://stt.example.lan:10300" in out
    assert "tts      openai http://tts.example.lan:8880/v1, voice af_heart" in out
    assert "llm      ollama http://llm.example.lan:11434, model m" in out
    assert "rooms    office, kitchen (default office)" in out
    assert "do-not-print" not in out


def test_check_config_reports_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("server: {port: 0}\n")
    assert main(["check-config", "-c", str(path)]) == 1
    err = capsys.readouterr().err
    assert f"Configuration error in {path}:" in err
    assert "  - server.port:" in err
    assert "  - llm: Field required" in err
