"""Backend checks at start-up and in `check-config --connect`: a wrong address or name shows before a conversation."""

import asyncio
import logging
import socket
from pathlib import Path
from typing import Any

import pytest
import yaml
from support.fakes.openai import fake_openai
from support.fakes.wyoming import fake_wyoming
from support.settings import make_settings

from homeduplex.app import report_backends
from homeduplex.backends.probe import probe_all
from homeduplex.cli import main


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def by_role(results: list[Any]) -> dict[str, Any]:
    return {r.role: r for r in results}


async def test_everything_answers() -> None:
    async with fake_wyoming() as stt, fake_wyoming(offers="tts", voice_names=["af_heart"]) as tts, fake_openai() as llm:
        settings = make_settings(
            stt={"type": "wyoming", "uri": stt.uri},
            tts={"type": "wyoming", "uri": tts.uri, "voice": "af_heart"},
            llm={"type": "ollama", "url": llm.ollama_url, "model": "m"},
        )
        results = by_role(await probe_all(settings))
    assert all(r.ok for r in results.values())
    assert results["speech-to-text"].detail == "fake-asr answers"
    assert results["text-to-speech"].detail == "fake-tts answers"
    assert results["language model"].detail == "Ollama answers, model m installed"


async def test_wrong_names_are_reported() -> None:
    async with fake_wyoming(offers="tts") as stt, fake_wyoming(offers="tts") as tts, fake_openai() as llm:
        settings = make_settings(
            stt={"type": "wyoming", "uri": stt.uri},  # a TTS server where speech-to-text should be
            tts={"type": "wyoming", "uri": tts.uri, "voice": "marin"},
            llm={"type": "ollama", "url": llm.ollama_url, "model": "llama9"},
        )
        results = by_role(await probe_all(settings))
    assert not any(r.ok for r in results.values())
    assert results["speech-to-text"].detail == "answers, but offers no transcription"
    assert results["text-to-speech"].detail == "fake-tts answers, but has no voice 'marin'"
    assert results["language model"].detail == "Ollama answers, but has no model 'llama9' (ollama pull?)"


async def test_unreachable_loopback_gets_the_container_hint() -> None:
    port = free_port()
    settings = make_settings(
        stt={"type": "wyoming", "uri": f"tcp://127.0.0.1:{port}"},
        tts={"type": "openai", "url": f"http://localhost:{port}/v1", "model": "t"},
        llm={"type": "ollama", "url": f"http://127.0.0.1:{port}", "model": "m"},
    )
    for result in await probe_all(settings, within=1):
        assert not result.ok
        assert "inside a container" in result.detail and "use the host's address" in result.detail


async def test_a_server_that_is_not_wyoming() -> None:
    async with fake_openai() as http:
        settings = make_settings(stt={"type": "wyoming", "uri": f"tcp://127.0.0.1:{http.port}"})
        result = by_role(await probe_all(settings, within=0.5))["speech-to-text"]
    assert not result.ok


async def test_openai_compatible_servers() -> None:
    async with fake_openai(required_key="right") as server:
        good = make_settings(llm={"type": "openai", "url": server.url, "model": "m", "api_key": "right"})
        bad = make_settings(llm={"type": "openai", "url": server.url, "model": "m", "api_key": "wrong"})
        assert by_role(await probe_all(good))["language model"].ok
        refused = by_role(await probe_all(bad))["language model"]
        server.required_key, server.behaviour = None, "error"
        failing = by_role(await probe_all(good))["language model"]
    assert refused.detail == "answers, but refused the API key (HTTP 401)"
    assert failing.detail == "answers with an error (HTTP 500)"


async def test_startup_logs_problems(caplog: pytest.LogCaptureFixture) -> None:
    async with fake_wyoming() as stt, fake_wyoming(offers="tts") as tts:
        settings = make_settings(
            stt={"type": "wyoming", "uri": stt.uri},
            tts={"type": "wyoming", "uri": tts.uri},
            llm={"type": "ollama", "url": f"http://127.0.0.1:{free_port()}", "model": "m"},
        )
        with caplog.at_level(logging.INFO, logger="homeduplex"):
            await report_backends(settings)
    levels = {r.getMessage().split(" ")[0]: r.levelname for r in caplog.records}
    assert levels == {"speech-to-text": "INFO", "text-to-speech": "INFO", "language": "WARNING"}


def write_config(tmp_path: Path, **sections: Any) -> str:
    path = tmp_path / "homeduplex.yaml"
    path.write_text(yaml.safe_dump(sections))
    return str(path)


async def test_check_config_connect_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    async with fake_wyoming() as stt, fake_wyoming(offers="tts") as tts, fake_openai() as llm:
        ok = write_config(
            tmp_path,
            stt={"type": "wyoming", "uri": stt.uri},
            tts={"type": "wyoming", "uri": tts.uri},
            llm={"type": "ollama", "url": llm.ollama_url, "model": "m"},
        )
        # check-config runs its own event loop: call it from a thread.
        assert await asyncio.to_thread(main, ["check-config", "-c", ok, "--connect"]) == 0
        assert "language model" in capsys.readouterr().out
        broken = write_config(
            tmp_path,
            stt={"type": "wyoming", "uri": stt.uri},
            tts={"type": "wyoming", "uri": tts.uri},
            llm={"type": "ollama", "url": llm.ollama_url, "model": "missing"},
        )
        assert await asyncio.to_thread(main, ["check-config", "-c", broken, "--connect"]) == 2
        assert await asyncio.to_thread(main, ["check-config", "-c", broken]) == 0  # without --connect
