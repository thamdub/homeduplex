from pathlib import Path

import pytest

from homeduplex.config import ConfigError, Settings, config_path, load
from homeduplex.config.schema import OllamaChat, OpenAIChat, WyomingSTT

EXAMPLES = Path(__file__).parent.parent / "examples"

MINIMAL = """
stt: {type: wyoming, uri: "tcp://stt.example.lan:10300"}
tts: {type: wyoming, uri: "tcp://tts.example.lan:10200"}
llm: {type: openai, url: "http://llm.example.lan:8080/v1/", model: m}
"""


def write(tmp_path: Path, text: str, name: str = "homeduplex.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def problems(path: Path, environ: dict[str, str] | None = None) -> list[str]:
    with pytest.raises(ConfigError) as info:
        load(path, environ or {})
    return info.value.problems


@pytest.mark.parametrize("example", sorted(EXAMPLES.glob("*.yaml")), ids=lambda p: p.name)
def test_examples_load(example: Path) -> None:
    load(example, {})


def test_minimal_config_gets_defaults(tmp_path: Path) -> None:
    settings = load(write(tmp_path, MINIMAL), {})
    assert settings.server.port == 8770
    assert settings.conversation.min_silence_ms == 800
    assert isinstance(settings.stt, WyomingSTT)
    assert isinstance(settings.llm, OpenAIChat)
    assert settings.llm.url == "http://llm.example.lan:8080/v1"  # trailing slash dropped


def test_backends_are_required(tmp_path: Path) -> None:
    assert sorted(problems(write(tmp_path, "server: {port: 9000}"))) == [
        "llm: Field required",
        "stt: Field required",
        "tts: Field required",
    ]


def test_unknown_setting_is_named(tmp_path: Path) -> None:
    found = problems(write(tmp_path, MINIMAL + "server: {prot: 9000}\n"))
    assert found == ["server.prot: unknown setting"]


def test_problem_location_hides_the_backend_tag(tmp_path: Path) -> None:
    text = MINIMAL.replace("model: m", "model: m, temperature: 2")
    assert problems(write(tmp_path, text)) == ["llm.temperature: unknown setting"]


def test_unknown_backend_type(tmp_path: Path) -> None:
    text = MINIMAL.replace("type: openai", "type: llamafile")
    [problem] = problems(write(tmp_path, text))
    assert problem.startswith("llm:") and "'openai'" in problem and "'ollama'" in problem


@pytest.mark.parametrize(
    ("bad", "expected"),
    [
        ('"stt.example.lan:10300"', "stt.uri: expected tcp://host:port"),
        ('"tcp://stt.example.lan"', "stt.uri: expected tcp://host:port"),
    ],
)
def test_wyoming_uri(tmp_path: Path, bad: str, expected: str) -> None:
    text = MINIMAL.replace('"tcp://stt.example.lan:10300"', bad)
    assert problems(write(tmp_path, text)) == [expected]


def test_http_url(tmp_path: Path) -> None:
    text = MINIMAL.replace('"http://llm.example.lan:8080/v1/"', "llm.example.lan:8080")
    assert problems(write(tmp_path, text)) == ["llm.url: expected an http:// or https:// URL"]


def test_ollama_dialect(tmp_path: Path) -> None:
    text = MINIMAL.replace(
        'llm: {type: openai, url: "http://llm.example.lan:8080/v1/", model: m}',
        "llm: {type: ollama, url: 'http://llm.example.lan:11434', model: m, options: {num_ctx: 16384},"
        " keep_alive: -1, think: false}",
    )
    llm = load(write(tmp_path, text), {}).llm
    assert isinstance(llm, OllamaChat)
    assert llm.options == {"num_ctx": 16384}
    assert llm.keep_alive == -1
    assert llm.think is False


def test_env_overrides_file(tmp_path: Path) -> None:
    env = {
        "HOMEDUPLEX_SERVER__PORT": "8780",
        "HOMEDUPLEX_LLM__API_KEY": "secret-value",
        "homeduplex_llm__model": "other",
        "HOMEDUPLEX_CONFIG": "ignored.yaml",
        "PATH": "/bin",
    }
    settings = load(write(tmp_path, MINIMAL), env)
    assert settings.server.port == 8780
    assert isinstance(settings.llm, OpenAIChat)
    assert settings.llm.model == "other"
    assert settings.llm.api_key is not None
    assert settings.llm.api_key.get_secret_value() == "secret-value"
    assert "secret-value" not in repr(settings)


def test_env_can_supply_a_whole_section_field(tmp_path: Path) -> None:
    text = MINIMAL.replace("model: m", "")
    settings = load(write(tmp_path, text), {"HOMEDUPLEX_LLM__MODEL": "from-env"})
    assert settings.llm.model == "from-env"


def test_bad_env_value_is_reported(tmp_path: Path) -> None:
    [problem] = problems(write(tmp_path, MINIMAL), {"HOMEDUPLEX_SERVER__PORT": "eighty"})
    assert problem.startswith("server.port:")


def test_default_room_must_exist(tmp_path: Path) -> None:
    text = MINIMAL + "rooms: {office: {area: Office}}\ndefault_room: kitchen\n"
    assert problems(write(tmp_path, text)) == ["default_room 'kitchen' is not in rooms"]


def test_extra_port_room_must_exist(tmp_path: Path) -> None:
    text = MINIMAL + "rooms: {office: {area: Office}}\nserver: {extra_ports: {8771: attic}}\n"
    assert problems(write(tmp_path, text)) == [
        "server.extra_ports: port 8771 names room 'attic', which is not in rooms"
    ]


def test_extra_port_must_differ_from_main_port(tmp_path: Path) -> None:
    text = MINIMAL + "rooms: {office: {}}\nserver: {port: 8771, extra_ports: {8771: office}}\n"
    assert problems(write(tmp_path, text)) == ["server.extra_ports: port 8771 is already server.port"]


def test_template_variables_from_rooms(tmp_path: Path) -> None:
    text = MINIMAL + (
        "rooms: {office: {area: Office, player: a}, kitchen: {area: Kitchen}}\n"
        "prompt: {context: 'In the {area} ({room}). Today is {now:%A}.'}\n"
    )
    settings = load(write(tmp_path, text), {})
    assert settings.template_variables() == {"room", "now", "area"}
    assert settings.warnings() == []


def test_time_of_day_in_context_warns(tmp_path: Path) -> None:
    settings = load(write(tmp_path, MINIMAL + "prompt: {context: 'It is {now:%A %-I:%M %p}.'}\n"), {})
    [warning] = settings.warnings()
    assert warning.startswith("prompt.context uses the time of day (%I, %M, %p)")
    assert "GetDateTime" in warning


def test_template_field_missing_from_a_room(tmp_path: Path) -> None:
    text = MINIMAL + (
        "rooms: {office: {area: Office, player: a}, kitchen: {area: Kitchen}}\nprompt: {context: 'Use {player}.'}\n"
    )
    [problem] = problems(write(tmp_path, text))
    assert problem.startswith("prompt.context: unknown variable {player}")


def test_old_prompt_slots_are_rejected(tmp_path: Path) -> None:
    assert sorted(problems(write(tmp_path, MINIMAL + "prompt: {room: a, turn: b}\n"))) == [
        "prompt.room: unknown setting",
        "prompt.turn: unknown setting",
    ]


def test_preamble_file_is_read_relative_to_config(tmp_path: Path) -> None:
    write(tmp_path, "Be brief.\n", "preamble.md")
    settings = load(write(tmp_path, MINIMAL + "prompt: {preamble_file: preamble.md}\n"), {})
    assert settings.prompt.preamble == "Be brief."


def test_missing_preamble_file(tmp_path: Path) -> None:
    [problem] = problems(write(tmp_path, MINIMAL + "prompt: {preamble_file: nope.md}\n"))
    assert problem.startswith("prompt.preamble_file: cannot read")


def test_preamble_and_file_are_exclusive(tmp_path: Path) -> None:
    text = MINIMAL + "prompt: {preamble: x, preamble_file: p.md}\n"
    assert problems(write(tmp_path, text)) == ["prompt: set preamble or preamble_file, not both"]


def test_timezone(tmp_path: Path) -> None:
    assert load(write(tmp_path, MINIMAL + "prompt: {timezone: Europe/Paris}\n"), {}).prompt.timezone == "Europe/Paris"
    [problem] = problems(write(tmp_path, MINIMAL + "prompt: {timezone: Mars/Olympus}\n"))
    assert problem.startswith("prompt.timezone: unknown time zone")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("stt: [unclosed", "not valid YAML"),
        ("- a\n- b\n", "the top level must be a mapping"),
    ],
)
def test_unreadable_yaml(tmp_path: Path, text: str, expected: str) -> None:
    [problem] = problems(write(tmp_path, text))
    assert problem.startswith(expected)


def test_missing_file(tmp_path: Path) -> None:
    [problem] = problems(tmp_path / "absent.yaml")
    assert problem.startswith("cannot read the file")


def test_config_path() -> None:
    assert config_path("a.yaml", {"HOMEDUPLEX_CONFIG": "b.yaml"}) == Path("a.yaml")
    assert config_path(None, {"HOMEDUPLEX_CONFIG": "b.yaml"}) == Path("b.yaml")
    assert config_path(None, {}) == Path("homeduplex.yaml")


def test_settings_are_immutable(tmp_path: Path) -> None:
    settings: Settings = load(write(tmp_path, MINIMAL), {})
    with pytest.raises(ValueError, match="frozen"):
        settings.server.port = 1
