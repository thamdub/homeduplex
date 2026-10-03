"""Load settings from a YAML file, with environment variables on top.

`HOMEDUPLEX_<SECTION>__<KEY>=value` sets `section.key` (any depth, case-insensitive), for secrets and container
deployments: `HOMEDUPLEX_LLM__API_KEY=…`, `HOMEDUPLEX_SERVER__PORT=8780`. Values are strings; the schema converts
them.
"""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from homeduplex.config.schema import Settings

ENV_PREFIX = "HOMEDUPLEX_"
CONFIG_ENV = "HOMEDUPLEX_CONFIG"
DEFAULT_PATH = Path("homeduplex.yaml")


class ConfigError(Exception):
    """Everything wrong with a configuration, one problem per line."""

    def __init__(self, source: str, problems: list[str]) -> None:
        self.source = source
        self.problems = problems
        super().__init__(f"{source}:\n" + "\n".join(f"  - {p}" for p in problems))


def config_path(explicit: str | Path | None, environ: Mapping[str, str] | None = None) -> Path:
    environ = os.environ if environ is None else environ
    if explicit is not None:
        return Path(explicit)
    return Path(environ.get(CONFIG_ENV) or DEFAULT_PATH)


def load(path: Path, environ: Mapping[str, str] | None = None) -> Settings:
    environ = os.environ if environ is None else environ
    data = _read_yaml(path)
    _apply_env(data, environ)
    try:
        settings = Settings.model_validate(data)
    except ValidationError as e:
        raise ConfigError(str(path), [_describe(err) for err in e.errors()]) from None
    return _read_preamble_file(settings, path)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ConfigError(str(path), [f"cannot read the file: {e.strerror}"]) from None
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ConfigError(str(path), [f"not valid YAML: {e}"]) from None
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(str(path), ["the top level must be a mapping (section: ...)"])
    return data


def _apply_env(data: dict[str, Any], environ: Mapping[str, str]) -> None:
    for name, value in sorted(environ.items()):
        if not name.upper().startswith(ENV_PREFIX) or name.upper() == CONFIG_ENV:
            continue
        keys = name[len(ENV_PREFIX) :].lower().split("__")
        if not all(keys):
            continue
        node = data
        for key in keys[:-1]:
            child = node.get(key)
            if not isinstance(child, dict):
                child = node[key] = {}
            node = child
        node[keys[-1]] = value


def _describe(error: Any) -> str:
    # Tagged unions add the tag to the location (llm.openai.model); the user wrote llm.model.
    loc = [str(part) for part in error["loc"] if part not in ("wyoming", "openai", "ollama")]
    where = ".".join(loc)
    message = str(error["msg"]).removeprefix("Value error, ")
    if error["type"] == "extra_forbidden":
        message = "unknown setting"
    return f"{where}: {message}" if where else message


def _read_preamble_file(settings: Settings, config_file: Path) -> Settings:
    file = settings.prompt.preamble_file
    if file is None:
        return settings
    if not file.is_absolute():
        file = config_file.parent / file
    try:
        text = file.read_text(encoding="utf-8")
    except OSError as e:
        raise ConfigError(str(config_file), [f"prompt.preamble_file: cannot read {file}: {e.strerror}"]) from None
    prompt = settings.prompt.model_copy(update={"preamble": text.strip(), "preamble_file": None})
    return settings.model_copy(update={"prompt": prompt})
