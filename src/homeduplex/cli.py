"""Command line: `homeduplex serve` and `homeduplex check-config`."""

import argparse
import asyncio
import sys
from collections.abc import Sequence

from homeduplex import __version__
from homeduplex.config import ConfigError, Settings, config_path, load


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="homeduplex", description=__doc__)
    parser.add_argument("--version", action="version", version=f"homeduplex {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    config_help = "settings file (default: $HOMEDUPLEX_CONFIG or ./homeduplex.yaml)"
    serve = commands.add_parser("serve", help="run the server")
    serve.add_argument("-c", "--config", help=config_help)
    serve.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    check = commands.add_parser("check-config", help="validate the settings file and print a summary")
    check.add_argument("-c", "--config", help=config_help)
    check.add_argument(
        "--connect", action="store_true", help="also check that each backend answers (exit code 2 if one doesn't)"
    )
    args = parser.parse_args(argv)

    try:
        settings = load(config_path(args.config))
    except ConfigError as e:
        print(f"Configuration error in {e}", file=sys.stderr)
        return 1

    if args.command == "check-config":
        print(summary(settings))
        for warning in settings.warnings():
            print(f"warning: {warning}", file=sys.stderr)
        if args.connect:
            from homeduplex.backends.probe import probe_all

            results = asyncio.run(probe_all(settings))
            for result in results:
                print(result.line())
            return 0 if all(r.ok for r in results) else 2
        return 0

    from homeduplex.app import run
    from homeduplex.observability import setup_logging

    setup_logging(args.log_level)
    asyncio.run(run(settings))
    return 0


def summary(settings: Settings) -> str:
    """What the server would use. Never prints secrets."""
    server = settings.server
    ports = ", ".join(f"{port} → {room}" for port, room in settings.server.extra_ports.items())
    lines = [
        f"listen   {server.host}:{server.port}" + (f" (+ {ports})" if ports else ""),
        f"auth     {'API key required' if server.api_key else 'none'}",
        f"stt      {_endpoint(settings.stt)}",
        f"tts      {_endpoint(settings.tts)}" + (f", voice {settings.tts.voice}" if settings.tts.voice else ""),
        f"llm      {_endpoint(settings.llm)}, model {settings.llm.model}",
        f"rooms    {', '.join(settings.rooms) or 'none'}"
        + (f" (default {settings.default_room})" if settings.default_room else ""),
    ]
    return "\n".join(lines)


def _endpoint(backend: object) -> str:
    kind = getattr(backend, "type", "?")
    where = getattr(backend, "uri", None) or getattr(backend, "url", None)
    return f"{kind} {where}"


if __name__ == "__main__":
    sys.exit(main())
