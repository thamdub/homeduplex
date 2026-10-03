"""Wires the settings to a running server: the only place that knows which concrete parts are used."""

import asyncio
import contextlib
import logging
import signal

from homeduplex import backends
from homeduplex.backends.probe import probe_all
from homeduplex.config import Settings
from homeduplex.scheduler.fair import FairQueue
from homeduplex.session.session import Services, Session
from homeduplex.transport.events import Send
from homeduplex.transport.room import Room
from homeduplex.transport.server import RealtimeServer, listeners

log = logging.getLogger(__name__)


def build_services(settings: Settings) -> Services:
    return Services(
        stt=backends.speech_to_text(settings.stt),
        llm=backends.language_model(settings.llm),
        tts=backends.text_to_speech(settings.tts),
        stt_queue=FairQueue("speech-to-text", settings.stt.slots),
        llm_queue=FairQueue("the language model", settings.llm.slots),
        tts_queue=FairQueue("text-to-speech", settings.tts.slots),
    )


async def close_services(services: Services) -> None:
    await asyncio.gather(services.stt.aclose(), services.llm.aclose(), services.tts.aclose())


def build_server(settings: Settings, services: Services) -> RealtimeServer:
    def new_session(send: Send, room: Room, model: str) -> Session:
        return Session(send, settings, services, room, model)

    return RealtimeServer(settings, new_session)


async def report_backends(settings: Settings) -> None:
    """Log whether each backend answers; a wrong address otherwise shows only when a conversation fails."""
    for result in await probe_all(settings):
        if result.ok:
            log.info("%s", result.line())
        else:
            log.warning("%s", result.line())


async def run(settings: Settings) -> None:
    """Serve until SIGINT or SIGTERM, then close every connection (which cancels their work)."""
    for warning in settings.warnings():
        log.warning("%s", warning)
    services = build_services(settings)
    server = build_server(settings, services)
    await server.start(listeners(settings))
    probes = asyncio.create_task(report_backends(settings), name="backend probes")
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):  # not on Windows
            loop.add_signal_handler(sig, stop.set)
    try:
        await stop.wait()
    finally:
        log.info("shutting down")
        probes.cancel()
        await server.close()
        await close_services(services)
