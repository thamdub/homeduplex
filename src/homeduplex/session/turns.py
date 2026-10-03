"""Who decides that the user has finished and the assistant should answer (docs/design.md, Turn-taking).

The session reports speech start and stop to a policy. With client-driven turns (`create_response: false`, Kiosk
Satellite) the policy does nothing: the client sends `response.create` or `response.cancel` itself. A server-driven
policy (`create_response: true`) will start and cancel answers from these hooks.
"""

from typing import Protocol


class TurnPolicy(Protocol):
    async def on_speech_started(self, item_id: str) -> None: ...

    async def on_speech_stopped(self, item_id: str) -> None: ...


class ClientTurns:
    async def on_speech_started(self, item_id: str) -> None:
        pass

    async def on_speech_stopped(self, item_id: str) -> None:
        pass
