import asyncio
from collections.abc import Callable


async def wait_for(condition: Callable[[], bool], within: float = 2) -> None:
    """Poll until `condition()` holds; TimeoutError after `within` seconds."""
    async with asyncio.timeout(within):
        while not condition():  # noqa: ASYNC110 (polling state owned by other code is the point)
            await asyncio.sleep(0.01)
