"""Fair access to a shared backend (docs/design.md, "Where concurrency really breaks").

A backend has a number of slots (a single-slot model server has one). When a slot frees up it goes to the next
conversation in round-robin order, not to whoever asked first, so a long answer in one room, spoken sentence by
sentence, can't starve a short command in another. A conversation cancelled while waiting leaves the queue.
"""

import asyncio
import contextlib
import logging
from collections import OrderedDict, deque
from collections.abc import AsyncIterator

log = logging.getLogger(__name__)

# Waits longer than this are logged: they are what users feel as slowness.
SLOW_WAIT_S = 0.5


class FairQueue:
    def __init__(self, name: str, slots: int) -> None:
        self.name = name
        self.slots = slots
        self._free = slots
        # Owner → its waiters, in the order owners take turns.
        self._waiting: OrderedDict[str, deque[asyncio.Future[None]]] = OrderedDict()

    @property
    def in_use(self) -> int:
        return self.slots - self._free

    @property
    def waiting(self) -> int:
        return sum(len(w) for w in self._waiting.values())

    @contextlib.asynccontextmanager
    async def slot(self, owner: str) -> AsyncIterator[None]:
        await self._acquire(owner)
        try:
            yield
        finally:
            self._release()

    async def _acquire(self, owner: str) -> None:
        if self._free > 0 and not self._waiting:
            self._free -= 1
            return
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[None] = loop.create_future()
        self._waiting.setdefault(owner, deque()).append(waiter)
        started = loop.time()
        try:
            await waiter
        except asyncio.CancelledError:
            if waiter.done() and not waiter.cancelled():
                self._release()  # the slot was handed over just as we were cancelled: pass it on
            else:
                self._forget(owner, waiter)
            raise
        waited = loop.time() - started
        if waited > SLOW_WAIT_S:
            log.info("waited %.1f s for %s", waited, self.name)

    def _release(self) -> None:
        while self._waiting:
            owner, waiters = next(iter(self._waiting.items()))
            waiter = waiters.popleft()
            # Round robin: this owner goes to the back of the line, or leaves it.
            del self._waiting[owner]
            if waiters:
                self._waiting[owner] = waiters
            if not waiter.done():
                waiter.set_result(None)  # the slot passes directly to the waiter
                return
        self._free += 1

    def _forget(self, owner: str, waiter: asyncio.Future[None]) -> None:
        waiters = self._waiting.get(owner)
        if waiters is None:
            return
        with contextlib.suppress(ValueError):
            waiters.remove(waiter)
        if not waiters:
            del self._waiting[owner]
