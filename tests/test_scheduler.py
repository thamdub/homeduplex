import asyncio

import pytest
from support.wait import wait_for

from homeduplex.scheduler.fair import FairQueue


async def hold(queue: FairQueue, owner: str, log: list[str], release: asyncio.Event, label: str = "") -> None:
    async with queue.slot(owner):
        log.append(label or owner)
        await release.wait()


async def test_slots_limit_concurrency() -> None:
    queue = FairQueue("test", 2)
    release = asyncio.Event()
    log: list[str] = []
    tasks = [asyncio.create_task(hold(queue, o, log, release)) for o in "abc"]
    await wait_for(lambda: len(log) == 2)
    assert queue.in_use == 2 and queue.waiting == 1
    release.set()
    await asyncio.gather(*tasks)
    assert sorted(log) == ["a", "b", "c"]
    assert queue.in_use == 0 and queue.waiting == 0


async def test_round_robin_between_owners() -> None:
    """Room A queued three sentences before room B queued one: B goes second, not fourth."""
    queue = FairQueue("test", 1)
    order: list[str] = []
    gate = asyncio.Event()

    async def job(owner: str, label: str) -> None:
        async with queue.slot(owner):
            order.append(label)
            await gate.wait()

    first = asyncio.create_task(job("a", "a1"))
    await wait_for(lambda: order == ["a1"])
    others = [asyncio.create_task(job("a", f"a{n}")) for n in (2, 3, 4)]
    await asyncio.sleep(0)
    others.append(asyncio.create_task(job("b", "b1")))
    await wait_for(lambda: queue.waiting == 4)
    gate.set()
    await asyncio.gather(first, *others)
    assert order == ["a1", "a2", "b1", "a3", "a4"]


async def test_fifo_within_an_owner() -> None:
    queue = FairQueue("test", 1)
    order: list[str] = []
    gate = asyncio.Event()
    tasks = [asyncio.create_task(hold(queue, "a", order, gate, f"{n}")) for n in range(4)]
    await wait_for(lambda: queue.waiting == 3)
    gate.set()
    await asyncio.gather(*tasks)
    assert order == ["0", "1", "2", "3"]


async def test_cancelled_waiter_leaves_the_queue() -> None:
    queue = FairQueue("test", 1)
    order: list[str] = []
    gate = asyncio.Event()
    holder = asyncio.create_task(hold(queue, "a", order, gate))
    await wait_for(lambda: order == ["a"])
    leaving = asyncio.create_task(hold(queue, "b", order, gate))
    staying = asyncio.create_task(hold(queue, "c", order, gate))
    await wait_for(lambda: queue.waiting == 2)
    leaving.cancel()
    await asyncio.gather(leaving, return_exceptions=True)
    assert queue.waiting == 1
    gate.set()
    await asyncio.gather(holder, staying)
    assert order == ["a", "c"]
    assert queue.in_use == 0


async def test_cancelled_just_as_the_slot_arrives_passes_it_on() -> None:
    queue = FairQueue("test", 1)
    order: list[str] = []
    async with queue.slot("a"):
        unlucky = asyncio.create_task(hold(queue, "b", order, asyncio.Event()))
        next_in_line = asyncio.create_task(hold(queue, "c", order, asyncio.Event()))
        await wait_for(lambda: queue.waiting == 2)
    # Leaving the slot handed it to b, which hasn't run yet; cancelling b now must pass the slot to c.
    assert queue.waiting == 1
    unlucky.cancel()
    await wait_for(lambda: order == ["c"])
    next_in_line.cancel()
    await asyncio.gather(unlucky, next_in_line, return_exceptions=True)
    assert queue.in_use == 0 and queue.waiting == 0


async def test_failure_releases_the_slot() -> None:
    queue = FairQueue("test", 1)
    with pytest.raises(RuntimeError):
        async with queue.slot("a"):
            raise RuntimeError("backend failed")
    assert queue.in_use == 0
    async with queue.slot("b"):
        assert queue.in_use == 1
