"""Shared HTTP plumbing for OpenAI-compatible servers: one connection pool per backend, errors as BackendError, and
requests that can be cancelled at any moment without leaking a connection."""

import asyncio
import contextlib
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
from pydantic import SecretStr

from homeduplex.backends.base import BackendError

CONNECT_TIMEOUT = 5.0


def make_client(url: str, api_key: SecretStr | None, timeout: float) -> httpx.AsyncClient:
    headers = {"Authorization": f"Bearer {api_key.get_secret_value()}"} if api_key else {}
    # Connect quickly or fail; reads are bounded per call by the adapters.
    return httpx.AsyncClient(
        base_url=url + "/", headers=headers, timeout=httpx.Timeout(timeout, connect=min(timeout, CONNECT_TIMEOUT))
    )


@contextlib.asynccontextmanager
async def open_stream(client: httpx.AsyncClient, method: str, url: str, **kwargs: Any) -> AsyncIterator[httpx.Response]:
    """`client.stream()`, safe to cancel at any moment.

    httpcore loses the socket when a cancellation lands while it is setting up a new connection (it awaits a trace
    hook before recording the connection): the socket then stays open until garbage collection, and the server
    keeps working on a request nobody wants. So the request runs in its own task, and a cancellation that arrives
    during connection setup waits for the setup to finish (bounded) before cancelling it. Once the request is being
    sent, httpx handles cancellation properly, so it is cancelled at once: a model server reading a long prompt is
    stopped immediately.
    """
    sending = asyncio.Event()

    async def trace(event: str, info: Any) -> None:
        if event.endswith("send_request_headers.started"):
            sending.set()

    request = client.build_request(method, url, extensions={"trace": trace}, **kwargs)
    opening = asyncio.ensure_future(client.send(request, stream=True))
    try:
        response = await asyncio.shield(opening)
    except asyncio.CancelledError:
        await _abandon(opening, sending)
        raise
    try:
        yield response
    finally:
        await response.aclose()


async def _abandon(opening: asyncio.Future[httpx.Response], sending: asyncio.Event) -> None:
    if not opening.done() and not sending.is_set():
        waiter: asyncio.Future[Any] = asyncio.ensure_future(sending.wait())
        pending: set[asyncio.Future[Any]] = {opening, waiter}
        await asyncio.wait(pending, timeout=CONNECT_TIMEOUT, return_when=asyncio.FIRST_COMPLETED)
        waiter.cancel()
    opening.cancel()
    try:
        response = await opening
    except (asyncio.CancelledError, httpx.HTTPError):
        return
    await response.aclose()


@contextlib.contextmanager
def http_errors(name: str, timeout: float) -> Iterator[None]:
    """Turn transport failures into BackendError with a short, secret-free message."""
    try:
        yield
    except httpx.ConnectTimeout:
        raise BackendError(f"{name}: cannot connect") from None
    except httpx.TimeoutException:
        raise BackendError(f"{name}: no answer within {timeout:g} s") from None
    except httpx.HTTPError as e:
        raise BackendError(f"{name}: {type(e).__name__}: {e}") from None


async def check_status(response: httpx.Response, name: str) -> None:
    if response.status_code < 400:
        return
    detail = (await response.aread()).decode(errors="replace").strip()[:300]
    raise BackendError(f"{name}: HTTP {response.status_code}: {detail}")
