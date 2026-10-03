import gc
from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def collect_garbage() -> Iterator[None]:
    """Leaked sockets and files warn when garbage-collected; collecting after each test blames the right one."""
    yield
    gc.collect()
