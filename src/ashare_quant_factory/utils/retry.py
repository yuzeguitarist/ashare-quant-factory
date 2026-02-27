from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Retry:
    attempts: int = 5
    base_sleep: float = 0.5
    max_sleep: float = 8.0
    jitter: float = 0.25  # seconds


def run_with_retry(fn: Callable[[], T], retry: Retry = Retry()) -> T:
    last_exc: Exception | None = None
    for i in range(retry.attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - retry wrapper
            last_exc = e
            if i == retry.attempts - 1:
                break
            sleep = min(retry.max_sleep, retry.base_sleep * (2**i)) + random.random() * retry.jitter
            time.sleep(sleep)
    assert last_exc is not None
    raise last_exc
