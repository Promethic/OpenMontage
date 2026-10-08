"""Trusted execution context, entered only by the service-owned worker."""

from contextlib import contextmanager
from contextvars import ContextVar

_WORKER = ContextVar("studio_provider_worker", default=False)


def in_worker() -> bool:
    return _WORKER.get()


@contextmanager
def service_worker():
    token = _WORKER.set(True)
    try:
        yield
    finally:
        _WORKER.reset(token)
