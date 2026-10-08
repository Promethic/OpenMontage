"""Explicit trusted context for mocked adapter unit tests, never public API tests."""

import pytest
from lib.providers.context import service_worker


@pytest.fixture
def private_provider_adapter():
    with service_worker():
        yield
