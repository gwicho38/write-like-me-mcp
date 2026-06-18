"""Shared pytest configuration for the write-like-me-mcp test suite.

The project does not depend on ``pytest-asyncio`` (keeping the dev surface
minimal), so this conftest provides a tiny hook that runs ``async def test_*``
coroutine tests on a fresh event loop. The ``@pytest.mark.asyncio`` markers used
in the async tests are registered here purely so pytest does not warn about an
unknown marker; the execution is handled by ``pytest_pyfunc_call`` below.
"""

from __future__ import annotations

import asyncio
import inspect


def pytest_configure(config):
    """Register the ``asyncio`` marker so it is not reported as unknown."""
    config.addinivalue_line(
        "markers", "asyncio: run an async test function on a fresh event loop"
    )


def pytest_pyfunc_call(pyfuncitem):
    """Execute coroutine test functions without requiring pytest-asyncio.

    If the collected test function is a coroutine function, run it to completion
    on a fresh event loop, passing through only the fixture arguments the test
    actually declares. Returning ``True`` tells pytest the call was handled.
    """
    test_func = pyfuncitem.obj
    if not inspect.iscoroutinefunction(test_func):
        return None

    sig = inspect.signature(test_func)
    kwargs = {
        name: pyfuncitem.funcargs[name]
        for name in sig.parameters
        if name in pyfuncitem.funcargs
    }

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(test_func(**kwargs))
    finally:
        loop.close()
    return True
