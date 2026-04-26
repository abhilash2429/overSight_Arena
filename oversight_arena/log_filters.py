"""Small runtime log filters shared by the Space UI and OpenEnv server."""

from __future__ import annotations

import sys
import traceback
from types import SimpleNamespace
from typing import Callable


_INSTALLED = False


def install_asyncio_stale_loop_unraisable_filter() -> None:
    """Suppress noisy asyncio cleanup warnings emitted during Gradio reload/shutdown."""
    global _INSTALLED
    if _INSTALLED:
        return

    previous_hook: Callable[[SimpleNamespace], None] = sys.unraisablehook

    def _filtered_hook(unraisable: SimpleNamespace) -> None:
        exc = unraisable.exc_value
        obj = unraisable.object
        tb = getattr(unraisable, "exc_traceback", None)
        stack = "" if tb is None else "".join(traceback.format_tb(tb))
        text = f"{exc!r} {obj!r}"
        stale_loop_noise = (
            "Event loop is closed" in text
            or "Invalid file descriptor" in text and "asyncio" in stack
            or "BaseEventLoop.__del__" in text
            or "ProactorBasePipeTransport.__del__" in text
        )
        if stale_loop_noise:
            return
        previous_hook(unraisable)

    sys.unraisablehook = _filtered_hook
    _INSTALLED = True
