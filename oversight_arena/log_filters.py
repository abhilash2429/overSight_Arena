# oversight_arena/log_filters.py
# ---------------------------
# CPython logs unraisable exceptions from destructors to stderr. When Gradio/Starlette/uvicorn
# discard a selector event loop, BaseEventLoop.__del__ can raise ValueError("Invalid file
# descriptor: -1") on Linux — harmless. Narrow filter so real issues still surface.

from __future__ import annotations

import sys
import traceback
from typing import Any


def install_asyncio_stale_loop_unraisable_filter() -> None:
    """Register sys.unraisablehook to drop known spurious asyncio loop teardown ValueErrors."""
    if getattr(sys, "unraisablehook", None) is None:  # pragma: no cover
        return
    if getattr(sys, "_oversight_arena_stale_loop_hooked", False):
        return
    _orig: Any = sys.__unraisablehook__  # type: ignore[misc]
    if not callable(_orig):
        return

    def _hook(uh: object) -> None:
        ex = getattr(uh, "exc_value", None)
        if (
            ex is not None
            and isinstance(ex, ValueError)
            and "Invalid file descriptor" in str(ex)
        ):
            tb = getattr(uh, "exc_traceback", None)
            stack = "" if not tb else "".join(traceback.format_tb(tb))
            if "asyncio" in stack and "base_events" in stack:
                return
        _orig(uh)

    sys.unraisablehook = _hook  # type: ignore[assignment]
    sys._oversight_arena_stale_loop_hooked = True  # type: ignore[attr-defined]
