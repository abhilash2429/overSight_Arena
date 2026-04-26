# server.py
"""
Oversight Arena — OpenEnv-compliant FastAPI server.

Run locally:
    python server.py
    uvicorn server:app --host 0.0.0.0 --port 8000

The environment is exposed via:
  POST /reset         — start a new episode
  POST /step          — submit a supervisor action (CallToolAction or raw string)
  GET  /state         — episode metadata (State object)
  GET  /health        — liveness probe
  WebSocket /ws       — streaming step/reset/state
  POST /mcp           — MCP JSON-RPC endpoint (tools/list, tools/call)

On the Hugging Face Space, this app is **mounted** at ``/openenv`` on the
public process (``app.py`` on port 7860), so the WebSocket URL path is
``/openenv/ws``. Use ``https://huggingface.co/spaces/<user>/oversight-arena/openenv``
as the OpenEnv *base* URL in clients. Use ``GET /health`` on the same origin for
Gradio liveness (not on this sub-app, unless the host also forwards it).

Port notes (local dev)
----------------------
``python app.py`` serves Gradio and mounts this app at ``/openenv`` on :7860.
``python server.py`` (or ``uvicorn server:app --port 8000``) still works for
local debugging on a dedicated port.
"""

import os

from oversight_arena.log_filters import install_asyncio_stale_loop_unraisable_filter

install_asyncio_stale_loop_unraisable_filter()

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
from openenv.core.env_server import serialization as _openenv_serialization
from openenv.core.env_server.types import Observation

from oversight_arena.environment import OversightArenaEnvironment

# openenv's default serializer omits ``metadata`` from the WebSocket JSON payload; RL
# training needs ``metadata["pipeline_text"]`` and reward breakdowns.
_serialize_observation_orig = _openenv_serialization.serialize_observation


def serialize_observation(observation: Observation) -> dict:  # type: ignore[valid-type]
    out = _serialize_observation_orig(observation)
    inner = out.get("observation")
    if not isinstance(inner, dict):
        inner = {}
    meta = getattr(observation, "metadata", None)
    if meta:
        inner = {**inner, "metadata": dict(meta)}
    out["observation"] = inner
    return out


_openenv_serialization.serialize_observation = serialize_observation

# Default to 1 concurrent session — OversightArenaEnvironment is not yet
# validated for multi-session use.  Override via env var when ready:
#   MAX_CONCURRENT_ENVS=4 uvicorn server:app --host 0.0.0.0 --port 8000
max_concurrent = int(os.getenv("MAX_CONCURRENT_ENVS", "1"))

# Pass the CLASS (not an instance) so create_app can create per-session instances
app = create_app(
    OversightArenaEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="oversight-arena",
    max_concurrent_envs=max_concurrent,
)


def main():
    import uvicorn

    # FastAPI runs on 8000 internally; Gradio takes 7860 for HF Spaces.
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
