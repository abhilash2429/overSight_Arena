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

Port notes (HF Spaces)
----------------------
HF Spaces exposes exactly one external port: 7860.
- Gradio UI  → port 7860  (external, HF-visible)
- FastAPI    → port 8000  (internal only; Gradio talks to it via localhost:8000)

The Dockerfile starts both processes:
  uvicorn server:app --host 0.0.0.0 --port 8000 & python app.py
"""

import os

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

from oversight_arena.environment import OversightArenaEnvironment

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
