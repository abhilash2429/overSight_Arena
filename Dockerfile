# =============================================================================
# Oversight Arena — Dockerfile
#
# Port layout (HF Spaces exposes exactly ONE port: 7860):
#   7860  — Gradio UI   (external; HF Spaces endpoint)
#   8000  — FastAPI/OpenEnv server (internal; Gradio talks to it via localhost)
#
# CMD starts both processes:
#   1. uvicorn serves the OpenEnv API on :8000 in the background.
#   2. python app.py launches the Gradio judge UI on :7860 in the foreground.
#
# Gradio is the process HF Spaces probes on :7860.
# The OpenEnv training client connects to the Space URL on :8000 only when
# running locally or from a notebook with port-forwarding.
# =============================================================================

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# HF Spaces exposes port 7860 (Gradio). FastAPI runs internally on 8000.
EXPOSE 7860

# OpenEnv is mounted on the public ASGI app at /openenv (see app.py); single process.
CMD ["python", "app.py"]
