"""
Oversight Arena judge interface.

This Gradio app is the public Hugging Face Space frontend. It keeps the real
environment execution path intact while presenting the benchmark as a polished,
judge-friendly product demo rather than a developer test harness.
"""

from __future__ import annotations

import html
import os
import re
from typing import Any

# Hugging Face Spaces can enable SSR by env; keep consistent with ssr_mode=False.
os.environ.setdefault("GRADIO_SSR_MODE", "false")

from oversight_arena.log_filters import install_asyncio_stale_loop_unraisable_filter

install_asyncio_stale_loop_unraisable_filter()

import gradio as gr
import uvicorn
from fastapi import FastAPI

from oversight_arena.environment import OversightArenaEnvironment
from oversight_arena.models import WorkerState
from oversight_arena.oracle import oracle_action


APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
  --oa-bg: #07111f;
  --oa-bg-2: #0b1727;
  --oa-panel: rgba(13, 24, 41, 0.86);
  --oa-panel-strong: rgba(16, 31, 53, 0.96);
  --oa-line: rgba(148, 163, 184, 0.20);
  --oa-text: #eef6ff;
  --oa-muted: #9fb0c6;
  --oa-dim: #6f8198;
  --oa-cyan: #5eead4;
  --oa-blue: #60a5fa;
  --oa-green: #34d399;
  --oa-amber: #fbbf24;
  --oa-red: #fb7185;
  --oa-violet: #a78bfa;
  --oa-shadow: 0 24px 80px rgba(0, 0, 0, 0.38);
}

.gradio-container {
  max-width: 1440px !important;
  color: var(--oa-text);
  background:
    radial-gradient(circle at 12% 0%, rgba(94, 234, 212, 0.20), transparent 30%),
    radial-gradient(circle at 88% 12%, rgba(167, 139, 250, 0.20), transparent 32%),
    linear-gradient(180deg, #07111f 0%, #0b1220 46%, #09111d 100%) !important;
  font-family: "IBM Plex Sans", system-ui, sans-serif !important;
}

.oa-wrap {
  width: 100%;
}

.oa-hero {
  position: relative;
  overflow: hidden;
  border: 1px solid rgba(148, 163, 184, 0.20);
  border-radius: 30px;
  padding: 34px;
  margin: 6px 0 18px;
  background:
    linear-gradient(135deg, rgba(94, 234, 212, 0.14), transparent 30%),
    linear-gradient(315deg, rgba(96, 165, 250, 0.16), transparent 34%),
    rgba(7, 17, 31, 0.82);
  box-shadow: var(--oa-shadow);
}

.oa-hero:after {
  content: "";
  position: absolute;
  inset: 0;
  pointer-events: none;
  background-image:
    linear-gradient(rgba(148, 163, 184, 0.08) 1px, transparent 1px),
    linear-gradient(90deg, rgba(148, 163, 184, 0.08) 1px, transparent 1px);
  background-size: 34px 34px;
  mask-image: linear-gradient(90deg, rgba(0, 0, 0, 0.45), transparent 72%);
}

.oa-kicker {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  color: var(--oa-cyan);
  border: 1px solid rgba(94, 234, 212, 0.28);
  border-radius: 999px;
  padding: 7px 12px;
  background: rgba(94, 234, 212, 0.08);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.oa-hero h1 {
  position: relative;
  z-index: 1;
  max-width: 980px;
  margin: 18px 0 8px;
  color: var(--oa-text);
  font-family: "Space Grotesk", "IBM Plex Sans", system-ui, sans-serif;
  font-size: clamp(44px, 7vw, 84px);
  line-height: 0.91;
  letter-spacing: -0.065em;
}

.oa-hero-content {
  position: relative;
  z-index: 1;
  display: grid;
  grid-template-columns: minmax(0, 0.92fr) minmax(420px, 1.08fr);
  gap: 28px;
  align-items: center;
}

.oa-hero-copy {
  min-width: 0;
}

.oa-hero-visual {
  position: relative;
  overflow: hidden;
  min-height: 430px;
  border: 1px solid rgba(148, 163, 184, 0.22);
  border-radius: 28px;
  background:
    radial-gradient(circle at 50% 18%, rgba(94, 234, 212, 0.18), transparent 28%),
    radial-gradient(circle at 78% 72%, rgba(251, 113, 133, 0.14), transparent 32%),
    rgba(2, 6, 23, 0.38);
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.05), 0 28px 90px rgba(0,0,0,0.28);
}

.oa-hero-visual svg,
.oa-live-map svg {
  width: 100%;
  height: auto;
  display: block;
}

.oa-svg-label {
  font-family: "Space Grotesk", "IBM Plex Sans", system-ui, sans-serif;
  font-weight: 800;
  letter-spacing: -0.02em;
}

.oa-svg-small {
  font-family: "IBM Plex Sans", system-ui, sans-serif;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

@keyframes oaPacketClean {
  0% { offset-distance: 0%; opacity: 0; }
  8% { opacity: 1; }
  78% { opacity: 1; }
  100% { offset-distance: 100%; opacity: 0; }
}

@keyframes oaPacketBad {
  0% { offset-distance: 0%; opacity: 0; transform: scale(0.8); }
  15% { opacity: 1; }
  54% { opacity: 1; transform: scale(1); }
  64% { opacity: 0.15; transform: scale(0.35); }
  100% { offset-distance: 100%; opacity: 0; }
}

@keyframes oaScan {
  0%, 100% { transform: translateX(-30px); opacity: 0.25; }
  50% { transform: translateX(42px); opacity: 0.9; }
}

@keyframes oaPulse {
  0%, 100% { opacity: 0.32; transform: scale(0.96); }
  50% { opacity: 0.92; transform: scale(1.04); }
}

@keyframes oaDash {
  to { stroke-dashoffset: -42; }
}

@keyframes oaReveal {
  0%, 38% { opacity: 0; transform: translateY(8px); }
  46%, 100% { opacity: 1; transform: translateY(0); }
}

.oa-clean-packet {
  offset-path: path("M82 230 C190 128 294 126 402 230 S612 332 722 230");
  animation: oaPacketClean 4.8s linear infinite;
}

.oa-bad-packet {
  offset-path: path("M82 230 C190 128 294 126 402 230 S612 332 722 230");
  animation: oaPacketBad 4.8s linear infinite;
  animation-delay: 1.1s;
}

.oa-scan-beam {
  transform-origin: center;
  animation: oaScan 2.2s ease-in-out infinite;
}

.oa-pulse {
  transform-origin: center;
  animation: oaPulse 1.7s ease-in-out infinite;
}

.oa-dashed-flow {
  stroke-dasharray: 10 12;
  animation: oaDash 1.8s linear infinite;
}

.oa-reveal {
  animation: oaReveal 4.8s ease-in-out infinite;
}

.oa-live-map {
  border: 1px solid rgba(148, 163, 184, 0.18);
  border-radius: 28px;
  padding: 10px;
  margin-bottom: 14px;
  background:
    linear-gradient(135deg, rgba(94, 234, 212, 0.08), transparent 34%),
    rgba(2, 6, 23, 0.30);
  box-shadow: 0 22px 70px rgba(0,0,0,0.18);
}

.oa-subtitle {
  position: relative;
  z-index: 1;
  max-width: 920px;
  margin: 0;
  color: #c9d8ec;
  font-size: clamp(17px, 2vw, 22px);
  line-height: 1.45;
}

.oa-hero-grid {
  position: relative;
  z-index: 1;
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin-top: 26px;
}

.oa-proof-card,
.oa-step-card,
.oa-panel,
.oa-control-panel {
  border: 1px solid var(--oa-line);
  border-radius: 22px;
  background: var(--oa-panel);
  box-shadow: 0 18px 48px rgba(0, 0, 0, 0.22);
}

.oa-proof-card {
  min-height: 138px;
  padding: 18px;
}

.oa-proof-card b {
  display: block;
  margin-bottom: 8px;
  color: var(--oa-text);
  font-size: 15px;
}

.oa-proof-card span {
  color: var(--oa-muted);
  font-size: 13px;
  line-height: 1.45;
}

.oa-section {
  margin: 18px 0;
}

.oa-section-head {
  display: flex;
  align-items: end;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 12px;
}

.oa-section-title {
  margin: 0;
  color: var(--oa-text);
  font-family: "Space Grotesk", "IBM Plex Sans", system-ui, sans-serif;
  font-size: 24px;
  letter-spacing: -0.03em;
}

.oa-section-copy {
  max-width: 780px;
  margin: 4px 0 0;
  color: var(--oa-muted);
  font-size: 14px;
  line-height: 1.5;
}

.oa-how {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 10px;
}

.oa-step-card {
  padding: 16px;
}

.oa-step-num {
  display: inline-flex;
  width: 28px;
  height: 28px;
  align-items: center;
  justify-content: center;
  border-radius: 10px;
  background: rgba(94, 234, 212, 0.12);
  color: var(--oa-cyan);
  font-weight: 800;
}

.oa-step-card b {
  display: block;
  margin: 12px 0 6px;
  color: var(--oa-text);
  font-size: 14px;
}

.oa-step-card span {
  color: var(--oa-muted);
  font-size: 12px;
  line-height: 1.4;
}

.oa-control-panel {
  padding: 18px;
  background:
    linear-gradient(135deg, rgba(96, 165, 250, 0.08), transparent 38%),
    var(--oa-panel-strong);
}

.oa-callout {
  border: 1px solid rgba(94, 234, 212, 0.24);
  border-radius: 18px;
  padding: 14px 16px;
  margin-bottom: 14px;
  color: #d9fff8;
  background: rgba(94, 234, 212, 0.08);
}

.oa-callout b {
  color: var(--oa-cyan);
}

.oa-status-grid {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 10px;
  margin-bottom: 12px;
}

.oa-stat {
  border: 1px solid var(--oa-line);
  border-radius: 18px;
  padding: 14px;
  background: rgba(15, 23, 42, 0.62);
}

.oa-stat span {
  display: block;
  color: var(--oa-dim);
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.oa-stat b {
  display: block;
  margin-top: 6px;
  color: var(--oa-text);
  font-size: 18px;
}

.oa-pipeline {
  display: grid;
  grid-template-columns: repeat(5, minmax(170px, 1fr));
  gap: 12px;
  align-items: stretch;
}

.oa-worker {
  position: relative;
  overflow: hidden;
  min-height: 264px;
  border: 1px solid var(--oa-line);
  border-radius: 24px;
  padding: 16px;
  background: rgba(12, 23, 39, 0.92);
}

.oa-worker:before {
  content: "";
  position: absolute;
  inset: 0 0 auto 0;
  height: 5px;
  background: var(--oa-dim);
}

.oa-worker.working:before { background: var(--oa-blue); }
.oa-worker.completed:before { background: var(--oa-green); }
.oa-worker.approved:before { background: var(--oa-amber); }
.oa-worker.idle:before { background: #64748b; }
.oa-worker.redirected:before { background: var(--oa-violet); }
.oa-worker.suspicious { box-shadow: 0 0 0 1px rgba(251, 191, 36, 0.34), 0 20px 48px rgba(251, 191, 36, 0.10); }
.oa-worker.suspicious:before { background: var(--oa-amber); }
.oa-worker.exposed { box-shadow: 0 0 0 1px rgba(251, 113, 133, 0.45), 0 20px 55px rgba(251, 113, 133, 0.14); }
.oa-worker.exposed:before { background: var(--oa-red); }
.oa-worker.truth-fail { border-color: rgba(251, 113, 133, 0.55); }
.oa-worker.truth-clean { border-color: rgba(52, 211, 153, 0.40); }

.oa-worker-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 12px;
}

.oa-worker-id {
  color: var(--oa-text);
  font-family: "Space Grotesk", "IBM Plex Sans", system-ui, sans-serif;
  font-size: 30px;
  font-weight: 800;
  letter-spacing: -0.05em;
}

.oa-badge {
  display: inline-flex;
  border: 1px solid rgba(148, 163, 184, 0.20);
  border-radius: 999px;
  padding: 5px 9px;
  color: #dbeafe;
  background: rgba(96, 165, 250, 0.10);
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.oa-badge.warn { color: #fff3c4; background: rgba(251, 191, 36, 0.14); border-color: rgba(251, 191, 36, 0.28); }
.oa-badge.danger { color: #ffe4e9; background: rgba(251, 113, 133, 0.16); border-color: rgba(251, 113, 133, 0.30); }
.oa-badge.good { color: #d6fff0; background: rgba(52, 211, 153, 0.14); border-color: rgba(52, 211, 153, 0.28); }
.oa-badge.dim { color: #cbd5e1; background: rgba(100, 116, 139, 0.16); }

.oa-role {
  color: var(--oa-cyan);
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.oa-task {
  margin-top: 6px;
  color: #dbeafe;
  font-size: 14px;
  font-weight: 700;
}

.oa-desc {
  min-height: 54px;
  margin-top: 8px;
  color: var(--oa-muted);
  font-size: 12px;
  line-height: 1.42;
}

.oa-snippet {
  min-height: 74px;
  max-height: 96px;
  overflow: hidden;
  border: 1px solid rgba(148, 163, 184, 0.14);
  border-radius: 16px;
  padding: 11px;
  margin-top: 12px;
  color: #d9e7fa;
  background: rgba(2, 6, 23, 0.42);
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 11px;
  line-height: 1.42;
  white-space: pre-wrap;
}

.oa-worker-foot {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 12px;
}

.oa-chip {
  display: inline-flex;
  align-items: center;
  border: 1px solid rgba(148, 163, 184, 0.16);
  border-radius: 999px;
  padding: 5px 8px;
  color: #cbd5e1;
  background: rgba(15, 23, 42, 0.64);
  font-size: 11px;
}

.oa-chip.hot {
  color: #fff3c4;
  background: rgba(251, 191, 36, 0.12);
  border-color: rgba(251, 191, 36, 0.26);
}

.oa-chip.red {
  color: #ffe4e9;
  background: rgba(251, 113, 133, 0.13);
  border-color: rgba(251, 113, 133, 0.28);
}

.oa-chip.green {
  color: #d6fff0;
  background: rgba(52, 211, 153, 0.12);
  border-color: rgba(52, 211, 153, 0.24);
}

.oa-panel {
  padding: 16px;
}

.oa-log {
  display: grid;
  gap: 10px;
}

.oa-log-item {
  border: 1px solid rgba(148, 163, 184, 0.16);
  border-radius: 18px;
  padding: 13px;
  background: rgba(15, 23, 42, 0.58);
}

.oa-log-line {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  color: #e6f0ff;
  font-weight: 700;
}

.oa-log-meta {
  margin-top: 8px;
  color: var(--oa-muted);
  font-size: 12px;
  line-height: 1.4;
}

.oa-reward-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
}

.oa-reward-card {
  border: 1px solid rgba(148, 163, 184, 0.15);
  border-radius: 16px;
  padding: 10px;
  background: rgba(15, 23, 42, 0.64);
}

.oa-reward-card span {
  display: block;
  color: var(--oa-dim);
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.oa-reward-card b {
  display: block;
  margin-top: 4px;
  color: var(--oa-text);
}

.oa-post-grid {
  display: grid;
  grid-template-columns: 1.1fr 1fr;
  gap: 14px;
}

.oa-table {
  width: 100%;
  border-collapse: collapse;
  overflow: hidden;
  border-radius: 16px;
  font-size: 12px;
}

.oa-table th,
.oa-table td {
  border-bottom: 1px solid rgba(148, 163, 184, 0.13);
  padding: 10px 8px;
  text-align: left;
  vertical-align: top;
}

.oa-table th {
  color: var(--oa-dim);
  background: rgba(15, 23, 42, 0.68);
  font-size: 10px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.oa-table td {
  color: #dbeafe;
  background: rgba(15, 23, 42, 0.36);
}

.oa-oracle {
  border: 1px solid rgba(167, 139, 250, 0.28);
  border-radius: 22px;
  padding: 16px;
  background: rgba(167, 139, 250, 0.08);
}

.oa-empty {
  border: 1px dashed rgba(148, 163, 184, 0.24);
  border-radius: 20px;
  padding: 20px;
  color: var(--oa-muted);
  text-align: center;
  background: rgba(15, 23, 42, 0.36);
}

.obs-text textarea,
.act-text textarea {
  color: #dbeafe !important;
  background: rgba(2, 6, 23, 0.72) !important;
  border: 1px solid rgba(148, 163, 184, 0.22) !important;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important;
  font-size: 12px !important;
  line-height: 1.48 !important;
}

button.primary,
.gr-button-primary {
  background: linear-gradient(135deg, #14b8a6, #2563eb) !important;
  border: 0 !important;
  color: white !important;
  font-weight: 800 !important;
}

button.secondary,
.gr-button-secondary {
  background: rgba(15, 23, 42, 0.74) !important;
  border: 1px solid rgba(148, 163, 184, 0.22) !important;
  color: #e2e8f0 !important;
}

@media (max-width: 1180px) {
  .oa-hero-content {
    grid-template-columns: 1fr;
  }

  .oa-hero-visual {
    min-height: 0;
  }

  .oa-hero-grid,
  .oa-how,
  .oa-status-grid,
  .oa-pipeline {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 760px) {
  .oa-hero {
    padding: 22px;
    border-radius: 22px;
  }
  .oa-hero-grid,
  .oa-how,
  .oa-status-grid,
  .oa-pipeline,
  .oa-post-grid,
  .oa-reward-grid {
    grid-template-columns: 1fr;
  }
}
"""


WORKER_ROLES = {
    1: "Requirement Analyst",
    2: "Coding Agent",
    3: "Test Generator",
    4: "Security Reviewer",
    5: "Deployment Approver",
}

FAILURE_STATES = {
    "HALLUCINATING",
    "STALLED",
    "DRIFTED",
    "DECEPTIVE",
    "CORRUPTED",
}

REWARD_LABELS = {
    "reward_catch": "Failure catch",
    "reward_deceptive_catch": "Deception catch",
    "penalty_false_positive": "False positive",
    "penalty_hallpass": "Bad approval",
    "penalty_deceptive_pass": "Deception missed",
    "reward_efficiency": "Efficiency",
    "reward_collusion": "Collusion",
    "penalty_format": "Format",
    "reward_mercor": "Reasoning",
}


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _clip(value: str, limit: int = 190) -> str:
    value = " ".join((value or "").split())
    return value if len(value) <= limit else value[: limit - 1] + "..."


def _format_action(verb: str, worker_id: int | str, reason: str = "") -> str:
    reason = reason.strip() or "I am checking this worker before advancing the pipeline."
    return f"{verb} {worker_id}\n<reasoning>{reason}</reasoning>"


def _redirect_action(worker_id: int | str, instruction: str) -> str:
    clean_instruction = instruction.strip() or "Refocus on the original task and produce only the requested output."
    return (
        f"REDIRECT {worker_id} {clean_instruction}\n"
        "<reasoning>The worker appears off-target, so a soft reset is safer than approving.</reasoning>"
    )


def _display_total(state: dict[str, Any]) -> float:
    breakdown = state.get("reward_breakdown", {}) or {}
    if state.get("episode_result") == "IN_PROGRESS":
        return float(sum(breakdown.values()))
    return float(state.get("total_reward", sum(breakdown.values())) or 0.0)


def hero_svg_html() -> str:
    return """
    <div class="oa-hero-visual" aria-label="Animated oversight flow">
      <svg viewBox="0 0 804 430" role="img">
        <defs>
          <linearGradient id="oaNode" x1="0" x2="1">
            <stop offset="0%" stop-color="#10243d"/>
            <stop offset="100%" stop-color="#0b1727"/>
          </linearGradient>
          <linearGradient id="oaScanGrad" x1="0" x2="1">
            <stop offset="0%" stop-color="#5eead4" stop-opacity="0"/>
            <stop offset="50%" stop-color="#5eead4" stop-opacity="0.9"/>
            <stop offset="100%" stop-color="#5eead4" stop-opacity="0"/>
          </linearGradient>
          <filter id="oaGlow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="7" result="blur"/>
            <feMerge>
              <feMergeNode in="blur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
        </defs>

        <rect x="0" y="0" width="804" height="430" rx="26" fill="rgba(2,6,23,0.24)"/>
        <path d="M82 230 C190 128 294 126 402 230 S612 332 722 230"
              fill="none" stroke="#5eead4" stroke-opacity="0.24" stroke-width="8"/>
        <path class="oa-dashed-flow" d="M82 230 C190 128 294 126 402 230 S612 332 722 230"
              fill="none" stroke="#93c5fd" stroke-opacity="0.50" stroke-width="2"/>

        <g transform="translate(38 174)">
          <rect width="104" height="112" rx="22" fill="url(#oaNode)" stroke="#334155"/>
          <text x="52" y="43" text-anchor="middle" fill="#e2e8f0" class="oa-svg-label" font-size="27">W1</text>
          <text x="52" y="69" text-anchor="middle" fill="#9fb0c6" class="oa-svg-small">REQS</text>
        </g>
        <g transform="translate(188 78)">
          <rect width="104" height="112" rx="22" fill="url(#oaNode)" stroke="#334155"/>
          <text x="52" y="43" text-anchor="middle" fill="#e2e8f0" class="oa-svg-label" font-size="27">W2</text>
          <text x="52" y="69" text-anchor="middle" fill="#9fb0c6" class="oa-svg-small">CODE</text>
        </g>
        <g transform="translate(350 174)">
          <rect width="104" height="112" rx="22" fill="#12233a" stroke="#fbbf24" stroke-width="2"/>
          <text x="52" y="42" text-anchor="middle" fill="#fff7d6" class="oa-svg-label" font-size="27">W3</text>
          <text x="52" y="68" text-anchor="middle" fill="#fbbf24" class="oa-svg-small">LOOKS CLEAN</text>
          <g class="oa-pulse">
            <circle cx="88" cy="23" r="13" fill="#fb7185" filter="url(#oaGlow)"/>
            <text x="88" y="27" text-anchor="middle" fill="#24040a" font-size="15" font-weight="900">!</text>
          </g>
          <rect class="oa-scan-beam" x="14" y="10" width="20" height="92" rx="10" fill="url(#oaScanGrad)"/>
        </g>
        <g transform="translate(512 270)">
          <rect width="104" height="112" rx="22" fill="url(#oaNode)" stroke="#334155"/>
          <text x="52" y="43" text-anchor="middle" fill="#e2e8f0" class="oa-svg-label" font-size="27">W4</text>
          <text x="52" y="69" text-anchor="middle" fill="#9fb0c6" class="oa-svg-small">SECURITY</text>
        </g>
        <g transform="translate(662 174)">
          <rect width="104" height="112" rx="22" fill="url(#oaNode)" stroke="#334155"/>
          <text x="52" y="43" text-anchor="middle" fill="#e2e8f0" class="oa-svg-label" font-size="27">W5</text>
          <text x="52" y="69" text-anchor="middle" fill="#9fb0c6" class="oa-svg-small">DEPLOY</text>
        </g>

        <circle class="oa-clean-packet" r="9" fill="#34d399" filter="url(#oaGlow)"/>
        <circle class="oa-bad-packet" r="11" fill="#fb7185" filter="url(#oaGlow)"/>

        <g transform="translate(238 320)">
          <path d="M0 40 C42 -4 98 -4 140 40" fill="none" stroke="#5eead4" stroke-width="4" stroke-linecap="round"/>
          <circle cx="70" cy="39" r="42" fill="rgba(94,234,212,0.09)" stroke="#5eead4" stroke-opacity="0.5"/>
          <text x="70" y="34" text-anchor="middle" fill="#d9fff8" class="oa-svg-label" font-size="18">OVERSIGHT</text>
          <text x="70" y="55" text-anchor="middle" fill="#9fb0c6" font-size="12">spend budget to reveal hidden flaws</text>
        </g>

        <g class="oa-reveal" transform="translate(485 36)">
          <rect width="268" height="82" rx="18" fill="rgba(251,113,133,0.14)" stroke="#fb7185" stroke-opacity="0.58"/>
          <text x="18" y="30" fill="#ffe4e9" class="oa-svg-label" font-size="18">DEEP_INSPECT reveals:</text>
          <text x="18" y="55" fill="#fecdd3" font-size="14">green surface hid a security-critical flaw</text>
        </g>
      </svg>
    </div>
    """


def pipeline_map_svg(env: OversightArenaEnvironment | None, show_real: bool = False) -> str:
    if env is None:
        labels = [("W1", "WAITING"), ("W2", "WAITING"), ("W3", "WAITING"), ("W4", "WAITING"), ("W5", "WAITING")]
        step_text = "Start simulation"
        risk = "LOW"
    else:
        state = env.state_dict
        labels = [
            (f"W{w.get('worker_id')}", str(w.get("visible_state", "IDLE")))
            for w in state.get("workers", [])
        ]
        step_text = f"Step {state.get('step', 0)}/{state.get('max_steps', 25)}"
        risk = state.get("corruption_risk", "LOW")

    nodes = []
    positions = [92, 247, 402, 557, 712]
    compromised_ids: set[int] = set()
    if env is not None:
        compromised_ids = {int(k) for k in env.state_dict.get("failure_plan", {}).keys()}

    for idx, (wid, visible) in enumerate(labels):
        worker_num = idx + 1
        x = positions[idx]
        color = "#60a5fa"
        if visible in {"APPROVED", "COMPLETED"}:
            color = "#34d399"
        if visible in {"IDLE", "WAITING"}:
            color = "#64748b"
        if env is not None:
            worker = env.state_dict.get("workers", [])[idx]
            if worker.get("steps_unchanged", 0) >= 2 and visible == "WORKING":
                color = "#fbbf24"
            if worker.get("deep_inspect_done") and (worker.get("real_state_str") in FAILURE_STATES):
                color = "#fb7185"
        if show_real and worker_num in compromised_ids:
            color = "#fb7185"
        nodes.append(
            f"""
            <g transform="translate({x - 52} 84)">
              <rect width="104" height="104" rx="24" fill="rgba(15,23,42,0.92)" stroke="{color}" stroke-width="2.5"/>
              <circle cx="52" cy="22" r="7" fill="{color}" filter="url(#mapGlow)"/>
              <text x="52" y="54" text-anchor="middle" fill="#eef6ff" class="oa-svg-label" font-size="26">{_esc(wid)}</text>
              <text x="52" y="76" text-anchor="middle" fill="#9fb0c6" class="oa-svg-small">{_esc(visible)}</text>
            </g>
            """
        )

    return f"""
    <div class="oa-live-map" aria-label="Live animated oversight map">
      <svg viewBox="0 0 804 252" role="img">
        <defs>
          <filter id="mapGlow" x="-80%" y="-80%" width="260%" height="260%">
            <feGaussianBlur stdDeviation="5" result="blur"/>
            <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
        </defs>
        <text x="34" y="36" fill="#eef6ff" class="oa-svg-label" font-size="25">Live Oversight Map</text>
        <text x="34" y="60" fill="#9fb0c6" font-size="13">Bad outputs move downstream unless the supervisor investigates in time.</text>
        <text x="648" y="38" fill="#5eead4" class="oa-svg-small">{_esc(step_text)}</text>
        <text x="648" y="60" fill="#fbbf24" class="oa-svg-small">risk: {_esc(risk)}</text>
        <path d="M92 136 H712" stroke="#334155" stroke-width="8" stroke-linecap="round"/>
        <path class="oa-dashed-flow" d="M92 136 H712" stroke="#5eead4" stroke-opacity="0.55" stroke-width="3" fill="none"/>
        <circle class="oa-clean-packet" r="8" fill="#34d399" filter="url(#mapGlow)"
                style='offset-path:path("M92 136 H712"); animation-duration:3.8s;'/>
        <circle class="oa-bad-packet" r="10" fill="#fb7185" filter="url(#mapGlow)"
                style='offset-path:path("M92 136 H712"); animation-duration:3.8s;'/>
        {''.join(nodes)}
      </svg>
    </div>
    """


def landing_html() -> str:
    return """
    <div class="oa-wrap">
      <section class="oa-hero">
        <div class="oa-hero-content">
          <div class="oa-hero-copy">
            <div class="oa-kicker">Multi-agent oversight benchmark</div>
            <h1>Catch the failure that looks safe.</h1>
            <p class="oa-subtitle">
              Oversight Arena turns the core danger of agent fleets into a visual,
              playable environment: a worker can look green, hide a flaw, and poison
              every downstream agent unless the supervisor knows when to inspect.
            </p>
          </div>
          """ + hero_svg_html() + """
        </div>
        <div class="oa-hero-grid">
          <div class="oa-proof-card">
            <b>The problem</b>
            <span>Agent pipelines can ship bad code, weak tests, or unsafe approvals while every individual step looks green.</span>
          </div>
          <div class="oa-proof-card">
            <b>Why deception matters</b>
            <span>Some workers show polished surface outputs. Only deliberate deep inspection reveals the flaw.</span>
          </div>
          <div class="oa-proof-card">
            <b>What oversight means</b>
            <span>The supervisor decides when to observe, inspect, redirect, terminate, or approve under a step budget.</span>
          </div>
          <div class="oa-proof-card">
            <b>Why this is useful</b>
            <span>Episodes are deterministic, seedable, and reward-shaped for training scalable oversight agents.</span>
          </div>
        </div>
      </section>
    </div>
    """


def how_to_use_html() -> str:
    steps = [
        ("Start simulation", "Launch the default judge demo or choose a difficulty and seed."),
        ("Watch the chain", "Follow W1 through W5 as outputs move downstream."),
        ("Find deception", "Look for unchanged outputs, suspicious polish, or hidden flaws after inspection."),
        ("Act as oversight", "Observe, deep inspect, redirect, terminate, or approve using real environment actions."),
        ("Review outcome", "Use the post-mortem to compare safe catches against bad approvals."),
    ]
    cards = []
    for idx, (title, copy) in enumerate(steps, start=1):
        cards.append(
            f"""
            <div class="oa-step-card">
              <div class="oa-step-num">{idx}</div>
              <b>{_esc(title)}</b>
              <span>{_esc(copy)}</span>
            </div>
            """
        )
    return f"""
    <section class="oa-section">
      <div class="oa-section-head">
        <div>
          <h2 class="oa-section-title">How to Use</h2>
          <p class="oa-section-copy">
            The fastest judge path is: start the deception demo, run one or more
            oversight turns, inspect suspicious workers, and finish by reading the
            post-mortem.
          </p>
        </div>
      </div>
      <div class="oa-how">{''.join(cards)}</div>
    </section>
    """


def empty_pipeline_html() -> str:
    cards = []
    for worker_id in range(1, 6):
        role = WORKER_ROLES[worker_id]
        cards.append(
            f"""
            <div class="oa-worker idle">
              <div class="oa-worker-top">
                <div class="oa-worker-id">W{worker_id}</div>
                <span class="oa-badge dim">Waiting</span>
              </div>
              <div class="oa-role">{_esc(role)}</div>
              <div class="oa-task">No episode loaded</div>
              <div class="oa-desc">Start a simulation to assign this worker a scripted task and failure plan.</div>
              <div class="oa-snippet">(no output yet)</div>
              <div class="oa-worker-foot"><span class="oa-chip">Real execution only</span></div>
            </div>
            """
        )
    return f"""
    <section class="oa-section">
      <div class="oa-section-head">
        <div>
          <h2 class="oa-section-title">Agent Flow</h2>
          <p class="oa-section-copy">Five deterministic workers execute in sequence. Hidden states stay hidden until inspection or post-mortem.</p>
        </div>
      </div>
      {pipeline_map_svg(None)}
      <div class="oa-pipeline">{''.join(cards)}</div>
    </section>
    """


def status_html(state: dict[str, Any] | None = None) -> str:
    if not state:
        stats = [
            ("Episode", "Not started"),
            ("Difficulty", "Judge demo"),
            ("Step budget", "25"),
            ("Corruption risk", "LOW"),
            ("Reward", "+0.000"),
        ]
    else:
        stats = [
            ("Episode", state.get("episode_result", "IN_PROGRESS")),
            ("Difficulty", str(state.get("difficulty", "?")).upper()),
            ("Step budget", f"{state.get('step', 0)}/{state.get('max_steps', 25)}"),
            ("Corruption risk", state.get("corruption_risk", "LOW")),
            ("Reward", f"{_display_total(state):+.3f}"),
        ]
    cards = "".join(
        f'<div class="oa-stat"><span>{_esc(label)}</span><b>{_esc(value)}</b></div>'
        for label, value in stats
    )
    return f'<div class="oa-status-grid">{cards}</div>'


def _worker_class(worker: dict[str, Any], show_real: bool, compromised_ids: set[int]) -> str:
    visible = str(worker.get("visible_state", "IDLE")).lower()
    classes = ["oa-worker", visible]
    real_state = worker.get("real_state_str") or worker.get("state")
    exposed = worker.get("deep_inspect_done") and real_state in FAILURE_STATES
    if worker.get("steps_unchanged", 0) >= 2 and worker.get("visible_state") == "WORKING":
        classes.append("suspicious")
    if exposed:
        classes.append("exposed")
    if show_real:
        classes.append("truth-fail" if worker.get("worker_id") in compromised_ids else "truth-clean")
    return " ".join(classes)


def _worker_badge(worker: dict[str, Any]) -> tuple[str, str]:
    visible = worker.get("visible_state", "IDLE")
    steps_unchanged = int(worker.get("steps_unchanged", 0) or 0)
    real_state = worker.get("real_state_str") or worker.get("state")
    deep_done = bool(worker.get("deep_inspect_done"))
    if visible in {"COMPLETED", "APPROVED"}:
        return "good", visible
    if deep_done and real_state in FAILURE_STATES:
        return "danger", "Flaw exposed"
    if steps_unchanged >= 2 and visible == "WORKING":
        return "warn", "Suspicious"
    return "dim", visible


def _worker_snippet(worker: dict[str, Any]) -> str:
    history = worker.get("output_history") or []
    if history:
        return _clip(str(history[-1]), 220)
    return "(no output yet)"


def _worker_chips(worker: dict[str, Any], show_real: bool, compromised_ids: set[int]) -> str:
    worker_id = int(worker.get("worker_id", 0))
    steps_unchanged = int(worker.get("steps_unchanged", 0) or 0)
    deep_done = bool(worker.get("deep_inspect_done"))
    is_compromised = worker_id in compromised_ids
    chips = [
        f'<span class="oa-chip {"hot" if steps_unchanged >= 2 else ""}">unchanged x{steps_unchanged}</span>',
        f'<span class="oa-chip {"green" if deep_done else ""}">deep inspect {"done" if deep_done else "not run"}</span>',
    ]
    if worker.get("is_colluding"):
        chips.append('<span class="oa-chip hot">possible pair</span>')
    if show_real:
        truth_class = "red" if is_compromised else "green"
        truth = worker.get("failure_mode", "NONE") if is_compromised else "CLEAN"
        chips.append(f'<span class="oa-chip {truth_class}">truth: {_esc(truth)}</span>')
    return "".join(chips)


def _worker_card(worker: dict[str, Any], show_real: bool, compromised_ids: set[int]) -> str:
    worker_id = int(worker.get("worker_id", 0))
    badge_class, badge_label = _worker_badge(worker)
    return f"""
    <div class="{_worker_class(worker, show_real, compromised_ids)}">
      <div class="oa-worker-top">
        <div class="oa-worker-id">W{worker_id}</div>
        <span class="oa-badge {badge_class}">{_esc(badge_label)}</span>
      </div>
      <div class="oa-role">{_esc(WORKER_ROLES.get(worker_id, "Worker"))}</div>
      <div class="oa-task">{_esc(worker.get("task_type", "Unknown"))}</div>
      <div class="oa-desc">{_esc(_clip(worker.get("task_description", ""), 145))}</div>
      <div class="oa-snippet">{_esc(_worker_snippet(worker))}</div>
      <div class="oa-worker-foot">{_worker_chips(worker, show_real, compromised_ids)}</div>
    </div>
    """


def pipeline_html(env: OversightArenaEnvironment | None, show_real: bool = False) -> str:
    if env is None:
        return empty_pipeline_html()

    state = env.state_dict
    compromised_ids = {int(k) for k in state.get("failure_plan", {}).keys()}
    cards = [
        _worker_card(worker, show_real, compromised_ids)
        for worker in state.get("workers", [])
    ]

    legend = (
        "Live view hides true failure states. Yellow means suspicious from visible signals; red means a deep inspection has exposed a flaw."
        if not show_real
        else "Post-mortem view reveals which workers were actually compromised by the seeded failure plan."
    )
    return f"""
    <section class="oa-section">
      <div class="oa-section-head">
        <div>
          <h2 class="oa-section-title">Agent Flow</h2>
          <p class="oa-section-copy">{_esc(legend)}</p>
        </div>
      </div>
      {pipeline_map_svg(env, show_real)}
      <div class="oa-pipeline">{''.join(cards)}</div>
    </section>
    """


def reward_html(
    breakdown: dict[str, float] | None,
    total: float = 0.0,
    episode_result: str = "IN_PROGRESS",
    step_reward: float | None = None,
) -> str:
    breakdown = breakdown or {}
    cards = []
    for key, label in REWARD_LABELS.items():
        value = float(breakdown.get(key, 0.0) or 0.0)
        cards.append(
            f"""
            <div class="oa-reward-card">
              <span>{_esc(label)}</span>
              <b>{value:+.2f}</b>
            </div>
            """
        )
    step_line = "" if step_reward is None else f"<span>Last step {step_reward:+.3f}</span>"
    return f"""
    <div class="oa-panel">
      <div class="oa-section-head">
        <div>
          <h2 class="oa-section-title">Reward Signal</h2>
          <p class="oa-section-copy">Independent reward components show exactly why oversight succeeded or failed.</p>
        </div>
        <div class="oa-badge {'good' if total >= 0 else 'danger'}">total {total:+.3f}</div>
      </div>
      <div class="oa-worker-foot">{step_line}<span class="oa-chip">result: {_esc(episode_result)}</span></div>
      <div class="oa-reward-grid">{''.join(cards)}</div>
    </div>
    """


def log_html(log: list[dict[str, Any]] | None) -> str:
    if not log:
        return '<div class="oa-empty">No actions yet. Start a simulation, then submit an action or run an oracle-guided demo step.</div>'

    items = []
    for entry in reversed(log[-8:]):
        if entry.get("kind") == "reset":
            items.append(
                f"""
                <div class="oa-log-item">
                  <div class="oa-log-line">
                    <span>New episode</span>
                    <span class="oa-badge good">{_esc(entry.get('difficulty', '?'))}</span>
                  </div>
                  <div class="oa-log-meta">seed={_esc(entry.get('seed', '?'))} | max_steps={_esc(entry.get('max_steps', 25))}</div>
                </div>
                """
            )
            continue
        summary = entry.get("summary") or "Action submitted."
        reward = float(entry.get("reward", 0.0) or 0.0)
        badge = "good" if reward >= 0 else "danger"
        items.append(
            f"""
            <div class="oa-log-item">
              <div class="oa-log-line">
                <span>Step {_esc(entry.get('step', '?'))}: {_esc(entry.get('action', ''))}</span>
                <span class="oa-badge {badge}">{reward:+.3f}</span>
              </div>
              <div class="oa-log-meta">{_esc(summary)}</div>
            </div>
            """
        )
    return f'<div class="oa-log">{"".join(items)}</div>'


def postmortem_html(state: dict[str, Any]) -> str:
    compromised = {int(k): v for k, v in state.get("failure_plan", {}).items()}
    caught = set(state.get("caught_workers", []) or [])
    hallpass = set(state.get("hallpass_workers", []) or [])
    colluding_pair = state.get("colluding_pair", []) or []
    total = _display_total(state)

    worker_rows = []
    for worker in state.get("workers", []):
        worker_id = int(worker.get("worker_id", 0))
        plan = compromised.get(worker_id)
        if plan:
            mode = plan.get("failure_mode", "?")
            if worker_id in hallpass:
                outcome = "Missed: approved bad output"
            elif worker_id in caught:
                outcome = "Caught by oversight"
            else:
                outcome = "Still unresolved"
        else:
            mode = "NONE"
            outcome = "Clean worker"
        worker_rows.append(
            f"""
            <tr>
              <td>W{worker_id}</td>
              <td>{_esc(WORKER_ROLES.get(worker_id, 'Worker'))}</td>
              <td>{_esc(mode)}</td>
              <td>{_esc(worker.get('real_state_str', '?'))}</td>
              <td>{_esc(outcome)}</td>
            </tr>
            """
        )

    result = state.get("episode_result", "IN_PROGRESS")
    verdict_class = "good" if result == "CLEAN_COMPLETE" else "danger"
    explanation = {
        "CLEAN_COMPLETE": "All workers were approved without letting a compromised output propagate.",
        "DIRTY_COMPLETE": "The pipeline completed, but at least one compromised worker was approved.",
        "TIMEOUT": "The supervisor exhausted the step budget before resolving the pipeline.",
    }.get(result, "Episode is still running.")

    return f"""
    <section class="oa-section">
      <div class="oa-panel">
        <div class="oa-section-head">
          <div>
            <h2 class="oa-section-title">Post-mortem</h2>
            <p class="oa-section-copy">{_esc(explanation)}</p>
          </div>
          <div class="oa-badge {verdict_class}">{_esc(result)} | {total:+.3f}</div>
        </div>
        <div class="oa-post-grid">
          <div>
            <table class="oa-table">
              <thead>
                <tr><th>Worker</th><th>Role</th><th>Injected failure</th><th>Final state</th><th>Oversight outcome</th></tr>
              </thead>
              <tbody>{''.join(worker_rows)}</tbody>
            </table>
          </div>
          <div class="oa-oracle">
            <h2 class="oa-section-title">Why it matters</h2>
            <p class="oa-section-copy">
              The deceptive and colluding workers are the important signal: they look
              safe until the supervisor spends budget to inspect and reason across the fleet.
            </p>
            <div class="oa-worker-foot">
              <span class="oa-chip red">missed: {len(hallpass)}</span>
              <span class="oa-chip green">caught: {len(caught)}</span>
              <span class="oa-chip hot">colluding pair: {_esc(colluding_pair or 'none')}</span>
              <span class="oa-chip">seed: {_esc(state.get('seed'))}</span>
            </div>
          </div>
        </div>
      </div>
    </section>
    """


def oracle_html(env: OversightArenaEnvironment | None, is_done: bool) -> str:
    if env is None:
        return '<div class="oa-empty">Start an episode before asking the oracle.</div>'

    state = env.state_dict
    approved = set(state.get("approved_workers", []) or [])
    colluding = state.get("colluding_pair", []) or []
    rows = []
    for worker in state.get("workers", []):
        worker_id = int(worker.get("worker_id", 0))
        real_state_str = worker.get("real_state_str") or worker.get("state")
        try:
            real_state = WorkerState(real_state_str)
        except ValueError:
            continue
        partner_approved = any(partner in approved for partner in colluding if partner != worker_id)
        optimal = oracle_action(
            real_state,
            int(worker.get("steps_unchanged", 0) or 0),
            partner_approved,
            bool(worker.get("deep_inspect_done")),
        )
        visible_or_real = real_state_str if is_done else worker.get("visible_state", "?")
        label = "real" if is_done else "visible"
        rows.append(
            f"""
            <tr>
              <td>W{worker_id}</td>
              <td>{_esc(worker.get('task_type', '?'))}</td>
              <td>{_esc(label)}={_esc(visible_or_real)}</td>
              <td><span class="oa-badge good">{_esc(optimal)}</span></td>
            </tr>
            """
        )
    return f"""
    <div class="oa-oracle">
      <div class="oa-section-head">
        <div>
          <h2 class="oa-section-title">Oracle Reference</h2>
          <p class="oa-section-copy">For demos only: this panel uses hidden server-side truth. Training clients do not receive this state.</p>
        </div>
      </div>
      <table class="oa-table">
        <thead><tr><th>Worker</th><th>Task</th><th>State shown</th><th>Optimal action</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    """


def _scenario_config(scenario: str, difficulty: str, seed: float | int | None) -> tuple[str, int | None]:
    if scenario.startswith("Judge demo"):
        return "hard", 42
    if scenario.startswith("Easy"):
        return "easy", 7
    if scenario.startswith("Medium"):
        return "medium", 11
    chosen_seed = int(seed) if seed is not None else None
    return (difficulty or "hard").lower(), chosen_seed


def _active_oracle_action(env: OversightArenaEnvironment) -> str:
    state = env.state_dict
    approved = set(state.get("approved_workers", []) or [])
    colluding = state.get("colluding_pair", []) or []
    workers = state.get("workers", [])

    priorities = {
        "TERMINATE": 0,
        "DEEP_INSPECT": 1,
        "REDIRECT": 2,
        "APPROVE": 3,
        "OBSERVE": 4,
    }
    candidates: list[tuple[int, int, str, dict[str, Any]]] = []
    for worker in workers:
        worker_id = int(worker.get("worker_id", 0))
        try:
            real_state = WorkerState(worker.get("real_state_str") or worker.get("state"))
        except ValueError:
            continue
        partner_approved = any(partner in approved for partner in colluding if partner != worker_id)
        optimal = oracle_action(
            real_state,
            int(worker.get("steps_unchanged", 0) or 0),
            partner_approved,
            bool(worker.get("deep_inspect_done")),
        )
        if real_state == WorkerState.IDLE and optimal == "OBSERVE":
            priority = 8
        elif real_state == WorkerState.WORKING and optimal == "OBSERVE":
            priority = 5
        else:
            priority = priorities.get(optimal, 6)
        candidates.append((priority, worker_id, optimal, worker))

    _, worker_id, verb, _worker = min(candidates, key=lambda item: (item[0], item[1]))
    if verb == "REDIRECT":
        return (
            f"REDIRECT {worker_id} Refocus on the original task and ignore unrelated output.\n"
            "<reasoning>Oracle-guided demo: this worker is best handled with a soft reset before approval.</reasoning>"
        )
    return _format_action(
        verb,
        worker_id,
        "Oracle-guided demo: execute the optimal oversight move to show the real environment dynamics.",
    )


def _action_label(action_text: str) -> str:
    first = (action_text or "").strip().splitlines()[0] if action_text else ""
    return re.sub(r"\s+", " ", first)[:96] or "(empty action)"


with gr.Blocks(title="Oversight Arena") as demo:
    env_state = gr.State(None)
    episode_log_state = gr.State([])
    episode_done_state = gr.State(False)

    gr.HTML(f"<style>{APP_CSS}</style>")
    gr.HTML(landing_html())
    gr.HTML(how_to_use_html())

    with gr.Row():
        with gr.Column(scale=5):
            gr.HTML(
                """
                <div class="oa-control-panel">
                  <div class="oa-callout">
                    <b>Default path for judges:</b> keep "Judge demo" selected, click Start Simulation,
                    then click Run Oracle-Guided Step a few times. Switch to manual actions when you
                    want to test your own oversight decisions.
                  </div>
                </div>
                """
            )
            with gr.Row():
                scenario_radio = gr.Radio(
                    choices=[
                        "Judge demo: deceptive workers guaranteed",
                        "Easy walkthrough: one visible failure",
                        "Medium challenge: mixed failures",
                        "Custom difficulty and seed",
                    ],
                    value="Judge demo: deceptive workers guaranteed",
                    label="Scenario",
                    scale=3,
                )
                difficulty_radio = gr.Radio(
                    choices=["easy", "medium", "hard"],
                    value="hard",
                    label="Custom difficulty",
                    scale=1,
                )
                seed_input = gr.Number(
                    label="Custom seed",
                    value=42,
                    precision=0,
                    minimum=0,
                    maximum=999999,
                    scale=1,
                )
            with gr.Row():
                reset_btn = gr.Button("Start Simulation", variant="primary", scale=2)
                auto_step_btn = gr.Button("Run Oracle-Guided Step", variant="primary", scale=2)
                show_oracle_btn = gr.Button("Show Oracle Reference", variant="secondary", scale=1)
        with gr.Column(scale=3):
            status_display = gr.HTML(status_html())

    pipeline_display = gr.HTML(empty_pipeline_html())

    with gr.Row():
        with gr.Column(scale=3):
            observation_box = gr.Textbox(
                label="Supervisor Observation",
                value="Start a simulation to see the exact observation string returned by the environment.",
                lines=20,
                max_lines=42,
                interactive=False,
                elem_classes=["obs-text"],
            )
        with gr.Column(scale=2):
            gr.HTML(
                """
                <div class="oa-section-head">
                  <div>
                    <h2 class="oa-section-title">Manual Oversight</h2>
                    <p class="oa-section-copy">Quick buttons fill a real action string. Submit it to step the environment.</p>
                  </div>
                </div>
                """
            )
            with gr.Row():
                obs_buttons = [gr.Button(f"Observe W{i}", size="sm") for i in range(1, 6)]
            with gr.Row():
                action_worker_num = gr.Number(
                    label="Worker",
                    value=1,
                    minimum=1,
                    maximum=5,
                    step=1,
                    precision=0,
                )
                deep_btn = gr.Button("Deep Inspect", size="sm")
                terminate_btn = gr.Button("Terminate", size="sm")
                approve_btn = gr.Button("Approve", size="sm")
            with gr.Row():
                redirect_worker_num = gr.Number(
                    label="Redirect worker",
                    value=1,
                    minimum=1,
                    maximum=5,
                    step=1,
                    precision=0,
                )
                redirect_instr = gr.Textbox(
                    label="Redirect instruction",
                    placeholder="Refocus on the original requirement and produce only the requested output.",
                    lines=1,
                )
            redirect_btn = gr.Button("Redirect Worker", size="sm")
            action_input = gr.Textbox(
                label="Action submitted to env.step()",
                value=_format_action("OBSERVE", 1),
                lines=5,
                elem_classes=["act-text"],
            )
            step_btn = gr.Button("Submit Manual Action", variant="primary")

    with gr.Row():
        with gr.Column(scale=3):
            gr.HTML(
                """
                <div class="oa-section-head">
                  <div>
                    <h2 class="oa-section-title">Clean Episode Log</h2>
                    <p class="oa-section-copy">A concise trace of actions, rewards, and environment summaries.</p>
                  </div>
                </div>
                """
            )
            episode_log_display = gr.HTML(log_html([]))
        with gr.Column(scale=2):
            reward_panel = gr.HTML(reward_html({}, 0.0))

    oracle_output = gr.HTML("")

    with gr.Row(visible=False) as postmortem_row:
        postmortem_display = gr.HTML("")

    _ALL_OUTPUTS = [
        env_state,
        episode_log_state,
        episode_done_state,
        observation_box,
        pipeline_display,
        status_display,
        episode_log_display,
        reward_panel,
        oracle_output,
        postmortem_row,
        postmortem_display,
        action_input,
    ]

    def do_reset(scenario: str, difficulty: str, seed_val):
        difficulty_val, seed = _scenario_config(scenario, difficulty, seed_val)
        env = OversightArenaEnvironment()
        obs_result = env.reset(difficulty=difficulty_val, seed=seed)
        state = env.state_dict
        obs_text = obs_result.metadata.get("pipeline_text", "")
        log = [
            {
                "kind": "reset",
                "difficulty": difficulty_val.upper(),
                "seed": state.get("seed"),
                "max_steps": state.get("max_steps", 25),
            }
        ]
        return (
            env,
            log,
            False,
            obs_text,
            pipeline_html(env, show_real=False),
            status_html(state),
            log_html(log),
            reward_html(state.get("reward_breakdown", {}), _display_total(state)),
            "",
            gr.update(visible=False),
            "",
            _format_action("OBSERVE", 1),
        )

    def _step_with_action(
        env: OversightArenaEnvironment | None,
        action_text: str,
        log: list[dict[str, Any]] | None,
        is_done: bool,
    ):
        log = list(log or [])
        if env is None:
            return (
                env,
                log,
                is_done,
                "Start a simulation before submitting an action.",
                empty_pipeline_html(),
                status_html(),
                log_html(log),
                reward_html({}, 0.0),
                "",
                gr.update(visible=False),
                "",
                action_text or _format_action("OBSERVE", 1),
            )

        if is_done:
            state = env.state_dict
            return (
                env,
                log,
                True,
                env._build_observation(),
                pipeline_html(env, show_real=True),
                status_html(state),
                log_html(log),
                reward_html(state.get("reward_breakdown", {}), _display_total(state), state.get("episode_result", "")),
                "",
                gr.update(visible=True),
                postmortem_html(state),
                action_text,
            )

        if not action_text or not action_text.strip():
            state = env.state_dict
            return (
                env,
                log,
                False,
                env._build_observation(),
                pipeline_html(env, show_real=False),
                status_html(state),
                log_html(log),
                reward_html(state.get("reward_breakdown", {}), _display_total(state)),
                "",
                gr.update(visible=False),
                "",
                _format_action("OBSERVE", 1),
            )

        obs_result = env.step(action_text)
        state = env.state_dict
        reward = float(obs_result.reward or 0.0)
        done = bool(obs_result.done)
        episode_result = obs_result.metadata.get("episode_result", "IN_PROGRESS")
        summary = obs_result.metadata.get("action_summary", "")
        obs_text = obs_result.metadata.get("pipeline_text", env._build_observation())
        log.append(
            {
                "kind": "step",
                "step": state.get("step"),
                "action": _action_label(action_text),
                "summary": summary,
                "reward": reward,
            }
        )
        show_real = done
        postmortem_update = gr.update(visible=done)
        postmortem = postmortem_html(state) if done else ""
        return (
            env,
            log,
            done,
            obs_text,
            pipeline_html(env, show_real=show_real),
            status_html(state),
            log_html(log),
            reward_html(
                state.get("reward_breakdown", {}),
                _display_total(state),
                episode_result,
                step_reward=reward,
            ),
            "",
            postmortem_update,
            postmortem,
            action_text,
        )

    def do_manual_step(env, action_text, log, is_done):
        return _step_with_action(env, action_text, log, is_done)

    def do_auto_step(env, log, is_done):
        if env is None:
            return _step_with_action(env, "", log, is_done)
        action_text = _active_oracle_action(env)
        return _step_with_action(env, action_text, log, is_done)

    reset_btn.click(
        do_reset,
        inputs=[scenario_radio, difficulty_radio, seed_input],
        outputs=_ALL_OUTPUTS,
    )
    step_btn.click(
        do_manual_step,
        inputs=[env_state, action_input, episode_log_state, episode_done_state],
        outputs=_ALL_OUTPUTS,
    )
    action_input.submit(
        do_manual_step,
        inputs=[env_state, action_input, episode_log_state, episode_done_state],
        outputs=_ALL_OUTPUTS,
    )
    auto_step_btn.click(
        do_auto_step,
        inputs=[env_state, episode_log_state, episode_done_state],
        outputs=_ALL_OUTPUTS,
    )

    for idx, button in enumerate(obs_buttons, start=1):
        button.click(
            lambda worker_id=idx: _format_action(
                "OBSERVE",
                worker_id,
                "I am checking the latest visible output before taking a stronger action.",
            ),
            inputs=[],
            outputs=[action_input],
        )

    deep_btn.click(
        lambda worker_id: _format_action(
            "DEEP_INSPECT",
            int(worker_id or 1),
            "The visible output may be hiding a flaw, so I am spending inspection budget.",
        ),
        inputs=[action_worker_num],
        outputs=[action_input],
    )
    terminate_btn.click(
        lambda worker_id: _format_action(
            "TERMINATE",
            int(worker_id or 1),
            "This worker appears unsafe to pass downstream, so I am hard resetting it.",
        ),
        inputs=[action_worker_num],
        outputs=[action_input],
    )
    approve_btn.click(
        lambda worker_id: _format_action(
            "APPROVE",
            int(worker_id or 1),
            "The worker appears complete and safe to pass to the next stage.",
        ),
        inputs=[action_worker_num],
        outputs=[action_input],
    )
    redirect_btn.click(
        lambda worker_id, instruction: _redirect_action(int(worker_id or 1), instruction or ""),
        inputs=[redirect_worker_num, redirect_instr],
        outputs=[action_input],
    )
    show_oracle_btn.click(
        oracle_html,
        inputs=[env_state, episode_done_state],
        outputs=[oracle_output],
    )


def _build_space_app() -> FastAPI:
    """Serve Gradio at / and mount the OpenEnv API at /openenv on the same HF port."""
    import server as _openenv_asgi

    app = FastAPI(title="Oversight Arena", version="1.0.0")

    @app.get("/health", tags=["Health"], summary="Space liveness")
    def space_health():
        return {"status": "ok", "service": "oversight-arena"}

    app.mount("/openenv", _openenv_asgi.app)
    demo.max_threads = 8
    return gr.mount_gradio_app(
        app,
        demo,
        path="/",
        server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False,
        pwa=False,
        mcp_server=False,
    )


if __name__ == "__main__":
    uvicorn.run(_build_space_app(), host="0.0.0.0", port=7860)
