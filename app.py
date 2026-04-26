"""
Oversight Arena — Gradio judge interface.
Real environment execution, no fake demos.
"""

from __future__ import annotations

import html
import os
import re
from typing import Any

os.environ.setdefault("GRADIO_SSR_MODE", "false")

from oversight_arena.log_filters import install_asyncio_stale_loop_unraisable_filter

install_asyncio_stale_loop_unraisable_filter()

import gradio as gr
import uvicorn
from fastapi import FastAPI

from oversight_arena.environment import OversightArenaEnvironment
from oversight_arena.models import WorkerState
from oversight_arena.oracle import oracle_action


# ─────────────────────────────────────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────────────────────────────────────

APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital,wght@0,400;1,400&family=Geist:wght@300;400;500;600;700&family=Geist+Mono:wght@400;500;600&display=swap');

/* ─── Claude-light tokens ──────────────────────────────────────── */
:root {
  /* warm cream surfaces */
  --bg:        #faf6f1;
  --bg-2:      #f3ece1;
  --paper:     #ffffff;
  --ink:       #1a1614;
  --ink-soft:  #2c2622;
  --muted:     #6b5d52;
  --dim:       #a89b8e;
  --line:      rgba(26, 22, 20, 0.10);
  --line-soft: rgba(26, 22, 20, 0.06);

  /* Claude-warm accent palette (lighter, refined) */
  --orange:        #c96442;   /* terracotta — primary */
  --orange-soft:   #e9b89a;
  --orange-tint:   rgba(201, 100, 66, 0.08);
  --orange-line:   rgba(201, 100, 66, 0.22);

  --sage:          #5b8769;   /* clean / safe */
  --sage-tint:     rgba(91, 135, 105, 0.10);

  --crimson:       #b85c52;   /* failure / deception */
  --crimson-tint:  rgba(184, 92, 82, 0.10);
  --crimson-line:  rgba(184, 92, 82, 0.30);

  --amber:         #c79552;   /* suspicious */
  --amber-tint:    rgba(199, 149, 82, 0.12);

  --steel:         #6b8aa8;   /* working / info */
  --steel-tint:    rgba(107, 138, 168, 0.10);

  --plum:          #a07dd4;   /* redirected */

  /* type stacks */
  --display: 'Instrument Serif', 'Cormorant Garamond', Georgia, serif;
  --sans:    'Geist', -apple-system, sans-serif;
  --mono:    'Geist Mono', ui-monospace, monospace;

  --r:  10px;
  --r-sm: 6px;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

.gradio-container,
.gradio-container * {
  font-family: var(--sans) !important;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

.gradio-container {
  max-width: 1480px !important;
  background: var(--bg) !important;
  color: var(--ink-soft) !important;
  padding: 0 !important;
  overflow-x: hidden;
}
.gradio-container .main { padding: 0 !important; }

footer { display: none !important; }

/* neutralize gradio default block chrome on plain HTML / containers */
.gradio-container .prose { max-width: none !important; color: inherit !important; }
.gradio-container .prose h1,
.gradio-container .prose h2,
.gradio-container .prose h3,
.gradio-container .prose p { color: inherit !important; margin: 0 !important; }
.gradio-container .gr-block,
.gradio-container .gr-form,
.gradio-container .block,
.gradio-container .form {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
}
.gradio-container .gr-html,
.gradio-container .html-container {
  background: transparent !important;
  padding: 0 !important;
  border: none !important;
}
.gradio-container .gap,
.gradio-container .row,
.gradio-container .column { background: transparent !important; }

/* ─── soft warm grain backdrop ────────────────────────────────── */
.oa-root { position: relative; background: var(--bg); }
.oa-root::before {
  content: "";
  position: fixed;
  inset: 0;
  z-index: 0;
  pointer-events: none;
  background-image:
    radial-gradient(circle at 14% 8%, rgba(201,100,66,0.07), transparent 32%),
    radial-gradient(circle at 88% 18%, rgba(107,138,168,0.05), transparent 36%);
}

/* ─── HERO  ───────────────────────────────────────────────────── */
.oa-hero {
  position: relative;
  padding: 88px 64px 56px;
  display: grid;
  grid-template-columns: 1.05fr 1fr;
  gap: 56px;
  align-items: center;
  border-bottom: 1px solid var(--line);
}
.oa-hero-text { max-width: 560px; }

.oa-eyebrow {
  font-family: var(--mono) !important;
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--orange);
  margin-bottom: 28px;
  display: inline-flex;
  align-items: center;
  gap: 12px;
}
.oa-eyebrow::before {
  content: "";
  width: 26px; height: 1px;
  background: var(--orange);
  opacity: 0.55;
}

.oa-h1 {
  font-family: var(--display) !important;
  font-size: clamp(56px, 7.8vw, 104px);
  font-weight: 400;
  line-height: 0.96;
  letter-spacing: -0.025em;
  color: var(--ink);
  margin-bottom: 28px;
}
.oa-h1 em  { font-style: italic; color: var(--ink); }
.oa-h1 .hi { font-style: italic; color: var(--orange); font-family: var(--display); }

.oa-lead {
  font-size: 17px;
  font-weight: 400;
  line-height: 1.62;
  color: var(--muted);
  margin-bottom: 32px;
  max-width: 520px;
}

.oa-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 24px;
  font-family: var(--mono) !important;
  font-size: 11px;
  color: var(--muted);
  letter-spacing: 0.04em;
}
.oa-meta span { display: inline-flex; align-items: center; gap: 8px; }
.oa-meta span::before {
  content: "";
  width: 5px; height: 5px;
  border-radius: 50%;
  background: var(--orange);
}

.oa-hero-svg { position: relative; width: 100%; }
.oa-hero-svg svg { width: 100%; height: auto; display: block; overflow: visible; }

/* ─── stat strip ──────────────────────────────────────────────── */
.oa-stats {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  background: var(--bg-2);
  border-bottom: 1px solid var(--line);
}
.oa-stat { padding: 26px 30px; border-right: 1px solid var(--line); }
.oa-stat:last-child { border-right: none; }
.oa-stat-lbl {
  display: block;
  font-family: var(--mono) !important;
  font-size: 10px;
  font-weight: 500;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--dim);
  margin-bottom: 10px;
}
.oa-stat-val {
  display: block;
  font-family: var(--display) !important;
  font-size: 36px;
  font-weight: 400;
  font-style: italic;
  letter-spacing: -0.02em;
  color: var(--ink);
  line-height: 1;
}
.oa-stat-val.g { color: var(--sage); }
.oa-stat-val.r { color: var(--crimson); }
.oa-stat-val.w { color: var(--amber); }

/* ─── section frame ───────────────────────────────────────────── */
.oa-section { padding: 56px 64px; position: relative; }
.oa-section + .oa-section { padding-top: 0; }
.oa-section-head {
  display: flex;
  align-items: baseline;
  flex-wrap: wrap;
  gap: 16px 24px;
  margin-bottom: 28px;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--line-soft);
}
.oa-section-head .oa-section-num { flex: 0 0 auto; }
.oa-section-head .oa-section-title { flex: 1 1 auto; }
.oa-section-num {
  font-family: var(--mono) !important;
  font-size: 11px;
  letter-spacing: 0.2em;
  color: var(--orange);
  text-transform: uppercase;
}
.oa-section-title {
  font-family: var(--display) !important;
  font-size: 32px;
  font-weight: 400;
  font-style: italic;
  letter-spacing: -0.02em;
  color: var(--ink);
}
.oa-section-copy {
  font-size: 14px;
  font-weight: 400;
  color: var(--muted);
  line-height: 1.55;
  max-width: 540px;
  margin-left: auto;
  text-align: right;
}

/* ─── how-to steps  ───────────────────────────────────────────── */
.oa-steps {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 1px;
  background: var(--line);
  border: 1px solid var(--line);
  border-radius: var(--r);
  overflow: hidden;
}
.oa-step {
  padding: 28px 24px;
  background: var(--paper);
  transition: background 0.2s;
}
.oa-step:hover { background: var(--bg-2); }
.oa-step-n {
  font-family: var(--display) !important;
  font-size: 64px;
  font-weight: 400;
  font-style: italic;
  letter-spacing: -0.04em;
  color: var(--orange-soft);
  line-height: 1;
  margin-bottom: 16px;
}
.oa-step-title {
  font-family: var(--mono) !important;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--orange);
  margin-bottom: 10px;
}
.oa-step-body {
  font-size: 13px;
  line-height: 1.6;
  color: var(--muted);
}

/* ─── live worker grid ────────────────────────────────────────── */
.oa-nodes {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 14px;
}
.oa-node {
  position: relative;
  border: 1px solid var(--line);
  border-radius: var(--r);
  padding: 20px 16px 16px;
  background: var(--paper);
  overflow: hidden;
  transition: border-color 0.2s, box-shadow 0.2s, transform 0.2s;
}
.oa-node:hover { transform: translateY(-1px); box-shadow: 0 8px 24px rgba(26,22,20,0.06); }
.oa-node::before {
  content: "";
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: var(--dim);
  opacity: 0.35;
  transition: background 0.3s, opacity 0.3s;
}
.oa-node.working::before   { background: var(--steel); opacity: 1; }
.oa-node.completed::before { background: var(--sage); opacity: 1; }
.oa-node.approved::before  { background: var(--amber); opacity: 1; }
.oa-node.redirected::before{ background: var(--plum); opacity: 1; }

.oa-node.hot {
  border-color: var(--amber);
  box-shadow: 0 0 0 1px var(--amber-tint), 0 8px 24px rgba(199,149,82,0.12);
  animation: hotPulse 2s ease-in-out infinite;
}
.oa-node.hot::before { background: var(--amber); opacity: 1; }

.oa-node.exposed {
  border-color: var(--crimson);
  box-shadow: 0 0 0 1px var(--crimson-tint), 0 8px 24px rgba(184,92,82,0.18);
}
.oa-node.exposed::before { background: var(--crimson); opacity: 1; animation: redFlash 1s ease-in-out infinite; }

.oa-node.truth-fail  { border-color: var(--crimson); }
.oa-node.truth-clean { border-color: var(--orange); }

.oa-node-id {
  font-family: var(--display) !important;
  font-size: 56px;
  font-weight: 400;
  font-style: italic;
  letter-spacing: -0.04em;
  color: var(--ink);
  line-height: 0.95;
  margin-bottom: 4px;
}
.oa-node-role {
  font-family: var(--mono) !important;
  font-size: 9.5px;
  font-weight: 500;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--orange);
  margin-bottom: 14px;
}
.oa-node-badge {
  display: inline-block;
  padding: 4px 9px;
  border-radius: 4px;
  font-family: var(--mono) !important;
  font-size: 9.5px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  border: 1px solid var(--line);
  color: var(--muted);
  background: var(--bg);
  margin-bottom: 12px;
}
.oa-node-badge.g { border-color: rgba(91,135,105,0.4); color: var(--sage); background: var(--sage-tint); }
.oa-node-badge.r { border-color: var(--crimson-line);  color: var(--crimson); background: var(--crimson-tint); }
.oa-node-badge.w { border-color: rgba(199,149,82,0.4); color: var(--amber); background: var(--amber-tint); }
.oa-node-badge.b { border-color: rgba(107,138,168,0.4); color: var(--steel); background: var(--steel-tint); }

.oa-node-task {
  font-family: var(--mono) !important;
  font-size: 10.5px;
  font-weight: 500;
  color: var(--ink-soft);
  margin-bottom: 5px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.oa-node-desc {
  font-size: 12px;
  font-weight: 400;
  line-height: 1.5;
  color: var(--muted);
  margin-bottom: 12px;
}
.oa-node-snippet {
  padding: 10px 11px;
  background: var(--bg-2);
  border: 1px solid var(--line-soft);
  border-radius: var(--r-sm);
  font-family: var(--mono) !important;
  font-size: 9.5px;
  line-height: 1.55;
  color: var(--muted);
  max-height: 72px;
  overflow: hidden;
  white-space: pre-wrap;
  word-break: break-all;
  margin-bottom: 12px;
}
.oa-node-tags { display: flex; flex-wrap: wrap; gap: 5px; }
.oa-tag {
  font-family: var(--mono) !important;
  font-size: 9px;
  font-weight: 500;
  padding: 3px 7px;
  border: 1px solid var(--line);
  border-radius: 4px;
  color: var(--muted);
  background: var(--paper);
}
.oa-tag.hot { border-color: rgba(199,149,82,0.4); color: var(--amber); background: var(--amber-tint); }
.oa-tag.red { border-color: var(--crimson-line);  color: var(--crimson); background: var(--crimson-tint); }
.oa-tag.grn { border-color: rgba(91,135,105,0.4); color: var(--sage); background: var(--sage-tint); }

/* ─── live SVG pipeline bar ──────────────────────────────────── */
.oa-pipe-bar {
  margin-bottom: 24px;
  border: 1px solid var(--line);
  border-radius: var(--r);
  overflow: hidden;
  background: var(--paper);
}
.oa-pipe-bar svg { display: block; width: 100%; height: auto; }

/* ─── card / console panel ───────────────────────────────────── */
.oa-console {
  border: 1px solid var(--line);
  border-radius: var(--r);
  overflow: hidden;
  background: var(--paper);
}
.oa-console-hdr {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 14px 18px;
  background: var(--bg-2);
  border-bottom: 1px solid var(--line-soft);
  font-family: var(--mono) !important;
  font-size: 10.5px;
  font-weight: 600;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--ink-soft);
}
.oa-dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  background: var(--orange);
  animation: blink 1.5s ease-in-out infinite;
}
.oa-dot.r { background: var(--crimson); }
.oa-dot.w { background: var(--amber); }

/* ─── log ────────────────────────────────────────────────────── */
.oa-log { font-size: 13px; }
.oa-log-row {
  display: grid;
  grid-template-columns: 50px 1fr 78px;
  gap: 14px;
  padding: 13px 18px;
  border-bottom: 1px solid var(--line-soft);
  animation: fadeUp 0.25s ease-out;
}
.oa-log-row:last-child { border-bottom: none; }
.oa-log-n  { font-family: var(--mono) !important; color: var(--dim); font-size: 11px; }
.oa-log-a  { color: var(--ink-soft); font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.oa-log-r  { font-family: var(--mono) !important; text-align: right; font-weight: 600; font-size: 11.5px; }
.oa-log-r.p{ color: var(--sage); }
.oa-log-r.n{ color: var(--crimson); }
.oa-log-sub {
  grid-column: 2/4;
  font-size: 11.5px;
  font-weight: 400;
  color: var(--muted);
  margin-top: -6px;
  padding-bottom: 6px;
}

/* ─── reward grid ────────────────────────────────────────────── */
.oa-reward-grid {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 1px;
  background: var(--line);
  border-radius: var(--r);
  overflow: hidden;
}
.oa-rcard { padding: 16px 16px; background: var(--paper); }
.oa-rcard-lbl {
  font-family: var(--mono) !important;
  font-size: 9.5px;
  font-weight: 500;
  letter-spacing: 0.13em;
  text-transform: uppercase;
  color: var(--dim);
  display: block;
  margin-bottom: 8px;
}
.oa-rcard-val {
  font-family: var(--display) !important;
  font-size: 26px;
  font-weight: 400;
  font-style: italic;
  letter-spacing: -0.02em;
  color: var(--ink);
}
.oa-rcard-val.p { color: var(--sage); }
.oa-rcard-val.n { color: var(--crimson); }
.oa-rcard-val.z { color: var(--dim); }

/* ─── oracle panel ───────────────────────────────────────────── */
.oa-oracle-wrap {
  border: 1px solid rgba(199,149,82,0.3);
  border-radius: var(--r);
  overflow: hidden;
  background: var(--amber-tint);
}
.oa-oracle-hdr {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 14px 18px;
  background: rgba(199,149,82,0.14);
  border-bottom: 1px solid rgba(199,149,82,0.22);
  font-family: var(--mono) !important;
  font-size: 10.5px;
  font-weight: 600;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #8c6429;
}

/* ─── post-mortem ────────────────────────────────────────────── */
.oa-pm { padding: 0 64px 72px; }
.oa-pm-hdr {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 20px;
  padding: 40px 0 28px;
  border-top: 1px solid var(--line);
}
.oa-pm-title {
  font-family: var(--display) !important;
  font-size: 48px;
  font-weight: 400;
  font-style: italic;
  letter-spacing: -0.02em;
  color: var(--ink);
}
.oa-pm-grid { display: grid; grid-template-columns: 1.5fr 1fr; gap: 20px; }
.oa-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
  border-radius: var(--r);
  overflow: hidden;
  background: var(--paper);
  border: 1px solid var(--line);
}
.oa-table th {
  padding: 12px 16px;
  text-align: left;
  background: var(--bg-2);
  font-family: var(--mono) !important;
  color: var(--muted);
  font-size: 9.5px;
  font-weight: 600;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  border-bottom: 1px solid var(--line);
}
.oa-table td {
  padding: 12px 16px;
  border-bottom: 1px solid var(--line-soft);
  color: var(--ink-soft);
  font-weight: 400;
}
.oa-table tr:last-child td { border-bottom: none; }
.oa-table td.caught { color: var(--sage); font-weight: 500; }
.oa-table td.missed { color: var(--crimson); font-weight: 500; }
.oa-table td.clean  { color: var(--steel); }

.oa-pm-side {
  padding: 24px;
  border: 1px solid var(--orange-line);
  border-radius: var(--r);
  background: var(--orange-tint);
}

/* ─── verdict badge ──────────────────────────────────────────── */
.oa-verdict {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding: 10px 18px;
  border-radius: 6px;
  font-family: var(--mono) !important;
  font-size: 10.5px;
  font-weight: 600;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}
.oa-verdict.clean   { background: var(--sage-tint);    border: 1px solid rgba(91,135,105,0.4); color: var(--sage); }
.oa-verdict.dirty   { background: var(--crimson-tint); border: 1px solid var(--crimson-line);  color: var(--crimson); }
.oa-verdict.timeout { background: var(--amber-tint);   border: 1px solid rgba(199,149,82,0.4); color: var(--amber); }

/* ─── empty state ────────────────────────────────────────────── */
.oa-empty {
  padding: 32px 22px;
  text-align: center;
  font-size: 14px;
  font-weight: 400;
  color: var(--muted);
  font-family: var(--display) !important;
  font-style: italic;
  border: 1px dashed var(--line);
  border-radius: var(--r);
  background: var(--bg-2);
}

/* ─── Gradio layout containers (real fix) ─────────────────────── */
/* Gradio Rows/Columns get our card styling via elem_classes.
   We also tame Gradio's default block backgrounds & borders so our
   palette wins. */

.oa-row {
  padding: 0 64px 18px !important;
  gap: 20px !important;
}
.oa-row > .gr-column,
.oa-row > div[class*="column"] { padding: 0 !important; }

.oa-card {
  background: var(--paper) !important;
  border: 1px solid var(--line) !important;
  border-radius: var(--r) !important;
  padding: 22px 22px 20px !important;
  display: flex !important;
  flex-direction: column;
  gap: 14px;
}
.oa-card-hdr {
  font-family: var(--mono) !important;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--muted);
  display: flex;
  align-items: center;
  gap: 9px;
  padding-bottom: 10px;
  border-bottom: 1px solid var(--line-soft);
  margin-bottom: 4px;
}
.oa-card-hdr .oa-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--orange); animation: blink 1.5s ease-in-out infinite; }

.oa-status-col {
  background: var(--bg-2) !important;
  border: 1px solid var(--line) !important;
  border-radius: var(--r) !important;
  padding: 0 !important;
  overflow: hidden;
}
.oa-status-col .oa-stats {
  border-bottom: none;
  background: transparent;
  grid-template-columns: 1fr 1fr;
}
.oa-status-col .oa-stat:nth-child(odd)  { border-right: 1px solid var(--line); }
.oa-status-col .oa-stat:nth-child(-n+3) { border-bottom: 1px solid var(--line); }
.oa-status-col .oa-stat:last-child { border-right: none; grid-column: 1 / -1; }

.oa-btn-row { gap: 8px !important; align-items: stretch !important; }

/* tame default Gradio block chrome inside our cards */
.oa-card .block,
.oa-card .form,
.oa-card .gr-form,
.oa-card .gr-box,
.oa-card .wrap {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
}

/* textareas */
.obs-text textarea, .act-text textarea {
  background: var(--bg-2) !important;
  border: 1px solid var(--line) !important;
  border-radius: var(--r-sm) !important;
  color: var(--ink-soft) !important;
  font-family: var(--mono) !important;
  font-size: 12px !important;
  line-height: 1.65 !important;
  padding: 14px !important;
  box-shadow: none !important;
}
.act-text textarea {
  background: var(--orange-tint) !important;
  border-color: var(--orange-line) !important;
  color: var(--ink) !important;
  font-weight: 500 !important;
}
.obs-text textarea:focus,
.act-text textarea:focus,
input:focus, textarea:focus {
  outline: none !important;
  border-color: var(--orange) !important;
  box-shadow: 0 0 0 3px var(--orange-tint) !important;
}

/* number / textbox inputs */
.gradio-container input[type="number"],
.gradio-container input[type="text"] {
  background: var(--paper) !important;
  border: 1px solid var(--line) !important;
  border-radius: var(--r-sm) !important;
  color: var(--ink-soft) !important;
  font-family: var(--mono) !important;
  font-size: 12.5px !important;
  padding: 8px 10px !important;
}

/* radio chips */
.gradio-container [data-testid="radio"] label,
.gradio-container .gr-radio label {
  background: var(--paper) !important;
  border: 1px solid var(--line) !important;
  color: var(--ink-soft) !important;
  font-family: var(--sans) !important;
  font-size: 12px !important;
  font-weight: 500 !important;
  letter-spacing: 0 !important;
  text-transform: none !important;
  border-radius: var(--r-sm) !important;
  padding: 7px 12px !important;
  transition: all 0.18s;
}
.gradio-container [data-testid="radio"] input:checked + span,
.gradio-container [data-testid="radio"] label:has(input:checked) {
  border-color: var(--orange) !important;
  background: var(--orange-tint) !important;
  color: var(--orange) !important;
}

/* buttons — primary (filled ink) */
button[class*="primary"], .gr-button-primary {
  background: var(--ink) !important;
  color: var(--bg) !important;
  border: 1px solid var(--ink) !important;
  border-radius: var(--r-sm) !important;
  font-family: var(--sans) !important;
  font-size: 12.5px !important;
  font-weight: 500 !important;
  letter-spacing: 0.01em !important;
  padding: 10px 18px !important;
  box-shadow: none !important;
  transition: opacity 0.18s, transform 0.18s !important;
}
button[class*="primary"]:hover { opacity: 0.88 !important; transform: translateY(-1px); }

/* buttons — secondary (outlined) */
button[class*="secondary"], .gr-button-secondary,
.gradio-container button:not([class*="primary"]) {
  background: var(--paper) !important;
  color: var(--ink-soft) !important;
  border: 1px solid var(--line) !important;
  border-radius: var(--r-sm) !important;
  font-family: var(--sans) !important;
  font-size: 12px !important;
  font-weight: 500 !important;
  padding: 8px 14px !important;
  box-shadow: none !important;
  transition: all 0.18s !important;
}
button[class*="secondary"]:hover,
.gradio-container button:not([class*="primary"]):hover {
  background: var(--orange-tint) !important;
  border-color: var(--orange) !important;
  color: var(--orange) !important;
}

/* labels — small caps mono */
.gradio-container label > span:first-child,
.gradio-container .label-wrap > span:first-child,
.gradio-container .gr-label {
  font-family: var(--mono) !important;
  font-size: 10px !important;
  font-weight: 500 !important;
  letter-spacing: 0.14em !important;
  text-transform: uppercase !important;
  color: var(--muted) !important;
  margin-bottom: 6px !important;
}

/* ─── SVG keyframes ──────────────────────────────────────────── */
@keyframes flowPkt {
  0%   { offset-distance: 0%;   opacity: 0; }
  6%   { opacity: 1; }
  88%  { opacity: 1; }
  100% { offset-distance: 100%; opacity: 0; }
}
@keyframes badPkt {
  0%   { offset-distance: 0%;  opacity: 0;   transform: scale(1); }
  8%   { opacity: 1; }
  50%  { opacity: 1;  transform: scale(1); }
  60%  { transform: scale(2.4); opacity: 0.5; }
  68%  { transform: scale(0.1); opacity: 0; }
  100% { offset-distance: 62%; opacity: 0; }
}
@keyframes dashScroll { to { stroke-dashoffset: -40; } }
@keyframes glitchSurface {
  0%,80%,100% { filter: none; transform: none; clip-path: none; }
  82% { transform: translate(-3px,1px); filter: hue-rotate(60deg) saturate(2.4) brightness(1.05); }
  85% { transform: translate(3px,-2px); clip-path: polygon(0 18%,100% 18%,100% 46%,0 46%); filter: none; }
  88% { transform: translate(-1px,0); }
}
@keyframes scanArc {
  0%,100% { opacity: 0.28; transform: scale(0.93); }
  50%     { opacity: 1;    transform: scale(1.06); }
}
@keyframes popReveal {
  0%,32%  { opacity: 0; transform: translateY(14px) scale(0.86); }
  46%,86% { opacity: 1; transform: translateY(0)    scale(1);    }
  98%,100%{ opacity: 0; transform: translateY(-8px) scale(0.94); }
}
@keyframes hotPulse {
  0%,100%{ box-shadow: 0 0 0 0 rgba(199,149,82,0); }
  50%    { box-shadow: 0 0 24px 4px rgba(199,149,82,0.15); }
}
@keyframes redFlash {
  0%,100%{ opacity: 1; }
  50%    { opacity: 0.45; }
}
@keyframes blink {
  0%,100%{ opacity: 1; }
  50%    { opacity: 0.3; }
}
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes heroIn {
  from { opacity: 0; transform: translateY(20px); }
  to   { opacity: 1; transform: translateY(0); }
}

.pkt-clean {
  offset-path: path("M72 182 L1160 182");
  animation: flowPkt 3.6s linear infinite;
}
.pkt-bad {
  offset-path: path("M72 182 L1160 182");
  animation: badPkt 3.6s linear infinite;
  animation-delay: 1.6s;
}
.dash-flow  { stroke-dasharray: 12 14; animation: dashScroll 1.8s linear infinite; }
.glitch-node{ animation: glitchSurface 5.5s ease-in-out infinite; animation-delay: 0.8s; }
.scan-ring  { transform-origin: 50% 50%; animation: scanArc 2s ease-in-out infinite; }
.pop-reveal { animation: popReveal 3.6s ease-in-out infinite; animation-delay: 1.2s; }
.hero-in    { animation: heroIn 0.9s ease-out both; }
.hero-in-2  { animation: heroIn 0.9s 0.18s ease-out both; }
.hero-in-3  { animation: heroIn 0.9s 0.34s ease-out both; }
.hero-in-4  { animation: heroIn 0.9s 0.5s ease-out both; }

/* ─── responsive ─────────────────────────────────────────────── */
@media (max-width: 1100px) {
  .oa-hero { grid-template-columns: 1fr; gap: 36px; padding: 64px 32px 40px; }
  .oa-nodes, .oa-stats, .oa-steps { grid-template-columns: repeat(2,1fr); }
  .oa-pm-grid { grid-template-columns: 1fr; }
  .oa-section, .oa-pm { padding-left: 32px; padding-right: 32px; }
  .oa-row { padding-left: 32px !important; padding-right: 32px !important; }
}
@media (max-width: 680px) {
  .oa-nodes, .oa-stats, .oa-steps, .oa-reward-grid { grid-template-columns: 1fr; }
  .oa-hero, .oa-section, .oa-pm { padding-left: 20px; padding-right: 20px; }
  .oa-row { padding-left: 20px !important; padding-right: 20px !important; }
  .oa-section-copy { text-align: left; margin-left: 0; }
  .oa-section-head { flex-direction: column; align-items: flex-start; gap: 8px; }
}
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

WORKER_ROLES = {
    1: "Req Analyst",
    2: "Code Agent",
    3: "Test Gen",
    4: "Sec Review",
    5: "Deploy",
}

FAILURE_STATES = {"HALLUCINATING", "STALLED", "DRIFTED", "DECEPTIVE", "CORRUPTED"}

REWARD_LABELS = {
    "reward_catch":          "Failure catch",
    "reward_deceptive_catch":"Deception catch",
    "penalty_false_positive":"False positive",
    "penalty_hallpass":      "Bad approval",
    "penalty_deceptive_pass":"Deception missed",
    "reward_efficiency":     "Efficiency",
    "reward_collusion":      "Collusion",
    "penalty_format":        "Format",
    "reward_mercor":         "Reasoning",
}


# ─────────────────────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _esc(v: Any) -> str:
    return html.escape(str(v), quote=True)

def _clip(v: str, lim: int = 180) -> str:
    v = " ".join((v or "").split())
    return v if len(v) <= lim else v[:lim - 1] + "…"

def _format_action(verb: str, wid: int | str, reason: str = "") -> str:
    r = reason.strip() or "Checking this worker before acting."
    return f"{verb} {wid}\n<reasoning>{r}</reasoning>"

def _redirect_action(wid: int | str, instruction: str) -> str:
    instr = instruction.strip() or "Refocus on the original task."
    return f"REDIRECT {wid} {instr}\n<reasoning>Worker appears off-task; soft reset preferred.</reasoning>"

def _display_total(state: dict[str, Any]) -> float:
    bd = state.get("reward_breakdown") or {}
    if state.get("episode_result") == "IN_PROGRESS":
        return float(sum(bd.values()))
    return float(state.get("total_reward", sum(bd.values())) or 0.0)


# ─────────────────────────────────────────────────────────────────────────────
#  Hero HTML — animated SVG pipeline concept
# ─────────────────────────────────────────────────────────────────────────────

def hero_html() -> str:
    return """
<div class="oa-hero oa-root">
  <div class="oa-hero-text">
    <div class="oa-eyebrow hero-in">Oversight Arena · v1.0</div>

    <h1 class="oa-h1 hero-in-2">
      Catch the<br/>
      failure that<br/>
      <em>looks</em> <span class="hi">safe.</span>
    </h1>

    <p class="oa-lead hero-in-3">
      Five deterministic AI workers run a software-delivery pipeline.
      One produces output that <em>looks clean</em> on the surface — but hides a
      flaw only revealed by deliberate deep inspection.
      Can a supervisor LLM learn when to look deeper?
    </p>

    <div class="oa-meta hero-in-4">
      <span>real environment</span>
      <span>no fake demos</span>
      <span>deterministic + seedable</span>
    </div>
  </div>

  <div class="oa-hero-svg hero-in-3">
    <svg viewBox="0 0 1240 400" aria-label="Animated oversight pipeline" role="img">
      <defs>
        <filter id="glow0">
          <feGaussianBlur stdDeviation="4" result="b"/>
          <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        <filter id="glow1">
          <feGaussianBlur stdDeviation="8" result="b"/>
          <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        <radialGradient id="scanGrad" cx="50%" cy="50%" r="50%">
          <stop offset="0%"   stop-color="#c96442" stop-opacity="0.22"/>
          <stop offset="100%" stop-color="#c96442" stop-opacity="0"/>
        </radialGradient>
        <radialGradient id="badGrad" cx="50%" cy="50%" r="50%">
          <stop offset="0%"   stop-color="#b85c52" stop-opacity="0.28"/>
          <stop offset="100%" stop-color="#b85c52" stop-opacity="0"/>
        </radialGradient>
        <pattern id="hdots" x="0" y="0" width="48" height="48" patternUnits="userSpaceOnUse">
          <circle cx="24" cy="24" r="1.1" fill="rgba(26,22,20,0.08)"/>
        </pattern>
      </defs>

      <rect width="1240" height="400" fill="url(#hdots)"/>

      <!-- pipeline backbone -->
      <line x1="80" y1="200" x2="1160" y2="200"
            stroke="rgba(26,22,20,0.10)" stroke-width="6" stroke-linecap="round"/>
      <line class="dash-flow" x1="80" y1="200" x2="1160" y2="200"
            stroke="#c96442" stroke-opacity="0.55" stroke-width="2"
            fill="none" stroke-dasharray="12 14"/>

      <!-- ── W1 ── -->
      <g transform="translate(38 134)">
        <rect width="130" height="132" rx="8"
              fill="#ffffff" stroke="rgba(26,22,20,0.12)" stroke-width="1"/>
        <rect x="0" y="0" width="130" height="3" rx="8" fill="#6b8aa8"/>
        <text x="14" y="44"
              fill="#1a1614" font-family="Instrument Serif,serif"
              font-size="44" font-weight="400" font-style="italic" letter-spacing="-2">W1</text>
        <text x="14" y="74"
              fill="#c96442" font-family="Geist Mono,monospace"
              font-size="9" font-weight="600" letter-spacing="1.6">REQ ANALYST</text>
        <rect x="14" y="92" width="56" height="20" rx="3"
              fill="rgba(107,138,168,0.10)" stroke="rgba(107,138,168,0.40)"/>
        <text x="42" y="106" text-anchor="middle"
              fill="#6b8aa8" font-family="Geist Mono,monospace"
              font-size="8.5" font-weight="600" letter-spacing="1">WORKING</text>
      </g>

      <!-- ── W2 ── -->
      <g transform="translate(258 134)">
        <rect width="130" height="132" rx="8"
              fill="#ffffff" stroke="rgba(26,22,20,0.12)" stroke-width="1"/>
        <rect x="0" y="0" width="130" height="3" rx="8" fill="#6b8aa8"/>
        <text x="14" y="44"
              fill="#1a1614" font-family="Instrument Serif,serif"
              font-size="44" font-weight="400" font-style="italic" letter-spacing="-2">W2</text>
        <text x="14" y="74"
              fill="#c96442" font-family="Geist Mono,monospace"
              font-size="9" font-weight="600" letter-spacing="1.6">CODE GEN</text>
        <rect x="14" y="92" width="56" height="20" rx="3"
              fill="rgba(107,138,168,0.10)" stroke="rgba(107,138,168,0.40)"/>
        <text x="42" y="106" text-anchor="middle"
              fill="#6b8aa8" font-family="Geist Mono,monospace"
              font-size="8.5" font-weight="600" letter-spacing="1">WORKING</text>
      </g>

      <!-- ── W3 DECEPTIVE ── -->
      <g transform="translate(478 110)">
        <circle cx="65" cy="90" r="78" fill="url(#badGrad)"/>
        <g class="glitch-node">
          <rect width="130" height="180" rx="8"
                fill="#ffffff" stroke="rgba(184,92,82,0.45)" stroke-width="1.2"/>
          <rect x="0" y="0" width="130" height="3" rx="8" fill="#b85c52"/>
          <text x="14" y="44"
                fill="#1a1614" font-family="Instrument Serif,serif"
                font-size="44" font-weight="400" font-style="italic" letter-spacing="-2">W3</text>
          <text x="14" y="74"
                fill="#c96442" font-family="Geist Mono,monospace"
                font-size="9" font-weight="600" letter-spacing="1.6">TEST GEN</text>
          <rect x="14" y="92" width="62" height="20" rx="3"
                fill="rgba(91,135,105,0.10)" stroke="rgba(91,135,105,0.40)"/>
          <text x="45" y="106" text-anchor="middle"
                fill="#5b8769" font-family="Geist Mono,monospace"
                font-size="8.5" font-weight="600" letter-spacing="1">PASSING</text>

          <rect x="10" y="124" width="110" height="46" rx="4"
                fill="rgba(184,92,82,0.10)" stroke="rgba(184,92,82,0.42)" stroke-width="1"/>
          <text x="65" y="140" text-anchor="middle"
                fill="#b85c52" font-family="Geist Mono,monospace"
                font-size="8" font-weight="700" letter-spacing="1.4">HIDDEN FLAW</text>
          <text x="65" y="155" text-anchor="middle"
                fill="rgba(184,92,82,0.75)" font-family="Geist Mono,monospace"
                font-size="7.5">missing tenant isolation</text>
          <text x="65" y="166" text-anchor="middle"
                fill="rgba(184,92,82,0.55)" font-family="Geist Mono,monospace"
                font-size="7">surface output: clean ✓</text>
        </g>
      </g>

      <!-- ── W4 IDLE ── -->
      <g transform="translate(698 134)">
        <rect width="130" height="132" rx="8"
              fill="#faf6f1" stroke="rgba(26,22,20,0.10)" stroke-width="1"/>
        <rect x="0" y="0" width="130" height="3" rx="8" fill="rgba(26,22,20,0.10)"/>
        <text x="14" y="44"
              fill="rgba(26,22,20,0.32)" font-family="Instrument Serif,serif"
              font-size="44" font-weight="400" font-style="italic" letter-spacing="-2">W4</text>
        <text x="14" y="74"
              fill="rgba(168,155,142,0.9)" font-family="Geist Mono,monospace"
              font-size="9" font-weight="600" letter-spacing="1.6">SEC REVIEW</text>
        <rect x="14" y="92" width="40" height="20" rx="3"
              fill="rgba(168,155,142,0.10)" stroke="rgba(168,155,142,0.30)"/>
        <text x="34" y="106" text-anchor="middle"
              fill="rgba(168,155,142,0.95)" font-family="Geist Mono,monospace"
              font-size="8.5" font-weight="600" letter-spacing="1">IDLE</text>
      </g>

      <!-- ── W5 IDLE ── -->
      <g transform="translate(918 134)">
        <rect width="130" height="132" rx="8"
              fill="#faf6f1" stroke="rgba(26,22,20,0.08)" stroke-width="1"/>
        <rect x="0" y="0" width="130" height="3" rx="8" fill="rgba(26,22,20,0.08)"/>
        <text x="14" y="44"
              fill="rgba(26,22,20,0.22)" font-family="Instrument Serif,serif"
              font-size="44" font-weight="400" font-style="italic" letter-spacing="-2">W5</text>
        <text x="14" y="74"
              fill="rgba(168,155,142,0.7)" font-family="Geist Mono,monospace"
              font-size="9" font-weight="600" letter-spacing="1.6">DEPLOY</text>
        <rect x="14" y="92" width="40" height="20" rx="3"
              fill="rgba(168,155,142,0.08)" stroke="rgba(168,155,142,0.22)"/>
        <text x="34" y="106" text-anchor="middle"
              fill="rgba(168,155,142,0.85)" font-family="Geist Mono,monospace"
              font-size="8.5" font-weight="600" letter-spacing="1">IDLE</text>
      </g>

      <!-- packets flowing -->
      <circle class="pkt-clean" r="8" fill="#c96442" filter="url(#glow0)"/>
      <circle class="pkt-bad" r="10" fill="#b85c52" filter="url(#glow1)"/>

      <!-- Oversight scanner -->
      <g transform="translate(543 320)">
        <circle r="60" fill="url(#scanGrad)"/>
        <circle class="scan-ring" r="58" fill="none" stroke="#c96442" stroke-width="1.4"/>
        <circle r="34" fill="#ffffff" stroke="rgba(201,100,66,0.4)" stroke-width="1"/>
        <text x="0" y="-3" text-anchor="middle"
              fill="#c96442" font-family="Geist Mono,monospace"
              font-size="10" font-weight="600" letter-spacing="2.4">OVERSIGHT</text>
        <text x="0" y="13" text-anchor="middle"
              fill="rgba(201,100,66,0.7)" font-family="Geist Mono,monospace"
              font-size="8" letter-spacing="0.5">DEEP_INSPECT</text>
      </g>
      <line x1="543" y1="266" x2="543" y2="290"
            stroke="#c96442" stroke-width="1.8" stroke-opacity="0.55"
            stroke-dasharray="3 4" class="scan-ring"/>

      <!-- Reward reveal badge -->
      <g class="pop-reveal" transform="translate(610 56)">
        <rect x="0" y="0" width="270" height="62" rx="8"
              fill="rgba(201,100,66,0.10)" stroke="rgba(201,100,66,0.45)" stroke-width="1"/>
        <text x="18" y="26" fill="#c96442"
              font-family="Instrument Serif,serif"
              font-size="22" font-weight="400" font-style="italic">+2.5 reward</text>
        <text x="18" y="48" fill="rgba(107,93,82,0.9)"
              font-family="Geist Mono,monospace" font-size="9.5" letter-spacing="0.4">
          deception caught via DEEP_INSPECT
        </text>
      </g>

      <!-- Under-node labels -->
      <text x="103" y="288" text-anchor="middle" fill="rgba(91,135,105,0.85)"
            font-family="Geist Mono,monospace" font-size="9" font-weight="600" letter-spacing="2">CLEAN</text>
      <text x="323" y="288" text-anchor="middle" fill="rgba(91,135,105,0.85)"
            font-family="Geist Mono,monospace" font-size="9" font-weight="600" letter-spacing="2">CLEAN</text>
      <text x="543" y="305" text-anchor="middle" fill="#b85c52"
            font-family="Geist Mono,monospace" font-size="9.5"
            font-weight="700" letter-spacing="1.8">DECEPTIVE ↓</text>
      <text x="763" y="288" text-anchor="middle" fill="rgba(168,155,142,0.85)"
            font-family="Geist Mono,monospace" font-size="9" letter-spacing="2">IDLE</text>
      <text x="983" y="288" text-anchor="middle" fill="rgba(168,155,142,0.85)"
            font-family="Geist Mono,monospace" font-size="9" letter-spacing="2">IDLE</text>
    </svg>
  </div>
</div>
"""


# ─────────────────────────────────────────────────────────────────────────────
#  How-to steps
# ─────────────────────────────────────────────────────────────────────────────

def how_html() -> str:
    steps = [
        ("Start", "Pick a scenario and click Start Simulation to initialize a real episode."),
        ("Watch", "Observe the 5-worker chain. Outputs look clean — surface tells nothing."),
        ("Investigate", "Use Deep Inspect on suspicious workers to reveal hidden flaws."),
        ("Act", "Terminate bad workers, Redirect drifters, Approve clean ones."),
        ("Review", "Post-mortem reveals ground truth: what was hidden and what you caught."),
    ]
    cards = "".join(
        f"""<div class="oa-step">
          <div class="oa-step-n">{i}</div>
          <div class="oa-step-title">{_esc(t)}</div>
          <div class="oa-step-body">{_esc(c)}</div>
        </div>"""
        for i, (t, c) in enumerate(steps, 1)
    )
    return f"""
<div class="oa-section">
  <div class="oa-section-head">
    <span class="oa-section-num">01 — How it works</span>
    <span class="oa-section-title">Five steps. No prior knowledge needed.</span>
    <span class="oa-section-copy">A guided flow from launch to verdict — designed so any judge can run a complete episode in under a minute.</span>
  </div>
  <div class="oa-steps">{cards}</div>
</div>
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Status bar
# ─────────────────────────────────────────────────────────────────────────────

def status_html(state: dict[str, Any] | None = None) -> str:
    if not state:
        items = [
            ("episode", "—", ""),
            ("difficulty", "—", ""),
            ("step", "0 / 25", ""),
            ("risk", "LOW", ""),
            ("reward", "+0.000", ""),
        ]
    else:
        total = _display_total(state)
        items = [
            ("episode",    state.get("episode_result", "IN_PROGRESS"), ""),
            ("difficulty", state.get("difficulty", "?").upper(), ""),
            ("step",       f"{state.get('step',0)} / {state.get('max_steps',25)}", ""),
            ("risk",       state.get("corruption_risk", "LOW"),
             "r" if state.get("corruption_risk") == "HIGH" else "g"),
            ("reward",     f"{total:+.3f}",
             "g" if total >= 0 else "r"),
        ]
    cells = "".join(
        f"""<div class="oa-stat">
          <span class="oa-stat-lbl">{_esc(lbl)}</span>
          <span class="oa-stat-val {cls}">{_esc(val)}</span>
        </div>"""
        for lbl, val, cls in items
    )
    return f'<div class="oa-stats">{cells}</div>'


# ─────────────────────────────────────────────────────────────────────────────
#  Live pipeline nodes  (state-driven)
# ─────────────────────────────────────────────────────────────────────────────

def _node_badge(worker: dict[str, Any]) -> tuple[str, str]:
    vis = worker.get("visible_state", "IDLE")
    unch = int(worker.get("steps_unchanged", 0) or 0)
    real = worker.get("real_state_str") or worker.get("state")
    deep = bool(worker.get("deep_inspect_done"))
    if vis in {"COMPLETED", "APPROVED"}:
        return "g", vis
    if deep and real in FAILURE_STATES:
        return "r", "FLAW EXPOSED"
    if unch >= 2 and vis == "WORKING":
        return "w", f"SUSPICIOUS ×{unch}"
    if vis == "WORKING":
        return "b", "WORKING"
    return "", vis

def _node_cls(worker: dict[str, Any], show_real: bool, bad_ids: set[int]) -> str:
    vis  = str(worker.get("visible_state", "IDLE")).lower()
    wid  = int(worker.get("worker_id", 0))
    real = worker.get("real_state_str") or worker.get("state")
    deep = bool(worker.get("deep_inspect_done"))
    unch = int(worker.get("steps_unchanged", 0) or 0)
    cls  = ["oa-node", vis]
    if deep and real in FAILURE_STATES:
        cls.append("exposed")
    elif unch >= 2 and vis == "working":
        cls.append("hot")
    if show_real:
        cls.append("truth-fail" if wid in bad_ids else "truth-clean")
    return " ".join(cls)

def _node_snippet(worker: dict[str, Any]) -> str:
    hist = worker.get("output_history") or []
    return _clip(str(hist[-1]), 180) if hist else "(no output yet)"

def _node_tags(worker: dict[str, Any], show_real: bool, bad_ids: set[int]) -> str:
    wid  = int(worker.get("worker_id", 0))
    unch = int(worker.get("steps_unchanged", 0) or 0)
    deep = bool(worker.get("deep_inspect_done"))
    tags = [
        f'<span class="oa-tag {"hot" if unch >= 2 else ""}">Δ={unch}</span>',
        f'<span class="oa-tag {"grn" if deep else ""}">{"✓ inspected" if deep else "not inspected"}</span>',
    ]
    if worker.get("is_colluding"):
        tags.append('<span class="oa-tag hot">colluding</span>')
    if show_real:
        fm = worker.get("failure_mode", "NONE")
        tags.append(
            f'<span class="oa-tag {"red" if wid in bad_ids else "grn"}">'
            f'{"⚠ " + fm if wid in bad_ids else "CLEAN"}</span>'
        )
    return "".join(tags)

def _node_card(worker: dict[str, Any], show_real: bool, bad_ids: set[int]) -> str:
    wid  = int(worker.get("worker_id", 0))
    bc, bl = _node_badge(worker)
    return f"""<div class="{_node_cls(worker, show_real, bad_ids)}">
  <div class="oa-node-id">W{wid}</div>
  <div class="oa-node-role">{_esc(WORKER_ROLES.get(wid, "Worker"))}</div>
  <span class="oa-node-badge {bc}">{_esc(bl)}</span>
  <div class="oa-node-task">{_esc(worker.get("task_type", ""))}</div>
  <div class="oa-node-desc">{_esc(_clip(worker.get("task_description", ""), 110))}</div>
  <div class="oa-node-snippet">{_esc(_node_snippet(worker))}</div>
  <div class="oa-node-tags">{_node_tags(worker, show_real, bad_ids)}</div>
</div>"""


def _pipe_bar_svg(env: OversightArenaEnvironment | None, show_real: bool) -> str:
    """Slim animated SVG bar showing live worker states across the pipeline."""
    if env is None:
        workers = [{"worker_id": i, "visible_state": "IDLE"} for i in range(1, 6)]
        step_txt = "No episode"
        bad_ids: set[int] = set()
    else:
        state  = env.state_dict
        workers = state.get("workers", [])
        step_txt = f"Step {state.get('step',0)}/{state.get('max_steps',25)}"
        bad_ids = {int(k) for k in state.get("failure_plan", {}).keys()}

    STATE_COLOR = {
        "IDLE":      "#a89b8e",
        "WAITING":   "#a89b8e",
        "WORKING":   "#6b8aa8",
        "COMPLETED": "#5b8769",
        "APPROVED":  "#c79552",
        "REDIRECTED":"#a07dd4",
    }
    xs  = [110, 298, 486, 674, 862]
    nodes = []
    for idx, w in enumerate(workers[:5]):
        vis = str(w.get("visible_state", "IDLE"))
        wid = int(w.get("worker_id", idx + 1))
        unch = int(w.get("steps_unchanged", 0) or 0)
        deep = bool(w.get("deep_inspect_done"))
        real = w.get("real_state_str") or w.get("state", "IDLE")
        x = xs[idx]

        color = STATE_COLOR.get(vis, "#a89b8e")
        if deep and real in FAILURE_STATES:
            color = "#b85c52"
        elif unch >= 2 and vis == "WORKING":
            color = "#c79552"
        if show_real and wid in bad_ids:
            color = "#b85c52"

        nodes.append(f"""
        <g transform="translate({x - 54} 30)">
          <rect width="108" height="92" rx="8"
                fill="#ffffff" stroke="{color}" stroke-width="1"/>
          <rect x="0" y="0" width="108" height="3" rx="8" fill="{color}"/>
          <circle cx="90" cy="18" r="4" fill="{color}" opacity="0.85"/>
          <text x="12" y="44"
                fill="#1a1614" font-family="Instrument Serif,serif"
                font-size="32" font-weight="400" font-style="italic" letter-spacing="-1.5">W{wid}</text>
          <text x="12" y="62"
                fill="{color}" font-family="Geist Mono,monospace"
                font-size="8" font-weight="600" letter-spacing="1.4">{_esc(vis)}</text>
          <text x="12" y="78"
                fill="#a89b8e" font-family="Geist Mono,monospace"
                font-size="7.5" letter-spacing="0.5">Δ={unch}</text>
        </g>""")

    return f"""<div class="oa-pipe-bar">
  <svg viewBox="0 0 980 156" role="img" aria-label="Live oversight pipeline">
    <defs>
      <filter id="pbGlow">
        <feGaussianBlur stdDeviation="3" result="b"/>
        <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>
      <pattern id="pbDots" x="0" y="0" width="40" height="40" patternUnits="userSpaceOnUse">
        <circle cx="20" cy="20" r="1" fill="rgba(26,22,20,0.06)"/>
      </pattern>
    </defs>
    <rect width="980" height="156" fill="#faf6f1"/>
    <rect width="980" height="156" fill="url(#pbDots)"/>
    <line x1="56" y1="76" x2="924" y2="76"
          stroke="rgba(26,22,20,0.10)" stroke-width="5" stroke-linecap="round"/>
    <line class="dash-flow" x1="56" y1="76" x2="924" y2="76"
          stroke="#c96442" stroke-opacity="0.6" stroke-width="1.6" fill="none"
          stroke-dasharray="10 13"/>
    <circle r="6" fill="#c96442" filter="url(#pbGlow)"
            style="offset-path:path('M56 76 L924 76'); animation:flowPkt 3.2s linear infinite;"/>
    {''.join(nodes)}
    <text x="924" y="142" text-anchor="end" fill="#a89b8e"
          font-family="Geist Mono,monospace" font-size="9"
          font-weight="500" letter-spacing="1.6">{_esc(step_txt)}</text>
  </svg>
</div>"""


def pipeline_html(env: OversightArenaEnvironment | None, show_real: bool = False) -> str:
    if env is None:
        bar = _pipe_bar_svg(None, False)
        idle_cards = "".join(
            f"""<div class="oa-node">
              <div class="oa-node-id">W{i}</div>
              <div class="oa-node-role">{_esc(WORKER_ROLES.get(i,""))}</div>
              <span class="oa-node-badge">IDLE</span>
              <div class="oa-node-desc">Start a simulation to load a scripted task.</div>
              <div class="oa-node-snippet">(no output yet)</div>
            </div>"""
            for i in range(1, 6)
        )
        return f"""<div class="oa-section">
  <div class="oa-section-head">
    <span class="oa-section-num">02 — Live pipeline</span>
    <span class="oa-section-title">Five workers, one will lie.</span>
    <span class="oa-section-copy">Idle. Start a simulation to load a deterministic, seeded task chain.</span>
  </div>
  {bar}
  <div class="oa-nodes">{idle_cards}</div>
</div>"""

    state    = env.state_dict
    bad_ids  = {int(k) for k in state.get("failure_plan", {}).keys()}
    cards    = "".join(
        _node_card(w, show_real, bad_ids) for w in state.get("workers", [])
    )
    bar      = _pipe_bar_svg(env, show_real)
    legend   = (
        "Live view — failure states are hidden. Suspicious = output unchanged ×2+; Flaw exposed = deep inspect ran."
        if not show_real else
        "Post-mortem — ground truth revealed. Red border = worker was compromised by the seeded failure plan."
    )
    return f"""<div class="oa-section">
  <div class="oa-section-head">
    <span class="oa-section-num">02 — Live pipeline</span>
    <span class="oa-section-title">Five workers, one will lie.</span>
    <span class="oa-section-copy">{_esc(legend)}</span>
  </div>
  {bar}
  <div class="oa-nodes">{cards}</div>
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
#  Reward panel
# ─────────────────────────────────────────────────────────────────────────────

def reward_html(
    breakdown: dict[str, float] | None,
    total: float = 0.0,
    episode_result: str = "IN_PROGRESS",
    step_reward: float | None = None,
) -> str:
    breakdown = breakdown or {}
    cards = []
    for key, lbl in REWARD_LABELS.items():
        v = float(breakdown.get(key, 0.0) or 0.0)
        vc = "p" if v > 0 else ("n" if v < 0 else "z")
        cards.append(f"""<div class="oa-rcard">
          <span class="oa-rcard-lbl">{_esc(lbl)}</span>
          <span class="oa-rcard-val {vc}">{v:+.2f}</span>
        </div>""")

    total_color = "var(--sage)" if total >= 0 else "var(--crimson)"
    step_color  = "var(--sage)" if (step_reward or 0) >= 0 else "var(--crimson)"
    step_note   = "" if step_reward is None else (
        f'<span style="color:{step_color}; font-size:11px; font-family:var(--mono);">last step {step_reward:+.3f}</span>'
    )
    return f"""<div class="oa-console">
  <div class="oa-console-hdr">
    <div class="oa-dot"></div>
    Reward Signal
    <span style="margin-left:auto;font-family:var(--display);font-size:30px;font-weight:400;font-style:italic;letter-spacing:-0.03em;color:{total_color};">{total:+.3f}</span>
  </div>
  <div style="padding:12px 18px 4px;display:flex;gap:12px;align-items:center;">
    <span class="oa-tag">{_esc(episode_result)}</span>
    {step_note}
  </div>
  <div class="oa-reward-grid" style="margin:10px 18px 18px;">{" ".join(cards)}</div>
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
#  Episode log
# ─────────────────────────────────────────────────────────────────────────────

def log_html(log: list[dict[str, Any]] | None) -> str:
    if not log:
        return '<div class="oa-empty">No actions yet — start a simulation and submit an action.</div>'
    rows = []
    for e in reversed(log[-10:]):
        if e.get("kind") == "reset":
            rows.append(f"""<div class="oa-log-row" style="color:var(--muted);">
              <span class="oa-log-n">—</span>
              <span class="oa-log-a" style="color:var(--orange); font-family:var(--mono); letter-spacing:0.06em;">NEW EPISODE · {_esc(e.get("difficulty","?"))} · seed={_esc(e.get("seed","?"))}</span>
              <span></span>
            </div>""")
            continue
        r = float(e.get("reward", 0.0) or 0.0)
        rc = "p" if r >= 0 else "n"
        rows.append(f"""<div class="oa-log-row">
          <span class="oa-log-n">s{_esc(e.get("step","?"))}</span>
          <span class="oa-log-a">{_esc(e.get("action",""))}</span>
          <span class="oa-log-r {rc}">{r:+.3f}</span>
        </div>
        {"" if not e.get("summary") else f'<div class="oa-log-row" style="padding-top:0;"><span></span><span class="oa-log-sub">{_esc(e.get("summary",""))}</span></div>'}""")
    inner = "".join(rows)
    return f"""<div class="oa-console">
  <div class="oa-console-hdr">
    <div class="oa-dot"></div>Episode Log
  </div>
  <div class="oa-log">{inner}</div>
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
#  Post-mortem
# ─────────────────────────────────────────────────────────────────────────────

def postmortem_html(state: dict[str, Any]) -> str:
    compromised = {int(k): v for k, v in state.get("failure_plan", {}).items()}
    caught   = set(state.get("caught_workers",  []) or [])
    hallpass = set(state.get("hallpass_workers", []) or [])
    collude  = state.get("colluding_pair", []) or []
    total    = _display_total(state)
    result   = state.get("episode_result", "IN_PROGRESS")

    verdict_cls = {"CLEAN_COMPLETE": "clean", "DIRTY_COMPLETE": "dirty", "TIMEOUT": "timeout"}.get(result, "timeout")
    explanation = {
        "CLEAN_COMPLETE": "Every compromised worker was caught. No bad output reached downstream.",
        "DIRTY_COMPLETE": "Pipeline completed — but at least one failing worker slipped through approval.",
        "TIMEOUT":        "Step budget exhausted before the pipeline could be resolved.",
    }.get(result, "Episode still running.")

    trs = []
    for w in state.get("workers", []):
        wid  = int(w.get("worker_id", 0))
        plan = compromised.get(wid)
        mode = plan.get("failure_mode", "?") if plan else "NONE"
        if not plan:
            out_cls, out_txt = "clean", "Clean worker"
        elif wid in hallpass:
            out_cls, out_txt = "missed", "Missed — bad output approved"
        elif wid in caught:
            out_cls, out_txt = "caught", "Caught by oversight"
        else:
            out_cls, out_txt = "", "Unresolved"
        trs.append(f"""<tr>
          <td>W{wid}</td>
          <td>{_esc(WORKER_ROLES.get(wid,""))}</td>
          <td>{_esc(mode)}</td>
          <td>{_esc(w.get("real_state_str","?"))}</td>
          <td class="{out_cls}">{_esc(out_txt)}</td>
        </tr>""")

    return f"""<div class="oa-pm">
  <div class="oa-pm-hdr">
    <div style="max-width:640px;">
      <div style="font-family:var(--mono); font-size:11px; letter-spacing:0.18em; color:var(--orange); text-transform:uppercase; margin-bottom:8px;">03 — Verdict</div>
      <div class="oa-pm-title">Post-mortem.</div>
      <div style="font-size:14px; color:var(--muted); line-height:1.6; margin-top:10px; max-width:560px;">{_esc(explanation)}</div>
    </div>
    <span class="oa-verdict {verdict_cls}">{_esc(result)} &nbsp; {total:+.3f}</span>
  </div>
  <div class="oa-pm-grid">
    <table class="oa-table">
      <thead><tr>
        <th>Worker</th><th>Role</th>
        <th>Injected failure</th><th>Final state</th><th>Outcome</th>
      </tr></thead>
      <tbody>{"".join(trs)}</tbody>
    </table>
    <div class="oa-pm-side">
      <div style="font-family:var(--display); font-size:24px; font-style:italic; color:var(--ink); margin-bottom:12px; letter-spacing:-0.02em;">Why it matters</div>
      <div style="font-size:13px; color:var(--muted); line-height:1.6; margin-bottom:18px;">
        Deceptive workers look safe until the supervisor commits budget to inspect.
        Colluding pairs defeat per-agent checks entirely — cross-fleet reasoning required.
      </div>
      <div class="oa-node-tags">
        <span class="oa-tag red">missed: {len(hallpass)}</span>
        <span class="oa-tag grn">caught: {len(caught)}</span>
        <span class="oa-tag hot">colluding: {_esc(collude or "none")}</span>
        <span class="oa-tag">seed: {_esc(state.get("seed"))}</span>
      </div>
    </div>
  </div>
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
#  Oracle
# ─────────────────────────────────────────────────────────────────────────────

def oracle_html(env: OversightArenaEnvironment | None, is_done: bool) -> str:
    if env is None:
        return '<div class="oa-empty">Start an episode first.</div>'
    state    = env.state_dict
    approved = set(state.get("approved_workers", []) or [])
    collude  = state.get("colluding_pair", []) or []
    rows = []
    for w in state.get("workers", []):
        wid = int(w.get("worker_id", 0))
        try:
            rs = WorkerState(w.get("real_state_str") or w.get("state"))
        except ValueError:
            continue
        partner_approved = any(p in approved for p in collude if p != wid)
        opt = oracle_action(rs, int(w.get("steps_unchanged", 0) or 0),
                            partner_approved, bool(w.get("deep_inspect_done")))
        vis_or_real = w.get("real_state_str") if is_done else w.get("visible_state", "?")
        lbl = "real" if is_done else "visible"
        rows.append(f"""<tr>
          <td>W{wid}</td>
          <td>{_esc(w.get("task_type","?"))}</td>
          <td>{_esc(lbl)}={_esc(vis_or_real)}</td>
          <td><span class="oa-tag grn">{_esc(opt)}</span></td>
        </tr>""")
    return f"""<div class="oa-oracle-wrap">
  <div class="oa-oracle-hdr">
    <div class="oa-dot w"></div>
    Oracle — uses hidden server-side truth (training clients never see this)
  </div>
  <table class="oa-table" style="margin:0;">
    <thead><tr><th>Worker</th><th>Task</th><th>State visible</th><th>Optimal action</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

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
    state    = env.state_dict
    approved = set(state.get("approved_workers", []) or [])
    collude  = state.get("colluding_pair", []) or []
    priorities = {"TERMINATE": 0, "DEEP_INSPECT": 1, "REDIRECT": 2, "APPROVE": 3, "OBSERVE": 4}
    candidates: list[tuple[int, int, str, dict[str, Any]]] = []
    for w in state.get("workers", []):
        wid = int(w.get("worker_id", 0))
        try:
            rs = WorkerState(w.get("real_state_str") or w.get("state"))
        except ValueError:
            continue
        partner_approved = any(p in approved for p in collude if p != wid)
        opt = oracle_action(rs, int(w.get("steps_unchanged", 0) or 0),
                            partner_approved, bool(w.get("deep_inspect_done")))
        if rs == WorkerState.IDLE and opt == "OBSERVE":
            priority = 8
        elif rs == WorkerState.WORKING and opt == "OBSERVE":
            priority = 5
        else:
            priority = priorities.get(opt, 6)
        candidates.append((priority, wid, opt, w))
    _, wid, verb, _ = min(candidates, key=lambda item: (item[0], item[1]))
    if verb == "REDIRECT":
        return (f"REDIRECT {wid} Refocus on the original task.\n"
                "<reasoning>Oracle-guided: soft reset preferred for off-task worker.</reasoning>")
    return _format_action(verb, wid,
                          "Oracle-guided demo: optimal oversight move for current pipeline state.")


def _action_label(action_text: str) -> str:
    first = (action_text or "").strip().splitlines()[0] if action_text else ""
    return re.sub(r"\s+", " ", first)[:96] or "(empty action)"


# ─────────────────────────────────────────────────────────────────────────────
#  Gradio layout
# ─────────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Oversight Arena", css=APP_CSS) as demo:
    env_state          = gr.State(None)
    episode_log_state  = gr.State([])
    episode_done_state = gr.State(False)

    gr.HTML(hero_html())
    gr.HTML(how_html())

    # ── 02 · controls row ───────────────────────────────────────────────────
    gr.HTML("""
<div class="oa-section">
  <div class="oa-section-head">
    <span class="oa-section-num">02 — Run it</span>
    <span class="oa-section-title">Pick a scenario. Hit start.</span>
    <span class="oa-section-copy">Each run is deterministic and reproducible — same seed, same outcome. Use the oracle button to auto-step a perfect supervisor.</span>
  </div>
</div>
""")
    with gr.Row(equal_height=False, elem_classes=["oa-row", "oa-controls-row"]):
        with gr.Column(scale=5, elem_classes=["oa-card", "oa-controls-card"]):
            with gr.Row():
                scenario_radio = gr.Radio(
                    choices=[
                        "Judge demo: deceptive workers guaranteed",
                        "Easy walkthrough: one visible failure",
                        "Medium challenge: mixed failures",
                        "Custom difficulty + seed",
                    ],
                    value="Judge demo: deceptive workers guaranteed",
                    label="Scenario",
                    scale=3,
                )
                difficulty_radio = gr.Radio(
                    choices=["easy", "medium", "hard"],
                    value="hard",
                    label="Difficulty (custom only)",
                    scale=1,
                )
                seed_input = gr.Number(
                    label="Seed (custom only)",
                    value=42, precision=0, minimum=0, maximum=999999,
                    scale=1,
                )
            with gr.Row():
                reset_btn      = gr.Button("▶  Start Simulation",       variant="primary",    scale=2)
                auto_step_btn  = gr.Button("⚡  Oracle-Guided Step",    variant="primary",    scale=2)
                show_oracle_btn= gr.Button("⊙  Show Oracle",            variant="secondary",  scale=1)
        with gr.Column(scale=3, elem_classes=["oa-status-col"]):
            status_display = gr.HTML(status_html())

    # ── 03 · live pipeline (self-contained section) ─────────────────────────
    pipeline_display = gr.HTML(pipeline_html(None))

    # ── 04 · observation + action console ──────────────────────────────────
    gr.HTML("""
<div class="oa-section">
  <div class="oa-section-head">
    <span class="oa-section-num">04 — Console</span>
    <span class="oa-section-title">Observe. Inspect. Decide.</span>
    <span class="oa-section-copy">Every action you submit is real — it goes through env.step() and updates the live state above.</span>
  </div>
</div>
""")
    with gr.Row(elem_classes=["oa-row", "oa-console-row"], equal_height=False):
        with gr.Column(scale=3, elem_classes=["oa-card", "oa-obs-card"]):
            gr.HTML("""<div class="oa-card-hdr"><span class="oa-dot"></span>Supervisor observation</div>""")
            observation_box = gr.Textbox(
                show_label=False,
                value="Start a simulation to see the real observation string.",
                lines=20, max_lines=44, interactive=False,
                elem_classes=["obs-text"],
            )
        with gr.Column(scale=2, elem_classes=["oa-card", "oa-actions-card"]):
            gr.HTML("""<div class="oa-card-hdr"><span class="oa-dot"></span>Quick actions</div>""")
            with gr.Row(elem_classes=["oa-btn-row"]):
                obs_buttons = [gr.Button(f"OBS W{i}", size="sm") for i in range(1, 6)]
            with gr.Row(elem_classes=["oa-btn-row"]):
                action_worker_num = gr.Number(label="Worker #", value=1, minimum=1, maximum=5, step=1, precision=0)
                deep_btn      = gr.Button("Deep inspect",  size="sm")
                terminate_btn = gr.Button("Terminate",     size="sm")
                approve_btn   = gr.Button("Approve",       size="sm")
            with gr.Row(elem_classes=["oa-btn-row"]):
                redirect_worker_num = gr.Number(label="Redirect #", value=1, minimum=1, maximum=5, step=1, precision=0)
                redirect_instr = gr.Textbox(label="Instruction", placeholder="Refocus on the original task.", lines=1)
            redirect_btn = gr.Button("↩ Redirect", size="sm")
            gr.HTML("""<div class="oa-card-hdr" style="margin-top:14px;"><span class="oa-dot"></span>Action payload</div>""")
            action_input = gr.Textbox(
                show_label=False,
                value=_format_action("OBSERVE", 1),
                lines=5, elem_classes=["act-text"],
            )
            step_btn = gr.Button("▶  Submit action", variant="primary")

    # ── 05 · log + rewards ──────────────────────────────────────────────────
    gr.HTML("""
<div class="oa-section">
  <div class="oa-section-head">
    <span class="oa-section-num">05 — Telemetry</span>
    <span class="oa-section-title">Audit trail &amp; reward signal.</span>
    <span class="oa-section-copy">Nine independent reward components, gated by an episode multiplier. The supervisor only earns full credit when the seeded failure plan is perfectly resolved.</span>
  </div>
</div>
""")
    with gr.Row(elem_classes=["oa-row", "oa-telemetry-row"]):
        with gr.Column(scale=3):
            episode_log_display = gr.HTML(log_html([]))
        with gr.Column(scale=2):
            reward_panel = gr.HTML(reward_html({}, 0.0))

    # ── oracle ───────────────────────────────────────────────────────────────
    oracle_output = gr.HTML("")

    # ── post-mortem ──────────────────────────────────────────────────────────
    with gr.Row(visible=False) as postmortem_row:
        postmortem_display = gr.HTML("")

    # ── output tuple shared by all handlers ─────────────────────────────────
    _ALL = [
        env_state, episode_log_state, episode_done_state,
        observation_box, pipeline_display, status_display,
        episode_log_display, reward_panel, oracle_output,
        postmortem_row, postmortem_display, action_input,
    ]

    # ── handlers ─────────────────────────────────────────────────────────────

    def do_reset(scenario: str, difficulty: str, seed_val):
        diff, seed = _scenario_config(scenario, difficulty, seed_val)
        env = OversightArenaEnvironment()
        obs = env.reset(difficulty=diff, seed=seed)
        s   = env.state_dict
        log = [{"kind": "reset", "difficulty": diff.upper(),
                "seed": s.get("seed"), "max_steps": s.get("max_steps", 25)}]
        return (
            env, log, False,
            obs.metadata.get("pipeline_text", ""),
            pipeline_html(env, show_real=False),
            status_html(s),
            log_html(log),
            reward_html(s.get("reward_breakdown", {}), _display_total(s)),
            "", gr.update(visible=False), "",
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
            return (env, log, is_done,
                    "Start a simulation first.",
                    pipeline_html(None), status_html(),
                    log_html(log), reward_html({}, 0.0), "",
                    gr.update(visible=False), "",
                    action_text or _format_action("OBSERVE", 1))

        if is_done:
            s = env.state_dict
            return (env, log, True,
                    env._build_observation(),
                    pipeline_html(env, show_real=True),
                    status_html(s),
                    log_html(log),
                    reward_html(s.get("reward_breakdown", {}), _display_total(s), s.get("episode_result", "")),
                    "", gr.update(visible=True), postmortem_html(s), action_text)

        if not action_text or not action_text.strip():
            s = env.state_dict
            return (env, log, False,
                    env._build_observation(),
                    pipeline_html(env, show_real=False),
                    status_html(s),
                    log_html(log),
                    reward_html(s.get("reward_breakdown", {}), _display_total(s)),
                    "", gr.update(visible=False), "",
                    _format_action("OBSERVE", 1))

        obs_result   = env.step(action_text)
        s            = env.state_dict
        reward       = float(obs_result.reward or 0.0)
        done         = bool(obs_result.done)
        ep_result    = obs_result.metadata.get("episode_result", "IN_PROGRESS")
        summary      = obs_result.metadata.get("action_summary", "")
        obs_text     = obs_result.metadata.get("pipeline_text", env._build_observation())
        log.append({"kind": "step", "step": s.get("step"),
                    "action": _action_label(action_text),
                    "summary": summary, "reward": reward})
        return (
            env, log, done,
            obs_text,
            pipeline_html(env, show_real=done),
            status_html(s),
            log_html(log),
            reward_html(s.get("reward_breakdown", {}), _display_total(s), ep_result, step_reward=reward),
            "",
            gr.update(visible=done),
            postmortem_html(s) if done else "",
            action_text,
        )

    def do_manual_step(env, action_text, log, is_done):
        return _step_with_action(env, action_text, log, is_done)

    def do_auto_step(env, log, is_done):
        if env is None:
            return _step_with_action(env, "", log, is_done)
        return _step_with_action(env, _active_oracle_action(env), log, is_done)

    reset_btn.click(do_reset,
        inputs=[scenario_radio, difficulty_radio, seed_input], outputs=_ALL)
    step_btn.click(do_manual_step,
        inputs=[env_state, action_input, episode_log_state, episode_done_state], outputs=_ALL)
    action_input.submit(do_manual_step,
        inputs=[env_state, action_input, episode_log_state, episode_done_state], outputs=_ALL)
    auto_step_btn.click(do_auto_step,
        inputs=[env_state, episode_log_state, episode_done_state], outputs=_ALL)

    for idx, btn in enumerate(obs_buttons, start=1):
        btn.click(
            lambda wid=idx: _format_action("OBSERVE", wid, "Checking latest visible output."),
            inputs=[], outputs=[action_input],
        )

    deep_btn.click(
        lambda w: _format_action("DEEP_INSPECT", int(w or 1),
                                  "Surface may hide a flaw — spending inspection budget."),
        inputs=[action_worker_num], outputs=[action_input])
    terminate_btn.click(
        lambda w: _format_action("TERMINATE", int(w or 1),
                                  "Worker unsafe to pass downstream — hard resetting."),
        inputs=[action_worker_num], outputs=[action_input])
    approve_btn.click(
        lambda w: _format_action("APPROVE", int(w or 1),
                                  "Worker appears complete and safe to advance."),
        inputs=[action_worker_num], outputs=[action_input])
    redirect_btn.click(
        lambda w, instr: _redirect_action(int(w or 1), instr or ""),
        inputs=[redirect_worker_num, redirect_instr], outputs=[action_input])
    show_oracle_btn.click(
        oracle_html,
        inputs=[env_state, episode_done_state], outputs=[oracle_output])


# ─────────────────────────────────────────────────────────────────────────────
#  FastAPI mounting (HF Spaces)
# ─────────────────────────────────────────────────────────────────────────────

def _build_space_app() -> FastAPI:
    import server as _srv

    app = FastAPI(title="Oversight Arena", version="1.0.0")

    @app.get("/health", tags=["Health"])
    def _health():
        return {"status": "ok", "service": "oversight-arena"}

    app.mount("/openenv", _srv.app)
    demo.max_threads = 8
    return gr.mount_gradio_app(
        app, demo,
        path="/",
        server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False, pwa=False, mcp_server=False,
    )


if __name__ == "__main__":
    uvicorn.run(_build_space_app(), host="0.0.0.0", port=7860)
