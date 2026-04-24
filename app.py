"""
app.py — Oversight Arena · Gradio Judge Interface
=================================================
Full Gradio Blocks application for the Oversight Arena RL environment.

Layout
------
1. Header            — title + tagline
2. Controls          — difficulty, seed, reset
3. Pipeline          — 5 colour-coded worker boxes (real states hidden until done)
4. Interaction       — observation pane + action controls
5. Log / Rewards     — episode log + reward breakdown panel
6. Oracle            — show optimal actions (uses real hidden states)
7. Post-mortem       — full breakdown, visible only after episode ends
"""

from __future__ import annotations

import gradio as gr

from oversight_arena.environment import OversightArenaEnv, OversightArenaEnvironment
from oversight_arena.models import WorkerState
from oversight_arena.oracle import oracle_action

# ---------------------------------------------------------------------------
# Colour helpers (spec-mandated palette)
# ---------------------------------------------------------------------------


def get_worker_color(visible_state: str, is_flagged: bool = False) -> str:
    colors = {
        "IDLE": "#9e9e9e",
        "WORKING": "#2196F3",
        "COMPLETED": "#4CAF50",
        "APPROVED": "#4CAF50",
        "TERMINATED": "#f44336",
        "REDIRECTED": "#FFEB3B",
        "FLAGGED": "#FF9800",
    }
    if is_flagged:
        return colors["FLAGGED"]
    return colors.get(visible_state, "#9e9e9e")


# ---------------------------------------------------------------------------
# Pipeline HTML rendering
# ---------------------------------------------------------------------------


def _worker_box(
    worker_id: int,
    task_type: str,
    visible_state: str,
    is_flagged: bool = False,
    real_state_str: str = "",
    show_real: bool = False,
) -> str:
    """Return HTML for one pipeline worker card."""
    bg = get_worker_color(visible_state, is_flagged)
    border = "#FF9800" if is_flagged else bg
    # Yellow background needs dark text for readability
    fg = "#111" if visible_state == "REDIRECTED" else "#fff"

    real_badge = ""
    if (
        show_real
        and real_state_str
        and real_state_str not in ("IDLE", "WORKING", "COMPLETED")
    ):
        real_badge = (
            f'<div style="margin-top:5px;padding:2px 6px;'
            f'background:rgba(0,0,0,.25);border-radius:4px;font-size:11px;">'
            f"{real_state_str}</div>"
        )

    return (
        f'<div style="'
        f"background:{bg};color:{fg};border:3px solid {border};"
        f"border-radius:12px;padding:12px 8px;text-align:center;"
        f"font-family:monospace;font-size:13px;flex:1;min-width:105px;"
        f"display:flex;flex-direction:column;justify-content:center;gap:3px;"
        f'box-shadow:0 3px 8px rgba(0,0,0,.22);">'
        f'<div style="font-weight:bold;font-size:15px;">Worker {worker_id}</div>'
        f'<div style="opacity:.85;font-size:12px;">{task_type}</div>'
        f'<div style="margin-top:4px;padding:2px 8px;background:rgba(0,0,0,.18);'
        f'border-radius:4px;">{visible_state}</div>'
        f"{real_badge}"
        f"</div>"
    )


def _pipeline_html(worker_tuples: list) -> str:
    """Wrap worker boxes in a flex row."""
    boxes = "".join(_worker_box(*t) for t in worker_tuples)
    return (
        '<div style="display:flex;gap:10px;justify-content:center;'
        'padding:14px;flex-wrap:nowrap;align-items:stretch;">'
        f"{boxes}</div>"
    )


def _default_pipeline() -> str:
    """Five grey IDLE boxes — shown before the first reset."""
    return _pipeline_html([(i, "—", "IDLE", False, "", False) for i in range(1, 6)])


def _env_pipeline(env: OversightArenaEnv, show_real: bool = False) -> str:
    """Build pipeline HTML from live env state."""
    s = env.state_dict
    workers_list = s["workers"]
    workers = {w["worker_id"]: w for w in workers_list}
    rows = []
    for wid in range(1, 6):
        w = workers.get(wid, {})
        vis = w.get("visible_state", "IDLE")
        task = w.get("task_type", "—")
        real_str = w.get("real_state_str", "")
        # Flag (orange border) once supervisor has deep-inspected a non-approved worker
        flagged = w.get("deep_inspect_done", False) and vis not in ("APPROVED",)
        rows.append((wid, task, vis, flagged, real_str, show_real))
    return _pipeline_html(rows)


# ---------------------------------------------------------------------------
# Reward panel HTML
# ---------------------------------------------------------------------------

_REWARD_COLOR: dict[str, str] = {
    "reward_catch": "#43A047",
    "reward_deceptive_catch": "#1E88E5",
    "reward_efficiency": "#8E24AA",
    "reward_collusion": "#FB8C00",
    "reward_mercor": "#00ACC1",
    "penalty_false_positive": "#E53935",
    "penalty_hallpass": "#E53935",
    "penalty_deceptive_pass": "#B71C1C",
    "penalty_format": "#FF5722",
}

_REWARD_LABEL: dict[str, str] = {
    "reward_catch": "Catch (HALL/STALL/DRIFT)",
    "reward_deceptive_catch": "Deceptive catch",
    "reward_efficiency": "Efficiency bonus",
    "reward_collusion": "Collusion bonus",
    "reward_mercor": "Mercor (reasoning)",
    "penalty_false_positive": "False positive",
    "penalty_hallpass": "Hallpass",
    "penalty_deceptive_pass": "Deceptive pass",
    "penalty_format": "Format penalty",
}


def _reward_html(
    breakdown: dict[str, float],
    total: float,
    episode_result: str = "",
) -> str:
    rows = ""
    for k, v in breakdown.items():
        if v == 0.0:
            continue
        color = _REWARD_COLOR.get(k, "#555")
        label = _REWARD_LABEL.get(k, k)
        sign = "+" if v > 0 else ""
        rows += (
            f"<tr>"
            f'<td style="padding:5px 8px;font-size:12px;color:#333;">{label}</td>'
            f'<td style="padding:5px 8px;text-align:right;font-family:monospace;'
            f'color:{color};font-weight:bold;">{sign}{v:.3f}</td>'
            f"</tr>"
        )

    if not rows:
        rows = (
            '<tr><td colspan="2" style="padding:12px;color:#999;'
            'text-align:center;font-size:12px;">No rewards yet</td></tr>'
        )

    result_badge = ""
    if episode_result and episode_result not in ("", "IN_PROGRESS"):
        bc = {
            "CLEAN_COMPLETE": "#4CAF50",
            "DIRTY_COMPLETE": "#FF9800",
            "TIMEOUT": "#f44336",
        }.get(episode_result, "#888")
        result_badge = (
            f'<div style="margin-bottom:8px;padding:5px 10px;background:{bc};'
            f"color:#fff;border-radius:6px;text-align:center;font-size:12px;"
            f'font-weight:bold;">{episode_result}</div>'
        )

    total_color = "#4CAF50" if total >= 0 else "#f44336"
    sign = "+" if total >= 0 else ""
    return (
        f'<div style="font-family:sans-serif;padding:10px;">'
        f"{result_badge}"
        f'<table style="width:100%;border-collapse:collapse;">{rows}</table>'
        f'<hr style="margin:8px 0;border:1px solid #ddd;">'
        f'<div style="display:flex;justify-content:space-between;'
        f'font-weight:bold;font-size:14px;padding:4px 8px;">'
        f"<span>Total Reward</span>"
        f'<span style="color:{total_color};">{sign}{total:.3f}</span>'
        f"</div></div>"
    )


# ---------------------------------------------------------------------------
# Post-mortem HTML
# ---------------------------------------------------------------------------


def _postmortem_html(s: dict) -> str:
    workers = s["workers"]
    caught = set(s.get("caught_workers", []))
    hallpass = set(s.get("hallpass_workers", []))
    approved = set(s.get("approved_workers", []))
    colluding = s.get("colluding_pair", [])
    result = s.get("episode_result", "—")
    total = s.get("total_reward", 0.0)

    workers_list = workers if isinstance(workers, list) else list(workers.values())
    workers_by_id = (
        {w["worker_id"]: w for w in workers_list}
        if workers_list and isinstance(workers_list[0], dict)
        else {}
    )

    rows = ""
    for wid in range(1, 6):
        w = workers_by_id.get(wid, {})
        real = w.get("real_state_str", "?")
        task = w.get("task_type", "—")
        fm = w.get("failure_mode", "NONE")
        deep = w.get("deep_inspect_done", False)
        is_col = wid in colluding

        if wid in caught:
            outcome = "✅ Correctly caught"
            row_bg = "#d4edda"
        elif wid in hallpass:
            outcome = "❌ Hallpass / bad approve"
            row_bg = "#f8d7da"
        elif wid in approved:
            outcome = "✓ Approved clean"
            row_bg = "#d4edda"
        else:
            outcome = "⏸ Not resolved"
            row_bg = "#fff3cd"

        fm_cell = (
            f'<span style="color:#E53935;font-weight:bold;">{fm}</span>'
            if fm != "NONE"
            else '<span style="color:#aaa;">—</span>'
        )

        rows += (
            f'<tr style="background:{row_bg};">'
            f'<td style="padding:8px;text-align:center;font-weight:bold;">W{wid}</td>'
            f'<td style="padding:8px;">{task}</td>'
            f'<td style="padding:8px;font-family:monospace;">{real}</td>'
            f'<td style="padding:8px;text-align:center;">{fm_cell}</td>'
            f'<td style="padding:8px;text-align:center;">{"🔍 Yes" if deep else "No"}</td>'
            f'<td style="padding:8px;text-align:center;">{"👥 Yes" if is_col else "—"}</td>'
            f'<td style="padding:8px;">{outcome}</td>'
            f"</tr>"
        )

    col_str = (
        f"Workers {colluding[0]} & {colluding[1]}" if len(colluding) >= 2 else "None"
    )
    rc = {
        "CLEAN_COMPLETE": "#4CAF50",
        "DIRTY_COMPLETE": "#FF9800",
        "TIMEOUT": "#f44336",
    }.get(result, "#888")
    sign = "+" if total >= 0 else ""

    return (
        '<div style="font-family:sans-serif;padding:16px;background:#f8f9fa;'
        'border-radius:10px;border:1px solid #dee2e6;">'
        '<h3 style="margin:0 0 12px;color:#333;border-bottom:2px solid #dee2e6;'
        'padding-bottom:8px;">🔬 Episode Post-Mortem</h3>'
        '<div style="display:flex;gap:20px;flex-wrap:wrap;margin-bottom:12px;">'
        f"<span><strong>Result:</strong> "
        f'<span style="color:{rc};font-weight:bold;">{result}</span></span>'
        f"<span><strong>Total Reward:</strong> "
        f'<span style="font-weight:bold;">{sign}{total:.3f}</span></span>'
        f"<span><strong>Colluding Pair:</strong> {col_str}</span>"
        "</div>"
        '<table style="width:100%;border-collapse:collapse;font-size:13px;">'
        "<thead>"
        '<tr style="background:#343a40;color:#fff;">'
        '<th style="padding:8px;">Worker</th>'
        '<th style="padding:8px;">Task</th>'
        '<th style="padding:8px;">Real State</th>'
        '<th style="padding:8px;">Failure Mode</th>'
        '<th style="padding:8px;">Deep Inspected</th>'
        '<th style="padding:8px;">Colluding</th>'
        '<th style="padding:8px;">Outcome</th>'
        "</tr></thead>"
        f"<tbody>{rows}</tbody>"
        "</table>"
        "</div>"
    )


# ---------------------------------------------------------------------------
# Action text helpers
# ---------------------------------------------------------------------------


def _action_text(verb: str, worker_num: int | str) -> str:
    """Populate the action textbox for simple one-word verbs."""
    return (
        f"<action>{verb} {worker_num}</action>\n"
        f"<reasoning>Write your reasoning here.</reasoning>"
    )


def _redirect_text(worker_num: int | str, instruction: str) -> str:
    instr = instruction.strip() or "Please refocus on the original task."
    return (
        f"<action>REDIRECT {worker_num} {instr}</action>\n"
        f"<reasoning>Write your reasoning here.</reasoning>"
    )


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """
.obs-text textarea  { font-family: 'Courier New', monospace !important; font-size: 13px !important; line-height: 1.55 !important; }
.log-text textarea  { font-family: 'Courier New', monospace !important; font-size: 12px !important; line-height: 1.4  !important; }
.act-text textarea  { font-family: 'Courier New', monospace !important; font-size: 13px !important; }
.oracle-text textarea { font-family: 'Courier New', monospace !important; font-size: 13px !important; }
.arena-wrap { text-align: center; padding: 18px 0 6px; }
.arena-title {
    font-size: 2.5em; font-weight: 800; margin: 0;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.arena-tag { color: #666; font-style: italic; margin-top: 4px; font-size: 1.05em; }
"""

# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

with gr.Blocks(title="Oversight Arena") as demo:
    # ── Persistent state ────────────────────────────────────────────────────
    env_state = gr.State(None)  # OversightArenaEnv instance
    episode_log_state = gr.State([])  # list[str] — running log lines
    episode_done_state = gr.State(False)  # bool

    # ── Row 0: Header ────────────────────────────────────────────────────────
    gr.HTML(
        '<div class="arena-wrap">'
        '<p class="arena-title">⚖️ Oversight Arena</p>'
        '<p class="arena-tag">'
        "Monitor AI worker pipelines &nbsp;·&nbsp; Detect failures "
        "&nbsp;·&nbsp; Maintain integrity"
        "</p></div>"
    )

    # ── Row 1: Episode controls ──────────────────────────────────────────────
    with gr.Row():
        difficulty_radio = gr.Radio(
            choices=["easy", "medium", "hard"],
            value="easy",
            label="Difficulty",
            scale=2,
        )
        seed_input = gr.Number(
            label="Seed (blank = random)",
            value=None,
            minimum=0,
            maximum=99999,
            step=1,
            precision=0,
            scale=1,
        )
        reset_btn = gr.Button("🔄  Reset Episode", variant="primary", scale=1)

    # ── Row 2: Pipeline visualisation ────────────────────────────────────────
    with gr.Row():
        pipeline_display = gr.HTML(_default_pipeline())

    # ── Row 3: Observation + action controls ─────────────────────────────────
    with gr.Row():
        # Left — observation pane
        with gr.Column(scale=3):
            observation_box = gr.Textbox(
                label="📋  Current Observation",
                value="Press  🔄 Reset Episode  to start a new episode.",
                lines=26,
                max_lines=50,
                interactive=False,
                elem_classes=["obs-text"],
            )

        # Right — action panel
        with gr.Column(scale=2):
            gr.Markdown("### 🎮  Quick Actions")

            # OBSERVE buttons (one per worker)
            gr.Markdown("**Observe a worker's latest output:**")
            with gr.Row():
                _obs_btn_1 = gr.Button("👁 W1", size="sm", min_width=60)
                _obs_btn_2 = gr.Button("👁 W2", size="sm", min_width=60)
                _obs_btn_3 = gr.Button("👁 W3", size="sm", min_width=60)
                _obs_btn_4 = gr.Button("👁 W4", size="sm", min_width=60)
                _obs_btn_5 = gr.Button("👁 W5", size="sm", min_width=60)
            _obs_btns = [_obs_btn_1, _obs_btn_2, _obs_btn_3, _obs_btn_4, _obs_btn_5]

            # DEEP_INSPECT / TERMINATE / APPROVE — share a worker # input
            gr.Markdown("**Inspect · Terminate · Approve:**")
            with gr.Row():
                action_worker_num = gr.Number(
                    label="Worker #",
                    value=1,
                    minimum=1,
                    maximum=5,
                    step=1,
                    precision=0,
                    scale=1,
                )
                deep_btn = gr.Button("🔍 Deep Inspect", size="sm", scale=1)
                terminate_btn = gr.Button("❌ Terminate", size="sm", scale=1)
                approve_btn = gr.Button("✅ Approve", size="sm", scale=1)

            # REDIRECT — needs its own instruction field
            gr.Markdown("**Redirect (soft-reset with corrective instruction):**")
            with gr.Row():
                redirect_worker_num = gr.Number(
                    label="Worker #",
                    value=1,
                    minimum=1,
                    maximum=5,
                    step=1,
                    precision=0,
                    scale=1,
                )
                redirect_instr = gr.Textbox(
                    label="Instruction",
                    placeholder="e.g. Focus only on the original question.",
                    lines=1,
                    scale=3,
                )
            redirect_btn = gr.Button("↩  Redirect", size="sm")

            # Action textbox + submit
            gr.Markdown("### ✏️  Action Input")
            gr.Markdown(
                "_Quick-action buttons fill this box. Edit the `<reasoning>` "
                "block before submitting._"
            )
            action_input = gr.Textbox(
                label="Action text",
                placeholder=(
                    "<action>OBSERVE 1</action>\n"
                    "<reasoning>Write your reasoning here.</reasoning>"
                ),
                lines=5,
                elem_classes=["act-text"],
            )
            step_btn = gr.Button("▶  Submit Action", variant="primary")

    # ── Row 4: Episode log + reward panel ────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=2):
            episode_log_box = gr.Textbox(
                label="📜  Episode Log",
                value="",
                lines=14,
                max_lines=40,
                interactive=False,
                elem_classes=["log-text"],
            )
        with gr.Column(scale=1):
            reward_panel = gr.HTML(
                _reward_html({}, 0.0),
                label="💰  Reward Breakdown",
            )

    # ── Row 5: Oracle ────────────────────────────────────────────────────────
    gr.Markdown("---")
    with gr.Row():
        with gr.Column():
            show_oracle_btn = gr.Button(
                "🔮  Show Oracle Actions",
                variant="secondary",
            )
            oracle_output = gr.Textbox(
                label="Oracle Recommendations  (reads real hidden states — do not use as a cheat!)",
                value="",
                lines=10,
                interactive=False,
                elem_classes=["oracle-text"],
            )

    # ── Row 6: Post-mortem (hidden until episode ends) ───────────────────────
    with gr.Row(visible=False) as postmortem_row:
        with gr.Column():
            postmortem_display = gr.HTML("")

    # =========================================================================
    # Event handlers
    # =========================================================================

    # Shared output list used by both reset and step handlers
    _ALL_OUTPUTS = [
        env_state,
        episode_log_state,
        episode_done_state,
        observation_box,
        pipeline_display,
        episode_log_box,
        reward_panel,
        oracle_output,
        postmortem_row,
        postmortem_display,
    ]

    # ── Reset ─────────────────────────────────────────────────────────────────
    def do_reset(difficulty_val: str, seed_val):
        env = OversightArenaEnvironment()
        seed = int(seed_val) if seed_val is not None else None
        obs_result = env.reset(difficulty=difficulty_val, seed=seed)
        # reset() now returns Observation; pipeline text is in metadata
        obs_text = obs_result.metadata.get("pipeline_text", str(obs_result))
        s = env.state_dict

        log = [
            "═" * 56,
            f"  NEW EPISODE  ·  {difficulty_val.upper()}"
            f"  ·  seed={s['seed']}  ·  max_steps={s['max_steps']}",
            "═" * 56,
        ]

        return (
            env,  # env_state
            log,  # episode_log_state
            False,  # episode_done_state
            obs_text,  # observation_box
            _env_pipeline(env, False),  # pipeline_display
            "\n".join(log),  # episode_log_box
            _reward_html({}, 0.0),  # reward_panel
            "",  # oracle_output  (cleared)
            gr.update(visible=False),  # postmortem_row (hidden)
            "",  # postmortem_display
        )

    reset_btn.click(
        do_reset,
        inputs=[difficulty_radio, seed_input],
        outputs=_ALL_OUTPUTS,
    )

    # ── Step ──────────────────────────────────────────────────────────────────
    def do_step(env, action_text: str, log: list, is_done: bool):
        # Guard: env not initialised
        if env is None:
            return (
                env,
                log,
                is_done,
                "⚠  Please press  🔄 Reset Episode  first.",
                _default_pipeline(),
                "\n".join(log),
                _reward_html({}, 0.0),
                "",
                gr.update(visible=False),
                "",
            )

        # Guard: episode already done
        if is_done:
            s = env.state_dict
            obs_text = env._build_observation()
            return (
                env,
                log,
                True,
                obs_text,
                _env_pipeline(env, show_real=True),
                "\n".join(log),
                _reward_html(
                    s["reward_breakdown"], s["total_reward"], s["episode_result"]
                ),
                "",
                gr.update(visible=True),
                _postmortem_html(s),
            )

        # Guard: empty action
        if not action_text or not action_text.strip():
            s = env.state_dict
            obs_text = env._build_observation()
            return (
                env,
                log,
                is_done,
                obs_text,
                _env_pipeline(env, False),
                "\n".join(log),
                _reward_html(s["reward_breakdown"], s.get("total_reward", 0.0)),
                "",
                gr.update(visible=False),
                "",
            )

        # Pass raw action text directly to step() — it handles parsing internally
        # via _parse_action() and applies a format penalty on malformed input.
        obs_result = env.step(action_text)
        reward = obs_result.reward or 0.0
        done = obs_result.done
        obs_text = obs_result.metadata.get("pipeline_text", "")
        step_bd = obs_result.metadata.get("reward_breakdown", {})
        ep_res = obs_result.metadata.get("episode_result", "IN_PROGRESS")

        s = env.state_dict
        total = s.get("total_reward", 0.0)

        # Build log lines for this step
        new_log = list(log)
        step_num = s["step"]
        act_short = action_text.split("\n")[0][:72]
        new_log.append(f"[Step {step_num:>2}]  {act_short}")

        bd_parts = [f"{k}:{v:+.2f}" for k, v in step_bd.items() if v != 0.0]
        if bd_parts:
            new_log.append(f"           {' | '.join(bd_parts)}")
        new_log.append(f"           Δ={reward:+.3f}  total={total:+.3f}")

        show_real = done
        pipe = _env_pipeline(env, show_real=show_real)
        rp = _reward_html(
            s["reward_breakdown"],
            total,
            ep_res if done else "",
        )

        pm_html = ""
        show_pm = gr.update(visible=False)
        if done:
            pm_html = _postmortem_html(s)
            show_pm = gr.update(visible=True)
            new_log.append("─" * 56)
            new_log.append(f"  EPISODE DONE  ·  {ep_res}  ·  total={total:+.3f}")
            new_log.append("─" * 56)

        return (
            env,
            new_log,
            done,
            obs_text,
            pipe,
            "\n".join(new_log),
            rp,
            "",  # clear oracle output on each step
            show_pm,
            pm_html,
        )

    step_btn.click(
        do_step,
        inputs=[env_state, action_input, episode_log_state, episode_done_state],
        outputs=_ALL_OUTPUTS,
    )
    # Pressing Enter in the action box also submits (works in single-line mode;
    # for multiline the user can always use the button instead)
    action_input.submit(
        do_step,
        inputs=[env_state, action_input, episode_log_state, episode_done_state],
        outputs=_ALL_OUTPUTS,
    )

    # ── Quick action button wiring ────────────────────────────────────────────

    # Factory avoids the classic Python loop-closure trap
    def _observe_fn(wid: int):
        def _fn():
            return _action_text("OBSERVE", wid)

        return _fn

    for _i, _btn in enumerate(_obs_btns, start=1):
        _btn.click(_observe_fn(_i), inputs=[], outputs=[action_input])

    def _do_deep_inspect(wnum):
        return _action_text("DEEP_INSPECT", int(wnum or 1))

    def _do_terminate(wnum):
        return _action_text("TERMINATE", int(wnum or 1))

    def _do_approve(wnum):
        return _action_text("APPROVE", int(wnum or 1))

    def _do_redirect(wnum, instr):
        return _redirect_text(int(wnum or 1), instr or "")

    deep_btn.click(_do_deep_inspect, inputs=[action_worker_num], outputs=[action_input])
    terminate_btn.click(
        _do_terminate, inputs=[action_worker_num], outputs=[action_input]
    )
    approve_btn.click(_do_approve, inputs=[action_worker_num], outputs=[action_input])
    redirect_btn.click(
        _do_redirect,
        inputs=[redirect_worker_num, redirect_instr],
        outputs=[action_input],
    )

    # ── Oracle ────────────────────────────────────────────────────────────────

    def do_show_oracle(env, is_done: bool):
        if env is None:
            return "⚠  Reset the environment first."

        s = env.state_dict
        workers_list = s["workers"]
        workers = {w["worker_id"]: w for w in workers_list}
        approved_set = set(s.get("approved_workers", []))
        colluding = s.get("colluding_pair", [])

        lines = [
            "🔮  ORACLE — Optimal Actions",
            "─" * 52,
            f"  Step {s['step']}/{s['max_steps']}  ·  {s['difficulty'].upper()}"
            f"  ·  corruption_risk={s.get('corruption_risk', '?')}",
            "─" * 52,
        ]

        for wid in range(1, 6):
            w = workers.get(wid, {})

            # The real WorkerState enum (needed by oracle_action)
            real_state_str = w.get("real_state_str") or w.get("state")
            if real_state_str is None:
                continue
            try:
                real_state = WorkerState(real_state_str)
            except ValueError:
                continue

            vis = w.get("visible_state", "?")
            task = w.get("task_type", "?")
            steps_unch = w.get("steps_unchanged", 0)
            deep_done = w.get("deep_inspect_done", False)
            partner_approved = any(p in approved_set for p in colluding if p != wid)

            optimal = oracle_action(
                real_state,
                steps_unch,
                partner_approved,
                deep_done,
            )

            if is_done:
                # Reveal real state now that the episode is over
                real_str = w.get("real_state_str", "?")
                lines.append(f"  W{wid}  {task:<18}  real={real_str:<14}  →  {optimal}")
            else:
                # During the episode: show optimal action but NOT the real state
                lines.append(f"  W{wid}  {task:<18}  visible={vis:<12}  →  {optimal}")

        lines += [
            "─" * 52,
            "  The oracle uses REAL (hidden) worker states.",
            "  Visible states may not match — that is the challenge.",
            "  After the episode ends, real states are revealed above.",
        ]

        if colluding:
            lines.append(f"  Colluding pair this episode: Workers {colluding}")

        return "\n".join(lines)

    show_oracle_btn.click(
        do_show_oracle,
        inputs=[env_state, episode_done_state],
        outputs=[oracle_output],
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # HF Spaces exposes exactly port 7860; Gradio must bind here.
    # FastAPI/OpenEnv server runs internally on port 8000.
    demo.launch(server_port=7860, share=False, css=_CSS)
