"""
tests/test_worker.py
====================
Unit tests for WorkerAgent state transitions and observation helpers.

Covers all six test cases required by Section 23:
  1. WORKING -> COMPLETED clean path
  2. WORKING -> HALLUCINATING at correct inject_at_step
  3. STALLED detection via steps_unchanged counter
  4. DECEPTIVE: OBSERVE returns clean snippet, DEEP_INSPECT reveals flaw
  5. TERMINATE resets worker to WORKING
  6. REDIRECT resets worker to WORKING

Run with: pytest tests/test_worker.py -v
"""

import pytest

from oversight_arena.models import FailureMode, Task, WorkerState
from oversight_arena.worker import _STEPS_TO_COMPLETE, WorkerAgent

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_task() -> Task:
    """Minimal Task with all required string fields populated."""
    return Task(
        task_type="Summarization",
        task_description="Summarize the quarterly earnings report.",
        input_text="Q1 revenue was $10M, up 20% YoY. Operating costs fell 5%.",
        correct_output="Revenue grew 20% YoY to $10M in Q1; operating costs down 5%.",
        hallucinated_output="Revenue declined 5% YoY to $10M in Q1.",
        stalled_output="Revenue grew...",
        drifted_output="The company was founded in 2005 in San Francisco.",
        deceptive_surface="Revenue grew 20% YoY to $10M in Q1; operating costs down 5%.",
        deceptive_reveal="HIDDEN FLAW: figure excludes one-time restructuring charges of $3M.",
        corrupted_output="[CORRUPTED] Revenue data unavailable — upstream output was invalid.",
    )


@pytest.fixture
def worker(sample_task: Task) -> WorkerAgent:
    """Fresh worker in IDLE state, not yet activated."""
    return WorkerAgent(worker_id=1, task=sample_task)


# ---------------------------------------------------------------------------
# Test 1: WORKING -> COMPLETED clean path
# ---------------------------------------------------------------------------


def test_working_to_completed_clean(worker: WorkerAgent) -> None:
    """
    A worker with no failure configured must transition:
      IDLE --activate()--> WORKING --advance() x _STEPS_TO_COMPLETE--> COMPLETED

    Verifies:
    - Initial state is IDLE.
    - activate("") moves worker to WORKING.
    - After exactly _STEPS_TO_COMPLETE advance() calls the worker is COMPLETED.
    - approved_output is None (supervisor has not yet issued APPROVE).
    """
    assert worker.state == WorkerState.IDLE

    worker.activate("")
    assert worker.state == WorkerState.WORKING

    # Advance until completion.  _STEPS_TO_COMPLETE is 3 (each advance()
    # increments steps_in_state before checking the threshold).
    for global_step in range(1, _STEPS_TO_COMPLETE + 1):
        worker.advance(global_step)

    assert worker.state == WorkerState.COMPLETED, (
        f"Expected COMPLETED after {_STEPS_TO_COMPLETE} advances, got {worker.state}"
    )
    assert worker.approved_output is None, (
        "approved_output must remain None until supervisor issues APPROVE"
    )


# ---------------------------------------------------------------------------
# Test 2: WORKING -> HALLUCINATING at correct inject_at_step
# ---------------------------------------------------------------------------


def test_working_to_hallucinating_at_correct_step(worker: WorkerAgent) -> None:
    """
    configure_failure(HALLUCINATING, inject_at_step=2) must cause the worker
    to enter HALLUCINATING exactly when steps_in_state reaches 2.

    Verifies:
    - After 1 advance the worker is still WORKING (failure not yet triggered).
    - After 2 advances the worker is HALLUCINATING.
    """
    worker.configure_failure(FailureMode.HALLUCINATING, inject_at_step=2)
    worker.activate("")

    # Step 1: failure threshold not reached
    worker.advance(1)
    assert worker.state == WorkerState.WORKING, (
        "Worker must remain WORKING before inject_at_step is reached"
    )

    # Step 2: failure fires (steps_in_state == 2 >= inject_at_step == 2)
    worker.advance(2)
    assert worker.state == WorkerState.HALLUCINATING, (
        "Worker must be HALLUCINATING once steps_in_state >= inject_at_step"
    )


# ---------------------------------------------------------------------------
# Test 3: STALLED detection via steps_unchanged counter
# ---------------------------------------------------------------------------


def test_stalled_steps_unchanged_increments(worker: WorkerAgent) -> None:
    """
    A STALLED worker emits the same fixed output every advance() call.
    steps_unchanged must increment on every step where the output is identical
    to the previous step.

    Verifies:
    - After failure fires, worker is STALLED.
    - steps_unchanged grows monotonically across consecutive advances.
    """
    worker.configure_failure(FailureMode.STALLED, inject_at_step=1)
    worker.activate("")

    # Advance once to fire the failure
    worker.advance(1)
    assert worker.state == WorkerState.STALLED

    # Record the unchanged counter just after entering STALLED.
    # On the first failure step the output just changed from "" to stalled_output,
    # so steps_unchanged may be 0.  Subsequent advances must increment it.
    before = worker.steps_unchanged

    worker.advance(2)
    after_one = worker.steps_unchanged
    assert after_one > before, (
        "steps_unchanged must increment when STALLED output repeats"
    )

    worker.advance(3)
    after_two = worker.steps_unchanged
    assert after_two > after_one, (
        "steps_unchanged must keep growing each step while output is identical"
    )


# ---------------------------------------------------------------------------
# Test 4: DECEPTIVE — OBSERVE returns clean snippet, DEEP_INSPECT reveals flaw
# ---------------------------------------------------------------------------


def test_deceptive_observe_vs_deep_inspect(
    worker: WorkerAgent, sample_task: Task
) -> None:
    """
    A DECEPTIVE worker must:
    - Return the clean deceptive_surface string via get_observe_snippet().
    - NOT include the deceptive_reveal string in get_observe_snippet().
    - Set deep_inspect_done = True after get_deep_inspect_output() is called.
    - Include deceptive_reveal in the string returned by get_deep_inspect_output().

    This is the core game mechanic: the supervisor cannot see the flaw without
    first spending 2 steps on DEEP_INSPECT.
    """
    worker.configure_failure(FailureMode.DECEPTIVE, inject_at_step=1)
    worker.activate("")
    worker.advance(1)  # fires failure → DECEPTIVE

    assert worker.state == WorkerState.DECEPTIVE

    # --- OBSERVE path: surface looks clean ---
    snippet = worker.get_observe_snippet()
    assert sample_task.deceptive_surface in snippet, (
        "get_observe_snippet() must return the clean deceptive surface"
    )
    assert sample_task.deceptive_reveal not in snippet, (
        "Hidden flaw must NOT appear in a plain OBSERVE snippet"
    )

    # --- DEEP_INSPECT path: flaw is revealed ---
    assert not worker.deep_inspect_done, "deep_inspect_done must start False"

    deep_out = worker.get_deep_inspect_output()

    assert worker.deep_inspect_done, (
        "get_deep_inspect_output() must set deep_inspect_done = True"
    )
    assert sample_task.deceptive_reveal in deep_out, (
        "deceptive_reveal must appear in the DEEP_INSPECT output"
    )


# ---------------------------------------------------------------------------
# Test 5: TERMINATE resets worker to WORKING
# ---------------------------------------------------------------------------


def test_terminate_resets_to_working(worker: WorkerAgent) -> None:
    """
    apply_terminate() on a HALLUCINATING worker must:
    - Transition the worker back to WORKING.
    - Clear output_history.
    - Clear failure_mode to NONE and inject_at_step to -1 so the worker
      gets a genuine retry without immediately re-failing.
    - Clear deep_inspect_done.

    Verifies the hard-reset semantics used after catching a bad worker.
    """
    worker.configure_failure(FailureMode.HALLUCINATING, inject_at_step=1)
    worker.activate("")
    worker.advance(1)
    assert worker.state == WorkerState.HALLUCINATING

    worker.apply_terminate()

    assert worker.state == WorkerState.WORKING, (
        "apply_terminate() must transition any failure state back to WORKING"
    )
    assert worker.failure_mode == FailureMode.NONE, (
        "apply_terminate() must clear failure_mode so the worker can retry cleanly"
    )
    assert worker.inject_at_step == -1, (
        "apply_terminate() must disable the failure injection"
    )
    assert len(worker.output_history) == 0, (
        "apply_terminate() must clear output_history"
    )
    assert not worker.deep_inspect_done, (
        "apply_terminate() must reset deep_inspect_done"
    )


# ---------------------------------------------------------------------------
# Test 6: REDIRECT resets worker to WORKING
# ---------------------------------------------------------------------------


def test_redirect_resets_to_working(worker: WorkerAgent) -> None:
    """
    apply_redirect(instruction) on a DRIFTED worker must:
    - Transition the worker back to WORKING.
    - Set self.redirected = True so get_visible_state_label() returns 'REDIRECTED'
      during the same timestep (transient — cleared on next advance()).
    - Clear failure_mode and inject_at_step so the worker retries cleanly.
    - Reset steps_unchanged.

    Verifies the soft-reset semantics: worker gets a corrective instruction
    and a genuine second chance without a full history wipe.
    """
    worker.configure_failure(FailureMode.DRIFTED, inject_at_step=1)
    worker.activate("")
    worker.advance(1)
    assert worker.state == WorkerState.DRIFTED

    instruction = "Re-focus on the original summarization task."
    worker.apply_redirect(instruction)

    assert worker.state == WorkerState.WORKING, (
        "apply_redirect() must transition any failure state back to WORKING"
    )
    assert worker.redirected is True, (
        "apply_redirect() must set redirected=True for the visible-state label"
    )
    assert worker.get_visible_state_label() == "REDIRECTED", (
        "get_visible_state_label() must return 'REDIRECTED' immediately after redirect"
    )
    assert worker.failure_mode == FailureMode.NONE, (
        "apply_redirect() must clear failure_mode"
    )
    assert worker.inject_at_step == -1, (
        "apply_redirect() must disable failure injection"
    )
    assert worker.steps_unchanged == 0, "apply_redirect() must reset steps_unchanged"
