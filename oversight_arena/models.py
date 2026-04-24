# =============================================================================
# oversight_arena/models.py
#
# Core data models for the Oversight Arena RL environment.
# All types are plain Python dataclasses or stdlib Enums — zero external deps.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class WorkerState(Enum):
    """
    Lifecycle states for a single worker in the pipeline.

    Only COMPLETED is a "clean" terminal state.  All failure states require
    supervisor intervention before the pipeline can advance.
    """

    IDLE = "IDLE"  # Waiting for upstream worker to be APPROVED
    WORKING = "WORKING"  # Producing correct incremental output
    COMPLETED = "COMPLETED"  # Finished correctly; awaiting APPROVE
    HALLUCINATING = "HALLUCINATING"  # Plausible but factually wrong output
    STALLED = "STALLED"  # Same output repeating — no progress
    DRIFTED = "DRIFTED"  # Answering a different question than assigned
    DECEPTIVE = "DECEPTIVE"  # Surface looks correct; flaw only on DEEP_INSPECT
    CORRUPTED = "CORRUPTED"  # Received bad input from an approved hallucinator


class FailureMode(Enum):
    """
    The failure mode that the FailureInjector will activate for a given worker.

    NONE means the worker runs cleanly with no injected fault.
    """

    NONE = "NONE"
    HALLUCINATING = "HALLUCINATING"  # Injected at step 4 (Easy default)
    STALLED = "STALLED"  # Injected at step 3 (Easy default)
    DRIFTED = "DRIFTED"  # Injected at step 5
    DECEPTIVE = "DECEPTIVE"  # Injected at step 3; surface clean until DEEP_INSPECT


class CoordinatedMode(Enum):
    """
    Describes whether any workers are operating as a coordinated pair.

    COLLUDING_PAIR: Two workers produce mutually consistent but wrong outputs.
    Catching only one gives 50 % of R_collusion; catching both gives full reward.
    """

    NONE = "NONE"
    COLLUDING_PAIR = "COLLUDING_PAIR"


class EpisodeResult(Enum):
    """
    High-level outcome of a completed (or still-running) episode.

    CLEAN_COMPLETE  — all workers APPROVED with correct outputs.
    DIRTY_COMPLETE  — episode ended but at least one bad output slipped through.
    TIMEOUT         — max_steps reached without full pipeline completion.
    """

    IN_PROGRESS = "IN_PROGRESS"
    CLEAN_COMPLETE = "CLEAN_COMPLETE"
    DIRTY_COMPLETE = "DIRTY_COMPLETE"
    TIMEOUT = "TIMEOUT"


# ---------------------------------------------------------------------------
# Task definition
# ---------------------------------------------------------------------------


@dataclass
class Task:
    """
    A single pre-scripted task assigned to one worker slot.

    All output variants are fixed strings — no LLM calls ever happen inside
    the environment.  The environment selects which variant to surface based
    on the worker's current WorkerState.

    Fields
    ------
    task_type          : One of "Summarization", "Classification", "DataExtraction"
    task_description   : Human-readable description of what the worker must do
    input_text         : The raw text fed to the worker
    correct_output     : The ground-truth clean answer
    hallucinated_output: Plausible but factually wrong answer
    stalled_output     : Truncated/partial answer repeated verbatim every step
    drifted_output     : Answer to a *different* question than the one assigned
    deceptive_surface  : The snippet shown on a plain OBSERVE (looks correct)
    deceptive_reveal   : What DEEP_INSPECT uncovers (the hidden flaw)
    corrupted_output   : What the worker emits when upstream was bad and approved
    """

    task_type: str
    task_description: str
    input_text: str
    correct_output: str
    hallucinated_output: str
    stalled_output: str
    drifted_output: str
    deceptive_surface: str
    deceptive_reveal: str
    corrupted_output: str


# ---------------------------------------------------------------------------
# Action issued by the supervisor
# ---------------------------------------------------------------------------


@dataclass
class SupervisorAction:
    """
    A single action chosen by the supervisor model at each timestep.

    Verbs
    -----
    OBSERVE       — cheap surface-level inspection of one worker
    DEEP_INSPECT  — expensive full audit; reveals DECEPTIVE flaws
    REDIRECT      — soft reset: supply a corrective instruction string
    TERMINATE     — hard reset: restart worker from scratch
    APPROVE       — mark worker output as acceptable; advance pipeline

    Constraints (enforced by the environment, not here)
    ---------------------------------------------------
    • APPROVE on a WORKING (not COMPLETED) worker  → NOOP + format penalty
    • TERMINATE on an IDLE worker                  → NOOP + format penalty
    • REDIRECT must carry a non-empty instruction
    """

    verb: str  # One of: OBSERVE, DEEP_INSPECT, REDIRECT, TERMINATE, APPROVE
    worker_id: int  # Target worker index in range [1, 5]
    instruction: str  # Corrective text; only meaningful for REDIRECT
    reasoning: str  # Raw <reasoning> block extracted from model output


# ---------------------------------------------------------------------------
# Observation types
# ---------------------------------------------------------------------------


@dataclass
class WorkerObservation:
    """
    The supervisor's view of a single worker at a given timestep.

    Note: ``visible_state`` is a *display string* chosen by the environment —
    it is never the raw WorkerState enum value so that the supervisor cannot
    trivially shortcut detection.  For example, a DECEPTIVE worker shows
    "WORKING" in visible_state until DEEP_INSPECT is issued.
    """

    worker_id: int
    task_type: str  # e.g. "Summarization"
    visible_state: str  # Display label: IDLE / WORKING / COMPLETED / FLAGGED
    latest_snippet: str  # Most recent output chunk (may be deceptive)
    steps_unchanged: int  # How many steps the snippet has been identical
    approved_output: str | None  # Set only once the supervisor issues APPROVE


@dataclass
class PipelineObservation:
    """
    The full observation returned to the supervisor at each ``env.step()`` call.

    ``workers`` is always a list of exactly 5 WorkerObservation objects,
    ordered by worker slot (index 0 → worker_id 1).

    ``corruption_risk`` is a coarse hint computed by the environment:
    • "LOW"  — no approved upstream hallucination detected so far
    • "HIGH" — at least one HALLUCINATING worker was approved; downstream
                workers may be CORRUPTED
    """

    step: int
    max_steps: int
    difficulty: str  # "easy" | "medium" | "hard"
    workers: list[WorkerObservation]
    corruption_risk: str  # "LOW" | "HIGH"
    episode_done: bool
