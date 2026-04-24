# =============================================================================
# tests/test_environment.py
#
# Integration tests for OversightArenaEnvironment.
#
# PRIORITY ORDER: Fix Section 22 issues -> pass Section 23 tests ->
# deploy to HF Space -> run baseline eval -> start training.
# Do not start training before the environment passes all tests.
#
# Run with: pytest tests/ -v
# =============================================================================

from __future__ import annotations

import random

import pytest

from oversight_arena.environment import OversightArenaEnvironment

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def env() -> OversightArenaEnvironment:
    """Fresh environment reset to EASY difficulty with a fixed seed."""
    e = OversightArenaEnvironment()
    e.reset(difficulty="easy", seed=42)
    return e


# ---------------------------------------------------------------------------
# Test 1: reset() returns a valid observation string for all 3 difficulties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_reset_returns_valid_observation(difficulty: str) -> None:
    """
    reset() must return an Observation whose metadata contains a non-empty
    'pipeline_text' string for every supported difficulty.
    """
    e = OversightArenaEnvironment()
    obs = e.reset(difficulty=difficulty, seed=0)

    assert obs is not None, "reset() must return an Observation"
    assert not obs.done, "Episode must not be done immediately after reset"
    assert obs.reward == 0.0, "reset() must return 0 reward"

    pipeline_text = obs.metadata.get("pipeline_text", "")
    assert isinstance(pipeline_text, str), "pipeline_text must be a string"
    assert len(pipeline_text) > 0, "pipeline_text must not be empty"
    # Sanity: the header banner should be present
    assert "Oversight Arena" in pipeline_text, (
        "pipeline_text should contain the environment header"
    )


# ---------------------------------------------------------------------------
# Test 2: Same seed produces identical episode
# ---------------------------------------------------------------------------


def test_same_seed_produces_identical_episode() -> None:
    """
    Two environments initialised with the same difficulty and seed must produce
    byte-identical initial observations and identical failure plans.
    """
    e1 = OversightArenaEnvironment()
    e2 = OversightArenaEnvironment()

    obs1 = e1.reset(difficulty="easy", seed=1234)
    obs2 = e2.reset(difficulty="easy", seed=1234)

    assert obs1.metadata["pipeline_text"] == obs2.metadata["pipeline_text"], (
        "Same seed must yield identical initial observations"
    )

    state1 = e1.state_dict
    state2 = e2.state_dict

    assert state1["failure_plan"] == state2["failure_plan"], (
        "Same seed must yield identical failure plans"
    )
    assert state1["seed"] == state2["seed"], "Stored seeds must match"


# ---------------------------------------------------------------------------
# Test 3: APPROVE on a WORKING worker returns format penalty
# ---------------------------------------------------------------------------


def test_approve_on_working_returns_format_penalty(
    env: OversightArenaEnvironment,
) -> None:
    """
    APPROVE on a WORKING (not yet completed) worker is an invalid action.
    The environment must return a format penalty of -0.1 and not advance
    the pipeline.
    """
    # After reset(), W1 is WORKING (activated); W2-W5 are IDLE.
    state = env.state_dict
    w1 = next(w for w in state["workers"] if w["worker_id"] == 1)
    assert w1["state"] == "WORKING", "Precondition: W1 must be WORKING after reset"

    obs = env.step("APPROVE 1")

    assert obs.reward == pytest.approx(-0.1), (
        "APPROVE on WORKING worker must yield -0.1 format penalty"
    )
    assert not obs.done, "Episode must not end from a single invalid action"


# ---------------------------------------------------------------------------
# Test 4: TERMINATE on an IDLE worker returns format penalty
# ---------------------------------------------------------------------------


def test_terminate_on_idle_returns_format_penalty(
    env: OversightArenaEnvironment,
) -> None:
    """
    TERMINATE on an IDLE worker (one that has not been activated yet) is an
    invalid action.  Must return a -0.1 format penalty.
    """
    # After reset(), W2 is IDLE.
    state = env.state_dict
    w2 = next(w for w in state["workers"] if w["worker_id"] == 2)
    assert w2["state"] == "IDLE", "Precondition: W2 must be IDLE after reset"

    obs = env.step("TERMINATE 2")

    assert obs.reward == pytest.approx(-0.1), (
        "TERMINATE on IDLE worker must yield -0.1 format penalty"
    )
    assert not obs.done, "Episode must not end from a single invalid action"


# ---------------------------------------------------------------------------
# Test 5: Episode terminates at step 25 (TIMEOUT)
# ---------------------------------------------------------------------------


def test_episode_terminates_at_step_25() -> None:
    """
    Repeatedly issuing OBSERVE actions must eventually trigger a TIMEOUT when
    the step budget (25) is exhausted.  The episode_result must be 'TIMEOUT'
    and the step counter must be >= max_steps.
    """
    e = OversightArenaEnvironment()
    e.reset(difficulty="easy", seed=7)

    done = False
    calls = 0
    safety_limit = 200  # prevents infinite loop if there's a bug

    while not done and calls < safety_limit:
        obs = e.step("OBSERVE 1")
        done = obs.done
        calls += 1

    assert done, f"Episode should have ended within {safety_limit} step() calls"

    state = e.state_dict
    assert state["step"] >= state["max_steps"], (
        f"step counter ({state['step']}) must be >= max_steps ({state['max_steps']})"
    )
    assert state["episode_result"] == "TIMEOUT", (
        f"Expected TIMEOUT, got {state['episode_result']!r}"
    )


# ---------------------------------------------------------------------------
# Test 6: Random agent achieves non-zero reward on EASY (10 episodes)
# ---------------------------------------------------------------------------


def test_random_agent_nonzero_reward_easy() -> None:
    """
    A random agent submitting random valid action strings must achieve at
    least one non-zero step reward across 10 EASY episodes.  This is a
    smoke test that verifies the reward pipeline is wired end-to-end.

    Non-zero here includes negative rewards (format penalties), which a
    random agent will routinely incur.  The test guards against a silent
    bug where every call returns 0.0 regardless of action.
    """
    rng = random.Random(99)
    verbs = ["OBSERVE", "DEEP_INSPECT", "APPROVE", "TERMINATE"]
    redirect_verbs = ["REDIRECT"]

    all_rewards: list[float] = []

    for seed in range(10):
        e = OversightArenaEnvironment()
        e.reset(difficulty="easy", seed=seed)

        done = False
        episode_reward = 0.0
        steps = 0

        while not done and steps < 50:
            wid = rng.randint(1, 5)
            verb = rng.choice(verbs + redirect_verbs)

            if verb == "REDIRECT":
                action = f"REDIRECT {wid} Please re-focus on the assigned task."
            else:
                action = f"{verb} {wid}"

            obs = e.step(action)
            episode_reward += obs.reward
            done = obs.done
            steps += 1

        all_rewards.append(episode_reward)

    # At minimum, format penalties (-0.1 each) ensure non-zero rewards.
    assert any(r != 0.0 for r in all_rewards), (
        "Expected at least one episode with non-zero reward; "
        f"got all_rewards={all_rewards}"
    )


# ---------------------------------------------------------------------------
# Test 7: DEEP_INSPECT costs 2 steps
# ---------------------------------------------------------------------------


def test_deep_inspect_costs_two_steps() -> None:
    """
    A DEEP_INSPECT action must advance the episode step counter by 2, not 1.
    All other actions advance it by 1.
    """
    e = OversightArenaEnvironment()
    e.reset(difficulty="easy", seed=5)

    # Baseline: OBSERVE costs 1 step
    step_before = e.state_dict["step"]
    e.step("OBSERVE 1")
    step_after = e.state_dict["step"]
    assert step_after - step_before == 1, (
        f"OBSERVE should cost 1 step, got {step_after - step_before}"
    )

    # DEEP_INSPECT costs 2 steps
    step_before = e.state_dict["step"]
    e.step("DEEP_INSPECT 1")
    step_after = e.state_dict["step"]
    assert step_after - step_before == 2, (
        f"DEEP_INSPECT should cost 2 steps, got {step_after - step_before}"
    )


# ---------------------------------------------------------------------------
# Test 8: Raw string step() path works end-to-end
# ---------------------------------------------------------------------------


def test_raw_string_step_accepted(env: OversightArenaEnvironment) -> None:
    """
    step() must accept plain action strings (not just CallToolAction objects).
    All five verbs must be parseable and return a valid Observation.
    """
    # OBSERVE — always valid on any active worker
    obs = env.step("OBSERVE 1")
    assert obs is not None
    assert "pipeline_text" in obs.metadata

    # DEEP_INSPECT — valid on any active worker
    obs = env.step("DEEP_INSPECT 1")
    assert obs is not None
    assert "pipeline_text" in obs.metadata


# ---------------------------------------------------------------------------
# Test 9: Malformed action string triggers format penalty
# ---------------------------------------------------------------------------


def test_malformed_action_triggers_penalty(env: OversightArenaEnvironment) -> None:
    """
    A raw string that cannot be parsed must trigger a -0.1 format penalty
    rather than raising an exception.
    """
    obs = env.step("this is not a valid action")
    assert obs.reward == pytest.approx(-0.1), (
        f"Malformed action should yield -0.1 penalty, got {obs.reward}"
    )
    assert "error" in obs.metadata, (
        "Malformed action response should include an 'error' key in metadata"
    )


# ---------------------------------------------------------------------------
# Test 10: Reasoning block extracts Mercor bonus
# ---------------------------------------------------------------------------


def test_reasoning_block_grants_mercor_bonus(env: OversightArenaEnvironment) -> None:
    """
    A raw action string that includes a valid <reasoning>…</reasoning> block
    and a correct action should accumulate a positive reward_mercor value in
    the reward breakdown.

    W1 is WORKING after reset; OBSERVE is the oracle-correct action for a
    WORKING worker, so action_was_correct=True and Mercor bonus fires.
    """
    action = (
        "<reasoning>"
        "Worker 1 appears to be in an early stage of progress. "
        "I should observe its output before taking any intervention."
        "</reasoning>\n"
        "OBSERVE 1"
    )
    obs = env.step(action)

    breakdown = obs.metadata.get("reward_breakdown", {})
    mercor = breakdown.get("reward_mercor", 0.0)

    assert mercor > 0.0, (
        f"Expected positive reward_mercor for correct action with reasoning, "
        f"got {mercor}. Full breakdown: {breakdown}"
    )
