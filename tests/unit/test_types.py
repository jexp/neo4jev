from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from neo4jev.types import (
    BeamState,
    FreeTextGoal,
    GoalSpec,
    NavCandidate,
    NavPath,
    NavResult,
    NavStep,
    PathIntentGoal,
    TargetNodeGoal,
    TerminationReason,
    assign_edge_keys,
)

TYPES_SOURCE = Path(__file__).resolve().parents[2] / "src/neo4jev/types.py"


def _make_candidate(edge_key: str = "e0") -> NavCandidate:
    return NavCandidate(
        edge_key=edge_key,
        rel_element_id="rel-1",
        rel_type="KNOWS",
        rel_props={"since": 2020},
        target_element_id="node-2",
        target_label="Person",
        target_props={"name": "Ada"},
        source_element_id="node-1",
        source_label="Person",
    )


def test_no_neo4j_or_typesafe_imports():
    tree = ast.parse(TYPES_SOURCE.read_text())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    assert not any(name.startswith("neo4j") or name.startswith("typesafe") for name in names)


def test_nav_candidate_construction():
    candidate = _make_candidate()
    assert candidate.edge_key == "e0"
    assert candidate.target_label == "Person"
    assert candidate.source_element_id == "node-1"


def test_nav_candidate_source_fields_default_for_hop_local_use():
    candidate = NavCandidate(
        edge_key="e0",
        rel_element_id="rel-1",
        rel_type="KNOWS",
        rel_props={},
        target_element_id="node-2",
        target_label="Person",
        target_props={},
    )
    assert candidate.source_element_id == ""
    assert candidate.source_label == ""


def test_assign_edge_keys_is_positional_and_does_not_collapse_same_rel_type():
    first = _make_candidate("")
    second = dataclasses.replace(
        first, rel_element_id="rel-2", target_element_id="node-3", target_props={"name": "Bob"}
    )

    keyed = assign_edge_keys([first, second])

    assert [candidate.edge_key for candidate in keyed] == ["e0", "e1"]
    assert keyed[1].target_element_id == "node-3"
    assert keyed[0].rel_type == keyed[1].rel_type


def test_assign_edge_keys_of_empty_list_returns_a_list():
    keyed = assign_edge_keys([])

    assert keyed == []
    assert isinstance(keyed, list)


def test_assign_edge_keys_leaves_the_input_candidates_untouched():
    original = _make_candidate("stale-key")

    keyed = assign_edge_keys([original])

    assert keyed[0].edge_key == "e0"
    assert original.edge_key == "stale-key"


def test_nav_step_keeps_all_candidates_considered_not_just_chosen():
    chosen = _make_candidate("e0")
    rejected = _make_candidate("e1")
    step = NavStep(
        node_id="node-1",
        chosen=[chosen],
        probabilities={"e0": 0.9, "e1": 0.1},
        log_prob=-0.10536,
        candidates=[chosen, rejected],
        noul=0.2,
    )
    assert [c.edge_key for c in step.candidates] == ["e0", "e1"]
    assert [c.edge_key for c in step.chosen] == ["e0"]
    assert step.noul == 0.2


def test_nav_step_noul_and_candidates_default():
    step = NavStep(node_id="node-1", chosen=[], probabilities={}, log_prob=0.0)
    assert step.candidates == []
    assert step.noul is None


def test_nav_path_and_result_hold_multiple_paths():
    step = NavStep(node_id="node-1", chosen=[_make_candidate()], probabilities={"e0": 1.0}, log_prob=0.0)
    path_a = NavPath(
        steps=[step], cumulative_log_prob=0.0, terminated_reason=TerminationReason.GOAL_REACHED
    )
    path_b = NavPath(
        steps=[step], cumulative_log_prob=-1.0, terminated_reason=TerminationReason.GOAL_REACHED
    )
    result = NavResult(start_element_id="node-0", paths=[path_a, path_b], neighborhood=[_make_candidate()])
    assert len(result.paths) == 2
    assert result.neighborhood[0].target_label == "Person"


def test_nav_result_neighborhood_defaults_empty():
    assert NavResult(start_element_id="node-0", paths=[]).neighborhood == []


def test_termination_reason_covers_all_four_cases():
    assert {r.value for r in TerminationReason} == {
        "goal_reached",
        "max_depth",
        "no_candidates",
        "budget_exhausted",
    }
    assert TerminationReason.MAX_DEPTH == "max_depth"
    assert TerminationReason.NO_CANDIDATES == "no_candidates"
    assert TerminationReason.BUDGET_EXHAUSTED == "budget_exhausted"


def test_beam_state_extend_does_not_mutate_parent():
    step = NavStep(node_id="node-1", chosen=[_make_candidate()], probabilities={"e0": 1.0}, log_prob=-0.5)
    parent = BeamState(
        node_id="node-0",
        path=[],
        cumulative_log_prob=0.0,
        visited=frozenset({"node-0"}),
        depth=0,
    )
    child = parent.extend(step, "node-1")

    assert parent.path == []
    assert parent.visited == frozenset({"node-0"})
    assert parent.depth == 0
    assert parent.cumulative_log_prob == 0.0

    assert child.node_id == "node-1"
    assert child.path == [step]
    assert child.cumulative_log_prob == -0.5
    assert child.visited == frozenset({"node-0", "node-1"})
    assert child.depth == 1


def test_frozen_dataclasses_reject_field_reassignment():
    candidate = _make_candidate()
    step = NavStep(node_id="node-1", chosen=[candidate], probabilities={"e0": 1.0}, log_prob=0.0)
    state = BeamState(
        node_id="node-0", path=[], cumulative_log_prob=0.0, visited=frozenset(), depth=0
    )
    for obj in (candidate, step, state):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(obj, dataclasses.fields(obj)[0].name, "mutated")


def test_goal_spec_discriminates_all_three_kinds():
    free_text = FreeTextGoal(goal="find something interesting")
    target_node = TargetNodeGoal(target_element_id="node-42")
    path_intent = PathIntentGoal(
        pattern_description="hop through related entities",
        top_n=5,
        stages=["find the company", "find its suppliers", "find articles mentioning them"],
    )

    goals: list[GoalSpec] = [free_text, target_node, path_intent]
    assert [g.kind for g in goals] == ["free_text", "target_node", "path_intent"]

    assert isinstance(free_text, FreeTextGoal)
    assert isinstance(target_node, TargetNodeGoal)
    assert isinstance(path_intent, PathIntentGoal)

    assert path_intent.top_n == 5
    assert path_intent.stages is not None
    assert len(path_intent.stages) == 3


def test_path_intent_stages_default_to_none_and_top_n_defaults_to_three():
    goal = PathIntentGoal(pattern_description="some pattern")
    assert goal.stages is None
    assert goal.top_n == 3
