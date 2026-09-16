from __future__ import annotations

import asyncio
import json
import math
from types import SimpleNamespace

import pytest
from typesafe_sdk import ChoiceAnswer, NoulAnswer, SystemOneResponse, Usage

from neo4jev import navigator as nav
from neo4jev.navigator import (
    CHOICE_QUESTION,
    NOUL_QUESTION,
    NavigatorConfig,
    NodeContext,
    decompose_stages,
    navigate,
    one_hop,
)
from neo4jev.types import (
    FreeTextGoal,
    NavCandidate,
    PathIntentGoal,
    TargetNodeGoal,
    TerminationReason,
)


def candidate(rel_id: str, target_id: str, *, rel_type: str = "REL", label: str = "Thing",
              rel_props: dict | None = None, target_props: dict | None = None) -> NavCandidate:
    return NavCandidate(
        edge_key=rel_id,
        rel_element_id=rel_id,
        rel_type=rel_type,
        rel_props=rel_props or {},
        target_element_id=target_id,
        target_label=label,
        target_props=target_props or {},
        source_element_id="n0",
        source_label="Thing",
    )


class FakeClient:
    """Minimal AsyncTypeSafeClient stand-in returning scripted per-node probabilities."""

    def __init__(self, weights: dict[str, dict[str, float]] | None = None,
                 noul: dict[str, float] | None = None) -> None:
        self.weights = weights or {}
        self.noul = noul or {}
        self.calls: list[SimpleNamespace] = []

    async def system_one(self, state, questions, **kwargs):
        self.calls.append(SimpleNamespace(state=state, questions=questions, kwargs=kwargs))
        node_id = state["current_node"]["element_id"]
        answers = {NOUL_QUESTION: NoulAnswer(noul=self.noul.get(node_id, 0.0))}
        if CHOICE_QUESTION in questions:
            keys = list(questions[CHOICE_QUESTION].criteria)
            scripted = self.weights.get(node_id)
            probabilities = (
                {key: 1 / len(keys) for key in keys}
                if scripted is None
                else {key: scripted.get(key, 0.0) for key in keys}
            )
            answers[CHOICE_QUESTION] = ChoiceAnswer(
                choice=max(probabilities, key=probabilities.get),
                confidence=1.0,
                probabilities=probabilities,
            )
        return SystemOneResponse(model="fake", usage=Usage(), answers=answers)


def fetcher(edges: dict[str, list[NavCandidate]]):
    calls: list[str] = []

    def fetch(node_id: str) -> list[NavCandidate]:
        calls.append(node_id)
        return list(edges.get(node_id, []))

    fetch.calls = calls
    return fetch


def run(coro):
    return asyncio.run(coro)


def test_one_hop_options_are_opaque_ids_mapped_back_to_candidates():
    # Same relationship type to three different targets: keying by rel type would collapse them.
    candidates = [
        candidate("r0", "n-a", rel_props={"weight": 1}, target_props={"name": "Ada"}),
        candidate("r1", "n-b", target_props={"name": "Bob"}),
        candidate("r2", "n-c", target_props={"name": "Cleo"}),
    ]
    client = FakeClient(weights={"n0": {"e0": 0.1, "e1": 0.7, "e2": 0.2}})

    hop = run(one_hop(client, NodeContext(element_id="n0", label="Thing"), candidates,
                      FreeTextGoal(goal="find Bob"), config=NavigatorConfig(top_k=1, cutoff=0.0)))

    criteria = client.calls[0].questions[CHOICE_QUESTION].criteria
    assert list(criteria) == ["e0", "e1", "e2"]
    assert "REL" not in criteria
    assert criteria["e1"] == {
        "relationship_type": "REL",
        "relationship_properties": {},
        "target_label": "Thing",
        "target_properties": {"name": "Bob"},
    }
    assert [c.edge_key for c in hop.candidates] == ["e0", "e1", "e2"]
    assert [c.target_element_id for c in hop.candidates] == ["n-a", "n-b", "n-c"]
    assert [c.target_element_id for c, _ in hop.chosen] == ["n-b"]
    assert hop.noul == 0.0


def test_one_hop_issues_one_call_carrying_both_questions_and_node_state():
    candidates = [candidate("r0", "n-a")]
    client = FakeClient()

    run(one_hop(client, NodeContext(element_id="n0", label="Company", properties={"name": "Apple"}),
                candidates, FreeTextGoal(goal="find a supplier")))

    assert len(client.calls) == 1
    call = client.calls[0]
    assert set(call.questions) == {CHOICE_QUESTION, NOUL_QUESTION}
    assert call.state["current_node"] == {
        "element_id": "n0",
        "label": "Company",
        "properties": {"name": "Apple"},
    }
    assert call.state["goal"]["mode"] == "free_text"
    assert call.state["goal"]["description"] == "find a supplier"
    assert call.state["hop_index"] == 0


def test_one_hop_asks_only_the_noul_when_a_node_has_no_relationships():
    client = FakeClient(noul={"n0": 0.7})
    hop = run(one_hop(client, NodeContext(element_id="n0"), [], FreeTextGoal(goal="x")))

    assert len(client.calls) == 1
    assert set(client.calls[0].questions) == {NOUL_QUESTION}
    assert hop.candidates == []
    assert hop.chosen == []
    assert hop.noul == 0.7
    assert hop.confidence is None  # no Choice was asked, so there is no Choice confidence


def test_one_hop_serializes_non_json_neo4j_property_values():
    class Temporal:
        def __str__(self) -> str:
            return "2026-09-16T00:00:00Z"

    candidates = [candidate("r0", "n-a", rel_props={"since": Temporal()})]
    client = FakeClient()

    run(one_hop(client, NodeContext(element_id="n0"), candidates, FreeTextGoal(goal="x")))

    criteria = client.calls[0].questions[CHOICE_QUESTION].criteria
    assert criteria["e0"]["relationship_properties"] == {"since": "2026-09-16T00:00:00Z"}


def _branches(probabilities: dict[str, float], **kwargs):
    by_key = {f"e{i}": candidate(f"r{i}", f"n{i}") for i in range(len(probabilities))}
    return nav._select_branches(probabilities, by_key, **kwargs)


def test_select_branches_applies_cutoff_then_top_k():
    selected = _branches(
        {"e0": 0.5, "e1": 0.3, "e2": 0.1, "e3": 0.05, "e4": 0.02},
        top_k=2,
        cutoff=0.05,
    )
    assert [(c.target_element_id, p) for c, p in selected] == [("n0", 0.5), ("n1", 0.3)]


def test_select_branches_limits_to_top_k():
    selected = _branches({"e0": 0.5, "e1": 0.5, "e2": 0.5}, top_k=2, cutoff=0.0)
    assert len(selected) == 2


def test_select_branches_falls_back_to_best_when_all_below_cutoff():
    selected = _branches({"e0": 0.01, "e1": 0.001}, top_k=3, cutoff=0.5)
    assert [(c.target_element_id, p) for c, p in selected] == [("n0", 0.01)]


def test_select_branches_ignores_probabilities_for_unknown_edge_keys():
    by_key = {"e0": candidate("r0", "n-a")}
    selected = nav._select_branches({"e0": 0.4, "e9": 0.9}, by_key, top_k=2, cutoff=0.0)
    assert [(c.target_element_id, p) for c, p in selected] == [("n-a", 0.4)]


def test_select_branches_handles_empty_probabilities():
    assert nav._select_branches({}, {}, top_k=2, cutoff=0.0) == []


def test_cumulative_log_probability_is_summed_over_hops():
    edges = {
        "n0": [candidate("r0", "n1")],
        "n1": [candidate("r1", "n2")],
        "n2": [],
    }
    client = FakeClient(weights={"n0": {"e0": 0.8}, "n1": {"e0": 0.25}})

    result = run(navigate(client, fetcher(edges), "n0", FreeTextGoal(goal="somewhere"),
                          config=NavigatorConfig(top_k=1, cutoff=0.0, max_depth=5)))

    assert len(result.paths) == 1
    path = result.paths[0]
    assert path.terminated_reason is TerminationReason.NO_CANDIDATES
    assert [step.node_id for step in path.steps] == ["n0", "n1"]
    assert [step.log_prob for step in path.steps] == [pytest.approx(math.log(0.8)),
                                                      pytest.approx(math.log(0.25))]
    assert path.cumulative_log_prob == pytest.approx(math.log(0.8) + math.log(0.25))


def test_free_text_returns_single_best_path_when_noul_crosses_threshold():
    edges = {"n0": [candidate("r0", "n1"), candidate("r1", "n2")], "n1": [], "n2": []}
    client = FakeClient(weights={"n0": {"e0": 0.9, "e1": 0.1}}, noul={"n1": 0.95})

    result = run(navigate(client, fetcher(edges), "n0", FreeTextGoal(goal="find a thing"),
                          config=NavigatorConfig(top_k=2, cutoff=0.05)))

    assert len(result.paths) == 1
    path = result.paths[0]
    assert path.terminated_reason is TerminationReason.GOAL_REACHED
    assert path.steps[-1].node_id == "n1"
    assert path.steps[-1].chosen == []
    assert path.steps[-1].noul == 0.95
    assert path.cumulative_log_prob == pytest.approx(math.log(0.9))


def test_target_node_ignores_a_high_noul_at_a_non_target_node():
    edges = {"n0": [candidate("r0", "n1")], "n1": [candidate("r1", "n2")], "n2": []}
    client = FakeClient(noul={"n1": 0.99})

    result = run(navigate(client, fetcher(edges), "n0", TargetNodeGoal(target_element_id="n2"),
                          config=NavigatorConfig(top_k=1, cutoff=0.0)))

    path = result.paths[0]
    assert path.terminated_reason is TerminationReason.GOAL_REACHED
    assert [step.node_id for step in path.steps] == ["n0", "n1"]
    assert path.steps[-1].chosen[0].target_element_id == "n2"
    assert path.steps[-1].noul == 0.99


def test_target_node_never_reached_falls_back_to_the_best_partial_path():
    edges = {"n0": [candidate("r0", "n1")], "n1": [candidate("r1", "n2")], "n2": []}

    result = run(navigate(FakeClient(), fetcher(edges), "n0",
                          TargetNodeGoal(target_element_id="unreachable"),
                          config=NavigatorConfig(top_k=1, cutoff=0.0, max_calls=1)))

    path = result.paths[0]
    assert path.terminated_reason is TerminationReason.BUDGET_EXHAUSTED
    assert path.steps[-1].chosen[0].target_element_id == "n1"


def test_target_node_terminates_on_exact_id_even_with_zero_noul():
    edges = {"n0": [candidate("r0", "n1")], "n1": [candidate("r1", "n2")], "n2": []}
    client = FakeClient(noul={})  # every hop answers "not reached"

    result = run(navigate(client, fetcher(edges), "n0", TargetNodeGoal(target_element_id="n2"),
                          config=NavigatorConfig(top_k=1, cutoff=0.0)))

    assert len(result.paths) == 1
    path = result.paths[0]
    assert path.terminated_reason is TerminationReason.GOAL_REACHED
    assert path.steps[-1].chosen[0].target_element_id == "n2"


def test_target_node_goal_at_start_node_terminates_without_any_call():
    client = FakeClient()
    result = run(navigate(client, fetcher({}), "n0", TargetNodeGoal(target_element_id="n0")))
    assert result.paths[0].terminated_reason is TerminationReason.GOAL_REACHED
    assert result.paths[0].steps == []
    assert client.calls == []


def test_visited_node_cycle_guard_stops_revisiting_a_node():
    edges = {"n0": [candidate("r0", "n1")], "n1": [candidate("r1", "n0")]}
    fetch = fetcher(edges)
    client = FakeClient()

    result = run(navigate(client, fetch, "n0", FreeTextGoal(goal="anywhere"),
                          config=NavigatorConfig(top_k=1, cutoff=0.0, max_depth=5)))

    assert fetch.calls == ["n0", "n1"]
    assert len(client.calls) == 2
    path = result.paths[0]
    assert path.terminated_reason is TerminationReason.NO_CANDIDATES
    assert [step.node_id for step in path.steps] == ["n0"]
    visited = [step.node_id for step in path.steps] + [step.chosen[0].target_element_id for step in path.steps]
    assert len(visited) == len(set(visited))


def test_max_depth_guard_stops_expansion():
    edges = {
        "n0": [candidate("r0", "n1")],
        "n1": [candidate("r1", "n2")],
        "n2": [candidate("r2", "n3")],
        "n3": [],
    }
    result = run(navigate(FakeClient(), fetcher(edges), "n0", FreeTextGoal(goal="deep"),
                          config=NavigatorConfig(top_k=1, cutoff=0.0, max_depth=2)))

    path = result.paths[0]
    assert path.terminated_reason is TerminationReason.MAX_DEPTH
    assert len(path.steps) == 2


def test_max_calls_budget_guard_stops_expansion():
    edges = {"n0": [candidate("r0", "n1")], "n1": [candidate("r1", "n2")], "n2": []}
    client = FakeClient()

    result = run(navigate(client, fetcher(edges), "n0", FreeTextGoal(goal="deep"),
                          config=NavigatorConfig(top_k=1, cutoff=0.0, max_depth=10, max_calls=1)))

    assert len(client.calls) == 1
    path = result.paths[0]
    assert path.terminated_reason is TerminationReason.BUDGET_EXHAUSTED
    assert len(path.steps) == 1


def test_dead_end_terminates_with_no_candidates():
    client = FakeClient()
    result = run(navigate(client, fetcher({"n0": []}), "n0", FreeTextGoal(goal="nowhere")))

    assert result.paths[0].terminated_reason is TerminationReason.NO_CANDIDATES
    assert result.paths[0].steps == []
    assert len(client.calls) == 1
    assert set(client.calls[0].questions) == {NOUL_QUESTION}


def test_leaf_node_can_still_satisfy_the_goal():
    edges = {"n0": [candidate("r0", "n1")], "n1": []}
    client = FakeClient(noul={"n1": 0.9})

    result = run(navigate(client, fetcher(edges), "n0", FreeTextGoal(goal="find a leaf")))

    assert result.paths[0].terminated_reason is TerminationReason.GOAL_REACHED
    assert [step.node_id for step in result.paths[0].steps] == ["n0", "n1"]


def test_neighborhood_collects_candidates_considered_deduped():
    shared = candidate("r-shared", "n2")
    edges = {"n0": [candidate("r0", "n1"), shared], "n1": [shared], "n2": []}
    result = run(navigate(FakeClient(), fetcher(edges), "n0", FreeTextGoal(goal="x"),
                          config=NavigatorConfig(top_k=1, cutoff=0.0)))

    rel_ids = [c.rel_element_id for c in result.neighborhood]
    assert rel_ids == ["r0", "r-shared"]


def test_path_intent_returns_multiple_distinct_paths_ranked_best_first():
    edges = {"n0": [candidate("r0", "n1"), candidate("r1", "n2")], "n1": [], "n2": []}
    client = FakeClient(weights={"n0": {"e0": 0.65, "e1": 0.35}})

    result = run(navigate(client, fetcher(edges), "n0",
                          PathIntentGoal(pattern_description="hop to a -> hop to b", top_n=3),
                          config=NavigatorConfig(top_k=2, cutoff=0.0)))

    assert len(result.paths) == 2
    assert [path.steps[0].chosen[0].target_element_id for path in result.paths] == ["n1", "n2"]
    assert result.paths[0].cumulative_log_prob > result.paths[1].cumulative_log_prob


def test_path_intent_respects_top_n():
    edges = {"n0": [candidate("r0", "n1"), candidate("r1", "n2"), candidate("r2", "n3")],
             "n1": [], "n2": [], "n3": []}
    client = FakeClient(weights={"n0": {"e0": 0.5, "e1": 0.3, "e2": 0.2}})

    result = run(navigate(client, fetcher(edges), "n0",
                          PathIntentGoal(pattern_description="a -> b", top_n=2),
                          config=NavigatorConfig(top_k=3, cutoff=0.0)))

    assert len(result.paths) == 2


def test_path_intent_deduplicates_paths_with_the_same_node_sequence():
    edges = {"n0": [candidate("r0", "n1"), candidate("r1", "n1")], "n1": []}
    client = FakeClient(weights={"n0": {"e0": 0.6, "e1": 0.4}})

    result = run(navigate(client, fetcher(edges), "n0",
                          PathIntentGoal(pattern_description="a -> b", top_n=3),
                          config=NavigatorConfig(top_k=2, cutoff=0.0)))

    assert len(result.paths) == 1


def test_path_intent_beam_width_limits_surviving_branches():
    edges = {
        "n0": [candidate("r0", "n1"), candidate("r1", "n2"), candidate("r2", "n3")],
        "n1": [candidate("r10", "n1a")],
        "n2": [candidate("r20", "n2a")],
        "n3": [candidate("r30", "n3a")],
        "n1a": [], "n2a": [], "n3a": [],
    }
    weights = {"n0": {"e0": 0.5, "e1": 0.3, "e2": 0.2}}
    goal = PathIntentGoal(pattern_description="a -> b", top_n=5)

    wide = run(navigate(FakeClient(weights), fetcher(edges), "n0", goal,
                        config=NavigatorConfig(top_k=3, cutoff=0.0, beam_width=3, max_depth=4)))
    narrow = run(navigate(FakeClient(weights), fetcher(edges), "n0", goal,
                          config=NavigatorConfig(top_k=3, cutoff=0.0, beam_width=1, max_depth=4)))

    assert len(wide.paths) == 3
    assert len(narrow.paths) == 1


def test_decompose_stages_splits_on_common_separators():
    assert decompose_stages("find the company -> find its suppliers then find the articles") == [
        "find the company",
        "find its suppliers",
        "find the articles",
    ]
    assert decompose_stages("a, b; c\nd => e") == ["a", "b", "c", "d", "e"]
    assert decompose_stages("single stage description") == ["single stage description"]


def test_path_intent_biases_each_hop_toward_the_current_stage():
    stages = ["find the company", "find its suppliers", "find the articles"]
    goal = PathIntentGoal(pattern_description="ignored", stages=stages)
    candidates = [candidate("r0", "n1")]
    client = FakeClient()

    run(one_hop(client, NodeContext(element_id="n0"), candidates, goal, hop_index=1))
    call = client.calls[0]

    assert stages[1] in call.questions[CHOICE_QUESTION].instructions
    assert call.state["goal"]["current_stage"] == stages[1]
    assert call.state["goal"]["current_stage_index"] == 1
    assert stages[-1] in call.questions[NOUL_QUESTION].criteria["true"]
    assert call.questions[NOUL_QUESTION].criteria["true"] != stages[1]


def test_path_intent_stage_pointer_clamps_at_the_final_stage():
    stages = ["one", "two"]
    goal = PathIntentGoal(pattern_description="ignored", stages=stages)
    client = FakeClient()

    run(one_hop(client, NodeContext(element_id="n0"), [candidate("r0", "n1")], goal, hop_index=7))

    assert client.calls[0].state["goal"]["current_stage"] == "two"


def test_goal_reached_uses_configured_threshold():
    edges = {"n0": [candidate("r0", "n1")], "n1": []}
    client = FakeClient(noul={"n0": 0.4})

    below = run(navigate(client, fetcher(edges), "n0", FreeTextGoal(goal="x"),
                         config=NavigatorConfig(goal_threshold=0.5, top_k=1, cutoff=0.0)))
    above = run(navigate(FakeClient(noul={"n0": 0.4}), fetcher(edges), "n0", FreeTextGoal(goal="x"),
                         config=NavigatorConfig(goal_threshold=0.3, top_k=1, cutoff=0.0)))

    assert below.paths[0].terminated_reason is TerminationReason.NO_CANDIDATES
    assert above.paths[0].terminated_reason is TerminationReason.GOAL_REACHED
    assert [step.node_id for step in above.paths[0].steps] == ["n0"]


def test_one_hop_caps_choice_options_at_the_api_limit():
    candidates = [candidate(f"r{i}", f"n{i}") for i in range(300)]
    client = FakeClient()

    hop = run(one_hop(client, NodeContext(element_id="n0"), candidates, FreeTextGoal(goal="x")))

    assert len(client.calls[0].questions[CHOICE_QUESTION].criteria) == nav.MAX_CHOICE_OPTIONS
    assert len(hop.candidates) == nav.MAX_CHOICE_OPTIONS


def test_one_hop_serializes_nested_property_containers():
    class Temporal:
        def __str__(self) -> str:
            return "2020"

    candidates = [candidate("r0", "n1", target_props={"a": {"b": [1, {"c": Temporal()}]}})]
    client = FakeClient()

    run(one_hop(client, NodeContext(element_id="n0"), candidates, FreeTextGoal(goal="x")))

    criteria = client.calls[0].questions[CHOICE_QUESTION].criteria
    assert criteria["e0"]["target_properties"] == {"a": {"b": [1, {"c": "2020"}]}}


def test_navigator_config_rejects_invalid_values():
    for field, value in (("beam_width", 0), ("max_depth", 0), ("max_calls", -1), ("top_k", 0),
                         ("cutoff", 1.5), ("goal_threshold", -0.1)):
        with pytest.raises(ValueError):
            NavigatorConfig(**{field: value})


def test_hop_passes_model_override_to_the_client():
    client = FakeClient()
    run(one_hop(client, NodeContext(element_id="n0"), [candidate("r0", "n1")],
                FreeTextGoal(goal="x"), config=NavigatorConfig(model="some-model")))

    assert client.calls[0].kwargs["model"] == "some-model"


def test_hop_round_trips_through_the_sdk_wire_format():
    httpx2 = pytest.importorskip("httpx2")
    from typesafe_sdk import AsyncTypeSafeClient

    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx2.Response(200, json={
            "model": "test-model",
            "usage": {},
            "answers": {
                CHOICE_QUESTION: {"type": "choice", "choice": "e1", "confidence": 0.8,
                                  "probabilities": {"e0": 0.25, "e1": 0.75}},
                NOUL_QUESTION: {"type": "noul", "noul": 0.1},
            },
        })

    client = AsyncTypeSafeClient(
        api_key="test-key",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    hop = run(one_hop(client, NodeContext(element_id="n0", label="Company"),
                      [candidate("r0", "n1"), candidate("r1", "n2")],
                      FreeTextGoal(goal="find a partner")))

    body = requests[0]
    assert body["state"]["current_node"]["element_id"] == "n0"
    assert set(body["questions"]) == {CHOICE_QUESTION, NOUL_QUESTION}
    assert set(body["questions"][CHOICE_QUESTION]["criteria"]) == {"e0", "e1"}
    assert hop.probabilities == {"e0": 0.25, "e1": 0.75}
    assert [(c.target_element_id, p) for c, p in hop.chosen] == [("n2", 0.75), ("n1", 0.25)]
    assert hop.noul == 0.1
    assert hop.confidence == 0.8
