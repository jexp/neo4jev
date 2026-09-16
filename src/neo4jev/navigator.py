"""Beam-search graph navigation driven by TypeSafe ``system_one`` Choice + Noul questions.

Each hop issues exactly one ``system_one`` call containing two questions: a ``Choice`` over the
outgoing relationships of the current node and a ``Noul`` asking whether the goal has been reached.
Top-k/cutoff selection over the returned probabilities turns that into a beam search scored by the
sum of log-probabilities. A node with no outgoing relationships still gets its single call, asking
the ``Noul`` alone (an empty ``Choice`` has nothing to choose between).
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from typesafe_sdk import Choice, Noul, SystemOneResponse
from typesafe_sdk import Question as TypeSafeQuestion

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
)

CHOICE_QUESTION = "next_edge"
NOUL_QUESTION = "goal_reached"

# Hard API limit on a Choice question's criteria map (REQ-NF-005).
MAX_CHOICE_OPTIONS = 255

_LOG_FLOOR = 1e-12
_STAGE_DELIMITER = "\x00"
_STAGE_SEPARATORS = ("->", "=>", " then ", ";", "\n", ",")

# Long text (article bodies, descriptions) dominates the request payload without helping the
# choice; vector/list properties are dropped outright.
MAX_PROPERTY_CHARS = 200
_DROP = object()


@dataclass(frozen=True)
class NavigatorConfig:
    max_depth: int = 4
    max_calls: int = 24
    top_k: int = 2
    cutoff: float = 0.05
    beam_width: int = 4
    goal_threshold: float = 0.5
    model: str | None = None

    def __post_init__(self) -> None:
        for name in ("max_depth", "max_calls", "top_k", "beam_width"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)}")
        for name in ("cutoff", "goal_threshold"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1, got {getattr(self, name)}")


@dataclass(frozen=True)
class NodeContext:
    """The data a hop is reasoned about: the node the search currently sits on."""

    element_id: str
    label: str = ""
    properties: Mapping[str, Any] = field(default_factory=dict)


class SystemOneClient(Protocol):
    async def system_one(
        self, state: Any, questions: Mapping[str, TypeSafeQuestion], **kwargs: Any
    ) -> SystemOneResponse: ...


# Injected rather than imported from neo4j_access so navigator stays unit-testable without a driver.
# The fetcher is responsible for capping its result (neo4j_access caps per type and in total);
# one_hop enforces the 255-option API limit only as a backstop.
CandidateFetcher = Callable[[str], Sequence[NavCandidate]]


@dataclass(frozen=True)
class HopResult:
    """One ``system_one`` round-trip at one node, plus the branches taken from it."""

    node_id: str
    candidates: list[NavCandidate]
    probabilities: dict[str, float]
    noul: float | None
    chosen: list[tuple[NavCandidate, float]]
    confidence: float | None = None


class _Budget:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = 0

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    def spend(self) -> None:
        self.used += 1


def _jsonable(value: Any) -> Any:
    # Neo4j hands back datetime/temporal/spatial objects that the request encoder rejects.
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    return str(value)


def _prompt_value(value: Any) -> Any:
    """A property value as the model should see it, or ``_DROP`` to omit the property."""
    if isinstance(value, (list, tuple, set, frozenset)):
        # Embeddings and other collections say nothing about the decision and cost input tokens.
        return _DROP
    if isinstance(value, Mapping):
        nested: dict[str, Any] = {}
        for key, item in value.items():
            trimmed = _prompt_value(item)
            if trimmed is not _DROP:
                nested[str(key)] = trimmed
        return nested
    if isinstance(value, str) and len(value) > MAX_PROPERTY_CHARS:
        return value[:MAX_PROPERTY_CHARS] + "..."
    return _jsonable(value)


def _prompt_props(props: Mapping[str, Any] | None) -> dict[str, Any]:
    """Properties with vectors/lists dropped and long text truncated, for request payloads."""
    reduced: dict[str, Any] = {}
    for key, value in (props or {}).items():
        trimmed = _prompt_value(value)
        if trimmed is not _DROP:
            reduced[str(key)] = trimmed
    return reduced


def _candidate_criteria(candidate: NavCandidate) -> dict[str, Any]:
    return {
        "relationship_type": candidate.rel_type,
        "relationship_properties": _prompt_props(candidate.rel_props),
        "target_label": candidate.target_label,
        "target_properties": _prompt_props(candidate.target_props),
    }


def decompose_stages(description: str) -> list[str]:
    """Split a path-intent pattern description into ordered stage-guidance strings.

    Always returns at least one stage, so downstream code can index the final stage safely.
    """
    normalized = description
    for separator in _STAGE_SEPARATORS:
        normalized = normalized.replace(separator, _STAGE_DELIMITER)
    stages = [stage.strip() for stage in normalized.split(_STAGE_DELIMITER)]
    return [stage for stage in stages if stage] or [description.strip()]


def _stages_for(goal: GoalSpec) -> list[str] | None:
    if not isinstance(goal, PathIntentGoal):
        return None
    if goal.stages:
        return list(goal.stages)
    return decompose_stages(goal.pattern_description)


def _stage_index(stages: Sequence[str], hop_index: int) -> int:
    # One stage per hop, clamped: hops beyond the stage count keep aiming at the final stage.
    return min(max(hop_index, 0), len(stages) - 1)


def _goal_description(goal: GoalSpec, stages: Sequence[str] | None) -> str:
    if isinstance(goal, FreeTextGoal):
        return goal.goal
    if isinstance(goal, TargetNodeGoal):
        return f"Reach the node whose element id is {goal.target_element_id}."
    assert stages is not None
    return f"Follow the pattern described by: {goal.pattern_description}"


def _noul_true_description(goal: GoalSpec, stages: Sequence[str] | None) -> str:
    if isinstance(goal, FreeTextGoal):
        return goal.goal
    if isinstance(goal, TargetNodeGoal):
        return (
            "The current node is the target node, whose element id is "
            f"{goal.target_element_id}."
        )
    assert stages is not None
    return (
        "The current node satisfies the final stage of the path pattern: "
        f"{stages[-1]}"
    )


def _choice_instructions(goal: GoalSpec, stages: Sequence[str] | None, stage_index: int | None) -> str:
    instruction = (
        "Choose the single outgoing relationship that best advances the navigation goal. "
        "The options are the relationships leaving the current node."
    )
    if stages is not None and stage_index is not None:
        instruction = (
            f"{instruction} The path pattern has {len(stages)} ordered stages: "
            f"{'; '.join(stages)}. This hop should advance stage {stage_index + 1}: "
            f"'{stages[stage_index]}'."
        )
    else:
        instruction = f"{instruction} Goal: {_goal_description(goal, stages)}"
    return instruction


def _node_state(
    node: NodeContext,
    goal: GoalSpec,
    *,
    hop_index: int,
    stages: Sequence[str] | None,
    stage_index: int | None,
    path: Sequence[NavStep],
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "current_node": {
            "element_id": node.element_id,
            "label": node.label,
            "properties": _prompt_props(node.properties),
        },
        "path_so_far": [
            {
                "from_node": step.node_id,
                "relationship_type": step.chosen[0].rel_type if step.chosen else None,
                "to_node": step.chosen[0].target_element_id if step.chosen else step.node_id,
                "to_label": step.chosen[0].target_label if step.chosen else None,
            }
            for step in path
        ],
        "hop_index": hop_index,
        "goal": {
            "mode": goal.kind,
            "description": _goal_description(goal, stages),
        },
    }
    if stages is not None and stage_index is not None:
        state["goal"]["stages"] = list(stages)
        state["goal"]["current_stage_index"] = stage_index
        state["goal"]["current_stage"] = stages[stage_index]
    return state


def _key_candidates(candidates: Sequence[NavCandidate]) -> list[NavCandidate]:
    # Options are re-keyed to opaque e0, e1, ... ids: two relationships of the same type to
    # different targets would otherwise collapse into one option.
    return [replace(candidate, edge_key=f"e{index}") for index, candidate in enumerate(candidates)]


def _select_branches(
    probabilities: Mapping[str, float],
    candidates_by_key: Mapping[str, NavCandidate],
    *,
    top_k: int,
    cutoff: float,
) -> list[tuple[NavCandidate, float]]:
    ranked = sorted(
        ((key, prob) for key, prob in probabilities.items() if key in candidates_by_key),
        key=lambda item: item[1],
        reverse=True,
    )
    if not ranked:
        return []
    above_cutoff = [(key, prob) for key, prob in ranked if prob >= cutoff]
    selected = above_cutoff or ranked[:1]
    return [(candidates_by_key[key], prob) for key, prob in selected[: max(1, top_k)]]


async def one_hop(
    client: SystemOneClient,
    node: NodeContext,
    candidates: Sequence[NavCandidate],
    goal: GoalSpec,
    *,
    config: NavigatorConfig | None = None,
    hop_index: int = 0,
    path: Sequence[NavStep] = (),
    model: str | None = None,
) -> HopResult:
    """Issue exactly one ``system_one`` call at ``node`` (a Choice plus a Noul question).

    A node with no outgoing relationships still gets its one call, asking the Noul alone: an empty
    Choice has no options to select between, and a dead-end goal node must still be recognizable.
    """
    settings = config or NavigatorConfig()

    keyed = _key_candidates(candidates[:MAX_CHOICE_OPTIONS])
    by_key = {candidate.edge_key: candidate for candidate in keyed}
    stages = _stages_for(goal)
    stage_index = _stage_index(stages, hop_index) if stages is not None else None

    questions: dict[str, TypeSafeQuestion] = {
        NOUL_QUESTION: Noul(
            instructions=(
                "Has the navigation goal already been reached at the current node, "
                "before following any further relationship?"
            ),
            criteria={
                "true": _noul_true_description(goal, stages),
                "false": "The current node does not satisfy the goal description.",
            },
        ),
    }
    if keyed:
        questions[CHOICE_QUESTION] = Choice(
            criteria={candidate.edge_key: _candidate_criteria(candidate) for candidate in keyed},
            instructions=_choice_instructions(goal, stages, stage_index),
        )
    state = _node_state(
        node, goal, hop_index=hop_index, stages=stages, stage_index=stage_index, path=path
    )
    response = await client.system_one(state, questions, model=model or settings.model)

    choice_answer = response.choices.get(CHOICE_QUESTION)
    probabilities = dict(choice_answer.probabilities) if choice_answer else {}
    noul_answer = response.nouls.get(NOUL_QUESTION)

    return HopResult(
        node_id=node.element_id,
        candidates=keyed,
        probabilities=probabilities,
        noul=noul_answer.noul if noul_answer else None,
        chosen=_select_branches(
            probabilities, by_key, top_k=settings.top_k, cutoff=settings.cutoff
        ),
        confidence=choice_answer.confidence if choice_answer else None,
    )


def _as_path(state: BeamState, reason: TerminationReason) -> NavPath:
    return NavPath(
        steps=list(state.path),
        cumulative_log_prob=state.cumulative_log_prob,
        terminated_reason=reason,
    )


def _terminal_step(node_id: str, hop: HopResult) -> NavStep:
    # Records where and why the search stopped: no edge was followed from this node.
    return NavStep(
        node_id=node_id,
        chosen=[],
        probabilities=hop.probabilities,
        log_prob=0.0,
        candidates=hop.candidates,
        noul=hop.noul,
    )


def _goal_reached_by_noul(goal: GoalSpec, noul: float | None, threshold: float) -> bool:
    if isinstance(goal, TargetNodeGoal):
        return False
    return noul is not None and noul >= threshold


def _path_signature(path: NavPath) -> tuple[str, ...]:
    # Node sequence only: parallel edges between the same pair collapse into one path.
    nodes = [step.node_id for step in path.steps]
    if path.steps and path.steps[-1].chosen:
        nodes.append(path.steps[-1].chosen[0].target_element_id)
    return tuple(nodes)


@dataclass
class _Expansion:
    children: list[BeamState] = field(default_factory=list)
    terminated: list[NavPath] = field(default_factory=list)
    candidates: list[NavCandidate] = field(default_factory=list)
    nodes: dict[str, NodeContext] = field(default_factory=dict)


async def _expand(
    client: SystemOneClient,
    state: BeamState,
    node: NodeContext,
    goal: GoalSpec,
    config: NavigatorConfig,
    fetch_candidates: CandidateFetcher,
    budget: _Budget,
) -> _Expansion:
    candidates = list(fetch_candidates(state.node_id))
    budget.spend()
    hop = await one_hop(
        client, node, candidates, goal, config=config, hop_index=state.depth, path=state.path
    )
    expansion = _Expansion(candidates=hop.candidates)

    if _goal_reached_by_noul(goal, hop.noul, config.goal_threshold):
        expansion.terminated.append(
            NavPath(
                steps=[*state.path, _terminal_step(state.node_id, hop)],
                cumulative_log_prob=state.cumulative_log_prob,
                terminated_reason=TerminationReason.GOAL_REACHED,
            )
        )
        return expansion

    for candidate, probability in hop.chosen:
        if candidate.target_element_id in state.visited:
            continue
        step = NavStep(
            node_id=state.node_id,
            chosen=[candidate],
            probabilities=hop.probabilities,
            log_prob=math.log(max(probability, _LOG_FLOOR)),
            candidates=hop.candidates,
            noul=hop.noul,
        )
        child = state.extend(step, candidate.target_element_id)
        if isinstance(goal, TargetNodeGoal) and candidate.target_element_id == goal.target_element_id:
            expansion.terminated.append(
                NavPath(
                    steps=list(child.path),
                    cumulative_log_prob=child.cumulative_log_prob,
                    terminated_reason=TerminationReason.GOAL_REACHED,
                )
            )
        else:
            expansion.children.append(child)
            expansion.nodes[candidate.target_element_id] = NodeContext(
                element_id=candidate.target_element_id,
                label=candidate.target_label,
                properties=candidate.target_props,
            )

    if not expansion.children and not expansion.terminated:
        expansion.terminated.append(_as_path(state, TerminationReason.NO_CANDIDATES))
    return expansion


def _rank(paths: Sequence[NavPath]) -> list[NavPath]:
    return sorted(paths, key=lambda path: path.cumulative_log_prob, reverse=True)


def _select_result_paths(
    goal: GoalSpec, terminated: Sequence[NavPath], leftover: Sequence[NavPath]
) -> list[NavPath]:
    ranked = _rank(list(terminated))
    if not isinstance(goal, PathIntentGoal):
        if ranked:
            return ranked[:1]
        return _rank(list(leftover))[:1]

    distinct: list[NavPath] = []
    seen: set[tuple[str, ...]] = set()
    for path in _rank([*terminated, *leftover]):
        signature = _path_signature(path)
        if signature in seen:
            continue
        seen.add(signature)
        distinct.append(path)
        if len(distinct) >= max(1, goal.top_n):
            break
    return distinct


def _to_node_context(start: NodeContext | str) -> NodeContext:
    return start if isinstance(start, NodeContext) else NodeContext(element_id=start)


def _neighborhood(candidates: Sequence[NavCandidate]) -> list[NavCandidate]:
    seen: set[str] = set()
    neighborhood: list[NavCandidate] = []
    for candidate in candidates:
        if candidate.rel_element_id in seen:
            continue
        seen.add(candidate.rel_element_id)
        neighborhood.append(candidate)
    return neighborhood


async def navigate(
    client: SystemOneClient,
    fetch_candidates: CandidateFetcher,
    start: NodeContext | str,
    goal: GoalSpec,
    *,
    config: NavigatorConfig | None = None,
) -> NavResult:
    """Beam-search a path from ``start`` toward ``goal``, one ``system_one`` call per hop."""
    settings = config or NavigatorConfig()
    start_node = _to_node_context(start)
    budget = _Budget(settings.max_calls)
    considered: list[NavCandidate] = []

    if isinstance(goal, TargetNodeGoal) and start_node.element_id == goal.target_element_id:
        return NavResult(
            start_element_id=start_node.element_id,
            paths=[
                NavPath(
                    steps=[],
                    cumulative_log_prob=0.0,
                    terminated_reason=TerminationReason.GOAL_REACHED,
                )
            ],
            start_label=start_node.label,
            start_props=dict(start_node.properties),
        )

    nodes: dict[str, NodeContext] = {start_node.element_id: start_node}
    frontier = [
        BeamState(
            node_id=start_node.element_id,
            path=[],
            cumulative_log_prob=0.0,
            visited=frozenset({start_node.element_id}),
            depth=0,
        )
    ]
    terminated: list[NavPath] = []

    while frontier and budget.remaining:
        depth_capped = [state for state in frontier if state.depth >= settings.max_depth]
        for state in depth_capped:
            terminated.append(_as_path(state, TerminationReason.MAX_DEPTH))
        expandable = [state for state in frontier if state.depth < settings.max_depth]
        if not expandable:
            frontier = []
            break

        batch = expandable[: budget.remaining]
        # Branches are independent round-trips, so a hop's frontier expands concurrently. Only the
        # system_one calls overlap: candidate fetching stays synchronous (a Neo4j driver hop).
        expansions = await asyncio.gather(
            *(
                _expand(
                    client,
                    state,
                    nodes.setdefault(state.node_id, NodeContext(element_id=state.node_id)),
                    goal,
                    settings,
                    fetch_candidates,
                    budget,
                )
                for state in batch
            )
        )
        next_frontier: list[BeamState] = []
        for expansion in expansions:
            next_frontier.extend(expansion.children)
            terminated.extend(expansion.terminated)
            considered.extend(expansion.candidates)
            nodes.update(
                {node_id: ctx for node_id, ctx in expansion.nodes.items() if node_id not in nodes}
            )
        frontier = _rank_states(next_frontier)[: settings.beam_width]

    leftover = [_as_path(state, TerminationReason.BUDGET_EXHAUSTED) for state in frontier]
    return NavResult(
        start_element_id=start_node.element_id,
        paths=_select_result_paths(goal, terminated, leftover),
        neighborhood=_neighborhood(considered),
        start_label=start_node.label,
        start_props=dict(start_node.properties),
    )


def _rank_states(states: Sequence[BeamState]) -> list[BeamState]:
    # BeamState is not orderable; key on (score, unique tiebreaker) instead.
    return sorted(
        states,
        key=lambda state: (-state.cumulative_log_prob, len(state.path), state.node_id),
    )


def run_navigate(
    client: SystemOneClient,
    fetch_candidates: CandidateFetcher,
    start: NodeContext | str,
    goal: GoalSpec,
    *,
    config: NavigatorConfig | None = None,
) -> NavResult:
    """Synchronous convenience wrapper around :func:`navigate`."""
    return asyncio.run(navigate(client, fetch_candidates, start, goal, config=config))


__all__ = [
    "CHOICE_QUESTION",
    "NOUL_QUESTION",
    "CandidateFetcher",
    "HopResult",
    "NavResult",
    "NavigatorConfig",
    "NodeContext",
    "SystemOneClient",
    "decompose_stages",
    "navigate",
    "one_hop",
    "run_navigate",
]
