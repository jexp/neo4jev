from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Literal, Union


class TerminationReason(str, Enum):
    GOAL_REACHED = "goal_reached"
    MAX_DEPTH = "max_depth"
    NO_CANDIDATES = "no_candidates"
    BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass(frozen=True)
class NavCandidate:
    edge_key: str
    rel_element_id: str
    rel_type: str
    rel_props: dict[str, Any]
    target_element_id: str
    target_label: str
    target_props: dict[str, Any]
    source_element_id: str = ""
    source_label: str = ""


def assign_edge_keys(candidates: Sequence[NavCandidate]) -> list[NavCandidate]:
    """Key candidates by synthetic opaque ids (``e0``, ``e1``, ...).

    Keying by relationship type would collapse several relationships of the same
    type to different targets into one option, so the ids stay positional.
    """
    return [
        replace(candidate, edge_key=f"e{index}")
        for index, candidate in enumerate(candidates)
    ]


@dataclass(frozen=True)
class NavStep:
    node_id: str
    chosen: list[NavCandidate]
    probabilities: dict[str, float]
    log_prob: float
    candidates: list[NavCandidate] = field(default_factory=list)
    noul: float | None = None


@dataclass(frozen=True)
class NavPath:
    steps: list[NavStep]
    cumulative_log_prob: float
    terminated_reason: TerminationReason


@dataclass(frozen=True)
class NavResult:
    start_element_id: str
    paths: list[NavPath]
    neighborhood: list[NavCandidate] = field(default_factory=list)
    start_label: str = ""
    start_props: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BeamState:
    # Not orderable/hashable: beam-ranking heaps must key on (score, unique_tiebreaker),
    # never compare two BeamStates directly.
    node_id: str
    path: list[NavStep]
    cumulative_log_prob: float
    visited: frozenset[str]
    depth: int

    def extend(self, step: NavStep, node_id: str) -> "BeamState":
        return replace(
            self,
            node_id=node_id,
            path=[*self.path, step],
            cumulative_log_prob=self.cumulative_log_prob + step.log_prob,
            visited=self.visited | {node_id},
            depth=self.depth + 1,
        )


@dataclass(frozen=True)
class FreeTextGoal:
    goal: str
    kind: Literal["free_text"] = "free_text"


@dataclass(frozen=True)
class TargetNodeGoal:
    target_element_id: str
    kind: Literal["target_node"] = "target_node"


@dataclass(frozen=True)
class PathIntentGoal:
    pattern_description: str
    top_n: int = 3
    stages: list[str] | None = None
    kind: Literal["path_intent"] = "path_intent"


GoalSpec = Union[FreeTextGoal, TargetNodeGoal, PathIntentGoal]
