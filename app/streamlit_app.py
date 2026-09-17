"""Streamlit front-end for the graph-navigation demo.

Drives ``src/neo4jev`` from the browser: pick a label, look up a start node with a
lookup mode auto-detected from that label's indexes, choose a goal mode, run the
TypeSafe beam search and inspect the resulting graph plus its per-hop trace.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import neo4j
import streamlit as st
from neo4j_viz.streamlit import display_widget
from typesafe_sdk import AsyncTypeSafeClient, TypeSafeError

from neo4jev import navigator, viz
from neo4jev.config import Settings
from neo4jev.neo4j_access import (
    GraphNode,
    IndexRef,
    LabelIndexes,
    LookupMode,
    Neo4jAccess,
)
from neo4jev.types import (
    FreeTextGoal,
    GoalSpec,
    NavCandidate,
    NavResult,
    NavStep,
    PathIntentGoal,
    TargetNodeGoal,
)

FREE_TEXT = "Free text"
TARGET_NODE = "Target node"
PATH_INTENT = "Path intent"
GOAL_MODES = (FREE_TEXT, TARGET_NODE, PATH_INTENT)

GRAPH_WIDGET_KEY = "nav_graph"
MAX_CAPTION_LEN = 80

# DriverError is a sibling of Neo4jError, not a parent: ServiceUnavailable (server
# unreachable) and ConfigurationError (bad URI) only descend from DriverError.
NEO4J_ERRORS = (neo4j.exceptions.Neo4jError, neo4j.exceptions.DriverError)

st.set_page_config(page_title="TypeSafe graph navigation", layout="wide")


@st.cache_resource(show_spinner=False)
def get_settings() -> Settings:
    return Settings.from_env(require_typesafe_key=False)


@st.cache_resource(show_spinner=False)
def get_access() -> Neo4jAccess:
    settings = get_settings()
    driver = neo4j.GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )
    driver.verify_connectivity()
    return Neo4jAccess(driver, database=settings.neo4j_database)


# Cached so the label/index introspection runs once per label instead of on every rerun
# (the graph widget's own interactions rerun the whole script).
@st.cache_data(show_spinner=False)
def list_labels_cached() -> tuple[str, ...]:
    return get_access().list_labels()


@st.cache_data(show_spinner=False)
def detect_indexes_cached(label: str) -> LabelIndexes:
    return get_access().detect_indexes(label)


def _short_name(props: dict[str, Any], label: str, fallback: str) -> str:
    strings = [
        value
        for value in props.values()
        if isinstance(value, str) and 0 < len(value) <= MAX_CAPTION_LEN
    ]
    name = max(strings, key=len) if strings else fallback
    return f"{label} · {name}" if label else name


def _node_caption(node: GraphNode) -> str:
    return _short_name(node.props, node.label, node.element_id)


def _candidate_caption(candidate: NavCandidate) -> str:
    label = candidate.next_label if candidate.direction == "in" else candidate.target_label
    props = candidate.next_props if candidate.direction == "in" else candidate.target_props
    fallback = candidate.next_element_id if candidate.direction == "in" else candidate.target_element_id
    arrow = "←" if candidate.direction == "in" else "→"
    return f"{arrow} {_short_name(props, label, fallback)}"


def _index_options(mode: LookupMode, indexes: LabelIndexes) -> tuple[IndexRef, ...]:
    return indexes.vector if mode == "vector" else indexes.fulltext if mode == "fulltext" else ()


def lookup_widget(prefix: str, label: str) -> GraphNode | None:
    """Search box whose available modes come from the label's live indexes.

    Returns the node the user picked, or ``None`` while nothing is selected.
    """
    indexes = detect_indexes_cached(label)
    modes = list(indexes.modes)
    st.caption(f"Lookup modes for `{label}`: {', '.join(modes)}")

    mode: LookupMode = st.radio(
        "Lookup mode",
        modes,
        key=f"{prefix}_mode",
        horizontal=True,
        help="Auto-detected from SHOW INDEXES for this label: exact is always available.",
    )
    query = st.text_input("Search text", key=f"{prefix}_query")

    index_refs = _index_options(mode, indexes)
    index_name: str | None = None
    if len(index_refs) > 1:
        index_name = st.selectbox("Index", [ref.name for ref in index_refs], key=f"{prefix}_index")
    if mode == "vector":
        st.caption(
            "Nearest-neighbour by embedding; this build uses a placeholder embedder "
            "unless one is injected."
        )

    if st.button("Search", key=f"{prefix}_search"):
        st.session_state[f"{prefix}_results"] = (label, _search(label, query, mode, index_name))

    stored = st.session_state.get(f"{prefix}_results")
    results: list[GraphNode] = stored[1] if stored and stored[0] == label else []
    if not results:
        return None

    picked = st.selectbox(
        f"{len(results)} match(es)",
        options=range(len(results)),
        format_func=lambda index: _node_caption(results[index]),
        key=f"{prefix}_pick",
    )
    return results[picked]


def _search(
    label: str, query: str, mode: LookupMode, index_name: str | None
) -> list[GraphNode]:
    if not query.strip():
        st.warning("Enter some search text first.")
        return []
    try:
        return get_access().search_start_nodes(label, query, mode, index_name=index_name)
    except (*NEO4J_ERRORS, ValueError) as error:
        st.error(f"Lookup failed: {error}")
        return []


def goal_widget(labels: Sequence[str], selected_label: str) -> GoalSpec | None:
    mode = st.radio("Goal mode", GOAL_MODES, key="goal_mode", horizontal=True)

    if mode == FREE_TEXT:
        text = st.text_input(
            "Goal description",
            key="goal_text",
            placeholder="e.g. a node that competes with the start node",
        )
        return FreeTextGoal(goal=text.strip()) if text.strip() else None

    if mode == TARGET_NODE:
        st.caption("Pick the node the search should navigate to.")
        default = labels.index(selected_label) if selected_label in labels else 0
        target_label = st.selectbox("Target label", list(labels), index=default, key="target_label")
        target = lookup_widget("target", target_label)
        return TargetNodeGoal(target_element_id=target.element_id) if target else None

    pattern = st.text_area(
        "Pattern description",
        key="goal_pattern",
        placeholder="describe the hop pattern, one stage per hop, e.g. a -> b -> c",
    )
    top_n = st.number_input("Paths to return (top N)", min_value=1, max_value=10, value=3, key="goal_top_n")
    if not pattern.strip():
        return None
    return PathIntentGoal(pattern_description=pattern.strip(), top_n=int(top_n))


def run_controls() -> navigator.NavigatorConfig:
    with st.sidebar:
        st.header("Run controls")
        top_k = st.number_input("Top-k per hop", min_value=1, max_value=10, value=2)
        cutoff = st.slider("Probability cutoff", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
        max_depth = st.number_input("Max depth", min_value=1, max_value=12, value=4)
        max_calls = st.number_input("Max TypeSafe calls", min_value=1, max_value=200, value=24)
        direction = st.radio(
            "Relationship direction",
            ("out", "in", "both"),
            horizontal=True,
            help="out = edges leaving each node; in = edges pointing at it (e.g. Chunk ← Article); both = all incident edges.",
        )
    return navigator.NavigatorConfig(
        top_k=int(top_k), cutoff=float(cutoff), max_depth=int(max_depth), max_calls=int(max_calls),
        direction=direction,
    )


def _fetch_candidates(node_id: str, direction: str) -> list[NavCandidate]:
    return list(get_access().get_outgoing_relationships(node_id, direction=direction))


async def _navigate(
    client: AsyncTypeSafeClient,
    start: GraphNode,
    goal: GoalSpec,
    config: navigator.NavigatorConfig,
) -> NavResult:
    fetch = lambda node_id: _fetch_candidates(node_id, config.direction)
    async with client:
        return await navigator.navigate(
            client,
            fetch,
            navigator.NodeContext(element_id=start.element_id, label=start.label, properties=start.props),
            goal,
            config=config,
        )


def run_navigation(
    start: GraphNode, goal: GoalSpec, config: navigator.NavigatorConfig
) -> NavResult | None:
    try:
        client = AsyncTypeSafeClient(api_key=get_settings().typesafe_api_key)
        return asyncio.run(_navigate(client, start, goal, config))
    except TypeSafeError as error:
        st.error(f"TypeSafe call failed: {error}")
    except (*NEO4J_ERRORS, ValueError) as error:
        st.error(f"Neo4j query failed while navigating: {error}")
    return None


def _inputs_signature(
    start: GraphNode | None, goal: GoalSpec | None, config: navigator.NavigatorConfig
) -> tuple[str, GoalSpec | None, navigator.NavigatorConfig]:
    return (start.element_id if start else "", goal, config)


def render_trace(result: NavResult) -> None:
    st.subheader("Per-hop trace")
    st.caption(
        f"Start `{result.start_element_id}` · {len(result.paths)} path(s) · "
        f"{len(result.neighborhood)} neighbouring edge(s) in view"
    )
    if not result.paths:
        st.info("No path was produced for this goal and configuration.")
        return

    for index, path in enumerate(result.paths):
        title = (
            f"Path {index + 1} · log-prob {path.cumulative_log_prob:.2f} · "
            f"{path.terminated_reason.value}"
        )
        with st.expander(title, expanded=index == 0):
            if not path.steps:
                st.write("The start node already satisfied the goal; no hop was taken.")
            for hop_index, step in enumerate(path.steps):
                _render_hop(step, hop_index)


def _render_hop(step: NavStep, hop_index: int) -> None:
    """One hop: what Jev was offered, what it picked, and the Noul verdict."""
    # Noul verdict — the goal-reached read on the current node.
    if step.noul is None:
        noul_text = "n/a (no Noul asked)"
    else:
        verdict = "goal reached" if step.noul >= 0.5 else "not yet"
        noul_text = f"{step.noul:.3f} → {verdict}"

    # What was picked.
    if step.chosen:
        picks = ", ".join(
            f"`{c.edge_key}` {c.rel_type} → **{_candidate_caption(c)}** (p {step.probabilities.get(c.edge_key, 0.0):.3f})"
            for c in step.chosen
        )
    else:
        picks = "— (none — path ends here)"

    st.markdown(
        f"**Hop {hop_index + 1}** from `{step.node_id}`\n\n"
        f"- Picked: {picks}\n"
        f"- Noul: {noul_text} · step log-prob {step.log_prob:.2f}"
    )

    rows = _probability_rows(step.candidates, step.probabilities, step.chosen)
    if rows:
        st.caption(f"Relationship options offered ({len(rows)}), ranked by probability:")
        st.dataframe(rows, hide_index=True)


_PROBABILITY_TABLE_MAX_ROWS = 20


def _probability_rows(
    candidates: Sequence[NavCandidate],
    probabilities: dict[str, float],
    chosen: Sequence[NavCandidate],
) -> list[dict[str, Any]]:
    by_key = {candidate.edge_key: candidate for candidate in candidates}
    chosen_keys = {candidate.edge_key for candidate in chosen}
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    rows = []
    for edge_key, probability in ranked[:_PROBABILITY_TABLE_MAX_ROWS]:
        candidate = by_key.get(edge_key)
        rows.append(
            {
                "": "→" if edge_key in chosen_keys else "",
                "edge": edge_key,
                "probability": round(probability, 4),
                "relationship": candidate.rel_type if candidate else "?",
                "target": _candidate_caption(candidate) if candidate else edge_key,
            }
        )
    if len(ranked) > _PROBABILITY_TABLE_MAX_ROWS:
        rows.append(
            {
                "": "",
                "edge": f"… +{len(ranked) - _PROBABILITY_TABLE_MAX_ROWS} more (probability 0)",
                "probability": None,
                "relationship": "",
                "target": "",
            }
        )
    return rows


def main() -> None:
    st.title("TypeSafe graph navigation")
    st.caption(
        "Beam-search through the graph one hop at a time, driven by TypeSafe `system_one` "
        "Choice + Noul questions."
    )

    try:
        settings = get_settings()
    except ValueError as error:
        st.error(f"Configuration error: {error}")
        st.stop()
        return

    try:
        labels = list_labels_cached()
    except (*NEO4J_ERRORS, ValueError) as error:
        st.error(f"Could not read the graph: {error}")
        st.stop()
        return

    typesafe_ready = bool(settings.typesafe_api_key.strip())
    with st.sidebar:
        st.header("Connection")
        st.caption(f"`{settings.neo4j_uri}` · db `{settings.neo4j_database}`")
        st.caption(
            "TypeSafe key: configured" if typesafe_ready else "TypeSafe key: **not configured**"
        )
        if st.button("Reload configuration"):
            # Settings and the driver are cached, so a key added to .env needs this.
            get_settings.clear()
            get_access.clear()
            list_labels_cached.clear()
            detect_indexes_cached.clear()
            st.session_state.pop("nav_result", None)
            st.rerun()
    config = run_controls()

    if not typesafe_ready:
        st.warning(
            "`TYPESAFE_API_KEY` is empty in `.env`: the label/index explorer works, but running "
            "a navigation needs a key. Add one and reload the page."
        )

    selected_label = st.selectbox("Node label", list(labels), key="start_label")
    start_node = lookup_widget("start", selected_label)

    st.divider()
    goal = goal_widget(labels, selected_label)

    if st.button("Run navigation", type="primary"):
        if start_node is None:
            st.error("Pick a start node first: search, then choose one of the matches.")
        elif goal is None:
            st.error("Describe the goal for the selected goal mode.")
        elif not typesafe_ready:
            st.error(
                "Cannot run the beam search: `TYPESAFE_API_KEY` is missing from `.env`. "
                "Set it and reload the page."
            )
        else:
            with st.spinner("Navigating…"):
                result = run_navigation(start_node, goal, config)
            if result is None:
                st.session_state.pop("nav_result", None)
            else:
                st.session_state["nav_result"] = (
                    _inputs_signature(start_node, goal, config),
                    result,
                )

    stored = st.session_state.get("nav_result")
    if stored is None:
        return

    st.divider()
    signature, result = stored
    if signature != _inputs_signature(start_node, goal, config):
        st.info(
            "Showing the result of an earlier run: the start node, goal or run controls "
            "changed since then. Press Run navigation to update it."
        )
    try:
        display_widget(viz.render_for_streamlit(result), key=GRAPH_WIDGET_KEY)
    except Exception as error:  # rendering must never take the whole app down
        st.error(f"Could not render the graph widget: {error}")
    render_trace(result)


main()
