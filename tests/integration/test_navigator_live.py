"""Live smoke test: one real ``system_one`` call combined with one real Neo4j hop."""

from __future__ import annotations

import asyncio

import pytest

from neo4jev.config import Settings
from neo4jev.navigator import NavigatorConfig, NodeContext, one_hop
from neo4jev.types import FreeTextGoal


def _settings_or_skip() -> Settings:
    try:
        settings = Settings.from_env()
    except ValueError as error:
        pytest.skip(f"Neo4j/TypeSafe credentials not configured: {error}")
    if not settings.typesafe_api_key.strip():
        pytest.skip("TYPESAFE_API_KEY is empty; live TypeSafe calls are not configured")
    return settings


@pytest.fixture(scope="module")
def live_access():
    settings = _settings_or_skip()
    neo4j_access = pytest.importorskip("neo4jev.neo4j_access")
    import neo4j

    try:
        with neo4j_access.open_access(settings) as access:
            # The driver connects lazily, so probe with a real query inside the guard.
            access.list_labels()
            yield access, settings
    except neo4j.exceptions.Neo4jError as error:  # pragma: no cover - network dependent
        pytest.skip(f"Companies KG unreachable: {error}")


def test_one_hop_against_live_graph_and_live_typesafe(live_access):
    import neo4j
    from typesafe_sdk import TypeSafeError

    access, settings = live_access
    try:
        start = access.search_start_nodes("Organization", "Apple", "fulltext", limit=1)
        if not start:
            pytest.skip("No start node found for the live lookup query")

        node = start[0]
        outgoing = access.get_outgoing_relationships(node.element_id, rel_type_cap=5, total_cap=10)
        if not outgoing:
            pytest.skip(f"Node {node.element_id} has no outgoing relationships to navigate")

        client = _client(settings)

        async def hop():
            async with client:
                return await one_hop(
                    client,
                    NodeContext(
                        element_id=node.element_id, label=node.label, properties=node.props
                    ),
                    list(outgoing),
                    FreeTextGoal(goal="find an organization related to the start organization"),
                    config=NavigatorConfig(top_k=2, cutoff=0.05),
                )

        result = asyncio.run(hop())
    except neo4j.exceptions.Neo4jError as error:  # pragma: no cover - network dependent
        pytest.skip(f"Companies KG unreachable: {error}")
    except TypeSafeError as error:  # pragma: no cover - network dependent
        pytest.skip(f"TypeSafe API call unavailable: {error}")

    assert result.node_id == node.element_id
    assert len(result.candidates) == len(outgoing)
    assert set(result.probabilities) <= {candidate.edge_key for candidate in result.candidates}
    assert result.probabilities
    assert result.noul is not None and 0.0 <= result.noul <= 1.0
    assert result.chosen
    for candidate, probability in result.chosen:
        assert candidate.target_element_id in {c.target_element_id for c in outgoing}
        assert probability >= 0.05
        assert candidate.rel_type


def _client(settings: Settings):
    from typesafe_sdk import AsyncTypeSafeClient

    return AsyncTypeSafeClient(api_key=settings.typesafe_api_key)
