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


def test_display_properties_derive_real_names_from_the_live_graph(live_access):
    """One live TypeSafe call per label decides which properties are the identity."""
    import neo4j
    from typesafe_sdk import TypeSafeError

    from neo4jev.neo4j_access import (
        clear_display_properties,
        identity_candidates,
        looks_like_reference,
        resolve_display_value,
    )

    access, _settings = live_access
    clear_display_properties()
    try:
        schema = access.describe_label("Organization")
        derived = access.display_properties("Organization")
        nodes = access.search_start_nodes("Organization", "Apple", "exact", limit=5)
    except neo4j.exceptions.Neo4jError as error:  # pragma: no cover - network dependent
        pytest.skip(f"Companies KG unreachable: {error}")
    except TypeSafeError as error:  # pragma: no cover - network dependent
        pytest.skip(f"TypeSafe API call unavailable: {error}")

    assert derived, "expected the model to name at least one display property"
    assert len(derived) <= 3
    assert set(derived) <= set(identity_candidates(schema))
    assert "name" in derived or "fullName" in derived
    # No derived property may be a URI/reference property.
    for key in derived:
        assert not any(looks_like_reference(value) for value in schema.sample_values[key])

    assert nodes
    captions = [resolve_display_value(node.props, "Organization") for node in nodes]
    assert any(caption in {"Apple", "APPLE"} for caption in captions), captions
    assert all(caption is None or not looks_like_reference(caption) for caption in captions)


def _client(settings: Settings):
    from typesafe_sdk import AsyncTypeSafeClient

    return AsyncTypeSafeClient(api_key=settings.typesafe_api_key)
