"""Integration tests against the live public Neo4j "Companies KG" demo instance.

Credentials are public and read from the environment / `.env` (see
`.env.example`). When no Neo4j connection is configured the whole module skips
instead of failing.
"""

from __future__ import annotations

import pytest

from neo4jev.config import Settings
from neo4jev.neo4j_access import Neo4jAccess, open_access, settings_from_env

# The Companies KG's own labels/types — used here as expectations, never as
# application logic.
EXPECTED_LABEL = "Organization"
EXPECTED_FULLTEXT_INDEX = "entity"
EXPECTED_VECTOR_LABEL = "Chunk"
EXPECTED_VECTOR_INDEX = "news"
WELL_KNOWN_COMPANY = "Apple"


def _settings_or_skip() -> Settings:
    try:
        return settings_from_env()
    except ValueError as exc:
        pytest.skip(
            f"No live Neo4j configured (missing .env) — skipping Companies KG tests: {exc}"
        )


@pytest.fixture(scope="module")
def access():
    with open_access(_settings_or_skip()) as live:
        assert isinstance(live, Neo4jAccess)
        yield live


def _company_with_outgoing_edges(access: Neo4jAccess):
    """A real company node that actually has outgoing edges.

    Same-named organizations tie on fulltext score and their relative order is
    not stable, so pick from the top hits instead of trusting a single one.
    """
    nodes = access.search_start_nodes(EXPECTED_LABEL, WELL_KNOWN_COMPANY, "fulltext", limit=5)
    assert nodes, "expected the live graph to know this company"
    for node in nodes:
        if access.get_outgoing_relationships(node.element_id, rel_type_cap=1).total_edges:
            return node
    pytest.fail("none of the top fulltext hits had outgoing relationships")


def test_list_labels_returns_expected_labels(access):
    labels = access.list_labels()

    assert labels == tuple(sorted(labels))
    for expected in (EXPECTED_LABEL, "Person", "Article", "Chunk"):
        assert expected in labels


def test_detect_indexes_finds_entity_fulltext_index_for_organization(access):
    indexes = access.detect_indexes(EXPECTED_LABEL)

    assert indexes.supports("exact")
    assert indexes.supports("fulltext")
    assert EXPECTED_FULLTEXT_INDEX in [ref.name for ref in indexes.fulltext]
    entity = next(ref for ref in indexes.fulltext if ref.name == EXPECTED_FULLTEXT_INDEX)
    assert entity.kind == "FULLTEXT"
    assert "name" in entity.properties


def test_detect_indexes_finds_news_vector_index_for_chunk(access):
    indexes = access.detect_indexes(EXPECTED_VECTOR_LABEL)

    assert indexes.supports("vector")
    news = next(
        ref for ref in indexes.vector if ref.name == EXPECTED_VECTOR_INDEX
    )
    assert news.dimensions and news.dimensions > 0
    assert news.properties


def test_fulltext_search_returns_real_organization(access):
    nodes = access.search_start_nodes(EXPECTED_LABEL, WELL_KNOWN_COMPANY, "fulltext", limit=5)

    assert nodes
    assert all(EXPECTED_LABEL in node.labels for node in nodes)
    assert all(node.score is not None for node in nodes)
    assert any(str(node.props.get("name", "")).startswith(WELL_KNOWN_COMPANY) for node in nodes)


def test_exact_search_finds_node_by_substring(access):
    nodes = access.search_start_nodes(EXPECTED_LABEL, WELL_KNOWN_COMPANY, "exact", limit=5)

    assert nodes
    assert any(node.props.get("name") == WELL_KNOWN_COMPANY for node in nodes)


def test_vector_search_executes_against_live_index(access):
    nodes = access.search_start_nodes(EXPECTED_VECTOR_LABEL, "renewable energy", "vector", limit=3)

    assert len(nodes) <= 3
    assert all(EXPECTED_VECTOR_LABEL in node.labels for node in nodes)


def test_get_outgoing_relationships_is_capped_and_truncated(access):
    node = _company_with_outgoing_edges(access)

    result = access.get_outgoing_relationships(node.element_id, rel_type_cap=10, total_cap=60)

    assert result.candidates
    assert len(result.candidates) <= 60
    assert result.total_edges >= len(result.candidates)
    assert result.source_element_id == node.element_id
    assert result.source_label == EXPECTED_LABEL
    assert [c.edge_key for c in result.candidates] == [
        f"e{i}" for i in range(len(result.candidates))
    ]
    assert all(c.target_element_id for c in result.candidates)
    assert all(c.target_label for c in result.candidates)
    assert all(c.rel_type for c in result.candidates)

    if result.truncated:
        assert result.note == (
            f"{result.considered_edges} of {result.total_edges} edges considered"
        )


def test_get_outgoing_relationships_per_type_cap_applies(access):
    node = _company_with_outgoing_edges(access)

    result = access.get_outgoing_relationships(node.element_id, rel_type_cap=2, total_cap=60)

    assert result.candidates, "expected a company with outgoing relationships"

    for rel_type, considered in result.by_type_considered.items():
        assert considered <= 2
        assert considered <= result.by_type_totals[rel_type]


def test_get_node_and_neighborhood_around_real_node(access):
    nodes = access.search_start_nodes(EXPECTED_LABEL, WELL_KNOWN_COMPANY, "fulltext", limit=1)
    assert nodes
    element_id = nodes[0].element_id

    node = access.get_node(element_id)
    assert node is not None
    assert EXPECTED_LABEL in node.labels
    assert node.props

    neighborhood = access.get_node_neighborhood([element_id], limit=25)

    assert neighborhood
    for candidate in neighborhood:
        assert candidate.rel_element_id
        assert candidate.rel_type
        assert candidate.source_element_id
        assert candidate.target_element_id
        assert element_id in (candidate.source_element_id, candidate.target_element_id)
        assert candidate.source_element_id != candidate.target_element_id
