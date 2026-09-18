"""Unit tests for the pure mapping/capping logic in neo4j_access, plus the
record-parsing paths of Neo4jAccess driven by a fake driver (no network)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import neo4j
import pytest

from neo4jev import neo4j_access
from neo4jev.config import Settings
from neo4jev.neo4j_access import (
    DEFAULT_REL_TYPE_CAP,
    DEFAULT_TOTAL_CAP,
    GraphNode,
    Neo4jAccess,
    cap_outgoing_edges,
    default_embedder,
    escape_lucene,
    open_access,
)
from neo4jev.neo4j_access import _quote_label
from neo4jev.types import NavCandidate


def _candidate(rel_type: str, index: int, *, target_label: str = "Organization") -> NavCandidate:
    return NavCandidate(
        edge_key="",
        rel_element_id=f"r{rel_type}{index}",
        rel_type=rel_type,
        rel_props={"index": index},
        target_element_id=f"t{rel_type}{index}",
        target_label=target_label,
        target_props={"name": f"{rel_type} {index}"},
    )


class FakeResult:
    def __init__(self, records):
        self.records = records


class FakeDriver:
    """Minimal stand-in for neo4j.Driver.execute_query."""

    def __init__(self, handler=None):
        self.handler = handler or (lambda query, params: [])
        self.calls = []
        self.closed = False
        self.verified = 0

    def verify_connectivity(self):
        self.verified += 1

    def execute_query(
        self,
        query_,
        parameters_=None,
        routing_=None,
        database_=None,
        **kwargs,
    ):
        self.calls.append(
            {
                "query": query_,
                "params": parameters_ or {},
                "routing": routing_,
                "database": database_,
            }
        )
        return FakeResult(self.handler(query_, parameters_ or {}))

    def close(self):
        self.closed = True


def _access(handler):
    return Neo4jAccess(FakeDriver(handler), database="companies")


# --------------------------------------------------------------------------
# capping + truncation reporting
# --------------------------------------------------------------------------


def test_cap_applies_per_type_cap():
    per_type = {
        "HAS_SUPPLIER": (2558, [_candidate("HAS_SUPPLIER", i) for i in range(2558)]),
    }

    result = cap_outgoing_edges(per_type, rel_type_cap=3, total_cap=60)

    assert len(result.candidates) == 3
    assert result.total_edges == 2558
    assert result.considered_edges == 3
    assert result.truncated is True
    assert result.by_type_totals == {"HAS_SUPPLIER": 2558}
    assert result.by_type_considered == {"HAS_SUPPLIER": 3}


def test_cap_applies_total_cap_across_types():
    per_type = {
        "HAS_COMPETITOR": (25, [_candidate("HAS_COMPETITOR", i) for i in range(25)]),
        "HAS_SUPPLIER": (2558, [_candidate("HAS_SUPPLIER", i) for i in range(2558)]),
        "HAS_SUBSIDIARY": (159, [_candidate("HAS_SUBSIDIARY", i) for i in range(159)]),
    }

    result = cap_outgoing_edges(per_type, rel_type_cap=10, total_cap=25)

    assert len(result.candidates) == 25
    assert result.total_edges == 25 + 2558 + 159
    assert result.truncated is True
    # Round-robin across types in name order: 25 slots across 3 types of 10 = 9/8/8,
    # the first type alphabetically taking the extra slot.
    assert result.by_type_considered == {
        "HAS_COMPETITOR": 9,
        "HAS_SUBSIDIARY": 8,
        "HAS_SUPPLIER": 8,
    }
    assert {c.rel_type for c in result.candidates} == {
        "HAS_COMPETITOR",
        "HAS_SUBSIDIARY",
        "HAS_SUPPLIER",
    }


def test_total_cap_round_robin_prevents_name_order_starvation():
    # The old name-order trim let an early-alphabet type consume the whole budget
    # (Apple's HAS_COMPETITOR reported "0 of 26"). Round-robin keeps every type in play.
    per_type = {
        "APPLIED_FOR": (200, [_candidate("APPLIED_FOR", i) for i in range(200)]),
        "HAS_CATEGORY": (300, [_candidate("HAS_CATEGORY", i) for i in range(300)]),
        "HAS_COMPETITOR": (26, [_candidate("HAS_COMPETITOR", i) for i in range(26)]),
        "HAS_SUPPLIER": (828, [_candidate("HAS_SUPPLIER", i) for i in range(828)]),
    }

    result = cap_outgoing_edges(per_type, rel_type_cap=10, total_cap=60)

    assert len(result.candidates) == 40
    assert result.by_type_considered["HAS_COMPETITOR"] == 10
    assert result.note.startswith("40 of ")


def test_total_cap_respects_per_type_cap_first():
    per_type = {
        "HAS_A": (50, [_candidate("HAS_A", i) for i in range(50)]),
        "HAS_B": (50, [_candidate("HAS_B", i) for i in range(50)]),
    }

    result = cap_outgoing_edges(per_type, rel_type_cap=2, total_cap=10)

    assert len(result.candidates) == 4
    assert result.by_type_considered == {"HAS_A": 2, "HAS_B": 2}


def test_cap_note_reports_n_of_m_edges_considered():
    per_type = {
        "HAS_SUPPLIER": (2558, [_candidate("HAS_SUPPLIER", i) for i in range(2558)]),
        "HAS_CEO": (1, [_candidate("HAS_CEO", 0)]),
    }

    result = cap_outgoing_edges(per_type, rel_type_cap=10, total_cap=60)

    assert result.note == "11 of 2559 edges considered"
    assert result.by_type_note["HAS_SUPPLIER"] == "10 of 2558 HAS_SUPPLIER edges considered"
    assert result.by_type_note["HAS_CEO"] == "1 of 1 HAS_CEO edges considered"


def test_cap_note_when_nothing_truncated():
    per_type = {"HAS_CEO": (2, [_candidate("HAS_CEO", i) for i in range(2)])}

    result = cap_outgoing_edges(per_type, rel_type_cap=10, total_cap=60)

    assert result.truncated is False
    assert result.note == "2 of 2 edges considered"


def test_cap_is_deterministic_and_assigns_sequential_edge_keys():
    per_type = {
        "HAS_B": (2, [_candidate("HAS_B", i) for i in range(2)]),
        "HAS_A": (2, [_candidate("HAS_A", i) for i in range(2)]),
    }

    first = cap_outgoing_edges(per_type, rel_type_cap=10, total_cap=60)
    second = cap_outgoing_edges(per_type, rel_type_cap=10, total_cap=60)

    assert [c.rel_type for c in first.candidates] == ["HAS_A", "HAS_B", "HAS_A", "HAS_B"]
    assert [c.edge_key for c in first.candidates] == ["e0", "e1", "e2", "e3"]
    assert [c.target_element_id for c in first.candidates] == [
        c.target_element_id for c in second.candidates
    ]
    assert first.mapping["e1"].rel_type == "HAS_B"
    assert first.source_element_id == second.source_element_id


def test_cap_with_no_edges():
    result = cap_outgoing_edges({}, rel_type_cap=10, total_cap=60, source_element_id="n1")

    assert result.candidates == []
    assert result.total_edges == 0
    assert result.truncated is False
    assert result.note == "0 of 0 edges considered"


def test_outgoing_candidates_is_list_like():
    per_type = {"HAS_CEO": (1, [_candidate("HAS_CEO", 0)])}
    result = cap_outgoing_edges(per_type, rel_type_cap=10, total_cap=60)

    assert len(result) == 1
    assert list(result)[0].rel_type == "HAS_CEO"
    assert result[0].edge_key == "e0"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def test_escape_lucene_escapes_syntax_characters():
    assert escape_lucene("Apple Inc.") == "Apple Inc."
    assert escape_lucene('a+b (c)') == "a\\+b \\(c\\)"
    assert escape_lucene('Acme: "The Best"') == 'Acme\\: \\"The Best\\"'


def test_default_embedder_is_deterministic_and_normalised():
    first = default_embedder("Apple", 16)
    second = default_embedder("Apple", 16)

    assert len(first) == 16
    assert first == second
    assert default_embedder("Google", 16) != first
    assert math.isclose(sum(v * v for v in first), 1.0, rel_tol=1e-9)


# --------------------------------------------------------------------------
# driver-backed methods against a fake driver
# --------------------------------------------------------------------------


def test_get_outgoing_relationships_parses_caps_and_reports_truncation():
    def handler(query, params):
        if "MATCH (n)-[r]->(t)" not in query:
            return []
        assert params["node_id"] == "n:1"
        return [
            {
                "rel_type": "HAS_SUPPLIER",
                "total": 2558,
                "kept": [
                    {
                        "rel_id": f"r{i}",
                        "rel_type": "HAS_SUPPLIER",
                        "rel_props": {"since": i},
                        "start_id": "n:1",
                        "start_labels": ["Organization"],
                        "start_props": {},
                        "end_id": f"t{i}",
                        "end_labels": ["Organization"],
                        "end_props": {"name": f"Supplier {i}"},
                    }
                    for i in range(params["rel_type_cap"])
                ],
                "node_labels": ["Organization", "Company"],
            },
            {
                "rel_type": "HAS_CEO",
                "total": 1,
                "kept": [
                    {
                        "rel_id": "rceo",
                        "rel_type": "HAS_CEO",
                        "rel_props": {},
                        "start_id": "n:1",
                        "start_labels": ["Organization"],
                        "start_props": {},
                        "end_id": "tceo",
                        "end_labels": ["Person"],
                        "end_props": {"name": "Tim Cook"},
                    }
                ],
                "node_labels": ["Organization", "Company"],
            },
        ]

    driver = FakeDriver(handler)
    access = Neo4jAccess(driver, database="companies")
    result = access.get_outgoing_relationships("n:1", rel_type_cap=10, total_cap=60)

    assert result.source_element_id == "n:1"
    assert result.source_label == "Organization"
    assert result.total_edges == 2559
    assert result.truncated is True
    assert result.note == "11 of 2559 edges considered"
    assert [c.edge_key for c in result.candidates] == [f"e{i}" for i in range(11)]
    assert all(c.source_element_id == "n:1" for c in result.candidates)
    assert all(c.source_label == "Organization" for c in result.candidates)
    # Relationship types are visited in name order: HAS_CEO first, then HAS_SUPPLIER.
    assert result.candidates[0].rel_type == "HAS_CEO"
    assert result.candidates[0].rel_props == {}
    assert result.candidates[0].target_label == "Person"
    assert result.candidates[-1].rel_type == "HAS_SUPPLIER"
    assert result.candidates[-1].rel_props == {"since": 9}
    assert result.mapping["e0"].target_element_id == "tceo"
    assert result.mapping["e10"].target_element_id == "t9"

    call = driver.calls[0]
    assert call["routing"] is neo4j.RoutingControl.READ
    assert call["database"] == "companies"


def test_get_outgoing_relationships_honours_total_cap_and_empty_node():
    def handler(query, params):
        return [
            {
                "rel_type": "HAS_SUPPLIER",
                "total": 12,
                "kept": [
                    {
                        "rel_id": f"r{i}",
                        "rel_type": "HAS_SUPPLIER",
                        "rel_props": {},
                        "target_id": f"t{i}",
                        "target_labels": ["Organization"],
                        "target_props": {},
                    }
                    for i in range(12)
                ],
                "node_labels": ["Organization"],
            }
        ]

    access = _access(handler)
    capped = access.get_outgoing_relationships("n:1", rel_type_cap=10, total_cap=4)
    assert len(capped.candidates) == 4
    assert capped.truncated is True
    assert capped.note == "4 of 12 edges considered"

    empty = _access(lambda query, params: []).get_outgoing_relationships("n:2")
    assert empty.candidates == []
    assert empty.source_element_id == "n:2"
    assert empty.note == "0 of 0 edges considered"


def test_get_outgoing_relationships_resolves_source_label_for_edgeless_node():
    def handler(query, params):
        if "MATCH (n)-[r]->(t)" in query:
            return []
        if "$element_id" in query:
            return [
                {
                    "element_id": "n:leaf",
                    "labels": ["Organization"],
                    "props": {"name": "Leaf Co"},
                }
            ]
        return []

    result = _access(handler).get_outgoing_relationships("n:leaf")

    assert result.candidates == []
    assert result.source_label == "Organization"
    assert result.note == "0 of 0 edges considered"


def test_get_outgoing_relationships_uses_default_caps():
    seen = {}

    def handler(query, params):
        if "rel_type_cap" in params:
            seen["cap"] = params["rel_type_cap"]
        return []

    _access(handler).get_outgoing_relationships("n:1")

    assert seen["cap"] == DEFAULT_REL_TYPE_CAP
    assert DEFAULT_TOTAL_CAP == 60


def test_list_labels_is_sorted():
    access = _access(
        lambda query, params: [
            {"label": "Organization"},
            {"label": "Article"},
            {"label": "Person"},
        ]
    )

    assert access.list_labels() == ("Article", "Organization", "Person")


def test_detect_indexes_reports_available_modes():
    def handler(query, params):
        assert "SHOW INDEXES" in query
        assert params["label"] == "Organization"
        return [
            {
                "name": "entity",
                "type": "FULLTEXT",
                "properties": ["name"],
                "options": {"indexConfig": {"fulltext.analyzer": "standard-no-stop-words"}},
            },
            {
                "name": "organization_name",
                "type": "VECTOR",
                "properties": ["name_embedding"],
                "options": {"indexConfig": {"vector.dimensions": 1536}},
            },
        ]

    indexes = _access(handler).detect_indexes("Organization")

    assert indexes.modes == ("exact", "fulltext", "vector")
    assert indexes.supports("exact") is True
    assert indexes.fulltext[0].name == "entity"
    assert indexes.fulltext[0].properties == ("name",)
    assert indexes.fulltext[0].dimensions is None
    assert indexes.vector[0].name == "organization_name"
    assert indexes.vector[0].dimensions == 1536


def test_detect_indexes_vector_only_label_has_no_fulltext_mode():
    def handler(query, params):
        return [
            {
                "name": "only_vector",
                "type": "VECTOR",
                "properties": ["embedding"],
                "options": {"indexConfig": {"vector.dimensions": 384}},
            }
        ]

    indexes = _access(handler).detect_indexes("Chunk")

    assert indexes.modes == ("exact", "vector")
    assert indexes.fulltext == ()
    assert indexes.supports("fulltext") is False


def test_detect_indexes_exact_only_when_no_fulltext_or_vector():
    indexes = _access(lambda query, params: []).detect_indexes("City")

    assert indexes.modes == ("exact",)
    assert indexes.supports("fulltext") is False
    assert indexes.supports("vector") is False


def test_get_node_returns_none_when_missing():
    access = _access(lambda query, params: [])

    assert access.get_node("n:missing") is None


def test_get_node_parses_record():
    access = _access(
        lambda query, params: [
            {"element_id": "n:1", "labels": ["Organization"], "props": {"name": "Apple"}}
        ]
    )

    node = access.get_node("n:1")

    assert node == GraphNode(
        element_id="n:1", labels=("Organization",), props={"name": "Apple"}, score=None
    )
    assert node.label == "Organization"


def test_search_start_nodes_exact_matches_any_string_property():
    captured = {}

    def handler(query, params):
        captured["query"] = query
        captured["params"] = params
        return [
            {"element_id": "n:1", "labels": ["Organization"], "props": {"name": "Apple"}}
        ]

    nodes = _access(handler).search_start_nodes("Organization", "Apple", "exact", limit=3)

    assert nodes[0].element_id == "n:1"
    assert "MATCH (n:`Organization`)" in captured["query"]
    assert "keys(n)" in captured["query"]
    assert captured["params"] == {"query": "Apple", "limit": 3}


def test_search_start_nodes_returns_empty_for_blank_query():
    access = _access(lambda query, params: pytest.fail("driver should not be queried"))

    assert access.search_start_nodes("Organization", "   ", "exact") == []


def test_search_start_nodes_fulltext_escapes_query_and_filters_label():
    captured = {}

    def handler(query, params):
        if "SHOW INDEXES" in query:
            return [
                {
                    "name": "entity",
                    "type": "FULLTEXT",
                    "properties": ["name"],
                    "options": {},
                }
            ]
        captured["query"] = query
        captured["params"] = params
        return [
            {
                "element_id": "n:2",
                "labels": ["Organization"],
                "props": {"name": "Apple"},
                "score": 4.5,
            }
        ]

    nodes = _access(handler).search_start_nodes("Organization", "Apple (Inc.)", "fulltext")

    assert nodes[0].score == 4.5
    assert "db.index.fulltext.queryNodes" in captured["query"]
    assert captured["params"]["index_name"] == "entity"
    assert captured["params"]["query"] == "Apple \\(Inc.\\)"
    assert captured["params"]["label"] == "Organization"


def test_search_start_nodes_vector_embeds_to_index_dimension():
    captured = {}

    def handler(query, params):
        if "SHOW INDEXES" in query:
            return [
                {
                    "name": "news",
                    "type": "VECTOR",
                    "properties": ["embedding"],
                    "options": {"indexConfig": {"vector.dimensions": 1536}},
                }
            ]
        captured["query"] = query
        captured["params"] = params
        return [
            {
                "element_id": "c:1",
                "labels": ["Chunk"],
                "props": {"text": "..."},
                "score": 0.9,
            }
        ]

    nodes = _access(handler).search_start_nodes("Chunk", "renewable energy", "vector", limit=4)

    assert nodes[0].element_id == "c:1"
    assert "db.index.vector.queryNodes" in captured["query"]
    assert captured["params"]["index_name"] == "news"
    assert captured["params"]["k"] == 4
    assert len(captured["params"]["embedding"]) == 1536


def test_search_start_nodes_vector_rejects_wrong_dimension_embedder():
    def handler(query, params):
        return [
            {
                "name": "news",
                "type": "VECTOR",
                "properties": ["embedding"],
                "options": {"indexConfig": {"vector.dimensions": 1536}},
            }
        ]

    access = Neo4jAccess(
        FakeDriver(handler),
        database="companies",
        embedder=lambda text, dimensions: [0.0, 1.0],
    )

    with pytest.raises(ValueError, match="dimensions"):
        access.search_start_nodes("Chunk", "energy", "vector")


@pytest.mark.parametrize("mode", ["fulltext", "vector"])
def test_search_start_nodes_raises_when_index_unavailable(mode):
    access = _access(lambda query, params: [])

    with pytest.raises(ValueError, match="No .* index available"):
        access.search_start_nodes("City", "Berlin", mode)


def _multi_vector_index_handler():
    def handler(query, params):
        if "SHOW INDEXES" in query:
            return [
                {
                    "name": "news_google",
                    "type": "VECTOR",
                    "properties": ["embedding_google"],
                    "options": {"indexConfig": {"vector.dimensions": 768}},
                },
                {
                    "name": "news",
                    "type": "VECTOR",
                    "properties": ["embedding"],
                    "options": {"indexConfig": {"vector.dimensions": 1536}},
                },
            ]
        return []

    return handler


def test_search_start_nodes_can_select_which_vector_index_to_use():
    captured = {}

    def handler(query, params):
        if "SHOW INDEXES" in query:
            return _multi_vector_index_handler()(query, params)
        captured.update(params)
        return []

    access = Neo4jAccess(
        FakeDriver(handler),
        database="companies",
        embedder=lambda text, dimensions: [0.5] * dimensions,
    )

    access.search_start_nodes("Chunk", "energy", "vector", index_name="news_google")
    assert captured["index_name"] == "news_google"
    assert len(captured["embedding"]) == 768

    # Without an explicit name the name-sorted first index is used.
    access.search_start_nodes("Chunk", "energy", "vector")
    assert captured["index_name"] == "news"


def test_search_start_nodes_rejects_index_name_not_matching_label_and_mode():
    access = Neo4jAccess(FakeDriver(_multi_vector_index_handler()), database="companies")

    with pytest.raises(ValueError, match="not a vector index"):
        access.search_start_nodes("Chunk", "energy", "vector", index_name="entity")

    with pytest.raises(ValueError, match="fulltext/vector lookup only"):
        access.search_start_nodes("Chunk", "energy", "exact", index_name="news")


def test_search_start_nodes_vector_requires_declared_dimensions():
    def handler(query, params):
        return [
            {
                "name": "news",
                "type": "VECTOR",
                "properties": ["embedding"],
                "options": {"indexConfig": {}},
            }
        ]

    access = _access(handler)

    with pytest.raises(ValueError, match="vector.dimensions"):
        access.search_start_nodes("Chunk", "energy", "vector")


def test_search_start_nodes_rejects_unknown_mode():
    access = _access(lambda query, params: [])

    with pytest.raises(ValueError, match="Unsupported lookup mode"):
        access.search_start_nodes("City", "Berlin", "magic")


def test_search_start_nodes_validates_mode_before_blank_query_short_circuit():
    access = _access(lambda query, params: [])

    with pytest.raises(ValueError, match="Unsupported lookup mode"):
        access.search_start_nodes("City", "   ", "magic")


def test_quote_label_escapes_backticks():
    assert _quote_label("Weird`Label") == "`Weird``Label`"
    with pytest.raises(ValueError):
        _quote_label("")


def test_get_outgoing_relationships_incoming_direction_marks_candidates_in():
    """direction='in' queries incoming edges and marks candidates direction='in',
    with source_* carrying the far endpoint (the node the traversal would reach)."""
    captured = {}

    def handler(query, params):
        captured["query"] = query
        return [
            {
                "rel_type": "HAS_CHUNK",
                "total": 3,
                "kept": [
                    {
                        "rel_id": "r1",
                        "rel_type": "HAS_CHUNK",
                        "rel_props": {},
                        "start_id": "a:1",
                        "start_labels": ["Article"],
                        "start_props": {"title": "Some article"},
                        "end_id": "c:9",  # current node (the Chunk)
                        "end_labels": ["Chunk"],
                        "end_props": {"text": "chunk text"},
                    }
                ],
                "node_labels": ["Chunk"],
            }
        ]

    access = _access(handler)
    result = access.get_outgoing_relationships("c:9", direction="in")

    assert "MATCH (n)<-[r]-(t)" in captured["query"]
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.direction == "in"
    # source_* is the edge's real start (the Article), the node you'd walk to.
    assert candidate.source_element_id == "a:1"
    assert candidate.source_label == "Article"
    # target_* is the edge's end: the current node (the Chunk).
    assert candidate.target_element_id == "c:9"
    assert candidate.target_label == "Chunk"
    assert candidate.target_props == {"text": "chunk text"}
    # next_* resolves to the far side (the source of the incoming edge).
    assert candidate.next_element_id == "a:1"
    assert candidate.next_label == "Article"
    assert candidate.next_props == {}


def test_get_outgoing_relationships_rejects_bad_direction():
    access = _access(lambda q, p: [])
    with pytest.raises(ValueError, match="direction"):
        access.get_outgoing_relationships("n:1", direction="sideways")


def test_navcandidate_next_resolves_direction():
    from neo4jev.types import NavCandidate

    out = NavCandidate(
        edge_key="", rel_element_id="r", rel_type="T", rel_props={},
        target_element_id="t", target_label="Target", target_props={"name": "T"},
        source_element_id="s", source_label="Source",
    )
    assert out.direction == "out"
    assert out.next_element_id == "t"
    assert out.next_label == "Target"
    assert out.next_props == {"name": "T"}

    incoming = NavCandidate(
        edge_key="", rel_element_id="r", rel_type="T", rel_props={},
        target_element_id="t", target_label="Target", target_props={"name": "T"},
        source_element_id="s", source_label="Source", direction="in",
    )
    assert incoming.direction == "in"
    assert incoming.next_element_id == "s"
    assert incoming.next_label == "Source"
    assert incoming.next_props == {}
    captured = {}

    def handler(query, params):
        captured["params"] = params
        return [
            {
                "rel_id": "r1",
                "rel_type": "MENTIONS",
                "rel_props": {"sentiment": 0.2},
                "source_id": "a:1",
                "source_labels": ["Article"],
                "target_id": "n:1",
                "target_labels": ["Organization"],
                "target_props": {"name": "Apple"},
            }
        ]

    candidates = _access(handler).get_node_neighborhood(["n:1"], limit=50)

    assert captured["params"] == {"node_ids": ["n:1"], "limit": 50}
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.edge_key == "e0"
    assert candidate.source_element_id == "a:1"
    assert candidate.source_label == "Article"
    assert candidate.target_element_id == "n:1"
    assert candidate.target_label == "Organization"
    assert candidate.target_props == {"name": "Apple"}
    assert candidate.rel_props == {"sentiment": 0.2}


def test_get_node_neighborhood_short_circuits_without_ids():
    access = _access(lambda query, params: pytest.fail("driver should not be queried"))

    assert access.get_node_neighborhood([]) == []


def test_open_access_builds_driver_from_settings_and_closes_it(monkeypatch):
    fake_driver = FakeDriver()
    created = {}

    def fake_driver_factory(uri, auth=None, **kwargs):
        created["uri"] = uri
        created["auth"] = auth
        return fake_driver

    monkeypatch.setattr(neo4j_access.neo4j.GraphDatabase, "driver", fake_driver_factory)
    settings = Settings(
        neo4j_uri="neo4j+s://demo.neo4jlabs.com:7687",
        neo4j_username="companies",
        neo4j_password="companies",
        neo4j_database="companies",
        typesafe_api_key="",
    )

    with open_access(settings) as access:
        assert isinstance(access, Neo4jAccess)
        assert access.database == "companies"

    assert created["uri"] == "neo4j+s://demo.neo4jlabs.com:7687"
    assert created["auth"] == ("companies", "companies")
    assert fake_driver.verified == 1
    assert fake_driver.closed is True


def test_open_access_without_settings_tolerates_a_missing_typesafe_key(monkeypatch):
    fake_driver = FakeDriver()

    def fake_driver_factory(uri, auth=None, **kwargs):
        return fake_driver

    monkeypatch.setattr(neo4j_access.neo4j.GraphDatabase, "driver", fake_driver_factory)
    monkeypatch.setattr("neo4jev.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("NEO4J_URI", "neo4j+s://demo.neo4jlabs.com:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "companies")
    monkeypatch.setenv("NEO4J_PASSWORD", "companies")
    monkeypatch.setenv("NEO4J_DATABASE", "companies")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)

    with open_access() as access:
        assert isinstance(access, Neo4jAccess)
        assert access.database == "companies"

    assert fake_driver.verified == 1
    assert fake_driver.closed is True


# --------------------------------------------------------------------------
# graph_schema
# --------------------------------------------------------------------------


def test_graph_schema_maps_labels_types_and_topology_triples():
    nodes = {
        "n1": SimpleNamespace(element_id="n1", labels=["Organization"]),
        "n2": SimpleNamespace(element_id="n2", labels=["Patent"]),
        "n3": SimpleNamespace(element_id="n3", labels=["Article"]),
    }
    rels = [
        SimpleNamespace(type="APPLIED_FOR", start_node=nodes["n1"], end_node=nodes["n2"]),
        SimpleNamespace(type="ASSIGNED", start_node=nodes["n1"], end_node=nodes["n2"]),
        SimpleNamespace(type="HAS_CHUNK", start_node=nodes["n3"], end_node=nodes["n3"]),
    ]

    def handler(query, params):
        if "db.labels" in query:
            return [{"label": "Patent"}, {"label": "Organization"}, {"label": "Article"}]
        if "db.relationshipTypes" in query:
            return [{"relationshipType": t} for t in ("HAS_CHUNK", "APPLIED_FOR", "ASSIGNED")]
        if "db.schema.visualization" in query:
            return [{"nodes": list(nodes.values()), "relationships": rels}]
        raise AssertionError(f"unexpected query: {query}")

    schema = _access(handler).graph_schema()

    assert schema.labels == ("Article", "Organization", "Patent")
    assert schema.relationship_types == ("APPLIED_FOR", "ASSIGNED", "HAS_CHUNK")
    assert schema.relationships == (
        ("Article", "HAS_CHUNK", "Article"),
        ("Organization", "APPLIED_FOR", "Patent"),
        ("Organization", "ASSIGNED", "Patent"),
    )
    assert schema.as_prompt()["topology"][0] == {
        "from": "Article",
        "relationship": "HAS_CHUNK",
        "to": "Article",
    }


def test_graph_schema_falls_back_to_labels_and_types_when_visualization_is_unavailable():
    import neo4j

    def handler(query, params):
        if "db.labels" in query:
            return [{"label": "Organization"}]
        if "db.relationshipTypes" in query:
            return [{"relationshipType": "APPLIED_FOR"}]
        raise neo4j.exceptions.Neo4jError("no such procedure")

    schema = _access(handler).graph_schema()

    assert schema.labels == ("Organization",)
    assert schema.relationship_types == ("APPLIED_FOR",)
    assert schema.relationships == ()
