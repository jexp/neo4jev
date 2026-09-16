"""Unit tests for the pure mapping/capping logic in neo4j_access, plus the
record-parsing paths of Neo4jAccess driven by a fake driver (no network)."""

from __future__ import annotations

import math

import neo4j
import pytest

from neo4jev import neo4j_access
from neo4jev.config import Settings
from neo4jev.neo4j_access import (
    DEFAULT_REL_TYPE_CAP,
    DEFAULT_TOTAL_CAP,
    GraphNode,
    Neo4jAccess,
    assign_edge_keys,
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
# edge_key generation / mapping
# --------------------------------------------------------------------------


def test_assign_edge_keys_is_positional_and_maps_back():
    candidates = [_candidate("HAS_SUBSIDIARY", 0), _candidate("HAS_SUBSIDIARY", 1)]

    keyed, mapping = assign_edge_keys(candidates)

    assert [c.edge_key for c in keyed] == ["e0", "e1"]
    assert set(mapping) == {"e0", "e1"}
    assert mapping["e1"].target_element_id == candidates[1].target_element_id
    # Same relationship type must not collapse into one key.
    assert keyed[0].edge_key != keyed[1].edge_key


def test_assign_edge_keys_of_empty_list():
    keyed, mapping = assign_edge_keys([])
    assert keyed == []
    assert mapping == {}


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
    # Types visited in name order: COMPETITOR(10) + SUBSIDIARY(10) + SUPPLIER(5)
    assert result.by_type_considered == {
        "HAS_COMPETITOR": 10,
        "HAS_SUBSIDIARY": 10,
        "HAS_SUPPLIER": 5,
    }
    assert {c.rel_type for c in result.candidates} == {
        "HAS_COMPETITOR",
        "HAS_SUBSIDIARY",
        "HAS_SUPPLIER",
    }


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

    assert [c.rel_type for c in first.candidates] == ["HAS_A", "HAS_A", "HAS_B", "HAS_B"]
    assert [c.edge_key for c in first.candidates] == ["e0", "e1", "e2", "e3"]
    assert [c.target_element_id for c in first.candidates] == [
        c.target_element_id for c in second.candidates
    ]
    assert first.mapping["e2"].rel_type == "HAS_B"
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
                        "target_id": f"t{i}",
                        "target_labels": ["Organization"],
                        "target_props": {"name": f"Supplier {i}"},
                    }
                    for i in range(params["rel_type_cap"])
                ],
                "source_labels": ["Organization", "Company"],
            },
            {
                "rel_type": "HAS_CEO",
                "total": 1,
                "kept": [
                    {
                        "rel_id": "rceo",
                        "rel_type": "HAS_CEO",
                        "rel_props": {},
                        "target_id": "tceo",
                        "target_labels": ["Person"],
                        "target_props": {"name": "Tim Cook"},
                    }
                ],
                "source_labels": ["Organization", "Company"],
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
                "source_labels": ["Organization"],
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


def test_get_node_neighborhood_maps_both_endpoints_and_skips_path_edges():
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


def test_settings_from_env_reads_neo4j_vars_without_typesafe_key(monkeypatch):
    for key in ("NEO4J_URL", "TYPESAFE_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("NEO4J_URI", "neo4j+s://demo.neo4jlabs.com:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "companies")
    monkeypatch.setenv("NEO4J_PASSWORD", "companies")
    monkeypatch.setenv("NEO4J_DATABASE", "companies")
    monkeypatch.setattr(neo4j_access, "load_dotenv", lambda *a, **k: None)

    settings = neo4j_access.settings_from_env()

    assert settings.neo4j_uri == "neo4j+s://demo.neo4jlabs.com:7687"
    assert settings.typesafe_api_key == ""


def test_settings_from_env_accepts_neo4j_url_alias_and_requires_uri(monkeypatch):
    monkeypatch.delenv("NEO4J_URI", raising=False)
    monkeypatch.setenv("NEO4J_URL", "neo4j+s://demo.neo4jlabs.com:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "companies")
    monkeypatch.setenv("NEO4J_PASSWORD", "companies")
    monkeypatch.setenv("NEO4J_DATABASE", "companies")
    monkeypatch.setattr(neo4j_access, "load_dotenv", lambda *a, **k: None)

    assert neo4j_access.settings_from_env().neo4j_uri.startswith("neo4j+s://")

    monkeypatch.delenv("NEO4J_URL", raising=False)
    with pytest.raises(ValueError, match="NEO4J_URI"):
        neo4j_access.settings_from_env()
