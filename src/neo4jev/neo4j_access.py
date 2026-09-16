"""Read-only Neo4j access layer for the graph-navigation demo.

Everything that touches the target graph lives here: driver lifecycle, label and
index introspection, start-node lookup (exact / fulltext / vector) and the
neighbourhood fetches the navigator and the visualisation need.

Nothing schema-specific is hardcoded: labels, relationship types and property
names are read from the live database (``SHOW INDEXES``, ``labels()``,
``type()``, ``keys()``).
"""

from __future__ import annotations

import hashlib
import os
import random
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import neo4j
from dotenv import load_dotenv

from neo4jev.config import Settings
from neo4jev.types import NavCandidate

LookupMode = Literal["exact", "fulltext", "vector"]
LOOKUP_MODES: tuple[LookupMode, ...] = ("exact", "fulltext", "vector")

DEFAULT_SEARCH_LIMIT = 10
DEFAULT_REL_TYPE_CAP = 10
DEFAULT_TOTAL_CAP = 60
DEFAULT_NEIGHBORHOOD_LIMIT = 200

# Embedder contract: (text, expected_dimensions) -> vector of that length.
Embedder = Callable[[str, int], Sequence[float]]

_LUCENE_SPECIAL = set('+-&|!(){}[]^"~*?\\:/')


@dataclass(frozen=True)
class GraphNode:
    """A node as returned by a lookup, with an optional lookup score."""

    element_id: str
    labels: tuple[str, ...]
    props: dict[str, Any]
    score: float | None = None

    @property
    def label(self) -> str:
        return self.labels[0] if self.labels else ""


@dataclass(frozen=True)
class IndexRef:
    name: str
    kind: str
    properties: tuple[str, ...]
    dimensions: int | None = None


@dataclass(frozen=True)
class LabelIndexes:
    """Which of the three lookup modes are usable for a label.

    ``exact`` is always available (property scan); ``fulltext``/``vector`` only
    when the database actually has a matching index for that label.
    """

    label: str
    fulltext: tuple[IndexRef, ...] = ()
    vector: tuple[IndexRef, ...] = ()

    @property
    def modes(self) -> tuple[LookupMode, ...]:
        modes: list[LookupMode] = ["exact"]
        if self.fulltext:
            modes.append("fulltext")
        if self.vector:
            modes.append("vector")
        return tuple(modes)

    def supports(self, mode: LookupMode) -> bool:
        return mode in self.modes


@dataclass(frozen=True)
class OutgoingCandidates:
    """Capped outgoing edges of one node, plus what was left out.

    Iterable/indexable over ``candidates`` so callers can treat it as a list.
    """

    source_element_id: str
    source_label: str
    candidates: list[NavCandidate] = field(default_factory=list)
    total_edges: int = 0
    by_type_totals: dict[str, int] = field(default_factory=dict)
    by_type_considered: dict[str, int] = field(default_factory=dict)

    @property
    def considered_edges(self) -> int:
        return len(self.candidates)

    @property
    def truncated(self) -> bool:
        return self.considered_edges < self.total_edges

    @property
    def note(self) -> str:
        return f"{self.considered_edges} of {self.total_edges} edges considered"

    @property
    def by_type_note(self) -> dict[str, str]:
        return {
            rel_type: f"{self.by_type_considered.get(rel_type, 0)} of {total} "
            f"{rel_type} edges considered"
            for rel_type, total in self.by_type_totals.items()
        }

    @property
    def mapping(self) -> dict[str, NavCandidate]:
        """edge_key -> candidate side table for the current hop."""
        return {candidate.edge_key: candidate for candidate in self.candidates}

    def __iter__(self) -> Iterator[NavCandidate]:
        return iter(self.candidates)

    def __len__(self) -> int:
        return len(self.candidates)

    def __getitem__(self, index: int) -> NavCandidate:
        return self.candidates[index]


def assign_edge_keys(
    candidates: Sequence[NavCandidate],
) -> tuple[list[NavCandidate], dict[str, NavCandidate]]:
    """Key candidates by synthetic opaque ids (``e0``, ``e1``, ...).

    Keying by relationship type would collapse several relationships of the same
    type to different targets into one option, so the ids stay positional.
    """
    keyed: list[NavCandidate] = []
    mapping: dict[str, NavCandidate] = {}
    for index, candidate in enumerate(candidates):
        edge_key = f"e{index}"
        keyed_candidate = replace(candidate, edge_key=edge_key)
        keyed.append(keyed_candidate)
        mapping[edge_key] = keyed_candidate
    return keyed, mapping


def cap_outgoing_edges(
    per_type_edges: Mapping[str, tuple[int, Sequence[NavCandidate]]],
    *,
    rel_type_cap: int = DEFAULT_REL_TYPE_CAP,
    total_cap: int = DEFAULT_TOTAL_CAP,
    source_element_id: str = "",
    source_label: str = "",
) -> OutgoingCandidates:
    """Apply the per-type and total caps and report the truncation.

    ``per_type_edges`` maps relationship type -> (true edge count, candidates).
    Relationship types are visited in name order so the result is deterministic;
    the total cap then trims from that ordered list.
    """
    selected: list[NavCandidate] = []
    by_type_totals: dict[str, int] = {}
    for rel_type, (total, candidates) in sorted(per_type_edges.items()):
        by_type_totals[rel_type] = total
        selected.extend(list(candidates)[:rel_type_cap])

    selected = selected[:total_cap]
    by_type_considered = {rel_type: 0 for rel_type in by_type_totals}
    for candidate in selected:
        by_type_considered[candidate.rel_type] = (
            by_type_considered.get(candidate.rel_type, 0) + 1
        )

    keyed, _ = assign_edge_keys(selected)
    return OutgoingCandidates(
        source_element_id=source_element_id,
        source_label=source_label,
        candidates=keyed,
        total_edges=sum(by_type_totals.values()),
        by_type_totals=by_type_totals,
        by_type_considered=by_type_considered,
    )


def escape_lucene(text: str) -> str:
    """Escape Lucene query syntax so raw user text is treated as literal text."""
    return "".join(f"\\{char}" if char in _LUCENE_SPECIAL else char for char in text)


def default_embedder(text: str, dimensions: int) -> list[float]:
    """Deterministic hash-seeded pseudo-embedding of the requested length.

    The demo has no embedding-provider credentials available (the server's
    ``genai`` plugin is unconfigured and no external provider key is supplied),
    so vector lookup works dimensionally but not semantically. Pass a real
    embedder to :class:`Neo4jAccess` to get meaningful neighbours.
    """
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)
    vector = [rng.uniform(-1.0, 1.0) for _ in range(dimensions)]
    norm = sum(value * value for value in vector) ** 0.5 or 1.0
    return [value / norm for value in vector]


def _quote_label(label: str) -> str:
    if not label:
        raise ValueError("label must be a non-empty string")
    return "`" + label.replace("`", "``") + "`"


def _optional(record: Mapping[str, Any], key: str) -> Any:
    try:
        return record[key]
    except (KeyError, IndexError):
        return None


def _node_from_record(record: Mapping[str, Any]) -> GraphNode:
    labels = tuple(record["labels"] or ())
    return GraphNode(
        element_id=record["element_id"],
        labels=labels,
        props=dict(record["props"] or {}),
        score=_optional(record, "score"),
    )


def _candidate_from_edge(
    edge: Mapping[str, Any],
    *,
    source_element_id: str = "",
    source_label: str = "",
) -> NavCandidate:
    target_labels = tuple(edge["target_labels"] or ())
    return NavCandidate(
        edge_key="",
        rel_element_id=edge["rel_id"],
        rel_type=edge["rel_type"],
        rel_props=dict(edge["rel_props"] or {}),
        target_element_id=edge["target_id"],
        target_label=target_labels[0] if target_labels else "",
        target_props=dict(edge["target_props"] or {}),
        source_element_id=source_element_id,
        source_label=source_label,
    )


def settings_from_env() -> Settings:
    """Neo4j-only settings built from the environment.

    Deliberately not ``Settings.from_env()``: this layer must not require a
    TypeSafe key, since index exploration and neighbourhood fetches never call
    the TypeSafe API.
    """
    load_dotenv()
    uri = os.environ.get("NEO4J_URI") or os.environ.get("NEO4J_URL")
    if not uri:
        raise ValueError(
            "Missing required environment variable: NEO4J_URI (or NEO4J_URL)"
        )
    return Settings(
        neo4j_uri=uri,
        neo4j_username=_require_env("NEO4J_USERNAME"),
        neo4j_password=_require_env("NEO4J_PASSWORD"),
        neo4j_database=_require_env("NEO4J_DATABASE"),
        typesafe_api_key=os.environ.get("TYPESAFE_API_KEY", ""),
    )


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


@contextmanager
def open_access(
    settings: Settings | None = None,
    *,
    embedder: Embedder | None = None,
) -> Iterator["Neo4jAccess"]:
    """Open a driver from settings (or the environment) and yield an access layer."""
    settings = settings or settings_from_env()
    driver = neo4j.GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )
    try:
        driver.verify_connectivity()
        yield Neo4jAccess(
            driver,
            database=settings.neo4j_database,
            embedder=embedder,
        )
    finally:
        driver.close()


class Neo4jAccess:
    """Read-only queries against one Neo4j database."""

    def __init__(
        self,
        driver: neo4j.Driver,
        *,
        database: str,
        embedder: Embedder | None = None,
    ) -> None:
        self._driver = driver
        self._database = database
        self._embedder = embedder or default_embedder

    @property
    def database(self) -> str:
        return self._database

    def _run(self, cypher: str, **parameters: Any) -> list[Mapping[str, Any]]:
        result = self._driver.execute_query(
            cypher,
            parameters_=parameters,
            routing_=neo4j.RoutingControl.READ,
            database_=self._database,
        )
        return list(result.records)

    def list_labels(self) -> tuple[str, ...]:
        records = self._run("CALL db.labels() YIELD label RETURN label")
        return tuple(sorted(record["label"] for record in records))

    def detect_indexes(self, label: str) -> LabelIndexes:
        records = self._run(
            """
            SHOW INDEXES YIELD name, type, labelsOrTypes, properties, options
            WHERE type IN ['FULLTEXT', 'VECTOR'] AND $label IN labelsOrTypes
            RETURN name, type, properties, options
            """,
            label=label,
        )
        fulltext: list[IndexRef] = []
        vector: list[IndexRef] = []
        for record in records:
            index_config = (record["options"] or {}).get("indexConfig") or {}
            ref = IndexRef(
                name=record["name"],
                kind=record["type"],
                properties=tuple(record["properties"] or ()),
                dimensions=index_config.get("vector.dimensions"),
            )
            (fulltext if record["type"] == "FULLTEXT" else vector).append(ref)
        return LabelIndexes(
            label=label,
            fulltext=tuple(sorted(fulltext, key=lambda ref: ref.name)),
            vector=tuple(sorted(vector, key=lambda ref: ref.name)),
        )

    def get_node(self, element_id: str) -> GraphNode | None:
        records = self._run(
            """
            MATCH (n) WHERE elementId(n) = $element_id
            RETURN elementId(n) AS element_id, labels(n) AS labels,
                   properties(n) AS props
            """,
            element_id=element_id,
        )
        return _node_from_record(records[0]) if records else None

    def search_start_nodes(
        self,
        label: str,
        query: str,
        mode: LookupMode = "exact",
        *,
        limit: int = DEFAULT_SEARCH_LIMIT,
        index_name: str | None = None,
    ) -> list[GraphNode]:
        if mode not in LOOKUP_MODES:
            raise ValueError(f"Unsupported lookup mode: {mode!r}")
        if not query.strip():
            return []
        if mode == "exact":
            if index_name is not None:
                raise ValueError("index_name applies to fulltext/vector lookup only")
            return self._search_exact(label, query, limit)
        if mode == "fulltext":
            return self._search_fulltext(label, query, limit, index_name)
        return self._search_vector(label, query, limit, index_name)

    def _search_exact(self, label: str, query: str, limit: int) -> list[GraphNode]:
        # No property name is assumed: every string-valued property of each node
        # of the label participates, with equality matches ranked first.
        records = self._run(
            f"""
            MATCH (n:{_quote_label(label)})
            WHERE any(k IN keys(n)
                      WHERE toStringOrNull(n[k]) IS NOT NULL
                        AND toLower(toStringOrNull(n[k])) CONTAINS toLower($query))
            WITH n, [k IN keys(n)
                     WHERE toLower(toStringOrNull(n[k])) = toLower($query)] AS exact_matches
            RETURN elementId(n) AS element_id, labels(n) AS labels,
                   properties(n) AS props
            ORDER BY size(exact_matches) DESC, elementId(n)
            LIMIT $limit
            """,
            query=query,
            limit=limit,
        )
        return [_node_from_record(record) for record in records]

    def _select_index(
        self, label: str, mode: Literal["fulltext", "vector"], index_name: str | None
    ) -> IndexRef:
        indexes = getattr(self.detect_indexes(label), mode)
        if not indexes:
            raise ValueError(f"No {mode} index available for label {label!r}")
        if index_name is None:
            return indexes[0]
        for ref in indexes:
            if ref.name == index_name:
                return ref
        raise ValueError(
            f"Index {index_name!r} is not a {mode} index for label {label!r}"
        )

    def _search_fulltext(
        self, label: str, query: str, limit: int, index_name: str | None
    ) -> list[GraphNode]:
        index = self._select_index(label, "fulltext", index_name)
        # As with the vector path, `limit` applies before the label filter, so a
        # multi-label index can yield fewer than `limit` rows.
        records = self._run(
            f"""
            CALL db.index.fulltext.queryNodes($index_name, $query, {{limit: {int(limit)}}})
            YIELD node, score
            WHERE $label IN labels(node)
            RETURN elementId(node) AS element_id, labels(node) AS labels,
                   properties(node) AS props, score
            """,
            index_name=index.name,
            query=escape_lucene(query),
            label=label,
        )
        return [_node_from_record(record) for record in records]

    def _search_vector(
        self, label: str, query: str, limit: int, index_name: str | None
    ) -> list[GraphNode]:
        index = self._select_index(label, "vector", index_name)
        if not index.dimensions:
            raise ValueError(
                f"Vector index {index.name!r} does not declare vector.dimensions"
            )
        embedding = list(self._embedder(query, int(index.dimensions)))
        if len(embedding) != int(index.dimensions):
            raise ValueError(
                f"Embedder returned {len(embedding)} dimensions, "
                f"index {index.name!r} expects {index.dimensions}"
            )
        # Label filtering happens after the ANN lookup, so a caller may get fewer
        # than `limit` rows when the index spans more than one label.
        # db.index.vector.queryNodes is the legacy-but-still-supported form; the
        # SEARCH clause only exists on 2026.x, and this demo also runs on 5.x.
        records = self._run(
            """
            CALL db.index.vector.queryNodes($index_name, $k, $embedding)
            YIELD node, score
            WHERE $label IN labels(node)
            RETURN elementId(node) AS element_id, labels(node) AS labels,
                   properties(node) AS props, score
            """,
            index_name=index.name,
            k=int(limit),
            embedding=embedding,
            label=label,
        )
        return [_node_from_record(record) for record in records]

    def get_outgoing_relationships(
        self,
        node_id: str,
        rel_type_cap: int = DEFAULT_REL_TYPE_CAP,
        total_cap: int = DEFAULT_TOTAL_CAP,
    ) -> OutgoingCandidates:
        # The per-type cap is applied in Cypher too, purely to bound how much data
        # a supernode ships over the wire; cap_outgoing_edges re-applies the same
        # rule and remains the single source of truth for the result shape.
        records = self._run(
            """
            MATCH (n) WHERE elementId(n) = $node_id
            MATCH (n)-[r]->(t)
            WITH type(r) AS rel_type, r, t, labels(n) AS source_labels
            ORDER BY elementId(r)
            WITH rel_type, source_labels,
                 collect({
                     rel_id: elementId(r),
                     rel_type: type(r),
                     rel_props: properties(r),
                     target_id: elementId(t),
                     target_labels: labels(t),
                     target_props: properties(t)
                 }) AS edges
            RETURN rel_type, size(edges) AS total, edges[0..$rel_type_cap] AS kept,
                   source_labels
            """,
            node_id=node_id,
            rel_type_cap=rel_type_cap,
        )

        per_type: dict[str, tuple[int, list[NavCandidate]]] = {}
        source_labels: tuple[str, ...] = ()
        for record in records:
            source_labels = source_labels or tuple(record["source_labels"] or ())
            per_type[record["rel_type"]] = (
                record["total"],
                [
                    _candidate_from_edge(
                        edge,
                        source_element_id=node_id,
                        source_label=source_labels[0] if source_labels else "",
                    )
                    for edge in record["kept"]
                ],
            )

        if not source_labels:
            # No edges means no row carrying labels(n), so look the node up rather
            # than reporting an empty source_label for exactly the leaf nodes.
            node = self.get_node(node_id)
            source_labels = node.labels if node else ()

        return cap_outgoing_edges(
            per_type,
            rel_type_cap=rel_type_cap,
            total_cap=total_cap,
            source_element_id=node_id,
            source_label=source_labels[0] if source_labels else "",
        )

    def get_node_neighborhood(
        self,
        node_ids: Sequence[str],
        *,
        limit: int = DEFAULT_NEIGHBORHOOD_LIMIT,
    ) -> list[NavCandidate]:
        """Edges connecting the given nodes to the rest of the graph.

        Edges whose both endpoints are in ``node_ids`` are excluded (the path
        itself already carries them). Each candidate keeps the relationship's
        real direction, so ``source_element_id``/``target_element_id`` mirror
        ``startNode(r)``/``endNode(r)``; ``NavCandidate`` has no source-property
        field, so only the target endpoint ships its properties.
        """
        if not node_ids:
            return []
        records = self._run(
            """
            MATCH (n) WHERE elementId(n) IN $node_ids
            MATCH (n)-[r]-(t)
            WHERE NOT elementId(t) IN $node_ids
            RETURN DISTINCT elementId(r) AS rel_id, type(r) AS rel_type,
                   properties(r) AS rel_props,
                   elementId(startNode(r)) AS source_id,
                   labels(startNode(r)) AS source_labels,
                   elementId(endNode(r)) AS target_id,
                   labels(endNode(r)) AS target_labels,
                   properties(endNode(r)) AS target_props
            ORDER BY rel_id
            LIMIT $limit
            """,
            node_ids=list(node_ids),
            limit=limit,
        )

        candidates: list[NavCandidate] = []
        for record in records:
            target_labels = tuple(record["target_labels"] or ())
            source_labels = tuple(record["source_labels"] or ())
            candidates.append(
                NavCandidate(
                    edge_key="",
                    rel_element_id=record["rel_id"],
                    rel_type=record["rel_type"],
                    rel_props=dict(record["rel_props"] or {}),
                    target_element_id=record["target_id"],
                    target_label=target_labels[0] if target_labels else "",
                    target_props=dict(record["target_props"] or {}),
                    source_element_id=record["source_id"],
                    source_label=source_labels[0] if source_labels else "",
                )
            )
        keyed, _ = assign_edge_keys(candidates)
        return keyed
