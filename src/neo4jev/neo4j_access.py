"""Read-only Neo4j access layer for the graph-navigation demo.

Everything that touches the target graph lives here: driver lifecycle, label and
index introspection, start-node lookup (exact / fulltext / vector) and the
neighbourhood fetches the navigator and the visualisation need.

This module also owns the per-label *identity map* (see :func:`resolve_display_value` and
:meth:`Neo4jAccess.display_properties`): which properties actually identify a node to a
human reader, derived from the live schema plus one TypeSafe call per label. Presentation
surfaces (viz, the app, notebooks) consult it instead of guessing from property values.

Nothing schema-specific is hardcoded: labels, relationship types and property
names are read from the live database (``SHOW INDEXES``, ``labels()``,
``type()``, ``keys()``).
"""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import neo4j
from typesafe_sdk import Choice, SystemOneResponse, TypeSafeClient, TypeSafeError

from neo4jev.config import Settings
from neo4jev.types import GraphSchema, NavCandidate, assign_edge_keys

LookupMode = Literal["exact", "fulltext", "vector"]
LOOKUP_MODES: tuple[LookupMode, ...] = ("exact", "fulltext", "vector")

# DriverError is a sibling of Neo4jError, not a parent: ServiceUnavailable (server unreachable)
# and ConfigurationError (bad URI) only descend from DriverError.
NEO4J_ERRORS = (neo4j.exceptions.Neo4jError, neo4j.exceptions.DriverError)

DEFAULT_SEARCH_LIMIT = 10
DEFAULT_REL_TYPE_CAP = 10
DEFAULT_TOTAL_CAP = 60
DEFAULT_NEIGHBORHOOD_LIMIT = 200

# Embedder contract: (text, expected_dimensions) -> vector of that length.
Embedder = Callable[[str, int], Sequence[float]]

# Identity / caption rules. A display property must be a short human-readable string; a
# caption used as a last resort must additionally be short enough that a description blob
# cannot win it (prose is longer than a name).
MAX_IDENTITY_VALUE_LEN = 120
MAX_FALLBACK_CAPTION_LEN = 48
MAX_IDENTITY_PROPERTIES = 3
IDENTITY_SAMPLE_SIZE = 25
IDENTITY_SAMPLES_PER_PROPERTY = 3
IDENTITY_QUESTION = "display_property"

_URI_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://")
# Lowercase only, deliberately: "yandex.com" is a host, but "E.ON", "Salesforce.com" and
# "St.Louis" are company/place names that happen to carry a dot.
_DOMAIN_LIKE_RE = re.compile(r"^(?:[a-z0-9-]+\.)+[a-z]{2,}(?:[/:?#].*)?$")
_OPAQUE_ID_RE = re.compile(r"^[0-9a-fA-F]{16,}$|^(?:[0-9a-fA-F]+-){3,}[0-9a-fA-F]+$")

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
    The per-type cap bounds each bucket first; the total cap then fills the budget by
    round-robin across types in name order, so a name-order-early type on a supernode
    cannot starve later types out of the result entirely.
    """
    capped: dict[str, list[NavCandidate]] = {}
    by_type_totals: dict[str, int] = {}
    for rel_type, (total, candidates) in sorted(per_type_edges.items()):
        by_type_totals[rel_type] = total
        capped[rel_type] = list(candidates)[:rel_type_cap]

    queues = {rel_type: list(candidates) for rel_type, candidates in capped.items()}
    selected: list[NavCandidate] = []
    while len(selected) < total_cap and any(queues.values()):
        for rel_type in sorted(queues):
            if len(selected) >= total_cap:
                break
            if queues[rel_type]:
                selected.append(queues[rel_type].pop(0))

    by_type_considered = {rel_type: 0 for rel_type in by_type_totals}
    for candidate in selected:
        by_type_considered[candidate.rel_type] = (
            by_type_considered.get(candidate.rel_type, 0) + 1
        )

    keyed = assign_edge_keys(selected)
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


def looks_like_reference(value: str) -> bool:
    """True for URIs, paths and opaque ids: they identify a node, but never a human.

    Covers ``scheme://…``, ``www.…``, bare host/path forms (``crunchbase.com/organization/x``,
    ``yandex.com``), long path-only segments and element-id/hash shapes — the property values
    that used to win the "longest string" caption heuristic. Dotted names such as ``E.ON`` or
    ``Salesforce.com`` are not references: a host is written in lowercase.
    """
    text = value.strip()
    if not text or any(char.isspace() for char in text):
        return False
    if _URI_SCHEME_RE.match(text) or text.lower().startswith("www."):
        return True
    if _DOMAIN_LIKE_RE.match(text):
        return True
    if text.count(":") >= 2:  # Neo4j element ids ("4:<uuid>:123") and other composite ids
        return True
    if _OPAQUE_ID_RE.match(text):
        return True
    return text.count("/") >= 2


def _short_identity_value(value: Any, max_len: int) -> str | None:
    """``value`` as a caption candidate, or None when it cannot identify anything."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or len(text) > max_len or looks_like_reference(text):
        return None
    return text


def fallback_display_value(
    props: Mapping[str, Any], *, max_len: int = MAX_FALLBACK_CAPTION_LEN
) -> str | None:
    """The longest short, non-reference string property, or None when there is none.

    Used only when the node carries none of its label's derived display properties; the
    length cap is what keeps a description blob or an article body out of a caption.
    """
    candidates = [
        text
        for raw in props.values()
        if (text := _short_identity_value(raw, max_len)) is not None
    ]
    return max(candidates, key=len) if candidates else None


_DISPLAY_PROPERTIES: dict[tuple[str, str], tuple[str, ...]] = {}


def remember_display_properties(
    label: str, properties: Sequence[str], *, database: str = ""
) -> None:
    """Record the derived display properties for ``label`` (keyed by database + label)."""
    _DISPLAY_PROPERTIES[(database, label)] = tuple(properties)


def display_properties_for(label: str, *, database: str | None = None) -> tuple[str, ...]:
    """Derived display properties for ``label``, or ``()`` when nothing was derived yet.

    Read-only by design: presentation code (viz, the app) knows a node's label but not the
    database it came from and must not trigger a query or an API call itself. Derivation is
    driven by :meth:`Neo4jAccess.display_properties`.
    """
    if database is not None:
        return _DISPLAY_PROPERTIES.get((database, label), ())
    matches = {
        properties for (_, cached_label), properties in _DISPLAY_PROPERTIES.items()
        if cached_label == label
    }
    # Two databases disagreeing about one label cannot be resolved from a label alone: a wrong
    # guess would caption a node with another schema's property names, so answer with nothing.
    return matches.pop() if len(matches) == 1 else ()


def clear_display_properties() -> None:
    """Forget every derived display property (tests, and reconnecting to another database)."""
    _DISPLAY_PROPERTIES.clear()


def resolve_display_value(props: Mapping[str, Any], label: str) -> str | None:
    """The value a human should read for a node, or None when it has no readable identity.

    A property derived as a display property for ``label`` wins, in the derived order; the
    result becomes a URI/too-long value only if the derived list is empty, and then only
    through the short, non-reference fallback.
    """
    for key in display_properties_for(label):
        value = _short_identity_value(props.get(key), MAX_IDENTITY_VALUE_LEN)
        if value is not None:
            return value
    return fallback_display_value(props)


@dataclass(frozen=True)
class LabelSchema:
    """Sampled schema of one label: the raw material the identity map is derived from."""

    label: str
    property_keys: tuple[str, ...] = ()
    property_types: dict[str, tuple[str, ...]] = field(default_factory=dict)
    sample_values: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    indexed_properties: tuple[str, ...] = ()
    sampled_nodes: int = 0

    @property
    def string_properties(self) -> tuple[str, ...]:
        return tuple(
            key for key in self.property_keys if self.property_types.get(key) == ("str",)
        )


def _samples_are_identity_like(values: Sequence[Any]) -> bool:
    # Majority, not all: one host-shaped or odd sample ("Yandex", "google.com") must not drop a
    # property that identifies most nodes of the label.
    if not values:
        return False
    readable = sum(
        1 for value in values if _short_identity_value(value, MAX_IDENTITY_VALUE_LEN) is not None
    )
    return readable * 2 >= len(values)


def _name_shaped(values: Sequence[Any]) -> bool:
    """A value shaped like a name rather than a code: one to four words."""
    for value in values:
        text = _short_identity_value(value, MAX_IDENTITY_VALUE_LEN)
        if text is not None and 1 <= len(text.split()) <= 4:
            return True
    return False


def identity_candidates(schema: LabelSchema) -> tuple[str, ...]:
    """Property keys that could plausibly identify a node to a human.

    String-only, and every sampled value short and non-reference — so URIs, embeddings,
    numbers, dates and long prose are gone before the model is even asked.
    """
    return tuple(
        key
        for key in schema.string_properties
        if _samples_are_identity_like(schema.sample_values.get(key, ()))
    )


def local_identity_order(schema: LabelSchema) -> tuple[str, ...]:
    """Deterministic ranking of :func:`identity_candidates`: indexed, name-shaped, alphabetical.

    This is both the option order offered to the model and the fallback used when the TypeSafe
    call is unavailable or fails.
    """
    indexed = set(schema.indexed_properties)
    return tuple(
        sorted(
            identity_candidates(schema),
            key=lambda key: (
                0 if key in indexed else 1,
                0 if _name_shaped(schema.sample_values.get(key, ())) else 1,
                key,
            ),
        )
    )


class IdentityClient(Protocol):
    """The sync TypeSafe surface the identity derivation needs (one call per label)."""

    def system_one(
        self, state: Any, questions: Mapping[str, Any], **kwargs: Any
    ) -> SystemOneResponse: ...


def _identity_state(schema: LabelSchema, candidates: Sequence[str]) -> dict[str, Any]:
    return {
        "label": schema.label,
        "sampled_nodes": schema.sampled_nodes,
        "candidate_properties": list(candidates),
    }


def _identity_choice(schema: LabelSchema, candidates: Sequence[str]) -> Choice:
    # Option keys are the property names themselves: unlike relationship types, property keys
    # on one label are unique, so nothing can collapse into a single option.
    return Choice(
        instructions=(
            f"Below are the candidate properties of the node label '{schema.label}'. Select every "
            "property that helps a human reader identify a single node of this label, most "
            "identifying first. Prefer short human-readable name/title-like string properties. "
            "Never select URLs or URIs, path-like references, image or document links, embeddings, "
            "numbers, dates, booleans, or long free text."
        ),
        criteria={
            key: {
                "sample_values": [
                    str(value)[:MAX_IDENTITY_VALUE_LEN]
                    for value in schema.sample_values.get(key, ())
                ],
                "indexed": key in schema.indexed_properties,
            }
            for key in candidates
        },
    )


def _default_identity_client() -> IdentityClient | None:
    """A sync TypeSafe client when a key is configured, else None (local ranking only)."""
    try:
        api_key = Settings.from_env(require_typesafe_key=False).typesafe_api_key.strip()
    except ValueError:
        return None
    if not api_key:
        return None
    try:
        return TypeSafeClient(api_key=api_key)
    except TypeSafeError:
        return None


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
    direction: str = "out",
) -> NavCandidate:
    # The query returns the edge's real start/end nodes, so both endpoints are
    # always known regardless of which direction the caller fetched in.
    end_labels = tuple(edge.get("end_labels") or edge.get("target_labels") or ())
    start_labels = tuple(edge.get("start_labels") or ())
    return NavCandidate(
        edge_key="",
        rel_element_id=edge["rel_id"],
        rel_type=edge["rel_type"],
        rel_props=dict(edge["rel_props"] or {}),
        target_element_id=edge.get("end_id", edge.get("target_id", "")),
        target_label=end_labels[0] if end_labels else "",
        target_props=dict(edge.get("end_props") or edge.get("target_props") or {}),
        source_element_id=edge.get("start_id", source_element_id),
        source_label=start_labels[0] if start_labels else source_label,
        direction=direction,
    )


@contextmanager
def open_access(
    settings: Settings | None = None,
    *,
    embedder: Embedder | None = None,
) -> Iterator["Neo4jAccess"]:
    """Open a driver from settings (or the environment) and yield an access layer.

    Falls back to ``Settings.from_env(require_typesafe_key=False)``: this layer never
    calls the TypeSafe API, so a missing key must not block graph access.
    """
    settings = settings or Settings.from_env(require_typesafe_key=False)
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

    def graph_schema(self) -> GraphSchema:
        """Live topology: labels, relationship types, and (from, type, to) triples.

        ``db.schema.visualization()`` is derived from live data, so nothing is hardcoded;
        a deployment without the procedure falls back to labels + types only.
        """
        labels = self.list_labels()
        type_records = self._run(
            "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
        )
        rel_types = tuple(sorted(record["relationshipType"] for record in type_records))
        try:
            viz = self._run(
                "CALL db.schema.visualization() YIELD nodes, relationships "
                "RETURN nodes, relationships"
            )
        except NEO4J_ERRORS:
            return GraphSchema(labels=labels, relationship_types=rel_types, relationships=())
        node_labels = {
            record_id: next(iter(labels_of), "")
            for row in viz
            for node in row["nodes"]
            for record_id, labels_of in [(node.element_id, node.labels)]
        }
        triples = {
            (node_labels.get(rel.start_node.element_id, ""), rel.type, node_labels.get(rel.end_node.element_id, ""))
            for row in viz
            for rel in row["relationships"]
        }
        return GraphSchema(
            labels=labels,
            relationship_types=rel_types,
            relationships=tuple(sorted(t for t in triples if all(t))),
        )

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

    def describe_label(self, label: str, *, sample: int = IDENTITY_SAMPLE_SIZE) -> LabelSchema:
        """Sampled schema of one label: property keys, value types, samples, indexed properties.

        Sampling (rather than a full scan) keeps this cheap on the 500k-node labels while still
        showing which properties are strings, which are vectors/numbers/dates, and what the
        values actually look like.
        """
        records = self._run(
            f"""
            MATCH (n:{_quote_label(label)})
            WITH n LIMIT $sample
            RETURN collect(properties(n)) AS property_maps, count(n) AS sampled
            """,
            sample=sample,
        )
        property_maps = list(records[0]["property_maps"] or []) if records else []
        sampled = int(records[0]["sampled"] or 0) if records else 0

        types: dict[str, set[str]] = {}
        samples: dict[str, list[Any]] = {}
        for properties in property_maps:
            for key, value in properties.items():
                key = str(key)
                types.setdefault(key, set()).add(type(value).__name__)
                bucket = samples.setdefault(key, [])
                if len(bucket) < IDENTITY_SAMPLES_PER_PROPERTY:
                    bucket.append(value)

        indexes = self.detect_indexes(label)
        indexed = tuple(
            sorted(
                {
                    property_name
                    for index in (*indexes.fulltext, *indexes.vector)
                    for property_name in index.properties
                }
            )
        )
        return LabelSchema(
            label=label,
            property_keys=tuple(sorted(types)),
            property_types={key: tuple(sorted(names)) for key, names in types.items()},
            sample_values={key: tuple(values) for key, values in samples.items()},
            indexed_properties=indexed,
            sampled_nodes=sampled,
        )

    def display_properties(
        self, label: str, *, client: IdentityClient | None = None
    ) -> tuple[str, ...]:
        """The properties that identify a node of ``label`` to a human reader (cached).

        Derived once per label: live schema introspection narrows the string properties, then a
        single TypeSafe call ranks them. Without a key (or when the call fails) the deterministic
        local ranking is used, so captions degrade but never break.
        """
        cached = _DISPLAY_PROPERTIES.get((self._database, label))
        if cached is not None:
            return cached
        schema = self.describe_label(label)
        candidates = local_identity_order(schema)
        derived = (
            self._select_display_properties(schema, candidates, client) if candidates else ()
        )
        remember_display_properties(label, derived, database=self._database)
        return derived

    def derive_display_properties(
        self, labels: Iterable[str], *, client: IdentityClient | None = None
    ) -> dict[str, tuple[str, ...]]:
        """Resolve display properties for several labels with one shared client."""
        resolved = client or _default_identity_client()
        return {
            label: self.display_properties(label, client=resolved) for label in labels
        }

    def derive_candidate_labels(
        self, candidates: Iterable[NavCandidate], *, client: IdentityClient | None = None
    ) -> None:
        """Derive the display properties of every label a hop is about to offer the model.

        A fetcher calls this on the candidates it just fetched, so the ``Choice`` criteria and
        the node state that follow already know each target's identity properties. Identity is
        presentation data: a graph error ends the derivation instead of failing the navigation
        that asked for it, and an already-derived label (empty result included) costs nothing.
        """
        labels = sorted({candidate.next_label for candidate in candidates} - {""})
        if not labels:
            return
        resolved = client or _default_identity_client()
        for label in labels:
            try:
                self.display_properties(label, client=resolved)
            except NEO4J_ERRORS:
                return

    def _select_display_properties(
        self,
        schema: LabelSchema,
        candidates: Sequence[str],
        client: IdentityClient | None,
    ) -> tuple[str, ...]:
        # `candidates` arrives in local_identity_order: the best local guess is also the first
        # option the model sees, and the stand-in when the call is unavailable or fails.
        ranked = tuple(candidates[:MAX_IDENTITY_PROPERTIES])
        client = client or _default_identity_client()
        if client is None:
            return ranked
        try:
            response = client.system_one(
                _identity_state(schema, candidates), {IDENTITY_QUESTION: _identity_choice(schema, candidates)}
            )
        except TypeSafeError:
            # A caption must never be the reason a page fails; the local ranking stands in.
            return ranked
        answer = response.choices.get(IDENTITY_QUESTION)
        if answer is None:
            return ranked
        candidate_set = set(candidates)
        ranked_options = sorted(
            answer.probabilities.items(), key=lambda item: item[1], reverse=True
        )
        picked = tuple(
            option
            for option, probability in ranked_options
            if probability > 0 and option in candidate_set
        )
        return picked[:MAX_IDENTITY_PROPERTIES] or ranked

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
        direction: str = "out",
    ) -> OutgoingCandidates:
        """Capped relationships from ``node_id``, in the requested direction.

        ``direction`` is "out" (edges leaving the node, the default), "in" (edges
        pointing at it — e.g. `Article -[:HAS_CHUNK]-> Chunk`), or "both". For
        incoming edges the candidate's ``source_*`` fields hold the far endpoint and
        ``direction`` is "in", so the navigator's ``next_element_id`` is the node the
        edge comes from.
        """
        if direction not in ("out", "in", "both"):
            raise ValueError(f"direction must be 'out', 'in' or 'both', got {direction!r}")
        # The per-type cap is applied in Cypher too, purely to bound how much data
        # a supernode ships over the wire; cap_outgoing_edges re-applies the same
        # rule and remains the single source of truth for the result shape.
        direction_clause = "(n)-[r]->(t)" if direction == "out" else "(n)<-[r]-(t)" if direction == "in" else "(n)-[r]-(t)"
        # For incoming edges the far endpoint (where following the edge would land) is
        # startNode(r), so the query returns the edge's real start/end explicitly rather
        # than aliasing `t` as the target.
        records = self._run(
            f"""
            MATCH (n) WHERE elementId(n) = $node_id
            MATCH {direction_clause}
            WITH type(r) AS rel_type, r, labels(n) AS node_labels,
                 startNode(r) AS edge_start, endNode(r) AS edge_end
            ORDER BY elementId(r)
            WITH rel_type, node_labels,
                 collect({{
                     rel_id: elementId(r),
                     rel_type: type(r),
                     rel_props: properties(r),
                     start_id: elementId(edge_start),
                     start_labels: labels(edge_start),
                     start_props: properties(edge_start),
                     end_id: elementId(edge_end),
                     end_labels: labels(edge_end),
                     end_props: properties(edge_end)
                 }}) AS edges
            RETURN rel_type, size(edges) AS total, edges[0..$rel_type_cap] AS kept,
                   node_labels
            """,
            node_id=node_id,
            rel_type_cap=rel_type_cap,
        )

        per_type: dict[str, tuple[int, list[NavCandidate]]] = {}
        source_labels: tuple[str, ...] = ()
        for record in records:
            source_labels = source_labels or tuple(record["node_labels"] or ())
            per_type[record["rel_type"]] = (
                record["total"],
                [
                    _candidate_from_edge(
                        edge,
                        direction=direction if direction != "both" else "out",
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
        return assign_edge_keys(candidates)
