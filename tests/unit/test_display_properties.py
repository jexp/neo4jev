"""Unit tests for the per-label identity map: schema sampling, the one-call TypeSafe
derivation, and the display/fallback resolution the caption sites share.

No network: the driver is faked and the TypeSafe client is scripted.
"""

from __future__ import annotations

import neo4j
import pytest
from typesafe_sdk import ChoiceAnswer, SystemOneResponse, TypeSafeError, Usage

from neo4jev import neo4j_access
from neo4jev.neo4j_access import (
    IDENTITY_QUESTION,
    MAX_FALLBACK_CAPTION_LEN,
    LabelSchema,
    Neo4jAccess,
    clear_display_properties,
    display_properties_for,
    fallback_display_value,
    identity_candidates,
    local_identity_order,
    looks_like_reference,
    remember_display_properties,
    resolve_display_value,
)
from neo4jev.types import NavCandidate

SHORT_NAME = "Apple Inc."
# 75 characters: long enough that the caption fallback must reject it, short enough that the
# model still gets to judge it as a candidate (it is the model's call, not a length rule).
DESCRIPTION = "Subsidiary of Microsoft, which is developing Skype and other services worldwide"
# Real prose (an article body or a full description) is filtered out before the model is asked.
PROSE = DESCRIPTION * 3
URI = "http://diffbot.com/entity/EIsFKrN_ZNLSWsvxdQfWutQ"


class FakeResult:
    def __init__(self, records):
        self.records = records


class FakeDriver:
    """Minimal stand-in for neo4j.Driver.execute_query."""

    def __init__(self, handler=None):
        self.handler = handler or (lambda query, params: [])
        self.calls: list[dict] = []

    def execute_query(self, query_, parameters_=None, routing_=None, database_=None, **kwargs):
        self.calls.append({"query": query_, "params": parameters_ or {}, "database": database_})
        return FakeResult(self.handler(query_, parameters_ or {}))


class FakeIdentityClient:
    """Sync TypeSafe stand-in returning scripted Choice probabilities for one label."""

    def __init__(self, probabilities=None, *, raises: bool = False) -> None:
        self.probabilities = probabilities or {}
        self.raises = raises
        self.calls: list[dict] = []

    def system_one(self, state, questions, **kwargs):
        self.calls.append({"state": state, "questions": questions, "kwargs": kwargs})
        if self.raises:
            raise TypeSafeError("model unavailable")
        answer = ChoiceAnswer(
            choice=next(iter(self.probabilities), None),
            confidence=1.0,
            probabilities=dict(self.probabilities),
        )
        return SystemOneResponse(model="fake", usage=Usage(), answers={IDENTITY_QUESTION: answer})


def schema(**overrides) -> LabelSchema:
    base: dict = {
        "label": "Organization",
        "property_keys": ("description", "embedding", "fullName", "name", "stockSymbol", "uri"),
        "property_types": {
            "name": ("str",),
            "fullName": ("str",),
            "stockSymbol": ("str",),
            "description": ("str",),
            "uri": ("str",),
            "embedding": ("list",),
        },
        "sample_values": {
            "name": ("YouTube", "Yandex HQ"),
            "fullName": ("YouTube, Inc.", "Nebius Group N.V."),
            "stockSymbol": ("GOOG", "NBIS"),
            "description": (PROSE,),
            "uri": (URI,),
            "embedding": ([0.1] * 4,),
        },
        "indexed_properties": ("fullName",),
        "sampled_nodes": 25,
    }
    return LabelSchema(**{**base, **overrides})


# One sampled Organization node, as describe_label would aggregate it.
ORG_NODE = {
    "name": "Apple",
    "fullName": SHORT_NAME,
    "stockSymbol": "AAPL",
    "description": PROSE,
    "uri": URI,
    "embedding": [0.1] * 4,
}


def handler_with_schema(property_maps, indexes=None):
    def handler(query, params):
        if "SHOW INDEXES" in query:
            return indexes or [
                {
                    "name": "organization_fullName",
                    "type": "FULLTEXT",
                    "properties": ["fullName"],
                    "options": {},
                }
            ]
        return [{"property_maps": property_maps, "sampled": len(property_maps)}]

    return handler


@pytest.fixture(autouse=True)
def _clean_identity_map():
    clear_display_properties()
    yield
    clear_display_properties()


# --------------------------------------------------------------------------
# reference detection (the bit that keeps URIs out of captions)
# --------------------------------------------------------------------------


def test_looks_like_reference_catches_uris_domains_and_paths():
    assert looks_like_reference(URI) is True
    assert looks_like_reference("https://www.example.com/a/b") is True
    assert looks_like_reference("www.example.com") is True
    assert looks_like_reference("yandex.com") is True
    assert looks_like_reference("crunchbase.com/organization/youtube") is True
    assert looks_like_reference("en.wikipedia.org/wiki/YouTube") is True
    assert looks_like_reference("one/two/three") is True


def test_looks_like_reference_catches_opaque_ids():
    assert looks_like_reference("4:14b688aa-3906-4af7-9f45-70b8a1c2d3e4:6431") is True
    assert looks_like_reference("9f4d2c1b8a7e6d5f4c3b2a1908f7e6d5") is True
    assert looks_like_reference("4:…:1742") is True


def test_looks_like_reference_leaves_names_and_short_codes_alone():
    assert looks_like_reference("Apple Inc.") is False
    assert looks_like_reference("Nebius Group N.V.") is False
    assert looks_like_reference("Insider Monkey") is False
    assert looks_like_reference("8-K") is False
    assert looks_like_reference("GOOG") is False
    assert looks_like_reference("") is False


def test_looks_like_reference_keeps_dotted_names_that_are_not_hosts():
    # Real companies/places in the live graph; a host is written in lowercase.
    for name in ("E.ON", "Salesforce.com", "St.Louis", "L.L.Bean"):
        assert looks_like_reference(name) is False
    assert looks_like_reference("yandex.com") is True
    assert looks_like_reference("google.com") is True


def test_looks_like_reference_treats_mixed_case_hosts_as_names():
    # Deliberate consequence of the lowercase rule: a title-cased host is read as a name,
    # because rejecting it would also reject "Salesforce.com" and "St.Louis".
    assert looks_like_reference("Yandex.COM") is False


# --------------------------------------------------------------------------
# candidate selection and local ranking
# --------------------------------------------------------------------------


def test_identity_candidates_drop_uris_embeddings_and_long_text():
    assert identity_candidates(schema()) == ("fullName", "name", "stockSymbol")


def test_identity_candidates_require_a_string_only_property():
    mixed = schema(
        property_keys=("name", "mixed"),
        property_types={"name": ("str",), "mixed": ("str", "NoneType")},
        sample_values={"name": ("Apple",), "mixed": ("Apple",)},
    )
    assert identity_candidates(mixed) == ("name",)


def test_identity_candidates_keep_a_property_whose_samples_are_mostly_readable():
    # One host-shaped sample must not drop the property that names every other node.
    mostly_readable = schema(
        property_keys=("name", "homepageUri"),
        property_types={"name": ("str",), "homepageUri": ("str",)},
        sample_values={"name": ("Yandex", "google.com"), "homepageUri": ("yandex.com", "google.com")},
        indexed_properties=(),
    )

    assert identity_candidates(mostly_readable) == ("name",)


def test_local_identity_order_prefers_indexed_then_name_shaped():
    ordered = local_identity_order(schema())
    # fullName is the indexed property; name is name-shaped; stockSymbol is neither.
    assert ordered == ("fullName", "name", "stockSymbol")


def test_local_identity_order_is_deterministic_for_equal_ranks():
    first = local_identity_order(schema(indexed_properties=()))
    second = local_identity_order(schema(indexed_properties=()))

    assert first == second == ("fullName", "name", "stockSymbol")


# --------------------------------------------------------------------------
# describe_label
# --------------------------------------------------------------------------


def test_describe_label_collects_keys_types_samples_and_indexed_properties():
    access = Neo4jAccess(
        FakeDriver(
            handler_with_schema(
                [
                    {"fullName": "Apple Inc.", "name": "Apple"},
                    {"fullName": "Google LLC", "name": "Google", "employees": 100},
                ]
            )
        ),
        database="companies2",
    )

    described = access.describe_label("Organization")

    assert described.label == "Organization"
    assert described.property_keys == ("employees", "fullName", "name")
    assert described.property_types == {
        "fullName": ("str",),
        "name": ("str",),
        "employees": ("int",),
    }
    assert described.sample_values["fullName"] == ("Apple Inc.", "Google LLC")
    assert described.indexed_properties == ("fullName",)
    assert described.sampled_nodes == 2
    assert described.string_properties == ("fullName", "name")


# --------------------------------------------------------------------------
# display_properties (the one TypeSafe call per label, cached)
# --------------------------------------------------------------------------


def test_display_properties_asks_once_and_keeps_the_model_order():
    client = FakeIdentityClient({"name": 0.7, "fullName": 0.3, "stockSymbol": 0.0})
    access = Neo4jAccess(FakeDriver(handler_with_schema([ORG_NODE])), database="db")

    derived = access.display_properties("Organization", client=client)

    assert derived == ("name", "fullName")
    assert len(client.calls) == 1
    criteria = client.calls[0]["questions"][IDENTITY_QUESTION].criteria
    assert set(criteria) == {"fullName", "name", "stockSymbol"}
    assert client.calls[0]["state"]["label"] == "Organization"


def test_display_properties_ignores_hallucinated_and_zero_probability_options():
    client = FakeIdentityClient({"uri": 0.9, "name": 0.5, "fullName": 0.4, "stockSymbol": 0.0})
    access = Neo4jAccess(FakeDriver(handler_with_schema([ORG_NODE])), database="db")

    assert access.display_properties("Organization", client=client) == ("name", "fullName")


def test_display_properties_falls_back_to_the_local_order_without_a_client(monkeypatch):
    monkeypatch.setattr(neo4j_access, "_default_identity_client", lambda: None)
    access = Neo4jAccess(FakeDriver(handler_with_schema([ORG_NODE])), database="db")

    assert access.display_properties("Organization") == ("fullName", "name", "stockSymbol")


def test_display_properties_falls_back_when_the_model_call_fails():
    client = FakeIdentityClient({"name": 0.9}, raises=True)
    access = Neo4jAccess(FakeDriver(handler_with_schema([ORG_NODE])), database="db")

    assert access.display_properties("Organization", client=client) == (
        "fullName",
        "name",
        "stockSymbol",
    )


def test_display_properties_is_cached_per_database_and_label():
    client = FakeIdentityClient({"name": 0.9, "fullName": 0.1})
    driver = FakeDriver(handler_with_schema([ORG_NODE]))
    access = Neo4jAccess(driver, database="db")

    first = access.display_properties("Organization", client=client)
    second = access.display_properties("Organization", client=client)

    assert first == second == ("name", "fullName")
    assert len(client.calls) == 1
    assert display_properties_for("Organization") == ("name", "fullName")
    assert display_properties_for("Organization", database="db") == ("name", "fullName")
    # Nothing known for a label that was never derived.
    assert display_properties_for("Person") == ()

    other = Neo4jAccess(driver, database="other-db")
    other.display_properties("Organization", client=client)
    assert len(client.calls) == 2


def test_display_properties_for_answers_nothing_when_two_databases_disagree_on_a_label():
    remember_display_properties("Organization", ("name",), database="db")
    assert display_properties_for("Organization") == ("name",)

    remember_display_properties("Organization", ("fullName",), database="other-db")

    # A label alone cannot resolve two conflicting schemas; a wrong guess would caption a node
    # with another database's property names.
    assert display_properties_for("Organization") == ()
    assert display_properties_for("Organization", database="db") == ("name",)
    assert display_properties_for("Organization", database="other-db") == ("fullName",)
    # A named database that never derived the label is unknown, not "whatever another one said".
    assert display_properties_for("Organization", database="third-db") == ()


def test_display_properties_for_answers_when_databases_agree_on_a_label():
    remember_display_properties("Organization", ("name",), database="db")
    remember_display_properties("Organization", ("name",), database="other-db")

    assert display_properties_for("Organization") == ("name",)


def test_display_properties_derives_nothing_for_a_label_without_string_properties():
    client = FakeIdentityClient({"name": 1.0})
    access = Neo4jAccess(
        FakeDriver(handler_with_schema([{"embedding": [0.1, 0.2], "tokenSize": 512}])),
        database="db",
    )

    assert access.display_properties("Chunk", client=client) == ()
    assert client.calls == []


def test_derive_display_properties_resolves_every_label_with_one_client():
    client = FakeIdentityClient({"name": 1.0})
    access = Neo4jAccess(FakeDriver(handler_with_schema([{"name": "Apple"}])), database="db")

    derived = access.derive_display_properties(["Organization", "Person"], client=client)

    assert derived == {"Organization": ("name",), "Person": ("name",)}
    assert len(client.calls) == 2


def test_derive_candidate_labels_covers_every_hop_label_once():
    client = FakeIdentityClient({"name": 1.0})
    access = Neo4jAccess(FakeDriver(handler_with_schema([{"name": "Apple"}])), database="db")
    outgoing = NavCandidate(
        edge_key="e0", rel_element_id="r0", rel_type="MENTIONS", rel_props={},
        target_element_id="t0", target_label="Organization", target_props={},
    )
    incoming = NavCandidate(
        edge_key="e1", rel_element_id="r1", rel_type="HAS_CHUNK", rel_props={},
        target_element_id="c0", target_label="Chunk", target_props={},
        source_element_id="p1", source_label="Person", direction="in",
    )

    access.derive_candidate_labels([outgoing, incoming, outgoing], client=client)

    # One call per distinct next-label (Organization, Person), never for the current side.
    assert [call["state"]["label"] for call in client.calls] == ["Organization", "Person"]
    assert display_properties_for("Organization") == ("name",)
    assert display_properties_for("Person") == ("name",)

    # Already-derived labels are not derived twice.
    access.derive_candidate_labels([outgoing, incoming], client=client)
    assert len(client.calls) == 2


def test_derive_candidate_labels_builds_one_default_client_for_every_label(monkeypatch):
    client = FakeIdentityClient({"name": 1.0})
    built = []
    monkeypatch.setattr(
        neo4j_access, "_default_identity_client", lambda: built.append(client) or client
    )
    access = Neo4jAccess(FakeDriver(handler_with_schema([{"name": "Apple"}])), database="db")
    candidates = [
        NavCandidate(
            edge_key=f"e{i}", rel_element_id=f"r{i}", rel_type="MENTIONS", rel_props={},
            target_element_id=f"t{i}", target_label=label, target_props={},
        )
        for i, label in enumerate(("Organization", "Person", "City"))
    ]

    access.derive_candidate_labels(candidates)

    assert len(built) == 1, "a client per label would open a connection per label"
    assert len(client.calls) == 3


def test_derive_candidate_labels_never_fails_the_navigation_on_a_graph_error(monkeypatch):
    def exploding_handler(query, params):
        raise neo4j.exceptions.ServiceUnavailable("server gone")

    access = Neo4jAccess(FakeDriver(exploding_handler), database="db")
    outgoing = NavCandidate(
        edge_key="e0", rel_element_id="r0", rel_type="MENTIONS", rel_props={},
        target_element_id="t0", target_label="Organization", target_props={},
    )

    access.derive_candidate_labels([outgoing], client=FakeIdentityClient({"name": 1.0}))

    assert display_properties_for("Organization") == ()


def test_derive_candidate_labels_does_nothing_without_candidates():
    client = FakeIdentityClient({"name": 1.0})
    access = Neo4jAccess(FakeDriver(handler_with_schema([])), database="db")

    access.derive_candidate_labels([], client=client)

    assert client.calls == []


# --------------------------------------------------------------------------
# resolve_display_value (what the caption sites use)
# --------------------------------------------------------------------------


def test_resolve_display_value_prefers_the_derived_property_in_derived_order():
    remember_display_properties("Chunk", ("siteName", "language"), database="db")
    props = {"language": "en", "siteName": "Insider Monkey", "id": URI}

    assert resolve_display_value(props, "Chunk") == "Insider Monkey"


def test_resolve_display_value_skips_a_derived_property_missing_or_reference_on_this_node():
    remember_display_properties("Chunk", ("siteName", "language"), database="db")

    assert resolve_display_value({"language": "en"}, "Chunk") == "en"
    assert resolve_display_value({"siteName": "google.com", "language": "en"}, "Chunk") == "en"


@pytest.mark.parametrize(
    "props",
    [
        {"uri": URI},
        {"description": DESCRIPTION},
        {"uri": URI, "description": DESCRIPTION},
        {"embedding": [0.1, 0.2], "index": 3},
        {},
    ],
)
def test_resolve_display_value_returns_none_when_a_node_has_no_readable_identity(props):
    assert resolve_display_value(props, "Organization") is None


def test_resolve_display_value_returns_a_real_name_when_there_is_one():
    remember_display_properties("Organization", ("fullName", "name"), database="db")
    props = {"name": "Apple", "fullName": "Apple Inc.", "description": DESCRIPTION, "uri": URI}

    assert resolve_display_value(props, "Organization") == "Apple Inc."


def test_fallback_display_value_caps_the_description_length():
    assert len(DESCRIPTION) > MAX_FALLBACK_CAPTION_LEN
    assert fallback_display_value({"description": DESCRIPTION}) is None
    assert fallback_display_value({"description": DESCRIPTION, "name": "Apple"}) == "Apple"


def test_fallback_display_value_keeps_the_longest_readable_string():
    assert fallback_display_value({"a": "Apple", "b": "Apple Inc.", "c": URI}) == "Apple Inc."
