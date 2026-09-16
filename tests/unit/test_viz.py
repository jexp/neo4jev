from __future__ import annotations

from IPython.display import HTML
from neo4j_viz import GraphWidget, Node, Relationship

from neo4jev.types import NavCandidate, NavPath, NavResult, NavStep, TerminationReason
from neo4jev.viz import (
    NEIGHBORHOOD_COLOR,
    PATH_COLORS,
    ROLE_KEY,
    START_COLOR,
    build_visualization,
    render_for_notebook,
    render_for_streamlit,
)


def _candidate(
    edge_key: str,
    rel_id: str,
    rel_type: str,
    target_id: str,
    target_label: str,
    target_props: dict | None = None,
    source_id: str = "",
    source_label: str = "",
) -> NavCandidate:
    return NavCandidate(
        edge_key=edge_key,
        rel_element_id=rel_id,
        rel_type=rel_type,
        rel_props={"weight": 1.0},
        target_element_id=target_id,
        target_label=target_label,
        target_props=target_props or {},
        source_element_id=source_id,
        source_label=source_label,
    )


def _step(node_id: str, chosen: list[NavCandidate]) -> NavStep:
    return NavStep(
        node_id=node_id,
        chosen=chosen,
        probabilities={c.edge_key: 1.0 / len(chosen) for c in chosen},
        log_prob=-0.1,
        candidates=list(chosen),
        noul=0.4,
    )


def _two_path_result() -> NavResult:
    path_0 = NavPath(
        steps=[
            _step("n0", [_candidate("e0", "r0", "SUPPLIES", "n1", "Company", {"name": "Beta Corp"})]),
            _step("n1", [_candidate("e0", "r1", "MENTIONS", "n2", "Article", {"title": "Beta in the news"})]),
        ],
        cumulative_log_prob=-0.2,
        terminated_reason=TerminationReason.GOAL_REACHED,
    )
    path_1 = NavPath(
        steps=[_step("n0", [_candidate("e0", "r2", "COMPETES_WITH", "n3", "Company", {"name": "Gamma Inc"})])],
        cumulative_log_prob=-0.5,
        terminated_reason=TerminationReason.MAX_DEPTH,
    )
    neighborhood = [
        _candidate(
            "e9",
            "r3",
            "MENTIONS",
            "n4",
            "Organization",
            {"name": "Delta"},
            source_id="n1",
            source_label="Company",
        ),
        _candidate("e9", "r4", "RELATED", "n5", "Topic", {"name": "Unresolved"}, source_id=""),
    ]
    return NavResult(start_element_id="n0", paths=[path_0, path_1], neighborhood=neighborhood)


def _node(graph, node_id: str) -> Node:
    return next(node for node in graph.nodes if node.id == node_id)


def _relationship(graph, rel_id: str) -> Relationship:
    return next(rel for rel in graph.relationships if rel.id == rel_id)


def _hex(color) -> str:
    return color.as_hex(format="long").lower()


def test_node_and_relationship_counts_cover_paths_and_neighborhood():
    graph = build_visualization(_two_path_result())

    # n5's candidate has no resolvable source among the path nodes, so neither it nor
    # its edge renders; n4 (sourced from path node n1) renders as neighborhood context.
    assert {node.id for node in graph.nodes} == {"n0", "n1", "n2", "n3", "n4"}
    assert {rel.id for rel in graph.relationships} == {"r0", "r1", "r2", "r3"}


def test_path_entities_are_colored_distinctly_from_neighborhood_entities():
    graph = build_visualization(_two_path_result())

    assert _hex(_node(graph, "n1").color) == PATH_COLORS[0].lower()
    assert _hex(_node(graph, "n2").color) == PATH_COLORS[0].lower()
    assert _hex(_node(graph, "n3").color) == PATH_COLORS[1].lower()

    # Neighborhood nodes fall back to the by-label default coloring.
    n4_label_color = _hex(_node(graph, "n4").color)
    n2_label = _node(graph, "n2").properties.get("label")
    assert n4_label_color != PATH_COLORS[0].lower()
    assert n4_label_color != START_COLOR.lower()

    assert _hex(_relationship(graph, "r0").color) == PATH_COLORS[0].lower()
    assert _hex(_relationship(graph, "r2").color) == PATH_COLORS[1].lower()
    assert _hex(_relationship(graph, "r3").color) == NEIGHBORHOOD_COLOR.lower()


def test_second_path_is_visually_distinguishable_from_the_first():
    graph = build_visualization(_two_path_result())

    assert _hex(_node(graph, "n1").color) != _hex(_node(graph, "n3").color)
    assert _hex(_relationship(graph, "r1").color) != _hex(_relationship(graph, "r2").color)


def test_start_node_and_path_nodes_differ_and_paths_are_sized_above_neighborhood():
    graph = build_visualization(_two_path_result())

    assert _hex(_node(graph, "n0").color) == START_COLOR.lower()
    assert _hex(_node(graph, "n0").color) != _hex(_node(graph, "n1").color)
    assert _node(graph, "n1").size > _node(graph, "n4").size
    assert _node(graph, "n0").size > _node(graph, "n4").size


def test_shared_node_keeps_the_lowest_path_color():
    shared = _candidate("e0", "r0", "KNOWS", "nX", "Company", {"name": "Shared"})
    result = NavResult(
        start_element_id="n0",
        paths=[
            NavPath([_step("n0", [shared])], -0.1, TerminationReason.GOAL_REACHED),
            NavPath([_step("n0", [shared])], -0.2, TerminationReason.GOAL_REACHED),
        ],
    )

    graph = build_visualization(result)

    assert len(graph.relationships) == 1
    assert _hex(_node(graph, "nX").color) == PATH_COLORS[0].lower()
    assert _hex(_relationship(graph, "r0").color) == PATH_COLORS[0].lower()


def test_color_nodes_metadata_and_captions_do_not_leak_into_the_graph():
    graph = build_visualization(_two_path_result())

    assert all(ROLE_KEY not in node.properties for node in graph.nodes)
    assert all(ROLE_KEY not in rel.properties for rel in graph.relationships)

    assert _node(graph, "n1").caption == "Beta Corp"
    assert _node(graph, "n2").caption == "Beta in the news"
    assert _node(graph, "n4").caption == "Delta"
    assert _relationship(graph, "r0").caption == "SUPPLIES"


def test_start_node_falls_back_to_its_element_id_as_caption():
    graph = build_visualization(_two_path_result())

    assert _node(graph, "n0").caption == "n0"


def test_node_seen_first_as_a_source_still_picks_up_properties_seen_later():
    # n1 is a path node (source of a chosen edge) so its neighborhood edges render.
    path = NavPath(
        [_step("n1", [_candidate("e0", "r2", "MENTIONS", "n7", "Article", {"title": "N1 News"}, source_id="n1")])],
        -0.1,
        TerminationReason.GOAL_REACHED,
    )
    result = NavResult(
        start_element_id="n0",
        paths=[path],
        neighborhood=[
            _candidate("e0", "r0", "MENTIONS", "n6", "Article", source_id="n4", source_label="Topic"),
            _candidate("e1", "r1", "RELATED", "n4", "Topic", {"name": "Late Props"}, source_id="n1"),
        ],
    )

    graph = build_visualization(result)

    # n4 appears (as a neighbor of path node n1) and picks up its properties;
    # n6 is sourced from n4, which the traversal never reached, so it does not render.
    assert _node(graph, "n4").caption == "Late Props"
    assert _node(graph, "n4").properties["name"] == "Late Props"
    assert {node.id for node in graph.nodes} == {"n0", "n1", "n7", "n4"}


def test_path_step_without_node_id_uses_the_candidate_source_instead_of_dangling():
    result = NavResult(
        start_element_id="n0",
        paths=[
            NavPath(
                [
                    _step(
                        "",
                        [_candidate("e0", "r0", "KNOWS", "n1", "Company", source_id="n9", source_label="Company")],
                    )
                ],
                -0.1,
                TerminationReason.GOAL_REACHED,
            )
        ],
    )

    graph = build_visualization(result)

    assert _relationship(graph, "r0").source == "n9"
    assert {node.id for node in graph.nodes} == {"n0", "n9", "n1"}


def test_zero_length_path_contributes_no_nodes_and_no_empty_legend_entry():
    result = NavResult(
        start_element_id="n0",
        paths=[NavPath([], 0.0, TerminationReason.NO_CANDIDATES)],
        neighborhood=[],
    )

    graph = build_visualization(result)

    assert [node.id for node in graph.nodes] == ["n0"]
    assert [entry.label for entry in graph.legend.nodes.entries] == ["Start"]


def test_legend_lists_every_style_that_appears_in_the_graph():
    graph = build_visualization(_two_path_result())

    node_legend = {entry.label: entry.color.lower() for entry in graph.legend.nodes.entries}
    rel_legend = {entry.label: entry.color.lower() for entry in graph.legend.relationships.entries}

    # Overlay roles (start + paths) appear in the legend; neighborhood falls back to
    # the by-label default coloring, which neo4j-viz renders without a legend entry.
    assert node_legend == {
        "Start": START_COLOR.lower(),
        "Path 1": PATH_COLORS[0].lower(),
        "Path 2": PATH_COLORS[1].lower(),
        "Neighborhood": NEIGHBORHOOD_COLOR.lower(),
    }
    assert rel_legend == {
        "Start": START_COLOR.lower(),
        "Path 1": PATH_COLORS[0].lower(),
        "Path 2": PATH_COLORS[1].lower(),
        "Neighborhood": NEIGHBORHOOD_COLOR.lower(),
    }
    assert graph.legend.visible is True


def test_number_of_path_colors_does_not_cap_the_number_of_distinct_paths():
    graph = build_visualization(_n_single_hop_paths(8))
    colors = {_hex(_node(graph, f"n{index + 1}").color) for index in range(8)}

    assert len(colors) == 8
    assert colors.isdisjoint({NEIGHBORHOOD_COLOR.lower(), START_COLOR.lower()})


def _n_single_hop_paths(count: int) -> NavResult:
    return NavResult(
        start_element_id="n0",
        paths=[
            NavPath(
                [_step("n0", [_candidate("e0", f"r{index}", "KNOWS", f"n{index + 1}", "Company")])],
                -0.1,
                TerminationReason.GOAL_REACHED,
            )
            for index in range(count)
        ],
    )


def test_empty_result_yields_only_the_start_node():
    graph = build_visualization(NavResult(start_element_id="n0", paths=[], neighborhood=[]))

    assert [node.id for node in graph.nodes] == ["n0"]
    assert graph.relationships == []
    assert [entry.label for entry in graph.legend.nodes.entries] == ["Start"]


def test_render_for_notebook_returns_html():
    rendered = render_for_notebook(_two_path_result(), height="300px")

    assert isinstance(rendered, HTML)
    assert "n0" in rendered.data


def test_render_for_streamlit_returns_a_graph_widget():
    widget = render_for_streamlit(_two_path_result())

    assert isinstance(widget, GraphWidget)
    assert {node.id for node in widget.nodes} == {"n0", "n1", "n2", "n3", "n4"}
