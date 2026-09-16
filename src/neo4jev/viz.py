"""Turn a :class:`~neo4jev.types.NavResult` into a styled ``neo4j-viz`` graph.

Path nodes/relationships are colored per path (one color per path index, all sharing a
single legend) while neighborhood-only entities get a muted neutral color, so the path(s)
taken stand out from their surrounding context.
"""

from __future__ import annotations

import colorsys
from typing import Any, Literal

from IPython.display import HTML
from neo4j_viz import GraphWidget, Layout, Node, Relationship, VisualizationGraph

from .types import NavCandidate, NavPath, NavResult

__all__ = ["build_visualization", "render_for_notebook", "render_for_streamlit"]

ROLE_KEY = "_neo4jev_role"
START_ROLE = "start"
NEIGHBORHOOD_ROLE = "neighborhood"
PATH_ROLE_PREFIX = "path_"

# Slots from neo4j-viz's own NEO4J_COLORS_DISCRETE palette (not a public export).
PATH_COLORS = ("#F79767", "#C990C0", "#56C7E4", "#8DCC93", "#DA7294", "#4D8DDA")
START_COLOR = "#FFC354"
NEIGHBORHOOD_COLOR = "#D8C7AE"

START_NODE_SIZE = 32.0
PATH_NODE_SIZE = 26.0
NEIGHBORHOOD_NODE_SIZE = 12.0
PATH_REL_WIDTH = 3.0
NEIGHBORHOOD_REL_WIDTH = 1.0

_MAX_CAPTION_LEN = 64
Theme = Literal["auto", "light", "dark"]


def _path_role(index: int) -> str:
    return f"{PATH_ROLE_PREFIX}{index}"


def _path_color(index: int) -> str:
    if index < len(PATH_COLORS):
        return PATH_COLORS[index]
    # Golden-ratio hue spacing keeps paths beyond the palette mutually distinguishable.
    red, green, blue = colorsys.hsv_to_rgb((index * 0.618034) % 1.0, 0.55, 0.95)
    return "#{:02X}{:02X}{:02X}".format(round(red * 255), round(green * 255), round(blue * 255))


def _role_color(role: str) -> str:
    if role == START_ROLE:
        return START_COLOR
    if role == NEIGHBORHOOD_ROLE:
        return NEIGHBORHOOD_COLOR
    return _path_color(int(role.removeprefix(PATH_ROLE_PREFIX)))


def _role_priority(role: str) -> int:
    if role == START_ROLE:
        return 0
    if role == NEIGHBORHOOD_ROLE:
        return 1_000_000
    return int(role.removeprefix(PATH_ROLE_PREFIX)) + 1


def _role_label(role: str) -> str:
    if role == START_ROLE:
        return "Start"
    if role == NEIGHBORHOOD_ROLE:
        return "Neighborhood"
    return f"Path {int(role.removeprefix(PATH_ROLE_PREFIX)) + 1}"


def _node_caption(label: str, props: dict[str, Any]) -> str:
    """Prefer the longest short string property (usually a name/title) over the raw label."""
    strings = [
        value
        for key, value in props.items()
        if key != ROLE_KEY and isinstance(value, str) and 0 < len(value) <= _MAX_CAPTION_LEN
    ]
    if strings:
        return max(strings, key=len)
    return label


def _node_size(role: str) -> float:
    return NEIGHBORHOOD_NODE_SIZE if role == NEIGHBORHOOD_ROLE else (START_NODE_SIZE if role == START_ROLE else PATH_NODE_SIZE)


def _caption_size(role: str) -> int:
    return 1 if role == NEIGHBORHOOD_ROLE else 2


class _GraphBuilder:
    """Accumulates viz entities, letting path entities win over neighborhood ones."""

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.relationships: dict[str, Relationship] = {}
        self._anonymous_rels = 0

    def add_node(self, element_id: str, label: str = "", props: dict[str, Any] | None = None, *, role: str) -> None:
        if not element_id:
            return

        props = props or {}
        existing = self.nodes.get(element_id)
        if existing is None:
            self.nodes[element_id] = Node(
                id=element_id,
                caption=_node_caption(label, props) or element_id,
                caption_size=_caption_size(role),
                size=_node_size(role),
                properties={**props, ROLE_KEY: role},
            )
            return

        if _role_priority(role) < _role_priority(existing.properties[ROLE_KEY]):
            existing.properties[ROLE_KEY] = role
            existing.size = _node_size(role)
            existing.caption_size = _caption_size(role)
        if props:
            # A node can first appear as a relationship source (label only) and only later
            # carry its properties as a relationship target: keep both.
            for key, value in props.items():
                existing.properties.setdefault(key, value)
            existing.caption = _node_caption(label, existing.properties) or existing.caption

    def add_candidate(self, candidate: NavCandidate, source_id: str, *, role: str) -> None:
        target_id = candidate.target_element_id
        self.add_node(source_id, candidate.source_label, role=role)
        self.add_node(target_id, candidate.target_label, candidate.target_props, role=role)
        if not source_id or not target_id:
            # No resolvable source: keep the target as context, skip the dangling edge.
            return

        rel_id = candidate.rel_element_id
        if not rel_id:
            self._anonymous_rels += 1
            rel_id = f"neo4jev-rel-{self._anonymous_rels}"
        if rel_id in self.relationships:
            return

        self.relationships[rel_id] = Relationship(
            id=rel_id,
            source=source_id,
            target=target_id,
            caption=candidate.rel_type,
            width=PATH_REL_WIDTH if role != NEIGHBORHOOD_ROLE else NEIGHBORHOOD_REL_WIDTH,
            properties={**candidate.rel_props, ROLE_KEY: role},
        )

    def add_path(self, path: NavPath, index: int) -> None:
        role = _path_role(index)
        for step in path.steps:
            self.add_node(step.node_id, role=role)
            for candidate in step.chosen:
                self.add_candidate(candidate, step.node_id or candidate.source_element_id, role=role)

    def add_neighborhood(self, result: NavResult) -> None:
        for candidate in result.neighborhood:
            self.add_candidate(candidate, candidate.source_element_id, role=NEIGHBORHOOD_ROLE)

    @staticmethod
    def _palette(roles: set[str]) -> dict[str, str]:
        return {role: _role_color(role) for role in sorted(roles, key=_role_priority)}

    @staticmethod
    def _legend(palette: dict[str, str]) -> dict[str, str]:
        return {_role_label(role): color for role, color in palette.items()}

    def to_graph(self) -> VisualizationGraph:
        graph = VisualizationGraph(nodes=list(self.nodes.values()), relationships=list(self.relationships.values()))

        node_roles = {node.properties[ROLE_KEY] for node in graph.nodes}
        rel_roles = {rel.properties[ROLE_KEY] for rel in graph.relationships}
        node_palette = self._palette(node_roles)
        rel_palette = self._palette(rel_roles)
        if node_roles:
            graph.color_nodes(property=ROLE_KEY, colors=node_palette)
        if rel_roles:
            graph.color_relationships(property=ROLE_KEY, colors=rel_palette)

        # The role is styling metadata, not graph data: drop it so the viz side panel
        # only ever shows real Neo4j properties (colors/legend above are already applied).
        for entity in (*graph.nodes, *graph.relationships):
            entity.properties.pop(ROLE_KEY, None)

        graph.set_legend(
            nodes=self._legend(node_palette) if node_roles else None,
            relationships=self._legend(rel_palette) if rel_roles else None,
            visible=True,
        )
        return graph


def build_visualization(result: NavResult) -> VisualizationGraph:
    """Build a :class:`~neo4j_viz.VisualizationGraph` for ``result``.

    Every path in ``result`` gets its own color (shared for its nodes and relationships);
    nodes/relationships reachable only through ``result.neighborhood`` are colored as
    "Neighborhood" and drawn smaller. A node that belongs to several paths takes the color
    of its lowest-numbered path, and the start node keeps a color of its own.
    """
    builder = _GraphBuilder()
    builder.add_node(result.start_element_id, role=START_ROLE)
    for index, path in enumerate(result.paths):
        builder.add_path(path, index)
    builder.add_neighborhood(result)
    return builder.to_graph()


def render_for_notebook(
    result: NavResult,
    *,
    layout: Layout | str | None = None,
    width: str = "100%",
    height: str = "600px",
    theme: Theme = "auto",
) -> HTML:
    """Render ``result`` as an :class:`IPython.display.HTML` object for inline notebook display."""
    return build_visualization(result).render(layout=layout, width=width, height=height, theme=theme)


def render_for_streamlit(
    result: NavResult,
    *,
    layout: Layout | str | None = None,
    width: str = "100%",
    height: str = "600px",
    theme: Theme = "auto",
) -> GraphWidget:
    """Build a :class:`~neo4j_viz.GraphWidget` for ``result``.

    Pass the returned widget straight to :func:`neo4j_viz.streamlit.display_widget`.
    """
    return build_visualization(result).render_widget(layout=layout, width=width, height=height, theme=theme)
