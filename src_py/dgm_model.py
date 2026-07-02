"""DGM graph schema — node and edge type definitions.

Pure Python data model. No Rust imports. No sovereign_core_rs imports.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum


class NodeKind(IntEnum):
    TASK       = 0
    AGENT      = 1
    RESOURCE   = 2
    CONSTRAINT = 3
    STATE      = 4
    GOAL       = 5


class EdgeKind(IntEnum):
    DEPENDS_ON = 0
    PRODUCES   = 1
    BLOCKS     = 2
    ENABLES    = 3
    ASSIGNS    = 4


@dataclass
class DgmNode:
    id: int
    kind: NodeKind
    label: str = ""
    attrs: dict = field(default_factory=dict)


@dataclass
class DgmEdge:
    from_id: int
    to_id: int
    kind: EdgeKind
    weight: float = 1.0


@dataclass
class DgmGraph:
    nodes: dict[int, DgmNode] = field(default_factory=dict)
    edges: list[DgmEdge] = field(default_factory=list)

    def add_node(self, node: DgmNode) -> None:
        self.nodes[node.id] = node

    def add_edge(self, edge: DgmEdge) -> None:
        self.edges.append(edge)

    def node_count(self) -> int:
        return len(self.nodes)

    def edge_count(self) -> int:
        return len(self.edges)
