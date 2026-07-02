"""DGM high-level reasoning strategies.

Builds graphs from tasks/workflows and runs graph operations
via DgmBridge. All intelligence stays in Python; all compute in Rust.
"""
from __future__ import annotations

from sovereign_core_rs import SemanticWord

from .dgm_bridge import DgmBridge
from .dgm_model import NodeKind, EdgeKind


class DgmReasoner:
    """Orchestrates DGM graph reasoning for the Sovereign OS."""

    def __init__(self) -> None:
        self.bridge = DgmBridge()

    # ── graph construction helpers ────────────────────────────────────────────

    def build_task_chain(self, task_ids: list[int]) -> None:
        """Build a linear task dependency chain: t[0] → t[1] → … → t[n]."""
        for tid in task_ids:
            self.bridge.add_node(tid, NodeKind.TASK)
        for i in range(len(task_ids) - 1):
            self.bridge.add_edge(task_ids[i], task_ids[i + 1], EdgeKind.DEPENDS_ON)

    def add_agent_assignment(self, agent_id: int, task_id: int) -> None:
        self.bridge.add_node(agent_id, NodeKind.AGENT)
        self.bridge.add_edge(agent_id, task_id, EdgeKind.ASSIGNS)

    def add_resource(self, resource_id: int, produced_by: int) -> None:
        self.bridge.add_node(resource_id, NodeKind.RESOURCE)
        self.bridge.add_edge(produced_by, resource_id, EdgeKind.PRODUCES)

    # ── reasoning operations ─────────────────────────────────────────────────

    def plan_execution_order(self) -> list[int] | None:
        """Return topological execution order, or None if there's a circular dep."""
        return self.bridge.query_topo()

    def reachable_from(self, start_id: int) -> list[int]:
        """All node IDs reachable from start via BFS."""
        result_block = self.bridge.query_bfs(start_id=start_id)
        if not result_block.words:
            return []
        sw = SemanticWord.decode(result_block.words[0])
        # payload_ref = number of reachable nodes; run raw BFS for the IDs
        return list(self.bridge.rt.run_bfs(start_id))

    def shortest_distances(self, start_id: int) -> dict[int, int]:
        return self.bridge.query_shortest_paths(start_id)

    def stats(self) -> dict:
        return {
            "nodes": self.bridge.node_count(),
            "edges": self.bridge.edge_count(),
        }
