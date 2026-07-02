"""DGM → dgm_rs Rust engine bridge.

This module:
  • Encodes graph queries into SemanticWord → word_int
  • Wraps them into sovereign_core_rs.KernelBlock
  • Sends them into dgm_rs.DgmRuntime (Rust) OR optional WASM graph agents
  • Returns the resulting KernelBlock(s)

NO Python sovereign-core imports.
NO circular imports.
NO GH05T3 expert logic.
Pure Rust bridge.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from sovereign_core_rs import KernelBlock, SemanticWord, WasmHost, WasmAgent
from dgm_rs import DgmRuntime

# Intent → WASM agent name mapping (matches compiled wasm files)
_INTENT_AGENT: dict[int, str] = {
    41: "graph_bfs",    # QUERY_GRAPH / BFS
    43: "graph_dfs",    # DFS
    44: "graph_shortest",
    45: "graph_topo",
}

_DEFAULT_WASM_DIR = Path(os.environ.get("DGM_WASM_DIR",
    Path(__file__).parent.parent / "wasm_agents" / "compiled"))


class DgmBridge:
    """Python → Rust DGM bridge with optional WASM agent dispatch."""

    def __init__(self, wasm_dir: Optional[Path] = None) -> None:
        self.rt = DgmRuntime()
        self._host = WasmHost()
        self._wasm_agents: dict[str, WasmAgent] = {}
        self._load_wasm_agents(Path(wasm_dir) if wasm_dir else _DEFAULT_WASM_DIR)

    def _load_wasm_agents(self, wasm_dir: Path) -> None:
        for name in _INTENT_AGENT.values():
            path = wasm_dir / f"{name}.wasm"
            if path.exists():
                agent = self._host.load_agent(path.read_bytes(), name)
                self._wasm_agents[name] = agent

    # ── graph construction ────────────────────────────────────────────────────

    def add_node(self, node_id: int, kind: int) -> None:
        self.rt.add_node(node_id, kind)

    def add_edge(self, from_id: int, to_id: int, kind: int) -> None:
        self.rt.add_edge(from_id, to_id, kind)

    def node_count(self) -> int:
        return self.rt.node_count()

    def edge_count(self) -> int:
        return self.rt.edge_count()

    # ── block dispatch ────────────────────────────────────────────────────────

    def _make_query_word(self, intent: int, payload_ref: int) -> int:
        return SemanticWord(
            type_=3, intent=intent, channel=0, priority=128,
            confidence=int(0.95 * 65535),
            payload_ref=payload_ref & 0xFFFF,
        ).encode()

    def _make_block(self, agent_id: int, genome_id: int,
                    creds_token: int, task_id: int, word_int: int) -> KernelBlock:
        return KernelBlock(agent_id=agent_id, genome_id=genome_id,
                           creds_token=creds_token, task_id=task_id,
                           words=[word_int], metrics_ref=0)

    def handle_block(self, block: KernelBlock) -> list[KernelBlock]:
        """Dispatch block: try WASM agent first, fall back to Rust handle_words."""
        if block.words:
            intent = (block.words[0] >> 48) & 0xFF
            agent_name = _INTENT_AGENT.get(intent)
            if agent_name and agent_name in self._wasm_agents:
                return self._host.call_agent(self._wasm_agents[agent_name], block)

        out_words = self.rt.handle_words(block.words)
        if not out_words:
            return []
        return [KernelBlock(agent_id=block.agent_id, genome_id=block.genome_id,
                            creds_token=block.creds_token, task_id=block.task_id,
                            words=out_words, metrics_ref=block.metrics_ref)]

    # ── public graph queries ──────────────────────────────────────────────────

    def query_bfs(self, start_id: int, agent_id: int = 1, genome_id: int = 1,
                  creds_token: int = 0b1111, task_id: int = 0) -> KernelBlock | None:
        word_int = self._make_query_word(intent=41, payload_ref=start_id)
        block = self._make_block(agent_id, genome_id, creds_token, task_id, word_int)
        out = self.handle_block(block)
        return out[0] if out else None

    def query_dfs(self, start_id: int, agent_id: int = 1, genome_id: int = 1,
                  creds_token: int = 0b1111, task_id: int = 0) -> KernelBlock | None:
        word_int = self._make_query_word(intent=43, payload_ref=start_id)
        block = self._make_block(agent_id, genome_id, creds_token, task_id, word_int)
        out = self.handle_block(block)
        return out[0] if out else None

    def query_shortest_paths(self, start_id: int) -> dict[int, int]:
        return {node: dist for node, dist in self.rt.run_shortest_paths(start_id)}

    def query_topo(self, agent_id: int = 1, genome_id: int = 1,
                   creds_token: int = 0b1111, task_id: int = 0) -> list[int] | None:
        return self.rt.run_topo_sort()

    def query_topo_sort(self, agent_id: int = 1, genome_id: int = 1,
                        creds_token: int = 0b1111, task_id: int = 0) -> list[int] | None:
        return self.rt.run_topo_sort()

    @property
    def wasm_agent_names(self) -> list[str]:
        return list(self._wasm_agents.keys())
