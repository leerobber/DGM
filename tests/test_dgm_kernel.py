"""Tests for DGM Rust graph engine + Python bridge/reasoner."""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dgm_rs import DgmRuntime
from sovereign_core_rs import KernelBlock, SemanticWord
from src_py.dgm_model import DgmNode, DgmEdge, DgmGraph, NodeKind, EdgeKind
from src_py.dgm_bridge import DgmBridge
from src_py.dgm_reasoner import DgmReasoner


# ── DgmRuntime (Rust) ─────────────────────────────────────────────────────────

class TestDgmRuntime:
    def test_new_runtime_empty(self):
        rt = DgmRuntime()
        assert rt.node_count() == 0
        assert rt.edge_count() == 0

    def test_add_node(self):
        rt = DgmRuntime()
        rt.add_node(1, 0)
        assert rt.node_count() == 1

    def test_add_multiple_nodes(self):
        rt = DgmRuntime()
        for i in range(5):
            rt.add_node(i, 0)
        assert rt.node_count() == 5

    def test_add_edge(self):
        rt = DgmRuntime()
        rt.add_node(1, 0)
        rt.add_node(2, 0)
        rt.add_edge(1, 2, 0)
        assert rt.edge_count() == 1

    def test_bfs_linear_chain(self):
        rt = DgmRuntime()
        for i in range(1, 5):
            rt.add_node(i, 0)
        for i in range(1, 4):
            rt.add_edge(i, i + 1, 0)
        order = rt.run_bfs(1)
        assert order == [1, 2, 3, 4]

    def test_bfs_missing_start_returns_empty(self):
        rt = DgmRuntime()
        assert rt.run_bfs(99) == []

    def test_dfs_chain(self):
        rt = DgmRuntime()
        for i in [10, 20, 30]:
            rt.add_node(i, 0)
        rt.add_edge(10, 20, 0)
        rt.add_edge(20, 30, 0)
        order = rt.run_dfs(10)
        assert order[0] == 10
        assert 20 in order and 30 in order

    def test_shortest_paths(self):
        rt = DgmRuntime()
        for i in range(1, 4):
            rt.add_node(i, 0)
        rt.add_edge(1, 2, 0)
        rt.add_edge(2, 3, 0)
        paths = dict(rt.run_shortest_paths(1))
        assert paths[1] == 0
        assert paths[2] == 1
        assert paths[3] == 2

    def test_topo_sort_dag(self):
        rt = DgmRuntime()
        for i in [1, 2, 3]:
            rt.add_node(i, 0)
        rt.add_edge(1, 2, 0)
        rt.add_edge(2, 3, 0)
        order = rt.run_topo_sort()
        assert order is not None
        assert order.index(1) < order.index(2) < order.index(3)

    def test_topo_sort_cycle_returns_none(self):
        rt = DgmRuntime()
        for i in [1, 2, 3]:
            rt.add_node(i, 0)
        rt.add_edge(1, 2, 0)
        rt.add_edge(2, 3, 0)
        rt.add_edge(3, 1, 0)  # cycle
        assert rt.run_topo_sort() is None

    def test_handle_words_bfs_intent(self):
        rt = DgmRuntime()
        rt.add_node(1, 0)
        rt.add_node(2, 0)
        rt.add_edge(1, 2, 0)
        word = SemanticWord(
            type_=3, intent=41, channel=0, priority=128,
            confidence=60000, payload_ref=1,
        ).encode()
        out = rt.handle_words([word])
        assert len(out) == 1
        sw = SemanticWord.decode(out[0])
        assert sw.type_ == 6  # RESULT
        assert sw.payload_ref == 2  # 2 nodes visited

    def test_handle_words_empty_returns_empty(self):
        rt = DgmRuntime()
        assert rt.handle_words([]) == []


# ── DgmModel ──────────────────────────────────────────────────────────────────

class TestDgmModel:
    def test_graph_add_node(self):
        g = DgmGraph()
        g.add_node(DgmNode(id=1, kind=NodeKind.TASK))
        assert g.node_count() == 1

    def test_graph_add_edge(self):
        g = DgmGraph()
        g.add_edge(DgmEdge(from_id=1, to_id=2, kind=EdgeKind.DEPENDS_ON))
        assert g.edge_count() == 1

    def test_node_kinds(self):
        assert NodeKind.TASK == 0
        assert NodeKind.AGENT == 1
        assert NodeKind.RESOURCE == 2


# ── DgmBridge ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def bridge():
    b = DgmBridge()
    b.add_node(1, NodeKind.TASK)
    b.add_node(2, NodeKind.TASK)
    b.add_node(3, NodeKind.TASK)
    b.add_edge(1, 2, EdgeKind.DEPENDS_ON)
    b.add_edge(2, 3, EdgeKind.DEPENDS_ON)
    return b


def test_bridge_node_count(bridge):
    assert bridge.node_count() == 3

def test_bridge_edge_count(bridge):
    assert bridge.edge_count() == 2

def test_bridge_query_bfs(bridge):
    result = bridge.query_bfs(start_id=1)
    assert isinstance(result, KernelBlock)
    assert len(result.words) == 1
    sw = SemanticWord.decode(result.words[0])
    assert sw.type_ == 6        # RESULT
    assert sw.payload_ref == 3  # 3 nodes reachable

def test_bridge_handle_block_roundtrip(bridge):
    word = SemanticWord(
        type_=3, intent=41, channel=0, priority=128,
        confidence=50000, payload_ref=1,
    ).encode()
    block = KernelBlock(
        agent_id=7, genome_id=2, creds_token=0b1111,
        task_id=99, words=[word], metrics_ref=0,
    )
    results = bridge.handle_block(block)
    assert len(results) == 1
    result = results[0]
    assert result.agent_id == 7
    assert result.task_id == 99
    assert len(result.words) == 1

def test_bridge_topo_sort(bridge):
    order = bridge.query_topo()
    assert order is not None
    assert order.index(1) < order.index(2) < order.index(3)

def test_bridge_shortest_paths(bridge):
    dists = bridge.query_shortest_paths(1)
    assert dists[1] == 0
    assert dists[2] == 1
    assert dists[3] == 2


# ── DgmReasoner ───────────────────────────────────────────────────────────────

@pytest.fixture()
def reasoner():
    r = DgmReasoner()
    r.build_task_chain([10, 20, 30])
    return r

def test_reasoner_stats(reasoner):
    s = reasoner.stats()
    assert s["nodes"] == 3
    assert s["edges"] == 2

def test_reasoner_plan_execution_order(reasoner):
    order = reasoner.plan_execution_order()
    assert order is not None
    assert order.index(10) < order.index(20) < order.index(30)

def test_reasoner_reachable_from(reasoner):
    reachable = reasoner.reachable_from(10)
    assert set(reachable) == {10, 20, 30}

def test_reasoner_reachable_from_middle(reasoner):
    reachable = reasoner.reachable_from(20)
    assert 10 not in reachable
    assert 20 in reachable and 30 in reachable

def test_reasoner_shortest_distances(reasoner):
    dists = reasoner.shortest_distances(10)
    assert dists[10] == 0
    assert dists[20] == 1
    assert dists[30] == 2

def test_reasoner_add_agent(reasoner):
    reasoner.add_agent_assignment(agent_id=99, task_id=10)
    assert reasoner.bridge.node_count() == 4
    assert reasoner.bridge.edge_count() == 3

def test_reasoner_add_resource(reasoner):
    reasoner.add_resource(resource_id=500, produced_by=30)
    assert reasoner.bridge.node_count() == 4
