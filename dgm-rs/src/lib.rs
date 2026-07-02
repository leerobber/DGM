mod graph_store;
mod graph_algorithms;
mod dgm_runtime;

use pyo3::prelude::*;
use dgm_runtime::DgmRuntime as _DgmRuntime;

/// Python-facing DGM Runtime.
#[pyclass(name = "DgmRuntime")]
pub struct PyDgmRuntime {
    inner: _DgmRuntime,
}

#[pymethods]
impl PyDgmRuntime {
    #[new]
    pub fn new() -> Self {
        Self { inner: _DgmRuntime::new() }
    }

    pub fn add_node(&mut self, id: u64, kind: u8) {
        self.inner.add_node(id, kind);
    }

    pub fn add_edge(&mut self, from: u64, to: u64, kind: u8) {
        self.inner.add_edge(from, to, kind);
    }

    pub fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    pub fn edge_count(&self) -> usize {
        self.inner.edge_count()
    }

    /// Dispatch SemanticWord-encoded u64 words, return output word_ints.
    pub fn handle_words(&mut self, words: Vec<u64>) -> Vec<u64> {
        self.inner.handle_words(&words)
    }

    /// BFS from start node; returns visited node IDs in order.
    pub fn run_bfs(&self, start: u64) -> Vec<u64> {
        self.inner.run_bfs(start)
    }

    /// DFS from start node; returns visited node IDs in order.
    pub fn run_dfs(&self, start: u64) -> Vec<u64> {
        self.inner.run_dfs(start)
    }

    /// Shortest-path distances from start; returns list of (node_id, distance).
    pub fn run_shortest_paths(&self, start: u64) -> Vec<(u64, u64)> {
        self.inner.run_shortest_paths(start)
    }

    /// Topological sort; returns None if the graph has a cycle.
    pub fn run_topo_sort(&self) -> Option<Vec<u64>> {
        self.inner.run_topo_sort()
    }
}

#[pymodule]
fn dgm_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDgmRuntime>()?;
    Ok(())
}
