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

    /// Build CSR arrays from the current graph.
    ///
    /// Returns (row_ptr, col_idx) where nodes are 0-indexed in sorted ID order.
    /// Pass directly to sovereign_gpu.GpuGraph via:
    ///     n_nodes = len(row_ptr) - 1
    ///     edges = list(zip(col_idx, ...))   # or use to_gpu_edges()
    pub fn to_csr(&self) -> (Vec<u32>, Vec<u32>) {
        self.inner.graph.to_csr()
    }

    /// Node IDs in sorted order — matches the 0-based row index in to_csr().
    pub fn node_ids(&self) -> Vec<u64> {
        self.inner.graph.node_ids_sorted()
    }

    /// Edge list as (from_idx, to_idx) pairs with 0-based indices.
    /// n_nodes = self.node_count(); pass both to sovereign_gpu.GpuGraph.
    pub fn to_gpu_edges(&self) -> (usize, Vec<(u32, u32)>) {
        let (row_ptr, col_idx) = self.inner.graph.to_csr();
        let n = row_ptr.len().saturating_sub(1);
        let mut edges = Vec::with_capacity(col_idx.len());
        for u in 0..n {
            let s = row_ptr[u] as usize;
            let e = row_ptr[u + 1] as usize;
            for &v in &col_idx[s..e] {
                edges.push((u as u32, v));
            }
        }
        (n, edges)
    }
}

#[pymodule]
fn dgm_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDgmRuntime>()?;
    Ok(())
}
