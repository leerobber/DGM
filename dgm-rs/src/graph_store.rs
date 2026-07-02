use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct Node {
    pub id: u64,
    pub kind: u8,
}

#[derive(Clone, Debug)]
pub struct Edge {
    pub from: u64,
    pub to: u64,
    pub kind: u8,
}

#[derive(Default)]
pub struct GraphStore {
    pub nodes: HashMap<u64, Node>,
    pub edges: Vec<Edge>,
}

impl GraphStore {
    pub fn add_node(&mut self, id: u64, kind: u8) {
        self.nodes.insert(id, Node { id, kind });
    }

    pub fn add_edge(&mut self, from: u64, to: u64, kind: u8) {
        self.edges.push(Edge { from, to, kind });
    }

    pub fn neighbors(&self, node_id: u64) -> Vec<u64> {
        self.edges
            .iter()
            .filter(|e| e.from == node_id)
            .map(|e| e.to)
            .collect()
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Build a CSR (row_ptr, col_idx) over the current graph.
    ///
    /// Nodes are 0-indexed in sorted order of their u64 IDs.
    /// Returns (row_ptr: Vec<u32>, col_idx: Vec<u32>).
    pub fn to_csr(&self) -> (Vec<u32>, Vec<u32>) {
        let mut node_ids: Vec<u64> = self.nodes.keys().cloned().collect();
        node_ids.sort_unstable();
        let n = node_ids.len();

        let id_to_idx: HashMap<u64, usize> = node_ids.iter()
            .enumerate()
            .map(|(i, &id)| (id, i))
            .collect();

        // Count out-degree per indexed node.
        let mut deg = vec![0usize; n];
        for e in &self.edges {
            if let Some(&fi) = id_to_idx.get(&e.from) {
                deg[fi] += 1;
            }
        }

        // Prefix-sum → row_ptr.
        let mut row_ptr = vec![0u32; n + 1];
        for i in 0..n {
            row_ptr[i + 1] = row_ptr[i] + deg[i] as u32;
        }

        // Fill col_idx.
        let m = row_ptr[n] as usize;
        let mut col_idx = vec![0u32; m];
        let mut pos: Vec<u32> = row_ptr[..n].to_vec();
        for e in &self.edges {
            if let (Some(&fi), Some(&ti)) = (id_to_idx.get(&e.from), id_to_idx.get(&e.to)) {
                col_idx[pos[fi] as usize] = ti as u32;
                pos[fi] += 1;
            }
        }

        (row_ptr, col_idx)
    }

    /// Node IDs in sorted order (matches the CSR row index).
    pub fn node_ids_sorted(&self) -> Vec<u64> {
        let mut ids: Vec<u64> = self.nodes.keys().cloned().collect();
        ids.sort_unstable();
        ids
    }
}
