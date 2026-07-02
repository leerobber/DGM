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
}
