/// DGM Runtime — takes SemanticWord-encoded KernelBlock fields, runs graph ops,
/// returns output word_ints.
///
/// Interface uses plain integers (u64 words) rather than importing sovereign_core_rs
/// types, avoiding cross-extension linkage issues. The Python dgm_bridge.py
/// reconstructs KernelBlock from the returned word_ints using sovereign_core_rs.
use crate::graph_algorithms::{bfs, dfs, shortest_paths, topo_sort};
use crate::graph_store::GraphStore;

// SemanticWord intent values that DGM responds to.
const INTENT_GRAPH_QUERY: u8  = 40;  // BUILD_GRAPH opcode value from S-ISA
const INTENT_GRAPH_UPDATE: u8 = 42;  // UPDATE_GRAPH
const INTENT_GRAPH_BFS: u8   = 41;  // QUERY_GRAPH (BFS by default)
const INTENT_PLAN: u8        = 2;

fn decode_intent(word: u64) -> u8 {
    ((word >> 48) & 0xFF) as u8
}

fn decode_payload_ref(word: u64) -> u16 {
    (word & 0xFFFF) as u16
}

fn encode_result_word(intent: u8, confidence: u16, payload_ref: u16) -> u64 {
    let type_: u64 = 6; // RESULT
    let channel: u64 = 0; // SYSTEM
    let priority: u64 = 200;
    (type_ << 56) | ((intent as u64) << 48) | (channel << 40) | (priority << 32)
        | ((confidence as u64) << 16) | (payload_ref as u64)
}

pub struct DgmRuntime {
    pub graph: GraphStore,
}

impl DgmRuntime {
    pub fn new() -> Self {
        Self { graph: GraphStore::default() }
    }

    pub fn add_node(&mut self, id: u64, kind: u8) {
        self.graph.add_node(id, kind);
    }

    pub fn add_edge(&mut self, from: u64, to: u64, kind: u8) {
        self.graph.add_edge(from, to, kind);
    }

    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Dispatch a list of SemanticWord-encoded u64s, return output word_ints.
    pub fn handle_words(&mut self, words: &[u64]) -> Vec<u64> {
        if words.is_empty() {
            return Vec::new();
        }
        let first = words[0];
        let intent = decode_intent(first);
        let payload_ref = decode_payload_ref(first);
        let start_node = payload_ref as u64;

        match intent {
            INTENT_GRAPH_BFS | INTENT_PLAN | INTENT_GRAPH_QUERY => {
                let order = bfs(&self.graph, start_node);
                let result_ref = (order.len() as u16).min(0xFFFF);
                let confidence = if order.is_empty() { 0 } else { 60000u16 };
                vec![encode_result_word(intent, confidence, result_ref)]
            }
            INTENT_GRAPH_UPDATE => {
                // Add nodes from remaining words' payload_refs
                for &w in &words[1..] {
                    let node_id = decode_payload_ref(w) as u64;
                    let kind = decode_intent(w);
                    self.graph.add_node(node_id, kind);
                }
                let result_ref = self.graph.node_count() as u16;
                vec![encode_result_word(intent, 65535, result_ref)]
            }
            _ => {
                // Unknown intent — run DFS and report
                let order = dfs(&self.graph, start_node);
                let result_ref = (order.len() as u16).min(0xFFFF);
                vec![encode_result_word(intent, 50000, result_ref)]
            }
        }
    }

    pub fn run_bfs(&self, start: u64) -> Vec<u64> {
        bfs(&self.graph, start)
    }

    pub fn run_dfs(&self, start: u64) -> Vec<u64> {
        dfs(&self.graph, start)
    }

    pub fn run_shortest_paths(&self, start: u64) -> Vec<(u64, u64)> {
        shortest_paths(&self.graph, start).into_iter().collect()
    }

    pub fn run_topo_sort(&self) -> Option<Vec<u64>> {
        topo_sort(&self.graph)
    }
}
