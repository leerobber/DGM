use crate::graph_store::GraphStore;
use std::collections::{HashMap, HashSet, VecDeque};

pub fn bfs(store: &GraphStore, start: u64) -> Vec<u64> {
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    let mut order = Vec::new();

    if !store.nodes.contains_key(&start) {
        return order;
    }

    visited.insert(start);
    queue.push_back(start);

    while let Some(node) = queue.pop_front() {
        order.push(node);
        for neighbor in store.neighbors(node) {
            if !visited.contains(&neighbor) {
                visited.insert(neighbor);
                queue.push_back(neighbor);
            }
        }
    }
    order
}

pub fn dfs(store: &GraphStore, start: u64) -> Vec<u64> {
    let mut visited = HashSet::new();
    let mut order = Vec::new();
    dfs_visit(store, start, &mut visited, &mut order);
    order
}

fn dfs_visit(store: &GraphStore, node: u64, visited: &mut HashSet<u64>, order: &mut Vec<u64>) {
    if visited.contains(&node) || !store.nodes.contains_key(&node) {
        return;
    }
    visited.insert(node);
    order.push(node);
    for neighbor in store.neighbors(node) {
        dfs_visit(store, neighbor, visited, order);
    }
}

/// Shortest-path lengths from `start` to all reachable nodes (BFS-based).
pub fn shortest_paths(store: &GraphStore, start: u64) -> HashMap<u64, u64> {
    let mut dist: HashMap<u64, u64> = HashMap::new();
    let mut queue = VecDeque::new();

    if !store.nodes.contains_key(&start) {
        return dist;
    }

    dist.insert(start, 0);
    queue.push_back(start);

    while let Some(node) = queue.pop_front() {
        let d = dist[&node];
        for neighbor in store.neighbors(node) {
            if !dist.contains_key(&neighbor) {
                dist.insert(neighbor, d + 1);
                queue.push_back(neighbor);
            }
        }
    }
    dist
}

/// Topological sort (Kahn's algorithm). Returns None if cycle detected.
pub fn topo_sort(store: &GraphStore) -> Option<Vec<u64>> {
    let mut in_degree: HashMap<u64, usize> = store.nodes.keys().map(|&k| (k, 0)).collect();
    for e in &store.edges {
        *in_degree.entry(e.to).or_insert(0) += 1;
    }

    let mut queue: VecDeque<u64> = in_degree
        .iter()
        .filter(|(_, &d)| d == 0)
        .map(|(&n, _)| n)
        .collect();
    let mut order = Vec::new();

    while let Some(node) = queue.pop_front() {
        order.push(node);
        for neighbor in store.neighbors(node) {
            let deg = in_degree.get_mut(&neighbor).unwrap();
            *deg -= 1;
            if *deg == 0 {
                queue.push_back(neighbor);
            }
        }
    }

    if order.len() == store.nodes.len() {
        Some(order)
    } else {
        None // cycle
    }
}
