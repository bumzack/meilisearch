use std::collections::HashSet;

use super::proximity_graph::ProximityGraph;

#[derive(Debug)]
pub struct Path {
    pub nodes: Vec<(usize, usize, u8)>,
    pub cost: u64,
}
impl PartialEq for Path {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}
impl Eq for Path {}
impl PartialOrd for Path {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.cost.partial_cmp(&other.cost)
    }
}
impl Ord for Path {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.cost.cmp(&other.cost)
    }
}

struct Dijkstra {
    unvisited: HashSet<usize>, // should be a small bitset
    distances: Vec<u64>,       // or binary heap (f64, usize)
    edge_costs: Vec<u8>,
    paths: Vec<Option<usize>>,
}

pub struct ShortestPathsState {
    to: usize,
    k: usize,
    paths_a: Vec<Path>,
    paths_b: Vec<Path>,
}

impl ProximityGraph {
    pub fn shortest_path(&self, from: usize, to: usize) -> Option<Path> {
        let mut dijkstra = Dijkstra {
            unvisited: (0..self.query.nodes.len()).collect(),
            distances: vec![u64::MAX; self.query.nodes.len()],
            edge_costs: vec![u8::MAX; self.query.nodes.len()],
            paths: vec![None; self.query.nodes.len()],
        };
        dijkstra.distances[from] = 0;

        while let Some(&cur_node) =
            dijkstra.unvisited.iter().min_by_key(|&&n| dijkstra.distances[n])
        {
            let cur_node_dist = dijkstra.distances[cur_node];
            if cur_node_dist == u64::MAX {
                return None;
            }
            if cur_node == to {
                break;
            }

            let succ_cur_node = &self.query.edges[cur_node].outgoing;
            let unvisited_succ_cur_node = succ_cur_node.intersection(&dijkstra.unvisited);
            for &succ in unvisited_succ_cur_node {
                let Some(edge_costs) = self.proximity_edges[cur_node].get(&succ) else {
                    continue;
                };
                let Some(&edge_cost) = edge_costs.iter().min() else {
                    continue;
                };

                // println!("cur node dist {cur_node_dist}");
                let old_dist_succ = &mut dijkstra.distances[succ];
                let new_potential_distance = cur_node_dist + edge_cost as u64;
                if new_potential_distance < *old_dist_succ {
                    *old_dist_succ = new_potential_distance;
                    dijkstra.edge_costs[succ] = edge_cost;
                    dijkstra.paths[succ] = Some(cur_node);
                }
            }
            dijkstra.unvisited.remove(&cur_node);
        }

        let mut cur = to;
        // let mut edge_costs = vec![];
        // let mut distances = vec![];
        let mut nodes = vec![];
        while let Some(n) = dijkstra.paths[cur] {
            nodes.push((n, cur, dijkstra.edge_costs[cur]));
            cur = n;
        }
        nodes.reverse();
        Some(Path { nodes, cost: dijkstra.distances[to] })
    }

    pub fn initialize_shortest_paths_state(
        &mut self,
        from: usize,
        to: usize,
    ) -> Option<ShortestPathsState> {
        let Some(shortest_path) = self.shortest_path(from, to) else {
            return None
        };
        let paths_a = vec![shortest_path];
        let paths_b = Vec::<Path>::new();
        let k = 0;
        Some(ShortestPathsState { to, k, paths_a, paths_b })
    }

    pub fn compute_next_shortest_path(&mut self, state: &mut ShortestPathsState) -> bool {
        state.k += 1;
        // println!("{:?}", state.paths_a);
        for (i, &(spur_node, _, _)) in state.paths_a[state.k - 1].nodes
            [..state.paths_a[state.k - 1].nodes.len() - 2]
            .iter()
            .enumerate()
        {
            let root_cost = state.paths_a[state.k - 1].nodes[..i]
                .iter()
                .fold(0, |sum, next| sum + next.2 as u64);
            // let (spur_node, root_cost) = A[k_i - 1].nodes[i];
            let root_path = &state.paths_a[state.k - 1].nodes[..i];
            // println!("spur_node: {spur_node}, root_cost: {root_cost}, root path: {root_path:?}");
            let mut removed_edges = vec![];
            for p in &state.paths_a {
                if root_path == &p.nodes[..i] {
                    // remove every edge from i to i+1 in the graph
                    // println!("remove edge: {:?}", p.nodes[i].1);
                    let prox_edges = &mut self.proximity_edges[p.nodes[i].0];

                    let prox_edges = prox_edges.get_mut(&p.nodes[i].1).unwrap();
                    let Some(pos) = prox_edges.iter().position(|&x| x == p.nodes[i].2) else {
                        continue;
                    };
                    let _cost = prox_edges.remove(pos);
                    // let edges_to_remove = self.proximity_edges[i].get_mut(&dest).unwrap();
                    removed_edges.push(p.nodes[i]);
                }
            }
            // println!("{}", self.graphviz());
            let spur_path = self.shortest_path(spur_node, state.to);

            for (removed_edge_src, removed_edge_dest, removed_edge_costs) in removed_edges {
                self.proximity_edges[removed_edge_src]
                    .entry(removed_edge_dest)
                    .or_default()
                    .push(removed_edge_costs);
            }

            // println!("root path: {root_path:?}");
            // println!("spur path: {spur_path:?}");

            let Some(spur_path) = spur_path else { continue; };

            let total_path = Path {
                nodes: root_path.iter().chain(spur_path.nodes.iter()).copied().collect(),
                cost: root_cost + spur_path.cost,
            };
            state.paths_b.push(total_path);
            state.paths_b.sort();
            state.paths_b.reverse();
        }
        if state.paths_b.is_empty() {
            return false;
        }
        state.paths_a.push(state.paths_b.pop().unwrap());
        true
    }
}

#[cfg(test)]
mod tests {
    use charabia::Tokenize;

    use crate::{
        index::tests::TempIndex,
        search::new::{
            proximity_graph::ProximityGraph,
            query_term::{word_derivations, LocatedQueryTerm},
            QueryGraph,
        },
    };

    #[test]
    fn build_graph() {
        let index = TempIndex::new();
        let fst = fst::Set::from_iter(["01", "234", "56"]).unwrap();

        let parts = LocatedQueryTerm::from_query(
            "0 1 \"2 3\" 4 5".tokenize(),
            Some(10),
            |word, is_prefix| word_derivations(&index, word, is_prefix, &fst),
        )
        .unwrap();
        // println!("{parts:?}");
        let graph = QueryGraph::from_query(parts, fst);
        // println!("{graph:?}");
        // println!("{}", graph.graphviz());

        // let positions_to_remove = vec![3, 6, 0, 4];
        // for p in positions_to_remove {
        //     graph.remove_words_at_position(p);
        //     println!("{}", graph.graphviz());
        // }

        let proximities = |w1: &str, w2: &str| -> Vec<i8> {
            if matches!((w1, w2), ("0", "1")) {
                // Instead of no proximities, it should be a constant (e.g. 8)
                // matching all documents
                vec![]
            } else {
                vec![1, 2]
            }
        };

        let mut prox_graph = ProximityGraph::from_query_graph(graph, proximities);

        println!("{}", prox_graph.graphviz());

        let mut state = prox_graph
            .initialize_shortest_paths_state(prox_graph.query.root_node, prox_graph.query.end_node)
            .unwrap();
        for _ in 0..6 {
            if !prox_graph.compute_next_shortest_path(&mut state) {
                break;
            }
            // println!("\n===========\n{}===========\n", prox_graph.graphviz());
        }
        // for Path { nodes, cost } in state.paths_a {
        //     println!("cost: {cost}");
        //     println!("nodes: {nodes:?}");
        // }
        // println!("{k_paths:?}");
    }
}
/*
/*
    function YenKSP(Graph, source, sink, K):
    // Determine the shortest path from the source to the sink.
    A[0] = Dijkstra(Graph, source, sink);
    // Initialize the set to store the potential kth shortest path.
    B = [];

    for k from 1 to K:
        // The spur node ranges from the first node to the next to last node in the previous k-shortest path.
        for i from 0 to size(A[k − 1]) − 2:

            // Spur node is retrieved from the previous k-shortest path, k − 1.
            spurNode = A[k-1].node(i);
            // The sequence of nodes from the source to the spur node of the previous k-shortest path.
            rootPath = A[k-1].nodes(0, i);

            for each path p in A:
                if rootPath == p.nodes(0, i):
                    // Remove the links that are part of the previous shortest paths which share the same root path.
                    remove p.edge(i,i + 1) from Graph;

            for each node rootPathNode in rootPath except spurNode:
                remove rootPathNode from Graph;

            // Calculate the spur path from the spur node to the sink.
            // Consider also checking if any spurPath found
            spurPath = Dijkstra(Graph, spurNode, sink);

            // Entire path is made up of the root path and spur path.
            totalPath = rootPath + spurPath;
            // Add the potential k-shortest path to the heap.
            if (totalPath not in B):
                B.append(totalPath);

            // Add back the edges and nodes that were removed from the graph.
            restore edges to Graph;
            restore nodes in rootPath to Graph;

        if B is empty:
            // This handles the case of there being no spur paths, or no spur paths left.
            // This could happen if the spur paths have already been exhausted (added to A),
            // or there are no spur paths at all - such as when both the source and sink vertices
            // lie along a "dead end".
            break;
        // Sort the potential k-shortest paths by cost.
        B.sort();
        // Add the lowest cost path becomes the k-shortest path.
        A[k] = B[0];
        // In fact we should rather use shift since we are removing the first element
        B.pop();

        return A;
    */

*/
