use std::collections::HashSet;

use super::proximity_graph::{ProximityEdge, ProximityEdges, ProximityGraph};

#[derive(Clone, Debug, PartialEq)]
pub struct PathNode {
    pub from: usize,
    pub to: usize,
    pub edge: ProximityEdge,
}

#[derive(Debug)]
pub struct Path {
    pub nodes: Vec<PathNode>,
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
    edges: Vec<ProximityEdge>,
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
            edges: vec![ProximityEdge::Unconditional { cost: u8::MAX }; self.query.nodes.len()],
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
                let Some(proximity_edges) = self.proximity_edges[cur_node].get(&succ) else {
                    continue;
                };
                let Some(cheapest_edge) = proximity_edges.cheapest_edge() else {
                    continue;
                };

                // println!("cur node dist {cur_node_dist}");
                let old_dist_succ = &mut dijkstra.distances[succ];
                let new_potential_distance = cur_node_dist + cheapest_edge.cost() as u64;
                if new_potential_distance < *old_dist_succ {
                    *old_dist_succ = new_potential_distance;
                    dijkstra.edges[succ] = cheapest_edge;
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
            nodes.push(PathNode { from: n, to: cur, edge: dijkstra.edges[cur].clone() });
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
        for (i, PathNode { from: spur_node, to: _, edge: _ }) in state.paths_a[state.k - 1].nodes
            [..state.paths_a[state.k - 1].nodes.len() - 2]
            .iter()
            .enumerate()
        {
            let root_cost = state.paths_a[state.k - 1].nodes[..i]
                .iter()
                .fold(0, |sum, next| sum + next.edge.cost() as u64);
            // let (spur_node, root_cost) = A[k_i - 1].nodes[i];
            let root_path = &state.paths_a[state.k - 1].nodes[..i];
            // println!("spur_node: {spur_node}, root_cost: {root_cost}, root path: {root_path:?}");
            let mut removed_edges = vec![];
            for p in &state.paths_a {
                if root_path == &p.nodes[..i] {
                    // remove every edge from i to i+1 in the graph
                    // println!("remove edge: {:?}", p.nodes[i].1);
                    let all_prox_edges = &mut self.proximity_edges[p.nodes[i].from];
                    let cost_to_remove = p.nodes[i].edge.cost();
                    let prox_edges = all_prox_edges.get_mut(&p.nodes[i].to).unwrap();
                    // TODO: we should verify that prox_edges contain `p.nodes[i].edge` here
                    match prox_edges {
                        ProximityEdges::NonExistent => {
                            // TODO: should this be impossible?
                            todo!();
                            continue;
                        }
                        ProximityEdges::Unconditional { cost } => {
                            if cost_to_remove == *cost {
                                *prox_edges = ProximityEdges::NonExistent;
                            } else {
                                // TODO: should this be impossible?
                                todo!();
                                continue;
                            }
                        }
                        ProximityEdges::Pairs(pairs) => {
                            if pairs[cost_to_remove as usize].is_empty() {
                                // TODO: should this be impossible?
                                todo!();
                                continue;
                            } else {
                                pairs[cost_to_remove as usize] = vec![];
                            }
                        }
                    }
                    // let edges_to_remove = self.proximity_edges[i].get_mut(&dest).unwrap();
                    removed_edges.push(p.nodes[i].clone());
                }
            }
            // println!("{}", self.graphviz());
            let spur_path = self.shortest_path(*spur_node, state.to);

            for PathNode { from: removed_edge_src, to: removed_edge_dest, edge: removed_edge } in
                removed_edges
            {
                self.proximity_edges[removed_edge_src]
                    .entry(removed_edge_dest)
                    .or_default()
                    // TODO: should also restore the word pairs here
                    // should the paths contain ProximityEdges?
                    .add_edge(removed_edge);
            }

            // println!("root path: {root_path:?}");
            // println!("spur path: {spur_path:?}");

            let Some(spur_path) = spur_path else { continue; };

            let total_path = Path {
                nodes: root_path.iter().chain(spur_path.nodes.iter()).cloned().collect(),
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
    use std::collections::HashMap;

    use charabia::Tokenize;

    use crate::{
        index::tests::TempIndex,
        search::new::{
            proximity_graph::{ProximityGraph, ProximityGraphCache, WordPairProximityCache},
            query_term::{word_derivations_max_typo_1, LocatedQueryTerm},
            QueryGraph,
        },
    };

    #[test]
    fn build_graph() {
        let index = TempIndex::new();
        let txn = index.read_txn().unwrap();
        let fst = index.words_fst(&txn).unwrap();
        let query =
            LocatedQueryTerm::from_query("0 1 \"2 3\" 4 5".tokenize(), None, |word, is_prefix| {
                word_derivations_max_typo_1(&index, &txn, word, is_prefix, &fst)
            })
            .unwrap();
        let mut graph = QueryGraph::from_query(&index, &txn, query).unwrap();

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

        let mut word_pair_proximity_cache = WordPairProximityCache { cache: HashMap::default() };
        let mut cache = ProximityGraphCache { word_pair_proximity: &mut word_pair_proximity_cache };

        let mut prox_graph =
            ProximityGraph::from_query_graph(&index, &txn, graph, &mut cache).unwrap();

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
