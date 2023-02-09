use super::proximity_graph::{ProximityEdges, ProximityGraph};
use std::{
    cmp::Reverse,
    collections::{BTreeMap, HashSet},
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Path {
    pub edges: Vec<PathEdgeId>,
    pub cost: u64,
}

struct DijkstraState {
    unvisited: HashSet<usize>, // should be a small bitset
    distances: Vec<u64>,       // or binary heap (f64, usize)
    edges: Vec<PathEdgeId>,
    paths: Vec<Option<usize>>,
}

pub struct KShortestPathGraphWrapper<'g> {
    graph: &'g ProximityGraph,
    removed_edges: HashSet<PathEdgeId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PathEdgeId {
    from: usize,
    to: usize,
    cost: u8,
}

impl<'g> KShortestPathGraphWrapper<'g> {
    fn cheapest_edge(&self, from: usize, to: usize) -> Option<PathEdgeId> {
        let edges = self.graph.proximity_edges[from].get(&to)?;
        match edges {
            ProximityEdges::Unconditional { cost } => {
                let edge_id = PathEdgeId { from, to, cost: *cost };

                if self.removed_edges.contains(&edge_id) {
                    None
                } else {
                    Some(edge_id)
                }
            }
            ProximityEdges::Pairs { pairs, leftover_cost_penalty } => {
                for (pair_idx, word_pairs) in pairs.iter().enumerate() {
                    let edge_id = PathEdgeId { from, to, cost: pair_idx as u8 };
                    if word_pairs.is_empty() || self.removed_edges.contains(&edge_id) {
                        continue;
                    } else {
                        return Some(edge_id);
                    }
                }
                let edge_id = PathEdgeId { from, to, cost: 8 + leftover_cost_penalty };
                if self.removed_edges.contains(&edge_id) {
                    None
                } else {
                    Some(edge_id)
                }
            }
        }
    }
}

pub struct KShortestPathsState {
    to: usize,
    shortest_paths: Vec<Path>,
    potential_shortest_paths: BTreeMap<u64, HashSet<Path>>,
}

impl<'g> KShortestPathGraphWrapper<'g> {
    pub fn shortest_path(&self, from: usize, to: usize) -> Option<Path> {
        let mut dijkstra = DijkstraState {
            unvisited: (0..self.graph.query.nodes.len()).collect(),
            distances: vec![u64::MAX; self.graph.query.nodes.len()],
            edges: vec![
                PathEdgeId { from: usize::MAX, to: usize::MAX, cost: u8::MAX };
                self.graph.query.nodes.len()
            ],
            paths: vec![None; self.graph.query.nodes.len()],
        };
        dijkstra.distances[from] = 0;

        // TODO: could use a binary heap here to store the distances
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

            let succ_cur_node =
                &self.graph.proximity_edges[cur_node].keys().copied().collect::<HashSet<_>>();
            // TODO: this intersection may be slow but shouldn't be,
            // can use a bitmap intersection instead
            let unvisited_succ_cur_node = succ_cur_node.intersection(&dijkstra.unvisited);
            for &succ in unvisited_succ_cur_node {
                let Some(cheapest_edge) = self.cheapest_edge(cur_node, succ) else {
                    continue;
                };

                // println!("cur node dist {cur_node_dist}");
                let old_dist_succ = &mut dijkstra.distances[succ];
                let new_potential_distance = cur_node_dist + cheapest_edge.cost as u64;
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
        let mut path_edges = vec![];
        while let Some(n) = dijkstra.paths[cur] {
            path_edges.push(dijkstra.edges[cur]);
            cur = n;
        }
        path_edges.reverse();
        Some(Path { edges: path_edges, cost: dijkstra.distances[to] })
    }

    pub fn initialize_shortest_paths_state(
        &mut self,
        from: usize,
        to: usize,
    ) -> Option<KShortestPathsState> {
        let Some(shortest_path) = self.shortest_path(from, to) else {
            return None
        };
        let shortest_paths = vec![shortest_path];
        let potential_shortest_paths = BTreeMap::new();
        Some(KShortestPathsState { to, shortest_paths, potential_shortest_paths })
    }

    pub fn compute_next_shortest_paths(&mut self, state: &mut KShortestPathsState) -> bool {
        let cur_shortest_path = &state.shortest_paths.last().unwrap();

        // edges of last shortest path except the last one
        let edges_of_cur_shortest_path =
            &cur_shortest_path.edges[..cur_shortest_path.edges.len() - 1];

        // for all nodes in the last shortest path (called spur_node), except last one...
        for (i, PathEdgeId { from: spur_node, .. }) in edges_of_cur_shortest_path.iter().enumerate()
        {
            let root_path = &cur_shortest_path.edges[..i];
            let root_cost = root_path.iter().fold(0, |sum, next| sum + next.cost as u64);

            // for all the paths already found that share a common prefix with the root path
            // we delete the edge from the spur node to the next one

            // TODO: this combination of for ... { if ... { } } could be written more efficiently using
            // better data structures. I expect the == between the two subpaths to be expensive
            // maybe storing a hash for each subpath would be more efficient, we'd only need to compute it once
            // or alternatively, I can have a prefix tree. That's the best solution
            for prev_short_path in &state.shortest_paths {
                // this should be replaced by a comparison of the nodes
                // from 0 to i
                if root_path == &prev_short_path.edges[..i] {
                    // remove every edge from i to i+1 in the graph
                    let edge_id_to_remove = &prev_short_path.edges[i];
                    self.removed_edges.insert(*edge_id_to_remove);
                }
            }

            // Compute the shortest path from the spur node to the destination
            // we will combine it with the root path to get a potential kth shortest path

            let spur_path = self.shortest_path(*spur_node, state.to);

            self.removed_edges.clear();

            let Some(spur_path) = spur_path else { continue; };
            let total_cost = root_cost + spur_path.cost;
            let total_path = Path {
                edges: root_path.iter().chain(spur_path.edges.iter()).cloned().collect(),
                cost: total_cost,
            };
            let entry = state.potential_shortest_paths.entry(total_cost).or_default();
            entry.insert(total_path);
        }
        if let Some(mut next_shortest_paths_entry) = state.potential_shortest_paths.first_entry() {
            // This could be implemented faster
            let next_shortest_paths = next_shortest_paths_entry.get_mut();
            let next_shortest_path = next_shortest_paths.iter().next().unwrap().clone();
            next_shortest_paths.remove(&next_shortest_path);
            state.shortest_paths.push(next_shortest_path.clone());

            if next_shortest_paths.is_empty() {
                next_shortest_paths_entry.remove();
            }

            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use charabia::Tokenize;

    use crate::{
        db_snap,
        index::tests::TempIndex,
        search::new::{
            proximity_graph::{
                ProximityEdges, ProximityGraph, ProximityGraphCache, WordPairProximityCache,
                WordPrefixPairProximityCache,
            },
            query_term::{word_derivations_max_typo_1, LocatedQueryTerm},
            shortest_paths::{KShortestPathGraphWrapper, Path},
            QueryGraph, QueryNode,
        },
    };

    use super::PathEdgeId;

    fn graphviz_path(graph: &ProximityGraph, path: &Path) -> String {
        let mut desc = String::new();
        desc.push_str("digraph G {\nrankdir = LR;\nnode [shape = \"record\"]\n");

        for node in 0..graph.query.nodes.len() {
            if matches!(graph.query.nodes[node], QueryNode::Deleted) {
                continue;
            }
            desc.push_str(&format!("{node} [label = {:?}]", &graph.query.nodes[node]));
            if node == graph.query.root_node {
                desc.push_str("[color = blue]");
            } else if node == graph.query.end_node {
                desc.push_str("[color = red]");
            }
            desc.push_str(";\n");

            for (destination, proximities) in graph.proximity_edges[node].iter() {
                match proximities {
                    ProximityEdges::Unconditional { cost } => {
                        let edge_id = PathEdgeId { from: node, to: *destination, cost: *cost };
                        let color = if path.edges.contains(&edge_id) {
                            ", color = red".to_owned()
                        } else {
                            String::new()
                        };
                        desc.push_str(&format!(
                            "{node} -> {destination} [label = \"always cost {cost}\"{color}];\n"
                        ));
                    }
                    ProximityEdges::Pairs { pairs, leftover_cost_penalty: leftover_cost } => {
                        for (pair_idx, pairs) in pairs.iter().enumerate() {
                            let cost = pair_idx as u8;
                            if !pairs.is_empty() {
                                let edge_id = PathEdgeId { from: node, to: *destination, cost };
                                let color = if path.edges.contains(&edge_id) {
                                    ", color = red".to_owned()
                                } else {
                                    String::new()
                                };
                                desc.push_str(&format!(
                                    "{node} -> {destination} [label = \"cost {cost}, {} pairs\"{color}];\n",
                                    pairs.len()
                                ));
                            }
                        }
                        let edge_id =
                            PathEdgeId { from: node, to: *destination, cost: 8 + leftover_cost };
                        let color = if path.edges.contains(&edge_id) {
                            ", color = red".to_owned()
                        } else {
                            String::new()
                        };
                        desc.push_str(&format!(
                            "{node} -> {destination} [label = \"remaining cost {}\"{color}];\n",
                            leftover_cost + 8
                        ));
                    }
                }
            }
            // for edge in self.edges[node].incoming.iter() {
            //     desc.push_str(&format!("{node} -> {edge} [color = grey];\n"));
            // }
        }

        desc.push('}');
        desc
    }

    #[test]
    fn build_graph() {
        let mut index = TempIndex::new();
        index.index_documents_config.autogenerate_docids = true;
        index
            .update_settings(|s| {
                s.set_searchable_fields(vec!["text".to_owned()]);
            })
            .unwrap();

        index
            .add_documents(documents!([
                {
                    "text": "0 1 2 3 4 5"
                },
                {
                    "text": "0 a 1 b 2 3 4 5"
                },
                {
                    "text": "0 a 1 b 3 a 4 b 5"
                },
                {
                    "text": "0 a a 1 b 2 3 4 5"
                },
                {
                    "text": "0 a a a a 1 b 3 45"
                },
            ]))
            .unwrap();

        db_snap!(index, word_pair_proximity_docids, @"679d1126b569b3e8b10dd937c3faedf9");

        let txn = index.read_txn().unwrap();
        let fst = index.words_fst(&txn).unwrap();
        let query =
            LocatedQueryTerm::from_query("0 1 2 3 4 5".tokenize(), None, |word, is_prefix| {
                word_derivations_max_typo_1(&index, &txn, word, is_prefix, &fst)
            })
            .unwrap();
        let graph = QueryGraph::from_query(&index, &txn, query).unwrap();

        let mut word_pair_proximity_cache = WordPairProximityCache { cache: HashMap::default() };
        let mut word_prefix_pair_proximity_cache =
            WordPrefixPairProximityCache { cache: HashMap::default() };
        let mut cache = ProximityGraphCache {
            word_pair_proximity: &mut word_pair_proximity_cache,
            word_prefix_pair_proximity: &mut word_prefix_pair_proximity_cache,
        };

        let prox_graph = ProximityGraph::from_query_graph(&index, &txn, graph, &mut cache).unwrap();

        println!("{}", prox_graph.graphviz());

        let mut prox_graph_dijkstra =
            KShortestPathGraphWrapper { graph: &prox_graph, removed_edges: HashSet::new() };

        let mut state = prox_graph_dijkstra
            .initialize_shortest_paths_state(
                prox_graph_dijkstra.graph.query.root_node,
                prox_graph_dijkstra.graph.query.end_node,
            )
            .unwrap();
        while state.shortest_paths.last().unwrap().cost <= 8 {
            if !prox_graph_dijkstra.compute_next_shortest_paths(&mut state) {
                break;
            }

            // println!("\n===========\n{}===========\n", prox_graph.graphviz());
        }
        drop(prox_graph_dijkstra);
        for path in state.shortest_paths {
            let cost = path.cost;
            let s = graphviz_path(&prox_graph, &path);
            println!("COST: {cost}\n{s}\n\n==========\n");

            // println!("nodes: {nodes:?}");
        }
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
