use std::collections::hash_map::DefaultHasher;
use std::fmt::Write;
use std::hash::{Hash, Hasher};

use super::shortest_paths::{Path, PathEdgeId};

#[derive(Default, Debug)]
pub struct PathsPrefixTree {
    nodes: Vec<(PathEdgeId, PathsPrefixTree)>,
}

impl PathsPrefixTree {
    pub fn from_paths(paths: &[Path]) -> Self {
        let mut result = Self::default();
        for p in paths {
            result.add_edges(&p.edges);
        }
        result
    }
    pub fn add_edges(&mut self, edges: &[PathEdgeId]) {
        match edges {
            [] => {}
            [first_edge, remaining_edges @ ..] => {
                // comment
                for (edge, next_node) in &mut self.nodes {
                    if edge == first_edge {
                        return next_node.add_edges(remaining_edges);
                    }
                }
                let mut rest = PathsPrefixTree::default();
                rest.add_edges(remaining_edges);
                self.nodes.push((*first_edge, rest));
            }
        }
    }
    fn remove_first_rec(&mut self, cur: &mut Vec<PathEdgeId>) -> bool {
        let Some((first_edge, rest)) = self.nodes.first_mut() else { return true };
        cur.push(*first_edge);
        let next_is_empty = rest.remove_first_rec(cur);
        if next_is_empty {
            self.nodes.remove(0);
            self.nodes.is_empty()
        } else {
            false
        }
    }
    pub fn remove_first(&mut self) -> Option<Vec<PathEdgeId>> {
        let mut result = vec![];
        self.remove_first_rec(&mut result);
        if result.is_empty() {
            None
        } else {
            Some(result)
        }
    }
    pub fn remove_forbidden_edge(&mut self, forbidden_edge: &PathEdgeId) {
        let mut i = 0;
        while i < self.nodes.len() {
            let should_remove = if &self.nodes[i].0 == forbidden_edge {
                true
            } else if !self.nodes[i].1.nodes.is_empty() {
                self.nodes[i].1.remove_forbidden_edge(forbidden_edge);
                self.nodes[i].1.nodes.is_empty()
            } else {
                false
            };
            if should_remove {
                self.nodes.remove(i);
            } else {
                i += 1;
            }
        }
    }
    pub fn remove_forbidden_prefix(&mut self, forbidden_prefix: &[PathEdgeId]) {
        let [first_edge, remaining_prefix @ ..] = forbidden_prefix else {
            self.nodes.clear();
            return;
        };

        let mut i = 0;
        while i < self.nodes.len() {
            let edge = self.nodes[i].0;
            let should_remove = if edge == *first_edge {
                self.nodes[i].1.remove_forbidden_prefix(remaining_prefix);
                self.nodes[i].1.nodes.is_empty()
            } else {
                false
            };
            if should_remove {
                self.nodes.remove(i);
            } else {
                i += 1;
            }
        }
    }

    // TODO: should it return yes if no path is equal to the given list of edges, but
    // there is a path which starts with these edges?
    // pub fn contains_edges(&self, edges: &[PathEdgeId]) -> bool {
    //     match edges {
    //         [] => true,
    //         [first_edge, remaining_edges @ ..] => {
    //             for (edge, next_node) in &self.nodes {
    //                 if edge == first_edge {
    //                     return next_node.contains_edges(remaining_edges);
    //                 }
    //             }
    //             false
    //         }
    //     }
    // }
    pub fn graphviz(&self) -> String {
        let mut desc = String::new();
        desc.push_str("digraph G {\n");
        self.graphviz_rec(&mut desc, vec![]);
        desc.push_str("\n}\n");
        desc
    }
    fn graphviz_rec(&self, desc: &mut String, path_from: Vec<u64>) {
        let id_from = {
            let mut h = DefaultHasher::new();
            path_from.hash(&mut h);
            h.finish()
        };
        for (edge, rest) in self.nodes.iter() {
            let PathEdgeId { from, to, cost } = edge;
            let mut path_to = path_from.clone();
            path_to.push({
                let mut h = DefaultHasher::new();
                edge.hash(&mut h);
                h.finish()
            });
            let id_to = {
                let mut h = DefaultHasher::new();
                path_to.hash(&mut h);
                h.finish()
            };
            writeln!(desc, "{id_to} [label = \"{from}→{to} [{cost}]\"];").unwrap();
            writeln!(desc, "{id_from} -> {id_to};").unwrap();

            rest.graphviz_rec(desc, path_to);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use charabia::Tokenize;

    use crate::db_snap;
    use crate::index::tests::TempIndex;
    use crate::search::new::proximity::paths_prefix_tree::PathsPrefixTree;
    use crate::search::new::proximity::proximity_graph::{
        ProximityEdgeCache, ProximityGraph, ProximityGraphCache, WordPairProximityCache,
        WordPrefixPairProximityCache,
    };
    use crate::search::new::proximity::shortest_paths::PathEdgeId;
    use crate::search::new::query_term::{word_derivations_max_typo_1, LocatedQueryTerm};
    use crate::search::new::QueryGraph;

    #[test]
    fn paths_tree() {
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
        let mut edge_cache = ProximityEdgeCache { cache: HashMap::default() };

        let mut cache = ProximityGraphCache {
            word_pair_proximity_docids: &mut word_pair_proximity_cache,
            word_prefix_pair_proximity_docids: &mut word_prefix_pair_proximity_cache,
            edge_docids: &mut edge_cache,
            empty_path_prefixes: <_>::default(),
        };

        let mut prox_graph =
            ProximityGraph::from_query_graph(&index, &txn, graph, &mut cache).unwrap();

        println!("{}", prox_graph.graphviz());

        let mut state = prox_graph
            .initialize_shortest_paths_state(prox_graph.query.root_node, prox_graph.query.end_node)
            .unwrap();
        while state.shortest_paths.last().unwrap().cost <= 6 {
            if !prox_graph.compute_next_shortest_paths(&mut state) {
                break;
            }

            // println!("\n===========\n{}===========\n", prox_graph.graphviz());
        }

        let mut path_tree = PathsPrefixTree::default();
        for path in state.shortest_paths {
            path_tree.add_edges(&path.edges);
        }

        let desc = path_tree.graphviz();
        println!("{desc}");

        path_tree.remove_forbidden_prefix(&[
            PathEdgeId { from: 0, to: 2, cost: 0 },
            PathEdgeId { from: 2, to: 3, cost: 2 },
        ]);
        let desc = path_tree.graphviz();
        println!("{desc}");

        // path_tree.remove_forbidden_edge(&PathEdgeId { from: 5, to: 6, cost: 1 });

        // let desc = path_tree.graphviz();
        // println!("AFTER REMOVING 5-6 [1]:\n{desc}");

        // path_tree.remove_forbidden_edge(&PathEdgeId { from: 3, to: 4, cost: 1 });

        // let desc = path_tree.graphviz();
        // println!("AFTER REMOVING 3-4 [1]:\n{desc}");

        // let p = path_tree.remove_first();
        // println!("PATH: {p:?}");
        // let desc = path_tree.graphviz();
        // println!("AFTER REMOVING: {desc}");

        // let p = path_tree.remove_first();
        // println!("PATH: {p:?}");
        // let desc = path_tree.graphviz();
        // println!("AFTER REMOVING: {desc}");

        // path_tree.remove_all_containing_edge(&PathEdgeId { from: 5, to: 6, cost: 2 });

        // let desc = path_tree.graphviz();
        // println!("{desc}");

        // let first_edges = path_tree.remove_first().unwrap();
        // println!("{first_edges:?}");
        // let desc = path_tree.graphviz();
        // println!("{desc}");

        // let first_edges = path_tree.remove_first().unwrap();
        // println!("{first_edges:?}");
        // let desc = path_tree.graphviz();
        // println!("{desc}");

        // let first_edges = path_tree.remove_first().unwrap();
        // println!("{first_edges:?}");
        // let desc = path_tree.graphviz();
        // println!("{desc}");

        // println!("{path_tree:?}");
    }
}
