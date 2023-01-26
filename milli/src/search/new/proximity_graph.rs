use std::collections::HashMap;

use super::{
    query_term::{LocatedQueryTerm, QueryTerm},
    QueryGraph, QueryNode,
};

pub struct ProximityGraph {
    pub query: QueryGraph,
    pub proximity_edges: Vec<HashMap<usize, Vec<u8>>>,
}

impl ProximityGraph {
    pub fn from_query_graph(
        query: QueryGraph,
        proximities: impl Fn(&str, &str) -> Vec<i8>,
    ) -> ProximityGraph {
        // for each node, look at its successors
        // then create an edge for each proximity available between these neighbours
        let mut prox_graph = ProximityGraph { query, proximity_edges: vec![] };
        for (node_idx, node) in prox_graph.query.nodes.iter().enumerate() {
            prox_graph.proximity_edges.push(HashMap::new());
            let prox_edges = &mut prox_graph.proximity_edges.last_mut().unwrap();
            let (word1, pos1) = match node {
                QueryNode::Term(LocatedQueryTerm { value: value1, positions: pos1 }) => {
                    match value1 {
                        QueryTerm::Word { original: word1, derivations } => {
                            (word1.as_str(), *pos1.end())
                        }
                        QueryTerm::Phrase(phrase1) => {
                            // TODO: remove second unwrap
                            (phrase1.last().unwrap().as_ref().unwrap().as_str(), *pos1.end())
                        }
                    }
                    // (word1.as_str(), pos1)
                }
                QueryNode::Start => ("", -100),
                _ => continue,
            };
            for &successor_idx in prox_graph.query.edges[node_idx].outgoing.iter() {
                match &prox_graph.query.nodes[successor_idx] {
                    QueryNode::Term(LocatedQueryTerm { value: value2, positions: pos2 }) => {
                        let (word2, pos2) = match value2 {
                            QueryTerm::Word { original: word2, derivations } => {
                                (word2.as_str(), *pos2.start())
                            }
                            QueryTerm::Phrase(phrase2) => {
                                // TODO: remove second unwrap
                                (phrase2.first().unwrap().as_ref().unwrap().as_str(), *pos2.start())
                            }
                        };
                        // TODO: here we would actually do it for each combination of word1 and word2
                        // and take the union of them
                        let proxs = if pos1 + 1 != pos2 {
                            vec![0]
                        } else {
                            proximities(word1, word2)
                                .into_iter()
                                .map(|x| x as u8)
                                .collect::<Vec<u8>>()
                        };
                        if !proxs.is_empty() {
                            prox_edges.insert(successor_idx, proxs);
                        }
                    }
                    QueryNode::End => {
                        prox_edges.insert(successor_idx, vec![0]);
                    }
                    _ => continue,
                }
            }
        }
        // TODO: simplify the proximity graph
        // by removing the dead end nodes. These kinds of algorithms
        // could be defined generically on a trait

        prox_graph.simplify();

        prox_graph
    }
}
impl ProximityGraph {
    pub fn remove_nodes(&mut self, nodes: &[usize]) {
        for &node in nodes {
            let proximity_edges = &mut self.proximity_edges[node];
            *proximity_edges = HashMap::new();
            let preds = &self.query.edges[node].incoming;
            for pred in preds {
                self.proximity_edges[*pred].remove(&node);
            }
        }
        self.query.remove_nodes(nodes);
    }
    fn simplify(&mut self) {
        loop {
            let mut nodes_to_remove = vec![];
            for (node_idx, node) in self.query.nodes.iter().enumerate() {
                if !matches!(node, QueryNode::End | QueryNode::Deleted)
                    && self.proximity_edges[node_idx].is_empty()
                {
                    nodes_to_remove.push(node_idx);
                }
            }
            if nodes_to_remove.is_empty() {
                break;
            } else {
                self.remove_nodes(&nodes_to_remove);
            }
        }
    }
    pub fn graphviz(&self) -> String {
        let mut desc = String::new();
        desc.push_str("digraph G {\nrankdir = LR;\n");

        for node in 0..self.query.nodes.len() {
            if matches!(self.query.nodes[node], QueryNode::Deleted) {
                continue;
            }
            desc.push_str(&format!("{node} [label = {:?}]", &self.query.nodes[node]));
            if node == self.query.root_node {
                desc.push_str("[color = blue]");
            } else if node == self.query.end_node {
                desc.push_str("[color = red]");
            }
            desc.push_str(";\n");

            for (destination, proximities) in self.proximity_edges[node].iter() {
                for prox in proximities {
                    desc.push_str(&format!("{node} -> {destination} [label = \"{prox}\"];\n"));
                }
            }
            // for edge in self.edges[node].incoming.iter() {
            //     desc.push_str(&format!("{node} -> {edge} [color = grey];\n"));
            // }
        }

        desc.push('}');
        desc
    }
}
