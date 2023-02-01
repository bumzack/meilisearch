use std::collections::{hash_map::Entry, HashMap};

use heed::{types::ByteSlice, RoTxn};
use itertools::Itertools;

use crate::{Index, Result};

use super::{
    query_term::{LocatedQueryTerm, QueryTerm, WordDerivations},
    QueryGraph, QueryNode,
};

pub enum ProximityEdges {
    NonExistent,
    Unconditional { cost: u8 },
    Pairs([Vec<WordPair>; 8]),
}
impl Default for ProximityEdges {
    fn default() -> Self {
        Self::NonExistent
    }
}
impl ProximityEdges {
    pub fn cheapest_edge(&self) -> Option<ProximityEdge> {
        match self {
            ProximityEdges::NonExistent => None,
            ProximityEdges::Unconditional { cost } => {
                Some(ProximityEdge::Unconditional { cost: *cost })
            }
            ProximityEdges::Pairs(pairs) => {
                for (cost, pairs) in pairs.iter().enumerate() {
                    if !pairs.is_empty() {
                        return Some(ProximityEdge::Pair {
                            cost: cost as u8,
                            word_pairs: pairs.clone(),
                        });
                    }
                }
                return None;
            }
        }
    }
    fn from_edge(edge: ProximityEdge) -> Self {
        match edge {
            ProximityEdge::Unconditional { cost } => Self::Unconditional { cost },
            ProximityEdge::Pair { cost, word_pairs } => {
                let mut pairs = std::array::from_fn(|_| vec![]);
                pairs[cost as usize] = word_pairs;
                Self::Pairs(pairs)
            }
        }
    }
    pub fn add_edge(&mut self, edge: ProximityEdge) {
        let result = match (std::mem::take(self), edge) {
            (ProximityEdges::NonExistent, edge) => Self::from_edge(edge),
            (ProximityEdges::Unconditional { cost }, _) => panic!(),
            (ProximityEdges::Pairs(_), ProximityEdge::Unconditional { cost }) => panic!(),
            (ProximityEdges::Pairs(pairs), ProximityEdge::Pair { cost, word_pairs }) => {
                let mut pairs = pairs.clone();
                assert!(pairs[cost as usize].is_empty());
                pairs[cost as usize] = word_pairs;
                Self::Pairs(pairs)
            }
        };
        *self = result;
    }
    // pub fn possible_costs(&self) -> Vec<u8> {
    //     match self {
    //         ProximityEdges::NonExistent => vec![],
    //         ProximityEdges::Unconditional { cost } => vec![*cost],
    //         ProximityEdges::Pairs(pairs) => {
    //             let mut costs = vec![];
    //             for (cost, pair) in pairs.iter().enumerate() {
    //                 if !pair.is_empty() {
    //                     costs.push(cost as u8);
    //                 }
    //             }
    //             costs
    //         }
    //     }
    // }
}
#[derive(Debug, Clone)]
pub enum ProximityEdge {
    Unconditional { cost: u8 },
    // TODO: Vec<WordPair> could be in a reference counted cell
    Pair { cost: u8, word_pairs: Vec<WordPair> },
}
impl PartialEq for ProximityEdge {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Unconditional { cost: l_cost }, Self::Unconditional { cost: r_cost }) => {
                l_cost == r_cost
            }
            (
                Self::Pair { cost: l_cost, word_pairs: l_word_pairs },
                Self::Pair { cost: r_cost, word_pairs: r_word_pairs },
            ) => l_cost == r_cost,
            _ => false,
        }
    }
}
impl ProximityEdge {
    pub fn cost(&self) -> u8 {
        match self {
            ProximityEdge::Unconditional { cost } => *cost,
            ProximityEdge::Pair { cost, .. } => *cost,
        }
    }
}

#[derive(Debug, Clone)]
pub enum WordPair {
    Words { left: String, right: String },
    WordPrefix { left: String, right_prefix: String },
}

pub struct WordPairProximityCache<'t> {
    // TODO: something more efficient than hashmap
    pub cache: HashMap<(u8, String, String), Option<&'t [u8]>>,
}

pub struct ProximityGraphCache<'c, 't> {
    pub word_pair_proximity: &'c mut WordPairProximityCache<'t>,
}
impl<'c, 't> ProximityGraphCache<'c, 't> {
    pub fn get_word_pair_proximity_docids(
        &mut self,
        index: &Index,
        txn: &'t RoTxn,
        word1: &str,
        word2: &str,
        proximity: u8,
    ) -> Result<Option<&'t [u8]>> {
        let key = (proximity, word1.to_owned(), word2.to_owned());
        match self.word_pair_proximity.cache.entry(key.clone()) {
            Entry::Occupied(bitmap_ptr) => Ok(bitmap_ptr.get().clone()),
            Entry::Vacant(entry) => {
                // Note that now, we really want to do a prefix iter over (w1, w2) to get all the possible proximities
                // but oh well
                //
                // Actually, we shouldn't greedily access this DB at all
                // a DB (w1, w2) -> [proximities] would be much better
                // We could even have a DB that is (w1) -> set of words such that (w1, w2) are in proximity
                // And if we worked with words encoded as integers, the set of words could be a roaring bitmap
                // Then, to find all the proximities between two list of words, we'd do:

                // inputs:
                //    - words1 (roaring bitmap)
                //    - words2 (roaring bitmap)
                // output:
                //    - [(word1, word2, [proximities])]
                // algo:
                //  let mut ouput = vec![];
                //  for word1 in words1 {
                //      let all_words_in_proximity_of_w1 = pair_words_db.get(word1);
                //      let words_in_proximity_of_w1 = all_words_in_proximity_of_w1 & words2;
                //      for word2 in words_in_proximity_of_w1 {
                //          let proximties = prox_db.get(word1, word2);
                //          output.push(word1, word2, proximities);
                //      }
                //  }
                let bitmap_ptr = index
                    .word_pair_proximity_docids
                    .remap_data_type::<ByteSlice>()
                    .get(txn, &(key.0, key.1.as_str(), key.2.as_str()))?;
                entry.insert(bitmap_ptr);
                Ok(bitmap_ptr)
            }
        }
    }
}

pub struct ProximityGraph {
    pub query: QueryGraph,
    pub proximity_edges: Vec<HashMap<usize, ProximityEdges>>,
}

impl ProximityGraph {
    pub fn from_query_graph<'t>(
        index: &Index,
        txn: &'t RoTxn,
        query: QueryGraph,
        cache: &mut ProximityGraphCache<'_, 't>,
    ) -> Result<ProximityGraph> {
        // for each node, look at its successors
        // then create an edge for each proximity available between these neighbours
        let mut prox_graph = ProximityGraph { query, proximity_edges: vec![] };
        for (node_idx, node) in prox_graph.query.nodes.iter().enumerate() {
            prox_graph.proximity_edges.push(HashMap::new());
            let prox_edges = &mut prox_graph.proximity_edges.last_mut().unwrap();
            let (derivations1, pos1) = match node {
                QueryNode::Term(LocatedQueryTerm { value: value1, positions: pos1 }) => {
                    match value1 {
                        QueryTerm::Word { derivations } => (derivations.clone(), *pos1.end()),
                        QueryTerm::Phrase(phrase1) => {
                            // TODO: remove second unwrap
                            let original = phrase1.last().unwrap().as_ref().unwrap().clone();
                            (
                                WordDerivations {
                                    original: original.clone(),
                                    zero_typo: Some(original),
                                    one_typo: vec![],
                                    two_typos: vec![],
                                    use_prefix_db: false,
                                },
                                *pos1.end(),
                            )
                        }
                    }
                }
                QueryNode::Start => (
                    WordDerivations {
                        original: String::new(),
                        zero_typo: None,
                        one_typo: vec![],
                        two_typos: vec![],
                        use_prefix_db: false,
                    },
                    -100,
                ),
                _ => continue,
            };
            for &successor_idx in prox_graph.query.edges[node_idx].outgoing.iter() {
                match &prox_graph.query.nodes[successor_idx] {
                    QueryNode::Term(LocatedQueryTerm { value: value2, positions: pos2 }) => {
                        let (derivations2, pos2) = match node {
                            QueryNode::Term(LocatedQueryTerm {
                                value: value2,
                                positions: pos2,
                            }) => {
                                match value2 {
                                    QueryTerm::Word { derivations } => {
                                        (derivations.clone(), *pos2.end())
                                    }
                                    QueryTerm::Phrase(phrase2) => {
                                        // TODO: remove second unwrap
                                        let original =
                                            phrase2.last().unwrap().as_ref().unwrap().clone();
                                        (
                                            WordDerivations {
                                                original: original.clone(),
                                                zero_typo: Some(original),
                                                one_typo: vec![],
                                                two_typos: vec![],
                                                use_prefix_db: false,
                                            },
                                            *pos2.end(),
                                        )
                                    }
                                }
                            }
                            QueryNode::Start => (
                                WordDerivations {
                                    original: String::new(),
                                    zero_typo: None,
                                    one_typo: vec![],
                                    two_typos: vec![],
                                    use_prefix_db: false,
                                },
                                -100,
                            ),
                            _ => continue,
                        };
                        // TODO: here we would actually do it for each combination of word1 and word2
                        // and take the union of them
                        let proxs = if pos1 + 1 != pos2 {
                            // TODO: how should this actually be handled?
                            // We want to effectively ignore this pair of terms
                            // Unconditionally walk through the edge without computing the docids
                            ProximityEdges::Unconditional { cost: 0 }
                        } else {
                            // TODO: manage the `use_prefix DB case`
                            // There are a few shortcuts to take there to avoid performing
                            // really expensive operations
                            let WordDerivations {
                                original: _,
                                zero_typo: zt1,
                                one_typo: ot1,
                                two_typos: tt1,
                                use_prefix_db: updb1,
                            } = &derivations1;
                            let WordDerivations {
                                original: _,
                                zero_typo: zt2,
                                one_typo: ot2,
                                two_typos: tt2,
                                use_prefix_db: _, // TODO
                            } = derivations2;

                            // left term cannot be a prefix
                            assert!(!updb1);

                            let derivations1 = zt1.iter().chain(ot1.iter()).chain(tt1.iter());
                            let derivations2 = zt2.iter().chain(ot2.iter()).chain(tt2.iter());
                            let product_derivations = derivations1.cartesian_product(derivations2);

                            let mut proximity_word_pairs: [_; 8] = std::array::from_fn(|_| vec![]);
                            for (word1, word2) in product_derivations {
                                for proximity in 0..7 {
                                    // TODO: do the opposite way with a proximity penalty as well!
                                    if cache
                                        .get_word_pair_proximity_docids(
                                            index, txn, word1, word2, proximity,
                                        )?
                                        .is_some()
                                    {
                                        proximity_word_pairs[proximity as usize].push(
                                            WordPair::Words {
                                                left: word1.to_owned(),
                                                right: word2.to_owned(),
                                            },
                                        );
                                    }
                                }
                            }
                            if proximity_word_pairs.is_empty() {
                                ProximityEdges::Unconditional { cost: 8 }
                            } else {
                                ProximityEdges::Pairs(proximity_word_pairs)
                            }
                        };

                        prox_edges.insert(successor_idx, proxs);
                    }
                    QueryNode::End => {
                        prox_edges.insert(successor_idx, ProximityEdges::Unconditional { cost: 0 });
                    }
                    _ => continue,
                }
            }
        }
        // TODO: simplify the proximity graph
        // by removing the dead end nodes. These kinds of algorithms
        // could be defined generically on a trait
        // TODO: why should it be simplified? There is no dead end node
        prox_graph.simplify();

        Ok(prox_graph)
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
                match proximities {
                    ProximityEdges::NonExistent => {}
                    ProximityEdges::Unconditional { cost } => {
                        desc.push_str(&format!(
                            "{node} -> {destination} [label = \"always cost {cost}\"];\n"
                        ));
                    }
                    ProximityEdges::Pairs(pairs) => {
                        for (cost, pairs) in pairs.iter().enumerate() {
                            desc.push_str(&format!(
                                "{node} -> {destination} [label = \"cost {cost}, {} pairs\"];\n",
                                pairs.len()
                            ));
                        }
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
}
