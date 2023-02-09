use std::{
    array,
    collections::{hash_map::Entry, HashMap},
};

use heed::{types::ByteSlice, RoTxn};
use itertools::Itertools;
use roaring::RoaringBitmap;

use crate::{Index, Result};

use super::{
    query_term::{LocatedQueryTerm, QueryTerm, WordDerivations},
    shortest_paths::Path,
    QueryGraph, QueryNode,
};

pub enum ProximityEdges {
    Unconditional {
        cost: u8,
    },
    Pairs {
        pairs: [Vec<WordPair>; 8],
        /// The additional cost, added to the minimum cost of `8`, for the remaining
        /// word pairs
        leftover_cost_penalty: u8,
    },
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
pub struct WordPrefixPairProximityCache<'t> {
    // TODO: something more efficient than hashmap
    pub cache: HashMap<(u8, String, String), Option<&'t [u8]>>,
}

pub struct ProximityGraphCache<'c, 't> {
    pub word_pair_proximity: &'c mut WordPairProximityCache<'t>,
    pub word_prefix_pair_proximity: &'c mut WordPrefixPairProximityCache<'t>,
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
    pub fn get_word_prefix_pair_proximity_docids(
        &mut self,
        index: &Index,
        txn: &'t RoTxn,
        word1: &str,
        prefix2: &str,
        proximity: u8,
    ) -> Result<Option<&'t [u8]>> {
        let key = (proximity, word1.to_owned(), prefix2.to_owned());
        match self.word_prefix_pair_proximity.cache.entry(key.clone()) {
            Entry::Occupied(bitmap_ptr) => Ok(*bitmap_ptr.get()),
            Entry::Vacant(entry) => {
                let bitmap_ptr = index
                    .word_prefix_pair_proximity_docids
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
    // Instead of an owned value of ProximityEdges, put each edge into a vector and
    // store a pointer there
    //
    // or maybe not
    //
    // The point is to make it easy to cache the result of each edge in a vector
    // of optional roaring bitmaps
    // The alternative is to have a PathEdgeId as a key to the cache
    //
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
                                    zero_typo: vec![original],
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
                        zero_typo: vec![],
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
                        let (derivations2, pos2) = match value2 {
                            QueryTerm::Word { derivations } => (derivations.clone(), *pos2.start()),
                            QueryTerm::Phrase(phrase2) => {
                                // TODO: remove second unwrap
                                let original = phrase2.last().unwrap().as_ref().unwrap().clone();
                                (
                                    WordDerivations {
                                        original: original.clone(),
                                        zero_typo: vec![original],
                                        one_typo: vec![],
                                        two_typos: vec![],
                                        use_prefix_db: false,
                                    },
                                    *pos2.start(),
                                )
                            }
                        };

                        // TODO: here we would actually do it for each combination of word1 and word2
                        // and take the union of them
                        let proxs = if pos1 + 1 != pos2 {
                            // TODO: how should this actually be handled?
                            // We want to effectively ignore this pair of terms
                            // Unconditionally walk through the edge without computing the docids
                            // But also what should the cost be?
                            ProximityEdges::Unconditional { cost: 0 }
                        } else {
                            // TODO: manage the `use_prefix_db` case
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
                                use_prefix_db: upd2, // TODO
                            } = derivations2;

                            // left term cannot be a prefix
                            assert!(!updb1);

                            let derivations1 = zt1.iter().chain(ot1.iter()).chain(tt1.iter());
                            let derivations2 = zt2.iter().chain(ot2.iter()).chain(tt2.iter());
                            let product_derivations = derivations1.cartesian_product(derivations2);

                            let mut proximity_word_pairs: [_; 8] = array::from_fn(|_| vec![]);
                            for (word1, word2) in product_derivations {
                                for proximity in 0..7 {
                                    // TODO: do the opposite way with a proximity penalty as well!
                                    // TODO: search for proximity+1, I guess?
                                    if upd2 {
                                        if cache
                                            .get_word_prefix_pair_proximity_docids(
                                                index, txn, word1, word2, proximity,
                                            )?
                                            .is_some()
                                        {
                                            proximity_word_pairs[proximity as usize].push(
                                                WordPair::WordPrefix {
                                                    left: word1.to_owned(),
                                                    right_prefix: word2.to_owned(),
                                                },
                                            );
                                        } // else what?
                                    } else if cache
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
                                    // else what?
                                }
                            }
                            ProximityEdges::Pairs {
                                pairs: proximity_word_pairs,
                                leftover_cost_penalty: 0,
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
        desc.push_str("digraph G {\nrankdir = LR;\nnode [shape = \"record\"]\n");

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
                    ProximityEdges::Unconditional { cost } => {
                        desc.push_str(&format!(
                            "{node} -> {destination} [label = \"always cost {cost}\"];\n"
                        ));
                    }
                    ProximityEdges::Pairs { pairs, leftover_cost_penalty: leftover_cost } => {
                        for (cost, pairs) in pairs.iter().enumerate() {
                            if !pairs.is_empty() {
                                desc.push_str(&format!(
                                    "{node} -> {destination} [label = \"cost {cost}, {} pairs\"];\n",
                                    pairs.len()
                                ));
                            }
                        }
                        desc.push_str(&format!(
                            "{node} -> {destination} [label = \"remaining cost {}\"];\n",
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
}

impl ProximityGraph {
    fn resolve_paths(
        &self,
        index: &Index,
        txn: &RoTxn,
        cache: &ProximityGraphCache,
        paths: &[Path],
    ) -> RoaringBitmap {
        // for each path, translate it to an intersection of cached roaring bitmaps
        // then do a union for all paths

        // get the docids of the given paths in the proximity graph
        // in the fastest possible way
        // 1. roaring MultiOps (before we can do the Frozen+AST thing)
        // 2. minimize number of operations
        todo!()
    }
}
