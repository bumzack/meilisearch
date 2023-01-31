use super::query_term::{QueryTerm, WordDerivations};
use super::QueryGraph;
use crate::{BoRoaringBitmapCodec, CboRoaringBitmapCodec, Index, Result, RoaringBitmapCodec};
use heed::types::ByteSlice;
use heed::{BytesDecode, RoTxn};
use roaring::{MultiOps, RoaringBitmap};
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet, VecDeque};

// TODO: manual performance metrics: access to DB, bitmap deserializations/operations, etc.

// TODO: record here which word derivation is empty, which node is empty, etc.
// to trim the query graph further later
pub struct ResolveQueryGraphCache<'c, 't> {
    pub word_docids: &'c mut WordDocIdsCache<'t>,
    pub prefix_docids: &'c mut PrefixDocIdsCache<'t>,
    pub node_docids: &'c mut NodeDocIdsCache,
}
#[derive(Default)]
pub struct NodeDocIdsCache {
    pub cache: HashMap<usize, RoaringBitmap>,
}
#[derive(Default)]
pub struct WordDocIdsCache<'t> {
    pub cache: HashMap<String, &'t [u8]>,
}
#[derive(Default)]
pub struct PrefixDocIdsCache<'t> {
    pub cache: HashMap<String, &'t [u8]>,
}
impl<'c, 't> ResolveQueryGraphCache<'c, 't> {
    pub fn get_word_docids(
        &mut self,
        index: &Index,
        txn: &'t RoTxn,
        word: &str,
    ) -> Result<&'t [u8]> {
        let bitmap_ptr = match self.word_docids.cache.entry(word.to_owned()) {
            Entry::Occupied(bitmap_ptr) => bitmap_ptr.get(),
            Entry::Vacant(entry) => {
                let Some(bitmap_ptr) = index.word_docids.remap_data_type::<ByteSlice>().get(txn, word)? else {
                    todo!();
                };
                entry.insert(bitmap_ptr);
                bitmap_ptr
            }
        };
        Ok(bitmap_ptr)
    }
    pub fn get_prefix_docids(
        &mut self,
        index: &Index,
        txn: &'t RoTxn,
        prefix: &str,
    ) -> Result<&'t [u8]> {
        let bitmap_ptr = match self.prefix_docids.cache.entry(prefix.to_owned()) {
            Entry::Occupied(bitmap_ptr) => bitmap_ptr.get(),
            Entry::Vacant(entry) => {
                let Some(bitmap_ptr) = index.word_prefix_docids.remap_data_type::<ByteSlice>().get(txn, prefix)? else {
                    todo!();
                };
                entry.insert(bitmap_ptr);
                bitmap_ptr
            }
        };
        Ok(bitmap_ptr)
    }
}

pub fn resolve_query_graph(
    index: &Index,
    txn: &RoTxn,
    q: &QueryGraph,
    universe: RoaringBitmap,
) -> Result<RoaringBitmap> {
    // TODO: these variables should be given as arguments
    // Maybe as a broader IndexCache?
    let mut word_docids = Default::default();
    let mut prefix_docids = Default::default();
    let mut node_docids = Default::default();

    let mut cache = ResolveQueryGraphCache {
        word_docids: &mut word_docids,
        prefix_docids: &mut prefix_docids,
        node_docids: &mut node_docids,
    };
    // resolve_query_graph_rec(index, txn, q, q.root_node, &mut docids, &mut cache)?;

    let mut nodes_resolved = HashSet::new();
    let mut nodes_docids = vec![RoaringBitmap::new(); q.nodes.len()];

    let mut next_nodes_to_visit = VecDeque::new();
    next_nodes_to_visit.push_front(q.root_node);

    while let Some(node) = next_nodes_to_visit.pop_front() {
        let predecessors = &q.edges[node].incoming;
        if !predecessors.is_subset(&nodes_resolved) {
            next_nodes_to_visit.push_back(node);
            continue;
        }
        // Take union of all predecessors
        let predecessors_iter = predecessors.iter().map(|p| &nodes_docids[*p]);
        let predecessors_docids = MultiOps::union(predecessors_iter);

        let n = &q.nodes[node];
        println!("resolving {node} {n:?}, predecessors: {predecessors:?}, their docids: {predecessors_docids:?}");
        let node_docids = match n {
            super::QueryNode::Term(located_term) => {
                let term = &located_term.value;
                match term {
                    QueryTerm::Phrase(_) => todo!("resolve phrase"),
                    QueryTerm::Word { original, derivations } => {
                        let mut or_docids = vec![];
                        let docids = cache.get_word_docids(index, txn, original)?;
                        let bitmap = RoaringBitmapCodec::bytes_decode(docids).unwrap();
                        println!("      get docids for word {original}: {bitmap:?}");
                        or_docids.push(docids);
                        match derivations {
                            WordDerivations::FromList(derived_words) => {
                                for word in derived_words {
                                    or_docids.push(cache.get_word_docids(index, txn, word)?);
                                }
                            }
                            WordDerivations::FromPrefixDB => {
                                or_docids.push(cache.get_prefix_docids(index, txn, original)?);
                            }
                        }
                        let or_iter = or_docids
                            .into_iter()
                            .map(|slice| RoaringBitmapCodec::bytes_decode(slice).unwrap());
                        let or = MultiOps::union(or_iter);
                        // TODO: if `or` is empty, register that somewhere, and immediately return an empty bitmap
                        predecessors_docids & or
                    }
                }
            }
            super::QueryNode::Deleted => {
                todo!()
            }
            super::QueryNode::Start => universe.clone(),
            super::QueryNode::End => {
                return Ok(predecessors_docids);
            }
        };
        nodes_resolved.insert(node);
        println!("resolved up to {node}: {node_docids:?}\n");
        nodes_docids[node] = node_docids;
        for &succ in q.edges[node].outgoing.iter() {
            if !next_nodes_to_visit.contains(&succ) && !nodes_resolved.contains(&succ) {
                next_nodes_to_visit.push_back(succ);
            }
        }
        for &prec in q.edges[node].incoming.iter() {
            if q.edges[prec].outgoing.is_subset(&nodes_resolved) {
                nodes_docids[prec].clear();
            }
        }
        println!("cached docids: {nodes_docids:?}");
    }

    panic!()
}

#[cfg(test)]
mod tests {
    use crate::{
        db_snap,
        index::tests::TempIndex,
        search::new::{
            query_term::{word_derivations_max_typo_1, LocatedQueryTerm},
            QueryGraph,
        },
    };
    use charabia::Tokenize;

    use super::resolve_query_graph;

    #[test]
    fn test_resolve_query_graph() {
        let index = TempIndex::new();

        index
            .update_settings(|s| {
                s.set_searchable_fields(vec!["text".to_owned()]);
            })
            .unwrap();

        index
            .add_documents(documents!([
                {"id": 0, "text": "0"},
                {"id": 1, "text": "1"},
                {"id": 2, "text": "2"},
                {"id": 3, "text": "3"},
                {"id": 4, "text": "4"},
                {"id": 5, "text": "5"},
                {"id": 6, "text": "6"},
                {"id": 7, "text": "7"},
                {"id": 8, "text": "0 1 2 3 4 5 6 7"},
                {"id": 9, "text": "7 6 5 4 3 2 1 0"},
                {"id": 10, "text": "01 234 56 7"},
                {"id": 11, "text": "7 56 0 1 23 5 4"},
                {"id": 12, "text": "0 1 2 3 4 5 6"},
                {"id": 13, "text": "01 23 4 5 7"},
            ]))
            .unwrap();

        // TODO: take the words fst from the index, and thus the construction of the query graph should already take
        // a read transaction as argument
        let fst = fst::Set::from_iter(["01", "23", "234", "56"]).unwrap();

        let graph = QueryGraph::from_query(
            LocatedQueryTerm::from_query("0 1 2 3 4 5 6 7".tokenize(), None, |word, is_prefix| {
                word_derivations_max_typo_1(&index, word, is_prefix, &fst)
            })
            .unwrap(),
            fst,
        );

        println!("{}", graph.graphviz());

        db_snap!(index, word_docids, @"7512d0b80659f6bf37d98b374ada8098");

        let txn = index.read_txn().unwrap();
        let universe = index.documents_ids(&txn).unwrap();
        insta::assert_debug_snapshot!(universe, @"RoaringBitmap<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]>");
        let docids = resolve_query_graph(&index, &txn, &graph, universe).unwrap();
        insta::assert_debug_snapshot!(docids, @"RoaringBitmap<[8, 9, 10, 11]>");

        // TODO: test with a reduced universe
    }
}
