pub mod proximity;
pub mod query_graph;
pub mod query_term;
pub mod ranking_rules;
pub mod resolve_query_graph;
pub mod sort;
pub mod words;

use charabia::Tokenize;
use heed::RoTxn;
pub use query_graph::*;
pub use ranking_rules::*;
use roaring::RoaringBitmap;

use self::query_term::{word_derivations_max_typo_1, LocatedQueryTerm};
use crate::Index;

pub enum BitmapOrAll<'s> {
    Bitmap(&'s RoaringBitmap),
    All,
}

pub fn make_query_graph(
    index: &Index,
    txn: &RoTxn,
    query: &str,
) -> crate::Result<QueryGraphOrPlaceholder> {
    if query.is_empty() {
        Ok(QueryGraphOrPlaceholder::Placeholder)
    } else {
        let fst = index.words_fst(txn).unwrap();
        let query = LocatedQueryTerm::from_query(query.tokenize(), None, |word, is_prefix| {
            word_derivations_max_typo_1(index, txn, word, is_prefix, &fst)
        })
        .unwrap();
        let graph = QueryGraph::from_query(index, txn, query)?;
        Ok(QueryGraphOrPlaceholder::QueryGraph(graph))
    }
}
