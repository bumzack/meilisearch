use heed::RoTxn;
use roaring::RoaringBitmap;

use super::resolve_query_graph::resolve_query_graph;
use super::QueryGraph;
use crate::new::words::Words;
// use crate::search::new::sort::Sort;
use crate::{Index, Result};

pub trait RankingRuleOutputIter<'transaction> {
    fn next_bucket(&mut self) -> Result<Option<RankingRuleOutput>>;
}

pub struct RankingRuleOutputIterWrapper<'transaction> {
    iter: Box<dyn Iterator<Item = Result<RankingRuleOutput>> + 'transaction>,
}
impl<'transaction> RankingRuleOutputIterWrapper<'transaction> {
    pub fn new(iter: Box<dyn Iterator<Item = Result<RankingRuleOutput>> + 'transaction>) -> Self {
        Self { iter }
    }
}
impl<'transaction> RankingRuleOutputIter<'transaction>
    for RankingRuleOutputIterWrapper<'transaction>
{
    fn next_bucket(&mut self) -> Result<Option<RankingRuleOutput>> {
        match self.iter.next() {
            Some(x) => x.map(Some),
            None => Ok(None),
        }
    }
}

pub trait RankingRule<'transaction> {
    fn init_bucket_iter(
        &mut self,
        index: &Index,
        txn: &'transaction RoTxn,
        parent_candidates: &RoaringBitmap,
        parent_query_tree: &QueryGraphOrPlaceholder,
    ) -> Result<()>;

    fn next_bucket(
        &mut self,
        index: &Index,
        txn: &'transaction RoTxn,
    ) -> Result<Option<RankingRuleOutput>>;

    fn reset_bucket_iter(&mut self, index: &Index, txn: &'transaction RoTxn);
}

#[derive(Debug)]
pub struct RankingRuleOutput {
    /// The query tree that must be used by the child ranking rule to fetch candidates.
    pub query_graph: QueryGraphOrPlaceholder,
    /// The allowed candidates for the child ranking rule
    pub candidates: RoaringBitmap,
}

#[derive(Debug, Clone)]
pub enum QueryGraphOrPlaceholder {
    QueryGraph(QueryGraph),
    Placeholder,
}

// This should find the fastest way to resolve the query tree, taking shortcuts if necessary
fn resolve_query_tree_or_placeholder<'t>(
    index: &Index,
    txn: &'t RoTxn,
    query_tree_or_placeholder: &QueryGraphOrPlaceholder,
    all: &RoaringBitmap,
) -> Result<RoaringBitmap> {
    match query_tree_or_placeholder {
        QueryGraphOrPlaceholder::QueryGraph(qt) => {
            let r = resolve_query_graph(index, txn, qt, all.clone())?;
            Ok(r)
        }
        QueryGraphOrPlaceholder::Placeholder => Ok(all.clone()),
    }
}

#[allow(unused)]
pub fn initial<'transaction>(
    index: &Index,
    txn: &'transaction RoTxn,
    query_graph: QueryGraphOrPlaceholder,
    universe: &RoaringBitmap,
    // mut distinct: Option<D>,
) -> Result<RoaringBitmap> {
    // resolve the whole query tree to retrieve an exhaustive list of documents matching the query.

    let candidates = resolve_query_tree_or_placeholder(index, txn, &query_graph, universe)?;

    // Distinct should be lazy if placeholder?
    //
    // // because the initial_candidates should be an exhaustive count of the matching documents,
    // // we precompute the distinct attributes.
    // let initial_candidates = match &mut distinct {
    //     Some(distinct) => {
    //         let mut initial_candidates = RoaringBitmap::new();
    //         for c in distinct.distinct(candidates.clone(), RoaringBitmap::new()) {
    //             initial_candidates.insert(c?);
    //         }
    //         initial_candidates
    //     }
    //     None => candidates.clone(),
    // };

    Ok(candidates)
}

pub fn execute_search<'transaction>(
    index: &Index,
    txn: &'transaction heed::RoTxn,
    universe: &RoaringBitmap,
    query_graph: QueryGraphOrPlaceholder,
    _from: usize,
    _length: usize,
) -> Result<Vec<u32>> {
    let words = Words::default();
    // let sort = Sort::new(index, txn, "sort1".to_owned(), true)?;

    let mut ranking_rules: Vec<Box<dyn RankingRule>> =
        vec![Box::new(words) /*  Box::new(sort) */];

    let ranking_rules_len = ranking_rules.len();
    ranking_rules[0].init_bucket_iter(index, txn, universe, &query_graph)?;

    // TODO: parent_candidates could be used only during debugging?
    let mut candidates = vec![RoaringBitmap::default(); ranking_rules_len];
    candidates[0] = universe.clone();

    let mut cur_ranking_rule_index = 0;

    macro_rules! back {
        () => {
            ranking_rules[cur_ranking_rule_index].reset_bucket_iter(index, txn);
            if cur_ranking_rule_index == 0 {
                break;
            } else {
                cur_ranking_rule_index -= 1;
            }
        };
    }

    let mut results = vec![];
    // TODO: skip buckets when we want to start from an offset
    while results.len() < 20 {
        let Some(next_bucket) = ranking_rules[cur_ranking_rule_index].next_bucket(index, txn)? else {
            back!();
            continue;
        };

        candidates[cur_ranking_rule_index] -= &next_bucket.candidates;

        match next_bucket.candidates.len() {
            0 => {
                // no progress anymore, go to the parent candidate
                // note that there is a risk here that the ranking rule was given documents it never returned
                // all ranking rules should make sure that the sum of their buckets equal the parent candidates
                // they were given during their initialisation
                // we check that this mistake wasn't made with this assertion
                assert!(candidates[cur_ranking_rule_index].is_empty());

                back!();
                continue;
            }
            1 => {
                // only one candidate, no need to sort through the child ranking rule
                results.extend(next_bucket.candidates);
                continue;
            }
            _ => {
                // many candidates, give to next ranking rule, if any
                if cur_ranking_rule_index == ranking_rules_len - 1 {
                    // TODO: don't extend too much, up to the limit only
                    results.extend(next_bucket.candidates);
                } else {
                    cur_ranking_rule_index += 1;
                    candidates[cur_ranking_rule_index] = next_bucket.candidates.clone();
                    // make new iterator from next ranking_rule
                    ranking_rules[cur_ranking_rule_index].init_bucket_iter(
                        index,
                        txn,
                        &next_bucket.candidates,
                        &next_bucket.query_graph,
                    )?;
                }
            }
        }
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::{execute_search, initial};
    use crate::index::tests::TempIndex;
    use crate::new::make_query_graph;

    #[test]
    fn execute_new_search() {
        let index = TempIndex::new();
        index
            .add_documents(documents!([
                {
                    "id": 0,
                    "text": "the quick brown fox jumps over",
                },
                {
                    "id": 1,
                    "text": "the quick brown fox jumps over the lazy dog",
                },
                {
                    "id": 2,
                    "text": "the quick brown cat jumps over the lazy dog",
                },
            ]))
            .unwrap();
        let txn = index.read_txn().unwrap();

        let query_graph = make_query_graph(&index, &txn, "the quick brown fox jumps over").unwrap();
        match &query_graph {
            super::QueryGraphOrPlaceholder::QueryGraph(qg) => {
                println!("{}", qg.graphviz());
            }
            super::QueryGraphOrPlaceholder::Placeholder => todo!(),
        }
        let universe = index.documents_ids(&txn).unwrap();
        // TODO: filters + maybe distinct attributes?
        let universe = initial(&index, &txn, query_graph.clone(), &universe).unwrap();
        println!("universe: {universe:?}");

        let results = execute_search(&index, &txn, &universe, query_graph, 0, 20).unwrap();
        println!("{results:?}")
    }
}
