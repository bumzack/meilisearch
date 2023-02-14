use heed::RoTxn;
use roaring::RoaringBitmap;

use super::resolve_query_graph::resolve_query_graph;
use super::{QueryGraphOrPlaceholder, RankingRule, RankingRuleOutput};
use crate::{Index, Result};

pub struct Words {
    universe: RoaringBitmap,
    query_graph: QueryGraphOrPlaceholder,
    exhausted: bool,
}
impl Default for Words {
    fn default() -> Self {
        Self {
            universe: RoaringBitmap::default(),
            query_graph: QueryGraphOrPlaceholder::Placeholder,
            exhausted: true,
        }
    }
}

impl<'transaction> RankingRule<'transaction> for Words {
    fn init_bucket_iter(
        &mut self,
        _index: &Index,
        _txn: &'transaction RoTxn,
        parent_candidates: &RoaringBitmap,
        parent_query_graph: &QueryGraphOrPlaceholder,
    ) -> Result<()> {
        self.universe = parent_candidates.clone();
        self.query_graph = parent_query_graph.clone();
        self.exhausted = false;
        Ok(())
    }

    fn next_bucket(
        &mut self,
        index: &Index,
        txn: &'transaction RoTxn,
    ) -> Result<Option<RankingRuleOutput>> {
        if self.exhausted {
            return Ok(None);
        }
        match self.query_graph.clone() {
            QueryGraphOrPlaceholder::QueryGraph(q) => {
                let next_bucket = resolve_query_graph(index, txn, &q, self.universe.clone())?;
                let cur_bucket = std::mem::replace(&mut self.universe, next_bucket);
                // TODO: find the nodes to remove and call q.remove_nodes(nodes);
                // pass the old `q` to the child ranking rule through the RankingRuleOutput
                // set the current query graph to `q`
                self.exhausted = true;
                Ok(Some(RankingRuleOutput {
                    query_graph: QueryGraphOrPlaceholder::QueryGraph(q),
                    candidates: cur_bucket,
                }))
            }
            QueryGraphOrPlaceholder::Placeholder => {
                self.exhausted = true;
                Ok(Some(RankingRuleOutput {
                    query_graph: QueryGraphOrPlaceholder::Placeholder,
                    candidates: std::mem::take(&mut self.universe),
                }))
            }
        }
    }

    fn reset_bucket_iter(&mut self, _index: &Index, _txn: &'transaction RoTxn) {
        self.exhausted = true;
    }
}

#[cfg(test)]
mod tests {
    // use charabia::Tokenize;
    // use roaring::RoaringBitmap;

    // use crate::{
    //     index::tests::TempIndex,
    //     search::{criteria::CriteriaBuilder, new::QueryGraphOrPlaceholder},
    // };

    // use super::Words;

    // fn placeholder() {
    //     let qt = QueryGraphOrPlaceholder::Placeholder;
    //     let index = TempIndex::new();
    //     let rtxn = index.read_txn().unwrap();

    //     let query = "a beautiful summer house by the beach overlooking what seems";
    //     // let mut builder = QueryTreeBuilder::new(&rtxn, &index).unwrap();
    //     // let (qt, parts, matching_words) = builder.build(query.tokenize()).unwrap().unwrap();

    //     // let cb = CriteriaBuilder::new(&rtxn, &index).unwrap();
    //     // let x = cb
    //     //     .build(
    //     //         Some(qt),
    //     //         Some(parts),
    //     //         None,
    //     //         None,
    //     //         false,
    //     //         None,
    //     //         crate::CriterionImplementationStrategy::OnlySetBased,
    //     //     )
    //     //     .unwrap();

    //     // let rr = Words::new(&index, &RoaringBitmap::from_sorted_iter(0..1000)).unwrap();
    // }
}
