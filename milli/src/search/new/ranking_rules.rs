// use crate::search::new::sort::Sort;
// use crate::search::new::words::Words;
use crate::search::query_tree::Operation;
use crate::search::WordDerivationsCache;
use crate::Result;
use roaring::RoaringBitmap;

pub trait RankingRuleOutputIter {
    fn next_bucket(&mut self) -> Result<Option<RankingRuleOutput>>;
}

pub struct RankingRuleOutputIterWrapper<'t> {
    iter: Box<dyn Iterator<Item = Result<RankingRuleOutput>> + 't>,
}
impl<'t> RankingRuleOutputIterWrapper<'t> {
    pub fn new(iter: Box<dyn Iterator<Item = Result<RankingRuleOutput>> + 't>) -> Self {
        Self { iter }
    }
}
impl<'t> RankingRuleOutputIter for RankingRuleOutputIterWrapper<'t> {
    fn next_bucket(&mut self) -> Result<Option<RankingRuleOutput>> {
        match self.iter.next() {
            Some(x) => x.map(Some),
            None => Ok(None),
        }
    }
}

pub trait RankingRule {
    fn init_bucket_iter(
        &mut self,
        parent_candidates: &RoaringBitmap,
        parent_query_tree: &QueryTreeOrPlaceholder,
        wdcache: &mut WordDerivationsCache,
    ) -> Result<()>;

    fn next_bucket(
        &mut self,
        wdcache: &mut WordDerivationsCache,
    ) -> Result<Option<RankingRuleOutput>>;

    fn reset_bucket_iter(&mut self);
}

#[derive(Debug)]
pub struct RankingRuleOutput {
    /// The query tree that must be used by the child ranking_rule to fetch candidates.
    query_tree: QueryTreeOrPlaceholder,
    /// The allowed candidates for the child ranking_rules
    candidates: RoaringBitmap,
}

#[derive(Debug, Clone)]
pub enum QueryTreeOrPlaceholder {
    QueryTree(Operation),
    Placeholder,
}

// TODO: uncomment this below

// // This should find the fastest way to resolve the query tree, taking shortcuts if necessary
// fn resolve_query_tree_or_placeholder(
//     index: &Index,
//     query_tree_or_placeholder: &QueryTreeOrPlaceholder,
//     wdcache: &mut WordDerivationsCache,
//     all: &RoaringBitmap,
// ) -> Result<RoaringBitmap> {
//     match query_tree_or_placeholder {
//         QueryTreeOrPlaceholder::QueryTree(qt) => {
//             let r = resolve_query_tree(ctx, qt, wdcache)?;
//             Ok(r)
//         }
//         QueryTreeOrPlaceholder::Placeholder => Ok(all.clone()),
//     }
// }

// #[allow(unused)]
// pub fn initial<'t>(
//     ctx: &'t dyn Context<'t>,
//     query_tree: QueryTreeOrPlaceholder,
//     universe: &RoaringBitmap,
//     // mut distinct: Option<D>,
//     wdcache: &mut WordDerivationsCache,
// ) -> Result<RoaringBitmap> {
//     // resolve the whole query tree to retrieve an exhaustive list of documents matching the query.
//     // then remove the potential soft deleted documents.

//     let candidates = resolve_query_tree_or_placeholder(ctx, &query_tree, wdcache, universe)?;

//     // Distinct should be lazy if placeholder?
//     //
//     // // because the initial_candidates should be an exhaustive count of the matching documents,
//     // // we precompute the distinct attributes.
//     // let initial_candidates = match &mut distinct {
//     //     Some(distinct) => {
//     //         let mut initial_candidates = RoaringBitmap::new();
//     //         for c in distinct.distinct(candidates.clone(), RoaringBitmap::new()) {
//     //             initial_candidates.insert(c?);
//     //         }
//     //         initial_candidates
//     //     }
//     //     None => candidates.clone(),
//     // };

//     Ok(candidates)
// }

// #[allow(unused)]
// pub fn execute_search<'t>(
//     index: &'t Index,
//     rtxn: &'t heed::RoTxn,
//     universe: &'t RoaringBitmap,
//     query_tree: QueryTreeOrPlaceholder,
//     ctx: &'t dyn Context<'t>,
//     _wdcache: &'t mut WordDerivationsCache,
// ) -> Result<Vec<u32>> {
//     let mut wdcache = WordDerivationsCache::default();

//     let words = Words::new(ctx, universe);
//     let sort = Sort::new(index, rtxn, "sort1".to_owned(), true)?;
//     let mut ranking_rules: Vec<Box<dyn RankingRule>> = vec![Box::new(words), Box::new(sort)];
//     let mut cur_ranking_rule_index = 0;

//     macro_rules! back {
//         () => {
//             ranking_rules[cur_ranking_rule_index].reset_bucket_iter();
//             if cur_ranking_rule_index == 0 {
//                 break;
//             } else {
//                 cur_ranking_rule_index -= 1;
//             }
//         };
//     }

//     let ranking_rules_len = ranking_rules.len();
//     ranking_rules[0].init_bucket_iter(universe, &query_tree, &mut wdcache)?;

//     let mut results = vec![];

//     while results.len() < 20 {
//         let Some(next_bucket) = ranking_rules[cur_ranking_rule_index].next_bucket(&mut wdcache)? else {
//             back!();
//             continue;
//         };
//         match next_bucket.candidates.len() {
//             0 => {
//                 // no progress anymore, go to the parent candidate
//                 back!();
//                 continue;
//             }
//             1 => {
//                 // only one candidate, no need to sort through the child ranking rule
//                 results.extend(next_bucket.candidates);
//                 continue;
//             }
//             _ => {
//                 // many candidates, give to next ranking rule, if any
//                 if cur_ranking_rule_index == ranking_rules_len - 1 {
//                     results.extend(next_bucket.candidates);
//                 } else {
//                     cur_ranking_rule_index += 1;
//                     // make new iterator from next ranking_rule
//                     ranking_rules[cur_ranking_rule_index].init_bucket_iter(
//                         &next_bucket.candidates,
//                         &next_bucket.query_tree,
//                         &mut wdcache,
//                     );
//                 }
//             }
//         }
//     }

//     Ok(results)
// }
