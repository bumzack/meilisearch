// use roaring::RoaringBitmap;

// use crate::{
//     search::{criteria::resolve_query_tree, query_tree::Operation, WordDerivationsCache},
//     Index, Result,
// };

// use super::{QueryTreeOrPlaceholder, RankingRule, RankingRuleOutput};

// enum WordsIter {
//     Branches { branches: std::vec::IntoIter<Operation> },
//     Placeholder { candidates: Option<RoaringBitmap> },
// }
// pub struct Words<'ctx, 'search> {
//     index: &'ctx Index,
//     #[allow(unused)]
//     universe: &'search RoaringBitmap,

//     iter: Option<WordsIter>,
// }
// impl<'index, 'search> Words<'index, 'search> {
//     pub fn new(index: &'index Index, universe: &'search RoaringBitmap) -> Self {
//         Self { index, universe, iter: None }
//     }
// }
// impl<'index: 'search, 'search> RankingRule for Words<'index, 'search> {
//     fn init_bucket_iter(
//         &mut self,
//         parent_candidates: &RoaringBitmap,
//         parent_query_tree: &QueryTreeOrPlaceholder,
//         _wdcache: &mut WordDerivationsCache,
//     ) -> Result<()> {
//         match parent_query_tree {
//             QueryTreeOrPlaceholder::QueryTree(qt) => {
//                 let branches = split_query_tree(qt.clone());
//                 self.iter = Some(WordsIter::Branches { branches: branches.into_iter() });
//             }
//             QueryTreeOrPlaceholder::Placeholder => {
//                 self.iter =
//                     Some(WordsIter::Placeholder { candidates: Some(parent_candidates.clone()) });
//             }
//         };
//         Ok(())
//     }

//     fn next_bucket(
//         &mut self,
//         wdcache: &mut WordDerivationsCache,
//     ) -> Result<Option<RankingRuleOutput>> {
//         let iter = self.iter.as_mut().unwrap();
//         match iter {
//             WordsIter::Branches { branches } => {
//                 let Some(branch) = branches.next() else { return Ok(None) };
//                 let qt_candidates = resolve_query_tree(self.index, &branch, wdcache)?;
//                 Ok(Some(RankingRuleOutput {
//                     query_tree: QueryTreeOrPlaceholder::QueryTree(branch),
//                     candidates: qt_candidates,
//                 }))
//             }
//             WordsIter::Placeholder { candidates } => {
//                 let Some(candidates) = candidates.take() else { return Ok(None) };
//                 Ok(Some(RankingRuleOutput {
//                     query_tree: QueryTreeOrPlaceholder::Placeholder,
//                     candidates,
//                 }))
//             }
//         }
//     }

//     fn reset_bucket_iter(&mut self) {
//         self.iter = None;
//     }
// }

// fn split_query_tree(query_tree: Operation) -> Vec<Operation> {
//     match query_tree {
//         Operation::Or(true, ops) => ops,
//         otherwise => vec![otherwise],
//     }
// }

// #[cfg(test)]
// mod tests {
//     use charabia::Tokenize;
//     use roaring::RoaringBitmap;

//     use crate::{
//         index::tests::TempIndex,
//         search::{
//             criteria::CriteriaBuilder, new::QueryTreeOrPlaceholder, query_tree::QueryTreeBuilder,
//         },
//     };

//     use super::Words;

//     fn placeholder() {
//         let qt = QueryTreeOrPlaceholder::Placeholder;
//         let index = TempIndex::new();
//         let rtxn = index.read_txn().unwrap();

//         let query = "a beautiful summer house by the beach overlooking what seems";
//         let mut builder = QueryTreeBuilder::new(&rtxn, &index).unwrap();
//         let (qt, parts, matching_words) = builder.build(query.tokenize()).unwrap().unwrap();

//         let cb = CriteriaBuilder::new(&rtxn, &index).unwrap();
//         let x = cb
//             .build(
//                 Some(qt),
//                 Some(parts),
//                 None,
//                 None,
//                 false,
//                 None,
//                 crate::CriterionImplementationStrategy::OnlySetBased,
//             )
//             .unwrap();

//         let rr = Words::new(&index, &RoaringBitmap::from_sorted_iter(0..1000)).unwrap();
//     }
// }
