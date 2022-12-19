// use std::borrow::Cow;
use std::cell::{RefCell, RefMut};
use std::rc::Rc;

// use roaring::RoaringBitmap;

// use super::{resolve_query_tree, Context, CriteriaBuilder};
// use crate::facet::FacetType;
// use crate::heed_codec::facet::FacetGroupKeyCodec;
// use crate::heed_codec::ByteSliceRefCodec;
// use crate::search::facet::{ascending_facet_sort, descending_facet_sort};
// use crate::search::query_tree::Operation;
// use crate::search::WordDerivationsCache;
// use crate::{FieldId, Index, Result};

// // need to be able to give back an owned version of the criterion
// pub type CriterionBucketIterator<'s> =
//     Box<dyn (Iterator<Item = Result<CriterionResultNew<'s>>>) + 's>;

// pub trait CriterionIter<'s> {
//     fn next(&mut self) -> Option<Result<CriterionResultNew<'s>>>;
// }

struct Y {
    x: u32,
}

struct Xs {
    v: Vec<Rc<RefCell<Y>>>,
}

impl Xs {
    fn get(&self, i: usize) -> RefMut<Y> {
        self.v[i].borrow_mut()
    }
    fn get2(&self, i: usize) -> (RefMut<Y>, RefMut<Y>) {
        (self.get(i), self.get(i))
    }
    fn dostuff(&self) {
        let (mut a, mut b) = self.get2(0);
        a.x += 1;
        b.x += 1;
    }
}

// trait Criterion2<'s> {
//     type Iter: CriterionIter<'s>;

//     fn to_bucket_iterator(
//         self,
//         parent_candidates: &'s CriterionCandidates,
//         parent_query_tree: &'s QueryTreeOrPlaceholder,
//         wdcache: &'s mut WordDerivationsCache,
//     ) -> Result<Self::Iter>;
//     fn to_criterion(iter: Self::Iter) -> Self;
// }

// // trait CriterionNew {
// //     // fn new(ctx: &'ctx dyn Context<'ctx>, universe: &'s UniverseCandidates) -> Self;
// //     fn bucket_iterator<'s, 's: 's>(
// //         &'s mut self,
// //         parent_candidates: &'s CriterionCandidates,
// //         parent_query_tree: &'s QueryTreeOrPlaceholder,
// //         wdcache: &'s mut WordDerivationsCache,
// //     ) -> Result<CriterionBucketIterator<'s>>;
// // }

// #[derive(Debug)]
// pub struct CriterionResultNew<'s> {
//     /// The query tree that must be used by the child criterion to fetch candidates.
//     query_tree: Cow<'s, QueryTreeOrPlaceholder>,
//     /// The allowed candidates for the child criteria
//     candidates: Cow<'s, CriterionCandidates<'s>>,
// }

// #[derive(Debug)]
// struct InitialCandidatesNew(UniverseCandidates);

// #[derive(Debug, Clone)]
// enum QueryTreeOrPlaceholder {
//     QueryTree(Operation),
//     Placeholder,
// }

// #[derive(Debug, Clone)]
// enum CriterionCandidates<'s> {
//     FullyComputed(ResolvedCriterionCandidates<'s>),
//     NotFullyComputed(ResolvedCriterionCandidates<'s>),
// }
// impl<'s> CriterionCandidates<'s> {
//     fn make_not_fully_computed(&'s self) -> Self {
//         match self {
//             CriterionCandidates::FullyComputed(c) => {
//                 CriterionCandidates::NotFullyComputed(c.as_ref())
//             }
//             CriterionCandidates::NotFullyComputed(parent) => {
//                 CriterionCandidates::NotFullyComputed(parent.as_ref())
//             }
//         }
//     }
//     fn known_length(&'s self) -> Option<u64> {
//         match self {
//             CriterionCandidates::FullyComputed(c) => c.known_length(),
//             CriterionCandidates::NotFullyComputed(_) => None,
//         }
//     }
// }
// #[derive(Debug, Clone)]
// enum ResolvedCriterionCandidates<'a> {
//     All,
//     Allowed(Cow<'a, RoaringBitmap>),
// }
// impl<'a> ResolvedCriterionCandidates<'a> {
//     fn as_ref(&'a self) -> Self {
//         match self {
//             ResolvedCriterionCandidates::All => ResolvedCriterionCandidates::All,
//             ResolvedCriterionCandidates::Allowed(a) => {
//                 ResolvedCriterionCandidates::Allowed(Cow::Borrowed(&a))
//             }
//         }
//     }
//     fn into_universe_candidates(self) -> UniverseCandidates {
//         match self {
//             ResolvedCriterionCandidates::All => UniverseCandidates::Excluded(RoaringBitmap::new()),
//             ResolvedCriterionCandidates::Allowed(a) => UniverseCandidates::Allowed(a.into_owned()),
//         }
//     }
//     fn restrict_to_universe(
//         &mut self,
//         universe: &UniverseCandidates,
//         // TODO: should return a reference? should be cached?
//         all: impl FnOnce() -> RoaringBitmap,
//     ) {
//         let new = match (std::mem::replace(self, ResolvedCriterionCandidates::All), universe) {
//             (Self::All, UniverseCandidates::Allowed(a)) => {
//                 // TODO: make borrowed instead of Owned
//                 Self::Allowed(Cow::Owned(a.clone()))
//             }
//             (Self::All, UniverseCandidates::Excluded(e)) => {
//                 if e.is_empty() {
//                     Self::All
//                 } else {
//                     let mut ids = all();
//                     ids -= e;
//                     Self::Allowed(Cow::Owned(ids))
//                 }
//             }
//             (Self::Allowed(s), UniverseCandidates::Allowed(a)) => {
//                 let mut s = s.into_owned();
//                 s &= a;
//                 Self::Allowed(Cow::Owned(s))
//             }
//             (Self::Allowed(s), UniverseCandidates::Excluded(e)) => {
//                 let mut s = s.into_owned();
//                 s -= e;
//                 Self::Allowed(Cow::Owned(s))
//             }
//         };
//         *self = new;
//     }

//     fn restrict(&mut self, other: &'a Self) {
//         let new = match (std::mem::replace(self, ResolvedCriterionCandidates::All), other) {
//             (s, ResolvedCriterionCandidates::All) => s,
//             (ResolvedCriterionCandidates::All, other) => other.as_ref(),
//             (ResolvedCriterionCandidates::Allowed(s), ResolvedCriterionCandidates::Allowed(a)) => {
//                 let mut s = s.into_owned();
//                 s &= a.as_ref();
//                 ResolvedCriterionCandidates::Allowed(Cow::Owned(s))
//             }
//         };
//         *self = new;
//     }
//     fn known_length(&self) -> Option<u64> {
//         match self {
//             ResolvedCriterionCandidates::All => None,
//             ResolvedCriterionCandidates::Allowed(a) => Some(a.len()),
//         }
//     }
// }

// #[derive(Debug, Clone)]
// enum UniverseCandidates {
//     Allowed(RoaringBitmap),
//     Excluded(RoaringBitmap),
// }

// impl Default for UniverseCandidates {
//     fn default() -> Self {
//         Self::Excluded(RoaringBitmap::default())
//     }
// }
// impl UniverseCandidates {
//     fn is_empty(&self) -> bool {
//         match self {
//             UniverseCandidates::Allowed(a) => a.is_empty(),
//             UniverseCandidates::Excluded(_) => false,
//         }
//     }
//     fn restrict(&mut self, other: &Self) {
//         let new = match (std::mem::take(self), other) {
//             (UniverseCandidates::Allowed(mut s), UniverseCandidates::Allowed(a)) => {
//                 s &= a;
//                 UniverseCandidates::Allowed(s)
//             }
//             (UniverseCandidates::Allowed(mut s), UniverseCandidates::Excluded(e)) => {
//                 s -= e;
//                 UniverseCandidates::Allowed(s)
//             }

//             (UniverseCandidates::Excluded(s), UniverseCandidates::Allowed(a)) => {
//                 let mut a = a.clone();
//                 a -= &s;
//                 UniverseCandidates::Allowed(a)
//             }
//             (UniverseCandidates::Excluded(mut s), UniverseCandidates::Excluded(e)) => {
//                 s |= e;
//                 UniverseCandidates::Excluded(s)
//             }
//         };
//         *self = new;
//     }
// }

// // This should find the fastest way to resolve the query tree, taking shortcuts if necessary
// fn resolve_query_tree_or_placeholder(
//     ctx: &dyn Context,
//     query_tree_or_placeholder: &QueryTreeOrPlaceholder,
//     wdcache: &mut WordDerivationsCache,
// ) -> Result<ResolvedCriterionCandidates<'static>> {
//     match query_tree_or_placeholder {
//         QueryTreeOrPlaceholder::QueryTree(qt) => {
//             let r = resolve_query_tree(ctx, qt, wdcache)?;
//             Ok(ResolvedCriterionCandidates::Allowed(Cow::Owned(r)))
//         }
//         QueryTreeOrPlaceholder::Placeholder => Ok(ResolvedCriterionCandidates::All),
//     }
// }

// fn initial<'t>(
//     ctx: &'t dyn Context<'t>,
//     query_tree: QueryTreeOrPlaceholder,
//     universe: &UniverseCandidates,
//     // mut distinct: Option<D>,
//     wdcache: &mut WordDerivationsCache,
// ) -> Result<UniverseCandidates> {
//     // resolve the whole query tree to retrieve an exhaustive list of documents matching the query.
//     // then remove the potential soft deleted documents.

//     let mut candidates =
//         resolve_query_tree_or_placeholder(ctx, &query_tree, wdcache)?.into_universe_candidates();
//     candidates.restrict(universe);

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

// pub struct WordsBucketIterator<'ctx, 's> {
//     criterion: Words<'ctx, 's>,
//     iter: Box<dyn Iterator<Item = Result<CriterionResultNew<'s>>> + 's>,
// }
// impl<'ctx, 's> CriterionIter<'s> for WordsBucketIterator<'ctx, 's> {
//     fn next(&mut self) -> Option<Result<CriterionResultNew<'s>>> {
//         self.iter.next()
//     }
// }

// pub struct Words<'ctx, 's> {
//     ctx: &'ctx dyn Context<'ctx>,
//     universe: &'s UniverseCandidates,
// }
// impl<'ctx, 's> Words<'ctx, 's> {
//     fn new(ctx: &'ctx dyn Context<'ctx>, universe: &'s UniverseCandidates) -> Self {
//         Self { ctx, universe }
//     }
// }
// impl<'ctx: 's, 's> Criterion2<'s> for Words<'ctx, 's> {
//     type Iter = WordsBucketIterator<'ctx, 's>;

//     fn to_bucket_iterator(
//         self,
//         parent_candidates: &'s CriterionCandidates,
//         parent_query_tree: &'s QueryTreeOrPlaceholder,
//         wdcache: &'s mut WordDerivationsCache,
//     ) -> Result<Self::Iter> {
//         let iter: CriterionBucketIterator<'s> = match parent_query_tree {
//             QueryTreeOrPlaceholder::QueryTree(qt) => {
//                 let branches = explode_query_tree(qt.clone());
//                 Box::new(branches.into_iter().map(move |query_tree| {
//                     Ok(CriterionResultNew {
//                         query_tree: Cow::Owned(QueryTreeOrPlaceholder::QueryTree(query_tree)),
//                         candidates: Cow::Owned(parent_candidates.make_not_fully_computed()),
//                     })
//                 }))
//             }
//             QueryTreeOrPlaceholder::Placeholder => {
//                 Box::new(std::iter::once(Ok(CriterionResultNew {
//                     query_tree: Cow::Owned(QueryTreeOrPlaceholder::Placeholder),
//                     candidates: Cow::Borrowed(parent_candidates),
//                 })))
//             }
//         };
//         Ok(WordsBucketIterator { criterion: self, iter })
//     }

//     fn to_criterion(iter: Self::Iter) -> Self {
//         iter.criterion
//     }
// }

// fn explode_query_tree(query_tree: Operation) -> Vec<Operation> {
//     match query_tree {
//         Operation::Or(true, ops) => ops,
//         otherwise => vec![otherwise],
//     }
// }

// pub struct Sort<'t, 's> {
//     index: &'t Index,
//     rtxn: &'t heed::RoTxn<'t>,
//     field_name: String,
//     field_id: Option<FieldId>,
//     is_ascending: bool,

//     universe: &'s UniverseCandidates,
// }
// impl<'t, 's> Sort<'t, 's> {
//     fn new(
//         index: &'t Index,
//         rtxn: &'t heed::RoTxn,
//         field_name: String,
//         is_ascending: bool,
//         universe: &'s UniverseCandidates,
//     ) -> Result<Self> {
//         let fields_ids_map = index.fields_ids_map(rtxn)?;
//         let field_id = fields_ids_map.id(&field_name);
//         let faceted_candidates = match field_id {
//             Some(field_id) => {
//                 let number_faceted =
//                     index.faceted_documents_ids(rtxn, field_id, FacetType::Number)?;
//                 let string_faceted =
//                     index.faceted_documents_ids(rtxn, field_id, FacetType::String)?;
//                 number_faceted | string_faceted
//             }
//             None => RoaringBitmap::default(),
//         };

//         Ok(Self { index, rtxn, field_name, field_id, is_ascending, universe })
//     }
// }
// struct SortBucketIter<'t, 's> {
//     criterion: Sort<'t, 's>,
//     iter: CriterionBucketIterator<'s>,
// }
// impl<'t, 's> CriterionIter<'s> for SortBucketIter<'t, 's> {
//     fn next(&mut self) -> Option<Result<CriterionResultNew<'s>>> {
//         self.iter.next()
//     }
// }
// impl<'t, 's> Criterion2<'s> for Sort<'t, 's> {
//     type Iter = SortBucketIter<'t, 's>;

//     fn to_bucket_iterator(
//         self,
//         parent_candidates: &'s CriterionCandidates,
//         parent_query_tree: &'s QueryTreeOrPlaceholder,
//         wdcache: &mut WordDerivationsCache,
//     ) -> Result<Self::Iter> {
//         // factor out the code below in a function
//         let resolved_parent_candidates = match parent_candidates {
//             CriterionCandidates::FullyComputed(r) => r.as_ref(),
//             CriterionCandidates::NotFullyComputed(parent) => {
//                 let context = CriteriaBuilder::new(self.rtxn, self.index)?;
//                 let mut resolved_qt_candidates =
//                     resolve_query_tree_or_placeholder(&context, parent_query_tree, wdcache)?;
//                 resolved_qt_candidates.restrict(parent);
//                 resolved_qt_candidates
//             }
//         };

//         let parent_candidates = match resolved_parent_candidates {
//             ResolvedCriterionCandidates::All => self.index.documents_ids(self.rtxn)?,
//             ResolvedCriterionCandidates::Allowed(a) => a.into_owned(),
//         };

//         let iter: CriterionBucketIterator = match self.field_id {
//             Some(field_id) => {
//                 let make_iter =
//                     if self.is_ascending { ascending_facet_sort } else { descending_facet_sort };

//                 let number_iter = make_iter(
//                     self.rtxn,
//                     self.index
//                         .facet_id_f64_docids
//                         .remap_key_type::<FacetGroupKeyCodec<ByteSliceRefCodec>>(),
//                     field_id,
//                     parent_candidates.clone(),
//                 )?;

//                 let string_iter = make_iter(
//                     self.rtxn,
//                     self.index
//                         .facet_id_string_docids
//                         .remap_key_type::<FacetGroupKeyCodec<ByteSliceRefCodec>>(),
//                     field_id,
//                     parent_candidates,
//                 )?;

//                 Box::new(number_iter.chain(string_iter).map(move |docids| {
//                     Ok(CriterionResultNew {
//                         query_tree: Cow::Borrowed(parent_query_tree),
//                         candidates: Cow::Owned(CriterionCandidates::FullyComputed(
//                             ResolvedCriterionCandidates::Allowed(Cow::Owned(docids?)),
//                         )),
//                     })
//                 }))
//             }
//             None => Box::new(std::iter::empty()),
//         };
//         Ok(SortBucketIter { criterion: self, iter })
//     }

//     fn to_criterion(iter: Self::Iter) -> Self {
//         iter.criterion
//     }
// }

// #[allow(clippy::if_same_then_else)]
// fn execute_search<'t>(
//     index: &'t Index,
//     rtxn: &'t heed::RoTxn,
//     universe: &'t UniverseCandidates,
//     query_tree: QueryTreeOrPlaceholder,
//     ctx: &'t dyn Context<'t>,
//     wdcache: &'t mut WordDerivationsCache,
// ) -> Result<Vec<u32>> {
//     let mut wdcache = WordDerivationsCache::default();

//     let mut words = Words::new(ctx, universe);
//     let mut sort = Sort::new(index, rtxn, "sort1".to_owned(), true, universe)?;
//     let criteria: Vec<Rc<RefCell<dyn Criterion2>>> =
//         vec![Rc::new(RefCell::new(words)), Rc::new(RefCell::new(sort))];

//     let mut starting_candidates = ResolvedCriterionCandidates::All;
//     starting_candidates.restrict_to_universe(universe, || index.documents_ids(rtxn).unwrap());
//     let starting_candidates = CriterionCandidates::FullyComputed(starting_candidates);

//     let mut cur_criterion_index = 0;

//     let criteria_len = criteria.len();
//     // for i in 0..criteria_len {
//     //     borrowed_criteria.push(criteria[i].borrow_mut());
//     // }

//     let starting_iter = criteria[0].borrow_mut().bucket_iterator(
//         &starting_candidates,
//         &QueryTreeOrPlaceholder::Placeholder,
//         &mut wdcache,
//     )?;

//     let mut criterion_iters = vec![starting_iter];

//     let mut results = vec![];

//     while let Some(cur_iter) = criterion_iters.last_mut() {
//         if results.len() > 20 {
//             break;
//         }
//         // TODO: condition on results length as well

//         let Some(next_bucket) = cur_iter.next() else {
//             // TODO: ?
//             criterion_iters.pop();
//             continue;
//         };
//         let next_bucket = next_bucket?;
//         if matches!(next_bucket.candidates.known_length(), Some(1)) {
//             // TODO: push into results;
//             continue;
//         } else if matches!(next_bucket.candidates.known_length(), Some(0)) {
//             // TODO: ?
//             criterion_iters.pop();
//             continue;
//         } else {
//             // make next criterion iter or add to results
//             cur_criterion_index += 1;
//             if cur_criterion_index == criteria_len {
//                 // TODO: push into results;
//             } else {
//                 // make new iterator from next criterion
//             }
//         }
//     }

//     Ok(results)
// }
