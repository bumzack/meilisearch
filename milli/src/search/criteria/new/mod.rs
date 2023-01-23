// #![allow(unused)]

use std::borrow::Cow;

use roaring::RoaringBitmap;

use super::{resolve_query_tree, Context};
use crate::facet::FacetType;
use crate::heed_codec::facet::FacetGroupKeyCodec;
use crate::heed_codec::ByteSliceRefCodec;
use crate::search::facet::{ascending_facet_sort, descending_facet_sort};
use crate::search::query_tree::Operation;
use crate::search::WordDerivationsCache;
use crate::{FieldId, Index, Result};

pub trait CriterionIter {
    fn next(&mut self) -> Result<Option<CriterionResultNew>>;
}

pub struct BucketIterWrapper<'t> {
    iter: Box<dyn Iterator<Item = Result<CriterionResultNew>> + 't>,
}
impl<'t> BucketIterWrapper<'t> {
    fn new(iter: Box<dyn Iterator<Item = Result<CriterionResultNew>> + 't>) -> Self {
        Self { iter }
    }
}
impl<'t> CriterionIter for BucketIterWrapper<'t> {
    fn next(&mut self) -> Result<Option<CriterionResultNew>> {
        match self.iter.next() {
            Some(x) => x.map(Some),
            None => Ok(None),
        }
    }
}

trait Criterion2 {
    fn start(
        &mut self,
        parent_candidates: &RoaringBitmap,
        parent_query_tree: &QueryTreeOrPlaceholder,
        wdcache: &mut WordDerivationsCache,
    ) -> Result<()>;

    fn next(&mut self, wdcache: &mut WordDerivationsCache) -> Result<Option<CriterionResultNew>>;

    fn reset(&mut self);
}

#[derive(Debug)]
pub struct CriterionResultNew {
    /// The query tree that must be used by the child criterion to fetch candidates.
    #[allow(unused)]
    query_tree: QueryTreeOrPlaceholder,
    /// The allowed candidates for the child criteria
    candidates: RoaringBitmap,
}

#[derive(Debug)]
struct InitialCandidatesNew(UniverseCandidates);

#[derive(Debug, Clone)]
pub enum QueryTreeOrPlaceholder {
    QueryTree(Operation),
    Placeholder,
}

#[derive(Debug, Clone)]
enum CriterionCandidates<'a> {
    All,
    #[allow(unused)]
    Allowed(Cow<'a, RoaringBitmap>),
}
impl<'a> CriterionCandidates<'a> {
    // fn as_ref(&'a self) -> Self {
    //     match self {
    //         CriterionCandidates::All => CriterionCandidates::All,
    //         CriterionCandidates::Allowed(a) => CriterionCandidates::Allowed(Cow::Borrowed(a)),
    //     }
    // }
    fn into_universe_candidates(self) -> UniverseCandidates {
        match self {
            CriterionCandidates::All => UniverseCandidates::Excluded(RoaringBitmap::new()),
            CriterionCandidates::Allowed(a) => UniverseCandidates::Allowed(a.into_owned()),
        }
    }
    fn restrict_to_universe(
        self,
        universe: &UniverseCandidates,
        // TODO: should return a reference? should be cached?
        all: &impl Fn() -> RoaringBitmap,
    ) -> RoaringBitmap {
        match (self, universe) {
            (Self::All, UniverseCandidates::Allowed(a)) => a.clone(),
            (Self::All, UniverseCandidates::Excluded(e)) => {
                if e.is_empty() {
                    all()
                } else {
                    let mut ids = all();
                    ids -= e;
                    ids
                }
            }
            (Self::Allowed(s), UniverseCandidates::Allowed(a)) => {
                let mut s = s.into_owned();
                s &= a;
                s
            }
            (Self::Allowed(s), UniverseCandidates::Excluded(e)) => {
                let mut s = s.into_owned();
                s -= e;
                s
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum UniverseCandidates {
    Allowed(RoaringBitmap),
    Excluded(RoaringBitmap),
}

impl Default for UniverseCandidates {
    fn default() -> Self {
        Self::Excluded(RoaringBitmap::default())
    }
}
impl UniverseCandidates {
    fn restrict(&mut self, other: &Self) {
        let new = match (std::mem::take(self), other) {
            (UniverseCandidates::Allowed(mut s), UniverseCandidates::Allowed(a)) => {
                s &= a;
                UniverseCandidates::Allowed(s)
            }
            (UniverseCandidates::Allowed(mut s), UniverseCandidates::Excluded(e)) => {
                s -= e;
                UniverseCandidates::Allowed(s)
            }

            (UniverseCandidates::Excluded(s), UniverseCandidates::Allowed(a)) => {
                let mut a = a.clone();
                a -= &s;
                UniverseCandidates::Allowed(a)
            }
            (UniverseCandidates::Excluded(mut s), UniverseCandidates::Excluded(e)) => {
                s |= e;
                UniverseCandidates::Excluded(s)
            }
        };
        *self = new;
    }
}

// This should find the fastest way to resolve the query tree, taking shortcuts if necessary
fn resolve_query_tree_or_placeholder(
    ctx: &dyn Context,
    query_tree_or_placeholder: &QueryTreeOrPlaceholder,
    wdcache: &mut WordDerivationsCache,
) -> Result<CriterionCandidates<'static>> {
    match query_tree_or_placeholder {
        QueryTreeOrPlaceholder::QueryTree(qt) => {
            let r = resolve_query_tree(ctx, qt, wdcache)?;
            Ok(CriterionCandidates::Allowed(Cow::Owned(r)))
        }
        QueryTreeOrPlaceholder::Placeholder => Ok(CriterionCandidates::All),
    }
}

#[allow(unused)]
fn initial<'t>(
    ctx: &'t dyn Context<'t>,
    query_tree: QueryTreeOrPlaceholder,
    universe: &UniverseCandidates,
    // mut distinct: Option<D>,
    wdcache: &mut WordDerivationsCache,
) -> Result<UniverseCandidates> {
    // resolve the whole query tree to retrieve an exhaustive list of documents matching the query.
    // then remove the potential soft deleted documents.

    let mut candidates =
        resolve_query_tree_or_placeholder(ctx, &query_tree, wdcache)?.into_universe_candidates();
    candidates.restrict(universe);

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

enum WordsIter {
    Branches {
        branches: std::vec::IntoIter<Operation>,
        #[allow(unused)]
        parent_candidates: RoaringBitmap,
    },
    Placeholder {
        candidates: Option<RoaringBitmap>,
    },
}
pub struct Words<'ctx, 's> {
    ctx: &'ctx dyn Context<'ctx>,
    #[allow(unused)]
    universe: &'s UniverseCandidates,

    iter: Option<WordsIter>,
}
impl<'ctx, 's> Words<'ctx, 's> {
    fn new(ctx: &'ctx dyn Context<'ctx>, universe: &'s UniverseCandidates) -> Self {
        Self { ctx, universe, iter: None }
    }
}
impl<'ctx: 's, 's> Criterion2 for Words<'ctx, 's> {
    fn start(
        &mut self,
        parent_candidates: &RoaringBitmap,
        parent_query_tree: &QueryTreeOrPlaceholder,
        _wdcache: &mut WordDerivationsCache,
    ) -> Result<()> {
        match parent_query_tree {
            QueryTreeOrPlaceholder::QueryTree(qt) => {
                let branches = explode_query_tree(qt.clone());
                self.iter = Some(WordsIter::Branches {
                    branches: branches.into_iter(),
                    parent_candidates: parent_candidates.clone(),
                });
            }
            QueryTreeOrPlaceholder::Placeholder => {
                self.iter =
                    Some(WordsIter::Placeholder { candidates: Some(parent_candidates.clone()) });
            }
        };
        Ok(())
    }

    fn next(&mut self, wdcache: &mut WordDerivationsCache) -> Result<Option<CriterionResultNew>> {
        let iter = self.iter.as_mut().unwrap();
        match iter {
            WordsIter::Branches { branches, parent_candidates: _ } => {
                let Some(branch) = branches.next() else { return Ok(None) };
                let qt_candidates = resolve_query_tree(self.ctx, &branch, wdcache)?;
                Ok(Some(CriterionResultNew {
                    query_tree: QueryTreeOrPlaceholder::QueryTree(branch),
                    candidates: qt_candidates,
                }))
            }
            WordsIter::Placeholder { candidates } => {
                let Some(candidates) = candidates.take() else { return Ok(None) };
                Ok(Some(CriterionResultNew {
                    query_tree: QueryTreeOrPlaceholder::Placeholder,
                    candidates,
                }))
            }
        }
    }

    fn reset(&mut self) {
        self.iter = None;
    }
}

fn explode_query_tree(query_tree: Operation) -> Vec<Operation> {
    match query_tree {
        Operation::Or(true, ops) => ops,
        otherwise => vec![otherwise],
    }
}

pub struct Sort<'t> {
    index: &'t Index,
    rtxn: &'t heed::RoTxn<'t>,
    #[allow(unused)]
    field_name: String,
    field_id: Option<FieldId>,
    is_ascending: bool,
    iter: Option<BucketIterWrapper<'t>>,
}
impl<'t> Sort<'t> {
    fn new(
        index: &'t Index,
        rtxn: &'t heed::RoTxn,
        field_name: String,
        is_ascending: bool,
    ) -> Result<Self> {
        let fields_ids_map = index.fields_ids_map(rtxn)?;
        let field_id = fields_ids_map.id(&field_name);
        #[allow(unused)]
        let faceted_candidates = match field_id {
            Some(field_id) => {
                let number_faceted =
                    index.faceted_documents_ids(rtxn, field_id, FacetType::Number)?;
                let string_faceted =
                    index.faceted_documents_ids(rtxn, field_id, FacetType::String)?;
                number_faceted | string_faceted
            }
            None => RoaringBitmap::default(),
        };

        Ok(Self { index, rtxn, field_name, field_id, is_ascending, iter: None })
    }
}

impl<'t> Criterion2 for Sort<'t> {
    fn start(
        &mut self,
        parent_candidates: &RoaringBitmap,
        parent_query_tree: &QueryTreeOrPlaceholder,
        _wdcache: &mut WordDerivationsCache,
    ) -> Result<()> {
        let iter: BucketIterWrapper = match self.field_id {
            Some(field_id) => {
                let make_iter =
                    if self.is_ascending { ascending_facet_sort } else { descending_facet_sort };

                let number_iter = make_iter(
                    self.rtxn,
                    self.index
                        .facet_id_f64_docids
                        .remap_key_type::<FacetGroupKeyCodec<ByteSliceRefCodec>>(),
                    field_id,
                    parent_candidates.clone(),
                )?;

                let string_iter = make_iter(
                    self.rtxn,
                    self.index
                        .facet_id_string_docids
                        .remap_key_type::<FacetGroupKeyCodec<ByteSliceRefCodec>>(),
                    field_id,
                    parent_candidates.clone(),
                )?;
                let query_tree = parent_query_tree.clone();
                BucketIterWrapper::new(Box::new(number_iter.chain(string_iter).map(
                    move |docids| {
                        Ok(CriterionResultNew {
                            query_tree: query_tree.clone(),
                            candidates: docids?,
                        })
                    },
                )))
            }
            None => BucketIterWrapper::new(Box::new(std::iter::empty())),
        };
        self.iter = Some(iter);
        Ok(())
    }

    fn next(&mut self, _wdcache: &mut WordDerivationsCache) -> Result<Option<CriterionResultNew>> {
        let iter = self.iter.as_mut().unwrap();
        iter.next()
    }

    fn reset(&mut self) {
        self.iter = None;
    }
}

#[allow(clippy::if_same_then_else, unused)]
pub fn execute_search<'t>(
    index: &'t Index,
    rtxn: &'t heed::RoTxn,
    universe: &'t UniverseCandidates,
    query_tree: QueryTreeOrPlaceholder,
    ctx: &'t dyn Context<'t>,
    _wdcache: &'t mut WordDerivationsCache,
) -> Result<Vec<u32>> {
    let mut wdcache = WordDerivationsCache::default();

    let starting_candidates = CriterionCandidates::All
        .restrict_to_universe(universe, &|| index.documents_ids(rtxn).unwrap());

    let words = Words::new(ctx, universe);
    let sort = Sort::new(index, rtxn, "sort1".to_owned(), true)?;

    let mut criteria: Vec<Box<dyn Criterion2>> = vec![Box::new(words), Box::new(sort)];

    let mut cur_criterion_index = 0;

    let criteria_len = criteria.len();

    criteria[0].start(&starting_candidates, &query_tree, &mut wdcache)?;

    let mut results = vec![];

    macro_rules! back {
        () => {
            criteria[cur_criterion_index].reset();
            if cur_criterion_index == 0 {
                break;
            } else {
                cur_criterion_index -= 1;
            }
        };
    }

    while results.len() < 20 {
        let Some(next_bucket) = criteria[cur_criterion_index].next(&mut wdcache)? else {
            back!();
            continue;
        };
        if next_bucket.candidates.len() == 1 {
            results.extend(next_bucket.candidates);
            continue;
        } else if next_bucket.candidates.is_empty() {
            back!();
            continue;
        } else {
            // make next criterion iter or add to results
            cur_criterion_index += 1;
            if cur_criterion_index == criteria_len {
                results.extend(next_bucket.candidates);
            } else {
                // make new iterator from next criterion
                criteria[cur_criterion_index].start(
                    &next_bucket.candidates,
                    &next_bucket.query_tree,
                    &mut wdcache,
                );
            }
        }
    }

    Ok(results)
}
