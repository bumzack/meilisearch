// use roaring::RoaringBitmap;

// use crate::{
//     // facet::FacetType,
//     heed_codec::{facet::FacetGroupKeyCodec, ByteSliceRefCodec},
//     search::{
//         facet::{ascending_facet_sort, descending_facet_sort},
//         WordDerivationsCache,
//     },
//     FieldId,
//     Index,
//     Result,
// };

// use super::{
//     QueryTreeOrPlaceholder, RankingRule, RankingRuleOutput, RankingRuleOutputIter,
//     RankingRuleOutputIterWrapper,
// };

// pub struct Sort<'t> {
//     index: &'t Index,
//     rtxn: &'t heed::RoTxn<'t>,
//     field_id: Option<FieldId>,
//     is_ascending: bool,
//     iter: Option<RankingRuleOutputIterWrapper<'t>>,
// }
// impl<'t> Sort<'t> {
//     pub fn new(
//         index: &'t Index,
//         rtxn: &'t heed::RoTxn,
//         field_name: String,
//         is_ascending: bool,
//     ) -> Result<Self> {
//         let fields_ids_map = index.fields_ids_map(rtxn)?;
//         let field_id = fields_ids_map.id(&field_name);

//         // TODO: What is this, why?
//         // let faceted_candidates = match field_id {
//         //     Some(field_id) => {
//         //         let number_faceted =
//         //             index.faceted_documents_ids(rtxn, field_id, FacetType::Number)?;
//         //         let string_faceted =
//         //             index.faceted_documents_ids(rtxn, field_id, FacetType::String)?;
//         //         number_faceted | string_faceted
//         //     }
//         //     None => RoaringBitmap::default(),
//         // };

//         Ok(Self { index, rtxn, field_id, is_ascending, iter: None })
//     }
// }

// impl<'t> RankingRule for Sort<'t> {
//     fn init_bucket_iter(
//         &mut self,
//         parent_candidates: &RoaringBitmap,
//         parent_query_tree: &QueryTreeOrPlaceholder,
//         _wdcache: &mut WordDerivationsCache,
//     ) -> Result<()> {
//         let iter: RankingRuleOutputIterWrapper = match self.field_id {
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
//                     parent_candidates.clone(),
//                 )?;
//                 let query_tree = parent_query_tree.clone();
//                 RankingRuleOutputIterWrapper::new(Box::new(number_iter.chain(string_iter).map(
//                     move |docids| {
//                         Ok(RankingRuleOutput {
//                             query_tree: query_tree.clone(),
//                             candidates: docids?,
//                         })
//                     },
//                 )))
//             }
//             None => RankingRuleOutputIterWrapper::new(Box::new(std::iter::empty())),
//         };
//         self.iter = Some(iter);
//         Ok(())
//     }

//     fn next_bucket(
//         &mut self,
//         _wdcache: &mut WordDerivationsCache,
//     ) -> Result<Option<RankingRuleOutput>> {
//         let iter = self.iter.as_mut().unwrap();
//         iter.next_bucket()
//     }

//     fn reset_bucket_iter(&mut self) {
//         self.iter = None;
//     }
// }
