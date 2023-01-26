use crate::{Index, Result};

use super::{query_term::QueryTerm, QueryGraph};
use roaring::RoaringBitmap;

pub fn resolve_query_graph(
    index: &Index,
    q: &QueryGraph,
    universe: RoaringBitmap,
) -> Result<RoaringBitmap> {
    // TODO: walk the query graph
    // for each new node, do an AND
    // for each alternative path, do an OR

    // maybe memoize some stuff
    // problem: many word derivations possible due to typos and prefixes!
    // but these things should be precomputed as much as possible?
    // we can also limit them easily, i.e. no more than 10 word derivations for typos allowed
    //    + we won't look for every word derived from the prefix because of the prefix databases

    // also after every ranking rule, we trim the query graph further?
    // there should also be a way to cache the results of database and roaring bitmap operations between
    // calls to ranking rules operating on different query graphs

    // TODO: add word derivations to the query graph
    todo!()
}

fn resolve_query_graph_rec(
    index: &Index,
    q: &QueryGraph,
    node: usize,
    docids: &mut RoaringBitmap,
) -> Result<()> {
    let n = &q.nodes[node];
    match n {
        super::QueryNode::Term(located_term) => {
            let term = &located_term.value;
            match term {
                QueryTerm::Phrase(_) => todo!(),
                QueryTerm::Word { original, derivations } => {
                    // here take the intersection of the existing bitmap and the docids of the OR of the rest
                }
            }
        }
        super::QueryNode::Deleted => {}
        super::QueryNode::Start => {}
        super::QueryNode::End => return Ok(()),
    }

    // then for each successor, do an AND

    // then take the OR of these ANDS

    // should memoize from the end?

    todo!()
}
