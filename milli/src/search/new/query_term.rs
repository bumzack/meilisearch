// TODO: put primitive query part in here

use crate::{Index, Result};
use charabia::{normalizer::NormalizedTokenIter, SeparatorKind, TokenKind};
use std::{mem, ops::RangeInclusive};

#[derive(Debug, Clone)]
pub enum WordDerivations {
    // TODO: The list should be split by number of typos
    // TODO: Should distinguish between typos and prefixes as well
    FromList(Vec<String>),
    FromPrefixDB,
}

pub fn word_derivations(
    index: &Index,
    word: &str,
    is_prefix: bool,
    fst: &fst::Set<Vec<u8>>,
) -> Result<WordDerivations> {
    todo!()
}

#[derive(Debug, Clone)]
pub enum QueryTerm {
    Phrase(Vec<Option<String>>),
    Word { original: String, derivations: WordDerivations },
}

#[derive(Debug, Clone)]
pub struct LocatedQueryTerm {
    pub value: QueryTerm, // value should be able to contain the word derivations as well
    pub positions: RangeInclusive<i8>,
}

impl LocatedQueryTerm {
    /// Create primitive query from tokenized query string,
    /// the primitive query is an intermediate state to build the query tree.
    pub fn from_query(
        query: NormalizedTokenIter<Vec<u8>>,
        words_limit: Option<usize>,
        derivations: impl Fn(&str, bool) -> Result<WordDerivations>,
    ) -> Result<Vec<LocatedQueryTerm>> {
        let mut primitive_query = Vec::new();
        let mut phrase = Vec::new();

        let mut quoted = false;

        let parts_limit = words_limit.unwrap_or(usize::MAX);

        let mut position = -1i8;
        let mut phrase_start = -1i8;
        let mut phrase_end = -1i8;

        let mut peekable = query.peekable();
        while let Some(token) = peekable.next() {
            position += 1;
            // early return if word limit is exceeded
            if primitive_query.len() >= parts_limit {
                return Ok(primitive_query);
            }

            match token.kind {
                TokenKind::Word | TokenKind::StopWord => {
                    // 1. if the word is quoted we push it in a phrase-buffer waiting for the ending quote,
                    // 2. if the word is not the last token of the query and is not a stop_word we push it as a non-prefix word,
                    // 3. if the word is the last token of the query we push it as a prefix word.
                    if quoted {
                        phrase_end = position;
                        if phrase.is_empty() {
                            phrase_start = position;
                        }
                        if let TokenKind::StopWord = token.kind {
                            phrase.push(None);
                        } else {
                            phrase.push(Some(token.lemma().to_string()));
                        }
                    } else if peekable.peek().is_some() {
                        if let TokenKind::StopWord = token.kind {
                        } else {
                            let derivations = derivations(token.lemma(), false)?;
                            let located_term = LocatedQueryTerm {
                                value: QueryTerm::Word {
                                    original: token.lemma().to_owned(),
                                    derivations,
                                },
                                positions: position..=position,
                            };
                            primitive_query.push(located_term);
                        }
                    } else {
                        let derivations = derivations(token.lemma(), true)?;
                        let located_term = LocatedQueryTerm {
                            value: QueryTerm::Word {
                                original: token.lemma().to_owned(),
                                derivations,
                            },
                            positions: position..=position,
                        };
                        primitive_query.push(located_term);
                    }
                }
                TokenKind::Separator(separator_kind) => {
                    let quote_count = token.lemma().chars().filter(|&s| s == '"').count();
                    // swap quoted state if we encounter a double quote
                    if quote_count % 2 != 0 {
                        quoted = !quoted;
                    }
                    // if there is a quote or a hard separator we close the phrase.
                    if !phrase.is_empty()
                        && (quote_count > 0 || separator_kind == SeparatorKind::Hard)
                    {
                        let located_query_term = LocatedQueryTerm {
                            value: QueryTerm::Phrase(mem::take(&mut phrase)),
                            positions: phrase_start..=phrase_end,
                        };
                        primitive_query.push(located_query_term);
                    }
                }
                _ => (),
            }
        }

        // If a quote is never closed, we consider all of the end of the query as a phrase.
        if !phrase.is_empty() {
            let located_query_term = LocatedQueryTerm {
                value: QueryTerm::Phrase(mem::take(&mut phrase)),
                positions: phrase_start..=phrase_end,
            };
            primitive_query.push(located_query_term);
        }

        Ok(primitive_query)
    }
}

impl LocatedQueryTerm {
    pub fn ngram2(
        x: &LocatedQueryTerm,
        y: &LocatedQueryTerm,
    ) -> Option<(String, RangeInclusive<i8>)> {
        if *x.positions.end() != y.positions.start() - 1 {
            return None;
        }
        match (&x.value, &y.value) {
            (QueryTerm::Word { original: w1, .. }, QueryTerm::Word { original: w2, .. }) => {
                let term = (format!("{w1}{w2}"), *x.positions.start()..=*y.positions.end());
                Some(term)
            }
            _ => None,
        }
    }
    pub fn ngram3(
        x: &LocatedQueryTerm,
        y: &LocatedQueryTerm,
        z: &LocatedQueryTerm,
    ) -> Option<(String, RangeInclusive<i8>)> {
        if *x.positions.end() != y.positions.start() - 1
            || *y.positions.end() != z.positions.start() - 1
        {
            return None;
        }
        match (&x.value, &y.value, &z.value) {
            (
                QueryTerm::Word { original: w1, .. },
                QueryTerm::Word { original: w2, .. },
                QueryTerm::Word { original: w3, .. },
            ) => {
                let term = (format!("{w1}{w2}{w3}"), *x.positions.start()..=*z.positions.end());
                Some(term)
            }
            _ => None,
        }
    }
}
