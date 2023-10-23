use super::basic::{BasicIdx, IdxColIter};
use super::utils as idx_utils;
use crate::bitmap::{Bitmap, BitmapInt};
use crate::db::{DBCol, DBColInner};
use crate::expr::EqualityConstraint;
use crate::file_table::{FileCol, FileTableRef};
use crate::tag::{Tag, TagSet};
use itertools::Itertools;
use rustc_hash::FxHashMap;
use std::borrow::Cow;
use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::hash::{BuildHasherDefault, Hash};
use std::rc::Rc;
use std::time::Instant;

pub struct TaggedIdx {
    tagged_idxs: Vec<(TagSet, BasicIdx)>,
    all_tags: TagSet, // This is not super necessary
}

pub struct TaggedIdxRef<'a>(Vec<&'a BasicIdx>);

impl TaggedIdx {
    pub fn new(table_ref: Rc<FileTableRef>, tags: &TagSet) -> Self {
        Self {
            tagged_idxs: vec![(tags.clone(), BasicIdx::new(table_ref))],
            all_tags: tags.clone(),
        }
    }

    pub fn col_iter(&self, table: &FileTableRef) -> MultiIdxColIter<'_> {
        MultiIdxColIter {
            iters: self
                .tagged_idxs
                .iter()
                .map(|(_, idx)| idx.col_iter(table))
                .collect(),
            iter_index: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.tagged_idxs.iter().map(|(_, idx)| idx.len()).sum()
    }

    pub fn num_rel_slices(&self) -> usize {
        self.tagged_idxs.len()
    }

    pub fn filter(&self, bmap: &Bitmap, filter_tags: &TagSet) -> Self {
        debug_assert!(!filter_tags.is_empty());
        let tagged_idxs: Vec<_> = self
            .tagged_idxs
            .iter()
            .scan(0, |cum_len, (tags, idx)| {
                let (start, end) = (*cum_len, *cum_len + idx.len() as BitmapInt);
                *cum_len += idx.len() as BitmapInt;
                Some((tags, idx, start, end))
            })
            .map(|(tags, idx, start, end)| {
                let apply_tags = filter_tags & tags;

                let mut ret = vec![];
                if !apply_tags.is_empty() {
                    if tags.difference(&apply_tags).next().is_some() {
                        let neg_idx = idx.filter(
                            &Bitmap::from_sorted_iter(
                                (Bitmap::from_sorted_iter(start..end).unwrap() - bmap)
                                    .into_iter()
                                    .map(|i| i - start),
                            )
                            .unwrap(),
                        );
                        if !neg_idx.is_empty() {
                            ret.push((tags - &apply_tags, neg_idx));
                        }
                    }

                    let pos_idx = idx.filter(
                        &Bitmap::from_sorted_iter(
                            (bmap & Bitmap::from_sorted_iter(start..end).unwrap())
                                .into_iter()
                                .map(|i| i - start),
                        )
                        .unwrap(),
                    );
                    if !pos_idx.is_empty() {
                        ret.push((tags | &apply_tags, pos_idx));
                    }
                } else {
                    ret.push((tags.clone(), idx.clone()));
                }

                ret
            })
            .flatten()
            .collect();

        let all_tags = tagged_idxs
            .iter()
            .fold(TagSet::new(), |all_tags, (tags, _)| &all_tags | &tags);

        debug_assert!(tagged_idxs.iter().all(|(tags, _)| !tags.is_empty()));

        Self {
            tagged_idxs,
            all_tags,
        }
    }

    pub fn join(&self, other: &Self, constraint: &EqualityConstraint) -> Self {
        let i_am_left = self
            .table_refs()
            .iter()
            .any(|t| t.as_ref() == constraint.left_table_ref.as_ref());

        if i_am_left {
            if self.len() <= other.len() {
                self.hash_join(
                    &constraint.left_col,
                    &constraint.left_table_ref,
                    other,
                    &constraint.right_col,
                    &constraint.right_table_ref,
                )
            } else {
                other.hash_join(
                    &constraint.right_col,
                    &constraint.right_table_ref,
                    self,
                    &constraint.left_col,
                    &constraint.left_table_ref,
                )
            }
        } else {
            if self.len() <= other.len() {
                self.hash_join(
                    &constraint.right_col,
                    &constraint.right_table_ref,
                    other,
                    &constraint.left_col,
                    &constraint.left_table_ref,
                )
            } else {
                other.hash_join(
                    &constraint.left_col,
                    &constraint.left_table_ref,
                    self,
                    &constraint.right_col,
                    &constraint.right_table_ref,
                )
            }
        }
    }

    /// Returns a TaggedIdxRef which contains the Idxs which were tagged with the given `tag`.
    /// Returns None, if tag was not present.
    pub fn ref_tagged(&self, tag: &Tag) -> Option<TaggedIdxRef<'_>> {
        if self.all_tags.contains(tag) {
            Some(TaggedIdxRef(
                self.tagged_idxs
                    .iter()
                    .filter_map(|(tags, idx)| if tags.contains(tag) { Some(idx) } else { None })
                    .collect(),
            ))
        } else {
            None
        }
    }

    fn hash_join(
        &self,
        my_col: &FileCol,
        my_table_ref: &FileTableRef,
        other: &Self,
        other_col: &FileCol,
        other_table_ref: &FileTableRef,
    ) -> Self {
        fn do_hash_join<T: Hash + Eq + fmt::Debug>(
            my_idx: &TaggedIdx,
            other_idx: &TaggedIdx,
            my_vals: DBColInner<T>,
            other_vals: DBColInner<T>,
        ) -> TaggedIdx {
            let mut rev_maps = Vec::with_capacity(my_idx.tagged_idxs.len());

            //let mut tag_map = HashMap::new();

            //match &my_vals {
            //    DBColInner::Values(values) => {
            //        let mut vals_iter = values.iter();
            //        for (i, (tags, idx)) in my_idx.tagged_idxs.iter().enumerate() {
            //            let mut idx_iter = idx.iter();
            //            for _ in 0..idx.len() {
            //                let val = vals_iter.next().unwrap();
            //                let row = idx_iter.next().unwrap();
            //                rev_maps[i]
            //                    .entry(val)
            //                    .or_insert(vec![])
            //                    .push(row.into_owned());
            //            }
            //        }
            //    }

            //    // XXX: Alternative is to just iterate once over values[index] (instead of
            //    // iterating twice to avoid hashing too many times)
            //    // TODO FIXME: We can get additional speedups here by using the RawTable API to
            //    // memoize hashes of values. Right now we have hash each value n times, n is the
            //    // number of mini-idxs which contain that value
            //    DBColInner::Projected { values, index } => {
            //        let mut all_rows = Vec::with_capacity(my_idx.tagged_idxs.len());
            //        let mut all_bmaps = Vec::with_capacity(my_idx.tagged_idxs.len());
            //        let mut proj_idx_iter = index.iter();
            //        for (i, (tags, idx)) in my_idx.tagged_idxs.iter().enumerate() {
            //            let mut rows = vec![vec![]; values.len()];
            //            for my_row in idx.iter() {

            //            }
            //            rows[i]
            //            bmap.insert(idx as u64);
            //            rows[i].push(row.into_owned());
            //        }

            //        for (i, (tags, idx)) in my_idx.tagged_idxs.iter().enumerate() {}
            //        for i in bmap {
            //            let mut rows = mem::take(&mut rows[i as usize]);
            //            rev_map
            //                .entry(&values[i as usize])
            //                .or_insert(vec![])
            //                .append(&mut rows);
            //        }
            //    }
            //}

            //let now = Instant::now();
            let mut my_vals_iter = my_vals.iter();
            for (tags, idx) in &my_idx.tagged_idxs {
                rev_maps.push((
                    tags.clone(),
                    idx_utils::build_rev_map(idx, &mut my_vals_iter),
                ));
            }
            //println!(
            //    "join build_rev_map_ms: {}, num_rev_maps: {}",
            //    now.elapsed().as_millis(),
            //    rev_maps.len(),
            //);

            //println!("other_vals {:?}", other_vals.len());
            //println!(
            //    "tagged_idxs {:?}",
            //    other_idx
            //        .tagged_idxs
            //        .iter()
            //        .map(|x| x.1.len())
            //        .collect::<Vec<_>>()
            //);
            //let now = Instant::now();
            let mut tagged_idxs = vec![];
            let mut num_read = 0;
            for (other_tags, idx) in &other_idx.tagged_idxs {
                for (my_tags, rev_map) in &rev_maps {
                    let tags = my_tags & other_tags;
                    if !tags.is_empty() {
                        tagged_idxs.push((
                            tags,
                            idx_utils::hash_join_vals(
                                rev_map,
                                my_idx.table_refs().into_owned(),
                                idx,
                                other_vals.slice(num_read..(num_read + idx.len())),
                            ),
                        ));
                    }
                }
                num_read += idx.len();
            }
            //println!("join doing_hash_join_ms: {}", now.elapsed().as_millis(),);

            //let tagged_idxs: Vec<_> =
            //    rev_maps
            //        .iter()
            //        .cartesian_product(other_idx.tagged_idxs.iter().scan(
            //            0,
            //            |cum_len, (tags, idx)| {
            //                let other_vals = &other_vals[*cum_len..(*cum_len + idx.len())];
            //                *cum_len += idx.len();
            //                Some((tags, idx, other_vals))
            //            },
            //        ))
            //        .filter_map(|((my_tags, rev_map), (other_tags, idx, other_vals))| {
            //            let tags = my_tags & other_tags;
            //            if !tags.is_empty() {
            //                Some((
            //                    tags,
            //                    Idx::Row(idx_utils::hash_join_vals(
            //                        rev_map,
            //                        my_idx.tables().into_owned(),
            //                        idx,
            //                        other_vals,
            //                    )),
            //                ))
            //            } else {
            //                None
            //            }
            //        })
            //        .collect();

            assert!(tagged_idxs.len() <= my_idx.tagged_idxs.len() * other_idx.tagged_idxs.len());
            TaggedIdx {
                tagged_idxs,
                all_tags: &my_idx.all_tags | &other_idx.all_tags,
            }
        }

        //let now = Instant::now();
        let my_vals = my_col.read(self.col_iter(my_table_ref));
        let other_vals = other_col.read(other.col_iter(other_table_ref));
        //println!(
        //    "join {} = {} read_time_ms: {}",
        //    my_col.full_name(),
        //    other_col.full_name(),
        //    now.elapsed().as_millis()
        //);
        match (my_vals, other_vals) {
            (DBCol::Int(my_vals), DBCol::Int(other_vals)) => {
                do_hash_join(self, other, my_vals, other_vals)
            }
            (DBCol::Long(my_vals), DBCol::Long(other_vals)) => {
                do_hash_join(self, other, my_vals, other_vals)
            }
            (DBCol::Str(my_vals), DBCol::Str(other_vals)) => {
                do_hash_join(self, other, my_vals, other_vals)
            }
            _ => {
                panic!(
                    "Unsupported join column type ({:?}, {:?})",
                    my_col.data_type(),
                    other_col.data_type(),
                );
            }
        }
    }

    fn table_refs(&self) -> Cow<'_, Vec<Rc<FileTableRef>>> {
        self.tagged_idxs
            .first()
            .map(|(_, idx)| idx.table_refs())
            .unwrap_or(Cow::Owned(vec![]))
    }
}

impl TaggedIdxRef<'_> {
    pub fn col_iter(&self, table: &FileTableRef) -> MultiIdxColIter<'_> {
        MultiIdxColIter {
            iters: self.0.iter().map(|idx| idx.col_iter(table)).collect(),
            iter_index: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.0.iter().map(|idx| idx.len()).sum()
    }
}

pub struct MultiIdxColIter<'a> {
    iters: Vec<IdxColIter<'a>>,
    iter_index: usize,
}

impl<'a> Iterator for MultiIdxColIter<'a> {
    type Item = BitmapInt;

    fn next(&mut self) -> Option<Self::Item> {
        while self.iter_index < self.iters.len() {
            let result = self.iters[self.iter_index].next();
            if result.is_some() {
                return result;
            } else {
                self.iter_index += 1;
            }
        }

        None
    }
}

//impl fmt::Debug for TaggedIdx {
//    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//        write!(
//            f,
//            "TaggedIdx([{}])",
//            self.tagged_idxs
//                .iter()
//                .map(|(tags, idx)| format!("[{}] {:?}", tags.iter().join(", "), idx))
//                .join(", ")
//        )
//    }
//}
