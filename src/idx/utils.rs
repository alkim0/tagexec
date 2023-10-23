use super::basic::{BasicIdx, IdxRow, RowIdx};
use crate::db::DBColInnerSlice;
use crate::file_table::FileTableRef;
use crate::utils;
use rustc_hash::FxHashMap;
use std::fmt;
use std::hash::BuildHasherDefault;
use std::hash::Hash;
use std::rc::Rc;

//pub fn build_rev_map<'a, T: Hash + Eq>(
//    idx: &Idx,
//    vals: &'a DBColInner<T>,
//) -> FxHashMap<&'a T, Vec<IdxRow>> {
//    let mut rev_map =
//        FxHashMap::with_capacity_and_hasher(vals.len(), BuildHasherDefault::default());
//    match vals {
//        DBColInner::Values(values) => {
//            for (row, val) in idx.iter().zip(values) {
//                rev_map.entry(val).or_insert(vec![]).push(row.into_owned());
//            }
//        }
//        // XXX: Alternative is to just iterate once over values[index] (instead of iterating twice
//        // to avoid hashing too many times)
//        DBColInner::Projected { values, index } => {
//            let mut rows = vec![vec![]; values.len()];
//            let mut bmap = Bitmap::new();
//            for (i, (row, &idx)) in idx.iter().zip(index).enumerate() {
//                bmap.insert(idx as u64);
//                rows[i].push(row.into_owned());
//            }
//            for i in bmap {
//                let mut rows = mem::take(&mut rows[i as usize]);
//                rev_map
//                    .entry(&values[i as usize])
//                    .or_insert(vec![])
//                    .append(&mut rows);
//            }
//        }
//    }
//    rev_map
//}

//pub fn hash_join_vals<T: Hash + Eq + fmt::Debug>(
//    rev_map: &FxHashMap<&T, Vec<IdxRow>>,
//    my_tables: Vec<Rc<FileTable>>,
//    other_idx: &Idx,
//    other_vals: &DBColInner<T>,
//) -> RowIdx {
//    let est_len = std::cmp::max(rev_map.len(), other_idx.len());
//    let mut rows = Vec::with_capacity(est_len);
//
//    match other_vals {
//        DBColInner::Values(values) => {
//            rows.extend(
//                other_idx
//                    .iter()
//                    .zip(values)
//                    .filter_map(|(other_row, val)| {
//                        rev_map.get(&val).map(|my_rows| {
//                            my_rows
//                                .iter()
//                                .cloned()
//                                .zip(std::iter::repeat(other_row.into_owned()))
//                                .map(|(my_row, other_row)| utils::append(my_row, other_row))
//                        })
//                    })
//                    .flatten(),
//            );
//        }
//
//        // XXX: Alternative is to just iterate once over values[index] (instead of iterating twice
//        // to avoid hashing too many times)
//        DBColInner::Projected { values, index } => {
//            let mut other_rows = vec![vec![]; values.len()];
//            let mut bmap = Bitmap::new();
//            for (i, (row, &idx)) in other_idx.iter().zip(index).enumerate() {
//                bmap.insert(idx as u64);
//                other_rows[i].push(row.into_owned());
//            }
//
//            rows.extend(
//                bmap.into_iter()
//                    .filter_map(|i| {
//                        rev_map.get(&&values[i as usize]).map(|my_rows| {
//                            let mut other_rows = mem::take(&mut other_rows[i as usize]);
//                            my_rows
//                                .into_iter()
//                                .cartesian_product(other_rows)
//                                .map(|(my_row, other_row)| utils::append(my_row.clone(), other_row))
//                        })
//                    })
//                    .flatten(),
//            );
//        }
//    }
//
//    RowIdx::new(
//        utils::append(my_tables, other_idx.tables().into_owned()),
//        rows,
//    )
//}

//pub fn build_rev_map<'a, T: Hash + Eq>(
//    idx: &Idx,
//    vals: &'a DBColInner<T>,
//) -> FxHashMap<&'a T, Vec<IdxRow>> {
//    let mut rev_map = FxHashMap::with_capacity_and_hasher(idx.len(), BuildHasherDefault::default());
//    for (row, val) in idx.iter().zip(vals.iter()) {
//        rev_map.entry(val).or_insert(vec![]).push(row.into_owned());
//    }
//    rev_map
//}

/// We pass in a &mut Iterator instead of the iterator itself here since after building the rev map,
/// we may want to continue using it. It is guaranteed to call next on `vals` no more than number
/// of rows in `idx`.
// TODO: Consider changing this to accept a DBColInnerSlice
pub fn build_rev_map<'a, T: Hash + Eq>(
    idx: &BasicIdx,
    vals: &mut impl Iterator<Item = &'a T>,
) -> FxHashMap<&'a T, Vec<IdxRow>> {
    let mut rev_map = FxHashMap::with_capacity_and_hasher(idx.len(), BuildHasherDefault::default());
    for row in idx.iter() {
        let val = vals.next().unwrap();
        rev_map.entry(val).or_insert(vec![]).push(row.into_owned());
    }
    rev_map
}

pub fn hash_join_vals<T: Hash + Eq + fmt::Debug>(
    rev_map: &FxHashMap<&T, Vec<IdxRow>>,
    my_table_refs: Vec<Rc<FileTableRef>>,
    other_idx: &BasicIdx,
    //other_vals: &DBColInner<T>,
    //other_vals: &mut impl Iterator<Item = &'a T>,
    other_vals: DBColInnerSlice<'_, T>,
) -> BasicIdx {
    let est_len = std::cmp::max(rev_map.len(), other_idx.len());
    let mut rows = Vec::with_capacity(est_len);
    rows.extend(
        other_idx
            .iter()
            .zip(other_vals)
            .filter_map(|(other_row, val)| {
                rev_map.get(&val).map(|my_rows| {
                    my_rows
                        .iter()
                        .cloned()
                        .zip(std::iter::repeat(other_row.into_owned()))
                        .map(|(my_row, other_row)| utils::append(my_row, other_row))
                })
            })
            .flatten(),
    );

    BasicIdx::Row(RowIdx::new(
        utils::append(my_table_refs, other_idx.table_refs().into_owned()),
        rows,
    ))
}

//pub trait IteratorFilterExt: Iterator {
//    /// Yields only the values whose positions are in the given `index`.
//    fn filter_by_index<'a>(self, index: Cow<'a, Bitmap>) -> FilterByIndex<'a, Self>
//    where
//        Self: Sized;
//}
//
//impl<I: Iterator> IteratorFilterExt for I {
//    fn filter_by_index<'a>(self, index: Cow<'a, Bitmap>) -> FilterByIndex<'a, Self>
//    where
//        Self: Sized,
//    {
//        FilterByIndex {
//            index,
//            iter: self.enumerate(),
//        }
//    }
//}
//
//pub struct FilterByIndex<'a, I> {
//    index: Cow<'a, Bitmap>,
//    iter: std::iter::Enumerate<I>,
//}
//
//impl<'a, I> Iterator for FilterByIndex<'a, I>
//where
//    I: Iterator,
//{
//    type Item = I::Item;
//
//    fn next(&mut self) -> Option<Self::Item> {
//        loop {
//            let out = self.iter.next();
//            if let Some((i, item)) = out {
//                if self.index.as_ref().contains(i as BitmapInt) {
//                    return Some(item);
//                }
//            } else {
//                return None;
//            }
//        }
//    }
//}
