use super::basic::{self, RowIdx};
use super::{BasicIdx, Idx, IdxRow};
use crate::bitmap::{Bitmap, BitmapInt};
use crate::db::{DBCol, DBColInner};
use crate::engine::EXEC_INFO;
use crate::expr::{EqualityConstraint, EvalContext};
use crate::file_table::FileTableRef;
use crate::pred::Pred;
use crate::tag::Tag;
use crate::utils::{self, IteratorFilterExt};
use either::Either;
use itertools::Itertools;
use log::info;
use rustc_hash::FxHashMap;
use std::borrow::Cow;
use std::collections::HashSet;
use std::fmt;
use std::hash::{BuildHasherDefault, Hash};
use std::rc::Rc;
use std::time::Instant;

// NOTE: An alternative design is to remove actual rows from the `idx`, but that's probably costly
// because not only must we modify `idx`, but we must update all of `tagged`'s bitmaps to be in
// line with `idx`'s modifications
pub struct TaggedIdx {
    idx: Rc<BasicIdx>,
    tagged: FxHashMap<Rc<Tag>, Rc<Bitmap>>,
}

impl TaggedIdx {
    pub fn new(table_ref: Rc<FileTableRef>, tags: impl IntoIterator<Item = Rc<Tag>>) -> Self {
        let idx = Rc::new(BasicIdx::new(table_ref));
        let full_bmap = Rc::new(Bitmap::from_sorted_iter(0..(idx.len() as BitmapInt)).unwrap());
        let tagged = tags
            .into_iter()
            .map(|tag| (tag, full_bmap.clone()))
            .collect();

        Self { idx, tagged }
    }

    /// Applies the given predicate and creates a new TaggedIdx with the tags according to the
    /// given `tag_map`. Here, `idx` is the `Idx` struct which wraps around self passed here so we
    /// can call `eval` on `pred`.
    // NOTE: Consider splitting out the evaluation from the tag bitmap business using
    // derive_tag_bmap and apply_result_bmap or something.
    pub fn apply_pred(
        &self,
        pred: &Pred,
        tag_map: &FxHashMap<Rc<Tag>, Vec<(bool, Rc<Tag>)>>,
        idx: &Idx,
    ) -> Self {
        let mut new_tagged = FxHashMap::with_hasher(BuildHasherDefault::default());
        let mut apply_bmaps = vec![];

        for (tag, results) in tag_map {
            if let Some(bmap) = self.tagged.get(tag) {
                if results.is_empty() {
                    new_tagged.insert(tag.clone(), bmap.clone());
                } else {
                    for (val, new_tag) in results {
                        apply_bmaps.push((bmap.clone(), *val, new_tag));
                    }
                }
            }
        }

        let unioned_bmap = Rc::new(
            apply_bmaps
                .iter()
                .map(|(bmap, _, _)| bmap)
                .fold(Bitmap::new(), |acc, bmap| acc | bmap.as_ref()),
        );
        let now = Instant::now();
        let unioned_bmap_len = unioned_bmap.len();
        let result_bmap = pred.eval(&EvalContext {
            idx,
            bmap: if unioned_bmap.len() as usize == idx.len() {
                None
            } else {
                Some(unioned_bmap)
            },
        });
        let pred_eval_time_ms = now.elapsed().as_millis();

        EXEC_INFO.with(|exec_info| {
            let mut exec_info = exec_info.borrow_mut();
            exec_info.stats.pred_eval_time_ms += pred_eval_time_ms;
            exec_info.stats.num_filter_tuples += unioned_bmap_len as u128;
        });

        if EXEC_INFO.with(|exec_info| exec_info.borrow().debug_times) {
            info!(
                "Filter {} eval time {} ms, num vals evaled {}",
                pred, pred_eval_time_ms, unioned_bmap_len,
            );
        }

        for (bmap, val, new_tag) in apply_bmaps {
            let bmap = if val {
                &result_bmap & bmap.as_ref()
            } else {
                bmap.as_ref() - &result_bmap
            };

            if !bmap.is_empty() {
                new_tagged
                    .entry(new_tag.clone())
                    .and_modify(|tag_bmap| *tag_bmap = Rc::new(tag_bmap.as_ref() | &bmap))
                    .or_insert_with(|| Rc::new(bmap));
            }
        }

        Self {
            idx: self.idx.clone(),
            tagged: new_tagged,
        }
    }

    pub fn join(
        &self,
        other: &Self,
        constraint: &EqualityConstraint,
        tag_map: &FxHashMap<(Rc<Tag>, Rc<Tag>), Option<Rc<Tag>>>,
        left_based: bool,
    ) -> Self {
        // NOTE: When building the join map, we can build it in a few ways. We can have a one hash
        // map for each tag. However, this requires us to hash the iterating side multiple times
        // (once for each tag on the hashed side which can yield a valid tag). This might be too
        // expensive. Instead, it's probably better to build a single giant hash map so each value
        // only needs to be hashed once, and in the values of the hash map, separate out by tag, so
        // we know which of these values to keep and which ones to not. This requires that each
        // value be hashed at least once, but that's probably okay since we assume each tag on each
        // side hash at least one tag on the other side they from a valid tag with.
        fn build_join_map<'a, T: Hash + Eq + Clone>(
            bmaps: Vec<&Bitmap>,
            base_idx: &'a BasicIdx,
            vals: DBColInner<T>,
            union_bmap: &Rc<Bitmap>,
        ) -> FxHashMap<T, Vec<Vec<Cow<'a, IdxRow>>>> {
            // TODO FIXME: Can make this even faster by using the structure of DBColInner
            // NOTE: We can do these in a couple ways. We can iterate over each bitmap and create
            // the hashmap from that (this will likely lead to some/lots of random io). Or, we can
            // try to iterate over all the bitmaps simultaneously taking, the next valid idx. This
            // requires the use of a priority queue and takes an extra log t time. The following
            // does the second. Maybe try the first way in the future.
            let mut join_map = FxHashMap::with_hasher(BuildHasherDefault::default());

            let idx_iter = base_idx.iter().filter_by_index2(union_bmap);
            let vals_iter = vals.into_iter();
            let merge_bmap_iter = utils::merge_iter_bmaps2(&bmaps);

            for (row, val, (i, _)) in itertools::izip!(idx_iter, vals_iter, merge_bmap_iter) {
                join_map.entry(val).or_insert(vec![vec![]; bmaps.len()])[i].push(row);
            }

            join_map
        }

        // XXX: The vals passed here have are actually the result of already filtered values, so
        // when iterating over them, no need to filter_by_index again.
        fn hash_join<T: Hash + Eq + Clone + fmt::Debug>(
            left_idx: &TaggedIdx,
            right_idx: &TaggedIdx,
            left_vals: DBColInner<T>,
            right_vals: DBColInner<T>,
            left_bmap: Rc<Bitmap>,
            right_bmap: Rc<Bitmap>,
            tag_map: &FxHashMap<(Rc<Tag>, Rc<Tag>), Option<Rc<Tag>>>,
            left_based: bool,
        ) -> TaggedIdx {
            let now = Instant::now();
            let (left_tag_map, left_bmaps): (FxHashMap<_, _>, Vec<_>) = left_idx
                .tagged
                .iter()
                .enumerate()
                .map(|(i, (tag, bmap))| ((tag.clone(), i), bmap.as_ref()))
                .unzip();
            let (right_tag_map, right_bmaps): (FxHashMap<_, _>, Vec<_>) = right_idx
                .tagged
                .iter()
                .enumerate()
                .map(|(i, (tag, bmap))| ((tag.clone(), i), bmap.as_ref()))
                .unzip();

            // tag_map includes all tags from the planning state. During execution, fewer tags may
            // be passed along the left and right idxs due to some tags having zero-length
            let out_tags = Vec::from_iter(
                tag_map
                    .iter()
                    .filter_map(|((left_tag, right_tag), out_tag)| {
                        if left_tag_map.contains_key(left_tag)
                            && right_tag_map.contains_key(right_tag)
                        {
                            out_tag.clone()
                        } else {
                            None
                        }
                    })
                    .collect::<HashSet<_>>(),
            );
            let mut out_bmaps = vec![Bitmap::new(); out_tags.len()];
            let out_tag_map: FxHashMap<_, _> = out_tags
                .iter()
                .cloned()
                .enumerate()
                .map(|(i, tag)| (tag, i))
                .collect();

            let mut tag_lookup_map = FxHashMap::with_hasher(BuildHasherDefault::default());
            for ((left_tag, right_tag), out_tag) in tag_map {
                if !(left_tag_map.contains_key(left_tag) && right_tag_map.contains_key(right_tag)) {
                    continue;
                }

                if let Some(out_tag) = out_tag {
                    let left_tag_i = *left_tag_map.get(left_tag).unwrap();
                    let right_tag_i = *right_tag_map.get(right_tag).unwrap();
                    let out_tag_i = *out_tag_map.get(out_tag).unwrap();

                    if left_based {
                        tag_lookup_map
                            .entry(right_tag_i)
                            .or_insert(vec![])
                            .push((left_tag_i, out_tag_i));
                    } else {
                        tag_lookup_map
                            .entry(left_tag_i)
                            .or_insert(vec![])
                            .push((right_tag_i, out_tag_i));
                    }
                }
            }
            let tag_manipulation_time_ms = now.elapsed().as_millis();

            let now = Instant::now();
            let mut out_rows = vec![];
            let (join_map, map_idx, other_vals, other_idx, other_bmaps, other_union_bmap) =
                if left_based {
                    let join_map = build_join_map(left_bmaps, &left_idx.idx, left_vals, &left_bmap);
                    (
                        join_map,
                        left_idx,
                        right_vals,
                        right_idx,
                        right_bmaps,
                        right_bmap,
                    )
                } else {
                    let join_map =
                        build_join_map(right_bmaps, &right_idx.idx, right_vals, &right_bmap);
                    (
                        join_map, right_idx, left_vals, left_idx, left_bmaps, left_bmap,
                    )
                };
            let join_map_time_ms = now.elapsed().as_millis();

            //println!("left_based {}", left_based);
            //println!("join_map {:?}", join_map);
            //println!("tag_lookup_map {:?}", tag_lookup_map);
            //println!("out_tags {:?}", out_tags);

            //println!(
            //    "Hash joining [{}] with [{}]",
            //    map_idx.idx.table_refs().iter().join(", "),
            //    other_idx.idx.table_refs().iter().join(", ")
            //);

            //println!(
            //    "tag_map\n{}",
            //    tag_map
            //        .iter()
            //        .map(|((left, right), result)| format!(
            //            "\t{}\n\tJOIN {}\n\t-> {:?}",
            //            left, right, result
            //        ))
            //        .join("\n\n")
            //);

            let now = Instant::now();
            let other_vals_iter = other_vals.into_iter();
            let other_idx_iter = other_idx.idx.iter().filter_by_index2(&other_union_bmap);
            let iter_create_time_ms = now.elapsed().as_millis();

            let now = Instant::now();
            let mut num_out_rows = 0;
            for (val, other_row, (i, idx)) in itertools::izip!(
                other_vals_iter,
                other_idx_iter,
                utils::merge_iter_bmaps2(&other_bmaps)
            ) {
                // NOTE: For now, make this an inner join
                if let Some(tag_rows) = join_map.get(&val) {
                    if let Some(join_tags) = tag_lookup_map.get(&i) {
                        for (&(map_i, out_i), other_row) in join_tags
                            .iter()
                            .zip(std::iter::repeat(other_row.into_owned()))
                        {
                            //let rows = &tag_rows[map_i];
                            let rows = unsafe { tag_rows.get_unchecked(map_i) };
                            //out_bmaps[out_i].insert_range(
                            let out_bmap = unsafe { out_bmaps.get_unchecked_mut(out_i) };
                            out_bmap.insert_range(
                                (num_out_rows as BitmapInt)
                                    ..((num_out_rows + rows.len()) as BitmapInt),
                            );

                            out_rows.extend(rows.iter().zip(std::iter::repeat(other_row)).map(
                                |(row, other_row)| {
                                    utils::append(row.clone().into_owned(), other_row)
                                },
                            ));
                            num_out_rows += rows.len();
                        }
                    }
                }
            }
            let join_iter_time_ms = now.elapsed().as_millis();

            //println!("left_idx {:?}", left_idx);
            //println!("right_idx {:?}", right_idx);
            //println!("out_rows {:?}", out_rows);
            //println!(
            //    "out tagged {:?}",
            //    out_tags.iter().zip(out_bmaps.iter()).collect::<Vec<_>>()
            //);

            let now = Instant::now();
            let table_refs = utils::append(
                map_idx.idx.table_refs().into_owned(),
                other_idx.idx.table_refs().into_owned(),
            );
            let tagged = out_tags
                .into_iter()
                .zip(out_bmaps.into_iter().map(|bmap| Rc::new(bmap)))
                .filter(|(_, bmap)| !bmap.is_empty())
                .collect();
            let idx = Rc::new(BasicIdx::Row(RowIdx::new(table_refs, out_rows)));
            let output_create_time_ms = now.elapsed().as_millis();

            if EXEC_INFO.with(|exec_info| exec_info.borrow().debug_times) {
                let left_table_refs = left_idx.idx.table_refs().iter().join(", ");
                let right_table_refs = right_idx.idx.table_refs().iter().join(", ");
                for (name, time_ms) in [
                    ("tag manipulation", tag_manipulation_time_ms),
                    ("join map creation", join_map_time_ms),
                    ("iter create time", iter_create_time_ms),
                    ("join iter time", join_iter_time_ms),
                    ("output creation", output_create_time_ms),
                ] {
                    info!(
                        "Join([{}] [{}]), {} {} ms",
                        left_table_refs, right_table_refs, name, time_ms
                    );
                }
                info!(
                    "Join([{}] [{}]), total time {} ms",
                    left_table_refs,
                    right_table_refs,
                    tag_manipulation_time_ms
                        + join_map_time_ms
                        + iter_create_time_ms
                        + join_iter_time_ms
                        + output_create_time_ms
                );
            }

            TaggedIdx { idx, tagged }
        }

        // Alternative version of hash join in which we don't care so much about iterating
        // DBColInner in consecutive order
        fn hash_join2<T: Hash + Eq + Clone + fmt::Debug>(
            left_idx: &TaggedIdx,
            right_idx: &TaggedIdx,
            left_vals: DBColInner<T>,
            right_vals: DBColInner<T>,
            left_union_bmap: Rc<Bitmap>,
            right_union_bmap: Rc<Bitmap>,
            tag_map: &FxHashMap<(Rc<Tag>, Rc<Tag>), Option<Rc<Tag>>>,
            left_based: bool,
        ) -> TaggedIdx {
            let (left_tag_map, left_bmaps): (FxHashMap<_, _>, Vec<_>) = left_idx
                .tagged
                .iter()
                .enumerate()
                .map(|(i, (tag, bmap))| ((tag.clone(), i), bmap.as_ref()))
                .unzip();
            let (right_tag_map, right_bmaps): (FxHashMap<_, _>, Vec<_>) = right_idx
                .tagged
                .iter()
                .enumerate()
                .map(|(i, (tag, bmap))| ((tag.clone(), i), bmap.as_ref()))
                .unzip();

            let out_tags = Vec::from_iter(
                tag_map
                    .values()
                    .filter_map(|tag| tag.clone())
                    .collect::<HashSet<_>>(),
            );
            let mut out_bmaps = vec![Bitmap::new(); out_tags.len()];
            let out_tag_map: FxHashMap<_, _> = out_tags
                .iter()
                .cloned()
                .enumerate()
                .map(|(i, tag)| (tag, i))
                .collect();

            let mut tag_lookup_map = FxHashMap::with_hasher(BuildHasherDefault::default());
            for ((left_tag, right_tag), out_tag) in tag_map {
                if let Some(out_tag) = out_tag {
                    let left_tag_i = *left_tag_map.get(left_tag).unwrap();
                    let right_tag_i = *right_tag_map.get(right_tag).unwrap();
                    let out_tag_i = *out_tag_map.get(out_tag).unwrap();

                    if left_based {
                        tag_lookup_map
                            .entry(right_tag_i)
                            .or_insert(vec![])
                            .push((left_tag_i, out_tag_i));
                    } else {
                        tag_lookup_map
                            .entry(left_tag_i)
                            .or_insert(vec![])
                            .push((right_tag_i, out_tag_i));
                    }
                }
            }

            //println!("left_tag_map {:?}", left_tag_map);
            //println!("right_tag_map {:?}", right_tag_map);
            //println!("left_based {}", left_based);
            //println!("tag_lookup_map {:?}", tag_lookup_map);

            let (join_map, map_idx, other_vals, other_idx, other_bmaps, other_union_bmap) =
                if left_based {
                    let join_map =
                        build_join_map(left_bmaps, &left_idx.idx, left_vals, &left_union_bmap);
                    (
                        join_map,
                        left_idx,
                        right_vals,
                        right_idx,
                        right_bmaps,
                        right_union_bmap,
                    )
                } else {
                    let join_map =
                        build_join_map(right_bmaps, &right_idx.idx, right_vals, &right_union_bmap);
                    (
                        join_map,
                        right_idx,
                        left_vals,
                        left_idx,
                        left_bmaps,
                        left_union_bmap,
                    )
                };

            //println!("join_map: {:?}", join_map);
            //println!("other_vals {:?}", other_vals);

            let mut out_rows = vec![];
            for (i, bmap) in other_bmaps.into_iter().enumerate() {
                let val_bmap =
                    Bitmap::from_sorted_iter(bmap.iter().map(|idx| other_union_bmap.rank(idx) - 1))
                        .unwrap();
                for &(map_i, out_i) in tag_lookup_map.get(&i).unwrap() {
                    let prev_num_out_rows = out_rows.len();
                    // XXX: Rembmer that other_vals is the already-filtered from when reading in
                    // the values, so we don't need to filter again here (i.e., other_vals[idx]).

                    for (idx, other_val) in bmap
                        .iter()
                        .zip(other_vals.iter().filter_by_index2(&val_bmap))
                    {
                        let other_row = other_idx.idx.index(idx as usize);
                        if let Some(map_rows) = join_map.get(other_val) {
                            out_rows.extend(map_rows[map_i].iter().map(|map_row| {
                                map_row
                                    .iter()
                                    .cloned()
                                    .chain(other_row.iter().cloned())
                                    .collect::<Vec<_>>()
                            }));
                        }
                    }
                    out_bmaps[out_i].insert_range(
                        (prev_num_out_rows as BitmapInt)..(out_rows.len() as BitmapInt),
                    );
                }
            }

            //println!(
            //    "Join([{}] [{}])",
            //    left_idx.idx.table_refs().iter().join(", "),
            //    right_idx.idx.table_refs().iter().join(", ")
            //);
            //println!("left_idx {:?}", left_idx);
            //println!("right_idx {:?}", right_idx);
            //println!("out_rows {:?}", out_rows);
            //println!(
            //    "out tagged {:?}",
            //    out_tags.iter().zip(out_bmaps.iter()).collect::<Vec<_>>()
            //);

            let table_refs = utils::append(
                map_idx.idx.table_refs().into_owned(),
                other_idx.idx.table_refs().into_owned(),
            );
            let tagged = out_tags
                .into_iter()
                .zip(out_bmaps.into_iter().map(|bmap| Rc::new(bmap)))
                .collect();
            let idx = Rc::new(BasicIdx::Row(RowIdx::new(table_refs, out_rows)));

            TaggedIdx { idx, tagged }
        }

        // We are left and we should be left on the constraint as well
        debug_assert!(self
            .idx
            .table_refs()
            .iter()
            .any(|t| t.as_ref() == constraint.left_table_ref.as_ref()));

        // TODO: There is a chance for a speedup here if one of the tags becomes completely
        // unneeded due to the fact that every tag that it would combine with is missing (because
        // they have length 0). Then we could avoid including that tag as one of the required
        // bitmaps here. This would also reduce the number of values we would have to read from
        // disk.
        let left_bmap = self.union_bmap();
        let right_bmap = other.union_bmap();

        if left_bmap.is_empty() || right_bmap.is_empty() {
            let table_refs = utils::append(
                self.idx.table_refs().into_owned(),
                other.idx.table_refs().into_owned(),
            );
            let idx = Rc::new(BasicIdx::Row(RowIdx::new(table_refs, vec![])));
            let tagged = FxHashMap::with_hasher(BuildHasherDefault::default());
            return TaggedIdx { idx, tagged };
        }

        let now = Instant::now();
        let left_vals = constraint
            .left_col
            .read(if left_bmap.len() as usize == self.idx.len() {
                Either::Left(self.idx.col_iter(&constraint.left_table_ref))
            } else {
                Either::Right(
                    self.idx
                        .col_iter(&constraint.left_table_ref)
                        .filter_by_index2(&left_bmap),
                )
            });
        let right_vals =
            constraint
                .right_col
                .read(if right_bmap.len() as usize == other.idx.len() {
                    Either::Left(other.idx.col_iter(&constraint.right_table_ref))
                } else {
                    Either::Right(
                        other
                            .idx
                            .col_iter(&constraint.right_table_ref)
                            .filter_by_index2(&right_bmap),
                    )
                });

        EXEC_INFO.with(|exec_info| {
            let mut exec_info = exec_info.borrow_mut();
            exec_info.stats.num_join_tuples += (left_vals.len() + right_vals.len()) as u128;
        });

        if EXEC_INFO.with(|exec_info| exec_info.borrow().debug_times) {
            info!(
                "Join {} val read time {} ms, num left vals {} num right vals {} left selec {} right selec {}",
                constraint,
                now.elapsed().as_millis(),
                left_vals.len(),
                right_vals.len(),
                left_bmap.len() as f64 / self.idx.len() as f64,
                right_bmap.len() as f64 / other.idx.len() as f64,
            );
        }
        let now = Instant::now();
        let out = match (left_vals, right_vals) {
            (DBCol::Int(left_vals), DBCol::Int(right_vals)) => hash_join(
                self, other, left_vals, right_vals, left_bmap, right_bmap, tag_map, left_based,
            ),
            (DBCol::Long(left_vals), DBCol::Long(right_vals)) => hash_join(
                self, other, left_vals, right_vals, left_bmap, right_bmap, tag_map, left_based,
            ),
            (DBCol::Str(left_vals), DBCol::Str(right_vals)) => hash_join(
                self, other, left_vals, right_vals, left_bmap, right_bmap, tag_map, left_based,
            ),
            _ => panic!(
                "Unsupported join column type ({:?}, {:?})",
                constraint.left_col.data_type(),
                constraint.right_col.data_type(),
            ),
        };

        if EXEC_INFO.with(|exec_info| exec_info.borrow().debug_times) {
            info!(
                "Total hash join {} for time {} ms",
                constraint,
                now.elapsed().as_millis()
            );
        }
        out
    }

    /// Returns the union of all the tags' bitmaps.
    pub fn union_bmap(&self) -> Rc<Bitmap> {
        if self.tagged.len() == 1 {
            self.tagged.values().next().unwrap().clone()
        } else {
            Rc::new(
                self.tagged
                    .values()
                    .fold(Bitmap::new(), |acc, bmap| acc | bmap.as_ref()),
            )
        }
    }

    /// Returns a vector of pairs with all the lengths of each tag
    pub fn tag_lens(&self) -> Vec<(Rc<Tag>, usize)> {
        self.tagged
            .iter()
            .map(|(tag, bmap)| (tag.clone(), bmap.len() as usize))
            .collect()
    }

    /// Note this just iterates over the underlying BasicIdx
    pub fn col_iter(&self, table_ref: &FileTableRef) -> basic::IdxColIter<'_> {
        self.idx.col_iter(table_ref)
    }

    /// Note this just gives the size of the underlying BasicIdx
    pub fn len(&self) -> usize {
        self.idx.len()
    }

    //pub fn col_iter(&self, table: &FileTableRef, tag_expr: Option<&TagExpr>) -> IdxColIter<'_> {
    //    if let Some(tag_expr) = tag_expr {
    //        let bmap = self.tag_expr_to_bmap(tag_expr);
    //        // TODO FIXME: If highly selective, may be worth creating another col_iter_indexed that
    //        Either::Right(self.idx.col_iter(table).filter_by_index(bmap))
    //    } else {
    //        Either::Left(self.idx.col_iter(table))
    //    }
    //}

    ///// Note this filter must operate on the same tag_expr as was used to read with col_iter
    //pub fn filter(&self, bmap: Bitmap, tag_expr: Option<&TagExpr>, new_tag: Tag) -> Self {
    //    let new_bmap = if let Some(tag_expr) = tag_expr {
    //        let base_bmap = self.tag_expr_to_bmap(tag_expr);
    //        Rc::new(base_bmap.iter().filter_by_index(AsRefable(&bmap)).collect())
    //    } else {
    //        Rc::new(bmap)
    //    };

    //    let tagged = self.tagged.clone();
    //    tagged.insert(new_tag, new_bmap);

    //    Self {
    //        idx: self.idx.clone(),
    //        tagged,
    //    }
    //}

    //// From a tagset, derive the bitmap which contains all relevant records.
    //fn derive_tagged_bitmap(&self, tags: &TagSet) -> Bitmap {
    //    tags.iter()
    //        .map(|tag| self.tagged.get(tag).unwrap())
    //        .fold(Bitmap::new(), |acc, x| &acc | x.as_ref())
    //}

    //pub fn iter(&self) -> impl Iterator<Item = Cow<'_, IdxRow>> + '_ {
    //    self.idx.iter().filter_by_index(Cow::Borrowed(&self.all))
    //}

    //pub fn iter_tagged(&self, tags: &TagSet) -> impl Iterator<Item = Cow<'_, IdxRow>> {
    //    let bmap = self.derive_tagged_bitmap(tags);
    //    self.idx.iter().filter_by_index(Cow::Owned(bmap))
    //}

    //pub fn tables(&self) -> Cow<'_, Vec<Rc<FileTable>>> {
    //    self.idx.tables()
    //}

    //pub fn join(&self, other: &Self, constraint: &EqualityConstraint) -> Self {
    //    let left_table = constraint.left.table();
    //    let i_am_left = self
    //        .tables()
    //        .iter()
    //        .any(|t| t.as_ref() == left_table.as_ref());

    //    if i_am_left {
    //        if self.len() <= other.len() {
    //            self.hash_join(other, &constraint.left, &constraint.right)
    //        } else {
    //            other.hash_join(self, &constraint.right, &constraint.left)
    //        }
    //    } else {
    //        if self.len() <= other.len() {
    //            self.hash_join(other, &constraint.right, &constraint.left)
    //        } else {
    //            other.hash_join(self, &constraint.left, &constraint.right)
    //        }
    //    }
    //}

    //fn hash_join(&self, other: &Self, my_col: &FileCol, other_col: &FileCol) -> Self {
    //    fn do_hash_join<'a, T: Hash + Eq + fmt::Debug>(
    //        my_idx: &TaggedIdxV2,
    //        other_idx: &TaggedIdxV2,
    //        my_vals: DBColInner<T>,
    //        other_vals: DBColInner<T>,
    //    ) -> TaggedIdxV2 {
    //        let my_tags: TagSet = my_idx.tagged.keys().cloned().collect();
    //        let other_tags: TagSet = other_idx.tagged.keys().cloned().collect();
    //        let common_tags = &my_tags & &other_tags;

    //        for tagset in unique_tagsets {}

    //        let bmap = my_idx.derive_tagged_bitmap(&common_tags);
    //        let mut rev_map =
    //            FxHashMap::with_capacity_and_hasher(my_idx.len(), BuildHasherDefault::default());
    //        for (row, val) in my_idx.iter_tagged(&common_tags).zip(my_vals.iter()) {
    //            rev_map.entry(val).or_insert(vec![]).push(row.into_owned());
    //        }

    //        let est_len = std::cmp::max(rev_map.len(), other_idx.len());
    //        let mut rows = Vec::with_capacity(est_len);
    //        rows.extend(
    //            other_idx
    //                .iter_tagged(&common_tags)
    //                .zip(other_vals.iter())
    //                .filter_map(|(other_row, val)| {
    //                    rev_map.get(&val).map(|my_rows| {
    //                        my_rows
    //                            .iter()
    //                            .cloned()
    //                            .zip(std::iter::repeat(other_row.into_owned()))
    //                            .map(|(my_row, other_row)| utils::append(my_row, other_row))
    //                    })
    //                })
    //                .flatten(),
    //        );

    //        Idx::Row(RowIdx::new(
    //            utils::append(
    //                my_idx.tables().into_owned(),
    //                other_idx.tables().into_owned(),
    //            ),
    //            rows,
    //        ))
    //    }

    //    let my_tags: TagSet = self.tagged.keys().cloned().collect();
    //    let other_tags: TagSet = other.tagged.keys().cloned().collect();
    //    let common_tags = &my_tags & &other_tags;

    //    let my_vals = my_col.read(self.col_iter(&my_col.table(), Some(&common_tags)));
    //    let other_vals = other_col.read(other.col_iter(&other_col.table(), Some(&common_tags)));
    //    match (my_vals, other_vals) {
    //        (DBCol::Int(my_vals), DBCol::Int(other_vals)) => {
    //            do_hash_join(self, other, my_vals, other_vals)
    //        }
    //        (DBCol::Long(my_vals), DBCol::Long(other_vals)) => {
    //            do_hash_join(self, other, my_vals, other_vals)
    //        }
    //        (DBCol::Str(my_vals), DBCol::Str(other_vals)) => {
    //            do_hash_join(self, other, my_vals, other_vals)
    //        }
    //        _ => {
    //            panic!(
    //                "Unsupported join column type ({:?}, {:?})",
    //                my_col.data_type(),
    //                other_col.data_type(),
    //            );
    //        }
    //    }
    //}
}

impl<'a> TryFrom<&'a Idx> for &'a TaggedIdx {
    type Error = &'static str;

    fn try_from(value: &'a Idx) -> Result<Self, Self::Error> {
        match value {
            Idx::Tagged(tagged) => Ok(tagged),
            _ => Err("Not tagged idx type"),
        }
    }
}

impl fmt::Debug for TaggedIdx {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TaggedIdx(idx={:?}, tagged={{{}}})",
            self.idx,
            self.tagged
                .iter()
                .map(|(tag, bmap)| format!("{}: {:?}", tag, bmap))
                .join(", ")
        )
    }
}
