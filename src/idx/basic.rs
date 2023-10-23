use super::utils as idx_utils;
use super::Idx;
use crate::bitmap::{Bitmap, BitmapInt, BitmapIter};
use crate::db::{DBCol, DBColInner};
use crate::engine::EXEC_INFO;
use crate::expr::EqualityConstraint;
use crate::file_table::{FileCol, FileTableRef};
use crate::utils::{self, IteratorAllEqExt, IteratorFilterExt};
use itertools::Itertools;
use log::info;
use rustc_hash::FxHashSet;
use std::borrow::Cow;
use std::collections::BTreeSet;
use std::fmt;
use std::hash::{BuildHasherDefault, Hash};
use std::rc::Rc;
use std::time::Instant;

// FIXME TODO: Think about whether having a column-oriented index can significantly speed up some
// things. It might be okay if we just separately store a bitmap of valid indices per table

pub type IdxRow = Vec<BitmapInt>;
pub type IdxCol = Vec<BitmapInt>;

// TODO: Think about whether we want a multi-table keyed index. This may be desirable when taking
// into account the fact that predicates don't need to be evaluated on the entire Idx, but rather
// on the subset of tables which are relevant to the predicate
// Also, when we do a join between two tables and there are further join-like constraints on top,
// we may multi-table keyed joins (see JOB)
#[derive(Clone)]
pub enum BasicIdx {
    Bmap(BmapIdx),
    Row(RowIdx),
    Col(ColIdx),
    Keyed(KeyedIdx),
}

#[derive(Clone)]
pub struct BmapIdx {
    table_ref: Rc<FileTableRef>,
    bmap: Bitmap,
}

#[derive(Clone)]
pub struct ColIdx {
    table_refs: Vec<Rc<FileTableRef>>,
    cols: Vec<IdxCol>,
}

#[derive(Clone)]
pub struct RowIdx {
    table_refs: Vec<Rc<FileTableRef>>,
    rows: Vec<IdxRow>,
}

/// Keyed on a single table. Implemented as a sparse vector for faster access.
#[derive(Clone)]
pub struct KeyedIdx {
    table_refs: Vec<Rc<FileTableRef>>,
    key: Rc<FileTableRef>,
    keyed_rows: Vec<Vec<IdxRow>>,
}

/// Iterator over `IdxRow`s
pub enum IdxRowIter<'a> {
    Bmap(BitmapIter<'a>),
    Row(std::slice::Iter<'a, IdxRow>),
    Col { idx: &'a ColIdx, counter: usize },
    Keyed(std::iter::Flatten<std::slice::Iter<'a, Vec<IdxRow>>>),
    //Tagged {
    //    iters: Vec<IdxRowIter<'a>>,
    //    iter_idx: usize,
    //},
}

/// Iterator over a single column.
pub enum IdxColIter<'a> {
    Bmap(BitmapIter<'a>),
    Row {
        iter: IdxRowIter<'a>,
        table_ref_idx: usize,
    },
    Col(std::slice::Iter<'a, BitmapInt>),
    //Tagged {
    //    iters: Vec<IdxColIter<'a>>,
    //    iter_idx: usize,
    //},
    // TODO: Special case keyed table?
    //Keyed {
    //    iter: IdxRowIter<'a>,
    //    table_idx: usize,
    //},
}

impl BasicIdx {
    pub fn new(table_ref: Rc<FileTableRef>) -> Self {
        Self::Bmap(BmapIdx {
            bmap: Bitmap::from_sorted_iter(0..table_ref.table.len() as BitmapInt).unwrap(),
            table_ref,
        })
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Bmap(idx) => idx.bmap.len() as usize,
            Self::Row(idx) => idx.rows.len(),
            Self::Col(idx) => idx.cols[0].len(),
            Self::Keyed(idx) => idx.keyed_rows.iter().map(|rows| rows.len()).sum(),
            //Self::Tagged(idx) => idx.0.iter().map(|idx| idx.1.len()).sum(),
        }
    }

    pub fn iter(&self) -> IdxRowIter<'_> {
        match self {
            Self::Bmap(idx) => IdxRowIter::Bmap(idx.bmap.iter()),
            Self::Row(idx) => IdxRowIter::Row(idx.rows.iter()),
            Self::Col(idx) => IdxRowIter::Col { idx, counter: 0 },
            Self::Keyed(idx) => IdxRowIter::Keyed(idx.keyed_rows.iter().flatten()),
            //Self::Tagged(idx) => IdxRowIter::Tagged {
            //    iters: idx.0.iter().map(|idx| idx.1.iter()).collect(),
            //    iter_idx: 0,
            //},
        }
    }

    pub fn col_iter(&self, table_ref: &FileTableRef) -> IdxColIter<'_> {
        let table_ref_idx = self.table_idx_ref(table_ref).expect("Table not included");
        match self {
            Self::Bmap(idx) => IdxColIter::Bmap(idx.bmap.iter()),
            Self::Row(_) => IdxColIter::Row {
                iter: self.iter(),
                table_ref_idx,
            },
            Self::Col(idx) => IdxColIter::Col(idx.cols[table_ref_idx].iter()),
            Self::Keyed(_) => IdxColIter::Row {
                iter: self.iter(),
                table_ref_idx,
            },
        }
    }

    pub fn join(&self, other: &Self, constraint: &EqualityConstraint, left_based: bool) -> Self {
        if left_based {
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
    }

    // Builds a hash table on self's values.
    fn hash_join(
        &self,
        my_col: &FileCol,
        my_table_ref: &FileTableRef,
        other: &Self,
        other_col: &FileCol,
        other_table_ref: &FileTableRef,
    ) -> Self {
        fn do_hash_join<T: Hash + Eq + fmt::Debug>(
            my_idx: &BasicIdx,
            other_idx: &BasicIdx,
            my_vals: DBColInner<T>,
            other_vals: DBColInner<T>,
        ) -> BasicIdx {
            let now = Instant::now();
            let rev_map = idx_utils::build_rev_map(my_idx, &mut my_vals.iter());
            let join_map_time_ms = now.elapsed().as_millis();

            let now = Instant::now();
            let out = idx_utils::hash_join_vals(
                &rev_map,
                my_idx.table_refs().into_owned(),
                other_idx,
                other_vals.slice(0..other_vals.len()),
            );
            let join_iter_time_ms = now.elapsed().as_millis();

            if EXEC_INFO.with(|exec_info| exec_info.borrow().debug_times) {
                let left_table_refs = my_idx.table_refs().iter().join(", ");
                let right_table_refs = other_idx.table_refs().iter().join(", ");
                info!(
                    "join([{}] [{}]): map create time {} ms",
                    left_table_refs, right_table_refs, join_map_time_ms
                );
                info!(
                    "join([{}] [{}]): map iter time {} ms",
                    left_table_refs, right_table_refs, join_iter_time_ms
                );
            }

            out
        }

        if self.len() == 0 || other.len() == 0 {
            return BasicIdx::Row(RowIdx::new(
                utils::append(
                    self.table_refs().into_owned(),
                    other.table_refs().into_owned(),
                ),
                vec![],
            ));
        }

        let now = Instant::now();
        let my_vals = my_col.read(self.col_iter(my_table_ref));
        let other_vals = other_col.read(other.col_iter(other_table_ref));

        EXEC_INFO.with(|exec_info| {
            let mut exec_info = exec_info.borrow_mut();
            exec_info.stats.num_join_tuples += (my_vals.len() + other_vals.len()) as u128;
        });

        if EXEC_INFO.with(|exec_info| exec_info.borrow().debug_times) {
            info!(
                "join {} = {} read_time_ms: {}, num left vals {} num right vals {}",
                my_col.full_name(),
                other_col.full_name(),
                now.elapsed().as_millis(),
                my_vals.len(),
                other_vals.len()
            );
        }

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

    pub fn filter(&self, bmap: &Bitmap) -> Self {
        match self {
            Self::Bmap(idx) => Self::Bmap(idx.filter(bmap)),
            Self::Row(idx) => Self::Row(idx.filter(bmap)),
            Self::Col(idx) => Self::Col(idx.filter(bmap)),
            Self::Keyed(idx) => Self::Keyed(idx.filter(bmap)),
            //Self::Tagged(idx) => {
            //    let mut idx_start = 0;
            //    Self::Tagged(TaggedIdx(
            //        idx.0
            //            .iter()
            //            .map(|(tag_id, idx)| {
            //                let result = idx.filter(
            //                    &(bmap
            //                        & Bitmap::from_sorted_iter(
            //                            idx_start..idx_start + idx.len() as BitmapInt,
            //                        )
            //                        .unwrap()),
            //                );
            //                idx_start += idx.len() as BitmapInt;
            //                (tag_id.clone(), result)
            //            })
            //            .collect(),
            //    ))
            //}
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // Get the index of the given table in the index.
    fn table_idx_ref(&self, table_ref: &FileTableRef) -> Option<usize> {
        match self {
            Self::Bmap(idx) => {
                if idx.table_ref.as_ref() == table_ref {
                    Some(0)
                } else {
                    None
                }
            }
            Self::Row(idx) => idx.table_refs.iter().position(|t| t.as_ref() == table_ref),
            Self::Col(idx) => idx.table_refs.iter().position(|t| t.as_ref() == table_ref),
            Self::Keyed(idx) => idx.table_refs.iter().position(|t| t.as_ref() == table_ref),
            //Self::Tagged(_) => panic!("A TaggedIdx has more than one possible table_idx"),
        }
    }

    //fn to_keyed(&self, key: &FileTableRef) -> Cow<'_, KeyedIdx> {
    //    match self {
    //        Self::Keyed(idx) if idx.key.as_ref() == key => Cow::Borrowed(idx),
    //        //Self::Tagged(_) => panic!("Use to_tagged_keyed for TaggedIdx"),
    //        _ => {
    //            let key_idx = self.table_idx(key).expect("Does not have key");
    //            let mut keyed_rows = vec![vec![]; key.len()];
    //            for row in self.iter() {
    //                keyed_rows[row[key_idx] as usize].push(row.into_owned())
    //            }

    //            let tables = self.tables().into_owned();
    //            Cow::Owned(KeyedIdx {
    //                key: tables[key_idx].clone(),
    //                tables,
    //                keyed_rows,
    //            })
    //        }
    //    }
    //}

    //
    //fn to_tagged_keyed(&self, key: &FileTable) -> Cow<'_, TaggedKeyedIdx> {
    //}

    pub fn table_refs(&self) -> Cow<'_, Vec<Rc<FileTableRef>>> {
        match self {
            Self::Bmap(idx) => Cow::Owned(vec![idx.table_ref.clone()]),
            Self::Row(idx) => Cow::Borrowed(&idx.table_refs),
            Self::Col(idx) => Cow::Borrowed(&idx.table_refs),
            Self::Keyed(idx) => Cow::Borrowed(&idx.table_refs),
            //Self::Tagged(idx) => idx.0.first().unwrap().1.tables(),
        }
    }

    pub fn index(&self, idx: usize) -> Cow<'_, IdxRow> {
        match self {
            Self::Bmap(bmap_idx) => {
                Cow::Owned(vec![bmap_idx.bmap.select(idx as BitmapInt).unwrap()])
            }
            Self::Row(row_idx) => Cow::Borrowed(&row_idx.rows[idx]),
            _ => panic!("Unsupported Idx type"),
        }
    }

    pub fn union(idxs: Vec<&Self>) -> Self {
        assert!(idxs
            .iter()
            .all_eq(|idx| BTreeSet::from_iter(idx.table_refs().iter().cloned())));
        let mut rows = FxHashSet::with_hasher(BuildHasherDefault::default());
        let table_refs = idxs.first().unwrap().table_refs().into_owned();
        for idx in idxs {
            let idx_table_refs = idx.table_refs();
            if &table_refs != &*idx_table_refs {
                let reorder_map: Vec<_> = table_refs
                    .iter()
                    .map(|table_ref| idx_table_refs.iter().position(|t| t == table_ref).unwrap())
                    .collect();
                rows.extend(
                    idx.iter()
                        .map(|row| reorder_map.iter().map(|&idx| row[idx]).collect()),
                );
            } else {
                rows.extend(idx.iter().map(|row| row.into_owned()));
            }
        }
        Self::Row(RowIdx::new(table_refs, Vec::from_iter(rows)))
    }
}

impl BmapIdx {
    fn filter(&self, bmap: &Bitmap) -> Self {
        Self {
            table_ref: self.table_ref.clone(),
            bmap: self.bmap.iter().filter_by_index2(bmap).collect(),
        }
    }
}

impl ColIdx {
    fn filter(&self, bmap: &Bitmap) -> Self {
        Self {
            table_refs: self.table_refs.clone(),
            cols: self
                .cols
                .iter()
                .map(|col| bmap.iter().map(|idx| col[idx as usize]).collect())
                .collect(),
        }
    }
}

impl RowIdx {
    pub fn new(table_refs: Vec<Rc<FileTableRef>>, rows: Vec<IdxRow>) -> Self {
        Self { table_refs, rows }
    }

    fn filter(&self, bmap: &Bitmap) -> Self {
        Self {
            table_refs: self.table_refs.clone(),
            rows: bmap
                .iter()
                .map(|idx| self.rows[idx as usize].clone())
                .collect(),
        }
    }

    //pub fn concat(idxs: impl Iterator<Item = Self>) -> Self {
    //    let mut rows = vec![];
    //    let mut tables = None;
    //    for mut idx in idxs {
    //        if let None = tables {
    //            tables = Some(idx.tables);
    //        }
    //        rows.append(&mut idx.rows);
    //    }
    //    Self {
    //        table_refs: table_refs.expect("Tables was never set, zero-length iterator"),
    //        rows,
    //    }
    //}
}

impl KeyedIdx {
    // Get the bitmap of indices the key table
    fn key_bmap(&self) -> Bitmap {
        Bitmap::from_sorted_iter(
            self.keyed_rows
                .iter()
                .enumerate()
                .filter_map(|(idx, rows)| {
                    if !rows.is_empty() {
                        Some(idx as BitmapInt)
                    } else {
                        None
                    }
                }),
        )
        .unwrap()
    }

    fn key_selectivity(&self) -> f64 {
        self.keyed_rows
            .iter()
            .filter(|rows| !rows.is_empty())
            .count() as f64
            / self.key.table.len() as f64
    }

    fn filter(&self, bmap: &Bitmap) -> Self {
        if let Some(max) = bmap.max() {
            let mut keyed_rows = vec![vec![]; self.keyed_rows.len()];
            let mut counter = 0;
            for (i, rows) in self.keyed_rows.iter().enumerate() {
                if counter > max {
                    break;
                }

                for row in rows {
                    if counter > max {
                        break;
                    }

                    if bmap.contains(counter) {
                        keyed_rows[i].push(row.clone());
                    }

                    counter += 1;
                }
            }

            Self {
                table_refs: self.table_refs.clone(),
                key: self.key.clone(),
                keyed_rows,
            }
        } else {
            self.clone()
        }
    }
}

impl<'a> Iterator for IdxRowIter<'a> {
    type Item = Cow<'a, IdxRow>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Bmap(iter) => iter.next().map(|v| Cow::Owned(vec![v])),
            Self::Row(iter) => iter.next().map(|row| Cow::Borrowed(row)),
            Self::Col { idx, counter } => idx.cols.first().and_then(|col| {
                if col.len() > *counter {
                    let row = idx.cols.iter().map(|col| col[*counter]).collect();
                    *counter += 1;
                    Some(Cow::Owned(row))
                } else {
                    None
                }
            }),
            Self::Keyed(iter) => iter.next().map(|row| Cow::Borrowed(row)),
            //Self::Tagged { iters, iter_idx } => {
            //    while *iter_idx < iters.len() {
            //        let result = iters[*iter_idx].next();
            //        if result.is_some() {
            //            return result;
            //        } else {
            //            *iter_idx += 1;
            //        }
            //    }
            //    None
            //}
        }
    }
}

impl<'a> Iterator for IdxColIter<'a> {
    type Item = BitmapInt;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Bmap(iter) => iter.next(),
            Self::Row {
                iter,
                table_ref_idx,
            } => iter.next().map(|row| row[*table_ref_idx]),
            Self::Col(iter) => iter.next().copied(),
            //Self::Keyed { iter, table_idx } => iter.next().map(|row| row[*table_idx]),
            //Self::Tagged { iters, iter_idx } => {
            //    while *iter_idx < iters.len() {
            //        let result = iters[*iter_idx].next();
            //        if result.is_some() {
            //            return result;
            //        } else {
            //            *iter_idx += 1;
            //        }
            //    }
            //    None
            //}
        }
    }
}

impl<'a> TryFrom<&'a Idx> for &'a BasicIdx {
    type Error = &'static str;

    fn try_from(value: &'a Idx) -> Result<Self, Self::Error> {
        match value {
            Idx::Basic(idx) => Ok(idx),
            _ => Err("Not tagged idx type"),
        }
    }
}

impl fmt::Debug for BasicIdx {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BasicIdx(type: {}, table_refs: [{}], idxs: [{}])",
            match self {
                Self::Bmap(_) => "Bmap".to_string(),
                Self::Row(_) => "Row".to_string(),
                Self::Col(_) => "Col".to_string(),
                Self::Keyed(idx) => format!("Keyed({})", idx.key),
                //Self::Tagged(_) => "Tagged".to_string(),
            },
            self.table_refs()
                .iter()
                //.map(|table| table.name())
                .join(", "),
            self.iter().map(|row| format!("{:?}", row)).join(", ")
        )
    }
}
