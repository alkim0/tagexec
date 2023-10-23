use crate::bitmap::{Bitmap, BitmapInt};
use crate::engine::EXEC_INFO;
use approx;
use auto_enums::auto_enum;
use chrono::{DateTime, Duration, Utc};
use either::Either;
use float_ord::FloatOrd;
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::mem;

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub enum DBType {
    Str,
    Int,
    Long,
    Float,
    Double,
    Bool,
    DateTime,
    Duration,
}

/// A vector of db values.
#[derive(Debug, Clone, PartialEq)]
pub enum DBCol {
    Int(DBColInner<i32>),
    Long(DBColInner<i64>),
    Bool(DBColInner<bool>),
    Float(DBColInner<f32>),
    Double(DBColInner<f64>),
    Str(DBColInner<String>),
    DateTime(DBColInner<DateTime<Utc>>),
    Duration(DBColInner<Duration>),
}

/// Singleton db value.
#[derive(Clone, PartialOrd, Debug)]
pub enum DBSingle {
    Int(i32),
    Long(i64),
    Bool(bool),
    Float(f32),
    Double(f64),
    Str(String),
    DateTime(DateTime<Utc>),
    Duration(Duration),
}

pub enum DBRef<'a> {
    Int(&'a i32),
    Long(&'a i64),
    Float(&'a f32),
    Double(&'a f64),
    Bool(&'a bool),
    Str(&'a str),
    DateTime(&'a DateTime<Utc>),
    Duration(&'a Duration),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DBVals {
    Single(DBSingle),
    Col(DBCol),
}

#[derive(Debug, PartialEq, Eq)]
pub struct DBResultSet(Vec<DBVals>);

impl DBResultSet {
    pub fn new(results: Vec<DBVals>) -> Self {
        Self(results)
    }

    pub fn iter(&self) -> impl Iterator<Item = &DBVals> {
        self.0.iter()
    }

    pub fn len(&self) -> usize {
        self.0[0].len()
    }
}

/// In general, all modifications should be done on the `Values` type. After projection, we expect
/// only to do read-only operations.
#[derive(Debug, Clone)]
pub enum DBColInner<T> {
    Values(Vec<T>),
    Projected { values: Vec<T>, index: Vec<usize> },
}

pub struct DBColInnerSlice<'a, T> {
    base: &'a DBColInner<T>,
    range: std::ops::Range<usize>,
}

pub struct DBColInnerSliceIter<'a, T> {
    slice: DBColInnerSlice<'a, T>,
    index: usize,
}

impl<'a, T> Iterator for DBColInnerSliceIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index > self.slice.range.end {
            None
        } else {
            let out = match &self.slice.base {
                DBColInner::Values(values) => Some(&values[self.index]),
                DBColInner::Projected { values, index } => Some(&values[index[self.index]]),
            };
            self.index += 1;
            out
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.slice.range.end - self.index;
        (size, Some(size))
    }
}

impl<'a, T> IntoIterator for DBColInnerSlice<'a, T> {
    type Item = &'a T;
    type IntoIter = DBColInnerSliceIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        let index = self.range.start;
        DBColInnerSliceIter { slice: self, index }
    }
}

impl<T> DBColInner<T> {
    #[auto_enum(Iterator)]
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        match self {
            Self::Values(values) => values.iter(),
            Self::Projected { values, index } => index.iter().map(|&i| &values[i]),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Values(values) => values.len(),
            Self::Projected { index, .. } => index.len(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self::Values(Vec::with_capacity(capacity))
    }

    pub fn push(&mut self, val: T) {
        match self {
            Self::Values(values) => values.push(val),
            Self::Projected { .. } => panic!("Do not expect to modify after projecting"),
        }
    }

    pub fn slice(&self, range: std::ops::Range<usize>) -> DBColInnerSlice<'_, T> {
        DBColInnerSlice { base: self, range }
    }

    fn project(self, output_idx: Vec<usize>) -> Self {
        match self {
            Self::Values(values) => Self::Projected {
                values,
                index: output_idx,
            },
            Self::Projected { values, index } => {
                let mut out = Vec::with_capacity(output_idx.len());
                out.extend(output_idx.into_iter().map(|i| index[i]));
                Self::Projected { values, index: out }
            }
        }
    }

    // XXX: Technically we know when we will reuse a DBCol or not, so we can decide ahead of time
    // whether we want to keep it in cache + filter to create a copy or add an into_filter to
    // consume the original
    fn filter(&self, bmap: &Bitmap) -> Self
    where
        T: Clone,
    {
        let threshold = EXEC_INFO.with(|exec_info| exec_info.borrow().selectivity_threshold);
        match self {
            Self::Values(values) => {
                let selectivity = bmap.len() as f64 / values.len() as f64;
                let mut out = Vec::with_capacity(bmap.len() as usize);
                if selectivity < threshold {
                    out.extend(bmap.iter().map(|i| values[i as usize].clone()));
                } else {
                    out.extend(values.iter().enumerate().filter_map(|(i, val)| {
                        if bmap.contains(i as BitmapInt) {
                            Some(val.clone())
                        } else {
                            None
                        }
                    }));
                }
                Self::Values(out)
            }
            Self::Projected { values, index } => {
                // XXX: Note that this doesn't actually shrink values, only modifies index.
                let selectivity = bmap.len() as f64 / index.len() as f64;
                let mut out = Vec::with_capacity(bmap.len() as usize);
                if selectivity < threshold {
                    out.extend(bmap.iter().map(|i| index[i as usize]));
                } else {
                    out.extend(index.iter().enumerate().filter_map(|(i, idx)| {
                        if bmap.contains(i as BitmapInt) {
                            Some(idx)
                        } else {
                            None
                        }
                    }));
                }
                Self::Projected {
                    values: values.clone(),
                    index: out,
                }
            }
        }
    }

    //pub fn build_rev_map<'a>(&self, mut row_iter: impl Iterator<Item = Cow<'a, IdxRow>>) -> FxHashMap<&T, Vec<Cow<'a, IdxRow>>> {
    //    match self {
    //        Self::Values(values) => {
    //            let mut rev_map = FxHashMap::with_capacity_and_hasher(values.len(), BuildHasherDefault::default());
    //            for (row, val) in idx_iter.zip(values) {
    //                rev_map.entry(val).or_insert(vec![]).push(row.to_owned
    //            }
    //        }
    //    }
    //}
}

impl<T: Clone> IntoIterator for DBColInner<T> {
    type Item = T;
    type IntoIter = DBColInnerIntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            Self::Values(values) => DBColInnerIntoIter::Values(values.into_iter()),
            Self::Projected { values, index } => DBColInnerIntoIter::Projected {
                values,
                index_iter: index.into_iter(),
            },
        }
    }
}

pub enum DBColInnerIntoIter<T> {
    Values(std::vec::IntoIter<T>),
    Projected {
        values: Vec<T>,
        index_iter: std::vec::IntoIter<usize>,
    },
}

impl<T: Clone> Iterator for DBColInnerIntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Values(iter) => iter.next(),
            Self::Projected { values, index_iter } => {
                index_iter.next().map(|idx| values[idx].clone())
            }
        }
    }
}

impl<T> Extend<T> for DBColInner<T> {
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        match self {
            Self::Values(values) => values.extend(iter),
            Self::Projected { .. } => panic!("Do not expect to modify after projecting"),
        }
    }
}

impl<T> std::ops::Index<usize> for DBColInner<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match self {
            Self::Values(values) => &values[index],
            Self::Projected {
                values,
                index: proj_idx,
            } => &values[proj_idx[index]],
        }
    }
}

impl<T: PartialEq> PartialEq for DBColInner<T> {
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}

impl From<&DBVals> for Bitmap {
    fn from(vals: &DBVals) -> Self {
        Bitmap::from_sorted_iter(vals.iter_as_bool().enumerate().filter_map(|(i, b)| {
            if *b {
                Some(i as BitmapInt)
            } else {
                None
            }
        }))
        .unwrap()
    }
}

macro_rules! impl_dbvals_iter_as {
    { $(($func:ident, $db_type:ident, $type:ty)),*$(,)? } => {
        $(
            pub fn $func(&self) -> impl Iterator<Item = &$type> + '_ {
                match self {
                    Self::Col(DBCol::$db_type(col)) => Either::Left(col.iter()),
                    Self::Single(DBSingle::$db_type(single)) => Either::Right(std::iter::once(single)),
                    _ => panic!("Not of type {}", stringify!($db_type)),
                }
            }
        )*
    }
}

macro_rules! impl_dbvals_into_iter_as {
    { $(($func:ident, $single_func:ident, $type:ty)),*$(,)? } => {
        $(
            pub fn $func(self) -> impl Iterator<Item = $type> {
                match self {
                    Self::Col(col) => Either::Left(col.$func()),
                    Self::Single(single) => Either::Right(std::iter::once(single.$single_func())),
                }
            }
        )*
    }
}

impl DBVals {
    pub fn len(&self) -> usize {
        match self {
            Self::Single(_) => 1,
            Self::Col(col) => col.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Self::Single(_) => false,
            Self::Col(col) => col.len() == 0,
        }
    }

    pub fn data_type(&self) -> DBType {
        match self {
            Self::Single(single) => single.data_type(),
            Self::Col(col) => col.data_type(),
        }
    }

    pub fn is_single(&self) -> bool {
        match self {
            Self::Single(_) => true,
            Self::Col(_) => false,
        }
    }

    pub fn filter(&self, bmap: &Bitmap) -> Self {
        match self {
            Self::Col(col) => Self::Col(col.filter(bmap)),
            Self::Single(_) => {
                panic!("You probably don't mean to use this function with a Single")
            }
        }
    }

    /// Differs from the From in that it can use a base bitmap (the first boolean value refers to
    /// the true/false value of the first bitmap index).
    pub fn to_bitmap(&self, base: Option<&Bitmap>) -> Bitmap {
        match base {
            None => Bitmap::from(self),
            Some(base) => Bitmap::from_sorted_iter(
                base.iter()
                    .zip(self.iter_as_bool())
                    .filter_map(|(idx, b)| if *b { Some(idx) } else { None }),
            )
            .unwrap(),
        }
    }

    pub fn iter_to_long(&self) -> impl Iterator<Item = i64> + '_ {
        match self {
            Self::Single(single) => Either::Left(std::iter::once(single.to_long())),
            Self::Col(col) => Either::Right(col.iter_to_long()),
        }
    }

    pub fn iter_to_double(&self) -> impl Iterator<Item = f64> + '_ {
        match self {
            Self::Single(single) => Either::Left(std::iter::once(single.to_double())),
            Self::Col(col) => Either::Right(col.iter_to_double()),
        }
    }

    impl_dbvals_iter_as! {
        (iter_as_int, Int, i32),
        (iter_as_long, Long, i64),
        (iter_as_float, Float, f32),
        (iter_as_double, Double, f64),
        (iter_as_str, Str, String),
        (iter_as_bool, Bool, bool),
        (iter_as_datetime, DateTime, DateTime<Utc>),
        (iter_as_duration, Duration, Duration),
    }

    /// NOTE: This is slower than traversing the explicit type, and should really only be used for
    /// debugging purposes.
    pub fn iter_as_dbref(&self) -> impl Iterator<Item = DBRef<'_>> {
        match self {
            Self::Col(col) => Either::Left(col.iter_as_dbref()),
            Self::Single(single) => Either::Right(std::iter::once(single.as_dbref())),
        }
    }

    //impl_dbvals_into_iter_as! {
    //    (into_iter_as_int, into_int, i32),
    //    (into_iter_as_long, into_long, i64),
    //    (into_iter_as_float, into_float, f32),
    //    (into_iter_as_double, into_double, f64),
    //    (into_iter_as_str, into_str, String),
    //    (into_iter_as_bool, into_bool, bool),
    //    (into_iter_as_datetime, into_datetime, DateTime<Utc>),
    //    (into_iter_as_duration, into_duration, Duration),
    //}
}

//macro_rules! dispatch_db_enum {
//    ($match:expr, $in_type:tt::_$(($val:pat))? => $out_type:tt::_$(($expr:expr))?) => {
//        dispatch_db_enum!($match, (Int, Long, Float, Double, Bool, Str, DateTime, Duration), $in_type::_$(($val))? => $out_type::_$(($expr))?)
//    };
//
//    ($match:expr, $in_type:tt::_$(($val:pat))? => $expr:expr) => {
//        dispatch_db_enum!($match, (Int, Long, Float, Double, Bool, Str, DateTime, Duration), $in_type::_$(($val))? => $expr)
//    };
//
//    ($match:expr, $in_type:tt::_$(($val:pat))? => $expr:expr => |$x:pat_param| $out_type:tt::_($outexpr:expr)) => {
//        dispatch_db_enum!($match, (Int, Long, Float, Double, Bool, Str, DateTime, Duration), $in_type::_$(($val))? => $expr => |$x| $out_type::_($outexpr))
//    };
//
//    ($match:expr, ($($subtype:tt),*), $in_type:tt::_($val:pat) => $out_type:tt::_($expr:expr)$(, _ => $default_expr:expr)?) => {
//        match $match {
//            $(
//                $in_type::$subtype($val) => $out_type::$subtype($expr)
//            ),*
//            $(,
//                _ => $default_expr
//            )?
//        }
//    };
//
//    ($match:expr, ($($subtype:tt),*), $in_type:tt::_($val:pat) => $out_type:tt::_$(, _ => $default_expr:expr)?) => {
//        match $match {
//            $(
//                $in_type::$subtype($val) => $out_type::$subtype
//            ),*
//            $(,
//                _ => $default_expr
//            )?
//        }
//    };
//
//    ($match:expr, ($($subtype:tt),*), $in_type:tt::_($val:pat) => $expr:expr$(, _ => $default_expr:expr)?) => {
//        match $match {
//            $(
//                $in_type::$subtype($val) => $expr
//            ),*
//            $(,
//                _ => $default_expr
//            )?
//        }
//    };
//
//    ($match:expr, ($($subtype:tt),*), $in_type:tt::_($val:pat) => $expr:expr => |$x:pat_param| $out_type:tt::_($outexpr:expr)$(, _ => $default_expr:expr)?) => {
//        match $match {
//            $(
//                $in_type::$subtype($val) => $expr.map(|$x| $out_type::$subtype($outexpr))
//            ),*
//            $(,
//                _ => $default_expr
//            )?
//        }
//    };
//
//    ($match:expr, ($($subtype:tt),*), $in_type:tt::_ => $out_type:tt::_($expr:expr)$(, _ => $default_expr:expr)?) => {
//        match $match {
//            $(
//                $in_type::$subtype => $out_type::$subtype($expr)
//            ),*
//            $(,
//                _ => $default_expr
//            )?
//        }
//    };
//
//    ($match:expr, ($($subtype:tt),*), $in_type:tt::_ => $out_type:tt::_$(, _ => $default_expr:expr)?) => {
//        match $match {
//            $(
//                $in_type::$subtype => $out_type::$subtype
//            ),*
//            $(,
//                _ => $default_expr
//            )?
//        }
//    };
//
//    ($match:expr, ($($subtype:tt),*), $in_type:tt::_ => $expr:expr$(, _ => $default_expr:expr)?) => {
//        match $match {
//            $(
//                $in_type::$subtype => $expr
//            ),*
//            $(,
//                _ => $default_expr
//            )?
//        }
//    };
//
//    ($match:expr, ($($subtype:tt),*), $in_type:tt::_ => $expr:expr => |$x:pat_param| $out_type:tt::_($outexpr:expr)$(, _ => $default_expr:expr)?) => {
//        match $match {
//            $(
//                $in_type::$subtype => $expr.map(|$x| $out_type::$subtype($outexpr))
//            ),*
//            $(,
//                _ => $default_expr
//            )?
//        }
//    };
//}

macro_rules! impl_iter_as {
    {$(($func:ident, $db_type:ident, $type:ty)),*$(,)?} => {
        $(
            pub fn $func(&self) -> impl Iterator<Item = &$type> + '_ {
                match self {
                    Self::$db_type(col) => col.iter(),
                    _ => panic!("Not of {} type", stringify!($db_type)),
                }
            }
        )*
    };
}

macro_rules! impl_into_iter_as {
    {$(($func:ident, $db_type:ident, $type:ty)),*$(,)?} => {
        $(
            pub fn $func(self) -> impl Iterator<Item = $type> {
                match self {
                    Self::$db_type(col) => col.into_iter(),
                    _ => panic!("Not of {} type", stringify!($db_type)),
                }
            }
        )*
    };
}

impl DBCol {
    pub fn len(&self) -> usize {
        match self {
            Self::Int(col) => col.len(),
            Self::Long(col) => col.len(),
            Self::Float(col) => col.len(),
            Self::Double(col) => col.len(),
            Self::Bool(col) => col.len(),
            Self::Str(col) => col.len(),
            Self::DateTime(col) => col.len(),
            Self::Duration(col) => col.len(),
        }
    }

    pub fn data_type(&self) -> DBType {
        match self {
            Self::Int(_) => DBType::Int,
            Self::Long(_) => DBType::Long,
            Self::Float(_) => DBType::Float,
            Self::Double(_) => DBType::Double,
            Self::Bool(_) => DBType::Bool,
            Self::Str(_) => DBType::Str,
            Self::DateTime(_) => DBType::DateTime,
            Self::Duration(_) => DBType::Duration,
        }
    }

    pub fn with_capacity(data_type: &DBType, capacity: usize) -> Self {
        match data_type {
            DBType::Int => Self::Int(DBColInner::with_capacity(capacity)),
            DBType::Long => Self::Long(DBColInner::with_capacity(capacity)),
            DBType::Float => Self::Float(DBColInner::with_capacity(capacity)),
            DBType::Double => Self::Double(DBColInner::with_capacity(capacity)),
            DBType::Bool => Self::Bool(DBColInner::with_capacity(capacity)),
            DBType::Str => Self::Str(DBColInner::with_capacity(capacity)),
            DBType::DateTime => Self::DateTime(DBColInner::with_capacity(capacity)),
            DBType::Duration => Self::Duration(DBColInner::with_capacity(capacity)),
        }
    }

    /// Roughly `output_idx.into_iter().map(|idx| self[idx]).collect()`.
    pub fn project(self, output_idx: Vec<usize>) -> Self {
        match self {
            Self::Int(inner) => Self::Int(inner.project(output_idx)),
            Self::Long(inner) => Self::Long(inner.project(output_idx)),
            Self::Float(inner) => Self::Float(inner.project(output_idx)),
            Self::Double(inner) => Self::Double(inner.project(output_idx)),
            Self::Bool(inner) => Self::Bool(inner.project(output_idx)),
            Self::Str(inner) => Self::Str(inner.project(output_idx)),
            Self::DateTime(inner) => Self::DateTime(inner.project(output_idx)),
            Self::Duration(inner) => Self::Duration(inner.project(output_idx)),
        }
    }

    pub fn filter(&self, bmap: &Bitmap) -> Self {
        match self {
            Self::Int(inner) => Self::Int(inner.filter(bmap)),
            Self::Long(inner) => Self::Long(inner.filter(bmap)),
            Self::Float(inner) => Self::Float(inner.filter(bmap)),
            Self::Double(inner) => Self::Double(inner.filter(bmap)),
            Self::Bool(inner) => Self::Bool(inner.filter(bmap)),
            Self::Str(inner) => Self::Str(inner.filter(bmap)),
            Self::DateTime(inner) => Self::DateTime(inner.filter(bmap)),
            Self::Duration(inner) => Self::Duration(inner.filter(bmap)),
        }
    }

    ///// Roughly `output_idx.into_iter().map(|idx| self[idx]).collect()`.
    //pub fn project<T: AsPrimitive<usize>>(&self, output_idx: Vec<T>) -> Self {
    //    match self {
    //        Self::Int(col) => {
    //            let mut out = Vec::with_capacity(output_idx.len());
    //            out.extend(output_idx.into_iter().map(|idx| col[idx.as_()]));
    //            Self::Int(out)
    //        }
    //        Self::Long(col) => {
    //            let mut out = Vec::with_capacity(output_idx.len());
    //            out.extend(output_idx.into_iter().map(|idx| col[idx.as_()]));
    //            Self::Long(out)
    //        }
    //        Self::Float(col) => {
    //            let mut out = Vec::with_capacity(output_idx.len());
    //            out.extend(output_idx.into_iter().map(|idx| col[idx.as_()]));
    //            Self::Float(out)
    //        }
    //        Self::Double(col) => {
    //            let mut out = Vec::with_capacity(output_idx.len());
    //            out.extend(output_idx.into_iter().map(|idx| col[idx.as_()]));
    //            Self::Double(out)
    //        }
    //        Self::Bool(col) => {
    //            let mut out = Vec::with_capacity(output_idx.len());
    //            out.extend(output_idx.into_iter().map(|idx| col[idx.as_()]));
    //            Self::Bool(out)
    //        }
    //        Self::Str(col) => {
    //            let mut out = Vec::with_capacity(output_idx.len());
    //            out.extend(output_idx.into_iter().map(|idx| col[idx.as_()].clone()));
    //            Self::Str(out)
    //        }
    //        Self::DateTime(col) => {
    //            let mut out = Vec::with_capacity(output_idx.len());
    //            out.extend(output_idx.into_iter().map(|idx| col[idx.as_()]));
    //            Self::DateTime(out)
    //        }
    //        Self::Duration(col) => {
    //            let mut out = Vec::with_capacity(output_idx.len());
    //            out.extend(output_idx.into_iter().map(|idx| col[idx.as_()]));
    //            Self::Duration(out)
    //        }
    //    }
    //}

    //pub fn filter(&self, bmap: &Bitmap) -> Self {
    //    match self {
    //        Self::Int(col) => {
    //            let mut out = Vec::with_capacity(bmap.len() as usize);
    //            out.extend(bmap.iter().map(|i| col[i as usize]));
    //            Self::Int(out)
    //        }
    //        Self::Long(col) => {
    //            let mut out = Vec::with_capacity(bmap.len() as usize);
    //            out.extend(bmap.iter().map(|i| col[i as usize]));
    //            Self::Long(out)
    //        }
    //        Self::Float(col) => {
    //            let mut out = Vec::with_capacity(bmap.len() as usize);
    //            out.extend(bmap.iter().map(|i| col[i as usize]));
    //            Self::Float(out)
    //        }
    //        Self::Double(col) => {
    //            let mut out = Vec::with_capacity(bmap.len() as usize);
    //            out.extend(bmap.iter().map(|i| col[i as usize]));
    //            Self::Double(out)
    //        }
    //        Self::Bool(col) => {
    //            let mut out = Vec::with_capacity(bmap.len() as usize);
    //            out.extend(bmap.iter().map(|i| col[i as usize]));
    //            Self::Bool(out)
    //        }
    //        Self::Str(col) => {
    //            let mut out = Vec::with_capacity(bmap.len() as usize);
    //            out.extend(bmap.iter().map(|i| col[i as usize].clone()));
    //            Self::Str(out)
    //        }
    //        Self::DateTime(col) => {
    //            let mut out = Vec::with_capacity(bmap.len() as usize);
    //            out.extend(bmap.iter().map(|i| col[i as usize]));
    //            Self::DateTime(out)
    //        }
    //        Self::Duration(col) => {
    //            let mut out = Vec::with_capacity(bmap.len() as usize);
    //            out.extend(bmap.iter().map(|i| col[i as usize]));
    //            Self::Duration(out)
    //        }
    //    }
    //}

    /// This should really only be used for debugging purposes.
    pub fn index<'a>(&'a self, idx: usize) -> DBRef<'a> {
        match self {
            Self::Int(col) => DBRef::Int(&col[idx]),
            Self::Long(col) => DBRef::Long(&col[idx]),
            Self::Float(col) => DBRef::Float(&col[idx]),
            Self::Double(col) => DBRef::Double(&col[idx]),
            Self::Bool(col) => DBRef::Bool(&col[idx]),
            Self::Str(col) => DBRef::Str(&col[idx]),
            Self::DateTime(col) => DBRef::DateTime(&col[idx]),
            Self::Duration(col) => DBRef::Duration(&col[idx]),
        }
    }

    /// NOTE: This is slower than traversing the explicit type, and should really only be used for
    /// debugging purposes.
    #[auto_enum(Iterator)]
    pub fn iter_as_dbref(&self) -> impl Iterator<Item = DBRef<'_>> {
        match self {
            Self::Int(col) => col.iter().map(|x| DBRef::Int(x)),
            Self::Long(col) => col.iter().map(|x| DBRef::Long(x)),
            Self::Float(col) => col.iter().map(|x| DBRef::Float(x)),
            Self::Double(col) => col.iter().map(|x| DBRef::Double(x)),
            Self::Bool(col) => col.iter().map(|x| DBRef::Bool(x)),
            Self::Str(col) => col.iter().map(|x| DBRef::Str(x)),
            Self::DateTime(col) => col.iter().map(|x| DBRef::DateTime(x)),
            Self::Duration(col) => col.iter().map(|x| DBRef::Duration(x)),
        }
    }

    //impl_iter_as! {
    //    (iter_as_int, Int, i32),
    //    (iter_as_long, Long, i64),
    //    (iter_as_float, Float, f32),
    //    (iter_as_double, Double, f64),
    //    (iter_as_str, Str, String),
    //    (iter_as_bool, Bool, bool),
    //    (iter_as_datetime, DateTime, DateTime<Utc>),
    //    (iter_as_duration, Duration, Duration),
    //}

    //impl_into_iter_as! {
    //    (into_iter_as_int, Int, i32),
    //    (into_iter_as_long, Long, i64),
    //    (into_iter_as_float, Float, f32),
    //    (into_iter_as_double, Double, f64),
    //    (into_iter_as_bool, Bool, bool),
    //    (into_iter_as_str, Str, String),
    //    (into_iter_as_datetime, DateTime, DateTime<Utc>),
    //    (into_iter_as_duration, Duration, Duration),
    //}

    /// Iterates over an integral DBCol and yield (possibly converted) i64 values
    #[auto_enum(Iterator)]
    pub fn iter_to_long(&self) -> impl Iterator<Item = i64> + '_ {
        match self {
            Self::Int(col) => col.iter().map(|i| *i as i64),
            Self::Long(col) => col.iter().copied(),
            _ => panic!("Is not integral type"),
        }
    }

    /// Iterates over a numeric DBCol and yield (possibly converted) f64 values
    #[auto_enum(Iterator)]
    pub fn iter_to_double(&self) -> impl Iterator<Item = f64> + '_ {
        match self {
            Self::Int(col) => col.iter().map(|i| *i as f64),
            Self::Long(col) => col.iter().map(|i| *i as f64),
            Self::Float(col) => col.iter().map(|i| *i as f64),
            Self::Double(col) => col.iter().copied(),
            _ => panic!("Is not numeric type"),
        }
    }
}

macro_rules! impl_as {
    {$(($func:ident, $db_type:ident, $type:ty)),*$(,)?} => {
        $(
            pub fn $func(&self) -> &$type {
                match self {
                    Self::$db_type(x) => x,
                    _ => panic!("Not of {} type", stringify!($db_type)),
                }
            }
        )*
    }
}

macro_rules! impl_into {
    {$(($func:ident, $db_type:ident, $type:ty)),*$(,)?} => {
        $(
            pub fn $func(self) -> $type {
                match self {
                    Self::$db_type(x) => x,
                    _ => panic!("Not of {} type", stringify!($db_type)),
                }
            }
        )*
    }
}

impl DBSingle {
    pub fn data_type(&self) -> DBType {
        match self {
            Self::Int(_) => DBType::Int,
            Self::Long(_) => DBType::Long,
            Self::Float(_) => DBType::Float,
            Self::Double(_) => DBType::Double,
            Self::Bool(_) => DBType::Bool,
            Self::Str(_) => DBType::Str,
            Self::DateTime(_) => DBType::DateTime,
            Self::Duration(_) => DBType::Duration,
        }
    }

    /// Retrieves and possibly converts an integral DBSingle to a i64
    pub fn to_long(&self) -> i64 {
        match self {
            Self::Int(i) => *i as i64,
            Self::Long(i) => *i as i64,
            _ => panic!("Not integral"),
        }
    }

    /// Retrieves and possibly converts a numeric DBSingle to a f64
    pub fn to_double(&self) -> f64 {
        match self {
            Self::Int(i) => *i as f64,
            Self::Long(i) => *i as f64,
            Self::Float(i) => *i as f64,
            Self::Double(i) => *i as f64,
            _ => panic!("Not numeric"),
        }
    }

    pub fn as_dbref(&self) -> DBRef<'_> {
        match self {
            Self::Int(v) => DBRef::Int(v),
            Self::Long(v) => DBRef::Long(v),
            Self::Float(v) => DBRef::Float(v),
            Self::Double(v) => DBRef::Double(v),
            Self::Bool(v) => DBRef::Bool(v),
            Self::Str(v) => DBRef::Str(v),
            Self::DateTime(v) => DBRef::DateTime(v),
            Self::Duration(v) => DBRef::Duration(v),
        }
    }

    //impl_as! {
    //    (as_int, Int, i32),
    //    (as_long, Long, i64),
    //    (as_float, Float, f32),
    //    (as_double, Double, f64),
    //    (as_bool, Bool, bool),
    //    (as_str, Str, String),
    //    (as_datetime, DateTime, DateTime<Utc>),
    //    (as_duration, Duration, Duration),
    //}

    //impl_into! {
    //    (into_int, Int, i32),
    //    (into_long, Long, i64),
    //    (into_float, Float, f32),
    //    (into_double, Double, f64),
    //    (into_bool, Bool, bool),
    //    (into_str, Str, String),
    //    (into_datetime, DateTime, DateTime<Utc>),
    //    (into_duration, Duration, Duration),
    //}
}

//impl Hash for DBVals {
//    fn hash<H>(&self, state: &mut H)
//    where
//        H: Hasher,
//    {
//        match self {
//            Self::Single(single) => single.hash(state),
//            Self::Col(col) => col.hash(state),
//        }
//    }
//}

impl Hash for DBSingle {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            Self::Int(v) => ("Int", v).hash(state),
            Self::Long(v) => ("Long", v).hash(state),
            Self::Bool(v) => ("Bool", v).hash(state),
            Self::Str(v) => ("Str", v).hash(state),
            Self::DateTime(v) => ("DateTime", v).hash(state),
            Self::Duration(v) => ("Duration", v).hash(state),
            // XXX: Allowing floatings to be hashed is kind of messy, but we currently use this
            // only in hashing predicate atoms (i.e., the floating number is given to us as an
            // input), so it should be okay to hash on those. Consider updating the corresponding
            // equality function to only allow equality in these cases.
            Self::Float(v) => ("Float", FloatOrd(*v)).hash(state),
            Self::Double(v) => ("Double", FloatOrd(*v)).hash(state),
        }
    }
}

//impl Hash for DBCol {
//    fn hash<H>(&self, state: &mut H)
//    where
//        H: Hasher,
//    {
//        match self {
//            Self::Int(v) => ("Int", v).hash(state),
//            Self::Long(v) => ("Long", v).hash(state),
//            Self::Bool(v) => ("Bool", v).hash(state),
//            Self::Str(v) => ("Str", v).hash(state),
//            Self::DateTime(v) => ("DateTime", v).hash(state),
//            Self::Duration(v) => ("Duration", v).hash(state),
//            _ => panic!("Did not expect to hash on floating type"),
//        }
//    }
//}

impl Eq for DBCol {}

impl PartialEq for DBSingle {
    fn eq(&self, other: &Self) -> bool {
        if self.data_type().is_integral() && other.data_type().is_integral() {
            self.to_long() == other.to_long()
        } else if self.data_type().is_numeric() && other.data_type().is_numeric() {
            approx::ulps_eq!(self.to_double(), other.to_double(), epsilon = 1e-6)
        } else {
            match (self, other) {
                (Self::Str(x), Self::Str(y)) => x == y,
                (Self::Bool(x), Self::Bool(y)) => x == y,
                (Self::DateTime(x), Self::DateTime(y)) => x == y,
                (Self::Duration(x), Self::Duration(y)) => x == y,
                _ => false,
            }
        }
    }
}
impl Eq for DBSingle {}
impl Ord for DBSingle {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

impl DBRef<'_> {
    pub fn to_owned(self) -> DBSingle {
        match self {
            Self::Int(v) => DBSingle::Int(*v),
            Self::Long(v) => DBSingle::Long(*v),
            Self::Float(v) => DBSingle::Float(*v),
            Self::Double(v) => DBSingle::Double(*v),
            Self::Bool(v) => DBSingle::Bool(*v),
            Self::Str(v) => DBSingle::Str(v.to_string()),
            Self::DateTime(v) => DBSingle::DateTime(*v),
            Self::Duration(v) => DBSingle::Duration(*v),
        }
    }
}

impl From<DBSingle> for DBCol {
    fn from(val: DBSingle) -> Self {
        match val {
            DBSingle::Int(v) => Self::Int(DBColInner::Values(vec![v])),
            DBSingle::Long(v) => Self::Long(DBColInner::Values(vec![v])),
            DBSingle::Float(v) => Self::Float(DBColInner::Values(vec![v])),
            DBSingle::Double(v) => Self::Double(DBColInner::Values(vec![v])),
            DBSingle::Bool(v) => Self::Bool(DBColInner::Values(vec![v])),
            DBSingle::Str(v) => Self::Str(DBColInner::Values(vec![v])),
            DBSingle::DateTime(v) => Self::DateTime(DBColInner::Values(vec![v])),
            DBSingle::Duration(v) => Self::Duration(DBColInner::Values(vec![v])),
        }
    }
}

impl DBType {
    pub fn size(&self) -> usize {
        match self {
            Self::Str => 128,
            Self::Int => mem::size_of::<i32>(),
            Self::Long => mem::size_of::<i64>(),
            Self::Float => mem::size_of::<f32>(),
            Self::Double => mem::size_of::<f64>(),
            Self::Bool => mem::size_of::<u8>(),
            Self::DateTime => mem::size_of::<i64>(),
            Self::Duration => mem::size_of::<i64>(),
        }
    }

    pub fn is_integral(&self) -> bool {
        match self {
            Self::Int | Self::Long => true,
            _ => false,
        }
    }

    pub fn is_numeric(&self) -> bool {
        match self {
            Self::Int | Self::Long | Self::Float | Self::Double => true,
            _ => false,
        }
    }
}

impl fmt::Display for DBSingle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int(v) => write!(f, "{}", v),
            Self::Long(v) => write!(f, "{}", v),
            Self::Float(v) => write!(f, "{}", v),
            Self::Double(v) => write!(f, "{}", v),
            Self::Bool(v) => write!(f, "{}", v),
            Self::Str(v) => write!(f, "{}", v),
            Self::DateTime(v) => write!(f, "{}", v),
            Self::Duration(v) => write!(f, "{}", v),
        }
    }
}

impl AsRef<DBVals> for DBVals {
    fn as_ref(&self) -> &DBVals {
        self
    }
}

impl From<&str> for DBType {
    fn from(s: &str) -> Self {
        match s {
            "string" => Self::Str,
            "int" => Self::Int,
            "long" => Self::Long,
            "float" => Self::Float,
            "double" => Self::Double,
            "boolean" => Self::Bool,
            "timestamp" | "date" => Self::DateTime,
            "interval" => Self::Duration,
            _ => panic!("Unknown type: {}", s),
        }
    }
}

//#[derive(Clone)]
//pub enum LongIter<'a> {
//    Col(LongColIter<'a>),
//    Single(LongSingleIter<'a>),
//}
//#[derive(Clone)]
//pub enum LongColIter<'a> {
//    Int(std::slice::Iter<'a, i32>),
//    Long(std::slice::Iter<'a, i64>),
//}
//#[derive(Clone)]
//pub enum LongSingleIter<'a> {
//    Int(std::iter::Once<&'a i32>),
//    Long(std::iter::Once<&'a i64>),
//}
//
//impl<'a> Iterator for LongIter<'a> {
//    type Item = i64;
//
//    fn next(&mut self) -> Option<Self::Item> {
//        match self {
//            Self::Col(LongColIter::Int(iter)) => iter.next().map(|&v| v as i64),
//            Self::Col(LongColIter::Long(iter)) => iter.next().map(|&v| v),
//            Self::Single(LongSingleIter::Int(iter)) => iter.next().map(|&v| v as i64),
//            Self::Single(LongSingleIter::Long(iter)) => iter.next().map(|&v| v as i64),
//        }
//    }
//}
//
//#[derive(Clone)]
//pub enum DoubleIter<'a> {
//    Col(DoubleColIter<'a>),
//    Single(DoubleSingleIter<'a>),
//}
//#[derive(Clone)]
//pub enum DoubleColIter<'a> {
//    Int(std::slice::Iter<'a, i32>),
//    Long(std::slice::Iter<'a, i64>),
//    Float(std::slice::Iter<'a, f32>),
//    Double(std::slice::Iter<'a, f64>),
//}
//#[derive(Clone)]
//pub enum DoubleSingleIter<'a> {
//    Int(std::iter::Once<&'a i32>),
//    Long(std::iter::Once<&'a i64>),
//    Float(std::iter::Once<&'a f32>),
//    Double(std::iter::Once<&'a f64>),
//}
//
//impl<'a> Iterator for DoubleIter<'a> {
//    type Item = f64;
//
//    fn next(&mut self) -> Option<Self::Item> {
//        match self {
//            Self::Col(DoubleColIter::Int(iter)) => iter.next().map(|&v| v as f64),
//            Self::Col(DoubleColIter::Long(iter)) => iter.next().map(|&v| v as f64),
//            Self::Col(DoubleColIter::Float(iter)) => iter.next().map(|&v| v as f64),
//            Self::Col(DoubleColIter::Double(iter)) => iter.next().map(|&v| v as f64),
//            Self::Single(DoubleSingleIter::Int(iter)) => iter.next().map(|&v| v as f64),
//            Self::Single(DoubleSingleIter::Long(iter)) => iter.next().map(|&v| v as f64),
//            Self::Single(DoubleSingleIter::Float(iter)) => iter.next().map(|&v| v as f64),
//            Self::Single(DoubleSingleIter::Double(iter)) => iter.next().map(|&v| v as f64),
//        }
//    }
//}
//
//#[derive(Clone)]
//pub enum DBIter<'a, T> {
//    Col(std::slice::Iter<'a, T>),
//    Single(std::iter::Once<&'a T>),
//}
//
//impl<'a, T> Iterator for DBIter<'a, T> {
//    type Item = &'a T;
//
//    fn next(&mut self) -> Option<Self::Item> {
//        match self {
//            Self::Col(iter) => iter.next(),
//            Self::Single(v) => v.next(),
//        }
//    }
//}
//
//#[derive(Clone)]
//pub struct StrIter<'a>(DBIter<'a, String>);
//
//impl<'a> Iterator for StrIter<'a> {
//    type Item = &'a str;
//
//    fn next(&mut self) -> Option<Self::Item> {
//        self.0.next().map(|s| s.as_str())
//    }
//}
//
//pub enum DBRefIter<'a> {
//    Col(DBRefColIter<'a>),
//    Single(Option<DBRef<'a>>),
//}
//
//pub enum DBRefColIter<'a> {
//    Int(std::slice::Iter<'a, i32>),
//    Long(std::slice::Iter<'a, i64>),
//    Float(std::slice::Iter<'a, f32>),
//    Double(std::slice::Iter<'a, f64>),
//    Str(std::slice::Iter<'a, String>),
//    Bool(std::slice::Iter<'a, bool>),
//    DateTime(std::slice::Iter<'a, DateTime<Utc>>),
//    Duration(std::slice::Iter<'a, Duration>),
//}
//
//impl<'a> Iterator for DBRefIter<'a> {
//    type Item = DBRef<'a>;
//
//    fn next(&mut self) -> Option<Self::Item> {
//        match self {
//            Self::Col(iter) => iter.next(),
//            Self::Single(v) => v.take(),
//        }
//    }
//}
//
//impl<'a> Iterator for DBRefColIter<'a> {
//    type Item = DBRef<'a>;
//
//    fn next(&mut self) -> Option<Self::Item> {
//        dispatch_db_enum!(self, Self::_(iter) => iter.next() => |v| DBRef::_(v))
//    }
//}
