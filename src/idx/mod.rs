use crate::bitmap::BitmapInt;
use crate::file_table::FileTableRef;

mod basic;
mod tagged;
mod utils;

pub use basic::{BasicIdx, IdxRow};
pub use tagged::TaggedIdx;
//use tagged::MultiIdxColIter;
//pub use tagged::{TaggedIdx, TaggedIdxRef};

#[derive(Debug)]
pub enum Idx {
    Basic(BasicIdx),
    Tagged(TaggedIdx),
}

impl Idx {
    pub fn col_iter(&self, table_ref: &FileTableRef) -> IdxColIter<'_> {
        match self {
            Self::Basic(idx) => IdxColIter::Basic(idx.col_iter(table_ref)),
            Self::Tagged(idx) => IdxColIter::Basic(idx.col_iter(table_ref)),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Basic(idx) => idx.len(),
            Self::Tagged(idx) => idx.len(),
        }
    }
}

///// Wrapper around normal and tagged indexes. To be used when evaluating expressions as part of
///// `EvalContext`.
//pub enum Idx<'a> {
//    Basic(BasicIdx),
//    //BasicRef(&'a BasicIdx),
//    Tagged(&'a TaggedIdx),
//    TaggedRef(&'a TaggedIdxRef<'a>),
//}
//
//impl<'a> Idx<'a> {
//    pub fn col_iter(&self, table_ref: &FileTableRef) -> IdxColIter<'a> {
//        match self {
//            Self::Basic(idx) => IdxColIter::Basic(idx.col_iter(table_ref)),
//            Self::Tagged(tagged_idx) => IdxColIter::Multi(tagged_idx.col_iter(table_ref)),
//            Self::TaggedRef(tagged_idx) => IdxColIter::Multi(tagged_idx.col_iter(table_ref)),
//        }
//    }
//
//    pub fn len(&self) -> usize {
//        match self {
//            Self::Basic(idx) => idx.len(),
//            Self::Tagged(tagged_idx) => tagged_idx.len(),
//            Self::TaggedRef(tagged_idx) => tagged_idx.len(),
//        }
//    }
//}

pub enum IdxColIter<'a> {
    Basic(basic::IdxColIter<'a>),
    //Multi(MultiIdxColIter<'a>),
}

impl<'a> Iterator for IdxColIter<'a> {
    type Item = BitmapInt;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Basic(iter) => iter.next(),
            //Self::Multi(iter) => iter.next(),
        }
    }
}
