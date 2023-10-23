use crate::bitmap::{Bitmap, BitmapInt, BitmapIter};
use chrono::{DateTime, Duration, NaiveDate, NaiveTime, TimeZone, Utc};
use priority_queue::PriorityQueue;
use std::alloc::{self, Layout};
use std::cmp::Reverse;
use std::slice;

pub fn parse_duration(s: &str) -> Duration {
    let tokens: Vec<&str> = s.split_whitespace().collect();
    if tokens.len() == 1 {
        assert!(tokens[0].contains(":"));
        let mut it = tokens[0].split(":");
        let hour = it
            .next()
            .unwrap()
            .parse::<i64>()
            .expect(&format!("Could not parse hour portion of {}", tokens[0]));
        let min = it
            .next()
            .unwrap()
            .parse::<i64>()
            .expect(&format!("Could not parse minute portion of {}", tokens[0]));

        Duration::minutes(hour * 60 + min)
    } else {
        assert_eq!(tokens.len(), 2);
        let num = tokens[0]
            .parse()
            .expect(&format!("Could not parse {} {}", tokens[0], tokens[1]));
        let unit = tokens[1];
        // XXX For simplicity, 1 year is 52 weeks and 1 month is 30 days
        match unit {
            _ if unit.starts_with("year") => Duration::weeks(num * 52),
            _ if unit.starts_with("mon") => Duration::days(num * 30),
            _ if unit.starts_with("week") => Duration::weeks(num),
            _ if unit.starts_with("day") => Duration::days(num),
            _ if unit.starts_with("hour") => Duration::hours(num),
            _ if unit.starts_with("minute") => Duration::minutes(num),
            _ if unit.starts_with("second") => Duration::seconds(num),
            _ => {
                panic!("Don't support unit {}", unit);
            }
        }
    }
}

pub fn parse_datetime(s: &str) -> DateTime<Utc> {
    s.parse::<DateTime<Utc>>()
        .or_else(|_| Utc.datetime_from_str(s, "%Y-%m-%d %H:%M:%S"))
        .or_else(|_| {
            NaiveDate::parse_from_str(s, "%Y-%m-%d").map(|date| {
                DateTime::from_utc(
                    date.and_time(NaiveTime::from_hms_opt(0, 0, 0).unwrap()),
                    Utc,
                )
            })
        })
        .expect(&format!("Could not parse {} as datetime", s))
}

pub fn alloc_aligned_buf<'a>(buf_size: usize, block_size: usize) -> &'a mut [u8] {
    unsafe {
        slice::from_raw_parts_mut(
            alloc::alloc(
                Layout::from_size_align(buf_size, block_size).expect(&format!(
                    "Error with alignment settings, size: {} align: {}",
                    buf_size, block_size
                )),
            ),
            buf_size,
        )
    }
}

pub fn alloc_aligned_ptr(buf_size: usize, block_size: usize) -> *mut u8 {
    unsafe {
        alloc::alloc(
            Layout::from_size_align(buf_size, block_size).expect(&format!(
                "Error with alignment settings, size: {} align: {}",
                buf_size, block_size
            )),
        )
    }
}

pub fn dealloc_aligned_buf(buf: &mut [u8], buf_size: usize, block_size: usize) {
    unsafe {
        alloc::dealloc(
            buf.as_mut_ptr(),
            Layout::from_size_align(buf_size, block_size).expect(&format!(
                "Error with alignment settings, size: {} align: {}",
                buf_size, block_size
            )),
        );
    }
}

pub fn dealloc_aligned_ptr(ptr: *mut u8, buf_size: usize, block_size: usize) {
    unsafe {
        alloc::dealloc(
            ptr,
            Layout::from_size_align(buf_size, block_size).expect(&format!(
                "Error with alignment settings, size: {} align: {}",
                buf_size, block_size
            )),
        );
    }
}

#[inline(always)]
pub fn slice_to_bytes<T>(vals: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            vals.as_ptr() as *const u8,
            vals.len() * std::mem::size_of::<T>(),
        )
    }
}

#[inline(always)]
pub fn bytes_to_val<T>(bytes: &[u8]) -> &T {
    unsafe { &*(bytes.as_ptr() as *const T) }
}

#[inline(always)]
pub fn bytes_to_slice<T>(bytes: &[u8]) -> &[T] {
    unsafe {
        std::slice::from_raw_parts(
            bytes.as_ptr() as *const T,
            bytes.len() / std::mem::size_of::<T>(),
        )
    }
}

/// Combines two vectors using append and returns the result.
#[inline(always)]
pub fn append<T>(mut v1: Vec<T>, mut v2: Vec<T>) -> Vec<T> {
    v1.append(&mut v2);
    v1
}

pub fn sql_pattern_to_regex(s: &str) -> String {
    let mut ret = String::new();
    for c in s.chars() {
        match c {
            '_' => ret.push('.'),
            '%' => ret.push_str(".*"),
            _ => ret.push(c),
        }
    }
    ret
}

pub enum OneOf<T1, T2, T3, T4> {
    One(T1),
    Two(T2),
    Three(T3),
    Four(T4),
}

impl<T1, T2, T3, T4> OneOf<T1, T2, T3, T4> {
    pub fn into_iter(self) -> OneOf<T1::IntoIter, T2::IntoIter, T3::IntoIter, T4::IntoIter>
    where
        T1: IntoIterator,
        T2: IntoIterator<Item = T1::Item>,
        T3: IntoIterator<Item = T1::Item>,
        T4: IntoIterator<Item = T1::Item>,
    {
        match self {
            Self::One(x) => OneOf::One(x.into_iter()),
            Self::Two(x) => OneOf::Two(x.into_iter()),
            Self::Three(x) => OneOf::Three(x.into_iter()),
            Self::Four(x) => OneOf::Four(x.into_iter()),
        }
    }
}

impl<T1, T2, T3, T4> Iterator for OneOf<T1, T2, T3, T4>
where
    T1: Iterator,
    T2: Iterator<Item = T1::Item>,
    T3: Iterator<Item = T1::Item>,
    T4: Iterator<Item = T1::Item>,
{
    type Item = T1::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::One(x) => x.next(),
            Self::Two(x) => x.next(),
            Self::Three(x) => x.next(),
            Self::Four(x) => x.next(),
        }
    }
}

pub trait IteratorFilterExt: Iterator {
    /// Yields only the values whose positions are in the given `index`.
    fn filter_by_index<T: AsRef<Bitmap>>(self, index: T) -> FilterByIndex<Self, T>
    where
        Self: Sized,
    {
        FilterByIndex {
            index,
            iter: self.enumerate(),
        }
    }

    fn filter_by_index2<'a>(self, index: &'a Bitmap) -> FilterByIndex2<'a, Self>
    where
        Self: Sized,
    {
        FilterByIndex2 {
            index_iter: index.iter(),
            last_seen: None,
            iter: self,
        }
    }
}

impl<I: Iterator> IteratorFilterExt for I {}

pub struct FilterByIndex<I, T: AsRef<Bitmap>> {
    index: T,
    iter: std::iter::Enumerate<I>,
}

impl<I, T> Iterator for FilterByIndex<I, T>
where
    I: Iterator,
    T: AsRef<Bitmap>,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let out = self.iter.next();
            if let Some((i, item)) = out {
                if self.index.as_ref().contains(i as BitmapInt) {
                    return Some(item);
                }
            } else {
                return None;
            }
        }
    }
}

pub struct FilterByIndex2<'a, I> {
    index_iter: BitmapIter<'a>,
    last_seen: Option<BitmapInt>,
    iter: I,
}

impl<I> Iterator for FilterByIndex2<'_, I>
where
    I: Iterator,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.index_iter.next().and_then(|idx| {
            let out = if let Some(last_seen) = self.last_seen {
                self.iter.nth((idx - last_seen - 1) as usize)
            } else {
                self.iter.nth(idx as usize)
            };
            self.last_seen = Some(idx);
            out
        })
    }
}

pub struct AsRefable<'a, T>(pub &'a T);

impl<T> AsRef<T> for AsRefable<'_, T> {
    fn as_ref(&self) -> &T {
        self.0
    }
}

/// Converts a collection type into a single value. Panics if there is not exactly one value.
#[inline]
pub fn convert_to_one<T>(collection: impl IntoIterator<Item = T>) -> T {
    let mut iter = collection.into_iter();
    let out = iter.next().expect("Expect at least one item");
    if iter.next().is_some() {
        panic!("Colleciton has more than one item");
    }
    out
}

pub fn merge_iter_bmaps<'a>(bmaps: &Vec<&'a Bitmap>) -> MergeBitmapIter<'a> {
    let mut pq = PriorityQueue::new();
    let iters: Vec<_> = bmaps
        .iter()
        .enumerate()
        .map(|(i, bmap)| {
            let mut iter = bmap.iter();
            if let Some(idx) = iter.next() {
                pq.push(i, Reverse(idx));
            }
            iter
        })
        .collect();
    MergeBitmapIter { iters, pq }
}

pub struct MergeBitmapIter<'a> {
    iters: Vec<BitmapIter<'a>>,
    pq: PriorityQueue<usize, Reverse<BitmapInt>>,
}

impl Iterator for MergeBitmapIter<'_> {
    type Item = (usize, BitmapInt);

    fn next(&mut self) -> Option<Self::Item> {
        self.pq.pop().map(|(i, idx)| {
            if let Some(new_idx) = self.iters[i].next() {
                self.pq.push(i, Reverse(new_idx));
            }
            (i, idx.0)
        })
    }
}

pub fn merge_iter_bmaps2<'a>(bmaps: &Vec<&'a Bitmap>) -> MergeBitmapIter2<'a> {
    let mut pq = PriorityQueue::new();
    let iters: Vec<_> = bmaps
        .iter()
        .enumerate()
        .map(|(i, bmap)| {
            let mut iter = bmap.iter();
            if let Some(idx) = iter.next() {
                pq.push(i, Reverse(idx));
            }
            iter
        })
        .collect();
    let min_iter_val = pq.pop().map(|(i, idx)| (i, idx.0));
    let second_val = pq.peek().map(|(_, val)| val.0);
    MergeBitmapIter2 {
        iters,
        pq,
        min_iter_val,
        second_val,
    }
}

pub struct MergeBitmapIter2<'a> {
    iters: Vec<BitmapIter<'a>>,
    pq: PriorityQueue<usize, Reverse<BitmapInt>>,
    min_iter_val: Option<(usize, BitmapInt)>,
    second_val: Option<BitmapInt>,
}

impl Iterator for MergeBitmapIter2<'_> {
    type Item = (usize, BitmapInt);

    fn next(&mut self) -> Option<Self::Item> {
        self.min_iter_val.map(|(i, idx)| {
            if let Some(new_idx) = self.iters[i].next() {
                if self
                    .second_val
                    .as_ref()
                    .map(|second_val| new_idx < *second_val)
                    .unwrap_or(true)
                {
                    self.min_iter_val = Some((i, new_idx));
                } else {
                    self.min_iter_val = self.pq.pop().map(|(i, idx)| (i, idx.0));
                    self.pq.push(i, Reverse(new_idx));
                    self.second_val = self.pq.peek().map(|(_, val)| val.0);
                }
            } else {
                self.min_iter_val = self.pq.pop().map(|(i, idx)| (i, idx.0));
                self.second_val = self.pq.peek().map(|(_, val)| val.0);
            }

            (i, idx)
        })
    }
}

/// Returns true if non-empty and if all elements are equal to each other
pub trait IteratorAllEqExt: Iterator {
    fn all_eq<F, T>(self, f: F) -> bool
    where
        F: FnMut(Self::Item) -> T,
        T: Eq;
}

impl<I: Iterator> IteratorAllEqExt for I {
    fn all_eq<F, T>(mut self, mut f: F) -> bool
    where
        F: FnMut(Self::Item) -> T,
        T: Eq,
    {
        if let Some(head) = self.next() {
            let head = f(head);
            self.all(|val| f(val) == head)
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_iter_bmap() {
        let a = Bitmap::from_sorted_iter([0, 3, 6, 7, 9]).unwrap();
        let b = Bitmap::from_sorted_iter([1, 2, 4, 8, 10]).unwrap();

        assert_eq!(
            merge_iter_bmaps(&vec![&a, &b]).collect::<Vec<_>>(),
            vec![
                (0, 0),
                (1, 1),
                (1, 2),
                (0, 3),
                (1, 4),
                (0, 6),
                (0, 7),
                (1, 8),
                (0, 9),
                (1, 10)
            ]
        );
    }
}
