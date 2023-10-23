use crate::utils;
use priority_queue::PriorityQueue;
use rustc_hash::{FxHashMap, FxHasher};
use std::cmp::Reverse;
use std::hash::BuildHasherDefault;
use std::path::{Path, PathBuf};

/// A simple LRU file cache object, which caches based on the given `FileCol` path and buffer index
/// within the column file.
///
/// WARNING: To make things simple to implement for the first draft, this can be used at most
/// `usize::MAX` times before overflowing (i.e., breaking).
pub struct FileCache {
    bufs: Vec<*mut u8>,
    map: FxHashMap<FileCacheEntry, usize>,
    pq: PriorityQueue<FileCacheEntry, Reverse<usize>, BuildHasherDefault<FxHasher>>,
    times_used: usize,
    buf_size: usize,
    block_size: usize,
    initialized: bool,
}

#[derive(PartialEq, Eq, Hash, Clone)]
struct FileCacheEntry {
    path: PathBuf,
    buf_idx: usize,
}

impl FileCache {
    pub fn new() -> Self {
        Self {
            bufs: vec![],
            map: Default::default(),
            pq: Default::default(),
            times_used: 0,
            buf_size: 0,
            block_size: 0,
            initialized: false,
        }
    }

    pub fn init(&mut self, num_bufs: usize, buf_size: usize, block_size: usize) {
        assert!(!self.initialized);

        self.bufs = (0..num_bufs)
            .map(|_| utils::alloc_aligned_ptr(buf_size, block_size))
            .collect();
        self.map = FxHashMap::with_capacity_and_hasher(num_bufs, BuildHasherDefault::default());
        self.pq = PriorityQueue::with_capacity_and_default_hasher(num_bufs);
        self.times_used = 0;
        self.buf_size = buf_size;
        self.block_size = block_size;
        self.initialized = true;
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Returns the buffer and a boolean flag which is set to true if the buffer was already in the
    /// cache (i.e., it's initialized with the correct values).
    pub fn get(&mut self, path: impl AsRef<Path>, buf_idx: usize) -> (*mut u8, bool) {
        debug_assert!(self.initialized);

        let cache_entry = FileCacheEntry {
            path: path.as_ref().to_owned(),
            buf_idx,
        };
        if let Some(bufs_vec_idx) = self.map.get(&cache_entry) {
            self.pq
                .change_priority(&cache_entry, Reverse(self.times_used));
            self.times_used += 1;
            //let buf = *self.bufs.get(*bufs_vec_idx).unwrap();
            //(buf, true)
            (self.bufs[*bufs_vec_idx], true)
        } else {
            let bufs_vec_idx = if self.map.len() == self.bufs.len() {
                // Cache is full
                let old_entry = self.pq.pop().unwrap().0;
                self.map.remove(&old_entry).unwrap()
            } else {
                // Cache is not full. Take next unused buffer
                self.map.len()
            };

            self.map.insert(cache_entry.clone(), bufs_vec_idx);
            self.pq.push(cache_entry, Reverse(self.times_used));
            self.times_used += 1;
            (self.bufs[bufs_vec_idx], false)
        }
    }

    pub fn clear(&mut self) {
        self.map.clear();
        self.pq.clear();
        self.times_used = 0;
    }
}

impl Drop for FileCache {
    fn drop(&mut self) {
        for ptr in self.bufs.drain(..) {
            utils::dealloc_aligned_ptr(ptr, self.buf_size, self.block_size);
        }
    }
}
