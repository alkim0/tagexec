use crate::bitmap::{Bitmap, BitmapInt};
use crate::db::{DBCol, DBType};
use crate::engine::{ReadAllType, ReadSomeType, EXEC_INFO};
//use auto_enums::auto_enum;
use crate::utils;
use byteorder::{ByteOrder, NativeEndian};
use chrono::{Duration, TimeZone, Utc};
use itertools::Itertools;
use snowflake::ProcessUniqueId;
//use futures_lite::{stream, StreamExt};
use crate::file_cache::FileCache;
use futures::{future, stream, StreamExt};
use glommio::{
    io::{ImmutableFileBuilder, MergedBufferLimit, ReadAmplificationLimit},
    LocalExecutor,
};
use libc::O_DIRECT;
use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::fs::{self, OpenOptions};
use std::hash::{Hash, Hasher};
use std::io::{self, Read, Seek, SeekFrom};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::rc::{Rc, Weak};
use std::time::Instant;

type TableId = ProcessUniqueId;
type RefId = ProcessUniqueId;

pub type FileTableSet = BTreeSet<Rc<FileTable>>;
pub type FileColSet = BTreeSet<Rc<FileCol>>;
pub type FileTableRefSet = BTreeSet<Rc<FileTableRef>>;

pub struct FileTable {
    id: TableId,
    name: String,
    path: PathBuf,
    cols: HashMap<String, Rc<FileCol>>,
    num_records: usize,
}

pub struct FileCol {
    name: String,
    path: PathBuf,
    data_type: DBType,
    table: RefCell<Option<Weak<FileTable>>>,
}

/// A reference to a `FileTable`. We may need to distinguish between references in the case that a
/// table is present more than one time in the join graph and/or self-joins.
pub struct FileTableRef {
    pub table: Rc<FileTable>,
    pub alias: Option<String>,
    id: RefId,
}

thread_local! {
    static FILE_CACHE: RefCell<FileCache> = RefCell::new(FileCache::new());
}

pub fn init_file_cache(num_bufs: usize, buf_size: usize, block_size: usize) {
    FILE_CACHE.with(|file_cache| {
        let mut file_cache = file_cache.borrow_mut();
        if !file_cache.is_initialized() {
            file_cache.init(num_bufs, buf_size, block_size);
        }
    })
}

pub fn clear_file_cache() {
    FILE_CACHE.with(|file_cache| {
        file_cache.borrow_mut().clear();
    })
}

impl FileTable {
    pub fn new(path: PathBuf) -> io::Result<Rc<Self>> {
        let cols = FileTableSchema::parse(&path)?;
        let num_records = cols.values().next().unwrap().est_num_records()?;
        let name = path
            .file_name()
            .ok_or(io::Error::new(io::ErrorKind::Other, "file name ends in .."))?
            .to_str()
            .ok_or(io::Error::new(io::ErrorKind::Other, "Invalid utf-8"))?
            .to_string();

        let table = Rc::new(Self {
            id: TableId::new(),
            name,
            path,
            cols,
            num_records,
        });

        for col in table.cols.values() {
            *col.table.borrow_mut() = Some(Rc::downgrade(&table));
        }

        Ok(table)
    }

    pub fn id(&self) -> TableId {
        self.id
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn find_col(&self, attr_name: &str) -> Option<&Rc<FileCol>> {
        self.cols.get(attr_name)
    }

    pub fn all_cols(&self) -> Vec<&Rc<FileCol>> {
        self.cols.values().collect()
    }

    pub fn len(&self) -> usize {
        self.num_records
    }
}

impl PartialEq for FileTable {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}
impl Eq for FileTable {}

impl Hash for FileTable {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl PartialOrd for FileTable {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Ord for FileTable {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id.cmp(&other.id)
    }
}

impl FileTableRef {
    pub fn new(table: Rc<FileTable>, alias: Option<String>) -> Self {
        Self {
            table,
            alias,
            id: RefId::new(),
        }
    }
}

impl PartialEq for FileTableRef {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for FileTableRef {}

impl Hash for FileTableRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl PartialOrd for FileTableRef {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Ord for FileTableRef {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id.cmp(&other.id)
    }
}

impl FileCol {
    fn est_num_records(&self) -> io::Result<usize> {
        let col_size = fs::metadata(&self.path)?.len();
        Ok(col_size as usize / self.data_type.size())
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the fully qualified name of the column including the `FileTable` name.
    pub fn full_name(&self) -> String {
        format!("{}.{}", self.table().name(), self.name())
    }

    pub fn table(&self) -> Rc<FileTable> {
        self.table
            .borrow()
            .as_ref()
            .expect("Didn't instantiate FileCol correctly")
            .upgrade()
            .expect("Could not upgrade to Rc<FileTable>")
    }

    pub fn data_type(&self) -> DBType {
        self.data_type
    }

    // TODO: Ideally, we would schedule file reads asynchronously and evaluate read results as
    // they're finished, but let's just make this simple for now.
    pub fn read(&self, idxs: impl IntoIterator<Item = BitmapInt>) -> DBCol {
        let mut output_idx = vec![];
        let mut bmap = Bitmap::new();
        let mut prev_idx = None;
        let mut in_order = true;
        for idx in idxs.into_iter() {
            bmap.insert(idx);
            output_idx.push(idx as usize);

            match prev_idx {
                Some(prev_idx) if in_order => {
                    if prev_idx > idx {
                        in_order = false;
                    }
                }
                _ => {}
            }
            prev_idx = Some(idx);
        }

        let output_idx = if in_order && bmap.len() as usize == output_idx.len() {
            None
        } else {
            Some(output_idx)
        };
        let output_idx = output_idx.map(|output_idx| {
            let rev_map: FxHashMap<usize, usize> = bmap
                .iter()
                .enumerate()
                .map(|(i, idx)| (idx as usize, i))
                .collect();
            output_idx
                .into_iter()
                .map(|idx| *rev_map.get(&idx).unwrap())
                .collect()
        });

        let selectivity = bmap.len() as f64 / self.table().num_records as f64;
        let (threshold, read_some_type, read_all_type) = EXEC_INFO.with(|exec_info| {
            let exec_info = exec_info.borrow();
            (
                exec_info.selectivity_threshold,
                exec_info.read_some_type,
                exec_info.read_all_type,
            )
        });
        let col = if selectivity < threshold {
            match read_some_type {
                ReadSomeType::Glommio => self.read_some(bmap, output_idx),
                ReadSomeType::Mmap => self.read_some_mmap(bmap, output_idx),
                ReadSomeType::DirectIO => self.read_some_direct_io(bmap, output_idx),
                ReadSomeType::CachedDirectIO => self.read_some_cached_direct_io(bmap, output_idx),
            }
        } else {
            match read_all_type {
                ReadAllType::Glommio => self.read_all(bmap, output_idx),
                ReadAllType::Simple => self.read_all_simple(bmap, output_idx),
                ReadAllType::Mmap => self.read_all_mmap(bmap, output_idx),
                ReadAllType::DirectIO => self.read_all_direct_io(bmap, output_idx),
                ReadAllType::CachedDirectIO => self.read_all_cached_direct_io(bmap, output_idx),
            }
        };
        col
    }

    fn read_some(&self, bmap: Bitmap, output_idx: Option<Vec<usize>>) -> DBCol {
        let now = Instant::now();
        let data_size = self.data_type.size();
        let mut col = DBCol::with_capacity(&self.data_type, bmap.len() as usize);

        let buf_size = EXEC_INFO.with(|exec_info| exec_info.borrow().buf_size);

        let ex = LocalExecutor::default();
        ex.run(async {
            let file = ImmutableFileBuilder::new(&self.path)
                .with_buffer_size(buf_size)
                .build_existing()
                .await
                .expect(format!("Cannot open {} for reading", self.path.display()).as_str());

            file.read_many(
                stream::iter(bmap).map(|idx| (idx * data_size as u64, data_size)),
                MergedBufferLimit::DeviceMaxSingleRequest,
                ReadAmplificationLimit::NoAmplification,
            )
            .for_each(|result| {
                let (_, result) = result.expect("Could not read");
                match &mut col {
                    DBCol::Int(col) => col.push(NativeEndian::read_i32(&*result)),
                    DBCol::Long(col) => col.push(NativeEndian::read_i64(&*result)),
                    DBCol::Float(col) => col.push(NativeEndian::read_f32(&*result)),
                    DBCol::Double(col) => col.push(NativeEndian::read_f64(&*result)),
                    DBCol::Bool(col) => col.push(unsafe { result.get_unchecked(0) } > &0),
                    DBCol::Str(col) => {
                        let str_len =
                            NativeEndian::read_u32(unsafe { result.get_unchecked(..4) }) as usize;
                        let mut str_buf = vec![0 as u8; str_len];
                        str_buf.copy_from_slice(unsafe { result.get_unchecked(4..(4 + str_len)) });
                        col.push(unsafe { String::from_utf8_unchecked(str_buf) })
                    }
                    DBCol::DateTime(col) => col.push(
                        Utc.timestamp_opt(NativeEndian::read_i64(&*result), 0)
                            .unwrap(),
                    ),
                    DBCol::Duration(col) => {
                        col.push(Duration::seconds(NativeEndian::read_i64(&*result)))
                    }
                }
                future::ready(())
            })
            .await;

            file.close().await.unwrap();
        });
        let read_time_ms = now.elapsed().as_millis();

        EXEC_INFO.with(|exec_info| {
            let mut exec_info = exec_info.borrow_mut();
            exec_info.stats.read_time_ms += read_time_ms;
            exec_info.stats.num_elems_read += col.len() as u128;
        });

        match output_idx {
            None => col,
            Some(output_idx) => col.project(output_idx),
        }
    }

    fn read_some_mmap(&self, bmap: Bitmap, output_idx: Option<Vec<usize>>) -> DBCol {
        let now = Instant::now();
        let data_size = self.data_type.size();
        let mut col = DBCol::with_capacity(&self.data_type, bmap.len() as usize);

        let file = std::fs::File::open(&self.path).unwrap();
        let buf = unsafe { memmap2::Mmap::map(&file).unwrap() };

        match &mut col {
            DBCol::Int(col) => col.extend(bmap.iter().map(|i| {
                let i = i as usize;
                NativeEndian::read_i32(unsafe {
                    buf.get_unchecked((i * data_size)..((i + 1) * data_size))
                })
            })),
            DBCol::Long(col) => col.extend(bmap.iter().map(|i| {
                let i = i as usize;
                NativeEndian::read_i64(unsafe {
                    buf.get_unchecked((i * data_size)..((i + 1) * data_size))
                })
            })),
            DBCol::Float(col) => col.extend(bmap.iter().map(|i| {
                let i = i as usize;
                NativeEndian::read_f32(unsafe {
                    buf.get_unchecked((i * data_size)..((i + 1) * data_size))
                })
            })),
            DBCol::Double(col) => col.extend(bmap.iter().map(|i| {
                let i = i as usize;
                NativeEndian::read_f64(unsafe {
                    buf.get_unchecked((i * data_size)..((i + 1) * data_size))
                })
            })),
            DBCol::Bool(col) => col.extend(bmap.iter().map(|i| {
                let i = i as usize;
                (unsafe { buf.get_unchecked(i) } > &0)
            })),
            DBCol::Str(col) => col.extend(bmap.iter().map(|i| {
                let i = i as usize;
                let str_len = NativeEndian::read_u32(unsafe {
                    buf.get_unchecked((i * data_size)..(i * data_size + 4))
                }) as usize;
                let mut str_buf = vec![0 as u8; str_len];
                str_buf.copy_from_slice(unsafe {
                    buf.get_unchecked((i * data_size + 4)..(i * data_size + 4 + str_len))
                });
                unsafe { String::from_utf8_unchecked(str_buf) }
            })),
            DBCol::DateTime(col) => col.extend(bmap.iter().map(|i| {
                let i = i as usize;
                Utc.timestamp_opt(
                    NativeEndian::read_i64(unsafe {
                        buf.get_unchecked((i * data_size)..((i + 1) * data_size))
                    }),
                    0,
                )
                .unwrap()
            })),
            DBCol::Duration(col) => col.extend(bmap.iter().map(|i| {
                let i = i as usize;
                Duration::seconds(NativeEndian::read_i64(unsafe {
                    buf.get_unchecked((i * data_size)..((i + 1) * data_size))
                }))
            })),
        }
        let read_time_ms = now.elapsed().as_millis();

        EXEC_INFO.with(|exec_info| {
            let mut exec_info = exec_info.borrow_mut();
            exec_info.stats.read_time_ms += read_time_ms;
            exec_info.stats.num_elems_read += col.len() as u128;
        });

        match output_idx {
            None => col,
            Some(output_idx) => col.project(output_idx),
        }
    }

    fn read_some_direct_io(&self, bmap: Bitmap, output_idx: Option<Vec<usize>>) -> DBCol {
        let now = Instant::now();
        let data_size = self.data_type.size();
        let num_records = self.table().num_records;

        let (buf_size, block_size) = EXEC_INFO.with(|exec_info| {
            let exec_info = exec_info.borrow();
            (exec_info.buf_size, exec_info.block_size)
        });

        let mut col = DBCol::with_capacity(&self.data_type, num_records);

        let buf = utils::alloc_aligned_buf(buf_size, block_size);
        let last_buf_is_not_full = (data_size * num_records) % buf_size > 0;
        let num_bufs =
            (data_size * num_records) / buf_size + if last_buf_is_not_full { 1 } else { 0 };

        let mut file = OpenOptions::new()
            .read(true)
            .custom_flags(O_DIRECT)
            .open(&self.path)
            .unwrap();

        let mut cur_buf_idx = None;
        for i in &bmap {
            let i = i as usize;
            let buf_idx = i * data_size / buf_size;

            match &cur_buf_idx {
                Some(cur_buf_idx) if cur_buf_idx == &buf_idx => {}
                _ => {
                    assert_eq!(
                        file.seek(SeekFrom::Start((buf_idx * buf_size) as u64))
                            .unwrap(),
                        (buf_idx * buf_size) as u64
                    );
                    if buf_idx == num_bufs - 1 {
                        // Last buf, may not be full
                        let read_size = (data_size * num_records) % buf_size;
                        file.read_exact(&mut buf[..read_size]).unwrap();
                    } else {
                        file.read_exact(buf).unwrap();
                    }
                    cur_buf_idx = Some(buf_idx);
                }
            }

            let offset = i * data_size % buf_size;
            match &mut col {
                DBCol::Int(col) => col.push(NativeEndian::read_i32(unsafe {
                    buf.get_unchecked(offset..(offset + data_size))
                })),
                DBCol::Long(col) => col.push(NativeEndian::read_i64(unsafe {
                    buf.get_unchecked(offset..(offset + data_size))
                })),
                DBCol::Float(col) => col.push(NativeEndian::read_f32(unsafe {
                    buf.get_unchecked(offset..(offset + data_size))
                })),
                DBCol::Double(col) => col.push(NativeEndian::read_f64(unsafe {
                    buf.get_unchecked(offset..(offset + data_size))
                })),
                DBCol::Bool(col) => col.push(unsafe { buf.get_unchecked(offset) } > &0),
                DBCol::Str(col) => {
                    let str_len =
                        NativeEndian::read_u32(unsafe { buf.get_unchecked(offset..(offset + 4)) })
                            as usize;
                    let mut str_buf = vec![0 as u8; str_len];
                    str_buf.copy_from_slice(unsafe {
                        buf.get_unchecked((offset + 4)..(offset + 4 + str_len))
                    });
                    col.push(unsafe { String::from_utf8_unchecked(str_buf) })
                }
                DBCol::DateTime(col) => col.push(
                    Utc.timestamp_opt(
                        NativeEndian::read_i64(unsafe {
                            buf.get_unchecked(offset..(offset + data_size))
                        }),
                        0,
                    )
                    .unwrap(),
                ),
                DBCol::Duration(col) => {
                    col.push(Duration::seconds(NativeEndian::read_i64(unsafe {
                        buf.get_unchecked(offset..(offset + data_size))
                    })))
                }
            }
        }

        let read_time_ms = now.elapsed().as_millis();

        EXEC_INFO.with(|exec_info| {
            let mut exec_info = exec_info.borrow_mut();
            exec_info.stats.read_time_ms += read_time_ms;
            exec_info.stats.num_elems_read += col.len() as u128;
        });

        utils::dealloc_aligned_buf(buf, buf_size, block_size);

        match output_idx {
            None => col,
            Some(output_idx) => col.project(output_idx),
        }
    }

    fn read_some_cached_direct_io(&self, bmap: Bitmap, output_idx: Option<Vec<usize>>) -> DBCol {
        let now = Instant::now();
        let data_size = self.data_type.size();
        let num_records = self.table().num_records;

        let buf_size = EXEC_INFO.with(|exec_info| exec_info.borrow().file_cache_buf_size);

        let mut col = DBCol::with_capacity(&self.data_type, num_records);

        let last_buf_is_not_full = (data_size * num_records) % buf_size > 0;
        let num_bufs =
            (data_size * num_records) / buf_size + if last_buf_is_not_full { 1 } else { 0 };

        let mut file = None;
        let mut buf = None;
        let mut cur_buf_idx = None;
        let mut file_buf_idx = 0;
        let mut num_elems_read = 0;

        for i in &bmap {
            let i = i as usize;
            let buf_idx = i * data_size / buf_size;

            match &cur_buf_idx {
                Some(cur_buf_idx) if cur_buf_idx == &buf_idx => {}
                _ => {
                    let (cached_buf, valid) = FILE_CACHE
                        .with(|file_cache| file_cache.borrow_mut().get(&self.path, buf_idx));
                    let cached_buf =
                        unsafe { std::slice::from_raw_parts_mut(cached_buf, buf_size) };

                    if !valid {
                        let file = file.get_or_insert_with(|| {
                            OpenOptions::new()
                                .read(true)
                                .custom_flags(O_DIRECT)
                                .open(&self.path)
                                .unwrap()
                        });

                        if file_buf_idx != buf_idx {
                            assert_eq!(
                                file.seek(SeekFrom::Start((buf_idx * buf_size) as u64))
                                    .unwrap(),
                                (buf_idx * buf_size) as u64
                            );
                        }

                        if buf_idx == num_bufs - 1 {
                            // Last buf, may not be full
                            let read_size = (data_size * num_records) % buf_size;
                            //file.read_exact(&mut cached_buf[..read_size]).unwrap();
                            assert_eq!(read_size, file.read(cached_buf).unwrap());
                            num_elems_read += read_size / data_size;
                        } else {
                            file.read_exact(cached_buf).unwrap();
                            num_elems_read += buf_size / data_size;
                        }
                        file_buf_idx = buf_idx + 1;
                    }
                    cur_buf_idx = Some(buf_idx);
                    buf = Some(cached_buf);
                }
            }

            let buf = buf.as_ref().unwrap();

            let offset = i * data_size % buf_size;
            match &mut col {
                DBCol::Int(col) => col.push(NativeEndian::read_i32(unsafe {
                    buf.get_unchecked(offset..(offset + data_size))
                })),
                DBCol::Long(col) => col.push(NativeEndian::read_i64(unsafe {
                    buf.get_unchecked(offset..(offset + data_size))
                })),
                DBCol::Float(col) => col.push(NativeEndian::read_f32(unsafe {
                    buf.get_unchecked(offset..(offset + data_size))
                })),
                DBCol::Double(col) => col.push(NativeEndian::read_f64(unsafe {
                    buf.get_unchecked(offset..(offset + data_size))
                })),
                DBCol::Bool(col) => col.push(unsafe { buf.get_unchecked(offset) } > &0),
                DBCol::Str(col) => {
                    let str_len =
                        NativeEndian::read_u32(unsafe { buf.get_unchecked(offset..(offset + 4)) })
                            as usize;
                    let mut str_buf = vec![0 as u8; str_len];
                    str_buf.copy_from_slice(unsafe {
                        buf.get_unchecked((offset + 4)..(offset + 4 + str_len))
                    });
                    col.push(unsafe { String::from_utf8_unchecked(str_buf) })
                }
                DBCol::DateTime(col) => col.push(
                    Utc.timestamp_opt(
                        NativeEndian::read_i64(unsafe {
                            buf.get_unchecked(offset..(offset + data_size))
                        }),
                        0,
                    )
                    .unwrap(),
                ),
                DBCol::Duration(col) => {
                    col.push(Duration::seconds(NativeEndian::read_i64(unsafe {
                        buf.get_unchecked(offset..(offset + data_size))
                    })))
                }
            }
        }

        let read_time_ms = now.elapsed().as_millis();

        EXEC_INFO.with(|exec_info| {
            let mut exec_info = exec_info.borrow_mut();
            exec_info.stats.read_time_ms += read_time_ms;
            exec_info.stats.num_elems_read += num_elems_read as u128;
        });

        match output_idx {
            None => col,
            Some(output_idx) => col.project(output_idx),
        }
    }

    fn read_all(&self, bmap: Bitmap, output_idx: Option<Vec<usize>>) -> DBCol {
        let now = Instant::now();
        let data_size = self.data_type.size();
        let num_records = self.table().num_records;

        let mut col = DBCol::with_capacity(&self.data_type, num_records);

        let read_all_buf_size = EXEC_INFO.with(|exec_info| exec_info.borrow().read_all_buf_size);

        let ex = LocalExecutor::default();
        ex.run(async {
            let file = ImmutableFileBuilder::new(&self.path)
                .with_buffer_size(read_all_buf_size)
                .build_existing()
                .await
                .expect(format!("Cannot open {} for reading", self.path.display()).as_str());

            let mut stream_reader = file
                .stream_reader()
                .with_buffer_size(read_all_buf_size)
                .build();

            let mut num_read = 0;
            loop {
                let result = stream_reader
                    .get_buffer_aligned(read_all_buf_size as u64)
                    .await
                    .unwrap();

                if result.len() == 0 {
                    break;
                }

                let num_data = result.len() / data_size;
                match &mut col {
                    DBCol::Int(col) => {
                        col.extend(
                            (0..num_data)
                                .filter(|i| bmap.contains((*i + num_read) as u64))
                                .map(|i| {
                                    //let slice = ReadResult::slice(&result, i * data_size, data_size).unwrap();
                                    //NativeEndian::read_i32(&*slice)
                                    NativeEndian::read_i32(unsafe {
                                        result.get_unchecked((i * data_size)..((i + 1) * data_size))
                                    })
                                }),
                        )
                    }
                    DBCol::Long(col) => col.extend(
                        (0..num_data)
                            .filter(|i| bmap.contains((*i + num_read) as u64))
                            .map(|i| {
                                NativeEndian::read_i64(unsafe {
                                    result.get_unchecked((i * data_size)..((i + 1) * data_size))
                                })
                            }),
                    ),
                    DBCol::Float(col) => col.extend(
                        (0..num_data)
                            .filter(|i| bmap.contains((*i + num_read) as u64))
                            .map(|i| {
                                NativeEndian::read_f32(unsafe {
                                    result.get_unchecked((i * data_size)..((i + 1) * data_size))
                                })
                            }),
                    ),
                    DBCol::Double(col) => col.extend(
                        (0..num_data)
                            .filter(|i| bmap.contains((*i + num_read) as u64))
                            .map(|i| {
                                NativeEndian::read_f64(unsafe {
                                    result.get_unchecked((i * data_size)..((i + 1) * data_size))
                                })
                            }),
                    ),
                    DBCol::Bool(col) => col.extend(
                        (0..num_data)
                            .filter(|i| bmap.contains((*i + num_read) as u64))
                            .map(|i| result[i] > 0),
                    ),
                    DBCol::Str(col) => col.extend(
                        (0..num_data)
                            .filter(|i| bmap.contains((*i + num_read) as u64))
                            .map(|i| {
                                let str_len = NativeEndian::read_u32(unsafe {
                                    result.get_unchecked((i * data_size)..(i * data_size + 4))
                                }) as usize;
                                let mut str_buf = vec![0 as u8; str_len];
                                str_buf.copy_from_slice(unsafe {
                                    result.get_unchecked(
                                        (i * data_size + 4)..(i * data_size + 4 + str_len),
                                    )
                                });
                                unsafe { String::from_utf8_unchecked(str_buf) }
                            }),
                    ),
                    DBCol::DateTime(col) => col.extend(
                        (0..num_data)
                            .filter(|i| bmap.contains((*i + num_read) as u64))
                            .map(|i| {
                                Utc.timestamp_opt(
                                    NativeEndian::read_i64(unsafe {
                                        result.get_unchecked((i * data_size)..((i + 1) * data_size))
                                    }),
                                    0,
                                )
                                .unwrap()
                            }),
                    ),
                    DBCol::Duration(col) => col.extend(
                        (0..num_data)
                            .filter(|i| bmap.contains((*i + num_read) as u64))
                            .map(|i| {
                                Duration::seconds(NativeEndian::read_i64(unsafe {
                                    result.get_unchecked((i * data_size)..((i + 1) * data_size))
                                }))
                            }),
                    ),
                }

                num_read += num_data;
            }

            stream_reader.close().await.unwrap();
            file.close().await.unwrap();
        });
        let read_time_ms = now.elapsed().as_millis();

        EXEC_INFO.with(|exec_info| {
            let mut exec_info = exec_info.borrow_mut();
            exec_info.stats.read_time_ms += read_time_ms;
            exec_info.stats.num_elems_read += col.len() as u128;
        });

        match output_idx {
            None => col,
            Some(output_idx) => col.project(output_idx),
        }
    }

    fn read_all_simple(&self, bmap: Bitmap, output_idx: Option<Vec<usize>>) -> DBCol {
        let now = Instant::now();
        let data_size = self.data_type.size();
        let num_records = self.table().num_records;

        let mut col = DBCol::with_capacity(&self.data_type, num_records);

        let mut buf = vec![];
        let mut file = std::fs::File::open(&self.path).unwrap();
        file.read_to_end(&mut buf).unwrap();

        // TODO: We could try diving up the buf into smaller vecs and passing them directly to the
        // string so we don't have to do copies in that case.

        match &mut col {
            DBCol::Int(col) => col.extend(bmap.iter().map(|i| {
                let i = i as usize;
                NativeEndian::read_i32(unsafe {
                    buf.get_unchecked((i * data_size)..((i + 1) * data_size))
                })
            })),
            DBCol::Long(col) => col.extend(bmap.iter().map(|i| {
                let i = i as usize;
                NativeEndian::read_i64(unsafe {
                    buf.get_unchecked((i * data_size)..((i + 1) * data_size))
                })
            })),
            DBCol::Float(col) => col.extend(bmap.iter().map(|i| {
                let i = i as usize;
                NativeEndian::read_f32(unsafe {
                    buf.get_unchecked((i * data_size)..((i + 1) * data_size))
                })
            })),
            DBCol::Double(col) => col.extend(bmap.iter().map(|i| {
                let i = i as usize;
                NativeEndian::read_f64(unsafe {
                    buf.get_unchecked((i * data_size)..((i + 1) * data_size))
                })
            })),
            DBCol::Bool(col) => col.extend(bmap.iter().map(|i| {
                let i = i as usize;
                (unsafe { buf.get_unchecked(i) } > &0)
            })),
            DBCol::Str(col) => col.extend(bmap.iter().map(|i| {
                let i = i as usize;
                let str_len = NativeEndian::read_u32(unsafe {
                    buf.get_unchecked((i * data_size)..(i * data_size + 4))
                }) as usize;
                let mut str_buf = vec![0 as u8; str_len];
                str_buf.copy_from_slice(unsafe {
                    buf.get_unchecked((i * data_size + 4)..(i * data_size + 4 + str_len))
                });
                unsafe { String::from_utf8_unchecked(str_buf) }
            })),
            DBCol::DateTime(col) => col.extend(bmap.iter().map(|i| {
                let i = i as usize;
                Utc.timestamp_opt(
                    NativeEndian::read_i64(unsafe {
                        buf.get_unchecked((i * data_size)..((i + 1) * data_size))
                    }),
                    0,
                )
                .unwrap()
            })),
            DBCol::Duration(col) => col.extend(bmap.iter().map(|i| {
                let i = i as usize;
                Duration::seconds(NativeEndian::read_i64(unsafe {
                    buf.get_unchecked((i * data_size)..((i + 1) * data_size))
                }))
            })),
        };
        let read_time_ms = now.elapsed().as_millis();

        EXEC_INFO.with(|exec_info| {
            let mut exec_info = exec_info.borrow_mut();
            exec_info.stats.read_time_ms += read_time_ms;
            exec_info.stats.num_elems_read += col.len() as u128;
        });

        match output_idx {
            None => col,
            Some(output_idx) => col.project(output_idx),
        }
    }

    fn read_all_mmap(&self, bmap: Bitmap, output_idx: Option<Vec<usize>>) -> DBCol {
        let now = Instant::now();
        let data_size = self.data_type.size();
        let num_records = self.table().num_records;

        let mut col = DBCol::with_capacity(&self.data_type, num_records);

        let file = std::fs::File::open(&self.path).unwrap();
        let mut mmap_options = memmap2::MmapOptions::new();
        let buf = unsafe { mmap_options.populate().map(&file).unwrap() };

        // TODO: We could try diving up the buf into smaller vecs and passing them directly to the
        // string so we don't have to do copies in that case.

        match &mut col {
            DBCol::Int(col) => col.extend(bmap.iter().map(|i| {
                let i = i as usize;
                NativeEndian::read_i32(unsafe {
                    buf.get_unchecked((i * data_size)..((i + 1) * data_size))
                })
            })),
            DBCol::Long(col) => col.extend(bmap.iter().map(|i| {
                let i = i as usize;
                NativeEndian::read_i64(unsafe {
                    buf.get_unchecked((i * data_size)..((i + 1) * data_size))
                })
            })),
            DBCol::Float(col) => col.extend(bmap.iter().map(|i| {
                let i = i as usize;
                NativeEndian::read_f32(unsafe {
                    buf.get_unchecked((i * data_size)..((i + 1) * data_size))
                })
            })),
            DBCol::Double(col) => col.extend(bmap.iter().map(|i| {
                let i = i as usize;
                NativeEndian::read_f64(unsafe {
                    buf.get_unchecked((i * data_size)..((i + 1) * data_size))
                })
            })),
            DBCol::Bool(col) => col.extend(bmap.iter().map(|i| {
                let i = i as usize;
                (unsafe { buf.get_unchecked(i) } > &0)
            })),
            DBCol::Str(col) => col.extend(bmap.iter().map(|i| {
                let i = i as usize;
                let str_len = NativeEndian::read_u32(unsafe {
                    buf.get_unchecked((i * data_size)..(i * data_size + 4))
                }) as usize;
                let mut str_buf = vec![0 as u8; str_len];
                str_buf.copy_from_slice(unsafe {
                    buf.get_unchecked((i * data_size + 4)..(i * data_size + 4 + str_len))
                });
                unsafe { String::from_utf8_unchecked(str_buf) }
            })),
            DBCol::DateTime(col) => col.extend(bmap.iter().map(|i| {
                let i = i as usize;
                Utc.timestamp_opt(
                    NativeEndian::read_i64(unsafe {
                        buf.get_unchecked((i * data_size)..((i + 1) * data_size))
                    }),
                    0,
                )
                .unwrap()
            })),
            DBCol::Duration(col) => col.extend(bmap.iter().map(|i| {
                let i = i as usize;
                Duration::seconds(NativeEndian::read_i64(unsafe {
                    buf.get_unchecked((i * data_size)..((i + 1) * data_size))
                }))
            })),
        };
        let read_time_ms = now.elapsed().as_millis();

        EXEC_INFO.with(|exec_info| {
            let mut exec_info = exec_info.borrow_mut();
            exec_info.stats.read_time_ms += read_time_ms;
            exec_info.stats.num_elems_read += col.len() as u128;
        });

        match output_idx {
            None => col,
            Some(output_idx) => col.project(output_idx),
        }
    }

    fn read_all_direct_io(&self, bmap: Bitmap, output_idx: Option<Vec<usize>>) -> DBCol {
        let now = Instant::now();
        let data_size = self.data_type.size();
        let num_records = self.table().num_records;

        let (read_all_buf_size, block_size) = EXEC_INFO.with(|exec_info| {
            let exec_info = exec_info.borrow();
            (exec_info.read_all_buf_size, exec_info.block_size)
        });

        let mut col = DBCol::with_capacity(&self.data_type, num_records);

        let buf = utils::alloc_aligned_buf(read_all_buf_size, block_size);
        let last_buf_is_not_full = (data_size * num_records) % read_all_buf_size > 0;
        let num_bufs = (data_size * num_records) / read_all_buf_size
            + if last_buf_is_not_full { 1 } else { 0 };

        let mut file = OpenOptions::new()
            .read(true)
            .custom_flags(O_DIRECT)
            .open(&self.path)
            .unwrap();

        let mut total_read = 0;
        for i in 0..num_bufs {
            let num_read = if last_buf_is_not_full && i == num_bufs - 1 {
                let read_size = (data_size * num_records) % read_all_buf_size;
                file.read_exact(&mut buf[..read_size]).unwrap();
                read_size
            } else {
                file.read_exact(buf).unwrap();
                read_all_buf_size
            };

            let num_data = num_read / data_size;
            match &mut col {
                DBCol::Int(col) => col.extend(
                    (0..num_data)
                        .filter(|i| bmap.contains((*i + total_read) as u64))
                        .map(|i| {
                            //let slice = ReadResult::slice(&buf, i * data_size, data_size).unwrap();
                            //NativeEndian::read_i32(&*slice)
                            NativeEndian::read_i32(unsafe {
                                buf.get_unchecked((i * data_size)..((i + 1) * data_size))
                            })
                        }),
                ),
                DBCol::Long(col) => col.extend(
                    (0..num_data)
                        .filter(|i| bmap.contains((*i + total_read) as u64))
                        .map(|i| {
                            NativeEndian::read_i64(unsafe {
                                buf.get_unchecked((i * data_size)..((i + 1) * data_size))
                            })
                        }),
                ),
                DBCol::Float(col) => col.extend(
                    (0..num_data)
                        .filter(|i| bmap.contains((*i + total_read) as u64))
                        .map(|i| {
                            NativeEndian::read_f32(unsafe {
                                buf.get_unchecked((i * data_size)..((i + 1) * data_size))
                            })
                        }),
                ),
                DBCol::Double(col) => col.extend(
                    (0..num_data)
                        .filter(|i| bmap.contains((*i + total_read) as u64))
                        .map(|i| {
                            NativeEndian::read_f64(unsafe {
                                buf.get_unchecked((i * data_size)..((i + 1) * data_size))
                            })
                        }),
                ),
                DBCol::Bool(col) => col.extend(
                    (0..num_data)
                        .filter(|i| bmap.contains((*i + total_read) as u64))
                        .map(|i| (unsafe { buf.get_unchecked(i) } > &0)),
                ),
                DBCol::Str(col) => col.extend(
                    (0..num_data)
                        .filter(|i| bmap.contains((*i + total_read) as u64))
                        .map(|i| {
                            let str_len = NativeEndian::read_u32(unsafe {
                                buf.get_unchecked((i * data_size)..(i * data_size + 4))
                            }) as usize;
                            let mut str_buf = vec![0 as u8; str_len];
                            str_buf.copy_from_slice(unsafe {
                                buf.get_unchecked(
                                    (i * data_size + 4)..(i * data_size + 4 + str_len),
                                )
                            });
                            unsafe { String::from_utf8_unchecked(str_buf) }
                        }),
                ),
                DBCol::DateTime(col) => col.extend(
                    (0..num_data)
                        .filter(|i| bmap.contains((*i + total_read) as u64))
                        .map(|i| {
                            Utc.timestamp_opt(
                                NativeEndian::read_i64(unsafe {
                                    buf.get_unchecked((i * data_size)..((i + 1) * data_size))
                                }),
                                0,
                            )
                            .unwrap()
                        }),
                ),
                DBCol::Duration(col) => col.extend(
                    (0..num_data)
                        .filter(|i| bmap.contains((*i + total_read) as u64))
                        .map(|i| {
                            Duration::seconds(NativeEndian::read_i64(unsafe {
                                buf.get_unchecked((i * data_size)..((i + 1) * data_size))
                            }))
                        }),
                ),
            }

            total_read += num_data;
        }
        let read_time_ms = now.elapsed().as_millis();

        EXEC_INFO.with(|exec_info| {
            let mut exec_info = exec_info.borrow_mut();
            exec_info.stats.read_time_ms += read_time_ms;
            exec_info.stats.num_elems_read += col.len() as u128;
        });

        utils::dealloc_aligned_buf(buf, read_all_buf_size, block_size);

        match output_idx {
            None => col,
            Some(output_idx) => col.project(output_idx),
        }
    }

    fn read_all_cached_direct_io(&self, bmap: Bitmap, output_idx: Option<Vec<usize>>) -> DBCol {
        let now = Instant::now();
        let data_size = self.data_type.size();
        let num_records = self.table().num_records;

        let (buf_size, block_size) = EXEC_INFO.with(|exec_info| {
            let exec_info = exec_info.borrow();
            (exec_info.file_cache_buf_size, exec_info.block_size)
        });

        let mut col = DBCol::with_capacity(&self.data_type, num_records);

        let last_buf_is_not_full = (data_size * num_records) % buf_size > 0;
        let num_bufs =
            (data_size * num_records) / buf_size + if last_buf_is_not_full { 1 } else { 0 };

        let mut file = None;
        let mut total_num_data = 0;
        let mut num_elems_read = 0;
        let mut file_buf_idx = 0;
        for i in 0..num_bufs {
            let (buf, valid) =
                FILE_CACHE.with(|file_cache| file_cache.borrow_mut().get(&self.path, i));
            let buf = unsafe { std::slice::from_raw_parts_mut(buf, buf_size) };

            let read_size = if last_buf_is_not_full && i == num_bufs - 1 {
                (data_size * num_records) % buf_size
            } else {
                buf_size
            };
            let num_data = read_size / data_size;

            if !valid {
                let file = file.get_or_insert_with(|| {
                    OpenOptions::new()
                        .read(true)
                        .custom_flags(O_DIRECT)
                        .open(&self.path)
                        .unwrap()
                });

                if file_buf_idx != i {
                    assert_eq!(
                        file.seek(SeekFrom::Start((i * buf_size) as u64)).unwrap(),
                        (i * buf_size) as u64
                    );
                }

                //file.read_exact(&mut buf[..read_size]).unwrap();
                assert_eq!(read_size, file.read(buf).unwrap());
                file_buf_idx = i + 1;
                num_elems_read += num_data;
            }

            match &mut col {
                DBCol::Int(col) => col.extend(
                    (0..num_data)
                        .filter(|i| bmap.contains((*i + total_num_data) as u64))
                        .map(|i| {
                            //let slice = ReadResult::slice(&buf, i * data_size, data_size).unwrap();
                            //NativeEndian::read_i32(&*slice)
                            NativeEndian::read_i32(unsafe {
                                buf.get_unchecked((i * data_size)..((i + 1) * data_size))
                            })
                        }),
                ),
                DBCol::Long(col) => col.extend(
                    (0..num_data)
                        .filter(|i| bmap.contains((*i + total_num_data) as u64))
                        .map(|i| {
                            NativeEndian::read_i64(unsafe {
                                buf.get_unchecked((i * data_size)..((i + 1) * data_size))
                            })
                        }),
                ),
                DBCol::Float(col) => col.extend(
                    (0..num_data)
                        .filter(|i| bmap.contains((*i + total_num_data) as u64))
                        .map(|i| {
                            NativeEndian::read_f32(unsafe {
                                buf.get_unchecked((i * data_size)..((i + 1) * data_size))
                            })
                        }),
                ),
                DBCol::Double(col) => col.extend(
                    (0..num_data)
                        .filter(|i| bmap.contains((*i + total_num_data) as u64))
                        .map(|i| {
                            NativeEndian::read_f64(unsafe {
                                buf.get_unchecked((i * data_size)..((i + 1) * data_size))
                            })
                        }),
                ),
                DBCol::Bool(col) => col.extend(
                    (0..num_data)
                        .filter(|i| bmap.contains((*i + total_num_data) as u64))
                        .map(|i| (unsafe { buf.get_unchecked(i) } > &0)),
                ),
                DBCol::Str(col) => col.extend(
                    (0..num_data)
                        .filter(|i| bmap.contains((*i + total_num_data) as u64))
                        .map(|i| {
                            let str_len = NativeEndian::read_u32(unsafe {
                                buf.get_unchecked((i * data_size)..(i * data_size + 4))
                            }) as usize;
                            let mut str_buf = vec![0 as u8; str_len];
                            str_buf.copy_from_slice(unsafe {
                                buf.get_unchecked(
                                    (i * data_size + 4)..(i * data_size + 4 + str_len),
                                )
                            });
                            unsafe { String::from_utf8_unchecked(str_buf) }
                        }),
                ),
                DBCol::DateTime(col) => col.extend(
                    (0..num_data)
                        .filter(|i| bmap.contains((*i + total_num_data) as u64))
                        .map(|i| {
                            Utc.timestamp_opt(
                                NativeEndian::read_i64(unsafe {
                                    buf.get_unchecked((i * data_size)..((i + 1) * data_size))
                                }),
                                0,
                            )
                            .unwrap()
                        }),
                ),
                DBCol::Duration(col) => col.extend(
                    (0..num_data)
                        .filter(|i| bmap.contains((*i + total_num_data) as u64))
                        .map(|i| {
                            Duration::seconds(NativeEndian::read_i64(unsafe {
                                buf.get_unchecked((i * data_size)..((i + 1) * data_size))
                            }))
                        }),
                ),
            }

            total_num_data += num_data;
        }
        let read_time_ms = now.elapsed().as_millis();

        EXEC_INFO.with(|exec_info| {
            let mut exec_info = exec_info.borrow_mut();
            exec_info.stats.read_time_ms += read_time_ms;
            exec_info.stats.num_elems_read += num_elems_read as u128;
        });

        match output_idx {
            None => col,
            Some(output_idx) => col.project(output_idx),
        }
    }
}

impl PartialEq for FileCol {
    fn eq(&self, other: &Self) -> bool {
        self.full_name() == other.full_name()
    }
}

impl Eq for FileCol {}

impl Hash for FileCol {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.full_name().hash(state);
    }
}

impl PartialOrd for FileCol {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.full_name().partial_cmp(&other.full_name())
    }
}

impl Ord for FileCol {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.full_name().cmp(&other.full_name())
    }
}

struct FileTableSchema {}

impl FileTableSchema {
    fn parse(table_path: &Path) -> io::Result<HashMap<String, Rc<FileCol>>> {
        let schema = fs::read_to_string(table_path.join("__schema__"))?;
        let mut lines = schema.split("\n");
        let attr_names: Vec<&str> = lines.next().unwrap().split(',').collect();
        let attr_types: Vec<&str> = lines.next().unwrap().split(',').collect();

        Ok(attr_names
            .into_iter()
            .zip(attr_types.into_iter())
            .map(|(name, data_type)| {
                (
                    name.to_string(),
                    Rc::new(FileCol {
                        name: name.to_string(),
                        path: table_path.join(name),
                        data_type: DBType::from(data_type),
                        table: RefCell::new(None),
                    }),
                )
            })
            .collect())
    }
}

impl fmt::Debug for FileTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FileTable(id: {}, name: {}, num_records: {}, cols: [{}])",
            self.id,
            self.name,
            self.num_records,
            self.cols.keys().join(", ")
        )
    }
}

impl fmt::Display for FileTableRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            self.alias
                .as_ref()
                .map(|s| s.as_str())
                .unwrap_or_else(|| self.table.name())
        )
    }
}

impl fmt::Debug for FileTableRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FileTableRef(table={}, alias={}, id={})",
            self.table.name(),
            self.alias.as_ref().unwrap_or(&"".to_string()),
            self.id
        )
    }
}

impl fmt::Debug for FileCol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FileCol({})", self.full_name())
    }
}
