use crate::file_table::FileTable;
use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::PathBuf;
use std::rc::Rc;

pub struct DataDir {
    path: PathBuf,
    tables: HashMap<String, Rc<FileTable>>,
}

impl DataDir {
    pub fn new(path: PathBuf) -> io::Result<Self> {
        let mut tables = HashMap::new();
        for entry in fs::read_dir(&path)? {
            let entry = entry?;
            let fname = entry.file_name().into_string().unwrap();
            if fname != "__join_keys__" {
                tables.insert(fname, FileTable::new(entry.path())?);
            }
        }

        Ok(Self { path, tables })
    }

    pub fn get_table(&self, table_name: &str) -> Option<&Rc<FileTable>> {
        self.tables.get(table_name)
    }
}
