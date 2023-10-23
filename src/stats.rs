use crate::bitmap::BitmapInt;
use crate::expr::EvalContext;
use crate::file_table::FileCol;
use crate::idx::{BasicIdx, Idx};
use crate::pred::PredAtom;
use crate::query::Query;
use crate::utils;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};

pub struct StatsGenerator {
    stats_dir: StatsDir,
}

pub struct StatsReader {
    stats_dir: StatsDir,
    pred_stats_cache: RefCell<HashMap<String, PredStats>>,
    unique_stats_cache: RefCell<HashMap<String, UniqueStats>>,
}

#[derive(Serialize, Deserialize)]
struct PredStats {
    selectivity: f64,
}

#[derive(Serialize, Deserialize)]
struct UniqueStats {
    num_unique: usize,
}

impl StatsGenerator {
    pub fn new(stats_dir: PathBuf) -> Self {
        Self {
            stats_dir: StatsDir(stats_dir),
        }
    }

    pub fn generate(&self, queries: &[Query]) -> io::Result<()> {
        self.stats_dir.ensure_exists();

        let mut existing_stats = HashSet::new();
        //let mut processed_preds: HashSet<&PredAtom> = HashSet::new();
        for query in queries {
            if let Some(pred_root) = &query.filter {
                for pred_atom in pred_root
                    .iter_leaves()
                    .map(|pred| <&PredAtom>::try_from(pred).unwrap())
                    .unique()
                {
                    if pred_atom.has_multiple_table_refs() {
                        let constraint = pred_atom.expr().as_equality_constraint().expect(
                            format!("Unexpected multi-table predicate: {}", pred_atom).as_str(),
                        );

                        for col in [constraint.left_col, constraint.right_col] {
                            let (tmp_path, perm_path) = self.stats_dir.unique_stats_paths(&col);
                            if perm_path.exists() {
                                continue;
                            }

                            {
                                let mut file = File::create(&tmp_path)?;
                                let vals = col.read(0..col.table().len() as BitmapInt);
                                let num_unique = HashSet::<_>::from_iter(vals.iter_to_long()).len();
                                rmp_serde::encode::write(&mut file, &UniqueStats { num_unique })
                                    .map_err(|err| io::Error::new(io::ErrorKind::Other, err))?
                            }

                            self.persist_path(&tmp_path, &perm_path);
                            existing_stats.insert(perm_path);
                        }
                    } else {
                        let (tmp_path, perm_path) = self.stats_dir.pred_stats_paths(&pred_atom);
                        if perm_path.exists() {
                            continue;
                        }

                        {
                            let mut file = File::create(&tmp_path)?;
                            let table_ref = utils::convert_to_one(pred_atom.file_table_refs());
                            let num_records = table_ref.table.len();
                            let idx = Idx::Basic(BasicIdx::new(table_ref));
                            let result = pred_atom.eval(&EvalContext {
                                idx: &idx,
                                bmap: None,
                            });
                            let selectivity = result.len() as f64 / num_records as f64;

                            rmp_serde::encode::write(&mut file, &PredStats { selectivity })
                                .map_err(|err| io::Error::new(io::ErrorKind::Other, err))?;
                        }

                        self.persist_path(&tmp_path, &perm_path);
                        existing_stats.insert(perm_path);
                    }
                }
            }
        }

        Ok(())
    }

    fn persist_path(&self, tmp_path: &Path, perm_path: &Path) {
        fs::rename(tmp_path, perm_path).expect(
            format!(
                "Error moving {} to {}",
                tmp_path.display(),
                perm_path.display()
            )
            .as_str(),
        );
    }
}

impl StatsReader {
    pub fn new(stats_dir: PathBuf) -> Self {
        Self {
            stats_dir: StatsDir(stats_dir),
            pred_stats_cache: RefCell::new(HashMap::new()),
            unique_stats_cache: RefCell::new(HashMap::new()),
        }
    }

    pub fn get_selectivity(&self, pred_atom: &PredAtom) -> f64 {
        assert!(!pred_atom.has_multiple_table_refs());
        let mut cache = self.pred_stats_cache.borrow_mut();
        cache
            .entry(pred_atom.to_string())
            .or_insert_with(|| {
                let (_, stats_path) = self.stats_dir.pred_stats_paths(pred_atom);
                let stats_file = File::open(&stats_path).expect(
                    format!("Could not find stats file: {}", stats_path.display()).as_str(),
                );
                let stats: PredStats = rmp_serde::decode::from_read(stats_file).unwrap();
                stats
            })
            .selectivity
    }

    pub fn get_num_unique(&self, col: &FileCol) -> usize {
        let mut cache = self.unique_stats_cache.borrow_mut();
        let stats = cache.entry(col.full_name()).or_insert_with(|| {
            let (_, stats_path) = self.stats_dir.unique_stats_paths(col);
            let stats_file = File::open(&stats_path)
                .expect(format!("Could not find stats file: {}", stats_path.display()).as_str());
            let stats: UniqueStats = rmp_serde::decode::from_read(stats_file).unwrap();
            stats
        });
        stats.num_unique
    }
}

struct StatsDir(PathBuf);

impl StatsDir {
    fn ensure_exists(&self) {
        if !self.0.exists() {
            fs::create_dir_all(&self.0)
                .expect(format!("Error creating dir {}", self.0.display()).as_str());
        }
    }

    /// Returns the pair (tmp_path, path), in which `tmp_path` is the temporary path to write to
    /// and `path` is the final path which we want to write to.
    fn pred_stats_paths(&self, pred_atom: &PredAtom) -> (PathBuf, PathBuf) {
        let suffix = format!("pred.{}", pred_atom);
        (
            self.0.join(format!("tmp.{}", &suffix)),
            self.0.join(&suffix),
        )
    }

    /// Returns the pair (tmp_path, path), in which `tmp_path` is the temporary path to write to
    /// and `path` is the final path which we want to write to.
    fn unique_stats_paths(&self, col: &FileCol) -> (PathBuf, PathBuf) {
        let suffix = format!("unique.{}", col.full_name());
        (
            self.0.join(format!("tmp.{}", &suffix)),
            self.0.join(&suffix),
        )
    }
}
