use crate::expr::Expr;
use crate::file_table::FileTableRef;
use crate::pred::Pred;
use crate::stats::StatsReader;
use itertools::Itertools;
use std::fmt;
use std::rc::Rc;

pub struct Query {
    pub projection: Vec<Expr>,
    pub group_by: Vec<Expr>,
    pub filter: Option<Rc<Pred>>,
    pub from: Vec<Rc<FileTableRef>>,
}

impl Query {
    pub fn update_stats(&mut self, stats_reader: &StatsReader) {
        if let Some(filter) = &mut self.filter {
            filter.update_stats(stats_reader);
        }
    }
}

impl fmt::Display for Query {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SELECT {} FROM {}{}",
            self.projection.iter().join(", "),
            self.from
                .iter()
                .map(|table_ref| format!(
                    "{}{}",
                    table_ref.table.name(),
                    table_ref
                        .alias
                        .as_ref()
                        .map(|alias| format!(" as {}", alias))
                        .unwrap_or("".to_string())
                ))
                .join(", "),
            match &self.filter {
                Some(filter) => format!(" WHERE {}", filter),
                None => "".to_string(),
            }
        )
    }
}
