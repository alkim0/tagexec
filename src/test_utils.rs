/// Utility module useful for testing. Be warned that many functions and structs in this module are
/// unoptimized, since they are meant to be used only during testing.
use crate::db::{DBResultSet, DBSingle};
use itertools::Itertools;
use std::collections::BTreeSet;
use std::fmt;

/// A row-oriented result set.
#[derive(PartialEq, Eq, Debug)]
pub struct RowResultSet(BTreeSet<Vec<DBSingle>>);

impl RowResultSet {
    pub fn difference<'a>(&'a self, other: &'a Self) -> impl Iterator<Item = &'a Vec<DBSingle>> {
        self.0.difference(&other.0)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }
}

impl FromIterator<Vec<DBSingle>> for RowResultSet {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = Vec<DBSingle>>,
    {
        RowResultSet(iter.into_iter().collect())
    }
}

impl From<DBResultSet> for RowResultSet {
    fn from(result_set: DBResultSet) -> Self {
        let mut iters: Vec<_> = result_set.iter().map(|vals| vals.iter_as_dbref()).collect();
        let mut rows = BTreeSet::new();
        loop {
            let row = iters
                .iter_mut()
                .map(|iter| iter.next().map(|val| val.to_owned()))
                .collect::<Option<_>>();
            if let Some(row) = row {
                rows.insert(row);
            } else {
                break;
            }
        }
        Self(rows)
    }
}

impl fmt::Display for RowResultSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            self.0
                .iter()
                .map(|vals| format!("[{}]", vals.iter().map(|x| x.to_string()).join(", ")))
                .join(", ")
        )
    }
}

#[macro_export]
macro_rules! make_row_result_set {
    {
        $subtypes:tt,
        {$($row:tt),* $(,)?}
    } => {
        $crate::test_utils::RowResultSet::from_iter([$(make_row_result_set! { @make_row $subtypes, $row }),*])
    };

    {
        @make_row ($($subtype:tt),*), ($($val:expr),*)
    } => {
        vec![$(make_row_result_set! { @make_single $subtype, $val}),*]
    };

    {
        @make_single Str, $val:expr
    } => {
        $crate::db::DBSingle::Str($val.to_string())
    };

    {
        @make_single $subtype:tt, $val:expr
    } => {
        $crate::db::DBSingle::$subtype($val)
    };

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::DBSingle;
    use std::collections::BTreeSet;

    #[test]
    fn make_result_set() {
        assert_eq!(
            make_row_result_set! { @make_row (Int, Long, Float, Bool, Str), (1, 2, 3., true, "a") },
            vec![
                DBSingle::Int(1),
                DBSingle::Long(2),
                DBSingle::Float(3.),
                DBSingle::Bool(true),
                DBSingle::Str("a".to_string())
            ]
        );

        assert_eq!(
            make_row_result_set! {
                (Int, Long, Float, Bool, Str),
                {
                    (1, 2, 3., true, "a"),
                    (3, 1, -2., false, "b"),
                    (2, 0, -1., false, "c"),
                }
            },
            RowResultSet(BTreeSet::from([
                vec![
                    DBSingle::Int(1),
                    DBSingle::Long(2),
                    DBSingle::Float(3.),
                    DBSingle::Bool(true),
                    DBSingle::Str("a".to_string())
                ],
                vec![
                    DBSingle::Int(3),
                    DBSingle::Long(1),
                    DBSingle::Float(-2.),
                    DBSingle::Bool(false),
                    DBSingle::Str("b".to_string())
                ],
                vec![
                    DBSingle::Int(2),
                    DBSingle::Long(0),
                    DBSingle::Float(-1.),
                    DBSingle::Bool(false),
                    DBSingle::Str("c".to_string())
                ],
            ]))
        );
    }
}
