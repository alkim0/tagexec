use crate::db::{DBSingle, DBType, DBVals};
use phf::phf_map;
use std::cmp::Ordering;

// Note that a difference with common SQL systems is that types are not often preserved and often
// promoted to either a long or double. This is not a fundamental limitation, but it does make
// coding simpler.

pub static FUNC_MAP: phf::Map<&'static str, fn(Vec<&DBVals>) -> DBVals> = phf_map! {
    "sum" => sum,
    "max" => max,
    "min" => min,
    "avg" => avg,
    "count" => count,
    "dummy_udf" => dummy_udf,
};

fn sum(args: Vec<&DBVals>) -> DBVals {
    assert!(args.len() == 1);
    let vals = args.into_iter().next().unwrap();
    if vals.data_type().is_integral() {
        DBVals::Single(DBSingle::Long(vals.iter_to_long().sum()))
    } else {
        DBVals::Single(DBSingle::Double(vals.iter_to_double().sum()))
    }
}

fn max(args: Vec<&DBVals>) -> DBVals {
    assert!(args.len() == 1);
    let vals = args.into_iter().next().unwrap();
    if vals.data_type().is_integral() {
        DBVals::Single(DBSingle::Long(vals.iter_to_long().max().unwrap_or(0)))
    } else {
        DBVals::Single(DBSingle::Double(
            vals.iter_to_double()
                .max_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Equal))
                .unwrap_or(0.),
        ))
    }
}

fn min(args: Vec<&DBVals>) -> DBVals {
    assert!(args.len() == 1);
    let vals = args.into_iter().next().unwrap();
    if vals.len() == 0 {
        return vals.clone();
    }

    DBVals::Single(match vals.data_type() {
        DBType::Str => DBSingle::Str(vals.iter_as_str().min().unwrap().to_string()),
        data_type if data_type.is_integral() => {
            DBSingle::Long(vals.iter_to_long().min().unwrap_or(0))
        }
        data_type if data_type.is_numeric() => DBSingle::Double(
            vals.iter_to_double()
                .min_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Equal))
                .unwrap_or(0.),
        ),
        data_type => panic!("Unexpected data type for min {:?}", data_type),
    })
}

fn avg(args: Vec<&DBVals>) -> DBVals {
    assert!(args.len() == 1);
    let vals = args.into_iter().next().unwrap();
    if vals.data_type().is_integral() {
        DBVals::Single(DBSingle::Long(
            vals.iter_to_long().sum::<i64>() / vals.len() as i64,
        ))
    } else {
        DBVals::Single(DBSingle::Double(
            vals.iter_to_double().sum::<f64>() / vals.len() as f64,
        ))
    }
}

fn count(args: Vec<&DBVals>) -> DBVals {
    assert!(args.len() == 1);
    let vals = args.into_iter().next().unwrap();
    DBVals::Single(DBSingle::Long(vals.len() as i64))
}

fn dummy_udf(_args: Vec<&DBVals>) -> DBVals {
    unimplemented!();
}
