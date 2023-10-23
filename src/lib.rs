pub mod bin_utils;
mod bitmap;
mod cost;
mod data_dir;
pub mod db;
mod engine;
mod expr;
mod file_cache;
mod file_table;
mod funcs;
mod idx;
mod parse;
mod plan;
mod pred;
mod query;
mod stats;
mod tag;
pub mod test_utils;
mod utils;

pub use data_dir::DataDir;
pub use db::DBResultSet;
pub use engine::{Engine, ExecStats, PlannerType, ReadAllType, ReadSomeType};
pub use parse::Parser;
//pub use plan::{Planner, TaggedPlanner};
pub use plan::PlanSpec;
pub use query::Query;
pub use stats::{StatsGenerator, StatsReader};

//pub fn run_queries(queries: &[&str], db_path: &Path) -> Vec<DBResultSet> {
//    let data_dir = DataDir::new(db_path.to_owned())
//        .expect(format!("Could not find db: {}", db_path.to_str().unwrap()).as_str());
//    let parser = Parser::new(&data_dir);
//    let queries: Vec<_> = queries
//        .into_iter()
//        .map(|query| {
//            let parsed = parser
//                .parse(query)
//                .expect(format!("Could not parse sql query: {}", query).as_str());
//            assert_eq!(parsed.len(), 1);
//            parsed.into_iter().next().unwrap()
//        })
//        .collect();
//    let StatsReader = StatsReader::new(
//    let planner = Planner::new();
//    let plans = planner.plan(queries);
//    let executor = Executor::new(&data_dir);
//    executor.exec(plans)
//}
