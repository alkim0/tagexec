use crate::cost::CostEstimator;
use crate::data_dir::DataDir;
use crate::db::DBResultSet;
use crate::file_table;
use crate::parse::Parser;
use crate::plan::{
    BasicConjPlanner, BasicDisjPlanner, BasicNoOptPlanner, PlanSpec, TaggedCombinedPlanner,
    TaggedConjPushdownPlanner, TaggedIterPushdownPlanner, TaggedNoOptPlanner, TaggedPullupPlanner,
    TaggedPushdownPlanner, TaggedSpecPlanner,
};
use crate::query::Query;
use crate::stats::{StatsGenerator, StatsReader};
use crate::utils;
use log::debug;
use serde::Serialize;
use std::cell::RefCell;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub struct Engine {
    data_dir: DataDir,
    stats_dir: PathBuf,
    exec_info: ExecInfo,
}

#[derive(Debug, Clone, Copy)]
pub enum ReadSomeType {
    Glommio,
    Mmap,
    DirectIO,
    CachedDirectIO,
}

#[derive(Debug, Clone, Copy)]
pub enum ReadAllType {
    Glommio,
    Simple,
    Mmap,
    DirectIO,
    CachedDirectIO,
}

#[derive(Debug, Clone, Copy, strum_macros::Display, strum_macros::EnumString)]
#[strum(serialize_all = "snake_case")]
pub enum PlannerType {
    Basic,
    BasicDisj,
    BasicNoOpt,
    //Tagged,
    TaggedPushdown,
    TaggedSpec,
    TaggedPullup,
    TaggedIterPushdown,
    TaggedConjPushdown,
    TaggedCombined,
    TaggedNoOpt,
}

#[derive(Clone)]
pub struct ExecInfo {
    pub buf_size: usize,
    pub read_all_buf_size: usize,
    pub block_size: usize,
    pub file_cache_size: usize, // This is in terms of number of bufs
    pub file_cache_buf_size: usize,
    pub smoothing_param: f64,
    pub selectivity_threshold: f64,
    pub read_some_type: ReadSomeType,
    pub read_all_type: ReadAllType,
    pub print_plan: bool,
    pub use_mqo: bool, // If true, uses the MQO planner
    pub debug_times: bool,
    pub planner_type: PlannerType,
    pub stats: ExecStats,
    pub cost_est: CostEstimator,
}

thread_local! {
    pub static EXEC_INFO: RefCell<ExecInfo> = RefCell::new(Default::default());
}

#[derive(Default, Clone, Debug, Serialize)]
pub struct ExecStats {
    pub plan_time_ms: u128,
    pub exec_time_ms: u128,
    pub read_time_ms: u128,      // Time to read and parse data from disk
    pub pred_eval_time_ms: u128, // Time taken only to evaluate predicates
    pub union_time_ms: u128,     // Time taken for unions
    pub num_pred_eval: u128,
    pub num_elems_read: u128,
    pub num_filter_tuples: u128, // Number of tuples touched during filtering
    pub num_join_tuples: u128,   // Number of tuples touched during joining
    pub num_union_tuples: u128,  // Number of tuples touched during unioning
}

impl Default for ExecInfo {
    fn default() -> Self {
        Self {
            buf_size: 32 << 10,
            read_all_buf_size: 4 << 20,
            block_size: 512,
            smoothing_param: 1e-9,
            file_cache_size: 500,
            file_cache_buf_size: 4 << 20,
            selectivity_threshold: 0.2,
            read_some_type: ReadSomeType::CachedDirectIO,
            read_all_type: ReadAllType::CachedDirectIO,
            print_plan: false,
            use_mqo: false,
            debug_times: false,
            planner_type: PlannerType::Basic,
            stats: Default::default(),
            cost_est: Default::default(),
        }
    }
}

impl Engine {
    pub fn new(db_path: &Path, stats_dir: &Path) -> io::Result<Self> {
        let stats_dir = if let Some(db_name) = db_path.file_name() {
            stats_dir.join(db_name)
        } else {
            stats_dir.to_owned()
        };
        Ok(Self {
            data_dir: DataDir::new(db_path.to_owned())?,
            stats_dir,
            exec_info: Default::default(),
        })
    }

    pub fn set_print_plan(&mut self, print_plan: bool) -> &mut Self {
        self.exec_info.print_plan = print_plan;
        self
    }

    pub fn set_use_mqo(&mut self, use_mqo: bool) -> &mut Self {
        self.exec_info.use_mqo = use_mqo;
        self
    }

    pub fn set_debug_times(&mut self, debug_times: bool) -> &mut Self {
        self.exec_info.debug_times = debug_times;
        self
    }

    pub fn with_buf_size(&mut self, buf_size: usize) -> &mut Self {
        self.exec_info.buf_size = buf_size;
        self
    }

    pub fn with_read_all_buf_size(&mut self, read_all_buf_size: usize) -> &mut Self {
        self.exec_info.read_all_buf_size = read_all_buf_size;
        self
    }

    pub fn with_read_some_type(&mut self, read_some_type: ReadSomeType) -> &mut Self {
        self.exec_info.read_some_type = read_some_type;
        self
    }

    pub fn with_read_all_type(&mut self, read_all_type: ReadAllType) -> &mut Self {
        self.exec_info.read_all_type = read_all_type;
        self
    }

    pub fn with_file_cache_size(&mut self, file_cache_size: usize) -> &mut Self {
        self.exec_info.file_cache_size = file_cache_size;
        self
    }

    pub fn with_file_cache_buf_size(&mut self, file_cache_buf_size: usize) -> &mut Self {
        self.exec_info.file_cache_buf_size = file_cache_buf_size;
        self
    }

    pub fn with_planner_type(&mut self, planner_type: PlannerType) -> &mut Self {
        self.exec_info.planner_type = planner_type;
        self
    }

    pub fn stats(&self) -> ExecStats {
        EXEC_INFO.with(|exec_info| exec_info.borrow().stats.clone())
    }

    pub fn make_parser(&self) -> Parser<'_> {
        Parser::new(&self.data_dir)
    }

    pub fn run<T: AsRef<str>>(&self, queries: &[T]) -> Vec<DBResultSet> {
        let parser = Parser::new(&self.data_dir);
        let queries: Vec<_> = queries
            .iter()
            .map(|query| {
                debug!("query {}", query.as_ref());
                let query = parser.parse(query.as_ref()).unwrap();
                assert_eq!(query.len(), 1);
                query.into_iter().next().unwrap()
            })
            .collect();
        self.run_queries(queries)
    }

    pub fn run_queries(&self, mut queries: Vec<Query>) -> Vec<DBResultSet> {
        // Set global EXEC_INFO. We never update the stats of the engine directly, so this also
        // effectively resets the stats to 0.
        EXEC_INFO.with(|exec_info| {
            *exec_info.borrow_mut() = self.exec_info.clone();
        });

        file_table::init_file_cache(
            self.exec_info.file_cache_size,
            self.exec_info.file_cache_buf_size,
            self.exec_info.block_size,
        );
        file_table::clear_file_cache();

        let stats_generator = StatsGenerator::new(self.stats_dir.clone());
        let stats_reader = StatsReader::new(self.stats_dir.clone());

        stats_generator
            .generate(&queries)
            .expect("Failed generating stats");
        for query in &mut queries {
            query.update_stats(&stats_reader);
        }

        let (planner_type, print_plan) = EXEC_INFO.with(|exec_info| {
            let exec_info = exec_info.borrow();
            (exec_info.planner_type, exec_info.print_plan)
        });

        let now = Instant::now();
        let plans: Vec<_> = match planner_type {
            PlannerType::Basic => {
                let planner = BasicConjPlanner::new(&stats_reader);

                queries.iter().map(|query| planner.plan(query)).collect()
            }
            PlannerType::BasicDisj => {
                let planner = BasicDisjPlanner::new(&stats_reader);

                queries.iter().map(|query| planner.plan(query)).collect()
            }
            PlannerType::BasicNoOpt => {
                let planner = BasicNoOptPlanner::new(&stats_reader);

                queries.iter().map(|query| planner.plan(query)).collect()
            }
            PlannerType::TaggedPushdown => {
                let planner = TaggedPushdownPlanner::new(&stats_reader);

                queries.iter().map(|query| planner.plan(query)).collect()
            }
            PlannerType::TaggedPullup => {
                let planner = TaggedPullupPlanner::new(&stats_reader);

                queries.iter().map(|query| planner.plan(query)).collect()
            }
            PlannerType::TaggedIterPushdown => {
                let planner = TaggedIterPushdownPlanner::new(&stats_reader);

                queries.iter().map(|query| planner.plan(query)).collect()
            }
            PlannerType::TaggedConjPushdown => {
                let planner = TaggedConjPushdownPlanner::new(&stats_reader);

                queries.iter().map(|query| planner.plan(query)).collect()
            }
            PlannerType::TaggedCombined => {
                let planner = TaggedCombinedPlanner::new(&stats_reader);

                queries.iter().map(|query| planner.plan(query)).collect()
            }
            PlannerType::TaggedNoOpt => {
                let planner = TaggedNoOptPlanner::new(&stats_reader);

                queries.iter().map(|query| planner.plan(query)).collect()
            }
            PlannerType::TaggedSpec => panic!("Use run_query_with_spec for this planner type"),
        };
        let plan_time_ms = now.elapsed().as_millis();

        let now = Instant::now();
        let results = plans.iter().map(|plan| plan.eval()).collect();
        let exec_time_ms = now.elapsed().as_millis();

        if print_plan {
            for (query, plan) in queries.iter().zip(&plans) {
                println!("Plan for: {}\n{}", query, plan);
                //plan.print_est_vs_real_cost();
            }
        }

        EXEC_INFO.with(|exec_info| {
            let mut exec_info = exec_info.borrow_mut();
            exec_info.stats.plan_time_ms = plan_time_ms;
            exec_info.stats.exec_time_ms = exec_time_ms;
        });

        results
    }

    pub fn run_with_spec(&self, query: &str, spec: PlanSpec) -> DBResultSet {
        let parser = Parser::new(&self.data_dir);
        let query = utils::convert_to_one(parser.parse(query).unwrap());
        self.run_query_with_spec(query, spec)
    }

    pub fn run_query_with_spec(&self, query: Query, spec: PlanSpec) -> DBResultSet {
        // Set global EXEC_INFO.
        EXEC_INFO.with(|exec_info| {
            *exec_info.borrow_mut() = self.exec_info.clone();
        });

        file_table::init_file_cache(
            self.exec_info.file_cache_size,
            self.exec_info.file_cache_buf_size,
            self.exec_info.block_size,
        );
        file_table::clear_file_cache();

        let stats_generator = StatsGenerator::new(self.stats_dir.clone());
        let stats_reader = StatsReader::new(self.stats_dir.clone());

        let queries = [query];
        stats_generator
            .generate(&queries)
            .expect("Failed generating stats");
        let mut query = utils::convert_to_one(queries);
        query.update_stats(&stats_reader);

        let print_plan = EXEC_INFO.with(|exec_info| {
            let exec_info = exec_info.borrow();
            exec_info.print_plan
        });

        let query_str = query.to_string();

        let planner = TaggedSpecPlanner::new(&stats_reader);
        let now = Instant::now();
        let plan = planner.plan(&query, &spec);
        let plan_time_ms = now.elapsed().as_millis();

        let now = Instant::now();
        let result = plan.eval();
        let exec_time_ms = now.elapsed().as_millis();

        if print_plan {
            println!("Plan for: {}\n{}", query_str, plan);
        }

        EXEC_INFO.with(|exec_info| {
            let mut exec_info = exec_info.borrow_mut();
            exec_info.stats.plan_time_ms = plan_time_ms;
            exec_info.stats.exec_time_ms = exec_time_ms;
        });

        result
    }
}

//impl fmt::Display for PlannerType {
//    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//        match self {
//            PlannerType::Basic => write!(f, "basic"),
//            PlannerType::BasicDisj => write!(f, "basic_disj"),
//            PlannerType::TaggedPushdown => write!(f, "tagged_pushdown"),
//            //PlannerType::Tagged => write!(f, "tagged"),
//            PlannerType::TaggedSpec => write!(f, "tagged_spec"),
//            PlannerType::TaggedPullup => write!(f, "tagged_pullup"),
//            PlannerType::TaggedIterPushdown => write!(f, "tagged_iter_pushdown"),
//            PlannerType::TaggedConjPushdown => write!(f, "tagged_conj_pushdown"),
//            PlannerType::TaggedCombined => write!(f, "tagged_combined"),
//        }
//    }
//}
