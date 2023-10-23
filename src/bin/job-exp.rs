use clap::Parser;
use gethostname::gethostname;
use regex::Regex;
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};
use tagjoin::{bin_utils, Engine, PlannerType};

#[derive(Parser)]
struct Args {
    #[arg(short = 't', long, default_value_t = 3)]
    num_trials: usize,

    #[arg(short, long, value_parser = bin_utils::parse_comma_range_num_list)]
    query: Option<std::vec::Vec<usize>>, // This needs to be std::vec::Vec to opt-out of clap
    // auto-inferring that we need to consume multiple strings
    // for this Vec argument
    #[arg(long)]
    output_prefix: Option<String>,

    #[arg(short, long)]
    output: Option<PathBuf>,

    #[arg(long)]
    no_output: bool,

    #[arg(long)]
    queries_dir: Option<PathBuf>,

    #[arg(long)]
    stats_dir: Option<PathBuf>,

    #[arg(long)]
    db_path: Option<PathBuf>,

    #[arg(long)]
    query_form: Option<String>,

    #[arg(long, value_parser = bin_utils::parse_comma_planner_type_list)]
    planner_type: Option<std::vec::Vec<PlannerType>>,
}

#[derive(Serialize)]
struct Record {
    query_num: usize,
    planner_type: String,
    trial: usize,
    plan_time_ms: u128,
    exec_time_ms: u128,
    read_time_ms: u128,
    pred_eval_time_ms: u128,
    union_time_ms: u128,
    num_pred_eval: u128,
    num_filter_tuples: u128,
    num_join_tuples: u128,
    num_union_tuples: u128,
}

fn main() {
    let args = Args::parse().with_defaults();

    let mut engine = Engine::new(&args.db_path.unwrap(), &args.stats_dir.unwrap()).unwrap();

    let mut records = vec![];

    for query_num in args.query.unwrap() {
        let queries = get_queries(query_num, args.queries_dir.as_ref().unwrap());
        for trial in 0..args.num_trials {
            let mut outputs = vec![];
            for &planner_type in args.planner_type.as_ref().unwrap() {
                println!(
                    "Running query {} trial {} planner type {:?}",
                    query_num, trial, planner_type
                );
                bin_utils::drop_caches();
                let query = {
                    let parser = engine.make_parser();
                    match args.query_form.as_ref().map(|s| s.as_str()) {
                        Some("conj") => parser.or_combine_queries_conj_root(&queries),
                        Some("disj") => parser.or_combine_queries_disj_root(&queries),
                        _ => match planner_type {
                            PlannerType::Basic
                            | PlannerType::TaggedConjPushdown
                            | PlannerType::TaggedCombined => {
                                parser.or_combine_queries_conj_root(&queries)
                            }
                            PlannerType::BasicDisj
                            | PlannerType::TaggedPushdown
                            | PlannerType::TaggedPullup
                            | PlannerType::TaggedIterPushdown => {
                                parser.or_combine_queries_disj_root(&queries)
                            }
                            _ => panic!(
                                "Haven't decided which combined query form to use for {:?}",
                                planner_type
                            ),
                        },
                    }
                };
                outputs.push(
                    engine
                        .with_planner_type(planner_type)
                        .set_print_plan(true)
                        .run_queries(vec![query]),
                );

                let stats = engine.stats();
                println!("{:?}", stats);
                records.push(Record {
                    query_num,
                    planner_type: planner_type.to_string(),
                    trial,
                    plan_time_ms: stats.plan_time_ms,
                    exec_time_ms: stats.exec_time_ms,
                    read_time_ms: stats.read_time_ms,
                    pred_eval_time_ms: stats.pred_eval_time_ms,
                    union_time_ms: stats.union_time_ms,
                    num_pred_eval: stats.num_pred_eval,
                    num_filter_tuples: stats.num_filter_tuples,
                    num_join_tuples: stats.num_join_tuples,
                    num_union_tuples: stats.num_union_tuples,
                });
            }

            if outputs.len() >= 2 {
                for output in &outputs {
                    assert_eq!(&outputs[0], output);
                }
            }
        }
    }

    if !args.no_output {
        bin_utils::write_records(&args.output.unwrap(), records).unwrap();
    }
}

fn get_queries(query_num: usize, queries_dir: &Path) -> Vec<String> {
    let re = Regex::new(format!("^{}[a-z].sql$", query_num).as_str()).unwrap();
    queries_dir
        .read_dir()
        .unwrap()
        .filter(|entry| {
            let entry = entry.as_ref().unwrap();
            let file_name = entry.file_name().into_string().unwrap();
            re.is_match(&file_name)
        })
        .map(|entry| {
            let entry = entry.as_ref().unwrap();
            String::from_utf8(fs::read(&entry.path()).unwrap()).unwrap()
        })
        .collect()
}

impl Args {
    fn with_defaults(mut self) -> Self {
        self.output
            .get_or_insert(bin_utils::default_output_dir().join(format!(
                    "{}-{}-{}.csv",
                    self.output_prefix
                        .as_ref()
                        .map(|s| s.as_str())
                        .unwrap_or("job-exp"),
                    gethostname().to_string_lossy(),
                    chrono::Local::now().format("%FT%H%M%S%z")
                )));
        self.queries_dir.get_or_insert(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("queries")
                .join("join-order-benchmark"),
        );

        self.db_path.get_or_insert(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("data")
                .join("imdb"),
        );

        self.stats_dir.get_or_insert(bin_utils::default_stats_dir());

        self.query.get_or_insert((1..34).into_iter().collect());

        self.planner_type
            .get_or_insert(bin_utils::default_planner_types());

        self
    }
}
