use clap::Parser;
use gethostname::gethostname;
use itertools::Itertools;
use serde::Serialize;
use std::path::PathBuf;
use std::str::FromStr;
use tagjoin::{bin_utils, Engine, ExecStats, PlannerType};

const DB_NAME: &'static str = "synth-mixed";

#[derive(Parser)]
struct Args {
    #[arg(short = 't', long, default_value_t = 3)]
    num_trials: usize,

    #[arg(long)]
    output_prefix: Option<String>,

    #[arg(short, long)]
    output: Option<PathBuf>,

    #[arg(long)]
    synth_dbs_dir: Option<PathBuf>,

    #[arg(long)]
    no_output: bool,

    #[arg(long)]
    stats_dir: Option<PathBuf>,

    #[arg(long)]
    debug: bool,

    #[arg(long, value_parser = parse_comma_exp_type_list)]
    exp_type: Option<std::vec::Vec<ExpType>>,

    #[arg(long, value_parser = bin_utils::parse_comma_planner_type_list)]
    planner_type: Option<std::vec::Vec<PlannerType>>,
}

#[derive(Clone, Copy, strum_macros::Display, strum_macros::EnumString)]
#[strum(serialize_all = "snake_case")]
enum ExpType {
    NumClauses,
    NumTables,
    Selectivity,
    NumRedundancies,
    OuterConjFactor,
    TableSize,
}

#[derive(Serialize)]
struct Record {
    header: ExpHeader,
    params: ExpParams,
    stats: ExecStats,
}

#[derive(Serialize)]
struct ExpHeader {
    exp_type: String,
    planner_type: String,
    trial: usize,
}

#[derive(Clone, Serialize)]
struct ExpParams {
    pred_selectivity: f64,
    outer_conj_factor: f64,
    table_size: usize,
    num_tables: usize,
    num_redundancies: usize,
    num_discriminating: usize,
    clause_size: usize,
    num_clauses: usize,
    is_cnf: bool,
}

impl Default for ExpParams {
    fn default() -> Self {
        Self {
            pred_selectivity: 0.2,
            outer_conj_factor: 1.,
            table_size: 1e4 as usize,
            num_tables: 3,
            num_redundancies: 0,
            num_discriminating: 0,
            clause_size: 2,
            num_clauses: 2,
            is_cnf: false,
        }
    }
}

fn table_name(table_size: usize, i: usize) -> String {
    format!("t_{table_size:e}_{i}")
}

impl ExpParams {
    fn build_query(&self) -> String {
        self.build_query_with_filter(&self.build_filter_from_vec(self.build_filter_vec()))
    }

    fn build_query_with_outer_conj_factor(&self) -> String {
        let mut preds = self.build_filter_vec();

        let outer_conj_factor_pred = format!(
            "{}.a0 < {}",
            table_name(self.table_size, 0),
            self.outer_conj_factor
        );
        if self.is_cnf {
            preds.push(vec![outer_conj_factor_pred]);
        } else {
            for clause in preds.iter_mut() {
                clause.push(outer_conj_factor_pred.clone());
            }
        }

        self.build_query_with_filter(&self.build_filter_from_vec(preds))
    }

    fn build_query_with_filter(&self, filter: &str) -> String {
        let tables: Vec<_> = (0..self.num_tables)
            .map(|i| table_name(self.table_size, i))
            .collect();

        format!(
            "select {} from {} where {} and ({})",
            tables
                .iter()
                .map(|table| format!("min({table}.pk)"))
                .join(", "),
            tables.iter().join(", "),
            tables[1..]
                .iter()
                .map(|table| format!("{}.pk = {}.fk0", tables[0], table))
                .join(" and "),
            filter
        )
    }

    fn build_filter_from_vec(&self, preds: Vec<Vec<String>>) -> String {
        preds
            .into_iter()
            .map(|clause| {
                format!(
                    "({})",
                    clause.join(if self.is_cnf { " or " } else { " and " })
                )
            })
            .join(if self.is_cnf { " and " } else { " or " })
    }

    fn build_filter_vec(&self) -> Vec<Vec<String>> {
        assert_eq!(self.clause_size, 2);
        (0..self.num_clauses)
            .map(|i| {
                vec![
                    format!(
                        "{}.a{} < {}",
                        table_name(self.table_size, 1),
                        if i <= self.num_redundancies { 0 } else { i },
                        self.pred_selectivity
                    ),
                    format!(
                        "{}.a{} < {}",
                        table_name(self.table_size, 2),
                        i,
                        self.pred_selectivity
                    ),
                ]
            })
            .collect()
    }

    fn get_planner_types(&self) -> Vec<PlannerType> {
        if self.is_cnf {
            vec![
                PlannerType::BasicNoOpt,
                PlannerType::TaggedPushdown,
                PlannerType::TaggedPullup,
                PlannerType::TaggedIterPushdown,
                PlannerType::TaggedCombined,
            ]
        } else {
            vec![
                PlannerType::BasicDisj,
                PlannerType::TaggedPushdown,
                PlannerType::TaggedPullup,
                PlannerType::TaggedIterPushdown,
                PlannerType::TaggedCombined,
            ]
        }
    }
}

fn exp_debug(args: &Args) {
    let exp_type = ExpType::OuterConjFactor.to_string();

    let default_params = ExpParams::default();
    let mut engine = Engine::new(
        &args.synth_dbs_dir.as_ref().unwrap().join(DB_NAME),
        &args.stats_dir.as_ref().unwrap(),
    )
    .unwrap();

    let mut records = vec![];

    for is_cnf in [true] {
        let outer_conj_factor = 0.5;
        let mut params = default_params.clone();
        params.is_cnf = is_cnf;
        params.outer_conj_factor = outer_conj_factor;

        let query = params.build_query_with_outer_conj_factor();
        //let planner_types = params.get_planner_types();
        let planner_types = [PlannerType::BasicNoOpt, PlannerType::TaggedPushdown];

        for trial in 0..1 {
            let mut outputs = vec![];

            for &planner_type in &planner_types {
                println!(
                    "Running exp {} {} {} trial {} planner type {}",
                    exp_type,
                    outer_conj_factor,
                    if is_cnf { "cnf" } else { "dnf" },
                    trial,
                    planner_type
                );
                bin_utils::drop_caches();
                outputs.push(
                    engine
                        .with_planner_type(planner_type)
                        .set_print_plan(true)
                        .run(&[&query]),
                );

                let stats = engine.stats();
                println!("{:?}", stats);
                records.push(Record {
                    header: ExpHeader {
                        exp_type: exp_type.clone(),
                        planner_type: planner_type.to_string(),
                        trial,
                    },
                    params: params.clone(),
                    stats,
                })
            }

            if outputs.len() >= 2 {
                for output in &outputs {
                    println!("{:?}", output);
                }
                for output in &outputs {
                    assert_eq!(&outputs[0], output);
                }
            }
        }
    }
}

/*
fn exp_num_discriminating(args: &Args) -> Vec<Record> {
    let pred_selectivity = 0.2;
    let db_size = 2e4 as usize;
    let db_name = "synth-mixed";
    let num_tables = 3;
    let num_redundancies = 0;
    let num_clauses = 5;
    let clause_size = 2;
    let pred_sleep_time_ms = 0;
    let outer_conj_factor = 1.;

    let mut engine = Engine::new(
        &args.synth_dbs_dir.as_ref().unwrap().join(db_name),
        &args.stats_dir.as_ref().unwrap(),
    )
    .unwrap();

    let mut records = vec![];

    for num_discriminating in 0..4 {
        for pred_is_cnf in [true, false] {
            let query = format!(
                "select min(t_2e4_0.pk), min(t_2e4_1.pk), min(t_2e4_2.pk) from t_2e4_0, t_2e4_1, t_2e4_2 where t_2e4_0.pk = t_2e4_1.fk0 and t_2e4_0.pk = t_2e4_2.fk0 and ({})",
                (0..num_clauses)
                    .map(|i| format!(
                        "(t_2e4_1.a{} < {} {} {}.a{} < {})",
                        i,
                        pred_selectivity,
                        if pred_is_cnf { "or" } else { "and" },
                        if i < num_discriminating {
                            "t_2e4_0"
                        } else {
                            "t_2e4_2"
                        },
                        i,
                        pred_selectivity
                    ))
                    .join(if pred_is_cnf { " and " } else { " or " })
            );

            let planner_types = if pred_is_cnf {
                vec![
                    PlannerType::BasicNoOpt,
                    PlannerType::TaggedPushdown,
                    PlannerType::TaggedPullup,
                    PlannerType::TaggedIterPushdown,
                    PlannerType::TaggedCombined,
                ]
            } else {
                vec![
                    PlannerType::BasicDisj,
                    PlannerType::TaggedPushdown,
                    PlannerType::TaggedPullup,
                    PlannerType::TaggedIterPushdown,
                    PlannerType::TaggedCombined,
                ]
            };

            for trial in 0..args.num_trials {
                let mut outputs = vec![];
                for &planner_type in &planner_types {
                    println!(
                        "Running exp num discriminating {} {} trial {} planner type {}",
                        num_discriminating,
                        if pred_is_cnf { "cnf" } else { "dnf" },
                        trial,
                        planner_type
                    );
                    bin_utils::drop_caches();
                    outputs.push(
                        engine
                            .with_planner_type(planner_type)
                            .set_print_plan(true)
                            .run(&[&query]),
                    );

                    let stats = engine.stats();
                    println!("{:?}", stats);
                    records.push(Record {
                        exp_type: ExpType::NumDiscriminating.to_string(),
                        planner_type: planner_type.to_string(),
                        trial,
                        pred_selectivity,
                        outer_conj_factor,
                        db_size,
                        num_tables,
                        num_redundancies,
                        num_discriminating,
                        clause_size,
                        num_clauses,
                        pred_sleep_time_ms,
                        conj_pred_root: pred_is_cnf,
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
    }

    records
}
*/

fn exp_num_clauses(args: &Args) -> Vec<Record> {
    let exp_type = ExpType::NumClauses.to_string();

    let default_params = ExpParams::default();
    let mut engine = Engine::new(
        &args.synth_dbs_dir.as_ref().unwrap().join(DB_NAME),
        &args.stats_dir.as_ref().unwrap(),
    )
    .unwrap();

    let mut records = vec![];

    for is_cnf in [true, false] {
        for num_clauses in 2..8 {
            let mut params = default_params.clone();
            params.is_cnf = is_cnf;
            params.num_clauses = num_clauses;

            let query = params.build_query();
            let planner_types = params.get_planner_types();

            for trial in 0..args.num_trials {
                let mut outputs = vec![];

                for &planner_type in &planner_types {
                    println!(
                        "Running exp {} {} {} trial {} planner type {}",
                        exp_type,
                        num_clauses,
                        if is_cnf { "cnf" } else { "dnf" },
                        trial,
                        planner_type
                    );
                    bin_utils::drop_caches();
                    outputs.push(
                        engine
                            .with_planner_type(planner_type)
                            .set_print_plan(true)
                            .run(&[&query]),
                    );

                    let stats = engine.stats();
                    println!("{:?}", stats);
                    records.push(Record {
                        header: ExpHeader {
                            exp_type: exp_type.clone(),
                            planner_type: planner_type.to_string(),
                            trial,
                        },
                        params: params.clone(),
                        stats,
                    })
                }

                if outputs.len() >= 2 {
                    for output in &outputs {
                        assert_eq!(&outputs[0], output);
                    }
                }
            }
        }
    }

    records
}

fn exp_num_tables(args: &Args) -> Vec<Record> {
    let exp_type = ExpType::NumTables.to_string();

    let default_params = ExpParams::default();
    let mut engine = Engine::new(
        &args.synth_dbs_dir.as_ref().unwrap().join(DB_NAME),
        &args.stats_dir.as_ref().unwrap(),
    )
    .unwrap();

    let mut records = vec![];

    for is_cnf in [true, false] {
        for num_tables in (3..7).rev() {
            let mut params = default_params.clone();
            params.is_cnf = is_cnf;
            params.table_size = 1e2 as usize;
            params.num_tables = num_tables;

            let query = params.build_query();
            let planner_types = params.get_planner_types();

            for trial in 0..args.num_trials {
                let mut outputs = vec![];

                for &planner_type in &planner_types {
                    println!(
                        "Running exp {} {} {} trial {} planner type {}",
                        exp_type,
                        num_tables,
                        if is_cnf { "cnf" } else { "dnf" },
                        trial,
                        planner_type
                    );
                    bin_utils::drop_caches();
                    outputs.push(
                        engine
                            .with_planner_type(planner_type)
                            .set_print_plan(true)
                            .run(&[&query]),
                    );

                    let stats = engine.stats();
                    println!("{:?}", stats);
                    records.push(Record {
                        header: ExpHeader {
                            exp_type: exp_type.clone(),
                            planner_type: planner_type.to_string(),
                            trial,
                        },
                        params: params.clone(),
                        stats,
                    })
                }

                if outputs.len() >= 2 {
                    for output in &outputs {
                        assert_eq!(&outputs[0], output);
                    }
                }
            }
        }
    }

    records
}

fn exp_selectivity(args: &Args) -> Vec<Record> {
    let exp_type = ExpType::Selectivity.to_string();

    let default_params = ExpParams::default();
    let mut engine = Engine::new(
        &args.synth_dbs_dir.as_ref().unwrap().join(DB_NAME),
        &args.stats_dir.as_ref().unwrap(),
    )
    .unwrap();

    let mut records = vec![];

    for is_cnf in [true, false] {
        for pred_selectivity in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            .into_iter()
            .rev()
        {
            let mut params = default_params.clone();
            params.is_cnf = is_cnf;
            params.pred_selectivity = pred_selectivity;

            let query = params.build_query();
            let planner_types = params.get_planner_types();

            for trial in 0..args.num_trials {
                let mut outputs = vec![];

                for &planner_type in &planner_types {
                    println!(
                        "Running exp {} {} {} trial {} planner type {}",
                        exp_type,
                        pred_selectivity,
                        if is_cnf { "cnf" } else { "dnf" },
                        trial,
                        planner_type
                    );
                    bin_utils::drop_caches();
                    outputs.push(
                        engine
                            .with_planner_type(planner_type)
                            .set_print_plan(true)
                            .run(&[&query]),
                    );

                    let stats = engine.stats();
                    println!("{:?}", stats);
                    records.push(Record {
                        header: ExpHeader {
                            exp_type: exp_type.clone(),
                            planner_type: planner_type.to_string(),
                            trial,
                        },
                        params: params.clone(),
                        stats,
                    })
                }

                if outputs.len() >= 2 {
                    for output in &outputs {
                        assert_eq!(&outputs[0], output);
                    }
                }
            }
        }
    }

    records
}

fn exp_num_redundancies(args: &Args) -> Vec<Record> {
    let exp_type = ExpType::NumRedundancies.to_string();

    let default_params = ExpParams::default();
    let mut engine = Engine::new(
        &args.synth_dbs_dir.as_ref().unwrap().join(DB_NAME),
        &args.stats_dir.as_ref().unwrap(),
    )
    .unwrap();

    let mut records = vec![];

    for is_cnf in [true, false] {
        for num_redundancies in 0..5 {
            let mut params = default_params.clone();
            params.is_cnf = is_cnf;
            params.num_clauses = 5;
            params.num_redundancies = num_redundancies;

            let query = params.build_query();
            let planner_types = params.get_planner_types();

            for trial in 0..args.num_trials {
                let mut outputs = vec![];

                for &planner_type in &planner_types {
                    println!(
                        "Running exp {} {} {} trial {} planner type {}",
                        exp_type,
                        num_redundancies,
                        if is_cnf { "cnf" } else { "dnf" },
                        trial,
                        planner_type
                    );
                    bin_utils::drop_caches();
                    outputs.push(
                        engine
                            .with_planner_type(planner_type)
                            .set_print_plan(true)
                            .run(&[&query]),
                    );

                    let stats = engine.stats();
                    println!("{:?}", stats);
                    records.push(Record {
                        header: ExpHeader {
                            exp_type: exp_type.clone(),
                            planner_type: planner_type.to_string(),
                            trial,
                        },
                        params: params.clone(),
                        stats,
                    })
                }

                if outputs.len() >= 2 {
                    for output in &outputs {
                        assert_eq!(&outputs[0], output);
                    }
                }
            }
        }
    }

    records
}

fn exp_outer_conj_factor(args: &Args) -> Vec<Record> {
    let exp_type = ExpType::OuterConjFactor.to_string();

    let default_params = ExpParams::default();
    let mut engine = Engine::new(
        &args.synth_dbs_dir.as_ref().unwrap().join(DB_NAME),
        &args.stats_dir.as_ref().unwrap(),
    )
    .unwrap();

    let mut records = vec![];

    for is_cnf in [true, false] {
        for outer_conj_factor in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            .into_iter()
            .rev()
        {
            let mut params = default_params.clone();
            params.is_cnf = is_cnf;
            params.outer_conj_factor = outer_conj_factor;

            let query = params.build_query_with_outer_conj_factor();
            let mut planner_types = params.get_planner_types();
            if params.is_cnf {
                planner_types.push(PlannerType::Basic);
            }

            for trial in 0..args.num_trials {
                let mut outputs = vec![];

                for &planner_type in &planner_types {
                    println!(
                        "Running exp {} {} {} trial {} planner type {}",
                        exp_type,
                        outer_conj_factor,
                        if is_cnf { "cnf" } else { "dnf" },
                        trial,
                        planner_type
                    );
                    bin_utils::drop_caches();
                    outputs.push(
                        engine
                            .with_planner_type(planner_type)
                            .set_print_plan(true)
                            .run(&[&query]),
                    );

                    let stats = engine.stats();
                    println!("{:?}", stats);
                    records.push(Record {
                        header: ExpHeader {
                            exp_type: exp_type.clone(),
                            planner_type: planner_type.to_string(),
                            trial,
                        },
                        params: params.clone(),
                        stats,
                    })
                }

                if outputs.len() >= 2 {
                    for output in &outputs {
                        assert_eq!(&outputs[0], output);
                    }
                }
            }
        }
    }

    records
}

fn exp_table_size(args: &Args) -> Vec<Record> {
    let exp_type = ExpType::TableSize.to_string();

    let default_params = ExpParams::default();
    let mut engine = Engine::new(
        &args.synth_dbs_dir.as_ref().unwrap().join(DB_NAME),
        &args.stats_dir.as_ref().unwrap(),
    )
    .unwrap();

    let mut records = vec![];

    for is_cnf in [true, false] {
        for table_size in [
            1e3 as usize,
            2e3 as usize,
            5e3 as usize,
            1e4 as usize,
            2e4 as usize,
            5e4 as usize,
            //1e5 as usize,
        ]
        .into_iter()
        .rev()
        {
            let mut params = default_params.clone();
            params.is_cnf = is_cnf;
            params.table_size = table_size;

            let query = params.build_query();
            let planner_types = params.get_planner_types();

            for trial in 0..args.num_trials {
                let mut outputs = vec![];

                for &planner_type in &planner_types {
                    println!(
                        "Running exp {} {:e} {} trial {} planner type {}",
                        exp_type,
                        table_size,
                        if is_cnf { "cnf" } else { "dnf" },
                        trial,
                        planner_type
                    );
                    bin_utils::drop_caches();
                    outputs.push(
                        engine
                            .with_planner_type(planner_type)
                            .set_print_plan(true)
                            .run(&[&query]),
                    );

                    let stats = engine.stats();
                    println!("{:?}", stats);
                    records.push(Record {
                        header: ExpHeader {
                            exp_type: exp_type.clone(),
                            planner_type: planner_type.to_string(),
                            trial,
                        },
                        params: params.clone(),
                        stats,
                    })
                }

                if outputs.len() >= 2 {
                    for output in &outputs {
                        assert_eq!(&outputs[0], output);
                    }
                }
            }
        }
    }

    records
}

fn main() {
    let args = Args::parse().with_defaults();

    if args.debug {
        exp_debug(&args);
        return;
    }

    let mut records = vec![];
    for exp_type in args.exp_type.clone().unwrap_or(vec![
        ExpType::NumClauses,
        ExpType::NumTables,
        ExpType::Selectivity,
        ExpType::NumRedundancies,
        ExpType::OuterConjFactor,
        ExpType::TableSize,
    ]) {
        match exp_type {
            ExpType::NumClauses => records.extend(exp_num_clauses(&args)),
            ExpType::NumTables => records.extend(exp_num_tables(&args)),
            ExpType::Selectivity => records.extend(exp_selectivity(&args)),
            ExpType::NumRedundancies => records.extend(exp_num_redundancies(&args)),
            ExpType::OuterConjFactor => records.extend(exp_outer_conj_factor(&args)),
            ExpType::TableSize => records.extend(exp_table_size(&args)),
        }
    }

    if !args.no_output {
        let records = records
            .into_iter()
            .map(|record| (record.header, record.params, record.stats))
            .collect();
        bin_utils::write_records(&args.output.as_ref().unwrap(), records).unwrap();
    }
}

impl Args {
    fn with_defaults(mut self) -> Self {
        self.output
            .get_or_insert(bin_utils::default_output_dir().join(format!(
                "{}-{}-{}.csv",
                self.output_prefix
                    .as_ref()
                    .map(|s| s.as_str())
                    .unwrap_or("synth-exp"),
                gethostname().to_string_lossy(),
                chrono::Local::now().format("%FT%H%M%S%z")
            )));

        self.synth_dbs_dir.get_or_insert(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("data")
                .join("synth"),
        );

        self.stats_dir.get_or_insert(bin_utils::default_stats_dir());

        self.planner_type
            .get_or_insert(bin_utils::default_planner_types());

        self
    }
}

fn parse_comma_exp_type_list(s: &str) -> Result<Vec<ExpType>, strum::ParseError> {
    s.split(",")
        .map(|exp_type| ExpType::from_str(exp_type))
        .collect::<Result<_, _>>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_query() {
        let params = ExpParams::default();
        assert_eq!(params.build_query(), "select min(t_1e4_0.pk), min(t_1e4_1.pk), min(t_1e4_2.pk) from t_1e4_0, t_1e4_1, t_1e4_2 where t_1e4_0.pk = t_1e4_1.fk0 and t_1e4_0.pk = t_1e4_2.fk0 and ((t_1e4_1.a0 < 0.2 and t_1e4_2.a0 < 0.2) or (t_1e4_1.a1 < 0.2 and t_1e4_2.a1 < 0.2))");

        let mut params = ExpParams::default();
        params.is_cnf = true;
        assert_eq!(params.build_query(), "select min(t_1e4_0.pk), min(t_1e4_1.pk), min(t_1e4_2.pk) from t_1e4_0, t_1e4_1, t_1e4_2 where t_1e4_0.pk = t_1e4_1.fk0 and t_1e4_0.pk = t_1e4_2.fk0 and ((t_1e4_1.a0 < 0.2 or t_1e4_2.a0 < 0.2) and (t_1e4_1.a1 < 0.2 or t_1e4_2.a1 < 0.2))");

        let mut params = ExpParams::default();
        params.table_size = 1e3 as usize;
        assert_eq!(params.build_query(), "select min(t_1e3_0.pk), min(t_1e3_1.pk), min(t_1e3_2.pk) from t_1e3_0, t_1e3_1, t_1e3_2 where t_1e3_0.pk = t_1e3_1.fk0 and t_1e3_0.pk = t_1e3_2.fk0 and ((t_1e3_1.a0 < 0.2 and t_1e3_2.a0 < 0.2) or (t_1e3_1.a1 < 0.2 and t_1e3_2.a1 < 0.2))");

        let mut params = ExpParams::default();
        params.table_size = 1e2 as usize;
        params.num_tables = 5 as usize;
        assert_eq!(params.build_query(), "select min(t_1e2_0.pk), min(t_1e2_1.pk), min(t_1e2_2.pk), min(t_1e2_3.pk), min(t_1e2_4.pk) from t_1e2_0, t_1e2_1, t_1e2_2, t_1e2_3, t_1e2_4 where t_1e2_0.pk = t_1e2_1.fk0 and t_1e2_0.pk = t_1e2_2.fk0 and t_1e2_0.pk = t_1e2_3.fk0 and t_1e2_0.pk = t_1e2_4.fk0 and ((t_1e2_1.a0 < 0.2 and t_1e2_2.a0 < 0.2) or (t_1e2_1.a1 < 0.2 and t_1e2_2.a1 < 0.2))");

        let mut params = ExpParams::default();
        params.pred_selectivity = 0.6;
        assert_eq!(params.build_query(), "select min(t_1e4_0.pk), min(t_1e4_1.pk), min(t_1e4_2.pk) from t_1e4_0, t_1e4_1, t_1e4_2 where t_1e4_0.pk = t_1e4_1.fk0 and t_1e4_0.pk = t_1e4_2.fk0 and ((t_1e4_1.a0 < 0.6 and t_1e4_2.a0 < 0.6) or (t_1e4_1.a1 < 0.6 and t_1e4_2.a1 < 0.6))");

        let mut params = ExpParams::default();
        params.num_clauses = 5;
        assert_eq!(params.build_query(), "select min(t_1e4_0.pk), min(t_1e4_1.pk), min(t_1e4_2.pk) from t_1e4_0, t_1e4_1, t_1e4_2 where t_1e4_0.pk = t_1e4_1.fk0 and t_1e4_0.pk = t_1e4_2.fk0 and ((t_1e4_1.a0 < 0.2 and t_1e4_2.a0 < 0.2) or (t_1e4_1.a1 < 0.2 and t_1e4_2.a1 < 0.2) or (t_1e4_1.a2 < 0.2 and t_1e4_2.a2 < 0.2) or (t_1e4_1.a3 < 0.2 and t_1e4_2.a3 < 0.2) or (t_1e4_1.a4 < 0.2 and t_1e4_2.a4 < 0.2))");

        let mut params = ExpParams::default();
        params.num_clauses = 5;
        params.num_redundancies = 3;
        assert_eq!(params.build_query(), "select min(t_1e4_0.pk), min(t_1e4_1.pk), min(t_1e4_2.pk) from t_1e4_0, t_1e4_1, t_1e4_2 where t_1e4_0.pk = t_1e4_1.fk0 and t_1e4_0.pk = t_1e4_2.fk0 and ((t_1e4_1.a0 < 0.2 and t_1e4_2.a0 < 0.2) or (t_1e4_1.a0 < 0.2 and t_1e4_2.a1 < 0.2) or (t_1e4_1.a0 < 0.2 and t_1e4_2.a2 < 0.2) or (t_1e4_1.a0 < 0.2 and t_1e4_2.a3 < 0.2) or (t_1e4_1.a4 < 0.2 and t_1e4_2.a4 < 0.2))");

        let mut params = ExpParams::default();
        params.outer_conj_factor = 0.5;
        assert_eq!(params.build_query_with_outer_conj_factor(), "select min(t_1e4_0.pk), min(t_1e4_1.pk), min(t_1e4_2.pk) from t_1e4_0, t_1e4_1, t_1e4_2 where t_1e4_0.pk = t_1e4_1.fk0 and t_1e4_0.pk = t_1e4_2.fk0 and ((t_1e4_1.a0 < 0.2 and t_1e4_2.a0 < 0.2 and t_1e4_0.a0 < 0.5) or (t_1e4_1.a1 < 0.2 and t_1e4_2.a1 < 0.2 and t_1e4_0.a0 < 0.5))");

        let mut params = ExpParams::default();
        params.outer_conj_factor = 0.5;
        params.is_cnf = true;
        assert_eq!(params.build_query_with_outer_conj_factor(), "select min(t_1e4_0.pk), min(t_1e4_1.pk), min(t_1e4_2.pk) from t_1e4_0, t_1e4_1, t_1e4_2 where t_1e4_0.pk = t_1e4_1.fk0 and t_1e4_0.pk = t_1e4_2.fk0 and ((t_1e4_1.a0 < 0.2 or t_1e4_2.a0 < 0.2) and (t_1e4_1.a1 < 0.2 or t_1e4_2.a1 < 0.2) and (t_1e4_0.a0 < 0.5))");
    }
}
