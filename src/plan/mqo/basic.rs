use super::utils as plan_utils;
use super::BenefitCostRatio;
use crate::db::DBResultSet;
use crate::expr::{EqualityConstraint, EvalContext, Expr};
use crate::file_table::FileTable;
use crate::idx::{ColIterable, Idx};
use crate::pred::Pred;
use crate::query::Query;
use crate::stats::StatsReader;
use crate::utils;
use either::Either;
use itertools::Itertools;
use std::borrow::Cow;
use std::cell::{Ref, RefCell};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::rc::Rc;
use std::time::Instant;

pub struct Planner<'a> {
    stats_reader: &'a StatsReader,
}

pub struct Plan {
    root: ProjectNode,
    total_time_ms: RefCell<u128>,
}

enum PlanNode {
    Join(JoinNode),
    Filter(FilterNode),
    Base(BaseNode),
}

impl<'a> Planner<'a> {
    pub fn new(stats_reader: &'a StatsReader) -> Self {
        Self { stats_reader }
    }

    pub fn plan(&self, queries: Vec<Query>) -> Vec<Plan> {
        fn plan_shared(
            preds: &HashSet<&Pred>,
            tables: &HashSet<Rc<FileTable>>,
            stats_reader: &StatsReader,
        ) -> Rc<PlanNode> {
            // For now, let's just push everything down and join the next cheapest table.
            let (join_graph, mut table_preds) = plan_utils::build_join_graph_and_table_preds(preds);

            for preds in table_preds.values_mut() {
                preds.sort_unstable_by_key(|pred| BenefitCostRatio::from(pred.est()));
                preds.reverse();
            }

            // Maps from table to PlanNode.
            let mut planned_nodes: HashMap<_, _> = tables
                .iter()
                .map(|table| {
                    (
                        table.clone(),
                        Rc::new(PlanNode::Base(BaseNode::new(table.clone()))),
                    )
                })
                .collect();

            for (table, preds) in table_preds {
                let node = planned_nodes.remove(&table).unwrap();
                let node = preds.into_iter().fold(node, |input, pred| {
                    Rc::new(PlanNode::Filter(FilterNode::new(input, pred.clone())))
                });
                planned_nodes.insert(table, node);
            }

            let mut planned_nodes: HashMap<_, _> = planned_nodes
                .into_iter()
                .map(|(table, node)| (table, Rc::new(RefCell::new(Some(node)))))
                .collect();

            let mut unjoined_links: Vec<_> = join_graph.values().flatten().collect();
            while !unjoined_links.is_empty() {
                unjoined_links.sort_unstable_by_key(|link| {
                    let left_ref = planned_nodes.get(&link.left.table()).unwrap().borrow();
                    let left_node = left_ref.as_ref().unwrap();
                    let right_ref = planned_nodes.get(&link.right.table()).unwrap().borrow();
                    let right_node = right_ref.as_ref().unwrap();
                    left_node.est_join(right_node, link, stats_reader)
                });

                let link = unjoined_links.swap_remove(0);
                let left_ref = planned_nodes.get(&link.left.table()).unwrap();
                let left_node = left_ref.take().unwrap();
                let right_node = planned_nodes
                    .get(&link.right.table())
                    .unwrap()
                    .take()
                    .unwrap();
                let right_tables = right_node.tables().into_owned();
                let node = Rc::new(PlanNode::Join(JoinNode::new(
                    left_node,
                    right_node,
                    link.clone(),
                    stats_reader,
                )));
                left_ref.replace(Some(node));
                let left_ref = left_ref.clone();
                for table in right_tables {
                    planned_nodes.insert(table, left_ref.clone());
                }
                unjoined_links = unjoined_links
                    .into_iter()
                    .filter(|link| {
                        let left_ref = planned_nodes.get(&link.left.table()).unwrap();
                        let right_ref = planned_nodes.get(&link.right.table()).unwrap();
                        !Rc::ptr_eq(left_ref, right_ref)
                    })
                    .collect();
            }

            planned_nodes.into_values().next().unwrap().take().unwrap()
        }

        fn plan_diff(
            query: &Query,
            shared_preds: &HashSet<&Pred>,
            shared_plan: Rc<PlanNode>,
        ) -> Plan {
            let query_preds = plan_utils::get_top_level_preds(query);
            let mut diff_preds: Vec<_> = (&query_preds - shared_preds).into_iter().collect();
            diff_preds.sort_unstable_by_key(|pred| BenefitCostRatio::from(pred.est()));
            diff_preds.reverse();
            let input = diff_preds.into_iter().fold(shared_plan, |input, pred| {
                Rc::new(PlanNode::Filter(FilterNode::new(input, pred.clone())))
            });

            Plan {
                root: ProjectNode {
                    input,
                    exprs: query.projection.clone(),
                },
                total_time_ms: RefCell::new(0),
            }
        }

        let (shared_preds, shared_tables) = plan_utils::find_shared(&queries);
        for query in &queries {
            //assert_eq!(
            //    HashSet::<&Rc<FileTable>>::from_iter(&query.from).len(),
            //    shared_tables.len()
            //);
            assert_eq!(query.from.len(), shared_tables.len());
        }
        let shared_plan = plan_shared(&shared_preds, &shared_tables, &self.stats_reader);

        let mut shared_plan = Rc::try_unwrap(shared_plan).unwrap();
        shared_plan.enable_cache();
        let shared_plan = Rc::new(shared_plan);

        queries
            .iter()
            .map(|query| plan_diff(query, &shared_preds, shared_plan.clone()))
            .collect()
    }
}

impl PlanNode {
    fn enable_cache(&mut self) {
        match self {
            Self::Base(base) => base.enable_cache(),
            Self::Filter(filter) => filter.enable_cache(),
            Self::Join(join) => join.enable_cache(),
        }
    }

    fn eval(&self) -> Either<Idx, Ref<'_, Idx>> {
        match self {
            Self::Base(base) => base.eval(),
            Self::Filter(filter) => filter.eval(),
            Self::Join(join) => join.eval(),
        }
    }

    fn tables(&self) -> Cow<'_, Vec<Rc<FileTable>>> {
        match self {
            Self::Join(join) => Cow::Borrowed(&join.tables),
            Self::Filter(filter) => Cow::Borrowed(&filter.tables),
            Self::Base(base) => Cow::Owned(vec![base.table.clone()]),
        }
    }

    // Will traverse in the DFS post ordering.
    fn iter(&self) -> PlanNodeIter<'_> {
        PlanNodeIter(vec![self])
    }

    fn est_len(&self) -> usize {
        match self {
            Self::Base(base) => base.est_len,
            Self::Filter(filter) => filter.est_len,
            Self::Join(join) => join.est_len,
        }
    }

    // Return the output estimated size of the join
    fn est_join(
        &self,
        other: &Self,
        constraint: &EqualityConstraint,
        stats_reader: &StatsReader,
    ) -> usize {
        let (left_num_unique, right_num_unique) = stats_reader.get_unique_counts(&constraint);
        ((self.est_len() * other.est_len()) as f64
            / std::cmp::max(left_num_unique, right_num_unique) as f64)
            .round() as usize
    }
}

impl Plan {
    pub fn eval(&self) -> DBResultSet {
        //println!("Evaling plan {:?}", self);
        let now = Instant::now();
        let out = self.root.eval();
        let time_ms = now.elapsed().as_millis();
        *self.total_time_ms.borrow_mut() = time_ms;
        out
    }
}

struct JoinNode {
    left: Rc<PlanNode>,
    right: Rc<PlanNode>,
    constraint: EqualityConstraint,
    tables: Vec<Rc<FileTable>>,
    est_len: usize,
    cache: Option<RefCell<Option<Idx>>>, // Outer option represents whether we want to cache
    stats: RefCell<Option<JoinNodeStats>>,
}

struct FilterNode {
    input: Rc<PlanNode>,
    pred: Pred,
    tables: Vec<Rc<FileTable>>,
    est_len: usize,
    cache: Option<RefCell<Option<Idx>>>, // Outer option represents whether we want to cache
    stats: RefCell<Option<FilterNodeStats>>,
}

struct BaseNode {
    table: Rc<FileTable>,
    est_len: usize,
    cache: Option<RefCell<Option<Idx>>>, // Outer option represents whether we want to cache
    stats: RefCell<Option<BaseNodeStats>>,
}

struct ProjectNode {
    input: Rc<PlanNode>,
    exprs: Vec<Expr>,
}

#[derive(Debug)]
struct JoinNodeStats {
    left_num: usize,
    right_num: usize,
    out_num: usize,
    time_ms: u128,
}

#[derive(Debug)]
struct FilterNodeStats {
    in_num: usize,
    out_num: usize,
    time_ms: u128,
}

#[derive(Debug)]
struct BaseNodeStats {
    out_num: usize,
    time_ms: u128,
}

impl JoinNode {
    fn new(
        left: Rc<PlanNode>,
        right: Rc<PlanNode>,
        constraint: EqualityConstraint,
        stats_reader: &StatsReader,
    ) -> Self {
        let tables = utils::append(left.tables().into_owned(), right.tables().into_owned());
        let est_len = left.est_join(&right, &constraint, stats_reader);
        Self {
            left,
            right,
            constraint,
            tables,
            est_len,
            cache: None,
            stats: RefCell::new(None),
        }
    }

    fn enable_cache(&mut self) {
        self.cache = Some(RefCell::new(None));
    }

    fn eval(&self) -> Either<Idx, Ref<'_, Idx>> {
        fn eval_inner(
            left: &PlanNode,
            right: &PlanNode,
            constraint: &EqualityConstraint,
        ) -> (Idx, JoinNodeStats) {
            let now = Instant::now();

            let left_idx = left.eval();
            let right_idx = right.eval();

            let left_idx = left_idx.as_ref().either(|idx| idx, |idx_ref| &*idx_ref);
            let right_idx = right_idx.as_ref().either(|idx| idx, |idx_ref| &*idx_ref);

            //println!("starting eval join {}", constraint);
            let out_idx = left_idx.join(right_idx, constraint);

            let time_ms = now.elapsed().as_millis();

            let stats = JoinNodeStats {
                left_num: left_idx.len(),
                right_num: right_idx.len(),
                out_num: out_idx.len(),
                time_ms,
            };

            //println!("constraint {} stats {:?}", constraint, stats);
            (out_idx, stats)
        }

        if let Some(cache) = &self.cache {
            {
                let mut cache = cache.borrow_mut();
                if cache.is_none() {
                    let (out, stats) = eval_inner(&self.left, &self.right, &self.constraint);
                    *self.stats.borrow_mut() = Some(stats);
                    *cache = Some(out);
                }
            }
            Either::Right(Ref::map(cache.borrow(), |idx| idx.as_ref().unwrap()))
        } else {
            let (out, stats) = eval_inner(&self.left, &self.right, &self.constraint);
            *self.stats.borrow_mut() = Some(stats);
            Either::Left(out)
        }

        //{
        //    let mut cache = self.cache.borrow_mut();
        //    if let None = cache.as_ref() {
        //        let left_idx = self.left.eval();
        //        let right_idx = self.right.eval();

        //        let left_idx = left_idx.either(|idx| &idx, |idx_ref| &idx_ref);
        //        let right_idx = right_idx.either(|idx| &idx, |idx_ref| &idx_ref);

        //        let now = Instant::now();
        //        *cache = Some(left_idx.join(right_idx, &self.constraint));
        //        let time_ms = now.elapsed().as_millis();

        //        EXEC_INFO.with(|exec_info| {
        //            if exec_info.borrow().node_stats {
        //                *self.stats.borrow_mut() = Some(JoinNodeStats {
        //                           left_num: left_idx.len(),
        //                    right_num: right_idx.len(),
        //                    out_num: cache.as_ref().unwrap().len(),
        //                    time_ms,
        //                });
        //            }
        //        });
        //    }
        //}
        //Ref::map(self.cache.borrow(), |idx| idx.as_ref().unwrap())
    }
}

impl FilterNode {
    fn new(input: Rc<PlanNode>, pred: Pred) -> Self {
        let est_len = (input.est_len() as f64 * pred.est().selectivity).round() as usize;
        let tables = input.tables().into_owned();
        Self {
            input,
            pred,
            est_len,
            tables,
            cache: None,
            stats: RefCell::new(None),
        }
    }

    fn enable_cache(&mut self) {
        self.cache = Some(RefCell::new(None));
    }

    fn eval(&self) -> Either<Idx, Ref<'_, Idx>> {
        fn eval_inner(input: &PlanNode, pred: &Pred) -> (Idx, FilterNodeStats) {
            let now = Instant::now();

            let in_idx = input.eval();

            let in_idx = in_idx.as_ref().either(|idx| idx, |idx_ref| &*idx_ref);

            //println!("starting eval filter {}", pred);
            let out_idx = in_idx.filter(&pred.eval(&EvalContext {
                idx: &ColIterable::Basic(in_idx),
                bmap: None,
                cache: None,
            }));

            let time_ms = now.elapsed().as_millis();

            let stats = FilterNodeStats {
                in_num: in_idx.len(),
                out_num: out_idx.len(),
                time_ms,
            };

            //println!("pred {} stats {:?}", pred, stats);
            (out_idx, stats)
        }

        if let Some(cache) = &self.cache {
            {
                let mut cache = cache.borrow_mut();
                if cache.is_none() {
                    let (out, stats) = eval_inner(&self.input, &self.pred);
                    *self.stats.borrow_mut() = Some(stats);
                    *cache = Some(out);
                }
            }
            Either::Right(Ref::map(cache.borrow(), |idx| idx.as_ref().unwrap()))
        } else {
            let (out, stats) = eval_inner(&self.input, &self.pred);
            *self.stats.borrow_mut() = Some(stats);
            Either::Left(out)
        }

        //{
        //    let mut cache = self.cache.borrow_mut();
        //    if let None = cache.as_ref() {
        //        let idx = self.input.eval();
        //        let now = Instant::now();
        //        *cache = Some(idx.filter(&self.pred.eval(&EvalContext {
        //            idx: &ColIterable::Basic(&idx),
        //            bmap: None,
        //            cache: None,
        //        })));
        //        let time_ms = now.elapsed().as_millis();

        //        EXEC_INFO.with(|exec_info| {
        //            if exec_info.borrow().node_stats {
        //                *self.stats.borrow_mut() = Some(FilterNodeStats {
        //                    in_num: idx.len(),
        //                    out_num: cache.as_ref().unwrap().len(),
        //                    time_ms,
        //                });
        //            }
        //        });
        //    }
        //}
        //Ref::map(self.cache.borrow(), |idx| idx.as_ref().unwrap())
    }
}

impl BaseNode {
    fn new(table: Rc<FileTable>) -> Self {
        let est_len = table.len();
        Self {
            table,
            est_len,
            cache: None,
            stats: RefCell::new(None),
        }
    }

    fn enable_cache(&mut self) {
        self.cache = Some(RefCell::new(None));
    }

    fn eval(&self) -> Either<Idx, Ref<'_, Idx>> {
        fn eval_inner(table: Rc<FileTable>) -> (Idx, BaseNodeStats) {
            //println!("starting eval base {}", table.name());

            let now = Instant::now();
            let idx = Idx::new(table.clone());
            let time_ms = now.elapsed().as_millis();

            let stats = BaseNodeStats {
                out_num: idx.len(),
                time_ms,
            };

            //println!("table {} stats {:?}", table.name(), stats);
            (idx, stats)
        }

        if let Some(cache) = &self.cache {
            {
                let mut cache = cache.borrow_mut();
                if cache.is_none() {
                    let (out, stats) = eval_inner(self.table.clone());
                    *self.stats.borrow_mut() = Some(stats);
                    *cache = Some(out);
                }
            }
            Either::Right(Ref::map(cache.borrow(), |idx| idx.as_ref().unwrap()))
        } else {
            let (out, stats) = eval_inner(self.table.clone());
            *self.stats.borrow_mut() = Some(stats);
            Either::Left(out)
        }

        //if let Some(cache) = &self.cache {
        //    {
        //        let mut cache = cache.borrow_mut();
        //        if let None = cache.as_ref() {
        //            let now = Instant::now();
        //            let idx = Idx::new(self.table.clone());
        //            let time_ms = now.elapsed().as_millis();

        //            EXEC_INFO.with(|exec_info| {
        //                if exec_info.borrow().node_stats {
        //                    *self.stats.borrow_mut() = Some(BaseNodeStats {
        //                        out_num: idx.len(),
        //                        time_ms,
        //                    });
        //                }
        //            });

        //            *cache = Some(idx);
        //        }
        //    }
        //    Either::Right(Ref::map(cache.borrow(), |idx| idx.as_ref().unwrap()))
        //} else {
        //    let now = Instant::now();
        //    let idx = Idx::new(self.table.clone());
        //    let time_ms = now.elapsed().as_millis();

        //    EXEC_INFO.with(|exec_info| {
        //        if exec_info.borrow().node_stats {
        //            *self.stats.borrow_mut() = Some(BaseNodeStats {
        //                out_num: idx.len(),
        //                time_ms,
        //            });
        //        }
        //    });

        //    Either::Left(idx)
        //}
    }
}

impl ProjectNode {
    fn eval(&self) -> DBResultSet {
        let idx = self.input.eval();
        let idx = idx.as_ref().either(|idx| idx, |idx_ref| &*idx_ref);
        DBResultSet::new(
            self.exprs
                .iter()
                .map(|expr| {
                    let vals = expr.eval(&EvalContext {
                        idx: &ColIterable::Basic(idx),
                        bmap: None,
                        cache: None,
                    });
                    vals.expect_right("We're not even using the cache...")
                })
                .collect(),
        )
    }
}

struct PlanNodeIter<'a>(Vec<&'a PlanNode>);

impl<'a> Iterator for PlanNodeIter<'a> {
    type Item = &'a PlanNode;

    fn next(&mut self) -> Option<Self::Item> {
        let node = self.0.pop();
        if let Some(PlanNode::Filter(filter)) = node {
            self.0.push(&filter.input);
        } else if let Some(PlanNode::Join(join)) = node {
            self.0.push(&join.left);
            self.0.push(&join.right);
        }
        node
    }
}

impl From<&Plan> for termtree::Tree<String> {
    fn from(plan: &Plan) -> Self {
        Self::new(format!(
            "Project({}) [total_time_ms = {}] ",
            plan.root
                .exprs
                .iter()
                .map(|expr| expr.to_string())
                .join(", "),
            plan.total_time_ms.borrow(),
        ))
        .with_leaves([plan.root.input.as_ref()])
    }
}

impl From<&PlanNode> for termtree::Tree<String> {
    fn from(node: &PlanNode) -> Self {
        match node {
            PlanNode::Base(base) => Self::new(format!(
                "Table({}, est_len = {}){}",
                base.table.name(),
                base.est_len,
                base.stats
                    .borrow()
                    .as_ref()
                    .map(|stats| format!(" [out = {}, time_ms = {}]", stats.out_num, stats.time_ms))
                    .unwrap_or("".to_string())
            )),
            PlanNode::Filter(filter) => Self::new(format!(
                "Filter({}, selectivity = {}, est_len = {}){}",
                filter.pred,
                filter.pred.est().selectivity,
                filter.est_len,
                filter
                    .stats
                    .borrow()
                    .as_ref()
                    .map(|stats| format!(
                        " [in = {}, out = {}, time_ms = {}]",
                        stats.in_num, stats.out_num, stats.time_ms
                    ))
                    .unwrap_or("".to_string())
            ))
            .with_leaves([filter.input.as_ref()]),
            PlanNode::Join(join) => Self::new(format!(
                "Join({}, est_len = {}){}",
                join.constraint,
                join.est_len,
                join.stats
                    .borrow()
                    .as_ref()
                    .map(|stats| format!(
                        " [left = {}, right = {}, out = {}, time_ms = {}]",
                        stats.left_num, stats.right_num, stats.out_num, stats.time_ms
                    ))
                    .unwrap_or("".to_string())
            ))
            .with_leaves([join.left.as_ref(), join.right.as_ref()]),
        }
    }
}

impl fmt::Debug for Plan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", termtree::Tree::from(self))
    }
}

impl fmt::Debug for PlanNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PlanNode")
    }
}
