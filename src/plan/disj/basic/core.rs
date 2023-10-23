//! Core for basic planners.
use crate::bitmap::BitmapInt;
use crate::cost::{cost_factors, Cost};
use crate::db::DBResultSet;
use crate::engine::EXEC_INFO;
use crate::expr::{EqualityConstraint, EvalContext, Expr};
use crate::file_table::{FileCol, FileTableRef};
use crate::idx::{BasicIdx, Idx};
use crate::plan::disj::utils as plan_utils;
use crate::plan::disj::{JoinGraph, TableNodeMap};
use crate::plan::{
    BaseNode, BaseNodeInner, FilterNode, FilterNodeInner, JoinNode, JoinNodeInner, PlanNode,
    ProjectNode, ProjectNodeInner, UnionNode, UnionNodeInner,
};
use crate::pred::Pred;
use crate::query::Query;
use crate::stats::StatsReader;
use float_ord::FloatOrd;
use log::info;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;
use std::time::Instant;

pub(super) struct BasicPlannerCore<'a> {
    stats_reader: &'a StatsReader,
}

pub(super) struct ProcessedQueryInfo {
    pub join_graph: JoinGraph,
    pub pred_root: Option<Rc<Pred>>,
    pub base_table_node_map: TableNodeMap,
}

impl<'a> BasicPlannerCore<'a> {
    pub fn new(stats_reader: &'a StatsReader) -> Self {
        Self { stats_reader }
    }

    pub fn process_query(&self, query: &Query) -> ProcessedQueryInfo {
        let base_table_node_map = query
            .from
            .iter()
            .map(|table_ref| (table_ref.clone(), self.make_base_node(table_ref.clone())))
            .collect();
        let (join_graph, pred_root) = query
            .filter
            .as_ref()
            .map(|filter| plan_utils::split_join_constraints(filter))
            .unwrap_or((HashMap::new(), None));

        ProcessedQueryInfo {
            join_graph,
            pred_root,
            base_table_node_map,
        }
    }

    pub fn join_table_node_map(
        &self,
        table_node_map: TableNodeMap,
        join_graph: &JoinGraph,
    ) -> Rc<PlanNode> {
        plan_utils::plan_greedy_join(
            join_graph.values().flatten().collect(),
            table_node_map,
            |left_node, right_node, constraint| {
                self.make_join_node(left_node, right_node, constraint)
            },
        )
    }

    pub fn make_base_node(&self, table_ref: Rc<FileTableRef>) -> Rc<PlanNode> {
        let inner = BaseNodeInner::Basic(BasicBaseNodeInner::new(&table_ref, &self.stats_reader));
        Rc::new(PlanNode::Base(BaseNode::new(table_ref, inner)))
    }

    pub fn make_filter_node(&self, input: Rc<PlanNode>, pred: Rc<Pred>) -> Rc<PlanNode> {
        let inner =
            FilterNodeInner::Basic(BasicFilterNodeInner::new(&input, &pred, &self.stats_reader));
        Rc::new(PlanNode::Filter(FilterNode::new(input, pred, inner)))
    }

    pub fn make_join_node(
        &self,
        left_node: Rc<PlanNode>,
        right_node: Rc<PlanNode>,
        constraint: EqualityConstraint,
    ) -> Rc<PlanNode> {
        let inner = JoinNodeInner::Basic(BasicJoinNodeInner::new(
            &left_node,
            &right_node,
            &constraint,
            &self.stats_reader,
        ));
        Rc::new(PlanNode::Join(JoinNode::new(
            left_node, right_node, constraint, inner,
        )))
    }

    pub fn make_union_node(&self, inputs: Vec<Rc<PlanNode>>) -> Rc<PlanNode> {
        let inner = UnionNodeInner::Basic(BasicUnionNodeInner::new(&inputs));
        Rc::new(PlanNode::Union(UnionNode::new(inputs, inner)))
    }

    pub fn make_project_node(&self, input: Rc<PlanNode>, exprs: &Vec<Expr>) -> ProjectNode {
        ProjectNode {
            input,
            exprs: exprs.clone(),
            inner: ProjectNodeInner::Basic(BasicProjectNodeInner),
        }
    }
}

struct EstInfo {
    est_len: f64,
    est_cum_cost: Cost,
    est_cost: Cost,
}

struct RunInfo {
    eval_time_ms: u128,
    num_output: usize,
    was_run: bool,
}

pub struct BasicBaseNodeInner {
    est_info: EstInfo,
    run_info: RefCell<RunInfo>,
}

pub struct BasicFilterNodeInner {
    est_info: EstInfo,
    run_info: RefCell<RunInfo>,
}

pub struct BasicJoinNodeInner {
    // When evaluating, if this is true, we build a hash map based on the left node's values.
    // Otherwise, we build a hash map based on the right node's values.
    left_based: bool,

    est_info: EstInfo,
    run_info: RefCell<RunInfo>,
}

pub struct BasicUnionNodeInner {
    est_info: EstInfo,
    run_info: RefCell<RunInfo>,
}

pub struct BasicProjectNodeInner;

impl BasicBaseNodeInner {
    pub(in crate::plan) fn new(table_ref: &Rc<FileTableRef>, stats_reader: &StatsReader) -> Self {
        Self {
            est_info: EstInfo {
                est_len: table_ref.table.len() as f64,
                est_cum_cost: 0.,
                est_cost: 0.,
            },
            run_info: RefCell::new(RunInfo {
                eval_time_ms: 0,
                num_output: 0,
                was_run: false,
            }),
        }
    }

    pub fn est_cum_cost(&self) -> Cost {
        self.est_info.est_cum_cost
    }

    pub fn est_cost(&self) -> Cost {
        self.est_info.est_cost
    }

    pub fn est_len(&self) -> f64 {
        self.est_info.est_len
    }

    pub(in crate::plan) fn est_unique(
        &self,
        table_ref: &Rc<FileTableRef>,
        col: &Rc<FileCol>,
        node: &BaseNode,
        stats_reader: &StatsReader,
    ) -> f64 {
        stats_reader.get_num_unique(col) as f64
    }

    pub(in crate::plan) fn eval(&self, node: &BaseNode) -> Idx {
        let now = Instant::now();
        let out = Idx::Basic(BasicIdx::new(node.table_ref.clone()));
        let mut run_info = self.run_info.borrow_mut();
        run_info.eval_time_ms = now.elapsed().as_millis();
        run_info.num_output = out.len();
        run_info.was_run = true;
        out
    }
}

impl BasicFilterNodeInner {
    pub(in crate::plan) fn new(
        input: &Rc<PlanNode>,
        pred: &Rc<Pred>,
        stats_reader: &StatsReader,
    ) -> Self {
        let input_len = input.est_len();
        let pred_est = pred.est();
        let est_len = input_len as f64 * pred_est.selectivity;
        let cost = input_len as f64 * pred_est.cost * cost_factors::FILTER_COST_FACTOR;

        Self {
            est_info: EstInfo {
                est_len,
                est_cum_cost: input.est_cum_cost() + cost,
                est_cost: cost,
            },
            run_info: RefCell::new(RunInfo {
                eval_time_ms: 0,
                num_output: 0,
                was_run: false,
            }),
        }
    }

    pub fn est_cum_cost(&self) -> Cost {
        self.est_info.est_cum_cost
    }

    pub fn est_cost(&self) -> Cost {
        self.est_info.est_cost
    }

    pub fn est_len(&self) -> f64 {
        self.est_info.est_len
    }

    pub(in crate::plan) fn est_unique(
        &self,
        table_ref: &Rc<FileTableRef>,
        col: &Rc<FileCol>,
        node: &FilterNode,
        stats_reader: &StatsReader,
    ) -> f64 {
        // XXX: This assumes predicate has no effect on the unique value. However, it obviously
        // can't be greater than the total size, so it must be smaller than that. This obviously
        // isn't super accurate (especially if the predicate is applied directly on the column we
        // are estimating the unique size of). Making this estimation is left as a todo.
        std::cmp::min(
            FloatOrd(node.input.est_unique(table_ref, col, stats_reader)),
            FloatOrd(self.est_info.est_len),
        )
        .0
    }

    pub(in crate::plan) fn eval(&self, node: &FilterNode) -> Idx {
        let in_idx = node.input.eval();

        let now = Instant::now();
        let result = node.pred.eval(&EvalContext {
            idx: &in_idx,
            bmap: None,
        });
        let pred_eval_time_ms = now.elapsed().as_millis();

        EXEC_INFO.with(|exec_info| {
            let mut exec_info = exec_info.borrow_mut();
            exec_info.stats.pred_eval_time_ms += pred_eval_time_ms;
            exec_info.stats.num_filter_tuples += in_idx.len() as u128;
        });

        if EXEC_INFO.with(|exec_info| exec_info.borrow().debug_times) {
            info!(
                "Filter {} eval time {} ms, num vals evaled {}",
                node.pred,
                pred_eval_time_ms,
                in_idx.len(),
            );
        }

        let in_idx = <&BasicIdx>::try_from(&in_idx).unwrap();
        let out = Idx::Basic(in_idx.filter(&result));

        if EXEC_INFO.with(|exec_info| exec_info.borrow().debug_times) {
            info!(
                "Filter {} total time {} ms",
                node.pred,
                now.elapsed().as_millis()
            );
        }

        let mut run_info = self.run_info.borrow_mut();
        run_info.eval_time_ms = now.elapsed().as_millis();
        run_info.num_output = out.len();
        run_info.was_run = true;

        out
    }
}

impl BasicJoinNodeInner {
    pub(in crate::plan) fn new(
        left: &Rc<PlanNode>,
        right: &Rc<PlanNode>,
        constraint: &EqualityConstraint,
        stats_reader: &StatsReader,
    ) -> Self {
        let left_len = left.est_len();
        let right_len = right.est_len();
        let left_num_unique = left.est_unique(
            &constraint.left_table_ref,
            &constraint.left_col,
            stats_reader,
        );
        let right_num_unique = right.est_unique(
            &constraint.right_table_ref,
            &constraint.right_col,
            stats_reader,
        );
        let unique_factor = std::cmp::max(FloatOrd(left_num_unique), FloatOrd(right_num_unique)).0;
        let est_len = (left_len * right_len) as f64 / unique_factor;

        let calc_est_cost = |left_len, right_len, left_num_unique: f64| {
            // NOTE: This doesn't include the cost of allocating the space for the values of the
            // hash map.
            // As a special case, if one side has 0 tuples, then the cost of the join is 0
            if left_len == 0. {
                return 0.;
            }

            let hash_map_build_cost = left_len as f64
                * constraint.left_col.data_type().size() as f64
                * cost_factors::HASH_COST_FACTOR
                + left_num_unique as f64
                    * constraint.left_col.data_type().size() as f64
                    * cost_factors::MEM_ALLOC_FACTOR;
            // For each tagged slice from the right side, there must be at least once tagged slice
            // on the left side which when combined does not result in None, so we must hash every
            // value on the right side.
            let hash_lookup_cost = right_len as f64
                * constraint.right_col.data_type().size() as f64
                * cost_factors::HASH_COST_FACTOR;
            let build_idx_cost = ((left.get_all_table_refs().len()
                + right.get_all_table_refs().len())
                * std::mem::size_of::<BitmapInt>()) as f64
                * est_len
                * cost_factors::MEM_ALLOC_FACTOR;

            hash_map_build_cost + hash_lookup_cost + build_idx_cost
        };

        let left_cost = calc_est_cost(left_len, right_len, left_num_unique);
        let right_cost = calc_est_cost(right_len, left_len, right_num_unique);
        let (left_based, est_cost) = if left_cost <= right_cost {
            (true, left_cost)
        } else {
            (false, right_cost)
        };

        Self {
            left_based,
            est_info: EstInfo {
                est_len,
                est_cum_cost: left.est_cum_cost() + right.est_cum_cost() + est_cost,
                est_cost,
            },
            run_info: RefCell::new(RunInfo {
                eval_time_ms: 0,
                num_output: 0,
                was_run: false,
            }),
        }
    }

    pub fn est_cum_cost(&self) -> Cost {
        self.est_info.est_cum_cost
    }

    pub fn est_cost(&self) -> Cost {
        self.est_info.est_cost
    }

    pub fn est_len(&self) -> f64 {
        self.est_info.est_len
    }

    pub(in crate::plan) fn est_unique(
        &self,
        table_ref: &Rc<FileTableRef>,
        col: &Rc<FileCol>,
        node: &JoinNode,
        stats_reader: &StatsReader,
    ) -> f64 {
        // XXX: The following estimates assumes inner joins
        let num_unique = if (&node.constraint.left_table_ref == table_ref
            && &node.constraint.left_col == col)
            || (&node.constraint.right_table_ref == table_ref && &node.constraint.right_col == col)
        {
            let left_num_unique = node.left.est_unique(
                &node.constraint.left_table_ref,
                &node.constraint.left_col,
                stats_reader,
            );
            let right_num_unique = node.right.est_unique(
                &node.constraint.right_table_ref,
                &node.constraint.right_col,
                stats_reader,
            );
            std::cmp::min(FloatOrd(left_num_unique), FloatOrd(right_num_unique)).0
        } else {
            if node.left.get_all_table_refs().contains(table_ref) {
                node.left.est_unique(table_ref, col, stats_reader)
            } else {
                node.right.est_unique(table_ref, col, stats_reader)
            }
        };

        std::cmp::min(FloatOrd(num_unique), FloatOrd(self.est_info.est_len)).0
    }

    pub(in crate::plan) fn eval(&self, node: &JoinNode) -> Idx {
        let left_idx = node.left.eval();
        let right_idx = node.right.eval();

        let left_idx = <&BasicIdx>::try_from(&left_idx).unwrap();
        let right_idx = <&BasicIdx>::try_from(&right_idx).unwrap();

        //println!("------ For join node (basic) {} ---------", node.constraint);
        //println!("left_idx {:?}", left_idx);
        //println!("right_idx {:?}", right_idx);

        let now = Instant::now();
        let out = Idx::Basic(left_idx.join(right_idx, &node.constraint, self.left_based));

        if EXEC_INFO.with(|exec_info| exec_info.borrow().debug_times) {
            info!(
                "Join {} total time {} ms",
                node.constraint,
                now.elapsed().as_millis()
            );
        }

        let mut run_info = self.run_info.borrow_mut();
        run_info.eval_time_ms = now.elapsed().as_millis();
        run_info.num_output = out.len();
        run_info.was_run = true;

        //println!("joined_idx {:?}", out);
        out
    }
}

impl BasicUnionNodeInner {
    fn new(inputs: &Vec<Rc<PlanNode>>) -> Self {
        let est_len = inputs.iter().map(|node| node.est_len()).sum();
        // XXX: This cost is never actually used for anything, so let's just set it to (n log n)
        // for now.
        let est_cost = (est_len as f64 * (est_len as f64).log2()) as Cost;
        let est_cum_cost = inputs.iter().map(|node| node.est_cum_cost()).sum::<f64>() + est_cost;
        Self {
            est_info: EstInfo {
                est_len,
                est_cum_cost,
                est_cost,
            },
            run_info: RefCell::new(RunInfo {
                eval_time_ms: 0,
                num_output: 0,
                was_run: false,
            }),
        }
    }

    pub fn est_cum_cost(&self) -> Cost {
        self.est_info.est_cum_cost
    }

    pub fn est_cost(&self) -> Cost {
        self.est_info.est_cost
    }

    pub fn est_len(&self) -> f64 {
        self.est_info.est_len
    }

    pub(in crate::plan) fn est_unique(
        &self,
        table_ref: &Rc<FileTableRef>,
        col: &Rc<FileCol>,
        node: &UnionNode,
        stats_reader: &StatsReader,
    ) -> f64 {
        unimplemented!();
    }

    pub(in crate::plan) fn eval(&self, node: &UnionNode) -> Idx {
        let in_idxs: Vec<_> = node.inputs.iter().map(|input| input.eval()).collect();

        let now = Instant::now();
        let in_idxs: Vec<_> = in_idxs
            .iter()
            .map(|idx| <&BasicIdx>::try_from(idx).unwrap())
            .collect();
        let num_in_rows: u128 = in_idxs.iter().map(|idx| idx.len() as u128).sum();
        let out_idx = Idx::Basic(BasicIdx::union(in_idxs));
        let eval_time_ms = now.elapsed().as_millis();

        EXEC_INFO.with(|exec_info| {
            let mut exec_info = exec_info.borrow_mut();
            exec_info.stats.union_time_ms += eval_time_ms;
            exec_info.stats.num_union_tuples += num_in_rows;
        });

        if EXEC_INFO.with(|exec_info| exec_info.borrow().debug_times) {
            info!("Union eval time {} ms", eval_time_ms);
        }

        let mut run_info = self.run_info.borrow_mut();
        run_info.eval_time_ms = eval_time_ms;
        run_info.num_output = out_idx.len();
        run_info.was_run = true;

        out_idx
    }
}

impl BasicProjectNodeInner {
    pub(in crate::plan) fn eval(&self, node: &ProjectNode) -> DBResultSet {
        let idx = node.input.eval();
        DBResultSet::new(
            node.exprs
                .iter()
                .map(|expr| {
                    let vals = expr.eval(&EvalContext {
                        idx: &idx,
                        bmap: None,
                    });
                    vals
                })
                .collect(),
        )
    }
}

impl fmt::Display for BasicBaseNodeInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let run_info = self.run_info.borrow();
        write!(
            f,
            "est_len={}{}",
            self.est_info.est_len,
            run_info
                .was_run
                .then(|| format!(" {}", run_info))
                .unwrap_or("".to_string())
        )
    }
}

impl fmt::Display for BasicFilterNodeInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let run_info = self.run_info.borrow();
        write!(
            f,
            "est_len={}{}",
            self.est_info.est_len,
            run_info
                .was_run
                .then(|| format!(" {}", run_info))
                .unwrap_or("".to_string())
        )
    }
}

impl fmt::Display for BasicJoinNodeInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let run_info = self.run_info.borrow();
        write!(
            f,
            "est_len={}{}",
            self.est_info.est_len,
            run_info
                .was_run
                .then(|| format!(" {}", run_info))
                .unwrap_or("".to_string())
        )
    }
}

impl fmt::Display for BasicUnionNodeInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let run_info = self.run_info.borrow();
        write!(
            f,
            "est_len={}{}",
            self.est_info.est_len,
            run_info
                .was_run
                .then(|| format!(" {}", run_info))
                .unwrap_or("".to_string())
        )
    }
}

impl fmt::Display for RunInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[num_output={}, eval_time_ms={}]",
            self.num_output, self.eval_time_ms
        )
    }
}
