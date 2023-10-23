//! A planner that extends `SimpleTaggedPlanner` by pulling up non-helpful predicates. This
//! predicate works by first pushing down all predicates to the table level. Then, in increasing
//! order of specifying benefit/cost, try pulling up the predicate to different levels to see if it
//! results in a better plan.
//!
//! Note that we only consider the movement of a single predicate at a time. After deciding to pull
//! up a predicate, it stays there. It does not come back down when considering the subsequent
//! predicates.
//!
//! The predicates pushed down to the table level are arranged in decreasing specifying benefit/cost.
use super::pushdown::TaggedPushdownPlanner;
use crate::plan::disj::utils as plan_utils;
use crate::plan::disj::{BENEFIT_EPSILON, COST_EPSILON};
use crate::plan::{FilterNode, Plan, PlanNode, PlanState};
use crate::pred::Pred;
use crate::query::Query;
use crate::stats::StatsReader;
use float_ord::FloatOrd;
use itertools::Itertools;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

pub struct TaggedPullupPlanner<'a> {
    pushdown_planner: TaggedPushdownPlanner<'a>,
}

impl<'a> TaggedPullupPlanner<'a> {
    pub fn new(stats_reader: &'a StatsReader) -> Self {
        Self {
            pushdown_planner: TaggedPushdownPlanner::new(stats_reader),
        }
    }

    pub fn plan(&self, query: &Query) -> Plan {
        fn is_matching_pred_node(node: &Rc<PlanNode>, pred: &Rc<Pred>) -> bool {
            match node.as_ref() {
                PlanNode::Filter(filter) => &filter.pred == pred,
                _ => false,
            }
        }

        let query_info = self.pushdown_planner.core.process_query(query);

        let pushdown_plan = self.pushdown_planner.plan(query);

        if let Some(pred_root) = &query_info.pred_root {
            let preds: HashSet<_> = pred_root.iter_leaves().cloned().collect();

            let mut best_tree = pushdown_plan.root.input;
            let pred_input_len_map: HashMap<_, _> = best_tree
                .find_pred_nodes(&preds)
                .into_iter()
                .map(|(pred, node)| {
                    let filter = <&FilterNode>::try_from(&*node).unwrap();
                    (pred, filter.input.est_len())
                })
                .collect();
            let preds = preds.into_iter().sorted_unstable_by_key(|pred| {
                let benefit = plan_utils::calc_spec_benefit(
                    &pred,
                    pred_input_len_map
                        .iter()
                        .filter(|(on_pred, _)| *on_pred != pred)
                        .map(|(on_pred, &input_len)| (on_pred, input_len)),
                );
                FloatOrd((benefit + BENEFIT_EPSILON) / (pred.est().cost + COST_EPSILON))
            });

            //println!(
            //    "{}",
            //    preds
            //        .clone()
            //        .map(|pred| {
            //            let benefit = plan_utils::calc_spec_benefit(
            //                &pred,
            //                pred_input_len_map
            //                    .iter()
            //                    .filter(|(on_pred, _)| *on_pred != &pred)
            //                    .map(|(on_pred, &input_len)| (on_pred, input_len)),
            //            );
            //            format!(
            //                "{} {} {} {}: {}",
            //                (benefit + BENEFIT_EPSILON) / (pred.est().cost + COST_EPSILON),
            //                benefit,
            //                pred.est().selectivity,
            //                pred.est().cost,
            //                pred
            //            )
            //        })
            //        .join("\n")
            //);

            for pred in preds {
                let mut cmp_tree = best_tree.clone();
                while !is_matching_pred_node(&cmp_tree, &pred) {
                    cmp_tree = self.pull_up_pred(&cmp_tree, &pred);
                    //println!("cmp_tree {} {}", cmp_tree.est_cum_cost(), cmp_tree);
                    //println!("best_tree {} {}", best_tree.est_cum_cost(), best_tree);
                    if cmp_tree.est_cum_cost() < best_tree.est_cum_cost() {
                        best_tree = cmp_tree.clone();
                    }
                }
            }

            Plan {
                root: self
                    .pushdown_planner
                    .core
                    .make_project_node(best_tree, &pushdown_plan.root.exprs),
                state: PlanState {
                    pred_root: query_info.pred_root,
                },
            }
        } else {
            pushdown_plan
        }
    }

    // Pulls up a predicate in a plan by one one node.
    fn pull_up_pred(&self, node: &Rc<PlanNode>, pred: &Rc<Pred>) -> Rc<PlanNode> {
        match node.as_ref() {
            PlanNode::Base(_) => panic!("Got to base node without finding pred"),
            PlanNode::Filter(filter) => match &*filter.input {
                PlanNode::Filter(input) if &input.pred == pred => {
                    self.pushdown_planner.core.make_filter_node(
                        self.pushdown_planner
                            .core
                            .make_filter_node(input.input.clone(), filter.pred.clone()),
                        pred.clone(),
                    )
                }
                _ => {
                    let modified = self.pull_up_pred(&filter.input, pred);
                    self.pushdown_planner
                        .core
                        .make_filter_node(modified, filter.pred.clone())
                }
            },
            PlanNode::Join(join) => match (&*join.left, &*join.right) {
                (PlanNode::Filter(left), _) if &left.pred == pred => {
                    self.pushdown_planner.core.make_filter_node(
                        self.pushdown_planner.core.make_join_node(
                            left.input.clone(),
                            join.right.clone(),
                            join.constraint.clone(),
                        ),
                        pred.clone(),
                    )
                }
                (_, PlanNode::Filter(right)) if &right.pred == pred => {
                    self.pushdown_planner.core.make_filter_node(
                        self.pushdown_planner.core.make_join_node(
                            join.left.clone(),
                            right.input.clone(),
                            join.constraint.clone(),
                        ),
                        pred.clone(),
                    )
                }
                (_, _) => {
                    if join.left.get_all_preds().contains(pred) {
                        let left = self.pull_up_pred(&join.left, pred);
                        self.pushdown_planner.core.make_join_node(
                            left,
                            join.right.clone(),
                            join.constraint.clone(),
                        )
                    } else {
                        let right = self.pull_up_pred(&join.right, pred);
                        self.pushdown_planner.core.make_join_node(
                            join.left.clone(),
                            right,
                            join.constraint.clone(),
                        )
                    }
                }
            },
            _ => panic!("Unexpected plan node"),
        }
    }
}
