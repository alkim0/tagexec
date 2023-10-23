//! A pushdown planner similar to `TaggedPushdownPlanner`, but pushes down one predicate at a time
//! (and only if the resulting plan is better). This is used to try to combat cases in which it
//! would be better to keep certain predicates until the end.
use super::core::TaggedPlannerCore;
use crate::plan::disj::utils as plan_utils;
use crate::plan::disj::{BENEFIT_EPSILON, COST_EPSILON};
use crate::plan::{Plan, PlanState};
use crate::query::Query;
use crate::stats::StatsReader;
use crate::utils;
use float_ord::FloatOrd;
use itertools::Itertools;
use std::cmp::Reverse;

pub struct TaggedIterPushdownPlanner<'a> {
    core: TaggedPlannerCore<'a>,
}

impl<'a> TaggedIterPushdownPlanner<'a> {
    pub fn new(stats_reader: &'a StatsReader) -> Self {
        Self {
            core: TaggedPlannerCore::new(stats_reader),
        }
    }

    pub fn plan(&self, query: &Query) -> Plan {
        let query_info = self.core.process_query(query);

        let mut best_tree = self.core.join_table_node_map(
            query_info.base_table_node_map.clone(),
            &query_info.join_graph,
            None,
        );

        if let Some(pred_root) = &query_info.pred_root {
            let preds: Vec<_> = pred_root.iter_leaves().unique().cloned().collect();
            let preds_clone = preds.clone();
            let preds: Vec<_> = preds
                .into_iter()
                .sorted_unstable_by_key(|pred| {
                    let benefit = plan_utils::calc_spec_benefit(
                        &pred,
                        preds_clone
                            .iter()
                            .filter(|on_pred| *on_pred != pred)
                            .map(|on_pred| (on_pred, 100.)),
                    );
                    Reverse(FloatOrd(
                        (benefit + BENEFIT_EPSILON) / (pred.est().cost + COST_EPSILON),
                    ))
                })
                .collect();

            best_tree = preds.iter().fold(best_tree, |input, pred| {
                self.core.make_filter_node(input, pred.clone())
            });

            let mut table_node_map = query_info.base_table_node_map;
            for pred in &preds {
                let mut pushdown_table_node_map = table_node_map.clone();
                pushdown_table_node_map
                    .entry(utils::convert_to_one(pred.file_table_refs()))
                    .and_modify(|node| {
                        *node = self.core.make_filter_node(node.clone(), pred.clone())
                    });
                let node = self.core.join_table_node_map(
                    pushdown_table_node_map.clone(),
                    &query_info.join_graph,
                    None,
                );
                let used_preds = node.get_all_preds();
                let cmp_tree = preds
                    .iter()
                    .filter(|&pred| !used_preds.contains(pred))
                    .fold(node, |input, pred| {
                        self.core.make_filter_node(input, pred.clone())
                    });

                if cmp_tree.est_cum_cost() < best_tree.est_cum_cost() {
                    best_tree = cmp_tree;
                    table_node_map = pushdown_table_node_map;
                }
            }
        }

        Plan {
            root: self.core.make_project_node(best_tree, &query.projection),
            state: PlanState {
                pred_root: query_info.pred_root,
            },
        }
    }
}
