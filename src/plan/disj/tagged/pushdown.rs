//! This contains the code for a very simple tagged planner, which simply just pushes down all the
//! predicates to the bottom and does a greedy join based on the results.
//!
//! Within a single table, the order of the predicates is arranged in decreasing specifying
//! benefit/cost first and increasing selectivity/cost second.
use super::core::TaggedPlannerCore;
use crate::plan::disj::utils as plan_utils;
use crate::plan::disj::{BENEFIT_EPSILON, COST_EPSILON, SELECTIVITY_EPSILON};
use crate::plan::{Plan, PlanState};
use crate::query::Query;
use crate::stats::StatsReader;
use float_ord::FloatOrd;
use itertools::Itertools;
use std::cmp::Reverse;

pub struct TaggedPushdownPlanner<'a> {
    pub(super) core: TaggedPlannerCore<'a>,
}

impl<'a> TaggedPushdownPlanner<'a> {
    pub fn new(stats_reader: &'a StatsReader) -> Self {
        Self {
            core: TaggedPlannerCore::new(stats_reader),
        }
    }

    pub fn plan(&self, query: &Query) -> Plan {
        let query_info = self.core.process_query(query);

        let mut table_node_map = query_info.base_table_node_map;

        if let Some(pred_root) = &query_info.pred_root {
            plan_utils::insert_preds_to_table_node_map(
                &mut table_node_map,
                pred_root.iter_leaves().unique().cloned(),
                |input, pred| self.core.make_filter_node(input, pred),
                |preds| {
                    let preds_clone = preds.clone();
                    preds.sort_unstable_by_key(|pred| {
                        let est = pred.est();
                        let spec_benefit = plan_utils::calc_spec_benefit(
                            pred,
                            preds_clone
                                .iter()
                                .filter(|&on_pred| on_pred != pred)
                                .map(|pred| (pred, 100.)),
                        );
                        (
                            Reverse(FloatOrd(
                                (spec_benefit + BENEFIT_EPSILON) / (est.cost + COST_EPSILON),
                            )),
                            FloatOrd(
                                (est.selectivity + SELECTIVITY_EPSILON) / (est.cost + COST_EPSILON),
                            ),
                        )
                    });
                },
            );
        }

        let node = self
            .core
            .join_table_node_map(table_node_map, &query_info.join_graph, None);

        Plan {
            root: self.core.make_project_node(node, &query.projection),
            state: PlanState {
                pred_root: query_info.pred_root,
            },
        }
    }
}
