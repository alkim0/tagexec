//! A tagged pushdown planner which operates similar to `BasicConjPlanner`. It expects a
//! conjunctive predicate root. All atomic clauses are pushed down to to their individual tables,
//! and the disjunctive clauses are performed after the joins. We can measure the "overhead" of
//! tagged execution by comparing the results of this planner with `BasicConjPlanner`.
use super::core::TaggedPlannerCore;
use crate::plan::disj::utils as plan_utils;
use crate::plan::{Plan, PlanState};
use crate::pred::Pred;
use crate::query::Query;
use crate::stats::StatsReader;
use crate::utils;

pub struct TaggedConjPushdownPlanner<'a> {
    core: TaggedPlannerCore<'a>,
}

impl<'a> TaggedConjPushdownPlanner<'a> {
    pub fn new(stats_reader: &'a StatsReader) -> Self {
        Self {
            core: TaggedPlannerCore::new(stats_reader),
        }
    }

    pub fn plan(&self, query: &Query) -> Plan {
        let query_info = self.core.process_query(query);

        let mut table_node_map = query_info.base_table_node_map;

        let disj_clause = if let Some(pred_root) = &query_info.pred_root {
            let (atomic_clauses, disj_clauses) = plan_utils::split_conj_root(pred_root);

            plan_utils::insert_preds_to_table_node_map(
                &mut table_node_map,
                atomic_clauses,
                |input, pred| self.core.make_filter_node(input, pred),
                plan_utils::sort_preds_by_selectivity_cost_ratio,
            );

            if disj_clauses.is_empty() {
                None
            } else if disj_clauses.len() == 1 {
                Some(utils::convert_to_one(disj_clauses))
            } else {
                Some(Pred::new_and(disj_clauses))
            }
        } else {
            None
        };

        let mut node = self
            .core
            .join_table_node_map(table_node_map, &query_info.join_graph, None);

        if let Some(clause) = disj_clause {
            node = self.core.make_filter_node(node, clause);
        }

        Plan {
            root: self.core.make_project_node(node, &query.projection),
            state: PlanState {
                pred_root: query_info.pred_root,
            },
        }
    }
}
