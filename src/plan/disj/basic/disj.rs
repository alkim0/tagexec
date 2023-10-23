//! Basic planner for queries with a disjunctive root predicate. Basically treats each clause as a
//! separate query. Within each clause, the children are pushed down to the individual tables, then
//! the results of the clauses are unioned together at the top.
use super::core::BasicPlannerCore;
use crate::plan::disj::utils as plan_utils;
use crate::plan::{Plan, PlanState};
use crate::query::Query;
use crate::stats::StatsReader;
use either::Either;

pub struct BasicDisjPlanner<'a> {
    core: BasicPlannerCore<'a>,
}

impl<'a> BasicDisjPlanner<'a> {
    pub fn new(stats_reader: &'a StatsReader) -> Self {
        Self {
            core: BasicPlannerCore::new(stats_reader),
        }
    }

    pub fn plan(&self, query: &Query) -> Plan {
        let query_info = self.core.process_query(query);

        let node = if let Some(pred_root) = &query_info.pred_root {
            assert!(pred_root.is_or());

            self.core.make_union_node(
                pred_root
                    .try_iter_children()
                    .unwrap()
                    .map(|clause| {
                        let mut table_node_map = query_info.base_table_node_map.clone();

                        let pred_iter = if clause.is_atom() {
                            Either::Left(std::iter::once(clause.clone()))
                        } else {
                            assert!(clause.is_and());
                            Either::Right(clause.try_iter_children().unwrap().cloned())
                        };
                        plan_utils::insert_preds_to_table_node_map(
                            &mut table_node_map,
                            pred_iter,
                            |input, pred| self.core.make_filter_node(input, pred),
                            plan_utils::sort_preds_by_selectivity_cost_ratio,
                        );

                        self.core
                            .join_table_node_map(table_node_map, &query_info.join_graph)
                    })
                    .collect(),
            )
        } else {
            self.core
                .join_table_node_map(query_info.base_table_node_map, &query_info.join_graph)
        };

        Plan {
            root: self.core.make_project_node(node, &query.projection),
            state: PlanState {
                pred_root: query_info.pred_root,
            },
        }
    }
}
