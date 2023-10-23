/// Basic planner which simply performs all the predicates at the end. This would be done for
/// predicates which can't be pushed down. Doesn't assume anything about the structure of the
/// predicate.
use super::core::BasicPlannerCore;
use crate::plan::{Plan, PlanState};
use crate::query::Query;
use crate::stats::StatsReader;

pub struct BasicNoOptPlanner<'a> {
    core: BasicPlannerCore<'a>,
}

impl<'a> BasicNoOptPlanner<'a> {
    pub fn new(stats_reader: &'a StatsReader) -> Self {
        Self {
            core: BasicPlannerCore::new(stats_reader),
        }
    }

    pub fn plan(&self, query: &Query) -> Plan {
        let query_info = self.core.process_query(query);

        let mut node = self
            .core
            .join_table_node_map(query_info.base_table_node_map, &query_info.join_graph);

        if let Some(pred_root) = &query_info.pred_root {
            node = self.core.make_filter_node(node, pred_root.clone());
        }

        Plan {
            root: self.core.make_project_node(node, &query.projection),
            state: PlanState {
                pred_root: query_info.pred_root,
            },
        }
    }
}
