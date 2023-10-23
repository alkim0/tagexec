//! A tagged planner that does no pushing down.
use super::core::TaggedPlannerCore;
use crate::plan::{Plan, PlanState};
use crate::query::Query;
use crate::stats::StatsReader;

pub struct TaggedNoOptPlanner<'a> {
    pub(super) core: TaggedPlannerCore<'a>,
}

impl<'a> TaggedNoOptPlanner<'a> {
    pub fn new(stats_reader: &'a StatsReader) -> Self {
        Self {
            core: TaggedPlannerCore::new(stats_reader),
        }
    }

    pub fn plan(&self, query: &Query) -> Plan {
        let query_info = self.core.process_query(query);

        let mut table_node_map = query_info.base_table_node_map;

        let mut node = self
            .core
            .join_table_node_map(table_node_map, &query_info.join_graph, None);

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
