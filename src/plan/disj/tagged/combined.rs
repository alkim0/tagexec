/// A combined planner which tries pullup, iterative pushdown, and conjunctive pushdown, estimates
/// plans and picks the best one. Note that if the input predicate has more than one non-atomic
/// clause in conjunctive root form, `TaggedConjPushdownPlanner` is not considered as a possible
/// planner.
use super::{TaggedConjPushdownPlanner, TaggedIterPushdownPlanner, TaggedPullupPlanner};
use crate::plan::Plan;
use crate::query::Query;
use crate::stats::StatsReader;
use float_ord::FloatOrd;

pub struct TaggedCombinedPlanner<'a> {
    pullup_planner: TaggedPullupPlanner<'a>,
    iter_pushdown_planner: TaggedIterPushdownPlanner<'a>,
    conj_pushdown_planner: TaggedConjPushdownPlanner<'a>,
}

impl<'a> TaggedCombinedPlanner<'a> {
    pub fn new(stats_reader: &'a StatsReader) -> Self {
        Self {
            pullup_planner: TaggedPullupPlanner::new(stats_reader),
            iter_pushdown_planner: TaggedIterPushdownPlanner::new(stats_reader),
            conj_pushdown_planner: TaggedConjPushdownPlanner::new(stats_reader),
        }
    }

    pub fn plan(&self, query: &Query) -> Plan {
        let plans = vec![
            self.pullup_planner.plan(query),
            self.iter_pushdown_planner.plan(query),
            self.conj_pushdown_planner.plan(query),
        ];

        plans
            .into_iter()
            .min_by_key(|plan| FloatOrd(plan.est_cost()))
            .unwrap()
    }
}
