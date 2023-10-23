//! This contains the code for a tagged planner which allows the user to specify where to apply
//! predicates. By default, all non-specified predicates are pushed down to the table and the join
//! order is determined greedily.
//!
//! The predicate specification is done with a `HashMap` of `BTreeSet<String>` keys and
//! `Vec<String>` values. The key is the "location" in which to apply the predicates and is
//! specified by the set of table alias/names which must be joined before applying the predicate.
//! The predicate values are applied in the order that they appear (the earlier predicates are
//! applied closer to the tables).
//!
//! For non-specified predicates, within a single table, the order of the predicates is arranged
//! increasing selectivity/cost, regardless of where they are in the predicate expression.
//!
//! NOTE: The planner panics if the "location" is not an exact match (i.e., the join order is
//! arranged so that exact location cannot be found).
//!
//! TODO: Provide a way to specify join order in the spec as well.
use super::core::TaggedPlannerCore;
use crate::plan::disj::utils as plan_utils;
use crate::plan::disj::{COST_EPSILON, SELECTIVITY_EPSILON};
use crate::plan::{Plan, PlanNode, PlanState};
use crate::pred::Pred;
use crate::query::Query;
use crate::stats::StatsReader;
use float_ord::FloatOrd;
use split_iter::Splittable;
use std::collections::{BTreeSet, HashMap};
use std::rc::Rc;

pub struct PlanSpec {
    // The key is a vector of table ref aliases/names. These predicates will be applied directly on
    // top (in order) of this join node (after any conjunctively-applied predicates).
    pub pred_spec: HashMap<BTreeSet<String>, Vec<String>>,
}

pub struct TaggedSpecPlanner<'a> {
    core: TaggedPlannerCore<'a>,
}

impl<'a> TaggedSpecPlanner<'a> {
    pub fn new(stats_reader: &'a StatsReader) -> Self {
        Self {
            core: TaggedPlannerCore::new(stats_reader),
        }
    }

    pub fn plan(&self, query: &Query, spec: &PlanSpec) -> Plan {
        let query_info = self.core.process_query(query);

        let mut table_node_map = query_info.base_table_node_map;

        let pred_spec_map = query_info.pred_root.as_ref().map(|pred_root| {
            let spec_pred_strs: BTreeSet<_> = spec.pred_spec.values().flatten().cloned().collect();

            let (nonspec_preds, spec_preds) = pred_root
                .iter_leaves()
                .split(|pred| spec_pred_strs.contains(&pred.to_string()));

            plan_utils::insert_preds_to_table_node_map(
                &mut table_node_map,
                nonspec_preds.cloned(),
                |input, pred| self.core.make_filter_node(input, pred),
                |preds| {
                    preds.sort_unstable_by_key(|pred| {
                        let est = pred.est();
                        FloatOrd(
                            (est.selectivity + SELECTIVITY_EPSILON) / (est.cost + COST_EPSILON),
                        )
                    });
                },
            );

            let mut spec_preds: HashMap<_, _> = spec_preds
                .map(|pred| (pred.to_string(), pred.clone()))
                .collect();
            spec.pred_spec
                .iter()
                .map(|(loc, preds)| {
                    let preds: Vec<_> = preds
                        .iter()
                        .map(|pred| spec_preds.remove(pred).unwrap())
                        .collect();
                    (loc.clone(), preds)
                })
                .collect::<HashMap<_, _>>()
        });

        let mut node = self
            .core
            .join_table_node_map(table_node_map, &query_info.join_graph, None);

        if let Some(pred_spec_map) = pred_spec_map {
            for (loc, preds) in pred_spec_map {
                node = self.insert_spec_preds(&node, &loc, &preds);
            }
        }

        Plan {
            root: self.core.make_project_node(node, &query.projection),
            state: PlanState {
                pred_root: query_info.pred_root,
            },
        }
    }

    fn insert_spec_preds(
        &self,
        node: &Rc<PlanNode>,
        loc: &BTreeSet<String>,
        preds: &Vec<Rc<Pred>>,
    ) -> Rc<PlanNode> {
        let table_refs: BTreeSet<_> = node
            .get_all_table_refs()
            .into_iter()
            .map(|table_ref| table_ref.to_string())
            .collect();

        if &table_refs == loc {
            preds.iter().fold(node.clone(), |node, pred| {
                self.core.make_filter_node(node, pred.clone())
            })
        } else {
            match node.as_ref() {
                PlanNode::Base(_) => panic!("Got to base node without inserting"),
                PlanNode::Filter(filter) => {
                    let inserted = self.insert_spec_preds(&filter.input, loc, preds);
                    self.core.make_filter_node(inserted, filter.pred.clone())
                }
                PlanNode::Join(join) => {
                    let left_table_refs: BTreeSet<_> = join
                        .left
                        .get_all_table_refs()
                        .into_iter()
                        .map(|table_ref| table_ref.to_string())
                        .collect();
                    if left_table_refs.is_superset(loc) {
                        self.core.make_join_node(
                            self.insert_spec_preds(&join.left, loc, preds),
                            join.right.clone(),
                            join.constraint.clone(),
                        )
                    } else {
                        self.core.make_join_node(
                            join.left.clone(),
                            self.insert_spec_preds(&join.right, loc, preds),
                            join.constraint.clone(),
                        )
                    }
                }
                _ => panic!("Unexpected plan node"),
            }
        }
    }
}
