use super::PlanNode;
use crate::cost::Cost;
use crate::expr::EqualityConstraint;
use crate::file_table::FileTableRef;
use std::collections::HashMap;
use std::rc::Rc;

mod basic;
mod tagged;
mod utils;

pub use basic::{
    BasicBaseNodeInner, BasicConjPlanner, BasicDisjPlanner, BasicFilterNodeInner,
    BasicJoinNodeInner, BasicNoOptPlanner, BasicProjectNodeInner, BasicUnionNodeInner,
};
pub use tagged::{
    PlanSpec, TaggedBaseNodeInner, TaggedCombinedPlanner, TaggedConjPushdownPlanner,
    TaggedFilterNodeInner, TaggedIterPushdownPlanner, TaggedJoinNodeInner, TaggedNoOptPlanner,
    TaggedProjectNodeInner, TaggedPullupPlanner, TaggedPushdownPlanner, TaggedSpecPlanner,
};

type JoinGraph = HashMap<Rc<FileTableRef>, Vec<EqualityConstraint>>;
type TableNodeMap = HashMap<Rc<FileTableRef>, Rc<PlanNode>>;

static SELECTIVITY_EPSILON: f64 = 1e-6;
static BENEFIT_EPSILON: f64 = 1e-6;
static COST_EPSILON: Cost = 1e-6;
