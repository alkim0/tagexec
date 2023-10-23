use crate::cost::Cost;
use crate::db::DBResultSet;
use crate::expr::{EqualityConstraint, Expr};
use crate::file_table::{FileCol, FileTableRef};
use crate::idx::Idx;
use crate::pred::Pred;
use crate::stats::StatsReader;
use itertools::Itertools;
use snowflake::ProcessUniqueId;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::rc::Rc;

mod disj;

use disj::{
    BasicBaseNodeInner, BasicFilterNodeInner, BasicJoinNodeInner, BasicProjectNodeInner,
    BasicUnionNodeInner, TaggedBaseNodeInner, TaggedFilterNodeInner, TaggedJoinNodeInner,
    TaggedProjectNodeInner,
};

pub use disj::{
    BasicConjPlanner, BasicDisjPlanner, BasicNoOptPlanner, PlanSpec, TaggedCombinedPlanner,
    TaggedConjPushdownPlanner, TaggedIterPushdownPlanner, TaggedNoOptPlanner, TaggedPullupPlanner,
    TaggedPushdownPlanner, TaggedSpecPlanner,
};

type NodeId = ProcessUniqueId;

pub struct Plan {
    root: ProjectNode,
    state: PlanState,
}

struct PlanState {
    // Planners often need the root predicate to be persist throughout the plan execution so we
    // hold a pointer here.
    pred_root: Option<Rc<Pred>>,
}

enum PlanNode {
    Join(JoinNode),
    Filter(FilterNode),
    Base(BaseNode),
    Union(UnionNode),
}

struct JoinNode {
    id: NodeId,
    left: Rc<PlanNode>,
    right: Rc<PlanNode>,
    constraint: EqualityConstraint,
    inner: JoinNodeInner,
}

struct FilterNode {
    id: NodeId,
    input: Rc<PlanNode>,
    pred: Rc<Pred>,
    inner: FilterNodeInner,
}

struct BaseNode {
    id: NodeId,
    table_ref: Rc<FileTableRef>,
    inner: BaseNodeInner,
}

struct UnionNode {
    id: NodeId,
    inputs: Vec<Rc<PlanNode>>,
    inner: UnionNodeInner,
}

struct ProjectNode {
    input: Rc<PlanNode>,
    exprs: Vec<Expr>,
    inner: ProjectNodeInner,
}

enum JoinNodeInner {
    Basic(BasicJoinNodeInner),
    Tagged(TaggedJoinNodeInner),
}

enum BaseNodeInner {
    Basic(BasicBaseNodeInner),
    Tagged(TaggedBaseNodeInner),
}

enum FilterNodeInner {
    Basic(BasicFilterNodeInner),
    Tagged(TaggedFilterNodeInner),
}

enum ProjectNodeInner {
    Basic(BasicProjectNodeInner),
    Tagged(TaggedProjectNodeInner),
}

enum UnionNodeInner {
    Basic(BasicUnionNodeInner),
}

impl Plan {
    pub fn eval(&self) -> DBResultSet {
        self.root.eval()
    }

    pub fn est_cost(&self) -> Cost {
        self.root.input.est_cum_cost()
    }

    //pub fn print_est_vs_real_cost(&self) {
    //    self.root.input.print_est_vs_real_cost();
    //}
}

impl PlanNode {
    fn eval(&self) -> Idx {
        match self {
            Self::Base(base) => match &base.inner {
                BaseNodeInner::Basic(inner) => inner.eval(base),
                BaseNodeInner::Tagged(inner) => inner.eval(base),
            },
            Self::Filter(filter) => match &filter.inner {
                FilterNodeInner::Basic(inner) => inner.eval(filter),
                FilterNodeInner::Tagged(inner) => inner.eval(filter),
            },
            Self::Join(join) => match &join.inner {
                JoinNodeInner::Basic(inner) => inner.eval(join),
                JoinNodeInner::Tagged(inner) => inner.eval(join),
            },
            Self::Union(union) => match &union.inner {
                UnionNodeInner::Basic(basic) => basic.eval(union),
            },
        }
    }

    fn est_len(&self) -> f64 {
        match self {
            Self::Base(base) => match &base.inner {
                BaseNodeInner::Basic(inner) => inner.est_len(),
                BaseNodeInner::Tagged(inner) => inner.est_len(),
            },
            Self::Filter(filter) => match &filter.inner {
                FilterNodeInner::Basic(inner) => inner.est_len(),
                FilterNodeInner::Tagged(inner) => inner.est_len(),
            },
            Self::Join(join) => match &join.inner {
                JoinNodeInner::Basic(inner) => inner.est_len(),
                JoinNodeInner::Tagged(inner) => inner.est_len(),
            },
            Self::Union(union) => match &union.inner {
                UnionNodeInner::Basic(basic) => basic.est_len(),
            },
        }
    }

    fn est_unique(
        &self,
        table_ref: &Rc<FileTableRef>,
        col: &Rc<FileCol>,
        stats_reader: &StatsReader,
    ) -> f64 {
        match self {
            Self::Base(base) => match &base.inner {
                BaseNodeInner::Basic(inner) => inner.est_unique(table_ref, col, base, stats_reader),
                BaseNodeInner::Tagged(inner) => {
                    inner.est_unique(table_ref, col, base, stats_reader)
                }
            },
            Self::Filter(filter) => match &filter.inner {
                FilterNodeInner::Basic(inner) => {
                    inner.est_unique(table_ref, col, filter, stats_reader)
                }
                FilterNodeInner::Tagged(inner) => {
                    inner.est_unique(table_ref, col, filter, stats_reader)
                }
            },
            Self::Join(join) => match &join.inner {
                JoinNodeInner::Basic(inner) => inner.est_unique(table_ref, col, join, stats_reader),
                JoinNodeInner::Tagged(inner) => {
                    inner.est_unique(table_ref, col, join, stats_reader)
                }
            },
            Self::Union(union) => match &union.inner {
                UnionNodeInner::Basic(basic) => {
                    basic.est_unique(table_ref, col, union, stats_reader)
                }
            },
        }
    }

    //fn table_refs(&self) -> Vec<Rc<FileTableRef>> {
    //    match self {
    //        Self::Base(base) => base.table_refs(),
    //        Self::Filter(filter) => filter.table_refs(),
    //        Self::Join(join) => join.table_refs(),
    //    }
    //}

    //fn est_join(
    //    &self,
    //    other: &Self,
    //    constraint: &EqualityConstraint,
    //    stats_reader: &StatsReader,
    //) -> usize {
    //    let (left_num_unique, right_num_unique) = stats_reader.get_unique_counts(constraint);
    //    ((self.est_len(stats_reader) * other.est_len(stats_reader)) as f64
    //        / std::cmp::max(left_num_unique, right_num_unique) as f64)
    //        .round() as usize
    //}

    /// Estimates the cost of this node.
    fn est_cost(&self) -> Cost {
        match self {
            Self::Base(base) => match &base.inner {
                BaseNodeInner::Basic(inner) => inner.est_cost(),
                BaseNodeInner::Tagged(inner) => inner.est_cost(),
            },
            Self::Filter(filter) => match &filter.inner {
                FilterNodeInner::Basic(inner) => inner.est_cost(),
                FilterNodeInner::Tagged(inner) => inner.est_cost(),
            },
            Self::Join(join) => match &join.inner {
                JoinNodeInner::Basic(inner) => inner.est_cost(),
                JoinNodeInner::Tagged(inner) => inner.est_cost(),
            },
            Self::Union(union) => match &union.inner {
                UnionNodeInner::Basic(basic) => basic.est_cost(),
            },
        }
    }

    /// Estimates the cumulative cost of all nodes up to this node.
    fn est_cum_cost(&self) -> Cost {
        match self {
            Self::Base(base) => match &base.inner {
                BaseNodeInner::Basic(inner) => inner.est_cum_cost(),
                BaseNodeInner::Tagged(inner) => inner.est_cum_cost(),
            },
            Self::Filter(filter) => match &filter.inner {
                FilterNodeInner::Basic(inner) => inner.est_cum_cost(),
                FilterNodeInner::Tagged(inner) => inner.est_cum_cost(),
            },
            Self::Join(join) => match &join.inner {
                JoinNodeInner::Basic(inner) => inner.est_cum_cost(),
                JoinNodeInner::Tagged(inner) => inner.est_cum_cost(),
            },
            Self::Union(union) => match &union.inner {
                UnionNodeInner::Basic(basic) => basic.est_cum_cost(),
            },
        }
    }

    fn get_all_preds(&self) -> HashSet<Rc<Pred>> {
        match self {
            Self::Base(_) => HashSet::new(),
            Self::Filter(filter) => {
                let mut all_preds = filter.input.get_all_preds();
                all_preds.insert(filter.pred.clone());
                all_preds
            }
            Self::Join(join) => {
                itertools::concat([join.left.get_all_preds(), join.right.get_all_preds()])
            }
            Self::Union(union) => {
                itertools::concat(union.inputs.iter().map(|input| input.get_all_preds()))
            }
        }
    }

    fn get_all_table_refs(&self) -> HashSet<Rc<FileTableRef>> {
        match self {
            Self::Base(base) => HashSet::from([base.table_ref.clone()]),
            Self::Filter(filter) => filter.input.get_all_table_refs(),
            Self::Join(join) => itertools::concat([
                join.left.get_all_table_refs(),
                join.right.get_all_table_refs(),
            ]),
            Self::Union(union) => {
                itertools::concat(union.inputs.iter().map(|input| input.get_all_table_refs()))
            }
        }
    }

    /// Find the nodes with the given predicates. Returns a hash map from each predicate to the
    /// node which hosts it.
    fn find_pred_nodes(
        self: &Rc<Self>,
        preds: &HashSet<Rc<Pred>>,
    ) -> HashMap<Rc<Pred>, Rc<PlanNode>> {
        match self.as_ref() {
            Self::Base(_) => HashMap::new(),
            Self::Filter(filter) => {
                let mut pred_node_map = filter.input.find_pred_nodes(preds);
                if preds.contains(&filter.pred) {
                    pred_node_map.insert(filter.pred.clone(), self.clone());
                }
                pred_node_map
            }
            Self::Join(join) => itertools::concat([
                join.left.find_pred_nodes(preds),
                join.right.find_pred_nodes(preds),
            ]),
            Self::Union(union) => {
                itertools::concat(union.inputs.iter().map(|node| node.find_pred_nodes(preds)))
            }
        }
    }

    //fn print_est_vs_real_cost(&self) {
    //    match self {
    //        Self::Join(join) => {
    //            join.left.print_est_vs_real_cost();
    //            if let JoinNodeInner::Tagged(inner) = &join.inner {
    //                inner.print_est_vs_real_cost();
    //            }
    //            join.right.print_est_vs_real_cost();
    //        }
    //        Self::Filter(filter) => {
    //            filter.input.print_est_vs_real_cost();
    //            if let FilterNodeInner::Tagged(inner) = &filter.inner {
    //                inner.print_est_vs_real_cost();
    //            }
    //        }
    //        _ => {}
    //    }
    //}
}

impl JoinNode {
    fn new(
        left: Rc<PlanNode>,
        right: Rc<PlanNode>,
        constraint: EqualityConstraint,
        inner: JoinNodeInner,
    ) -> Self {
        Self {
            id: NodeId::new(),
            left,
            right,
            constraint,
            inner,
        }
    }
}

impl FilterNode {
    fn new(input: Rc<PlanNode>, pred: Rc<Pred>, inner: FilterNodeInner) -> Self {
        Self {
            id: NodeId::new(),
            input,
            pred,
            inner,
        }
    }
}

impl BaseNode {
    fn new(table_ref: Rc<FileTableRef>, inner: BaseNodeInner) -> Self {
        Self {
            id: NodeId::new(),
            table_ref,
            inner,
        }
    }
}

impl UnionNode {
    fn new(inputs: Vec<Rc<PlanNode>>, inner: UnionNodeInner) -> Self {
        Self {
            id: NodeId::new(),
            inputs,
            inner,
        }
    }
}

impl ProjectNode {
    fn eval(&self) -> DBResultSet {
        match &self.inner {
            ProjectNodeInner::Basic(basic) => basic.eval(self),
            ProjectNodeInner::Tagged(tagged) => tagged.eval(self),
        }
    }
}

impl<'a> TryFrom<&'a PlanNode> for &'a FilterNode {
    type Error = &'static str;

    fn try_from(value: &'a PlanNode) -> Result<Self, Self::Error> {
        match value {
            PlanNode::Filter(filter) => Ok(filter),
            _ => Err("Plan node is not of filter type"),
        }
    }
}

impl From<&ProjectNode> for termtree::Tree<String> {
    fn from(node: &ProjectNode) -> Self {
        Self::new(format!(
            "Project({}, est_cum_cost={})",
            node.exprs.iter().map(|expr| expr.to_string()).join(", "),
            node.input.est_cum_cost()
        ))
        .with_leaves([node.input.as_ref()])
    }
}

impl From<&PlanNode> for termtree::Tree<String> {
    fn from(node: &PlanNode) -> Self {
        match node {
            PlanNode::Base(base) => {
                Self::new(format!("Table({}, inner=({}))", base.table_ref, base.inner))
            }
            PlanNode::Filter(filter) => {
                Self::new(format!("Filter({}, inner=({}))", filter.pred, filter.inner))
                    .with_leaves([filter.input.as_ref()])
            }
            PlanNode::Join(join) => {
                Self::new(format!("Join({}, inner=({}))", join.constraint, join.inner))
                    .with_leaves([join.left.as_ref(), join.right.as_ref()])
            }
            PlanNode::Union(union) => Self::new(format!("Union(inner=({}))", union.inner))
                .with_leaves(union.inputs.iter().map(|input| input.as_ref())),
        }
    }
}

impl fmt::Display for Plan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", termtree::Tree::from(&self.root))
    }
}

impl fmt::Display for PlanNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", termtree::Tree::<String>::from(self))
    }
}

impl fmt::Display for BaseNodeInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Basic(basic) => basic.fmt(f),
            Self::Tagged(tagged) => tagged.fmt(f),
        }
    }
}

impl fmt::Display for FilterNodeInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Basic(basic) => basic.fmt(f),
            Self::Tagged(tagged) => tagged.fmt(f),
        }
    }
}

impl fmt::Display for JoinNodeInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Basic(basic) => basic.fmt(f),
            Self::Tagged(tagged) => tagged.fmt(f),
        }
    }
}

impl fmt::Display for UnionNodeInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Basic(basic) => basic.fmt(f),
        }
    }
}
