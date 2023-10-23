use super::utils as plan_utils;
use super::BenefitCostRatio;
use crate::db::DBResultSet;
use crate::expr::{EqualityConstraint, EvalContext, Expr};
use crate::file_table::FileTable;
use crate::idx::ColIterable;
use crate::idx::TaggedIdx;
use crate::pred::Pred;
use crate::query::Query;
use crate::stats::StatsReader;
use crate::tag::{Tag, TagSet};
use crate::utils;
use either::Either;
use itertools::Itertools;
use log::debug;
use std::borrow::Cow;
use std::cell::{Ref, RefCell};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::rc::Rc;
use std::time::Instant;

pub struct TaggedPlanner<'a> {
    stats_reader: &'a StatsReader,
}

pub struct TaggedPlan {
    root: ProjectNode,
    tag: Tag,
    total_time_ms: RefCell<u128>,
}

enum PlanNode {
    Join(JoinNode),
    Filter(FilterNode),
    Base(BaseNode),
}

impl TaggedPlan {
    pub fn eval(&self) -> DBResultSet {
        let now = Instant::now();
        let out = self.root.eval(&self.tag);
        let time_ms = now.elapsed().as_millis();
        *self.total_time_ms.borrow_mut() = time_ms;
        out
    }
}

impl<'a> TaggedPlanner<'a> {
    pub fn new(stats_reader: &'a StatsReader) -> Self {
        Self { stats_reader }
    }

    pub fn plan(&self, queries: Vec<Query>) -> Vec<TaggedPlan> {
        let (shared_preds, shared_tables) = plan_utils::find_shared(&queries);
        for query in &queries {
            //assert_eq!(
            //    HashSet::<&Rc<FileTable>>::from_iter(&query.from).len(),
            //    shared_tables.len()
            //);
            assert_eq!(query.from.len(), shared_tables.len());
        }

        let mut all_preds = HashMap::new();
        for (query_num, query) in queries.iter().enumerate() {
            let query_preds = plan_utils::get_top_level_preds(query);
            for pred in query_preds {
                all_preds
                    .entry(pred)
                    .or_insert(TagSet::new())
                    .insert(Tag::new(query_num.to_string()));
            }
        }

        let (join_graph, mut table_preds) =
            plan_utils::build_join_graph_and_table_preds(all_preds.keys());

        for preds in table_preds.values_mut() {
            preds.sort_unstable_by_key(|pred| {
                (
                    shared_preds.contains(pred),
                    BenefitCostRatio::from(pred.est()),
                )
            });
            preds.reverse();
        }

        let mut planned_nodes: HashMap<_, _> = shared_tables
            .iter()
            .map(|table| {
                (
                    table.clone(),
                    Rc::new(PlanNode::Base(BaseNode::new(
                        table.clone(),
                        (0..queries.len())
                            .map(|i| Tag::new(i.to_string()))
                            .collect(),
                    ))),
                )
            })
            .collect();

        for (table, preds) in table_preds {
            let node = planned_nodes.remove(&table).unwrap();
            let node = preds.into_iter().fold(node, |input, pred| {
                Rc::new(PlanNode::Filter(FilterNode::new(
                    input,
                    pred.clone(),
                    all_preds.get(pred).cloned().unwrap_or(TagSet::new()),
                )))
            });
            planned_nodes.insert(table, node);
        }

        let mut planned_nodes: HashMap<_, _> = planned_nodes
            .into_iter()
            .map(|(table, node)| (table, Rc::new(RefCell::new(Some(node)))))
            .collect();

        let mut unjoined_links: Vec<_> = join_graph.values().flatten().collect();
        while !unjoined_links.is_empty() {
            unjoined_links.sort_unstable_by_key(|link| {
                let left_ref = planned_nodes.get(&link.left.table()).unwrap().borrow();
                let left_node = left_ref.as_ref().unwrap();
                let right_ref = planned_nodes.get(&link.right.table()).unwrap().borrow();
                let right_node = right_ref.as_ref().unwrap();
                left_node.est_join(right_node, link, &self.stats_reader)
            });

            let link = unjoined_links.swap_remove(0);
            let left_ref = planned_nodes.get(&link.left.table()).unwrap();
            let left_node = left_ref.take().unwrap();
            let right_node = planned_nodes
                .get(&link.right.table())
                .unwrap()
                .take()
                .unwrap();
            let right_tables = right_node.tables().into_owned();
            let node = Rc::new(PlanNode::Join(JoinNode::new(
                left_node,
                right_node,
                link.clone(),
                &self.stats_reader,
            )));
            // XXX: Don't need untagged when we have all output tags already
            left_ref.replace(Some(node));
            let left_ref = left_ref.clone();
            for table in right_tables {
                planned_nodes.insert(table, left_ref.clone());
            }
            unjoined_links = unjoined_links
                .into_iter()
                .filter(|link| {
                    let left_ref = planned_nodes.get(&link.left.table()).unwrap();
                    let right_ref = planned_nodes.get(&link.right.table()).unwrap();
                    !Rc::ptr_eq(left_ref, right_ref)
                })
                .collect();
        }
        let base_plan = planned_nodes.into_values().next().unwrap().take().unwrap();

        let mut base_plan = Rc::try_unwrap(base_plan).unwrap();
        base_plan.enable_cache();
        let base_plan = Rc::new(base_plan);

        queries
            .into_iter()
            .enumerate()
            .map(|(query_num, query)| TaggedPlan {
                root: ProjectNode {
                    input: base_plan.clone(),
                    exprs: query.projection,
                },
                tag: Tag::new(query_num.to_string()),
                total_time_ms: RefCell::new(0),
            })
            .collect()
    }
}

impl PlanNode {
    fn enable_cache(&mut self) {
        match self {
            Self::Base(base) => base.enable_cache(),
            Self::Filter(filter) => filter.enable_cache(),
            Self::Join(join) => join.enable_cache(),
        }
    }

    fn eval(&self) -> Either<TaggedIdx, Ref<'_, TaggedIdx>> {
        match self {
            Self::Base(base) => base.eval(),
            Self::Filter(filter) => filter.eval(),
            Self::Join(join) => join.eval(),
        }
    }

    fn tables(&self) -> Cow<'_, Vec<Rc<FileTable>>> {
        match self {
            Self::Join(join) => Cow::Borrowed(&join.tables),
            Self::Filter(filter) => Cow::Borrowed(&filter.tables),
            Self::Base(base) => Cow::Owned(vec![base.table.clone()]),
        }
    }

    fn est_len(&self) -> usize {
        match self {
            Self::Base(base) => base.est_len,
            Self::Filter(filter) => filter.est_len,
            Self::Join(join) => join.est_len,
        }
    }

    fn est_join(
        &self,
        other: &Self,
        constraint: &EqualityConstraint,
        stats_reader: &StatsReader,
    ) -> usize {
        let (left_num_unique, right_num_unique) = stats_reader.get_unique_counts(&constraint);
        ((self.est_len() * other.est_len()) as f64
            / std::cmp::max(left_num_unique, right_num_unique) as f64)
            .round() as usize
    }

    fn tags(&self) -> &TagSet {
        match self {
            Self::Base(base) => &base.tags,
            Self::Filter(filter) => &filter.tags,
            Self::Join(join) => &join.tags,
        }
    }
}

struct JoinNode {
    left: Rc<PlanNode>,
    right: Rc<PlanNode>,
    tables: Vec<Rc<FileTable>>,
    constraint: EqualityConstraint,
    tags: TagSet,
    est_len: usize,
    cache: Option<RefCell<Option<TaggedIdx>>>,
    stats: RefCell<Option<JoinNodeStats>>,
}

struct FilterNode {
    input: Rc<PlanNode>,
    pred: Pred,
    tables: Vec<Rc<FileTable>>,
    filter_tags: TagSet,
    tags: TagSet,
    est_len: usize,
    cache: Option<RefCell<Option<TaggedIdx>>>,
    stats: RefCell<Option<FilterNodeStats>>,
}

struct BaseNode {
    table: Rc<FileTable>,
    est_len: usize,
    tags: TagSet,
    cache: Option<RefCell<Option<TaggedIdx>>>,
    stats: RefCell<Option<BaseNodeStats>>,
}

struct ProjectNode {
    input: Rc<PlanNode>,
    exprs: Vec<Expr>,
}

struct JoinNodeStats {
    left_num: usize,
    right_num: usize,
    out_num: usize,
    left_num_rel_slices: usize,
    right_num_rel_slices: usize,
    out_num_rel_slices: usize,
    time_ms: u128,
}

struct FilterNodeStats {
    in_num: usize,
    out_num: usize,
    in_num_rel_slices: usize,
    out_num_rel_slices: usize,
    time_ms: u128,
}

struct BaseNodeStats {
    out_num: usize,
    time_ms: u128,
}

impl JoinNode {
    fn new(
        left: Rc<PlanNode>,
        right: Rc<PlanNode>,
        constraint: EqualityConstraint,
        stats_reader: &StatsReader,
    ) -> Self {
        let tables = utils::append(left.tables().into_owned(), right.tables().into_owned());
        let est_len = left.est_join(&right, &constraint, stats_reader);
        let tags = left.tags() | right.tags();
        Self {
            left,
            right,
            constraint,
            tables,
            tags,
            est_len,
            cache: None,
            stats: RefCell::new(None),
        }
    }

    fn enable_cache(&mut self) {
        self.cache = Some(RefCell::new(None));
    }

    fn eval(&self) -> Either<TaggedIdx, Ref<'_, TaggedIdx>> {
        fn eval_inner(
            left: &PlanNode,
            right: &PlanNode,
            constraint: &EqualityConstraint,
        ) -> (TaggedIdx, JoinNodeStats) {
            let now = Instant::now();

            let left_idx = left.eval();
            let right_idx = right.eval();

            let left_idx = left_idx.as_ref().either(|idx| idx, |idx_ref| &*idx_ref);
            let right_idx = right_idx.as_ref().either(|idx| idx, |idx_ref| &*idx_ref);

            let out_idx = left_idx.join(right_idx, constraint);

            let time_ms = now.elapsed().as_millis();

            let stats = JoinNodeStats {
                left_num: left_idx.len(),
                right_num: right_idx.len(),
                left_num_rel_slices: left_idx.num_rel_slices(),
                right_num_rel_slices: right_idx.num_rel_slices(),
                out_num: out_idx.len(),
                out_num_rel_slices: out_idx.num_rel_slices(),
                time_ms,
            };

            debug!("constraint {} idx {:?}", constraint, out_idx);
            (out_idx, stats)
        }

        if let Some(cache) = &self.cache {
            {
                let mut cache = cache.borrow_mut();
                if cache.is_none() {
                    let (out, stats) = eval_inner(&self.left, &self.right, &self.constraint);
                    *self.stats.borrow_mut() = Some(stats);
                    *cache = Some(out);
                }
            }
            Either::Right(Ref::map(cache.borrow(), |idx| idx.as_ref().unwrap()))
        } else {
            let (out, stats) = eval_inner(&self.left, &self.right, &self.constraint);
            *self.stats.borrow_mut() = Some(stats);
            Either::Left(out)
        }
    }
}

impl FilterNode {
    fn new(input: Rc<PlanNode>, pred: Pred, filter_tags: TagSet) -> Self {
        let est_len = if filter_tags.is_superset(input.tags()) {
            (input.est_len() as f64 * pred.est().selectivity).round() as usize
        } else {
            input.est_len()
        };
        let tables = input.tables().into_owned();
        let tags = input.tags() | &filter_tags;

        Self {
            input,
            pred,
            tables,
            filter_tags,
            tags,
            est_len,
            cache: None,
            stats: RefCell::new(None),
        }
    }

    fn enable_cache(&mut self) {
        self.cache = Some(RefCell::new(None));
    }

    fn eval(&self) -> Either<TaggedIdx, Ref<'_, TaggedIdx>> {
        fn eval_inner(
            input: &PlanNode,
            pred: &Pred,
            filter_tags: &TagSet,
        ) -> (TaggedIdx, FilterNodeStats) {
            let now = Instant::now();

            let in_idx = input.eval();

            let in_idx = in_idx.as_ref().either(|idx| idx, |idx_ref| &*idx_ref);

            let out_idx = in_idx.filter(
                &pred.eval(&EvalContext {
                    idx: &ColIterable::Tagged(in_idx),
                    bmap: None,
                    cache: None,
                }),
                filter_tags,
            );

            let time_ms = now.elapsed().as_millis();

            let stats = FilterNodeStats {
                in_num: in_idx.len(),
                in_num_rel_slices: in_idx.num_rel_slices(),
                out_num: out_idx.len(),
                out_num_rel_slices: out_idx.num_rel_slices(),
                time_ms,
            };

            (out_idx, stats)
        }

        if let Some(cache) = &self.cache {
            {
                let mut cache = cache.borrow_mut();
                if cache.is_none() {
                    let (out, stats) = eval_inner(&self.input, &self.pred, &self.filter_tags);
                    *self.stats.borrow_mut() = Some(stats);
                    *cache = Some(out);
                }
            }
            Either::Right(Ref::map(cache.borrow(), |idx| idx.as_ref().unwrap()))
        } else {
            let (out, stats) = eval_inner(&self.input, &self.pred, &self.filter_tags);
            *self.stats.borrow_mut() = Some(stats);
            Either::Left(out)
        }
    }
}

impl BaseNode {
    fn new(table: Rc<FileTable>, tags: TagSet) -> Self {
        let est_len = table.len();
        Self {
            table,
            est_len,
            tags,
            cache: None,
            stats: RefCell::new(None),
        }
    }

    fn enable_cache(&mut self) {
        self.cache = Some(RefCell::new(None));
    }

    fn eval(&self) -> Either<TaggedIdx, Ref<'_, TaggedIdx>> {
        fn eval_inner(table: Rc<FileTable>, tags: &TagSet) -> (TaggedIdx, BaseNodeStats) {
            let now = Instant::now();
            let idx = TaggedIdx::new(table.clone(), tags);
            let time_ms = now.elapsed().as_millis();

            let stats = BaseNodeStats {
                out_num: idx.len(),
                time_ms,
            };

            (idx, stats)
        }

        if let Some(cache) = &self.cache {
            {
                let mut cache = cache.borrow_mut();
                if cache.is_none() {
                    let (out, stats) = eval_inner(self.table.clone(), &self.tags);
                    *self.stats.borrow_mut() = Some(stats);
                    *cache = Some(out);
                }
            }
            Either::Right(Ref::map(cache.borrow(), |idx| idx.as_ref().unwrap()))
        } else {
            let (out, stats) = eval_inner(self.table.clone(), &self.tags);
            *self.stats.borrow_mut() = Some(stats);
            Either::Left(out)
        }
    }
}

impl ProjectNode {
    fn eval(&self, tag: &Tag) -> DBResultSet {
        let idx = self.input.eval();
        let idx = idx.as_ref().either(|l| l, |r| &*r);
        let idx_ref = idx.ref_tagged(tag);
        let iterable = idx_ref
            .as_ref()
            .map(|idx_ref| ColIterable::TaggedRef(idx_ref))
            .unwrap_or(ColIterable::Tagged(&idx));
        DBResultSet::new(
            self.exprs
                .iter()
                .map(|expr| {
                    let vals = expr.eval(&EvalContext {
                        idx: &iterable,
                        bmap: None,
                    });
                    //vals.expect_right("We're not even using the cache")
                    vals
                })
                .collect(),
        )
    }
}

impl From<&TaggedPlan> for termtree::Tree<String> {
    fn from(plan: &TaggedPlan) -> Self {
        Self::new(format!(
            "Project({}) tag = {} [total_time_ms = {}]",
            plan.root
                .exprs
                .iter()
                .map(|expr| expr.to_string())
                .join(", "),
            plan.tag,
            plan.total_time_ms.borrow()
        ))
        .with_leaves([plan.root.input.as_ref()])
    }
}

impl From<&PlanNode> for termtree::Tree<String> {
    fn from(node: &PlanNode) -> Self {
        match node {
            PlanNode::Base(base) => Self::new(format!(
                "Table({}, est_len = {}){}",
                base.table.name(),
                base.est_len,
                base.stats
                    .borrow()
                    .as_ref()
                    .map(|stats| format!(" [out = {}, time_ms = {}]", stats.out_num, stats.time_ms))
                    .unwrap_or("".to_string())
            )),
            PlanNode::Filter(filter) => Self::new(format!(
                "Filter({}, selectivity = {}, est_len = {}, filter_tags=[{}]){}",
                filter.pred,
                filter.pred.est().selectivity,
                filter.est_len,
                filter.filter_tags.iter().join(", "),
                filter
                    .stats
                    .borrow()
                    .as_ref()
                    .map(|stats| format!(
                        " [in = {}, out = {}, in_rel_slices = {}, out_rel_slices = {}, time_ms = {}]",
                        stats.in_num,
                        stats.out_num,
                        stats.in_num_rel_slices,
                        stats.out_num_rel_slices,
                        stats.time_ms
                    ))
                    .unwrap_or("".to_string())
            ))
            .with_leaves([filter.input.as_ref()]),
            PlanNode::Join(join) => Self::new(format!(
                "Join({}, est_len = {}){}",
                join.constraint, join.est_len,
                join.stats
                    .borrow()
                    .as_ref()
                    .map(|stats| format!(
                        " [left = {}, right = {}, out = {}, left_rel_slices = {}, right_rel_slices = {}, out_rel_slices = {}, time_ms = {}]",
                        stats.left_num, stats.right_num, stats.out_num, stats.left_num_rel_slices,
                        stats.right_num_rel_slices, stats.out_num_rel_slices, stats.time_ms
                    ))
                    .unwrap_or("".to_string())
            ))
            .with_leaves([join.left.as_ref(), join.right.as_ref()]),
        }
    }
}

impl fmt::Debug for TaggedPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", termtree::Tree::from(self))
    }
}

impl fmt::Debug for PlanNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PlanNode")
    }
}
