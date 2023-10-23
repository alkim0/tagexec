//! Core for tagged planners.
use crate::bitmap::BitmapInt;
use crate::cost::{cost_factors, Cost};
use crate::db::DBResultSet;
use crate::engine::EXEC_INFO;
use crate::expr::{EqualityConstraint, EvalContext, Expr};
use crate::file_table::{FileCol, FileTableRef};
use crate::idx::{Idx, TaggedIdx};
use crate::plan::disj::utils as plan_utils;
use crate::plan::disj::{JoinGraph, TableNodeMap, BENEFIT_EPSILON, COST_EPSILON};
use crate::plan::{
    BaseNode, BaseNodeInner, FilterNode, FilterNodeInner, JoinNode, JoinNodeInner, PlanNode,
    ProjectNode, ProjectNodeInner,
};
use crate::pred::Pred;
use crate::query::Query;
use crate::stats::StatsReader;
use crate::tag::{Tag, TagManager};
use float_ord::FloatOrd;
use itertools::Itertools;
use log::info;
use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;
use std::hash::BuildHasherDefault;
use std::rc::Rc;
use std::time::Instant;

pub(super) struct TaggedPlannerCore<'a> {
    stats_reader: &'a StatsReader,
    tag_manager: Rc<TagManager>,
}

type JoinPaths = HashMap<Rc<FileTableRef>, HashMap<Rc<FileTableRef>, Vec<Vec<EqualityConstraint>>>>;

// A float representing the benefit / ratio of a predicate or set of predicates in which:
// benefit = how much cheaper joining the remaining nodes would be
// cost = how much it costs to inlcude the applied predicate or predicate set to the current plan
// Used for dealing with predicates in a discriminating context
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
struct DisBenefitCostRatio(FloatOrd<f64>);

// A float representing the benefit / ratio of a predicate or set of predicates in which:
// benefit = how the selectivity is reduced for othe predicates
// cost = cost factor of the predicate
// Used for dealing with predicates in a specifying context
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
struct SpecBenefitCostRatio(FloatOrd<f64>);

// Reference Information for Planner
pub(super) struct ProcessedQueryInfo {
    pub join_graph: JoinGraph,
    pub pred_root: Option<Rc<Pred>>,
    pub base_table_node_map: TableNodeMap,
    pub join_paths: JoinPaths,
}

impl<'a> TaggedPlannerCore<'a> {
    pub fn new(stats_reader: &'a StatsReader) -> Self {
        Self {
            stats_reader,
            tag_manager: TagManager::new(),
        }
    }

    pub fn process_query(&self, query: &Query) -> ProcessedQueryInfo {
        let base_table_node_map = query
            .from
            .iter()
            .map(|table_ref| (table_ref.clone(), self.make_base_node(table_ref.clone())))
            .collect();

        let (join_graph, pred_root) = query
            .filter
            .as_ref()
            .map(|filter| plan_utils::split_join_constraints(filter))
            .unwrap_or((HashMap::new(), None));

        let join_paths = self.build_join_paths(&join_graph);

        ProcessedQueryInfo {
            join_graph,
            pred_root,
            base_table_node_map,
            join_paths,
        }
    }

    pub fn get_min_dis_pred_info(&self, pred_root: &Rc<Pred>) -> MinDisPredInfo {
        MinDisPredInfo::new(pred_root)
    }

    /// Joins the nodes in the given `table_node_map` with `base` if given. `base` represents the
    /// already partially formed plan and tables may already be joined in it.
    pub fn join_table_node_map(
        &self,
        mut table_node_map: TableNodeMap,
        join_graph: &JoinGraph,
        base: Option<Rc<PlanNode>>,
    ) -> Rc<PlanNode> {
        let join_edges = if let Some(base) = base {
            let table_refs = base.get_all_table_refs();
            let join_edges = join_graph
                .values()
                .flatten()
                .filter(|edge| {
                    // Only return edges which have not already been joined in base (i.e., edges where
                    // both ends appear in table_refs have already been joined in base).
                    !(table_refs.contains(&edge.left_table_ref)
                        && table_refs.contains(&edge.right_table_ref))
                })
                .collect();
            for table_ref in table_refs {
                table_node_map.insert(table_ref, base.clone());
            }
            join_edges
        } else {
            join_graph.values().flatten().collect()
        };

        plan_utils::plan_greedy_join(
            join_edges,
            table_node_map,
            |left_node, right_node, constraint| {
                self.make_join_node(left_node, right_node, constraint)
            },
        )
    }

    pub fn make_base_node(&self, table_ref: Rc<FileTableRef>) -> Rc<PlanNode> {
        let inner = BaseNodeInner::Tagged(TaggedBaseNodeInner::new(
            &table_ref,
            &self.tag_manager,
            &self.stats_reader,
        ));
        Rc::new(PlanNode::Base(BaseNode::new(table_ref, inner)))
    }

    pub fn make_filter_node(&self, input: Rc<PlanNode>, pred: Rc<Pred>) -> Rc<PlanNode> {
        let inner = FilterNodeInner::Tagged(TaggedFilterNodeInner::new(
            &input,
            &pred,
            &self.stats_reader,
        ));
        Rc::new(PlanNode::Filter(FilterNode::new(input, pred, inner)))
    }

    pub fn make_join_node(
        &self,
        left_node: Rc<PlanNode>,
        right_node: Rc<PlanNode>,
        constraint: EqualityConstraint,
    ) -> Rc<PlanNode> {
        let inner = JoinNodeInner::Tagged(TaggedJoinNodeInner::new(
            &left_node,
            &right_node,
            &constraint,
            &self.stats_reader,
        ));
        Rc::new(PlanNode::Join(JoinNode::new(
            left_node, right_node, constraint, inner,
        )))
    }

    pub fn make_project_node(&self, input: Rc<PlanNode>, exprs: &Vec<Expr>) -> ProjectNode {
        ProjectNode {
            input,
            exprs: exprs.clone(),
            inner: ProjectNodeInner::Tagged(TaggedProjectNodeInner),
        }
    }

    fn build_join_paths(&self, join_graph: &JoinGraph) -> JoinPaths {
        fn build(
            node: &Rc<FileTableRef>,
            path: &Vec<EqualityConstraint>,
            join_graph: &JoinGraph,
            source: &Rc<FileTableRef>,
            seen: &mut HashSet<Rc<FileTableRef>>,
        ) -> HashMap<Rc<FileTableRef>, Vec<Vec<EqualityConstraint>>> {
            seen.insert(node.clone());
            let mut paths = HashMap::new();
            for edge in join_graph.get(node).unwrap() {
                // If this edge is the edge to get to this node, skip
                if path
                    .last()
                    .map(|last_edge| edge == last_edge)
                    .unwrap_or(false)
                {
                    continue;
                }
                let other_node = edge.other_table_ref(node);
                if other_node == source || seen.contains(other_node) {
                    continue;
                }

                let mut path = path.clone();
                path.push(edge.clone());
                for (dest, dest_paths) in build(other_node, &path, join_graph, source, seen) {
                    paths.entry(dest).or_insert(vec![]).extend(dest_paths);
                }
                paths.entry(other_node.clone()).or_insert(vec![]).push(path);
            }
            seen.remove(node);
            paths
        }

        join_graph
            .keys()
            .map(|source| {
                (
                    source.clone(),
                    build(source, &vec![], join_graph, source, &mut HashSet::new()),
                )
            })
            .collect()
    }

    //fn get_other_min_dis_pred_sets(
    //    &self,
    //    unused_preds: HashSet<Rc<Pred>>,
    //    min_dis_pred_map: &HashMap<Rc<Pred>, Vec<Rc<HashSet<Rc<Pred>>>>>,
    //) -> Vec<Rc<HashSet<Rc<Pred>>>> {
    //    // A wrapper around the HashSet of Preds which compares based on Rc pointers rather than
    //    // the values.
    //    struct PredSet(Rc<HashSet<Rc<Pred>>>);
    //    impl PartialEq for PredSet {
    //        fn eq(&self, other: &Self) -> bool {
    //            Rc::ptr_eq(&self.0, &other.0)
    //        }
    //    }
    //    impl Eq for PredSet {}
    //    impl Hash for PredSet {
    //        fn hash<H: Hasher>(&self, state: &mut H) {
    //            std::ptr::hash(Rc::as_ptr(&self.0), state);
    //        }
    //    }

    //    unused_preds
    //        .iter()
    //        .map(|pred| {
    //            min_dis_pred_map
    //                .get(pred)
    //                .unwrap()
    //                .iter()
    //                .filter(|pred_set| pred_set.is_subset(&unused_preds))
    //                .map(|pred_set| PredSet(pred_set.clone()))
    //        })
    //        .flatten()
    //        .collect::<HashSet<_>>()
    //        .into_iter()
    //        .map(|pred_set| pred_set.0)
    //        .collect()
    //}

    //// Returns a list of additionally discriminating predicates. The returned predicates are not arranged in any specific order.
    //fn get_add_dis_preds(
    //    &self,
    //    min_dis_pred_map: &HashMap<Rc<Pred>, Vec<Rc<HashSet<Rc<Pred>>>>>,
    //    all_preds: &HashSet<Rc<Pred>>,
    //    used_preds: &HashSet<Rc<Pred>>,
    //) -> HashSet<Rc<Pred>> {
    //    (all_preds - used_preds)
    //        .into_iter()
    //        .filter(|pred| {
    //            min_dis_pred_map
    //                .get(pred)
    //                .map(|pred_sets| {
    //                    pred_sets.iter().any(|pred_set| {
    //                        // Here, `pred` should be the only predicate that is part of `pred_set` but not in
    //                        // `used_preds`.
    //                        (pred_set.as_ref() - used_preds).len() == 1
    //                    })
    //                })
    //                .unwrap_or(false)
    //        })
    //        .collect()
    //}
}

type PredSet = BTreeSet<Rc<Pred>>;

pub(super) struct MinDisPredInfo {
    pub pred_sets: Vec<Rc<PredSet>>,
    pub pred_map: HashMap<Rc<Pred>, Vec<Rc<PredSet>>>,
}

impl MinDisPredInfo {
    fn new(pred_root: &Rc<Pred>) -> Self {
        fn get_pred_sets(pred: &Rc<Pred>) -> Vec<PredSet> {
            if pred.is_atom() {
                vec![BTreeSet::from([pred.clone()])]
            } else if pred.is_and() {
                pred.try_iter_children()
                    .unwrap()
                    .map(|child| get_pred_sets(child))
                    .concat()
            } else {
                pred.try_iter_children()
                    .unwrap()
                    .map(|child| get_pred_sets(child))
                    .multi_cartesian_product()
                    .map(|pred_sets| pred_sets.into_iter().concat())
                    .collect()
            }
        }

        let pred_sets: Vec<_> = get_pred_sets(pred_root)
            .into_iter()
            .map(|pred_set| Rc::new(pred_set))
            .collect();

        let mut pred_map = HashMap::new();
        for pred_set in &pred_sets {
            for pred in pred_set.iter() {
                pred_map
                    .entry(pred.clone())
                    .or_insert(vec![])
                    .push(pred_set.clone());
            }
        }

        Self {
            pred_sets,
            pred_map,
        }
    }
}

struct EstInfo {
    est_total_len: f64,
    est_len: FxHashMap<Rc<Tag>, f64>,
    est_cum_cost: Cost, // Total cost up to this node.
    est_cost: Cost,     // Est cost of this node
}

struct RunInfo {
    eval_time_ms: u128,
    num_output: Vec<(Rc<Tag>, usize)>,
    was_run: bool,
}

pub struct TaggedBaseNodeInner {
    est_info: EstInfo,
    run_info: RefCell<RunInfo>,
}

pub struct TaggedFilterNodeInner {
    // Maps from input tag to whether we should include the current filter node's predicate's
    // true/false values. Tags we shouldn't process have neither value.
    tag_map: FxHashMap<Rc<Tag>, Vec<(bool, Rc<Tag>)>>,

    est_info: EstInfo,
    run_info: RefCell<RunInfo>,
}

pub struct TaggedJoinNodeInner {
    // Map from (left, right) to output tag.
    tag_map: FxHashMap<(Rc<Tag>, Rc<Tag>), Option<Rc<Tag>>>,

    // When evaluating, if this is true, we build a hash map based on the left node's values.
    // Otherwise, we build a hash map based on the right node's values.
    left_based: bool,

    est_info: EstInfo,
    run_info: RefCell<RunInfo>,
}

pub struct TaggedProjectNodeInner;

impl TaggedBaseNodeInner {
    pub(in crate::plan) fn new(
        table_ref: &FileTableRef,
        tag_manager: &Rc<TagManager>,
        _stats_reader: &StatsReader,
    ) -> Self {
        let tag = tag_manager.empty_tag();
        let est_total_len = table_ref.table.len() as f64;
        let est_len = FxHashMap::from_iter(std::iter::once((tag.clone(), est_total_len)));

        Self {
            est_info: EstInfo {
                est_total_len,
                est_len,
                est_cum_cost: 0.,
                est_cost: 0.,
            },
            run_info: RefCell::new(RunInfo {
                eval_time_ms: 0,
                num_output: vec![],
                was_run: false,
            }),
        }
    }

    pub fn est_len(&self) -> f64 {
        self.est_info.est_total_len
    }

    pub fn est_cost(&self) -> Cost {
        self.est_info.est_cost
    }

    pub fn est_cum_cost(&self) -> Cost {
        self.est_info.est_cum_cost
    }

    pub(in crate::plan) fn est_unique(
        &self,
        table_ref: &Rc<FileTableRef>,
        col: &Rc<FileCol>,
        node: &BaseNode,
        stats_reader: &StatsReader,
    ) -> f64 {
        stats_reader.get_num_unique(col) as f64
    }

    pub(in crate::plan) fn eval(&self, node: &BaseNode) -> Idx {
        let now = Instant::now();
        let out = TaggedIdx::new(
            node.table_ref.clone(),
            self.est_info.est_len.keys().cloned(),
        );

        let mut run_info = self.run_info.borrow_mut();
        run_info.eval_time_ms = now.elapsed().as_millis();
        run_info.num_output = out.tag_lens();
        run_info.was_run = true;

        Idx::Tagged(out)
    }
}

impl TaggedFilterNodeInner {
    pub(in crate::plan) fn new(
        input: &PlanNode,
        pred: &Rc<Pred>,
        stats_reader: &StatsReader,
    ) -> Self {
        let input_est_info = match input {
            PlanNode::Base(base) => &<&TaggedBaseNodeInner>::try_from(base).unwrap().est_info,
            PlanNode::Filter(filter) => {
                &<&TaggedFilterNodeInner>::try_from(filter).unwrap().est_info
            }
            PlanNode::Join(join) => &<&TaggedJoinNodeInner>::try_from(join).unwrap().est_info,
            _ => panic!("Unexpected plan node"),
        };

        let mut est_len = FxHashMap::with_hasher(BuildHasherDefault::default());
        let mut tag_map = FxHashMap::with_hasher(BuildHasherDefault::default());
        let ancestor_lines = pred.ancestor_lines();
        let mut est_cost = 0.;

        for (tag, &tag_len) in &input_est_info.est_len {
            if !ancestor_lines.is_empty()
                && ancestor_lines.iter().all(|line| {
                    line.iter()
                        .any(|ancestor| tag.assign.contains_key(ancestor))
                })
            {
                // We do not need to perform the predicate on this tagged slice since one of the
                // predicate's ancestors in each line has a static value.
                est_len
                    .entry(tag.clone())
                    .and_modify(|len| *len += tag_len)
                    .or_insert(tag_len);
                tag_map.insert(tag.clone(), vec![]);
            } else {
                let tag_map_vals = tag_map.entry(tag.clone()).or_insert(vec![]);
                for val in [true, false] {
                    if let Some(new_tag) = tag.concat(std::iter::once((pred.clone(), val))) {
                        let pred_est = pred.est();
                        let result_len = if val {
                            tag_len as f64 * pred_est.selectivity
                        } else {
                            tag_len as f64 * (1. - pred_est.selectivity)
                        };

                        est_len
                            .entry(new_tag.clone())
                            .and_modify(|len| *len += result_len)
                            .or_insert(result_len);
                        tag_map_vals.push((val, new_tag));

                        // NOTE: The cost here is calculated in a very naive manner. It's simply
                        // the number of tuples * a cost factor for the predicate. A more complex
                        // cost function would look into the separate factors. Specifically, the
                        // cost of the projection may impact the cost heavily, and certain
                        // situations (e.g., spin disks), we would want to do a full scan instead
                        // of reading selectively.
                        // 1. Cost of reading in the specified values + projection
                        // 2. Cost of applying the predicate
                        // 3. Cost of creating a tagged index from it = 0 (because bitmap)
                        // TODO FIXME: Include the cost to hit disk here, since the filter does
                        // eventually read from disk.
                        est_cost +=
                            pred_est.cost * tag_len as f64 * cost_factors::FILTER_COST_FACTOR;
                    }
                }
            }
        }

        let est_total_len = est_len.values().sum();

        Self {
            tag_map,
            est_info: EstInfo {
                est_total_len,
                est_len,
                est_cum_cost: input_est_info.est_cum_cost + est_cost,
                est_cost,
            },
            run_info: RefCell::new(RunInfo {
                eval_time_ms: 0,
                num_output: vec![],
                was_run: false,
            }),
        }
    }

    pub fn est_len(&self) -> f64 {
        self.est_info.est_total_len
    }

    pub fn est_cost(&self) -> Cost {
        self.est_info.est_cost
    }

    pub fn est_cum_cost(&self) -> Cost {
        self.est_info.est_cum_cost
    }

    pub(in crate::plan) fn est_unique(
        &self,
        table_ref: &Rc<FileTableRef>,
        col: &Rc<FileCol>,
        node: &FilterNode,
        stats_reader: &StatsReader,
    ) -> f64 {
        // XXX: This assumes predicate has no effect on the unique value. However, it obviously
        // can't be greater than the total size, so it must be smaller than that. This obviously
        // isn't super accurate (especially if the predicate is applied directly on the column we
        // are estimating the unique size of). Making this estimation is left as a todo.
        std::cmp::min(
            FloatOrd(node.input.est_unique(table_ref, col, stats_reader)),
            FloatOrd(self.est_info.est_total_len),
        )
        .0
    }

    pub(in crate::plan) fn eval(&self, node: &FilterNode) -> Idx {
        let in_idx = node.input.eval();
        let tagged_in_idx = <&TaggedIdx>::try_from(&in_idx).unwrap();

        //println!("Print tag map and est_len for node: {}", node.pred);
        //println!("est_len {:#?}", self.est_info.est_len);
        //println!("tag_map {:#?}", self.tag_map);

        let now = Instant::now();
        let out = tagged_in_idx.apply_pred(&node.pred, &self.tag_map, &in_idx);

        if EXEC_INFO.with(|exec_info| exec_info.borrow().debug_times) {
            info!(
                "Filter {} total time {} ms",
                node.pred,
                now.elapsed().as_millis()
            );
        }

        let mut run_info = self.run_info.borrow_mut();
        run_info.eval_time_ms = now.elapsed().as_millis();
        run_info.num_output = out.tag_lens();
        run_info.was_run = true;

        Idx::Tagged(out)
    }

    //pub fn print_est_vs_real_cost(&self) {
    //    let est_cost = self.est_info.est_cost;
    //    let real_cost = self.run_info.borrow().eval_time_ms;
    //    if real_cost >= 80 || est_cost > 1e5 {
    //        println!(
    //            "F est {} real {} ratio {}",
    //            est_cost,
    //            real_cost,
    //            est_cost / real_cost as f64
    //        );
    //    }
    //}

    //pub fn print_tags(&self) {
    //    for tag in self.est_info.est_len.keys() {
    //        println!("printing preds and parents for tag {}", tag);
    //        for pred in tag.assign.keys() {
    //            println!("For tag, pred {} parent {:?}", pred, pred.parent());
    //        }
    //    }
    //}
}

impl TaggedJoinNodeInner {
    pub(in crate::plan) fn new(
        left: &PlanNode,
        right: &PlanNode,
        constraint: &EqualityConstraint,
        stats_reader: &StatsReader,
    ) -> Self {
        let left_est_info = match left {
            PlanNode::Base(base) => &<&TaggedBaseNodeInner>::try_from(base).unwrap().est_info,
            PlanNode::Filter(filter) => {
                &<&TaggedFilterNodeInner>::try_from(filter).unwrap().est_info
            }
            PlanNode::Join(join) => &<&TaggedJoinNodeInner>::try_from(join).unwrap().est_info,
            _ => panic!("Unexpected plan node"),
        };
        let right_est_info = match right {
            PlanNode::Base(base) => &<&TaggedBaseNodeInner>::try_from(base).unwrap().est_info,
            PlanNode::Filter(filter) => {
                &<&TaggedFilterNodeInner>::try_from(filter).unwrap().est_info
            }
            PlanNode::Join(join) => &<&TaggedJoinNodeInner>::try_from(join).unwrap().est_info,
            _ => panic!("Unexpected plan node"),
        };

        let mut est_len: FxHashMap<Rc<Tag>, f64> =
            FxHashMap::with_hasher(BuildHasherDefault::default());
        let mut tag_map = FxHashMap::with_hasher(BuildHasherDefault::default());

        let left_num_unique = left.est_unique(
            &constraint.left_table_ref,
            &constraint.left_col,
            stats_reader,
        );
        let right_num_unique = right.est_unique(
            &constraint.right_table_ref,
            &constraint.right_col,
            stats_reader,
        );
        let unique_factor = std::cmp::max(FloatOrd(left_num_unique), FloatOrd(right_num_unique)).0;

        for ((left_tag, left_tag_len), (right_tag, right_tag_len)) in left_est_info
            .est_len
            .iter()
            .cartesian_product(&right_est_info.est_len)
        {
            if let Some(new_tag) = left_tag.combine(right_tag) {
                let result_len = (left_tag_len * right_tag_len) as f64 / unique_factor;
                est_len
                    .entry(new_tag.clone())
                    .and_modify(|len| *len += result_len)
                    .or_insert(result_len);
                tag_map.insert((left_tag.clone(), right_tag.clone()), Some(new_tag));
            } else {
                tag_map.insert((left_tag.clone(), right_tag.clone()), None);
            }
        }

        let est_total_len = est_len.values().sum();

        //println!("constraint {} left total len {} rigth total len {} left unique {} right unique {} total est {} this_sum {} ",
        //    constraint,
        //    left_est_info.est_total_len,
        //    right_est_info.est_total_len,
        //    left_num_unique,
        //    right_num_unique,
        //    est_total_len, this_sum);

        let calc_est_cost =
            |left_est_info: &EstInfo, right_est_info: &EstInfo, left_num_unique: f64| {
                // NOTE: This doesn't include the cost of allocating the space for the values of the
                // hash map.
                // As a special case, if one side has 0 tuples, then the cost of the join is 0
                if left_est_info.est_total_len == 0. {
                    return 0.;
                }

                let hash_map_build_cost = left_est_info.est_total_len as f64
                    * constraint.left_col.data_type().size() as f64
                    * cost_factors::HASH_COST_FACTOR
                    + left_num_unique as f64
                        * constraint.left_col.data_type().size() as f64
                        * cost_factors::MEM_ALLOC_FACTOR;
                // For each tagged slice from the right side, there must be at least once tagged slice
                // on the left side which when combined does not result in None, so we must hash every
                // value on the right side.
                let hash_lookup_cost = right_est_info.est_total_len as f64
                    * constraint.right_col.data_type().size() as f64
                    * cost_factors::HASH_COST_FACTOR;
                let build_idx_cost = ((left.get_all_table_refs().len()
                    + right.get_all_table_refs().len())
                    * std::mem::size_of::<BitmapInt>()) as f64
                    * est_total_len
                    * cost_factors::MEM_ALLOC_FACTOR;

                hash_map_build_cost + hash_lookup_cost + build_idx_cost
            };

        let left_cost = calc_est_cost(left_est_info, right_est_info, left_num_unique);
        let right_cost = calc_est_cost(right_est_info, left_est_info, right_num_unique);

        let (left_based, est_cost) = if left_cost <= right_cost {
            (true, left_cost)
        } else {
            (false, right_cost)
        };

        Self {
            tag_map,
            left_based,
            est_info: EstInfo {
                est_total_len,
                est_len,
                est_cum_cost: left_est_info.est_cum_cost + right_est_info.est_cum_cost + est_cost,
                est_cost,
            },
            run_info: RefCell::new(RunInfo {
                eval_time_ms: 0,
                num_output: vec![],
                was_run: false,
            }),
        }
    }

    pub fn est_len(&self) -> f64 {
        self.est_info.est_total_len
    }

    pub fn est_cost(&self) -> Cost {
        self.est_info.est_cost
    }

    pub fn est_cum_cost(&self) -> Cost {
        self.est_info.est_cum_cost
    }

    pub(in crate::plan) fn est_unique(
        &self,
        table_ref: &Rc<FileTableRef>,
        col: &Rc<FileCol>,
        node: &JoinNode,
        stats_reader: &StatsReader,
    ) -> f64 {
        // XXX: The following estimates assumes inner joins
        let num_unique = if (&node.constraint.left_table_ref == table_ref
            && &node.constraint.left_col == col)
            || (&node.constraint.right_table_ref == table_ref && &node.constraint.right_col == col)
        {
            let left_num_unique = node.left.est_unique(
                &node.constraint.left_table_ref,
                &node.constraint.left_col,
                stats_reader,
            );
            let right_num_unique = node.right.est_unique(
                &node.constraint.right_table_ref,
                &node.constraint.right_col,
                stats_reader,
            );
            std::cmp::min(FloatOrd(left_num_unique), FloatOrd(right_num_unique)).0
        } else {
            if node.left.get_all_table_refs().contains(table_ref) {
                node.left.est_unique(table_ref, col, stats_reader)
            } else {
                node.right.est_unique(table_ref, col, stats_reader)
            }
        };

        std::cmp::min(FloatOrd(num_unique), FloatOrd(self.est_info.est_total_len)).0
    }

    pub(in crate::plan) fn eval(&self, node: &JoinNode) -> Idx {
        let left_idx = node.left.eval();
        let right_idx = node.right.eval();

        let tagged_left_idx = <&TaggedIdx>::try_from(&left_idx).unwrap();
        let tagged_right_idx = <&TaggedIdx>::try_from(&right_idx).unwrap();

        //println!(
        //    "------ For join node (tagged) {} ---------",
        //    node.constraint
        //);
        //println!("tagged_left_idx {:?}", tagged_left_idx);
        //println!("tagged_right_idx {:?}", tagged_right_idx);

        //println!(
        //    "Print tag map and est_len for join node: [{}] join [{}]",
        //    node.left
        //        .get_all_table_refs()
        //        .iter()
        //        .map(|t| t.to_string())
        //        .join(", "),
        //    node.right
        //        .get_all_table_refs()
        //        .iter()
        //        .map(|t| t.to_string())
        //        .join(", ")
        //);
        //println!("est_len {:#?}", self.est_info.est_len);
        //println!("tag_map {:#?}", self.tag_map);

        let now = Instant::now();
        let out = tagged_left_idx.join(
            &tagged_right_idx,
            &node.constraint,
            &self.tag_map,
            self.left_based,
        );

        if EXEC_INFO.with(|exec_info| exec_info.borrow().debug_times) {
            info!(
                "Join {} total time {} ms",
                node.constraint,
                now.elapsed().as_millis()
            );
        }

        let mut run_info = self.run_info.borrow_mut();
        run_info.eval_time_ms = now.elapsed().as_millis();
        run_info.num_output = out.tag_lens();
        run_info.was_run = true;

        //println!("joined_idx {:?}", out);
        Idx::Tagged(out)
    }

    //pub fn print_est_vs_real_cost(&self) {
    //    let est_cost = self.est_info.est_cost;
    //    let real_cost = self.run_info.borrow().eval_time_ms;
    //    if real_cost >= 80 || est_cost > 1e5 {
    //        println!(
    //            "J est {} real {} ratio {}",
    //            est_cost,
    //            real_cost,
    //            est_cost / real_cost as f64
    //        );
    //    }
    //}
}

impl TaggedProjectNodeInner {
    pub(in crate::plan) fn eval(&self, node: &ProjectNode) -> DBResultSet {
        let idx = node.input.eval();
        let tagged_idx = <&TaggedIdx>::try_from(&idx).unwrap();
        let bmap = Some(tagged_idx.union_bmap());
        DBResultSet::new(
            node.exprs
                .iter()
                .map(|expr| {
                    let vals = expr.eval(&EvalContext {
                        idx: &idx,
                        bmap: bmap.clone(),
                    });
                    vals
                })
                .collect(),
        )
    }
}

impl<'a> TryFrom<&'a BaseNode> for &'a TaggedBaseNodeInner {
    type Error = &'static str;

    fn try_from(value: &'a BaseNode) -> Result<Self, Self::Error> {
        match &value.inner {
            BaseNodeInner::Tagged(inner) => Ok(inner),
            _ => Err("Non-tagged plan node"),
        }
    }
}

impl<'a> TryFrom<&'a FilterNode> for &'a TaggedFilterNodeInner {
    type Error = &'static str;

    fn try_from(value: &'a FilterNode) -> Result<Self, Self::Error> {
        match &value.inner {
            FilterNodeInner::Tagged(inner) => Ok(inner),
            _ => Err("Non-tagged plan node"),
        }
    }
}

impl<'a> TryFrom<&'a JoinNode> for &'a TaggedJoinNodeInner {
    type Error = &'static str;

    fn try_from(value: &'a JoinNode) -> Result<Self, Self::Error> {
        match &value.inner {
            JoinNodeInner::Tagged(inner) => Ok(inner),
            _ => Err("Non-tagged plan node"),
        }
    }
}

impl DisBenefitCostRatio {
    fn new(benefit: f64, cost: Cost) -> Self {
        Self(FloatOrd(
            (benefit + BENEFIT_EPSILON) / (cost + COST_EPSILON),
        ))
    }
}

impl SpecBenefitCostRatio {
    fn new(benefit: f64, cost: Cost) -> Self {
        Self(FloatOrd(
            (benefit + BENEFIT_EPSILON) / (cost + COST_EPSILON),
        ))
    }
}

impl fmt::Display for TaggedBaseNodeInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let run_info = self.run_info.borrow();
        write!(
            f,
            "est_len={}, est_cost={}, {}",
            //"tags={{{}}}{}",
            //self.est_info
            //    .est_len
            //    .iter()
            //    .map(|(tag, tag_len)| format!("{}: {}", tag, tag_len))
            //    .join(", "),
            self.est_len(),
            self.est_cost(),
            run_info
        )
    }
}

impl fmt::Display for TaggedFilterNodeInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let run_info = self.run_info.borrow();
        write!(
            f,
            "est_len={}, est_cost={}, {}",
            //"tags={{{}}}, est_len={}, {}",
            //self.est_info
            //    .est_len
            //    .iter()
            //    .map(|(tag, tag_len)| format!("{}: {}", tag, tag_len))
            //    .join(", "),
            self.est_len(),
            self.est_cost(),
            run_info
        )
    }
}

impl fmt::Display for TaggedJoinNodeInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let run_info = self.run_info.borrow();
        write!(
            f,
            "est_len={}, est_cost={}, {}",
            //"tags={{{}}}, est_len={}, {}",
            //self.est_info
            //    .est_len
            //    .iter()
            //    .map(|(tag, tag_len)| format!(
            //        "Tag(\n{}\n): {}",
            //        tag.assign
            //            .iter()
            //            .map(|(pred, val)| format!("\t{} <- {}", val, pred))
            //            .join("\n"),
            //        tag_len
            //    ))
            //    .join("\n"),
            self.est_len(),
            self.est_cost(),
            run_info
        )
    }
}

impl fmt::Display for RunInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.was_run {
            write!(f, "")
        } else {
            write!(
                f,
                "[num_output={}, eval_time_ms={}]",
                self.num_output.iter().map(|(_, len)| len).sum::<usize>(),
                self.eval_time_ms
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_dir::DataDir;
    use crate::parse::Parser;
    use crate::stats::StatsReader;
    use crate::utils;
    use pretty_assertions::assert_eq;
    use std::path::PathBuf;

    #[test]
    fn test_build_join_paths() {
        let db_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("data")
            .join("test-data")
            .join("join-test");
        let working_dir = tempfile::tempdir().unwrap();
        let data_dir = DataDir::new(db_path).unwrap();
        let stats_reader = StatsReader::new(working_dir.path().to_owned());
        let core = TaggedPlannerCore::new(&stats_reader);
        let parser = Parser::new(&data_dir);

        fn stringify_join_paths(
            join_paths: &HashMap<
                Rc<FileTableRef>,
                HashMap<Rc<FileTableRef>, Vec<Vec<EqualityConstraint>>>,
            >,
        ) -> HashMap<String, HashMap<String, HashSet<Vec<String>>>> {
            join_paths
                .iter()
                .map(|(source, dest_paths)| {
                    (
                        source.to_string(),
                        dest_paths
                            .iter()
                            .map(|(dest, paths)| {
                                (
                                    dest.to_string(),
                                    paths
                                        .iter()
                                        .map(|path| {
                                            path.iter().map(|path| path.to_string()).collect()
                                        })
                                        .collect(),
                                )
                            })
                            .collect(),
                    )
                })
                .collect()
        }

        fn s(s: &str) -> String {
            s.to_string()
        }

        let query = utils::convert_to_one(parser.parse("select * from table1, table2, table3 where table1.id = table2.fid and table1.id = table3.fid").unwrap());
        let (join_graph, _) = plan_utils::split_join_constraints(query.filter.as_ref().unwrap());
        let join_paths = core.build_join_paths(&join_graph);

        assert_eq!(
            stringify_join_paths(&join_paths),
            HashMap::from([
                (
                    s("table1"),
                    HashMap::from([
                        (
                            s("table2"),
                            HashSet::from([vec![s("table1.id = table2.fid")]])
                        ),
                        (
                            s("table3"),
                            HashSet::from([vec![s("table1.id = table3.fid")]])
                        )
                    ])
                ),
                (
                    s("table2"),
                    HashMap::from([
                        (
                            s("table1"),
                            HashSet::from([vec![s("table1.id = table2.fid")]])
                        ),
                        (
                            s("table3"),
                            HashSet::from([vec![
                                s("table1.id = table2.fid"),
                                s("table1.id = table3.fid")
                            ]])
                        )
                    ])
                ),
                (
                    s("table3"),
                    HashMap::from([
                        (
                            s("table1"),
                            HashSet::from([vec![s("table1.id = table3.fid")]])
                        ),
                        (
                            s("table2"),
                            HashSet::from([vec![
                                s("table1.id = table3.fid"),
                                s("table1.id = table2.fid")
                            ]])
                        )
                    ])
                ),
            ])
        );

        let query = utils::convert_to_one(parser.parse("select * from table1, table2, table3, table4 where table1.id = table2.fid and table1.id = table3.fid and table1.id = table4.fid and table2.fid = table3.fid").unwrap());
        let (join_graph, _) = plan_utils::split_join_constraints(query.filter.as_ref().unwrap());
        let join_paths = core.build_join_paths(&join_graph);

        assert_eq!(
            stringify_join_paths(&join_paths),
            HashMap::from([
                (
                    s("table1"),
                    HashMap::from([
                        (
                            s("table2"),
                            HashSet::from([
                                vec![s("table1.id = table2.fid")],
                                vec![s("table1.id = table3.fid"), s("table2.fid = table3.fid")]
                            ])
                        ),
                        (
                            s("table3"),
                            HashSet::from([
                                vec![s("table1.id = table3.fid")],
                                vec![s("table1.id = table2.fid"), s("table2.fid = table3.fid")]
                            ])
                        ),
                        (
                            s("table4"),
                            HashSet::from([vec![s("table1.id = table4.fid")]])
                        )
                    ])
                ),
                (
                    s("table2"),
                    HashMap::from([
                        (
                            s("table1"),
                            HashSet::from([
                                vec![s("table1.id = table2.fid")],
                                vec![s("table2.fid = table3.fid"), s("table1.id = table3.fid")]
                            ])
                        ),
                        (
                            s("table3"),
                            HashSet::from([
                                vec![s("table1.id = table2.fid"), s("table1.id = table3.fid")],
                                vec![s("table2.fid = table3.fid")]
                            ])
                        ),
                        (
                            s("table4"),
                            HashSet::from([
                                vec![s("table1.id = table2.fid"), s("table1.id = table4.fid")],
                                vec![
                                    s("table2.fid = table3.fid"),
                                    s("table1.id = table3.fid"),
                                    s("table1.id = table4.fid")
                                ]
                            ]),
                        )
                    ])
                ),
                (
                    s("table3"),
                    HashMap::from([
                        (
                            s("table1"),
                            HashSet::from([
                                vec![s("table1.id = table3.fid")],
                                vec![s("table2.fid = table3.fid"), s("table1.id = table2.fid")]
                            ])
                        ),
                        (
                            s("table2"),
                            HashSet::from([
                                vec![s("table1.id = table3.fid"), s("table1.id = table2.fid")],
                                vec![s("table2.fid = table3.fid")]
                            ]),
                        ),
                        (
                            s("table4"),
                            HashSet::from([
                                vec![s("table1.id = table3.fid"), s("table1.id = table4.fid")],
                                vec![
                                    s("table2.fid = table3.fid"),
                                    s("table1.id = table2.fid"),
                                    s("table1.id = table4.fid")
                                ]
                            ]),
                        ),
                    ])
                ),
                (
                    s("table4"),
                    HashMap::from([
                        (
                            s("table1"),
                            HashSet::from([vec![s("table1.id = table4.fid")]])
                        ),
                        (
                            s("table2"),
                            HashSet::from([
                                vec![s("table1.id = table4.fid"), s("table1.id = table2.fid")],
                                vec![
                                    s("table1.id = table4.fid"),
                                    s("table1.id = table3.fid"),
                                    s("table2.fid = table3.fid")
                                ]
                            ]),
                        ),
                        (
                            s("table3"),
                            HashSet::from([
                                vec![s("table1.id = table4.fid"), s("table1.id = table3.fid")],
                                vec![
                                    s("table1.id = table4.fid"),
                                    s("table1.id = table2.fid"),
                                    s("table2.fid = table3.fid")
                                ]
                            ]),
                        ),
                    ])
                ),
            ])
        );
    }
}
