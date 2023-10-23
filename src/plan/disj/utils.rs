use super::{JoinGraph, TableNodeMap, COST_EPSILON, SELECTIVITY_EPSILON};
use crate::expr::EqualityConstraint;
use crate::file_table::FileTableRef;
use crate::plan::PlanNode;
use crate::pred::{Pred, PredAtom};
use crate::utils;
use float_ord::FloatOrd;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

/// Inserts the given predicates into the table node map. If multiple predicates refer to the same
/// table ref, the order in which they are inserted is dependent on the given the given
/// `sort_preds` function.
pub(super) fn insert_preds_to_table_node_map(
    table_node_map: &mut TableNodeMap,
    preds: impl IntoIterator<Item = Rc<Pred>>,
    make_filter_node: impl Fn(Rc<PlanNode>, Rc<Pred>) -> Rc<PlanNode>,
    sort_preds: fn(&mut Vec<Rc<Pred>>),
) {
    let mut table_pred_map = HashMap::new();
    for pred in preds {
        table_pred_map
            .entry(utils::convert_to_one(pred.file_table_refs()))
            .or_insert(vec![])
            .push(pred);
    }
    for preds in table_pred_map.values_mut() {
        sort_preds(preds);
    }
    for (table_ref, preds) in table_pred_map {
        let node = table_node_map.remove(&table_ref).unwrap();
        let node = preds
            .into_iter()
            .fold(node, |node, pred| make_filter_node(node, pred));
        table_node_map.insert(table_ref, node);
    }
}

/// Sorts predicates in increasing selectivity/cost
pub fn sort_preds_by_selectivity_cost_ratio(preds: &mut Vec<Rc<Pred>>) {
    preds.sort_unstable_by_key(|pred| {
        let est = pred.est();
        FloatOrd((est.selectivity + SELECTIVITY_EPSILON) / (est.cost + COST_EPSILON))
    });
}

/// Tries to join the remaining edges based on a greedy algorithm (i.e., tries to perform the joins
/// that lead to the smallest output sizes first).
/// Partially joined components are allowed. The `table_node_map` should then point to these joined
/// nodes. In this case, `join_edges` must contain any edges of already joined components.
pub(super) fn plan_greedy_join(
    mut join_edges: HashSet<&EqualityConstraint>,
    mut table_node_map: HashMap<Rc<FileTableRef>, Rc<PlanNode>>,
    make_join_node: impl Fn(Rc<PlanNode>, Rc<PlanNode>, EqualityConstraint) -> Rc<PlanNode>,
) -> Rc<PlanNode> {
    while !join_edges.is_empty() {
        // NOTE: Maybe the nodes should be sorted by the ratio est_total_len / est_cost of the
        // node.
        let mut poss_joins: Vec<_> = join_edges
            .iter()
            .map(|join_edge| {
                let left_node = table_node_map.get(&join_edge.left_table_ref).unwrap();
                let right_node = table_node_map.get(&join_edge.right_table_ref).unwrap();
                make_join_node(left_node.clone(), right_node.clone(), (*join_edge).clone())
            })
            .collect();

        poss_joins.sort_unstable_by_key(|node| FloatOrd(node.est_len()));
        let new_node = poss_joins.into_iter().next().unwrap();

        let joined_table_refs = new_node.get_all_table_refs();
        join_edges.retain(|edge| {
            !joined_table_refs.contains(&edge.left_table_ref)
                || !joined_table_refs.contains(&edge.right_table_ref)
        });
        for table_ref in joined_table_refs {
            table_node_map.insert(table_ref, new_node.clone());
        }
    }

    table_node_map.into_values().next().unwrap()
}

/// Removes multi-table predicates from the tree and returns the join graph constructed from those
/// multi-table predicates and the predicate tree without those predicates. Note that the
/// predicates will be of a separate instance using the `clone_tree` method of the `Pred` object.
/// Assumes a conjunctive root, and that all multi-table predicates (i.e., join constraints) are
/// direct children of the root.
pub fn split_join_constraints(root: &Rc<Pred>) -> (JoinGraph, Option<Rc<Pred>>) {
    let mut join_graph = JoinGraph::new();

    let mut process_clause = |pred: &Rc<Pred>| {
        if pred.is_atom() {
            let pred_atom = <&PredAtom>::try_from(pred).unwrap();
            if pred_atom.has_multiple_table_refs() {
                let constraint = pred_atom.expr().as_equality_constraint().unwrap();
                join_graph
                    .entry(constraint.right_table_ref.clone())
                    .or_insert(vec![])
                    .push(constraint.clone());
                join_graph
                    .entry(constraint.left_table_ref.clone())
                    .or_insert(vec![])
                    .push(constraint);
                None
            } else {
                pred.clear_parents();
                Some(pred.clone())
            }
        } else {
            assert!(pred.is_or());
            assert!(
                pred.try_iter_children()
                    .unwrap()
                    .all(|child| !child.has_multi_table_atom()),
                "Unexpected filter form. ORs of join constraints"
            );
            pred.clear_parents();
            Some(pred.clone())
        }
    };

    let pred = if root.is_atom() || root.is_or() {
        process_clause(root)
    } else {
        let clauses: Vec<_> = root
            .try_iter_children()
            .unwrap()
            .filter_map(|child| process_clause(child))
            .collect();
        if clauses.is_empty() {
            None
        } else {
            Some(Pred::new_and(clauses))
        }
    };

    (join_graph, pred.map(|pred| pred.normalize()))
}

/// Splits a conjunctive root into its atomic and non-atomic (i.e., has disjunctions) clauses.
/// If either a atomic or disjunctive-root predicate is passed in, it is assumed it is the sole
/// child of the conjunctive root, and parses it accordingly.
///
/// NOTE: For the purposes of this function, even non-atomic clauses which refer to only one file
/// table ref are considered atomic clauses.
pub fn split_conj_root(pred_root: &Rc<Pred>) -> (Vec<Rc<Pred>>, Vec<Rc<Pred>>) {
    fn clause_is_atomic(clause: &Rc<Pred>) -> bool {
        clause.is_atom() || (clause.file_table_refs().len() == 1)
    }

    if clause_is_atomic(pred_root) {
        (vec![pred_root.clone()], vec![])
    } else if pred_root.is_or() {
        (vec![], vec![pred_root.clone()])
    } else {
        let mut atomic_clauses = vec![];
        let mut disj_clauses = vec![];
        for child in pred_root.try_iter_children().unwrap() {
            if clause_is_atomic(child) {
                atomic_clauses.push(child.clone());
            } else {
                assert!(child.is_or(), "Un-normalized predicate");
                disj_clauses.push(child.clone());
            }
        }
        (atomic_clauses, disj_clauses)
    }
}

/// Calculates the specifying benefit of applying predicate atom `pred` with respect to the
/// predicates listed in `on_preds`. Here, `on_preds` is an iterator of `(&Rc<Pred>, usize)`
/// pairs, in which the second element of the pair is the number of tuples we would evaluate the
/// predicate on. The specifying benefit is calculated by, for each predicate in `on_pred`,
/// checking to see whether that predicate is a descendant of one of `pred`'s parents. If so, we
/// estimate the number of tuples we can avoid evaluating with the `on_preds` predicate by
/// evaluating `pred` first.
///
/// Note that this a conservative estimate of a predicate's specification benefit. It only
/// calculates the benefits that a single predicate has, not the benefits that it would have in
/// conjunction with other predicates.
pub fn calc_spec_benefit<'a>(
    pred: &Rc<Pred>,
    on_preds: impl Iterator<Item = (&'a Rc<Pred>, f64)>,
) -> f64 {
    let parents = pred.parents();
    if parents.is_empty() {
        0.
    } else {
        on_preds
            .map(|(on_pred, num_tuples)| {
                // To calculate the specifying benefit of `pred` on `on_pred`, we look at each
                // ancestor line of `on_pred`. In each ancestor line of `on_pred`, we check to see
                // if a parent of `pred` appears. If so, it is possible that `on_pred` gets some
                // benefit from `pred`. To calculate the exact benefit, we have to check whether
                // the parent of `pred` is an AND node or an OR node. It is possible to meet
                // multiple parents of `pred` in a single ancestor line of `on_pred`. If every
                // parent is of AND type, then we get the benefits of AND node (once). If both AND
                // and OR types of parents exist along the ancestor line, then we do not need to
                // evaluate `on_pred` after `pred` according to this ancestor line. This is because
                // for any tuple, if `pred` evaluates to true, then there is no need to evaluate
                // any children of `pred`'s OR parent, and if `pred` evaluates to false, then there
                // is no need to evaluate any children of `pred`'s AND parent.
                //
                // For each ancestor line, we check to see if we get none, AND, OR, or both
                // benefits of `pred`. In general, we look at the ancestor line with the "minimum"
                // amount of benefits. If there is an ancestor line with no benefits, then overall,
                // `on_pred` gets no specifying benefit from `pred`. If all ancestor lines get an
                // AND benefit, then `on_pred` gets the AND benefit from `pred`. If there are some
                // ancestor lines with AND benefits and some with OR benefits, this is the same as
                // getting no benefit since `pred` evaluating to true gets rid of the `on_pred`
                // which are descendants of the OR parent but not descendants of the AND parent,
                // while `pred` evaluating to false gets rid of the `on_pred` which are descendants
                // of AND parents but not the descendants of the OR parent.

                let (and_benefit, or_benefit) = on_pred
                    .ancestor_lines()
                    .into_iter()
                    .map(|line| {
                        parents
                            .iter()
                            .fold((false, false), |(found_and, found_or), parent| {
                                if line.contains(parent) {
                                    if parent.is_and() {
                                        (true, found_or)
                                    } else {
                                        (found_and, true)
                                    }
                                } else {
                                    (found_and, found_or)
                                }
                            })
                    })
                    .reduce(|acc, (found_and, found_or)| (acc.0 && found_and, acc.1 && found_or))
                    .unwrap();

                let selectivity = pred.est().selectivity;
                match (and_benefit, or_benefit) {
                    (true, true) => num_tuples as f64,
                    (true, false) => (1. - selectivity) * num_tuples as f64,
                    (false, true) => selectivity * num_tuples as f64,
                    (false, false) => 0.,
                }
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_dir::DataDir;
    use crate::parse::Parser;
    use crate::pred::PredAtom;
    use crate::utils;

    #[test]
    fn test_calc_spec_benefit() {
        let db_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("data")
            .join("test-data")
            .join("join-test");
        let data_dir = DataDir::new(db_path).unwrap();
        let parser = Parser::new(&data_dir);

        // Basic And benefit
        {
            let query = "select table1.id from table1 where a = 3 and (c > 0 or id = 5)";
            let query = utils::convert_to_one(parser.parse(query).unwrap());

            let pred_root = query.filter.unwrap();
            let pred = pred_root.find_by_str("3 = table1.a").unwrap();
            let pred_atom = <&PredAtom>::try_from(&pred).unwrap();
            pred_atom.update_selectivity(2. / 8.);

            let on_pred1 = pred_root.find_by_str("0 < table1.c").unwrap();
            let on_pred2 = pred_root.find_by_str("5 = table1.id").unwrap();
            assert_eq!(
                calc_spec_benefit(&pred, [(&on_pred1, 100.)].into_iter()),
                75.
            );
            assert_eq!(
                calc_spec_benefit(&pred, [(&on_pred1, 100.), (&on_pred2, 400.)].into_iter()),
                375.
            );
        }

        // Basic Or benefit
        {
            let query = "select table1.id from table1 where a = 3 or (c > 0 or id = 5)";
            let query = utils::convert_to_one(parser.parse(query).unwrap());

            let pred_root = query.filter.unwrap();
            let pred = pred_root.find_by_str("3 = table1.a").unwrap();
            let pred_atom = <&PredAtom>::try_from(&pred).unwrap();
            pred_atom.update_selectivity(2. / 8.);

            let on_pred1 = pred_root.find_by_str("0 < table1.c").unwrap();
            let on_pred2 = pred_root.find_by_str("5 = table1.id").unwrap();
            assert_eq!(
                calc_spec_benefit(&pred, [(&on_pred1, 100.)].into_iter()),
                25.
            );
            assert_eq!(
                calc_spec_benefit(&pred, [(&on_pred1, 100.), (&on_pred2, 400.)].into_iter()),
                125.
            );
        }

        // And + Or benefit
        {
            let query = "select table1.id from table1 where (a = 3 or (c > 0 or id = 5)) and a = 3";
            let query = utils::convert_to_one(parser.parse(query).unwrap());

            let pred_root = query.filter.unwrap();
            let pred = pred_root.find_by_str("3 = table1.a").unwrap();
            let pred_atom = <&PredAtom>::try_from(&pred).unwrap();
            pred_atom.update_selectivity(2. / 8.);

            let on_pred1 = pred_root.find_by_str("0 < table1.c").unwrap();
            assert_eq!(
                calc_spec_benefit(&pred, [(&on_pred1, 100.)].into_iter()),
                100.
            );
        }

        // Multiple Or benefits
        {
            let query = "select table1.id from table1 where (a = 3 or (c > 0 or id = 5)) and (a = 3 or c > 0)";
            let query = utils::convert_to_one(parser.parse(query).unwrap());

            let pred_root = query.filter.unwrap();
            let pred = pred_root.find_by_str("3 = table1.a").unwrap();
            let pred_atom = <&PredAtom>::try_from(&pred).unwrap();
            pred_atom.update_selectivity(2. / 8.);

            let on_pred1 = pred_root.find_by_str("0 < table1.c").unwrap();
            assert_eq!(
                calc_spec_benefit(&pred, [(&on_pred1, 100.)].into_iter()),
                25.
            );
        }

        // Mixed And + Or benefits
        {
            let query = "select table1.id from table1 where (a = 3 or (c > 0 or id = 5)) and (id = 5 or (a = 3 and c > 0))";
            let query = utils::convert_to_one(parser.parse(query).unwrap());

            let pred_root = query.filter.unwrap();
            let pred = pred_root.find_by_str("3 = table1.a").unwrap();
            let pred_atom = <&PredAtom>::try_from(&pred).unwrap();
            pred_atom.update_selectivity(2. / 8.);

            let on_pred1 = pred_root.find_by_str("0 < table1.c").unwrap();
            assert_eq!(
                calc_spec_benefit(&pred, [(&on_pred1, 100.)].into_iter()),
                0.
            );
        }

        // One line has no benefit
        {
            let query = "select table1.id from table1 where ((a = 3 or (c > 0 or id = 5)) and a = 3) or (id = 5 or c > 0) ";
            let query = utils::convert_to_one(parser.parse(query).unwrap());

            let pred_root = query.filter.unwrap();
            let pred = pred_root.find_by_str("3 = table1.a").unwrap();
            let pred_atom = <&PredAtom>::try_from(&pred).unwrap();
            pred_atom.update_selectivity(2. / 8.);

            let on_pred1 = pred_root.find_by_str("0 < table1.c").unwrap();
            assert_eq!(
                calc_spec_benefit(&pred, [(&on_pred1, 100.)].into_iter()),
                0.
            );
        }
    }
}
