/// Contains the structs for predicates. Predicates are distinct from expressions in that they are
/// always expected to evaluate to true/false. Also, evaluating them on a column returns a bitmap
/// instead of a `DBVals` object.
///
/// Note that all equality comparisons (and hashing) between `Pred` node objects are done based on
/// the Id of the `Pred` object. However, if `parse::Parser::parse_pred` was used to create the
/// `Rc<Pred>` objects, then predicate nodes which are semantically equivalent should point to the
/// same object.
///
/// On the other hand, `PredAtom` always use the same semantic equivalence as `Expr` objects.
use crate::bitmap::{Bitmap, BitmapInt};
use crate::cost::{cost_factors, Cost};
use crate::engine::EXEC_INFO;
use crate::expr::{EvalContext, Expr};
use crate::file_table::{FileColSet, FileTableRefSet};
use crate::stats::StatsReader;
use crate::utils;
use either::Either;
use itertools::Itertools;
use log::info;
use snowflake::ProcessUniqueId;
use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::rc::{Rc, Weak};
use std::time::Instant;
use traversal::DftPre;

pub type PredId = ProcessUniqueId;

pub struct Pred {
    inner: PredInner,
    meta: PredMeta,
}

#[derive(Clone)]
enum PredInner {
    And(Vec<Rc<Pred>>),
    Or(Vec<Rc<Pred>>),
    Atom(PredAtom),
}

pub struct PredMeta {
    id: PredId,
    parents: RefCell<Vec<Weak<Pred>>>,
}

#[derive(Clone)]
pub struct PredAtom {
    expr: Expr,
    est: RefCell<PredEst>,
}

#[derive(Clone)]
pub struct PredEst {
    pub cost: Cost,
    pub selectivity: f64,
}

impl Pred {
    pub fn new_and(children: Vec<Rc<Self>>) -> Rc<Self> {
        let pred = Rc::new(Pred {
            inner: PredInner::And(children),
            meta: PredMeta {
                id: PredId::new(),
                parents: RefCell::new(vec![]),
            },
        });
        pred.set_children_parent();
        pred
    }

    pub fn new_or(children: Vec<Rc<Self>>) -> Rc<Self> {
        let pred = Rc::new(Pred {
            inner: PredInner::Or(children),
            meta: PredMeta {
                id: PredId::new(),
                parents: RefCell::new(vec![]),
            },
        });
        pred.set_children_parent();
        pred
    }

    pub fn new_atom(atom: PredAtom) -> Rc<Self> {
        Rc::new(Pred {
            inner: PredInner::Atom(atom),
            meta: PredMeta {
                id: PredId::new(),
                parents: RefCell::new(vec![]),
            },
        })
    }

    //pub fn print_parents(&self) {
    //    println!("Parent of {} is {:?}", self, self.parents());
    //    if let Ok(children) = self.try_iter_children() {
    //        for child in children {
    //            child.print_parents();
    //        }
    //    }
    //}

    /// Normalizes a predicate tree by ensuring all children of an `And` node are either `Or` or
    /// `Atom` nodes, and all children of an `Or` node are either `And~ or `Atom` nodes.
    pub fn normalize(self: Rc<Self>) -> Rc<Self> {
        fn _normalize(pred: &Rc<Pred>) -> Rc<Pred> {
            match &pred.inner {
                PredInner::Atom { .. } => pred.clone(),
                PredInner::And(children) | PredInner::Or(children) => {
                    let is_and = pred.is_and();
                    let children: Vec<_> = children
                        .iter()
                        .map(|child| {
                            let child = _normalize(child);
                            match (&child.inner, is_and) {
                                (PredInner::And(grandchildren), true)
                                | (PredInner::Or(grandchildren), false) => grandchildren.clone(),
                                _ => vec![child],
                            }
                        })
                        .flatten()
                        .collect();

                    assert!(!children.is_empty());

                    if children.len() == 1 {
                        utils::convert_to_one(children)
                    } else {
                        Rc::new(Pred {
                            inner: if is_and {
                                PredInner::And(children)
                            } else {
                                PredInner::Or(children)
                            },
                            meta: PredMeta {
                                id: PredId::new(),
                                parents: RefCell::new(vec![]),
                            },
                        })
                    }
                }
            }
        }

        fn sort_by_canon_str(pred: Rc<Pred>) -> Rc<Pred> {
            let is_and = pred.is_and();
            match &pred.inner {
                PredInner::Atom(_) => pred,
                PredInner::And(children) | PredInner::Or(children) => {
                    let children = children
                        .iter()
                        .map(|child| sort_by_canon_str(child.clone()))
                        .sorted_unstable_by_key(|child| child.to_string())
                        .collect();
                    Rc::new(Pred {
                        inner: if pred.is_and() {
                            PredInner::And(children)
                        } else {
                            PredInner::Or(children)
                        },
                        meta: PredMeta {
                            id: PredId::new(),
                            parents: RefCell::new(vec![]),
                        },
                    })
                }
            }
        }

        let pred = _normalize(&self);
        sort_by_canon_str(pred).deduplicate()
    }

    pub fn deduplicate(self: Rc<Pred>) -> Rc<Pred> {
        fn _deduplicate(
            pred: Rc<Pred>,
            pred_atom_map: &mut HashMap<PredAtom, Rc<Pred>>,
            pred_map: &mut HashMap<(bool, BTreeSet<Rc<Pred>>), Rc<Pred>>,
        ) -> Rc<Pred> {
            match &pred.inner {
                PredInner::Atom(atom) => {
                    if let Some(pred) = pred_atom_map.get(atom) {
                        pred.clone()
                    } else {
                        pred_atom_map.insert(atom.clone(), pred.clone());
                        pred
                    }
                }
                PredInner::And(children) | PredInner::Or(children) => {
                    let is_and = pred.is_and();
                    let children: BTreeSet<_> = children
                        .iter()
                        .map(|child| _deduplicate(child.clone(), pred_atom_map, pred_map))
                        .collect();
                    let key = (is_and, children);
                    if let Some(pred) = pred_map.get(&key) {
                        pred.clone()
                    } else {
                        let children = Vec::from_iter(key.1.clone());
                        let pred = Rc::new(Pred {
                            inner: if is_and {
                                PredInner::And(children)
                            } else {
                                PredInner::Or(children)
                            },
                            meta: PredMeta {
                                id: PredId::new(),
                                parents: RefCell::new(vec![]),
                            },
                        });
                        pred_map.insert(key, pred.clone());
                        pred
                    }
                }
            }
        }

        let mut pred_atom_map = HashMap::new();
        let mut pred_map = HashMap::new();
        let pred = _deduplicate(self, &mut pred_atom_map, &mut pred_map);
        pred.reset_all_parents();
        pred
    }

    fn reset_all_parents(self: &Rc<Pred>) {
        fn clear_all_parents(pred: &Rc<Pred>) {
            pred.meta.parents.borrow_mut().clear();
            if let Ok(child_iter) = pred.try_iter_children() {
                for child in child_iter {
                    clear_all_parents(child);
                }
            }
        }

        fn set_all_parents(pred: &Rc<Pred>) {
            if let Ok(child_iter) = pred.try_iter_children() {
                for child in child_iter {
                    set_all_parents(child);
                }
                pred.set_children_parent();
            }
        }

        clear_all_parents(self);
        set_all_parents(self);
    }

    //pub fn sort_by_canon_str(self: Rc<Self>) -> Rc<Self> {
    //    let is_and = self.is_and();
    //    match Rc::try_unwrap(self).unwrap() {
    //        atom @ Self::Atom { .. } => Rc::new(atom),
    //        //Self::Group(pred) => Self::Group(Box::new(pred.sort_by_canon_str())),
    //        Self::And { meta, mut children } | Self::Or { meta, mut children } => {
    //            children.sort_unstable_by_key(|child| child.to_string());
    //            let pred = Rc::new(if is_and {
    //                Self::And { meta, children }
    //            } else {
    //                Self::Or { meta, children }
    //            });
    //            pred.set_children_parent();
    //            pred
    //        }
    //    }
    //}

    pub fn try_iter_children(&self) -> Result<impl Iterator<Item = &Rc<Pred>>, &'static str> {
        match &self.inner {
            PredInner::And(children) | PredInner::Or(children) => Ok(children.iter()),
            PredInner::Atom(_) => Err("Atomic predicate has no children"),
        }
    }

    fn set_children_parent(self: &Rc<Self>) {
        if let Ok(child_iter) = self.try_iter_children() {
            for child in child_iter {
                let mut parents = child.meta.parents.borrow_mut();
                if !parents.iter().any(|parent| {
                    parent
                        .upgrade()
                        .map(|parent| &parent == self)
                        .unwrap_or(false)
                }) {
                    parents.push(Rc::downgrade(self));
                }
            }
        } else {
            panic!("Setting children of atom?");
        }
    }

    #[inline]
    pub fn is_atom(&self) -> bool {
        match &self.inner {
            PredInner::Atom { .. } => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_and(&self) -> bool {
        match &self.inner {
            PredInner::And { .. } => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_or(&self) -> bool {
        match &self.inner {
            PredInner::Or { .. } => true,
            _ => false,
        }
    }

    pub fn clear_parents(&self) {
        self.meta.parents.borrow_mut().clear();
    }

    /// Note that the `cost` returned by this function is on per record it is evaluated against and
    /// likely needs to multiplied against number of rows.
    pub fn est(&self) -> PredEst {
        match &self.inner {
            PredInner::Atom(atom) => atom.est.borrow().clone(),
            PredInner::And(children) => {
                let (cost, selectivity) =
                    children
                        .iter()
                        .fold((0., 1.), |(total_cost, total_selectivity), child| {
                            let est = child.est();
                            (
                                total_cost + total_selectivity * est.cost,
                                total_selectivity * est.selectivity,
                            )
                        });
                PredEst { cost, selectivity }
            }
            PredInner::Or(children) => {
                let (cost, selectivity) =
                    children
                        .iter()
                        .fold((0., 0.), |(total_cost, total_selectivity), child| {
                            let est = child.est();
                            (
                                total_cost + (1. - total_selectivity) * est.cost,
                                total_selectivity + est.selectivity * (1. - total_selectivity),
                            )
                        });
                PredEst { cost, selectivity }
            }
        }
    }

    pub fn eval(&self, context: &EvalContext) -> Bitmap {
        let smoothing_param = EXEC_INFO.with(|exec_info| exec_info.borrow().smoothing_param);
        match &self.inner {
            PredInner::And(children) => {
                let get_weight = |pred: &Pred| {
                    let est = pred.est();
                    (est.cost + smoothing_param) / (1. - est.selectivity + smoothing_param)
                };
                let mut children: Vec<&Rc<Pred>> = children.iter().collect();
                children
                    .sort_unstable_by(|a, b| get_weight(a).partial_cmp(&get_weight(b)).unwrap());

                let mut bmap = context.bmap.clone().unwrap_or_else(|| {
                    Rc::new(Bitmap::from_sorted_iter(0..context.idx.len() as BitmapInt).unwrap())
                });
                for child in children {
                    if bmap.is_empty() {
                        break;
                    }
                    bmap = Rc::new(child.eval(&EvalContext {
                        idx: context.idx,
                        bmap: Some(bmap),
                    }));
                }
                Rc::try_unwrap(bmap).unwrap()
            }
            PredInner::Or(children) => {
                let get_weight = |pred: &Pred| {
                    let est = pred.est();
                    (est.cost + smoothing_param) / (est.selectivity + smoothing_param)
                };
                let mut children: Vec<&Rc<Pred>> = children.iter().collect();
                children
                    .sort_unstable_by(|a, b| get_weight(a).partial_cmp(&get_weight(b)).unwrap());

                let total_bmap = context.bmap.clone().unwrap_or_else(|| {
                    Rc::new(Bitmap::from_sorted_iter(0..context.idx.len() as BitmapInt).unwrap())
                });

                let mut done = Bitmap::new();
                for child in children {
                    let todo = if done.is_empty() {
                        context.bmap.clone()
                    } else {
                        Some(Rc::new(total_bmap.as_ref() - &done))
                    };
                    if matches!(&todo, Some(todo) if todo.is_empty()) {
                        break;
                    }
                    done |= child.eval(&EvalContext {
                        idx: context.idx,
                        bmap: todo,
                    });
                }
                done
            }
            PredInner::Atom(atom) => atom.eval(context),
        }
    }

    /// Returns true if the predicate contains a multi-table predicate atom.
    pub fn has_multi_table_atom(self: &Rc<Self>) -> bool {
        self.iter_leaves().any(|pred| {
            let pred_atom = <&PredAtom>::try_from(pred).unwrap();
            pred_atom.has_multiple_table_refs()
        })
    }

    pub fn file_table_refs(self: &Rc<Self>) -> FileTableRefSet {
        self.iter_leaves()
            .map(|pred| {
                let pred_atom = <&PredAtom>::try_from(pred).unwrap();
                pred_atom.file_table_refs()
            })
            .concat()
    }

    pub fn file_cols(self: &Rc<Self>) -> FileColSet {
        self.iter_leaves()
            .map(|pred| {
                let pred_atom = <&PredAtom>::try_from(pred).unwrap();
                pred_atom.file_cols()
            })
            .concat()
    }

    pub fn iter_leaves<'a>(self: &'a Rc<Self>) -> impl Iterator<Item = &'a Rc<Self>> {
        DftPre::new(self, |pred| match &pred.inner {
            PredInner::And(children) | PredInner::Or(children) => Either::Left(children.iter()),
            PredInner::Atom(_) => Either::Right(std::iter::empty()),
        })
        .filter_map(|(_, pred)| pred.is_atom().then_some(pred))
    }

    pub fn update_stats(self: &Rc<Self>, stats_reader: &StatsReader) {
        for pred in self.iter_leaves() {
            let pred_atom = <&PredAtom>::try_from(pred).unwrap();
            if !pred_atom.has_multiple_table_refs() {
                pred_atom.update_selectivity(stats_reader.get_selectivity(pred_atom));
            }
        }
    }

    /// Returns the depth of the predicate tree. A single predicate atom has depth 0. Every level
    /// of AND/OR adds one to the depth
    pub fn depth(&self) -> usize {
        match &self.inner {
            PredInner::Atom(_) => 0,
            PredInner::And(children) | PredInner::Or(children) => {
                children
                    .iter()
                    .map(|child| child.depth())
                    .max()
                    .expect("And/Or node with no children?")
                    + 1
            }
        }
    }

    pub fn is_cnf(&self) -> bool {
        match &self.inner {
            PredInner::Atom(_) => true,
            PredInner::Or(_) => self.depth() == 1,
            PredInner::And(children) => children.iter().all(|child| match &child.inner {
                PredInner::Atom(_) => true,
                PredInner::Or(_) => child.depth() == 1,
                PredInner::And(_) => panic!("why was this not normalized"),
            }),
        }
    }

    fn id(&self) -> PredId {
        self.meta.id
    }

    pub fn parents(&self) -> Vec<Rc<Pred>> {
        self.meta
            .parents
            .borrow()
            .iter()
            .filter_map(|parent| parent.upgrade())
            .collect()
    }

    /// Gets the root by repeatedly calling parent of this `Pred`.
    pub fn get_root(self: &Rc<Self>) -> Rc<Pred> {
        std::iter::successors(Some(self.clone()), |node| node.parents().into_iter().next())
            .last()
            .unwrap()
    }

    /// Returns the ancestors of this predicate. Note that there may be more than one "line" of
    /// ancestors if the same predicate appears multiple times in the predicate tree.
    pub fn ancestors(&self) -> HashSet<Rc<Pred>> {
        let mut fringe = VecDeque::from(self.parents());
        let mut ancestors = HashSet::new();
        while !fringe.is_empty() {
            let ancestor = fringe.pop_front().unwrap();
            fringe.extend(ancestor.parents());
            ancestors.insert(ancestor);
        }
        ancestors
    }

    /// Returns separate "lines" to the root, one for each time the predicate appears in the
    /// predicate tree.
    pub fn ancestor_lines(&self) -> Vec<HashSet<Rc<Pred>>> {
        let mut lines = vec![];
        let mut fringe: VecDeque<_> = self
            .parents()
            .into_iter()
            .map(|parent| vec![parent])
            .collect();
        while !fringe.is_empty() {
            let mut line = fringe.pop_front().unwrap();
            let parents = line.last().unwrap().parents();
            if parents.is_empty() {
                lines.push(line);
            } else if parents.len() == 1 {
                line.push(utils::convert_to_one(parents));
                fringe.push_back(line);
            } else {
                for parent in parents {
                    let mut new_line = line.clone();
                    new_line.push(parent);
                    fringe.push_back(new_line);
                }
            }
        }

        lines
            .into_iter()
            .map(|line| HashSet::from_iter(line))
            .collect()
    }

    /// Returns the descendants of this predicate.
    pub fn descendants(&self) -> HashSet<Rc<Pred>> {
        if let Ok(child_iter) = self.try_iter_children() {
            let mut fringe = VecDeque::from_iter(child_iter);
            let mut descendants = HashSet::new();
            while !fringe.is_empty() {
                let descendant = fringe.pop_front().unwrap();
                if !descendants.contains(descendant) {
                    descendants.insert(descendant.clone());
                    if let Ok(child_iter) = descendant.try_iter_children() {
                        fringe.extend(child_iter);
                    }
                }
            }
            descendants
        } else {
            HashSet::new()
        }
    }

    /// Returns true if this predicate is an ancestor of `other`.
    pub fn is_ancestor_of(&self, other: &Self) -> bool {
        self.descendants().contains(other)
    }

    /// Returns true if this predicate is an descendant of `other`.
    pub fn is_descendant_of(&self, other: &Self) -> bool {
        other.descendants().contains(self)
    }

    /// Finds the predicate subexpression based on whether it matches the given str. This should
    /// mostly be used for testing and debugging purposes.
    pub fn find_by_str(self: &Rc<Self>, s: &str) -> Option<Rc<Pred>> {
        if &self.to_string() == s {
            Some(self.clone())
        } else {
            if let Ok(child_iter) = self.try_iter_children() {
                for child in child_iter {
                    let result = child.find_by_str(s);
                    if result.is_some() {
                        return result;
                    }
                }
            }
            None
        }
    }
}

impl PredAtom {
    pub fn new(expr: Expr) -> Self {
        let cost = expr
            .file_cols()
            .into_iter()
            .map(|col| col.data_type().size())
            .sum::<usize>() as Cost
            * cost_factors::PRED_COST_FACTOR;
        Self {
            expr,
            est: RefCell::new(PredEst {
                //cost: 1.,
                cost,
                selectivity: 1.,
            }),
        }
    }

    pub fn update_selectivity(&self, selectivity: f64) {
        self.est.borrow_mut().selectivity = selectivity;
    }

    pub fn update_cost(&self, cost: Cost) {
        self.est.borrow_mut().cost = cost;
    }

    pub fn expr(&self) -> &Expr {
        &self.expr
    }

    pub fn has_multiple_table_refs(&self) -> bool {
        self.file_table_refs().len() > 1
    }

    pub fn file_table_refs(&self) -> FileTableRefSet {
        self.expr.file_table_refs()
    }

    fn file_cols(&self) -> FileColSet {
        self.expr.file_cols()
    }

    pub fn eval(&self, context: &EvalContext) -> Bitmap {
        let now = Instant::now();
        EXEC_INFO.with(|exec_info| {
            let mut exec_info = exec_info.borrow_mut();
            exec_info.stats.num_pred_eval += if let Some(bmap) = &context.bmap {
                bmap.len() as u128
            } else {
                context.idx.len() as u128
            };
        });
        let prev_read_time_ms = EXEC_INFO.with(|exec_info| exec_info.borrow().stats.read_time_ms);
        let vals = self.expr.eval(context);
        let after_read_time_ms = EXEC_INFO.with(|exec_info| exec_info.borrow().stats.read_time_ms);
        if EXEC_INFO.with(|exec_info| exec_info.borrow().debug_times) {
            info!(
                "Evaluating pred atom {} took {} ms, on {} vals with selectivity {} read time {} ms",
                self,
                now.elapsed().as_millis(),
                vals.len(),
                vals.len() as f64 / context.idx.len() as f64,
                after_read_time_ms - prev_read_time_ms
            );
        }
        let now = Instant::now();
        let vals = AsRef::as_ref(&vals);
        let out = vals.to_bitmap(context.bmap.as_ref().map(|bmap| bmap.as_ref()));
        //println!(
        //    "Converting {} and reprojecting based on given bmap took {} ms",
        //    self,
        //    now.elapsed().as_millis()
        //);
        out
    }
}

impl<'a> TryFrom<&'a Rc<Pred>> for &'a PredAtom {
    type Error = String;

    fn try_from(value: &'a Rc<Pred>) -> Result<Self, Self::Error> {
        match &value.inner {
            PredInner::Atom(atom) => Ok(atom),
            _ => Err(format!("{} is not an atom", value.to_string())),
        }
    }
}

impl PartialEq for Pred {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}

impl Eq for Pred {}

impl Hash for Pred {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.id().hash(state);
    }
}

impl PartialOrd for Pred {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.id().partial_cmp(&other.id())
    }
}

impl Ord for Pred {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id().cmp(&other.id())
    }
}

impl PartialEq for PredAtom {
    fn eq(&self, other: &Self) -> bool {
        self.expr == other.expr
    }
}

impl Eq for PredAtom {}

impl Hash for PredAtom {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.expr.hash(state);
    }
}

pub struct PredAtomIter<'a>(Vec<&'a Rc<Pred>>);

impl<'a> Iterator for PredAtomIter<'a> {
    type Item = &'a Rc<Pred>;

    fn next(&mut self) -> Option<Self::Item> {
        while !self.0.is_empty() {
            let pred = self.0.pop()?;
            match &pred.inner {
                PredInner::Atom(_) => {
                    return Some(pred);
                }
                PredInner::And(children) | PredInner::Or(children) => {
                    self.0.extend(children);
                }
            }
        }
        None
    }
}

impl fmt::Debug for Pred {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl fmt::Display for Pred {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let is_and = self.is_and();
        match &self.inner {
            PredInner::Atom(atom) => write!(f, "{}", atom),
            PredInner::And(children) | PredInner::Or(children) => {
                let mut children: Vec<_> = children.iter().map(|c| c.to_string()).collect();
                children.sort_unstable();
                write!(
                    f,
                    "({})",
                    children.join(if is_and { " and " } else { " or " })
                )
            } //Self::Group(group) => write!(f, "{}", group.to_string()),
        }
    }
}

impl fmt::Debug for PredAtom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl fmt::Display for PredAtom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.expr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_dir::DataDir;
    use crate::parse::Parser;

    #[test]
    fn test_normalize() {
        let db_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("data")
            .join("test-data")
            .join("join-test");
        let data_dir = DataDir::new(db_path).unwrap();
        let parser = Parser::new(&data_dir);

        let query = utils::convert_to_one(
            parser
                .parse("select id from table1 where a = 1 and (b = 'a' and (c > 0 or id = 3))")
                .unwrap(),
        );
        let pred = query.filter.as_ref().unwrap();
        assert_eq!(pred.depth(), 2);
        assert!(pred.is_cnf());
        assert_eq!(
            query.to_string().as_str(),
            "SELECT table1.id FROM table1 WHERE ('a' = table1.b and (0 < table1.c or 3 = table1.id) and 1 = table1.a)"
        );

        let query = utils::convert_to_one(
            parser
                .parse("select id from table1 where a = 1 and (b = 'a' and ((c > 0 and (c < 10 and id = 4)) or id = 3))")
                .unwrap(),
        );
        let pred = query.filter.as_ref().unwrap();
        assert_eq!(pred.depth(), 3);
        assert_eq!(
            query.to_string().as_str(),
            "SELECT table1.id FROM table1 WHERE ('a' = table1.b and ((0 < table1.c and 4 = table1.id and table1.c < 10) or 3 = table1.id) and 1 = table1.a)"
        );
    }

    #[test]
    fn test_deduplicate() {
        let db_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("data")
            .join("test-data")
            .join("join-test");
        let data_dir = DataDir::new(db_path).unwrap();
        let parser = Parser::new(&data_dir);

        let query = utils::convert_to_one(
            parser
                .parse("select * from table1 where (a = 1 and b = 'a') or (a = 1 and b = 'b')")
                .unwrap(),
        );
        let pred = query.filter.unwrap();
        let mut child_iter = pred.try_iter_children().unwrap();
        let pred1 = child_iter
            .next()
            .unwrap()
            .find_by_str("1 = table1.a")
            .unwrap();
        let pred2 = child_iter
            .next()
            .unwrap()
            .find_by_str("1 = table1.a")
            .unwrap();

        assert!(Rc::ptr_eq(&pred1, &pred2));
        assert_eq!(pred1, pred2);

        let query = utils::convert_to_one(
            parser
                .parse("select * from table1 where (a = 1 or b = 'a') and (b = 'b' or (c > 0 and (a = 1 or b = 'a')))")
                .unwrap(),
        );
        let root = query.filter.unwrap();
        let mut child_iter = root.try_iter_children().unwrap();
        let pred1 = child_iter
            .next()
            .unwrap()
            .find_by_str("('a' = table1.b or 1 = table1.a)")
            .unwrap();
        let pred2 = child_iter
            .next()
            .unwrap()
            .find_by_str("('a' = table1.b or 1 = table1.a)")
            .unwrap();

        assert_eq!(root.depth(), 4);
        assert_eq!(pred1, pred2);
        assert!(Rc::ptr_eq(&pred1, &pred2));
        let parents = pred1.parents();
        assert_eq!(parents.len(), 2);
        assert!(
            (Rc::ptr_eq(&parents[0], &root)
                && parents[1].to_string() == "(('a' = table1.b or 1 = table1.a) and 0 < table1.c)")
                || (Rc::ptr_eq(&parents[1], &root)
                    && parents[0].to_string()
                        == "(('a' = table1.b or 1 = table1.a) and 0 < table1.c)")
        );
    }

    #[test]
    fn test_ancestor_lines() {
        let db_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("data")
            .join("test-data")
            .join("join-test");
        let data_dir = DataDir::new(db_path).unwrap();
        let parser = Parser::new(&data_dir);

        let query = utils::convert_to_one(
            parser
                .parse("select * from table1 where (a = 1 or b = 'a') and (b = 'b' or (c > 0 and (a = 1 or b = 'a')))")
                .unwrap(),
        );

        let root = query.filter.unwrap();
        let pred = root.find_by_str("1 = table1.a").unwrap();
        let lines = pred.ancestor_lines();
        assert_eq!(lines.len(), 2);
        let lines: HashSet<_> = lines
            .into_iter()
            .map(|line| {
                line.iter()
                    .map(|pred| pred.to_string())
                    .collect::<BTreeSet<_>>()
            })
            .collect();
        pretty_assertions::assert_eq!(
            lines,
            HashSet::from([
                BTreeSet::from([
                    "('a' = table1.b or 1 = table1.a)".to_string(),
                    "(('a' = table1.b or 1 = table1.a) and ('b' = table1.b or (('a' = table1.b or 1 = table1.a) and 0 < table1.c)))"
                        .to_string()
                ]),
                BTreeSet::from([
                    "('a' = table1.b or 1 = table1.a)".to_string(),
                    "(('a' = table1.b or 1 = table1.a) and 0 < table1.c)".to_string(),
                    "('b' = table1.b or (('a' = table1.b or 1 = table1.a) and 0 < table1.c))".to_string(),
                    "(('a' = table1.b or 1 = table1.a) and ('b' = table1.b or (('a' = table1.b or 1 = table1.a) and 0 < table1.c)))"
                        .to_string()
                ])
            ])
        );
    }
}
