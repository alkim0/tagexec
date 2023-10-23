//! This file contains the code for dealing with tags. Creating new tags should be done through a
//! `TagManager` object which keeps track of all tags and ensures that there exists a unique tag
//! for each set of predicate assignments. Whenever a new tag is created (either by the
//! `TagManager` or by combining existing tags), the assignments are propagated upwards and only
//! the assignments for the predicate nodes closest to the root are recorded. For example, if the
//! predicate has the form (A | B) ^ C, and A = true, then (A | B) = true is the assignment that is
//! saved.

use crate::pred::Pred;
use itertools::Itertools;
use rustc_hash::FxHashMap;
use snowflake::ProcessUniqueId;
use std::cell::RefCell;
use std::collections::{BTreeMap, HashSet, VecDeque};
use std::fmt;
use std::hash::{BuildHasherDefault, Hash};
use std::rc::Rc;

type TagId = ProcessUniqueId;

type TagAssignMap = BTreeMap<Rc<Pred>, bool>;

pub struct Tag {
    id: TagId,
    pub assign: TagAssignMap,
    manager: Rc<TagManager>,
}

pub struct TagManager {
    tag_map: RefCell<FxHashMap<TagAssignMap, Rc<Tag>>>,
}

impl TagManager {
    pub fn new() -> Rc<Self> {
        Rc::new(Self {
            tag_map: RefCell::new(FxHashMap::with_hasher(BuildHasherDefault::default())),
        })
    }

    pub fn empty_tag(self: &Rc<Self>) -> Rc<Tag> {
        let mut tag_map = self.tag_map.borrow_mut();
        tag_map
            .entry(TagAssignMap::new())
            .or_insert_with(|| {
                Rc::new(Tag {
                    id: TagId::new(),
                    assign: TagAssignMap::new(),
                    manager: self.clone(),
                })
            })
            .clone()
    }

    pub fn new_tag(
        self: &Rc<Self>,
        assign: impl IntoIterator<Item = (Rc<Pred>, bool)>,
    ) -> Option<Rc<Tag>> {
        Tag::propagate_assignments(assign).map(|assign| self.tag_from_assign(assign))
    }

    /// Checks to see if the assignment exists in tag_map and returns the existing tag if it
    /// exists. Otherwise, a new tag is created and that that is returned.
    fn tag_from_assign(self: &Rc<Self>, assign: TagAssignMap) -> Rc<Tag> {
        let mut tag_map = self.tag_map.borrow_mut();
        tag_map
            .entry(assign)
            .or_insert_with_key(|assign| {
                Rc::new(Tag {
                    id: TagId::new(),
                    assign: assign.clone(),
                    manager: self.clone(),
                })
            })
            .clone()
    }
}

impl Tag {
    /// Creates a new tag based on the old tag's assignments concatenated with the given
    /// assignments. Also propagates the assignments up the predicate tree, so ancestors with
    /// static values due to the assignments are also stored in the `assign` hash map. If root has
    /// a static value of false, None is returned instead.
    pub fn concat(
        &self,
        new_assign: impl IntoIterator<Item = (Rc<Pred>, bool)>,
    ) -> Option<Rc<Self>> {
        Tag::propagate_assignments(self.assign.clone().into_iter().chain(new_assign))
            .map(|assign| self.manager.tag_from_assign(assign))
    }

    /// Creates a new tag based on combining this tag with another. Also propagates the assignments
    /// up the predicate tree, so ancestors with static values due to the assignments are also
    /// stored in the `assign` hash map. If root has a static value of false, None is returned
    /// instead.
    // NOTE: It should not be possible for tags to conflict (i.e., for two different tags to have
    // different values assigned to the same predicate node), especially if we assume predicates
    // are unique in the predicate tree. If predicates are not unique, we may need to alter
    // `propagate_assignments` to check for these conflicts.
    pub fn combine(&self, other: &Self) -> Option<Rc<Self>> {
        Tag::propagate_assignments(self.assign.clone().into_iter().chain(other.assign.clone()))
            .map(|assign| self.manager.tag_from_assign(assign))
    }

    fn propagate_assignments(
        assign: impl IntoIterator<Item = (Rc<Pred>, bool)>,
    ) -> Option<TagAssignMap> {
        fn clear_assignments(assign: &mut BTreeMap<Rc<Pred>, bool>, pred_root: &Rc<Pred>) {
            // Returns all predicates which have been assigned and do not have any ancestors that
            // are assigned.
            fn get_topmost_assigned_preds(
                pred: &Rc<Pred>,
                assign: &BTreeMap<Rc<Pred>, bool>,
            ) -> HashSet<Rc<Pred>> {
                if assign.contains_key(pred) {
                    HashSet::from([pred.clone()])
                } else {
                    pred.try_iter_children()
                        .map(|child_iter| {
                            child_iter
                                .map(|child| get_topmost_assigned_preds(child, assign))
                                .concat()
                        })
                        .unwrap_or(HashSet::new())
                }
            }

            let topmost_assigned_preds = get_topmost_assigned_preds(pred_root, assign);
            assign.retain(|pred, val| topmost_assigned_preds.contains(pred));
        }

        let mut fringe: VecDeque<_> = assign.into_iter().collect();
        let mut assign = TagAssignMap::new();
        //let mut assign_counter = FxHashMap::with_hasher(BuildHasherDefault::default());
        while !fringe.is_empty() {
            let (pred, val) = fringe.pop_front().unwrap();
            if assign.contains_key(&pred) {
                continue;
            }

            let parents = pred.parents();
            assign.insert(pred, val);

            if parents.is_empty() {
                if val {
                    break;
                } else {
                    return None;
                }
            } else {
                for parent in parents {
                    if parent.is_and() && !val {
                        fringe.push_back((parent, false));
                    } else if parent.is_or() && val {
                        fringe.push_back((parent, true));
                    } else {
                        // FIXME: This should contain an explicit check to true/false depending on
                        // the parent, otherwise this may not work if the fringe contains
                        // assignments for all children but some of them true/false
                        if parent
                            .try_iter_children()
                            .unwrap()
                            .all(|child| assign.contains_key(child))
                        {
                            fringe.push_back((parent, val));
                        }
                    }
                }
            }

            //    match (pred.parent(), val) {
            //        (None, false) => {
            //            return None;
            //        }
            //        (None, true) => {
            //            assign.insert(pred, val);
            //            break;
            //        }
            //        (Some(parent), false) if parent.is_and() => {
            //            fringe.push_back((parent, false));
            //        }
            //        (Some(parent), true) if parent.is_or() => {
            //            fringe.push_back((parent, true));
            //        }
            //        (Some(parent), val) => {
            //            let num_children = match parent.as_ref() {
            //                Pred::And { children, .. } => children.len(),
            //                Pred::Or { children, .. } => children.len(),
            //                Pred::Atom { .. } => panic!("wtf"),
            //            };
            //            let counter = assign_counter.entry(parent.clone()).or_insert(num_children);
            //            *counter -= 1;
            //            if *counter == 0 {
            //                //if let Pred::And { children, .. } | Pred::Or { children, .. } =
            //                //    parent.as_ref()
            //                //{
            //                //    for child in children {
            //                //        assign.remove(child);
            //                //    }
            //                //}
            //                fringe.push_back((parent, val));
            //            } else {
            //                assign.insert(pred, val);
            //            }
            //        }
            //    }
        }

        if !assign.is_empty() {
            let root = assign.keys().next().unwrap().get_root();
            clear_assignments(&mut assign, &root);
        }

        Some(assign)
    }
}

impl fmt::Display for Tag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl fmt::Debug for Tag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            write!(
                f,
                "Tag(\n{}\n)",
                self.assign
                    .iter()
                    .map(|(pred, val)| format!("    {}({})", val, pred))
                    .join("\n")
            )
        } else {
            write!(
                f,
                "Tag({})",
                self.assign
                    .iter()
                    .map(|(pred, val)| format!("{}({})", val, pred))
                    .join(", ")
            )
        }
    }
}

impl PartialEq for Tag {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Tag {}

impl Hash for Tag {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

//#[derive(Debug, PartialEq, Eq, Hash, Clone, PartialOrd, Ord)]
//pub struct Tag(TagId);
////pub struct Tag(String);
//
//pub type TagSet = BTreeSet<Tag>;
//
//// TODO FIXME: Isn't this just a bunch of assignments. Do I need complex and/or/minus/not?
//pub enum TagExpr {
//    And(Vec<TagExpr>),
//    Or(Vec<TagExpr>),
//    Minus {
//        from: Box<TagExpr>,
//        to: Box<TagExpr>,
//    },
//    Not(Box<TagExpr>),
//    Atom(Tag),
//}
//
//pub struct TagBuilder {
//    tags: FxHashMap<Tag, String>,
//}
//
//impl TagBuilder {
//    pub fn new() -> Self {
//        Self {
//            tags: HashMap::with_hasher(BuildHasherDefault::default()),
//        }
//    }
//
//    pub fn new_tag(&mut self, s: String) -> Tag {
//        let tag = Tag(TagId::new());
//        self.tags.insert(tag.clone(), s);
//        tag
//    }
//
//    pub fn tag_as_str(&self, tag: &Tag) -> &'_ str {
//        self.tags.get(tag).unwrap()
//    }
//}

//pub struct TagBitsConverter {
//    tag_to_bit_map: BTreeMap<Tag, u64>, // Maps from Tag to the bit mask for that tag
//    bit_to_tag_map: BTreeMap<u64, Tag>, // Maps from bit mask to Tag
//}
//
//impl TagBitsConverter {
//    pub fn new(all_tags: &TagSet) -> Self {
//        let mut tag_to_bit_map = BTreeMap::new();
//        let mut bit_to_tag_map = BTreeMap::new();
//        for (i, tag) in all_tags.iter().enumerate() {
//            tag_to_bit_map.insert(tag.clone(), (1 as u64) << i);
//            bit_to_tag_map.insert((1 as u64) << i, tag.clone());
//        }
//
//        Self {
//            tag_to_bit_map,
//            bit_to_tag_map,
//        }
//    }
//
//    pub fn tags_to_bits<'a>(&self, tags: impl IntoIterator<Item = &'a Tag>) -> u64 {
//        let mut out = 0;
//        for tag in tags {
//            out |= self.tag_to_bit_map.get(tag).unwrap();
//        }
//        out
//    }
//
//    pub fn bits_to_tags(&self, tag_bits: u64) -> Tags {
//        let mut out = 0;
//        for tag in tags {
//            out |= self.tag_to_bit_map.get(tag).unwrap();
//        }
//        out
//    }
//}

//impl Tag {
//    pub fn new(s: String) -> Self {
//        Self(s)
//    }
//}

//impl fmt::Display for Tag {
//    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//        write!(f, "\"{}\"", self.0)
//    }
//}

//pub type TagId = ProcessUniqueId;
//
//pub enum TagConstraint {
//    Filter(TagExpr),
//    Join {
//        left: Option<TagExpr>,
//        right: Option<TagExpr>,
//    },
//}
//
//pub enum TagExpr {
//    And(Vec<TagId>),
//    Or(Vec<TagId>),
//}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_dir::DataDir;
    use crate::parse::Parser;
    use crate::utils;
    use std::path::PathBuf;

    #[test]
    fn test_propagate_assignments() {
        let db_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("data")
            .join("test-data")
            .join("join-test");
        let data_dir = DataDir::new(db_path).unwrap();
        let parser = Parser::new(&data_dir);

        // Singleton
        let query =
            utils::convert_to_one(parser.parse("select * from table1 where a = 1").unwrap());
        let pred = query.filter.unwrap();
        let tag_manager = TagManager::new();
        assert_eq!(
            tag_manager.new_tag([(pred.clone(), true)]).unwrap().assign,
            BTreeMap::from([(pred.clone(), true)])
        );
        assert_eq!(tag_manager.new_tag([(pred.clone(), false)]), None);

        // Depth-1 AND node
        let query = utils::convert_to_one(
            parser
                .parse("select * from table1 where a = 1 and a = 2 and a = 3")
                .unwrap(),
        );
        let root = query.filter.unwrap();
        let tag_manager = TagManager::new();
        let (a1, a2, a3) = (
            root.find_by_str("1 = table1.a").unwrap(),
            root.find_by_str("2 = table1.a").unwrap(),
            root.find_by_str("3 = table1.a").unwrap(),
        );
        assert_eq!(tag_manager.new_tag([(a2.clone(), false)]), None);
        assert_eq!(
            tag_manager.new_tag([(a3.clone(), true)]).unwrap().assign,
            BTreeMap::from([(a3.clone(), true)])
        );
        assert_eq!(
            tag_manager
                .new_tag([(a1.clone(), true), (a2.clone(), true), (a3.clone(), true)])
                .unwrap()
                .assign,
            BTreeMap::from([(root.clone(), true)])
        );

        // Depth-1 OR node
        let query = utils::convert_to_one(
            parser
                .parse("select * from table1 where a = 1 or a = 2 or a = 3")
                .unwrap(),
        );
        let root = query.filter.unwrap();
        let tag_manager = TagManager::new();
        let (a1, a2, a3) = (
            root.find_by_str("1 = table1.a").unwrap(),
            root.find_by_str("2 = table1.a").unwrap(),
            root.find_by_str("3 = table1.a").unwrap(),
        );
        assert_eq!(
            tag_manager.new_tag([(a2.clone(), true)]).unwrap().assign,
            BTreeMap::from([(root.clone(), true)])
        );
        assert_eq!(
            tag_manager.new_tag([(a3.clone(), false)]).unwrap().assign,
            BTreeMap::from([(a3.clone(), false)])
        );
        assert_eq!(
            tag_manager.new_tag([
                (a1.clone(), false),
                (a2.clone(), false),
                (a3.clone(), false)
            ]),
            None
        );

        // CNF
        let query = utils::convert_to_one(
            parser
                .parse("select * from table1 where a = 1 and (a = 2 or a = 3) and (a = 4 or a = 5)")
                .unwrap(),
        );
        let root = query.filter.unwrap();
        let tag_manager = TagManager::new();
        let (a1, a2, a3, a4, a5) = (
            root.find_by_str("1 = table1.a").unwrap(),
            root.find_by_str("2 = table1.a").unwrap(),
            root.find_by_str("3 = table1.a").unwrap(),
            root.find_by_str("4 = table1.a").unwrap(),
            root.find_by_str("5 = table1.a").unwrap(),
        );
        assert_eq!(tag_manager.new_tag([(a1.clone(), false)]), None);
        assert_eq!(
            tag_manager.new_tag([(a2.clone(), true)]).unwrap().assign,
            BTreeMap::from([(utils::convert_to_one(a2.parents()), true)])
        );
        assert_eq!(
            tag_manager.new_tag([(a2.clone(), false), (a3.clone(), false)]),
            None
        );
        assert_eq!(
            tag_manager
                .new_tag([(a1.clone(), true), (a2.clone(), true), (a4.clone(), true)])
                .unwrap()
                .assign,
            BTreeMap::from([(root.clone(), true)])
        );
        //assert_eq!(
        //    Tag::new([(a1.clone(), false)]).assign,
        //    FxHashMap::from_iter([(a1.clone(), false), (root.clone(), false)]),
        //);
        //assert_eq!(
        //    Tag::new([(a2.clone(), true)]).assign,
        //    FxHashMap::from_iter([(a2.clone(), true), (a2.parent().unwrap(), true)]),
        //);
        //assert_eq!(
        //    Tag::new([(a2.clone(), false), (a3.clone(), false)]).assign,
        //    FxHashMap::from_iter([
        //        (a2.clone(), false),
        //        (a3.clone(), false),
        //        (a2.parent().unwrap(), false),
        //        (root.clone(), false)
        //    ]),
        //);
        //assert_eq!(
        //    Tag::new([(a1.clone(), true), (a2.clone(), true), (a4.clone(), true)]).assign,
        //    FxHashMap::from_iter([
        //        (a1.clone(), true),
        //        (a2.clone(), true),
        //        (a4.clone(), true),
        //        (a2.parent().unwrap(), true),
        //        (a4.parent().unwrap(), true),
        //        (root.clone(), true)
        //    ]),
        //);

        // DNF
        let query = utils::convert_to_one(
            parser
                .parse("select * from table1 where a = 1 or (a = 2 and a = 3) or (a = 4 and a = 5)")
                .unwrap(),
        );
        let root = query.filter.unwrap();
        let tag_manager = TagManager::new();
        let (a1, a2, a3, a4, a5) = (
            root.find_by_str("1 = table1.a").unwrap(),
            root.find_by_str("2 = table1.a").unwrap(),
            root.find_by_str("3 = table1.a").unwrap(),
            root.find_by_str("4 = table1.a").unwrap(),
            root.find_by_str("5 = table1.a").unwrap(),
        );
        assert_eq!(
            tag_manager.new_tag([(a1.clone(), true)]).unwrap().assign,
            BTreeMap::from([(root.clone(), true)])
        );
        assert_eq!(
            tag_manager.new_tag([(a2.clone(), false)]).unwrap().assign,
            BTreeMap::from([(utils::convert_to_one(a2.parents()), false)])
        );
        assert_eq!(
            tag_manager
                .new_tag([(a2.clone(), true), (a3.clone(), true)])
                .unwrap()
                .assign,
            BTreeMap::from([(root.clone(), true)])
        );
        assert_eq!(
            tag_manager.new_tag([
                (a1.clone(), false),
                (a2.clone(), false),
                (a4.clone(), false)
            ]),
            None
        );
        //assert_eq!(
        //    Tag::new([(a1.clone(), true)]).assign,
        //    FxHashMap::from_iter([(a1.clone(), true), (root.clone(), true)]),
        //);
        //assert_eq!(
        //    Tag::new([(a2.clone(), false)]).assign,
        //    FxHashMap::from_iter([(a2.clone(), false), (a2.parent().unwrap(), false)]),
        //);
        //assert_eq!(
        //    Tag::new([(a2.clone(), true), (a3.clone(), true)]).assign,
        //    FxHashMap::from_iter([
        //        (a2.clone(), true),
        //        (a3.clone(), true),
        //        (a2.parent().unwrap(), true),
        //        (root.clone(), true)
        //    ]),
        //);
        //assert_eq!(
        //    Tag::new([
        //        (a1.clone(), false),
        //        (a2.clone(), false),
        //        (a4.clone(), false)
        //    ])
        //    .assign,
        //    FxHashMap::from_iter([
        //        (a1.clone(), false),
        //        (a2.clone(), false),
        //        (a4.clone(), false),
        //        (a2.parent().unwrap(), false),
        //        (a4.parent().unwrap(), false),
        //        (root.clone(), false)
        //    ]),
        //);

        // Combining test for assigned root + assigned pred atom
        let query = utils::convert_to_one(
            parser
                .parse("select * from table1 where a = 1 or (a = 2 and a = 3)")
                .unwrap(),
        );
        let root = query.filter.unwrap();
        let tag_manager = TagManager::new();
        let (a1, a2, a3) = (
            root.find_by_str("1 = table1.a").unwrap(),
            root.find_by_str("2 = table1.a").unwrap(),
            root.find_by_str("3 = table1.a").unwrap(),
        );
        assert_eq!(
            tag_manager
                .new_tag([(root.clone(), true)])
                .unwrap()
                .combine(&tag_manager.new_tag([(a2.clone(), true)]).unwrap())
                .unwrap()
                .assign,
            BTreeMap::from([(root.clone(), true)])
        );
        assert_eq!(
            tag_manager
                .new_tag([(root.clone(), true)])
                .unwrap()
                .combine(&tag_manager.new_tag([(a2.clone(), false)]).unwrap())
                .unwrap()
                .assign,
            BTreeMap::from([(root.clone(), true)])
        );
    }

    #[test]
    fn test_propagate_duplicate_preds() {
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
        let tag_manager = TagManager::new();
        let pred = root.find_by_str("1 = table1.a").unwrap();
        let parent = utils::convert_to_one(pred.parents());
        assert_eq!(
            parent,
            root.find_by_str("('a' = table1.b or 1 = table1.a)")
                .unwrap()
        );

        let tag = tag_manager.new_tag([(pred.clone(), true)]).unwrap();
        assert_eq!(tag.assign, BTreeMap::from([(parent, true)]));

        let tag = tag_manager
            .new_tag([(root.find_by_str("0 < table1.c").unwrap(), false)])
            .unwrap();
        assert_eq!(
            tag.assign,
            BTreeMap::from([(
                root.find_by_str("(('a' = table1.b or 1 = table1.a) and 0 < table1.c)")
                    .unwrap(),
                false
            )])
        );
        let ancestor_lines = pred.ancestor_lines();
        assert_eq!(
            ancestor_lines.iter().all(|line| line
                .iter()
                .any(|ancestor| tag.assign.contains_key(ancestor))),
            false
        );
        assert_eq!(
            ancestor_lines.iter().any(|line| line
                .iter()
                .any(|ancestor| tag.assign.contains_key(ancestor))),
            true
        );

        // Should propagate and keep original assignment in some cases.
        {
            let query = utils::convert_to_one(
                parser
                .parse("select * from table1 where (a = 1 or b = 'b') and (b = 'b' or (c > 0 and a = 1))")
                .unwrap(),
            );
            let root = query.filter.unwrap();
            let tag_manager = TagManager::new();
            let pred = root.find_by_str("1 = table1.a").unwrap();
            let tag = tag_manager.new_tag([(pred.clone(), false)]).unwrap();
            assert_eq!(
                tag.assign,
                BTreeMap::from([
                    (pred, false),
                    (
                        root.find_by_str("(0 < table1.c and 1 = table1.a)").unwrap(),
                        false
                    )
                ])
            );
        }
    }
}
