use itertools::Itertools;

pub fn find_shared(queries: &[Query]) -> (HashSet<&Pred>, HashSet<Rc<FileTable>>) {
    // We expect all tables to be shared among the given queries and for a certain subset
    // of the AND-root children to be shared among the queries.

    assert!(queries.len() > 0);
    let first = queries.first().unwrap();
    let mut shared_tables = HashSet::from_iter(first.from.clone());

    let mut shared_preds = get_top_level_preds(first);

    for query in queries {
        let query_preds = get_top_level_preds(query);
        shared_tables = shared_tables
            .into_iter()
            .filter(|table| query.from.contains(table))
            .collect();
        shared_preds = shared_preds
            .into_iter()
            .filter(|pred| query_preds.contains(pred))
            .collect();
    }

    (shared_preds, shared_tables)
}

pub fn build_join_graph_and_table_preds<'a>(
    preds: impl IntoIterator<Item = &'a &'a Pred>,
) -> (JoinGraph, TablePreds<'a>) {
    let mut join_graph = HashMap::new();
    let mut table_preds = HashMap::new();
    for pred in preds.into_iter().unique() {
        if pred.has_multi_table_atom() {
            if let Pred::Atom(atom) = pred {
                let constraint = atom.expr().as_equality_constraint().unwrap();
                join_graph
                    .entry(constraint.left.table())
                    .or_insert(vec![])
                    .push(constraint.clone());
                join_graph
                    .entry(constraint.right.table())
                    .or_insert(vec![])
                    .push(constraint);
            } else {
                panic!("Unexpected multi-table pred {}", pred);
            }
        } else {
            let table = pred.file_tables().into_iter().next().unwrap();
            table_preds
                .entry(table)
                .or_insert(vec![])
                .push(pred.clone());
        }
    }
    (join_graph, table_preds)
}
