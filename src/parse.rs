use crate::data_dir::DataDir;
use crate::expr::Expr;
use crate::file_table::{FileCol, FileTableRef};
use crate::pred::{Pred, PredAtom};
use crate::query::Query;
use crate::stats::StatsReader;
use crate::utils::{self, IteratorAllEqExt};
use sqlparser::ast;
use sqlparser::dialect::PostgreSqlDialect;
use sqlparser::parser::{Parser as RawParser, ParserError as RawParseError};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

pub struct ParseContext {
    pub table_refs: Vec<Rc<FileTableRef>>,
    pub aliases: HashMap<String, Rc<FileTableRef>>,
}

#[derive(Debug)]
pub enum ParseError {
    Raw(RawParseError),
    NonSelect(String),
    NoTable(String),
    Expr(String),
}

pub type Result<T> = std::result::Result<T, ParseError>;

pub struct Parser<'a> {
    data_dir: &'a DataDir,
    stats_reader: RefCell<Option<&'a StatsReader>>,
}

impl<'a> Parser<'a> {
    pub fn new(data_dir: &'a DataDir) -> Self {
        Self {
            data_dir,
            stats_reader: RefCell::new(None),
        }
    }

    pub fn update_stats_reader(&self, stats_reader: &'a StatsReader) {
        *self.stats_reader.borrow_mut() = Some(stats_reader);
    }

    pub fn parse(&self, sql: &str) -> Result<Vec<Query>> {
        let stmts =
            RawParser::parse_sql(&PostgreSqlDialect {}, sql).map_err(|err| ParseError::Raw(err))?;

        stmts
            .into_iter()
            .map(|stmt| {
                let stmt_str = stmt.to_string();
                let query = if let ast::Statement::Query(query) = stmt {
                    Ok(query)
                } else {
                    Err(ParseError::NonSelect(stmt_str.clone()))
                }?;
                let query = if let ast::SetExpr::Select(query) = *query.body {
                    Ok(query)
                } else {
                    Err(ParseError::NonSelect(stmt_str.clone()))
                }?;

                let context = self.parse_from(query.from)?;
                let filter = query
                    .selection
                    .map(|filter| self.parse_pred(filter, &context))
                    .transpose()?;
                //.map(|pred| pred.normalize().sort_by_canon_str());
                let projection = self.parse_projection(query.projection, &context)?;
                let group_by = self.parse_group_by(query.group_by, &context)?;

                Ok(Query {
                    projection,
                    group_by,
                    filter,
                    from: context.table_refs,
                })
            })
            .collect()
    }

    fn parse_pred(&self, expr: ast::Expr, context: &ParseContext) -> Result<Rc<Pred>> {
        fn _parse_pred(
            expr: ast::Expr,
            context: &ParseContext,
            stats_reader: &Option<&StatsReader>,
        ) -> Result<Rc<Pred>> {
            match expr {
                ast::Expr::BinaryOp {
                    left,
                    op: op @ (ast::BinaryOperator::And | ast::BinaryOperator::Or),
                    right,
                } => {
                    let left = _parse_pred(*left, context, stats_reader)?;
                    let right = _parse_pred(*right, context, stats_reader)?;
                    if let ast::BinaryOperator::And = op {
                        Ok(Pred::new_and(vec![left, right]))
                    } else {
                        Ok(Pred::new_or(vec![left, right]))
                    }
                }
                ast::Expr::Nested(expr) => _parse_pred(*expr, context, stats_reader),
                _ => {
                    let pred_atom = PredAtom::new(Expr::new(&expr, context)?);
                    if let Some(stats_reader) = stats_reader {
                        let selectivity = stats_reader.get_selectivity(&pred_atom);
                        pred_atom.update_selectivity(selectivity);
                    }
                    Ok(Pred::new_atom(pred_atom))
                }
            }
        }

        let stats_reader = self.stats_reader.borrow();
        let pred = _parse_pred(expr, context, &*stats_reader);
        pred.map(|pred| pred.normalize())
    }

    fn parse_group_by(
        &self,
        group_by: Vec<ast::Expr>,
        context: &ParseContext,
    ) -> Result<Vec<Expr>> {
        group_by
            .into_iter()
            .map(|expr| Expr::new(&expr, context))
            .collect()
    }

    fn parse_projection(
        &self,
        projection: Vec<ast::SelectItem>,
        context: &ParseContext,
    ) -> Result<Vec<Expr>> {
        Ok(projection
            .into_iter()
            .map(|item| Expr::new_select_item(&item, context))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect())
    }

    fn parse_from(&self, from: Vec<ast::TableWithJoins>) -> Result<ParseContext> {
        fn parse_table(parser: &Parser, table: ast::TableFactor) -> Result<ParseContext> {
            match table {
                ast::TableFactor::Table { name, alias, .. } => {
                    assert_eq!(name.0.len(), 1);
                    let name = name.0.into_iter().next().unwrap().value;
                    let table = parser
                        .data_dir
                        .get_table(&name)
                        .ok_or(ParseError::NoTable(name.clone()))?;

                    let alias = if let Some(ast::TableAlias { name: alias, .. }) = alias {
                        Some(alias.value)
                    } else {
                        None
                    };

                    let table_ref = Rc::new(FileTableRef::new(table.clone(), alias.clone()));

                    let mut aliases = HashMap::new();
                    if let Some(alias) = alias {
                        aliases.insert(alias, table_ref.clone());
                    }

                    Ok(ParseContext {
                        table_refs: vec![table_ref],
                        aliases,
                    })
                }
                _ => {
                    panic!("Unsupported table type: {:?}", table);
                }
            }
        }

        fn parse_table_with_joins(
            parser: &Parser,
            table: ast::TableWithJoins,
        ) -> Result<ParseContext> {
            assert!(table.joins.is_empty());
            parse_table(parser, table.relation)
        }

        let mut table_refs = vec![];
        let mut aliases = HashMap::new();
        for table in from {
            let mut context = parse_table_with_joins(self, table)?;
            table_refs.append(&mut context.table_refs);
            aliases.extend(context.aliases);
        }

        Ok(ParseContext {
            table_refs,
            aliases,
        })
    }

    /// Combines multiple queries into one by taking the OR between the queries' predicates. It is
    /// assumed that the queries join using the same join constraints and require the same tables.
    /// In the case that the projection expression differs across different queries, the first
    /// query's projection expression is used.
    ///
    /// Note that this function assumes the join constraints to appear as part of the predicate
    /// (instead of as separate join constraint). Since these constraints are assumed to be the
    /// same across all queries, they are pulled out and the predicate ultimately ends up with a
    /// root predicate. However, all non-constraint predicates are arranged in DNF (not exactly
    /// DNF, since the depth might be greater than 2, but it will have a disjunctive root anyway).
    ///
    /// We expect the final form of the predicate to look something like:
    /// join-constraint-1 and join-constraint-2 and ... and (query-1-clause or query-2-clause or ...)
    /// in which query-1-clause are the non-constraint predicates (ANDed together).
    pub fn or_combine_queries_disj_root<T: AsRef<str>>(&self, queries: &[T]) -> Query {
        let queries: Vec<_> = queries
            .iter()
            .map(|query| utils::convert_to_one(self.parse(query.as_ref()).unwrap()))
            .collect();

        assert!(queries
            .iter()
            .all(|query| query.filter.as_ref().unwrap().is_and()));
        assert!(queries.iter().all_eq(|query| query
            .group_by
            .iter()
            .map(|x| x.to_string())
            .collect::<HashSet<_>>()));
        assert!(queries.iter().all_eq(|query| query
            .from
            .iter()
            .map(|x| x.to_string())
            .collect::<HashSet<_>>()));

        let pred = self
            .or_combine_preds_disj_root(queries.iter().map(|query| query.filter.clone().unwrap()));

        let first_query = queries.iter().next().unwrap();

        let query = Query {
            projection: first_query.projection.clone(),
            group_by: first_query.group_by.clone(),
            filter: Some(pred),
            from: first_query.from.clone(),
        };

        //println!("{}", query);
        utils::convert_to_one(self.parse(&query.to_string()).unwrap())
    }

    /// Combines multiple queries into one by taking the OR between the queries' predicates. It is
    /// assumed that the queries join using the same join constraints and require the same tables.
    /// In the case that the projection expression differs across different queries, the first
    /// query's projection expression is used.
    ///
    /// Note that this function assumes the join constraints to appear as part of the predicate
    /// (instead of as separate join constraint). Since these constraints are assumed to be the
    /// same across all queries, they are pulled out and the predicate ultimately ends up with a
    /// root predicate.
    ///
    /// The difference with `or_combine_preds_disj_root` is that predicates that appear commonly are
    /// also pulled out of the DNF expression.
    ///
    /// We expect the final form of the predicate to look something like:
    /// join-constraint-1 and join-constraint-2 and ... and common-pred-1 and common-pred-2 and ...
    /// and (query-1-clause or query-2-clause or ...)
    /// in which query-1-clause are the non-constraint, non-common predicates (ANDed together).
    pub fn or_combine_queries_conj_root<T: AsRef<str>>(&self, queries: &[T]) -> Query {
        let queries: Vec<_> = queries
            .iter()
            .map(|query| utils::convert_to_one(self.parse(query.as_ref()).unwrap()))
            .collect();

        assert!(queries
            .iter()
            .all(|query| query.filter.as_ref().unwrap().is_and()));
        assert!(queries.iter().all_eq(|query| query
            .group_by
            .iter()
            .map(|x| x.to_string())
            .collect::<HashSet<_>>()));
        assert!(queries.iter().all_eq(|query| query
            .from
            .iter()
            .map(|x| x.to_string())
            .collect::<HashSet<_>>()));

        let pred = self
            .or_combine_preds_conj_root(queries.iter().map(|query| query.filter.clone().unwrap()));

        let first_query = queries.iter().next().unwrap();

        let query = Query {
            projection: first_query.projection.clone(),
            group_by: first_query.group_by.clone(),
            filter: Some(pred),
            from: first_query.from.clone(),
        };

        utils::convert_to_one(self.parse(&query.to_string()).unwrap())
    }

    // Combines an iterator of Pred roots into one using disjunctions. The final outcome should be
    // in DNF. Assumes each pred is in conjunctive at the root. Note that the returned predicate is
    // still conjunctive at the root since we assume join predicates appear as part of the
    // predicate expression.
    fn or_combine_preds_disj_root(&self, preds: impl Iterator<Item = Rc<Pred>>) -> Rc<Pred> {
        let mut join_constraints = HashSet::new();
        let mut clauses = vec![];
        for pred in preds {
            assert!(pred.is_and());

            let mut clause = vec![];
            for child in pred.try_iter_children().unwrap() {
                // This makes sure we don't have join constraints as a descendant of an OR node or
                // something.
                if !child.is_atom() {
                    assert!(!child.has_multi_table_atom());
                }

                if child.is_atom() && child.has_multi_table_atom() {
                    join_constraints.insert(StrHashPred(child.clone()));
                } else {
                    clause.push(child.clone());
                }
            }

            clauses.push(Pred::new_and(clause));
        }

        assert!(!join_constraints.is_empty());
        let mut top_clauses = Vec::from_iter(
            join_constraints
                .into_iter()
                .map(|str_hash_pred| str_hash_pred.0),
        );
        top_clauses.push(Pred::new_or(clauses));
        Pred::new_and(top_clauses)

        //Pred::new_or(preds.collect())
    }

    fn or_combine_preds_conj_root(&self, preds: impl Iterator<Item = Rc<Pred>>) -> Rc<Pred> {
        let mut join_constraints = HashSet::new();
        let mut pred_counts = HashMap::new();
        let mut clauses = vec![];
        let mut num_preds = 0;
        for pred in preds {
            assert!(pred.is_and());

            let mut clause = HashSet::new();
            for child in pred.try_iter_children().unwrap() {
                // This makes sure we don't have join constraints as a descendant of an OR node or
                // something.
                if !child.is_atom() {
                    assert!(!child.has_multi_table_atom());
                }

                if child.is_atom() && child.has_multi_table_atom() {
                    join_constraints.insert(StrHashPred(child.clone()));
                } else {
                    clause.insert(StrHashPred(child.clone()));
                    *pred_counts.entry(StrHashPred(child.clone())).or_insert(0) += 1;
                }
            }

            clauses.push(clause);
            num_preds += 1;
        }

        let common_preds: Vec<_> = pred_counts
            .into_iter()
            .filter_map(|(pred, count)| if count == num_preds { Some(pred) } else { None })
            .collect();

        for clause in &mut clauses {
            for common_pred in &common_preds {
                clause.remove(common_pred);
            }
        }

        assert!(!join_constraints.is_empty());
        let mut top_clauses = Vec::from_iter(
            join_constraints
                .into_iter()
                .map(|str_hash_pred| str_hash_pred.0),
        );
        top_clauses.extend(
            common_preds
                .into_iter()
                .map(|str_hash_pred| str_hash_pred.0),
        );
        top_clauses.push(Pred::new_or(
            clauses
                .into_iter()
                .filter_map(|clause| {
                    if clause.is_empty() {
                        None
                    } else {
                        Some(Pred::new_and(
                            clause
                                .into_iter()
                                .map(|str_hash_pred| str_hash_pred.0)
                                .collect(),
                        ))
                    }
                })
                .collect(),
        ));

        Pred::new_and(top_clauses)
    }
}

impl ParseContext {
    pub fn find_table_ref(&self, table_name_or_alias: &str) -> Option<&Rc<FileTableRef>> {
        self.aliases.get(table_name_or_alias).or_else(|| {
            self.table_refs
                .iter()
                .find(|table_ref| table_ref.table.name() == table_name_or_alias)
        })
    }

    pub fn find_col(
        &self,
        col_name: &str,
        table_name_or_alias: Option<&str>,
    ) -> Option<(&Rc<FileCol>, &Rc<FileTableRef>)> {
        match table_name_or_alias {
            Some(table_name_or_alias) => {
                self.find_table_ref(table_name_or_alias)
                    .and_then(|table_ref| {
                        table_ref
                            .table
                            .find_col(col_name)
                            .map(|col| (col, table_ref))
                    })
            }
            None => self.table_refs.iter().find_map(|table_ref| {
                table_ref
                    .table
                    .find_col(col_name)
                    .map(|col| (col, table_ref))
            }),
        }
    }
}

struct StrHashPred(Rc<Pred>);

impl PartialEq for StrHashPred {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_string() == other.0.to_string()
    }
}

impl Eq for StrHashPred {}

impl std::hash::Hash for StrHashPred {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_string().hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_sqlparser() {
        use sqlparser::dialect::GenericDialect;
        use sqlparser::parser::Parser;

        let dialect = GenericDialect {}; // or AnsiDialect

        let sql = "SELECT a, b, 123, myfunc(b) \
           FROM table_1 join table_2 on table_1.id = table_2.id \
           WHERE a > b AND b < 100 \
           ORDER BY a DESC, b";

        let ast = Parser::parse_sql(&dialect, sql).unwrap();

        println!("AST: {:?}", ast);
    }

    #[test]
    fn test_or_combine_preds_disj_root() {
        let db_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("data")
            .join("test-data")
            .join("join-test");
        let data_dir = DataDir::new(db_path).unwrap();
        let parser = Parser::new(&data_dir);

        let query1 = "select table1.id from table1, table2, table4 where table1.id = table2.fid and table1.a = table4.fid and a = 1 and b = 'b'";
        let query2 = "select table1.id from table1, table2, table4 where table1.id = table2.fid and table1.a = table4.fid and a = 1 and table2.d = 'b'";
        let query3 = "select table1.id from table1, table2, table4 where table1.id = table2.fid and table1.a = table4.fid and a = 1 and (table2.d = 'c' or c > 0)";

        let query = parser.or_combine_queries_disj_root(&[query1, query2, query3]);
        assert_eq!(query.to_string(),
            "SELECT table1.id FROM table1, table2, table4 WHERE ((('b' = table1.b and 1 = table1.a) or ('b' = table2.d and 1 = table1.a) or (('c' = table2.d or 0 < table1.c) and 1 = table1.a)) and table1.a = table4.fid and table1.id = table2.fid)"
        );
    }

    #[test]
    fn test_or_combine_preds_conj_root() {
        let db_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("data")
            .join("test-data")
            .join("join-test");
        let data_dir = DataDir::new(db_path).unwrap();
        let parser = Parser::new(&data_dir);

        let query1 = "select table1.id from table1, table2, table4 where table1.id = table2.fid and table1.a = table4.fid and a = 1 and b = 'b'";
        let query2 = "select table1.id from table1, table2, table4 where table1.id = table2.fid and table1.a = table4.fid and a = 1 and table2.d = 'b'";
        let query3 = "select table1.id from table1, table2, table4 where table1.id = table2.fid and table1.a = table4.fid and a = 1 and (table2.d = 'c' or c > 0)";

        let query = parser.or_combine_queries_conj_root(&[query1, query2, query3]);
        assert_eq!(query.to_string(),
            "SELECT table1.id FROM table1, table2, table4 WHERE (('b' = table1.b or 'b' = table2.d or 'c' = table2.d or 0 < table1.c) and 1 = table1.a and table1.a = table4.fid and table1.id = table2.fid)"
        );
    }
}
