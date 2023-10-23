use crate::bitmap::Bitmap;
use crate::db::{DBCol, DBColInner, DBSingle, DBType, DBVals};
use crate::file_table::{FileCol, FileColSet, FileTableRef, FileTableRefSet};
use crate::funcs::FUNC_MAP;
use crate::idx::Idx;
use crate::parse::{self, ParseContext, ParseError};
use crate::utils::{self, IteratorFilterExt, OneOf};
use auto_enums::auto_enum;
use either::Either;
use itertools::{izip, Itertools};
use regex::Regex;
use sqlparser::ast;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use traversal::DftPre;

// Forget about CTEs and nested subqueries for now.

pub struct EvalContext<'a> {
    pub idx: &'a Idx,
    pub bmap: Option<Rc<Bitmap>>,
}

macro_rules! do_op {
    (@col $iter:expr, $len:expr, $out_type:ident) => {{
        let mut out = Vec::with_capacity($len);
        out.extend($iter);
        DBVals::Col(DBCol::$out_type(DBColInner::Values(out)))
    }};

    (@single $expr:expr, $out_type:ident) => {
        DBVals::Single(DBSingle::$out_type($expr))
    };
}

//macro_rules! do_op {
//    (@zip $left_iter:expr, $right_iter:expr, $len:expr, $out_type:ident, $map:expr) => {{
//        let mut out = Vec::with_capacity($len);
//        out.extend($left_iter.zip($right_iter).map($map));
//        DBVals::Col(DBCol::$out_type(out))
//    }};
//
//    (@one_col $iter:expr, $len:expr, $out_type:ident, $map:expr) => {{
//        let mut out = Vec::with_capacity($len);
//        out.extend($iter.map($map));
//        DBVals::Col(DBCol::$out_type(out))
//    }};
//
//    (@single $out_type:ident, $expr:expr) => {
//        DBVals::Single(DBSingle::$out_type($expr))
//    };
//}

macro_rules! iter_or_repeat {
    ($vals:expr, $db_type:ident) => {
        match $vals {
            DBVals::Col(DBCol::$db_type(col)) => Either::Left(col.iter()),
            DBVals::Single(DBSingle::$db_type(single)) => Either::Right(std::iter::repeat(single)),
            _ => panic!("Unexpected type {}", stringify!($db_type)),
        }
    };

    ($vals:expr, $col_func:ident, $single_func:ident) => {
        match $vals {
            DBVals::Col(col) => Either::Left(col.$col_func()),
            DBVals::Single(single) => Either::Right(std::iter::repeat(single.$single_func())),
        }
    };

    (@map $vals:expr, $db_type:ident, $map:expr) => {
        match $vals {
            DBVals::Col(DBCol::$db_type(col)) => Either::Left(col.iter().map($map)),
            DBVals::Single(DBSingle::$db_type(single)) => {
                Either::Right(std::iter::repeat(($map)(single)))
            }
            _ => panic!("Unexpected type {}", stringify!($db_type)),
        }
    }; //($vals:expr, $db_type:ident, $map:expr) => {
       //    match $vals {
       //        DBVals::Col(DBCol::$db_type(col)) => Either::Left(col.iter().map($map)),
       //        DBVals::Single(DBSingle::$db_type(single)) => Either::Right(std::iter::repeat(($map)(single))),
       //    }
       //};
}

// TODO: Implemented iter_as_converted_long and iter_as_converted_double, also change normal iter
// and clean up db in general

macro_rules! do_number_op {
    ($left:expr, $right:expr, $op:tt) => {{
        let is_integral = $left.data_type().is_integral() && $right.data_type().is_integral();
        match ($left, $right) {
            (DBVals::Single(left), DBVals::Single(right)) if is_integral => {
                do_op!(@single left.to_long() $op right.to_long(), Long)
            }
            (DBVals::Single(left), DBVals::Single(right)) => {
                do_op!(@single left.to_double() $op right.to_double(), Double)
            }
            (left, right) if is_integral => {
                let left_iter = iter_or_repeat!(left, iter_to_long, to_long);
                let right_iter = iter_or_repeat!(right, iter_to_long, to_long);
                do_op!(@col left_iter.zip(right_iter).map(|(x, y)| x $op y), left.len(), Long)
            }
            (left, right) => {
                let left_iter = iter_or_repeat!(left, iter_to_double, to_double);
                let right_iter = iter_or_repeat!(right, iter_to_double, to_double);
                do_op!(@col left_iter.zip(right_iter).map(|(x, y)| x $op y), left.len(), Double)
            }
            //(DBVals::Col(left), DBVals::Col(right)) if is_integral => {
            //    do_op!(@zip left.iter_to_long(), right.iter_to_long(), left.len(), Long, |(x, y)| x $op y)
            //}
            //(DBVals::Col(left), DBVals::Col(right)) => {
            //    do_op!(@zip left.iter_to_double(), right.iter_to_double(), left.len(), Double, |(x, y)| x $op y)
            //}

            //(DBVals::Col(left), DBVals::Single(right)) if is_integral => {
            //    let val = right.to_long();
            //    do_op!(@one_col left.iter_to_long(), left.len(), Long, |x| x $op val)
            //}
            //(DBVals::Col(left), DBVals::Single(right)) => {
            //    let val = right.to_double();
            //    do_op!(@one_col left.iter_to_double(), left.len(), Double, |x| x $op val)
            //}

            //(DBVals::Single(left), DBVals::Col(right)) if is_integral => {
            //    let val = left.to_long();
            //    do_op!(@one_col right.iter_to_long(), right.len(), Long, |x| val $op x)
            //}
            //(DBVals::Single(left), DBVals::Col(right)) => {
            //    let val = left.to_double();
            //    do_op!(@one_col right.iter_to_double(), right.len(), Double, |x| val $op x)
            //}

            //(DBVals::Single(left), DBVals::Single(right)) if is_integral => {
            //    do_op!(@single Long, left.to_long() $op right.to_long())
            //}
            //(DBVals::Single(left), DBVals::Single(right)) => {
            //    do_op!(@single Double, left.to_double() $op right.to_double())
            //}
        }
    }}
}

macro_rules! do_datetime_op {
    ($left:expr, $right:expr, $op:tt) => {
        match ($left.data_type(), $right.data_type()) {
            (DBType::DateTime, DBType::Duration) => {
                match ($left, $right) {
                    (DBVals::Single(DBSingle::DateTime(left)), DBVals::Single(DBSingle::Duration(right))) => {
                        do_op!(@single *left $op *right, DateTime)
                    }
                    (left, right) => {
                        let left_iter = iter_or_repeat!(left, DateTime);
                        let right_iter = iter_or_repeat!(right, Duration);
                        do_op!(@col left_iter.zip(right_iter).map(|(x, y)| *x $op *y), left.len(), DateTime)
                    }
                }
            }
            (DBType::DateTime, DBType::Str) => {
                match ($left, $right) {
                    (DBVals::Single(DBSingle::DateTime(left)), DBVals::Single(DBSingle::Str(right))) => {
                        do_op!(@single *left $op utils::parse_duration(right), DateTime)
                    }
                    (left, right) => {
                        let left_iter = iter_or_repeat!(left, DateTime);
                        let right_iter = iter_or_repeat!(@map right, Str, |s| utils::parse_duration(s));
                        do_op!(@col left_iter.zip(right_iter).map(|(x, y)| *x $op y), left.len(), DateTime)
                    }
                }
            }
            _ => do_number_op!($left, $right, $op),

            //(DBVals::Col(DBCol::DateTime(left)), DBVals::Col(DBCol::Duration(right))) => {
            //    do_op!(@zip left.iter(), right.iter(), left.len(), DateTime, |(x, y)| *x $op *y)
            //}
            //(DBVals::Col(DBCol::DateTime(left)), DBVals::Col(DBCol::Str(right))) => {
            //    do_op!(@zip left.iter(), right.iter().map(|s| utils::parse_duration(s)), left.len(), DateTime, |(x, y)| *x $op y)
            //}

            //(DBVals::Col(DBCol::DateTime(left)), DBVals::Single(DBSingle::Duration(right))) => {
            //    do_op!(@one_col left.iter(), left.len(), DateTime, |x| *x $op *right)
            //}
            //(DBVals::Col(DBCol::DateTime(left)), DBVals::Single(DBSingle::Str(right))) => {
            //    let val = utils::parse_duration(right.as_str());
            //    do_op!(@one_col left.iter(), left.len(), DateTime, |x| *x $op val)
            //}

            //(DBVals::Single(DBSingle::DateTime(left)), DBVals::Col(DBCol::Duration(right))) => {
            //    do_op!(@one_col right.iter(), right.len(), DateTime, |x| *left $op *x)
            //}
            //(DBVals::Single(DBSingle::DateTime(left)), DBVals::Col(DBCol::Str(right))) => {
            //    do_op!(@one_col right.iter().map(|s| utils::parse_duration(s)), right.len(), DateTime, |x| *left $op x)
            //}

            //(DBVals::Single(DBSingle::DateTime(left)), DBVals::Single(DBSingle::Duration(right))) => {
            //    do_op!(@single DateTime, *left $op *right)
            //}
            //(DBVals::Single(DBSingle::DateTime(left)), DBVals::Single(DBSingle::Str(right))) => {
            //    do_op!(@single DateTime, *left $op utils::parse_duration(right.as_str()))
            //}

            //(left, right) => do_number_op!(left, right, $op),
        }
    }
}

macro_rules! do_plus_op {
    ($left:expr, $right:expr) => {
        match ($left.data_type(), $right.data_type()) {
            (DBType::Str, DBType::Str) => {
                match ($left, $right) {
                    (DBVals::Single(DBSingle::Str(left)), DBVals::Single(DBSingle::Str(right))) => {
                        do_op!(@single left.to_owned() + right, Str)
                    }
                    (left, right) => {
                        let left_iter = iter_or_repeat!(left, Str);
                        let right_iter = iter_or_repeat!(right, Str);
                        do_op!(@col left_iter.zip(right_iter).map(|(x, y)| x.to_owned() + y), left.len(), Str)
                    }
                }
            }
            _ => do_datetime_op!($left, $right, +),
        }
        //match ($left, $right) {
        //    (DBVals::Col(DBCol::Str(left)), DBVals::Col(DBCol::Str(right))) => {
        //        do_op!(@zip left.iter(), right.iter(), left.len(), Str, |(x, y)| x.to_owned() + y)
        //    }
        //    (DBVals::Col(DBCol::Str(left)), DBVals::Single(DBSingle::Str(right))) => {
        //        do_op!(@one_col left.iter(), left.len(), Str, |x| x.to_owned() + right)
        //    }
        //    (DBVals::Single(DBSingle::Str(left)), DBVals::Col(DBCol::Str(right))) => {
        //        do_op!(@one_col right.iter(), right.len(), Str, |x| left.to_owned() + x)
        //    }
        //    (DBVals::Single(DBSingle::Str(left)), DBVals::Single(DBSingle::Str(right))) => {
        //        do_op!(@single Str, left.to_owned() + right)
        //    }
        //    (left, right) => do_datetime_op!(left, right, +),
        //}
    }
}

macro_rules! do_bool_op {
    (@inner $left:expr, $right:expr, $op:tt, $db_type:ident) => {
        match ($left, $right) {
            (DBVals::Single(DBSingle::$db_type(left)), DBVals::Single(DBSingle::$db_type(right))) => {
                do_op!(@single left $op right, Bool)
            }
            (left, right) => {
                let left_iter = iter_or_repeat!(left, $db_type);
                let right_iter = iter_or_repeat!(right, $db_type);
                do_op!(@col left_iter.zip(right_iter).map(|(x, y)| x $op y), left.len(), Bool)
            }
        }
    };

    (@inner $left:expr, $right:expr, $op:tt, $col_func:ident, $single_func:ident) => {
        match ($left, $right) {
            (DBVals::Single(left), DBVals::Single(right)) => {
                do_op!(@single left.$single_func() $op right.$single_func(), Bool)
            }
            (left, right) => {
                let left_iter = iter_or_repeat!(left, $col_func, $single_func);
                let right_iter = iter_or_repeat!(right, $col_func, $single_func);
                do_op!(@col left_iter.zip(right_iter).map(|(x, y)| x $op y), left.len(), Bool)
            }
        }
    };

    (@inner $left:expr, $right:expr, $op:tt, ($left_db_type:ident, $right_db_type:ident), $right_map:expr) => {
        match ($left, $right) {
            (DBVals::Single(DBSingle::$left_db_type(left)), DBVals::Single(DBSingle::$right_db_type(right))) => {
                do_op!(@single left $op &($right_map)(right), Bool)
            }
            (left, right) => {
                let left_iter = iter_or_repeat!(left, $left_db_type);
                let right_iter = iter_or_repeat!(@map right, $right_db_type, $right_map);
                do_op!(@col left_iter.zip(right_iter).map(|(x, y)| x $op &y), left.len(), Bool)
            }
        }
    };

    ($left:expr, $right:expr, $op:tt) => {
        match ($left.data_type(), $right.data_type()) {
            (DBType::Str, DBType::Str) => do_bool_op!(@inner $left, $right, $op, Str),
            (DBType::Bool, DBType::Bool) => do_bool_op!(@inner $left, $right, $op, Bool),
            (DBType::DateTime, DBType::DateTime) => do_bool_op!(@inner $left, $right, $op, DateTime),
            (DBType::DateTime, DBType::Str) => do_bool_op!(@inner $left, $right, $op, (DateTime, Str), |s| utils::parse_datetime(s)),
            (DBType::Duration, DBType::Duration) => do_bool_op!(@inner $left, $right, $op, Duration),
            (DBType::Duration, DBType::Str) => do_bool_op!(@inner $left, $right, $op, (Duration, Str), |s| utils::parse_duration(s)),
            (left_type, right_type) if left_type.is_integral() && right_type.is_integral() => {
                do_bool_op!(@inner $left, $right, $op, iter_to_long, to_long)
            }
            _ => do_bool_op!(@inner $left, $right, $op, iter_to_double, to_double),
        }
    };
}

#[derive(Clone, PartialEq, Eq)]
pub enum Expr {
    ColRef {
        table_ref: Rc<FileTableRef>,
        col: Rc<FileCol>,
    },
    IsNull {
        expr: Box<Expr>,
        negated: bool,
    },
    BinaryOp {
        left: Box<Expr>,
        right: Box<Expr>,
        op: BinaryOperator,
    },
    UnaryOp {
        expr: Box<Expr>,
        op: UnaryOperator,
    },
    Value(DBSingle),
    Function {
        name: String,
        args: Vec<Expr>,
    },
    Between {
        expr: Box<Expr>,
        low: Box<Expr>,
        high: Box<Expr>,
    },
    Like {
        expr: Box<Expr>,
        pattern: String,
        negated: bool,
    },
    InList {
        expr: Box<Expr>,
        list: Vec<Expr>,
        negated: bool,
    },
    //Case {
    //    cases: Vec<(Expr, Expr)>,
    //    else_result: Option<Box<Expr>>,
    //},
    //Nested(Box<Expr>),
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub enum BinaryOperator {
    Plus,
    Minus,
    Multiply,
    Divide,
    Modulo,
    //Gt,
    Lt,
    //GtEq,
    LtEq,
    Eq,
    NotEq,
}

// XXX: Assumes that EqualityConstraint's are canonicalized in the beginning and remain unchanged
// throughout execution
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct EqualityConstraint {
    pub left_col: Rc<FileCol>,
    pub left_table_ref: Rc<FileTableRef>,
    pub right_col: Rc<FileCol>,
    pub right_table_ref: Rc<FileTableRef>,
}

macro_rules! convert_binary_op {
    (@return_result $expr:expr, ($($op:ident),*)) => {
        match $expr {
            $(
                ast::BinaryOperator::$op => Ok(BinaryOperator::$op)
            ),*,
            _ => Err(ParseError::Expr(format!("Unxpected op {}", $expr).to_string())),
        }
    };

    ($expr:expr, ($($op:ident),*)) => {
        match $expr {
            $(
                ast::BinaryOperator::$op => BinaryOperator::$op
            ),*,
            _ => {
                panic!("Unxpected op {}", $expr);
            }
        }
    };
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub enum UnaryOperator {
    Plus,
    Minus,
    Not,
}

impl Expr {
    pub fn new(ast_expr: &ast::Expr, context: &ParseContext) -> parse::Result<Self> {
        // XXX: When constructing a new Expr, a certain structure will always be preferred. This
        // makes comparing Expr's easier later. For example, all x > y expressions will be turned
        // into y < x expressions.
        match ast_expr {
            ast::Expr::Identifier(ast::Ident { value, .. }) => context
                .find_col(value, None)
                .map(|(col, table_ref)| Self::make_col_ref(col, table_ref))
                .ok_or(ParseError::Expr(format!("Could find col: {}", value))),
            ast::Expr::CompoundIdentifier(idents) => {
                assert!(idents.len() == 2);
                context
                    .find_col(&idents[1].value, Some(&idents[0].value))
                    .map(|(col, table_ref)| Self::make_col_ref(col, table_ref))
                    .ok_or(ParseError::Expr(format!(
                        "Could find col: {}.{}",
                        &idents[0], &idents[1]
                    )))
            }
            ast::Expr::Value(val) => match val {
                ast::Value::Number(val, _) => val
                    .parse::<i64>()
                    .map(|val| Self::Value(DBSingle::Long(val)))
                    .or_else(|_| {
                        val.parse::<f64>()
                            .map(|val| Self::Value(DBSingle::Double(val)))
                    })
                    .or(Err(ParseError::Expr(format!(
                        "Could not parse value: {}",
                        val
                    )))),
                ast::Value::SingleQuotedString(s) => Ok(Self::Value(DBSingle::Str(s.clone()))),
                ast::Value::Boolean(b) => Ok(Self::Value(DBSingle::Bool(*b))),
                _ => {
                    panic!("Unknown type for val {:?}", val);
                }
            },
            ast::Expr::BinaryOp { left, right, op } => {
                let left = Box::new(Expr::new(left, context)?);
                let right = Box::new(Expr::new(right, context)?);
                match op {
                    // Commutative operations
                    ast::BinaryOperator::Plus
                    | ast::BinaryOperator::Multiply
                    | ast::BinaryOperator::Eq
                    | ast::BinaryOperator::NotEq => {
                        let op = convert_binary_op!(op, (Plus, Multiply, Eq, NotEq));
                        if left.to_string() <= right.to_string() {
                            Ok(Self::BinaryOp { left, right, op })
                        } else {
                            Ok(Self::BinaryOp {
                                left: right,
                                right: left,
                                op,
                            })
                        }
                    }
                    // Operations which can be flipped
                    ast::BinaryOperator::Gt => Ok(Self::BinaryOp {
                        left: right,
                        right: left,
                        op: BinaryOperator::Lt,
                    }),
                    ast::BinaryOperator::GtEq => Ok(Self::BinaryOp {
                        left: right,
                        right: left,
                        op: BinaryOperator::LtEq,
                    }),
                    // Remaining operations
                    _ => {
                        let op = convert_binary_op!(@return_result op, (Minus, Divide, Modulo, Lt, LtEq))?;
                        Ok(Self::BinaryOp { left, right, op })
                    }
                }
            }
            ast::Expr::UnaryOp { expr, op } => {
                let expr = Expr::new(expr, context)?;
                let op = match op {
                    ast::UnaryOperator::Plus => {
                        return Ok(expr);
                    }
                    ast::UnaryOperator::Minus => UnaryOperator::Minus,
                    ast::UnaryOperator::Not => UnaryOperator::Not,
                    _ => {
                        panic!("Unsupported {}", op);
                    }
                };
                Ok(Self::UnaryOp {
                    expr: Box::new(expr),
                    op,
                })
            }
            ast::Expr::Nested(expr) => Expr::new(expr, context),
            ast::Expr::Function(ast::Function { name, args, .. }) => {
                assert_eq!(name.0.len(), 1);
                let name = name.0[0].value.to_lowercase();
                if let "extract_" | "timezone" | "array_agg" | "json_path_lookup" = name.as_str() {
                    return Err(ParseError::Expr(format!(
                        "Unimplemented function: {}",
                        name
                    )));
                }

                let args = args
                    .iter()
                    .map(|arg| {
                        if let ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(arg)) = arg {
                            Expr::new(arg, context)
                        } else {
                            Err(ParseError::Expr(format!(
                                "Do not yet support named or wildcard function arguments: {}",
                                arg
                            )))
                        }
                    })
                    .collect::<parse::Result<_>>()?;

                Ok(Expr::Function { name, args })
            }
            ast::Expr::Between {
                expr,
                negated,
                low,
                high,
            } => {
                if *negated {
                    Err(ParseError::Expr(
                        "Don't support not between expressions".to_string(),
                    ))
                } else {
                    Ok(Expr::Between {
                        expr: Box::new(Expr::new(expr, context)?),
                        low: Box::new(Expr::new(low, context)?),
                        high: Box::new(Expr::new(high, context)?),
                    })
                }
            }
            //ast::Expr::IsNull(subexpr) => Ok(Expr::IsNull(Box::new(Expr::new(subexpr, context)?))),
            ast::Expr::IsNull(expr) => Ok(Expr::IsNull {
                expr: Box::new(Expr::new(expr, context)?),
                negated: false,
            }),
            ast::Expr::IsNotNull(expr) => Ok(Expr::IsNull {
                expr: Box::new(Expr::new(expr, context)?),
                negated: true,
            }),
            //ast::Expr::Case {
            //    conditions,
            //    results,
            //    else_result,
            //    operand,
            //} => {
            //    unimplemented!("Do not support cases yet");
            //    // TODO: Do not support operand yet
            //    assert!(operand.is_none());
            //    let cases = conditions
            //        .iter()
            //        .zip(results.iter())
            //        .map(|(cond, result)| {
            //            let cond = Expr::new(cond, context)?;
            //            let result = Expr::new(result, context)?;
            //            Ok((cond, result))
            //        })
            //        .collect::<parse::Result<_>>()?;
            //    let else_result = else_result
            //        .as_ref()
            //        .map(|expr| Expr::new(expr, context))
            //        .transpose()?
            //        .map(|expr| Box::new(expr));
            //    Ok(Expr::Case { cases, else_result })
            //}
            ast::Expr::Like {
                negated,
                expr,
                pattern,
                escape_char,
            } => {
                // Do not support escape char yet
                assert!(escape_char.is_none());
                let expr = Box::new(Expr::new(expr, context)?);
                let pattern = Expr::new(pattern, context)?;
                let pattern = if let Expr::Value(DBSingle::Str(s)) = pattern {
                    s
                } else {
                    return Err(ParseError::Expr(format!("Unexpected pattern {}", pattern)));
                };
                Ok(Expr::Like {
                    expr,
                    pattern,
                    negated: *negated,
                })
            }
            ast::Expr::InList {
                expr,
                list,
                negated,
            } => {
                let expr = Box::new(Expr::new(expr, context)?);
                let list = list
                    .iter()
                    .map(|expr| Expr::new(expr, context))
                    .collect::<parse::Result<_>>()?;
                Ok(Expr::InList {
                    expr,
                    list,
                    negated: *negated,
                })
            }
            //// TODO Implement in list/between in future
            //ast::Expr::InList { .. } | ast::Expr::Between { .. } => {
            //    Ok(Expr::Value(DBCol::Bool(vec![true])))
            //}
            _ => {
                panic!("Expression not supported {:?}", ast_expr);
            }
        }
    }

    pub fn new_select_item(
        ast_item: &ast::SelectItem,
        context: &ParseContext,
    ) -> parse::Result<Vec<Self>> {
        match ast_item {
            ast::SelectItem::UnnamedExpr(expr) | ast::SelectItem::ExprWithAlias { expr, .. } => {
                Ok(vec![Self::new(&expr, context)?])
            }
            // TODO: Implement exclude/except options
            ast::SelectItem::Wildcard(_) => Ok(context
                .table_refs
                .iter()
                .map(|table_ref| std::iter::repeat(table_ref).zip(table_ref.table.all_cols()))
                .flatten()
                .map(|(table_ref, col)| Self::make_col_ref(col, table_ref))
                .collect()),
            ast::SelectItem::QualifiedWildcard(table_name, _) => {
                assert_eq!(table_name.0.len(), 1);
                let table_name = &table_name.0[0].value;
                let table_ref = context
                    .find_table_ref(table_name)
                    .ok_or(ParseError::NoTable(format!(
                        "Could not find {}",
                        table_name
                    )))?;
                Ok(table_ref
                    .table
                    .all_cols()
                    .into_iter()
                    .map(|col| Self::make_col_ref(&col, table_ref))
                    .collect())
            }
        }
    }

    fn make_col_ref(col: &Rc<FileCol>, table_ref: &Rc<FileTableRef>) -> Self {
        Self::ColRef {
            table_ref: table_ref.clone(),
            col: col.clone(),
        }
    }

    /// Iterates over the children of an expression.
    #[auto_enum(Iterator)]
    fn iter_children<'a>(&'a self) -> impl Iterator<Item = &'a Self> {
        match self {
            Self::ColRef { .. } | Self::Value(_) => std::iter::empty(),
            Self::BinaryOp { left, right, .. } => vec![left.as_ref(), right.as_ref()].into_iter(),
            Self::UnaryOp { expr, .. } | Self::IsNull { expr, .. } | Self::Like { expr, .. } => {
                vec![expr.as_ref()].into_iter()
            }
            Self::Function { args, .. } => args.iter(),
            Self::Between { expr, low, high } => {
                vec![expr.as_ref(), low.as_ref(), high.as_ref()].into_iter()
            }
            Self::InList { expr, list, .. } => std::iter::once(expr.as_ref()).chain(list.iter()),
        }
    }

    /// Iterates over the expression tree in dft order.
    fn iter_tree<'a>(&'a self) -> impl Iterator<Item = &'a Self> {
        DftPre::new(self, |expr| expr.iter_children()).map(|(_, expr)| expr)
    }

    /// Iterates over all `ColRef` variants referred to by this expression.
    fn iter_col_refs<'a>(&'a self) -> impl Iterator<Item = &'a Self> {
        self.iter_tree()
            .filter(|&expr| matches!(expr, Self::ColRef { .. }))
    }

    /// Get all `FileCol`s referred to by this expression
    pub fn file_cols(&self) -> FileColSet {
        self.iter_col_refs()
            .map(|expr| match expr {
                Self::ColRef { col, .. } => col.clone(),
                _ => panic!("Wtf"),
            })
            .collect()
        //match self {
        //    Self::ColRef { col, .. } => FileColSet::from([col.clone()]),
        //    Self::BinaryOp { left, right, .. } => {
        //        itertools::concat([left.file_cols(), right.file_cols()])
        //    }
        //    Self::UnaryOp { expr, .. } => expr.file_cols(),
        //    Self::Function { args, .. } => {
        //        itertools::concat(args.iter().map(|arg| arg.file_cols()))
        //    }
        //    Self::Value(_) => FileColSet::new(),
        //    Self::Between { expr, low, high } => {
        //        itertools::concat([expr.file_cols(), low.file_cols(), high.file_cols()])
        //    }
        //    Self::Like { expr, .. } => expr.file_cols(),
        //    Self::InList { expr, list, .. } => {
        //        let mut cols = expr.file_cols();
        //        cols.extend(itertools::concat(list.iter().map(|expr| expr.file_cols())));
        //        cols
        //    }
        //    Self::IsNull { expr, .. } => expr.file_cols(),
        //}
    }

    /// Get all `FileTableRef`s referred to by this expression
    pub fn file_table_refs(&self) -> FileTableRefSet {
        self.iter_col_refs()
            .map(|expr| match expr {
                Self::ColRef { table_ref, .. } => table_ref.clone(),
                _ => panic!("Wtf"),
            })
            .collect()
        //match self {
        //    Self::ColRef { table_ref, .. } => FileTableRefSet::from([table_ref.clone()]),
        //    Self::BinaryOp { left, right, .. } => {
        //        itertools::concat([left.file_table_refs(), right.file_table_refs()])
        //    }
        //    Self::UnaryOp { expr, .. } => expr.file_table_refs(),
        //    Self::Value(_) => FileTableRefSet::new(),
        //    Self::Function { args, .. } => {
        //        itertools::concat(args.iter().map(|arg| arg.file_table_refs()))
        //    }
        //    Self::Between { expr, low, high } => itertools::concat([
        //        expr.file_table_refs(),
        //        low.file_table_refs(),
        //        high.file_table_refs(),
        //    ]),
        //    Self::Like { expr, .. } => expr.file_table_refs(),
        //    Self::InList { expr, list, .. } => {
        //        let mut table_refs = expr.file_table_refs();
        //        table_refs.extend(itertools::concat(
        //            list.iter().map(|expr| expr.file_table_refs()),
        //        ));
        //        table_refs
        //    }
        //    Self::IsNull { expr, .. } => expr.file_table_refs(),
        //}
    }

    /// Note that `Rc<DBVals>` is only possibly returned when the cache is used.
    pub fn eval(&self, context: &EvalContext<'_>) -> DBVals {
        match self {
            Self::ColRef { .. } => self.eval_col_ref(context),
            Self::BinaryOp { .. } => self.eval_binary_op(context),
            Self::UnaryOp { .. } => self.eval_unary_op(context),
            //Self::Nested(subexpr) => subexpr.eval(context),
            Self::Value(val) => DBVals::Single(val.clone()),
            Self::Function { .. } => self.eval_func(context),
            //Self::Case { .. } => self.eval_case(context),
            Self::Between { .. } => self.eval_between(context),
            Self::Like { .. } => self.eval_like(context),
            Self::InList { .. } => self.eval_in_list(context),
            Self::IsNull { .. } => self.eval_is_null(context),
            //Self::Combiner { .. } => self.eval_combiner(context),
        }
    }

    pub fn as_equality_constraint(&self) -> Option<EqualityConstraint> {
        if let Expr::BinaryOp {
            left,
            right,
            op: BinaryOperator::Eq,
        } = self
        {
            if let (
                Expr::ColRef {
                    col: left_col,
                    table_ref: left_table_ref,
                },
                Expr::ColRef {
                    col: right_col,
                    table_ref: right_table_ref,
                },
            ) = (left.as_ref(), right.as_ref())
            {
                Some(EqualityConstraint {
                    left_col: left_col.clone(),
                    left_table_ref: left_table_ref.clone(),
                    right_col: right_col.clone(),
                    right_table_ref: right_table_ref.clone(),
                })
            } else {
                None
            }
        } else {
            None
        }
    }

    fn eval_col_ref(&self, context: &EvalContext<'_>) -> DBVals {
        if let Self::ColRef { table_ref, col } = self {
            DBVals::Col(if let Some(bmap) = &context.bmap {
                //col.read(context.idx.col_iter(table_ref).filter_by_index(bmap))
                col.read(context.idx.col_iter(table_ref).filter_by_index2(bmap))
            } else {
                col.read(context.idx.col_iter(table_ref))
            })
        } else {
            panic!("wtf");
        }
    }

    fn eval_binary_op(&self, context: &EvalContext<'_>) -> DBVals {
        if let Self::BinaryOp { left, right, op } = self {
            let left_vals = left.eval(context);
            let right_vals = right.eval(context);

            let left_vals = AsRef::as_ref(&left_vals);
            let right_vals = AsRef::as_ref(&right_vals);

            match op {
                BinaryOperator::Plus => do_plus_op!(left_vals, right_vals),
                BinaryOperator::Minus => do_datetime_op!(left_vals, right_vals, -),
                BinaryOperator::Multiply => do_number_op!(left_vals, right_vals, *),
                BinaryOperator::Divide => do_number_op!(left_vals, right_vals, /),
                BinaryOperator::Modulo => do_number_op!(left_vals, right_vals, %),
                BinaryOperator::Lt => do_bool_op!(left_vals, right_vals, <),
                BinaryOperator::LtEq => do_bool_op!(left_vals, right_vals, <=),
                BinaryOperator::Eq => do_bool_op!(left_vals, right_vals, ==),
                BinaryOperator::NotEq => do_bool_op!(left_vals, right_vals, !=),
            }
        } else {
            panic!("wtf");
        }
    }

    fn eval_unary_op(&self, context: &EvalContext<'_>) -> DBVals {
        if let Self::UnaryOp { expr, op } = self {
            let vals = expr.eval(context);

            match op {
                UnaryOperator::Plus => vals,
                UnaryOperator::Minus if vals.data_type().is_integral() => {
                    do_op!(@col vals.iter_to_long().map(|x| -x), vals.len(), Long)
                }
                UnaryOperator::Minus => {
                    do_op!(@col vals.iter_to_double().map(|x| -x), vals.len(), Double)
                }
                UnaryOperator::Not => {
                    do_op!(@col vals.iter_as_bool().map(|x| !x), vals.len(), Bool)
                }
            }
        } else {
            panic!("wtf");
        }
    }

    fn eval_between(&self, context: &EvalContext<'_>) -> DBVals {
        if let Self::Between { expr, low, high } = self {
            let expr_vals = expr.eval(context);
            let low_vals = low.eval(context);
            let high_vals = high.eval(context);

            let expr_vals = AsRef::as_ref(&expr_vals);
            let low_vals = AsRef::as_ref(&low_vals);
            let high_vals = AsRef::as_ref(&high_vals);

            match (
                expr_vals.data_type(),
                low_vals.data_type(),
                high_vals.data_type(),
            ) {
                (DBType::Str, DBType::Str, DBType::Str) => {
                    let low_iter = iter_or_repeat!(&low_vals, Str);
                    let high_iter = iter_or_repeat!(&high_vals, Str);
                    do_op!(@col izip!(expr_vals.iter_as_str(), low_iter, high_iter).map(|(expr, low, high)| low <= expr && expr <= high), expr_vals.len(), Bool)
                }
                (expr_type, low_type, high_type)
                    if expr_type.is_integral()
                        && low_type.is_integral()
                        && high_type.is_integral() =>
                {
                    let low_iter = iter_or_repeat!(&low_vals, iter_to_long, to_long);
                    let high_iter = iter_or_repeat!(&high_vals, iter_to_long, to_long);
                    do_op!(@col izip!(expr_vals.iter_to_long(), low_iter, high_iter).map(|(expr, low, high)| low <= expr && expr <= high), expr_vals.len(), Bool)
                }
                _ => {
                    let low_iter = iter_or_repeat!(&low_vals, iter_to_double, to_double);
                    let high_iter = iter_or_repeat!(&high_vals, iter_to_double, to_double);
                    do_op!(@col izip!(expr_vals.iter_to_double(), low_iter, high_iter).map(|(expr, low, high)| low <= expr && expr <= high), expr_vals.len(), Bool)
                }
            }
        } else {
            panic!("wtf");
        }
    }

    fn eval_func(&self, context: &EvalContext<'_>) -> DBVals {
        if let Self::Function { name, args } = self {
            let args: Vec<_> = args.iter().map(|arg| arg.eval(context)).collect();
            let args_as_ref = args.iter().map(|arg| AsRef::as_ref(arg)).collect();
            FUNC_MAP[&name](args_as_ref)
        } else {
            panic!("wtf");
        }
    }

    fn eval_like(&self, context: &EvalContext<'_>) -> DBVals {
        if let Self::Like {
            expr,
            pattern,
            negated,
        } = self
        {
            let vals = expr.eval(context);
            let vals = AsRef::as_ref(&vals);
            let re = Regex::new(&utils::sql_pattern_to_regex(pattern)).unwrap();
            do_op!(@col vals.iter_as_str().map(|s| *negated ^ re.is_match(s)), vals.len(), Bool)
        } else {
            panic!("wtf");
        }
    }

    fn eval_in_list(&self, context: &EvalContext<'_>) -> DBVals {
        if let Self::InList {
            expr,
            list,
            negated,
        } = self
        {
            let expr_vals = expr.eval(context);
            let expr_vals = AsRef::as_ref(&expr_vals);
            let list_vals: Vec<_> = list.iter().map(|e| e.eval(context)).collect();
            match expr_vals.data_type() {
                DBType::Str => {
                    let list_vals: Vec<_> = list_vals
                        .iter()
                        .map(|v| AsRef::as_ref(v).iter_as_str().next().unwrap())
                        .collect();
                    if *negated {
                        do_op!(@col expr_vals.iter_as_str().map(|v| list_vals.iter().all(|list_v| v != list_v.as_str())), expr_vals.len(), Bool)
                    } else {
                        do_op!(@col expr_vals.iter_as_str().map(|v| list_vals.iter().any(|list_v| v == list_v.as_str())), expr_vals.len(), Bool)
                    }
                }

                expr_type
                    if expr_type.is_integral()
                        && list_vals
                            .iter()
                            .all(|v| AsRef::as_ref(v).data_type().is_integral()) =>
                {
                    let list_vals: Vec<_> = list_vals
                        .iter()
                        .map(|v| AsRef::as_ref(v).iter_to_long().next().unwrap())
                        .collect();
                    if *negated {
                        do_op!(@col expr_vals.iter_to_long().map(|v| list_vals.iter().all(|list_v| v != *list_v)), expr_vals.len(), Bool)
                    } else {
                        do_op!(@col expr_vals.iter_to_long().map(|v| list_vals.iter().any(|list_v| v == *list_v)), expr_vals.len(), Bool)
                    }
                }

                _ => {
                    let list_vals: Vec<_> = list_vals
                        .iter()
                        .map(|v| AsRef::as_ref(v).iter_to_double().next().unwrap())
                        .collect();
                    if *negated {
                        do_op!(@col expr_vals.iter_to_double().map(|v| list_vals.iter().all(|list_v| v != *list_v)), expr_vals.len(), Bool)
                    } else {
                        do_op!(@col expr_vals.iter_to_double().map(|v| list_vals.iter().any(|list_v| v == *list_v)), expr_vals.len(), Bool)
                    }
                }
            }
        } else {
            panic!("wtf");
        }
    }

    // FIXME: The way we implement nulls is a hack. We don't implement separate null lists, so we
    // just pick a specific value within the domain of each type and say that is the null value.
    // For example, the empty string is null for str types and i32::MIN is null for
    // ints
    fn eval_is_null(&self, context: &EvalContext<'_>) -> DBVals {
        if let Self::IsNull { expr, negated } = self {
            let vals = expr.eval(context);
            match vals.data_type() {
                DBType::Str => {
                    do_op!(@col vals.iter_as_str().map(|s| *negated ^ s.is_empty()), vals.len(), Bool)
                }
                DBType::Int => {
                    do_op!(@col vals.iter_as_int().map(|i| *negated ^ (*i == i32::MIN)), vals.len(), Bool)
                }
                _ => panic!("Null not implemented for {:?} type", vals.data_type()),
            }
            //DBVals::Col(DBCol::Bool(DBColInner::Values(vec![*negated; vals.len()])))
        } else {
            panic!("wtf");
        }
    }
}

impl EqualityConstraint {
    pub fn flip(&self) -> Self {
        Self {
            left_col: self.right_col.clone(),
            left_table_ref: self.right_table_ref.clone(),
            right_col: self.left_col.clone(),
            right_table_ref: self.left_table_ref.clone(),
        }
    }

    #[inline]
    pub fn other_table_ref(&self, table_ref: &Rc<FileTableRef>) -> &Rc<FileTableRef> {
        if table_ref == &self.left_table_ref {
            &self.right_table_ref
        } else if table_ref == &self.right_table_ref {
            &self.left_table_ref
        } else {
            panic!("Table ref doesn't match either in constraint");
        }
    }

    pub fn canonicalize(&self) -> Self {
        if format!("{}.{}", self.left_table_ref, self.left_col.name())
            <= format!("{}.{}", self.right_table_ref, self.right_col.name())
        {
            self.clone()
        } else {
            self.flip()
        }
    }
}

impl Hash for Expr {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        match self {
            Self::ColRef { col, table_ref } => {
                ("ColRef", col.full_name(), table_ref).hash(state);
            }
            Self::BinaryOp { left, right, op } => {
                ("BinaryOp", left, right, op).hash(state);
            }
            Self::UnaryOp { expr, op } => {
                ("UnaryOp", expr, op).hash(state);
            }
            Self::Value(val) => {
                ("Value", val).hash(state);
            }
            Self::Function { name, args } => {
                ("Function", name, args).hash(state);
            }
            Self::Between { expr, low, high } => {
                ("Between", expr, low, high).hash(state);
            }
            Self::Like {
                expr,
                pattern,
                negated,
            } => {
                ("Like", expr, pattern, negated).hash(state);
            }
            Self::InList {
                expr,
                list,
                negated,
            } => {
                ("InList", expr, list, negated).hash(state);
            }
            Self::IsNull { expr, negated } => {
                ("IsNull", expr, negated).hash(state);
            } //Self::Combiner { children, is_and } => {
              //    ("Combiner", children, is_and).hash(state);
              //}
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::ColRef { col, table_ref } => {
                write!(f, "{}.{}", table_ref, col.name())
            }
            Expr::BinaryOp { left, right, op } => {
                let op = match op {
                    BinaryOperator::Plus => "+",
                    BinaryOperator::Minus => "-",
                    BinaryOperator::Multiply => "*",
                    BinaryOperator::Divide => "/",
                    BinaryOperator::Modulo => "%",
                    //BinaryOperator::Gt => ">",
                    BinaryOperator::Lt => "<",
                    //BinaryOperator::GtEq => ">=",
                    BinaryOperator::LtEq => "<=",
                    BinaryOperator::Eq => "=",
                    BinaryOperator::NotEq => "!=",
                };
                write!(f, "{} {} {}", left, op, right)
            }
            Expr::UnaryOp { expr, op } => {
                let op = match op {
                    UnaryOperator::Plus => "+",
                    UnaryOperator::Minus => "-",
                    UnaryOperator::Not => "NOT ",
                };
                write!(f, "{}{}", op, expr)
            }
            //Expr::Nested(subexpr) => write!(f, "({})", subexpr),
            Expr::Value(val) => match val {
                DBSingle::Str(_) => write!(f, "'{}'", val),
                _ => write!(f, "{}", val),
            },
            Expr::Function { name, args } => {
                let args: Vec<String> = args.iter().map(|arg| arg.to_string()).collect();
                write!(f, "{}({})", name, args.join(", "))
            }
            Expr::Between { expr, low, high } => {
                write!(f, "{} BETWEEN {} AND {}", expr, low, high)
            }
            Self::Like {
                expr,
                pattern,
                negated,
            } => {
                write!(
                    f,
                    "{} {}LIKE '{}'",
                    expr,
                    if *negated { "NOT " } else { "" },
                    pattern
                )
            }
            Self::InList {
                expr,
                list,
                negated,
            } => {
                write!(
                    f,
                    "{} {}IN ({})",
                    expr,
                    if *negated { "NOT " } else { "" },
                    list.iter().map(|e| e.to_string()).join(", ")
                )
            }
            Self::IsNull { expr, negated } => {
                write!(f, "{} IS {}NULL", expr, if *negated { "NOT " } else { "" })
            } //Self::Combiner {
              //    children, is_and, ..
              //} => {
              //    write!(
              //        f,
              //        "({})",
              //        children.iter().join(if *is_and { " and " } else { " or " })
              //    )
              //}
        }
    }
}

impl fmt::Debug for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl fmt::Display for EqualityConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}.{} = {}.{}",
            self.left_table_ref,
            self.left_col.name(),
            self.right_table_ref,
            self.right_col.name()
        )
    }
}

impl fmt::Debug for EqualityConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}
