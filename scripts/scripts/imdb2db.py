import csv
from pathlib import Path

import click

import sqlparse

import clevercsv

from . import db

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# We use this instead of null lists
MIN_INT = -(1 << 31)


def get_schema(schema_path: Path) -> dict[str, list[tuple[str, db.DBType]]]:
    with open(schema_path) as f:
        stmts = sqlparse.parse(f.read())

    def get_tokens(token_list):
        token = token_list.token_first(skip_ws=True, skip_cm=True)
        tidx = 0
        tokens = [token]
        while True:
            tidx, token = token_list.token_next(tidx, skip_ws=True, skip_cm=True)
            if token is None:
                break
            tokens.append(token)
        return tokens

    schema = {}
    for stmt in stmts:
        tokens = get_tokens(stmt)
        table_name = [
            t.value for t in tokens if isinstance(t, sqlparse.sql.Identifier)
        ][0]
        fields_token = [t for t in tokens if isinstance(t, sqlparse.sql.Parenthesis)][0]
        fields = []
        for field in fields_token.value[1:-1].split(","):
            field_name, field_type = field.strip().split(maxsplit=1)
            if field_type.startswith("integer"):
                fields.append((field_name, db.DBType.INT))
            elif field_type.startswith("character varying"):
                fields.append((field_name, db.DBType.STR))
            else:
                raise Exception(f"Unknown field type {field_type}")

        schema[table_name] = fields

    return schema


def write_table(table: str, schema: dict, indir: Path, outdir: Path):
    vals = [[] for _ in range(len(schema[table]))]
    for row in clevercsv.stream_table(
        indir / f"{table}.csv",
        clevercsv.dialect.SimpleDialect(",", '"', "\\"),
    ):
        if table == "title" and row[0] == "2522636":
            # This needs to be special cased
            row = [
                "2522636",
                "\\Frag'ile\\",
                "",
                "1",
                "2010",
                "",
                "F624",
                "",
                "",
                "",
                "",
                "c0b2e279bce6d3b1717e750a2591bb6d",
            ]
        elif table == "person_info" and row[0] == "2671660":
            row = [
                "2671660",
                "2604773",
                "17",
                "Daughter of Irish actor and raconteur 'Niall Toibin' (qv);",
                "",
            ]

        for i, entry in enumerate(row):
            if schema[table][i][1] == db.DBType.STR:
                vals[i].append(entry)
            elif schema[table][i][1] == db.DBType.INT:
                if entry:
                    vals[i].append(int(entry))
                else:
                    vals[i].append(MIN_INT)
            else:
                raise Exception(f"Unexpected type {schema[table][i][1]}")
    cols = [
        db.Col(name=info[0], data_type=info[1], vals=col_vals)
        for info, col_vals in zip(schema[table], vals)
    ]

    table_len = len(cols[0].vals)
    for col in cols:
        if len(col.vals) != table_len:
            raise Exception(f"Uneven column lengths for table {table}")

    db.Table(name=table, cols=cols).write(outdir)


@click.command()
@click.option("-t", "--table")
@click.option("-o", "--outdir", type=Path, default=DATA_DIR / "imdb")
@click.argument("imdb-dir", type=Path)
def cli(table: str, outdir: Path, imdb_dir: Path) -> None:
    outdir.mkdir(exist_ok=True)
    schema = get_schema(imdb_dir / "schematext.sql")
    print("Finished parsing schema")
    if table is not None:
        print(f"Converting table {table} ... ", end="", flush=True)
        write_table(table, schema, imdb_dir, outdir)
        print("Done!", flush=True)
    else:
        for table in schema:
            print(f"Converting table {table} ... ", end="", flush=True)
            write_table(table, schema, imdb_dir, outdir)
            print("Done!", flush=True)
