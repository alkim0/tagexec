from pathlib import Path

import click

import numpy as np

from .db import DB

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

FK_ZIPF_PARAM = 1.5
LONG_MAX = (1 << 63) - 1


def gen_synth_db(outdir: Path) -> None:
    def make_table(
        name: str, size: int, num_fk: int, num_a: int, num_o: int
    ) -> tuple[str, list[dict[str, object]]]:
        def make_col(name: str, type_: str, vals=None, rng=None) -> dict[str, object]:
            assert vals is not None or rng is not None
            if vals is not None:
                return {"name": name, "type": type_, "vals": vals}
            else:
                return {"name": name, "type": type_, "rng": rng}

        def gen_o_vals():
            vals = np.random.default_rng().random(size)
            vals.sort()
            return vals

        return (
            name,
            [make_col("pk", "long", vals=range(size))]
            + [
                make_col(
                    f"fk{i}",
                    "long",
                    rng=lambda: (
                        np.random.default_rng().zipf(FK_ZIPF_PARAM, size) % LONG_MAX
                    ),
                )
                for i in range(num_fk)
            ]
            + [
                make_col(
                    f"a{i}",
                    "double",
                    rng=lambda: np.random.default_rng().random(size),
                )
                for i in range(num_a)
            ]
            + [
                make_col(
                    f"o{i}",
                    "double",
                    rng=gen_o_vals,
                )
                for i in range(num_o)
            ],
        )

    tables = []
    for db_label, db_size in (
        ("1e3", int(1e3)),
        ("2e3", int(2e3)),
        ("5e3", int(5e3)),
        ("1e4", int(1e4)),
        ("2e4", int(2e4)),
        ("5e4", int(5e4)),
        ("1e5", int(1e5)),
    ):
        tables.extend(
            [make_table(f"t_{db_label}_{i}", db_size, 3, 10, 3) for i in range(3)]
        )
    tables.extend([make_table(f"t_1e2_{i}", int(1e2), 1, 3, 0) for i in range(10)])

    db = DB.from_dict(
        "synth-mixed", {table_name: table_dict for table_name, table_dict in tables}
    )
    db.write(outdir, verbose=True)


@click.command()
@click.option("-o", "--outdir", type=Path, default=DATA_DIR / "synth")
def cli(outdir: Path) -> None:
    outdir.mkdir(exist_ok=True)
    gen_synth_db(outdir)
