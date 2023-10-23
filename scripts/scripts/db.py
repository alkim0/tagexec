from typing import Callable
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
import struct

MAX_STR_LEN = 124  # 128 - 4 bytes to store length


class DBType(Enum):
    INT = "int"
    LONG = "long"
    FLOAT = "float"
    DOUBLE = "double"
    STR = "string"
    BOOL = "boolean"

    @classmethod
    def from_str(cls, s: str) -> "DBType":
        return {
            "int": cls.INT,
            "long": cls.LONG,
            "float": cls.FLOAT,
            "double": cls.DOUBLE,
            "string": cls.STR,
            "boolean": cls.BOOL,
        }[s]


def cut_str(s: str) -> bytes:
    b = s.encode()
    if len(b) <= MAX_STR_LEN:
        return b

    num = 1
    while len(b) > MAX_STR_LEN:
        b = s[:-num].encode()
        num += 1

    return b


class Col:
    def __init__(
        self,
        name: str,
        data_type: DBType,
        vals: list | range | None = None,
        rng: Callable | None = None,
    ) -> None:
        self.name = name
        self.data_type = data_type

        assert vals is not None or rng is not None
        if vals is not None:
            self.vals = vals
        else:
            self.rng = rng

    def write(self, table_path: Path, verbose=False) -> None:
        col_path = table_path / self.name

        def write_val(f, val) -> None:
            if self.data_type == DBType.STR:
                val = cut_str(val)
                f.write(struct.pack(f"i{MAX_STR_LEN}s", len(val), val))
                # val = val.encode()
                # f.write(
                #    struct.pack(
                #        f"i{MAX_STR_LEN}s",
                #        min(MAX_STR_LEN, len(val)),
                #        val[:MAX_STR_LEN],
                #    )
                # )
            else:
                fmt = {
                    DBType.INT: "i",
                    DBType.LONG: "q",
                    DBType.FLOAT: "f",
                    DBType.DOUBLE: "d",
                    DBType.BOOL: "b",
                }[self.data_type]
                f.write(struct.pack(fmt, val))

        with open(col_path, "wb") as f:
            if hasattr(self, "vals"):
                for val in self.vals:
                    write_val(f, val)
            else:
                assert self.rng is not None
                for val in self.rng():
                    write_val(f, val)

    @classmethod
    def from_dict(cls, data: dict) -> "Col":
        return cls(
            data["name"],
            DBType.from_str(data["type"]),
            data.get("vals"),
            data.get("rng"),
        )


@dataclass
class Table:
    name: str
    cols: list[Col]

    def write(self, db_path: Path, verbose=False) -> None:
        table_path = db_path / self.name
        table_path.mkdir(exist_ok=True)
        for col in self.cols:
            if verbose:
                print(f"Writing column {self.name}.{col.name} ... ", end="", flush=True)

            col.write(table_path, verbose)

            if verbose:
                print("Done!")

        with open(table_path / "__schema__", "w") as f:
            print(",".join([col.name for col in self.cols]), file=f)
            print(",".join([col.data_type.value for col in self.cols]), file=f)


@dataclass
class DB:
    name: str
    tables: list[Table]

    def write(self, outdir: Path, verbose=False) -> None:
        db_path = outdir / self.name
        db_path.mkdir(exist_ok=True)

        if verbose:
            print(f"Writing db {self.name}")

        for table in self.tables:
            table.write(db_path, verbose)

        if verbose:
            print(f"Finished writing db {self.name}")

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "DB":
        return cls(
            name,
            [
                Table(table_name, [Col.from_dict(col) for col in cols])
                for table_name, cols in data.items()
            ],
        )
