from pathlib import Path

from .db import DB

TEST_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "test-data"
TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)


join_test_db = (
    "join-test",
    {
        "table1": [
            {"name": "a", "type": "int", "vals": [1, 2, 3, 5, 3, -3, 6, 8]},
            {
                "name": "b",
                "type": "string",
                "vals": ["a", "b", "c", "a", "a", "b", "b", "a"],
            },
            {
                "name": "c",
                "type": "float",
                "vals": [1.5, 23.5, -1.5, 0, 0.234, 42.2, 1.90, 2.87],
            },
            {"name": "id", "type": "int", "vals": range(8), "pk": True},
        ],
        "table2": [
            {"name": "fid", "type": "int", "vals": [0, 1, 0, 1, 1, 5, 6, 7]},
            {
                "name": "d",
                "type": "string",
                "vals": ["c", "b", "a", "c", "b", "a", "b", "b"],
            },
            {
                "name": "e",
                "type": "float",
                "vals": [
                    3.15,
                    3.53,
                    -5.3,
                    0.4,
                    0.56,
                    20.432,
                    1.3,
                    0,
                ],
            },
            {"name": "f", "type": "int", "vals": [3, 2, -4, 10, 30, -20, 0, 7]},
        ],
        "table3": [
            {"name": "fid", "type": "int", "vals": [2, 3, 10, 3, 3]},
            {"name": "d", "type": "string", "vals": ["c", "b", "a", "c", "b"]},
            {"name": "e", "type": "float", "vals": [3.15, 3.53, -5.3, 0.4, 0.56]},
            {"name": "f", "type": "int", "vals": [3, 2, -4, 10, 30]},
        ],
        "table4": [
            {"name": "fid", "type": "int", "vals": [0, 1, 0, 1, 1, 5, 6, 7, 10, 0, 2]},
            {
                "name": "d",
                "type": "string",
                "vals": ["c", "b", "a", "c", "b", "a", "b", "b", "d", "a", "b"],
            },
            {
                "name": "e",
                "type": "float",
                "vals": [3.15, 3.53, -5.3, 0.4, 0.56, 20.432, 1.3, 0, 2.4, -1.4, 1],
            },
            {
                "name": "f",
                "type": "int",
                "vals": [3, 2, -4, 10, 30, -20, 0, 7, 0, 3, -4],
            },
        ],
        "table5": [
            {"name": "id", "type": "int", "vals": [0, 1, 0, 1, 1, 5, 6, 7, 10, 0, 2]},
            {
                "name": "d",
                "type": "string",
                "vals": ["c", "b", "a", "c", "b", "a", "b", "b", "d", "a", "b"],
            },
            {
                "name": "e",
                "type": "float",
                "vals": [3.15, 3.53, -5.3, 0.4, 0.56, 20.432, 1.3, 0, 2.4, -1.4, 1],
            },
            {
                "name": "f",
                "type": "int",
                "vals": [3, 2, -4, 10, 30, -20, 0, 7, 0, 3, -4],
            },
        ],
        "table6": [
            {"name": "fid", "type": "int", "vals": [3, 10, 2, 0, 5, 1]},
            {
                "name": "e",
                "type": "float",
                "vals": [8.95, 2.31, 6.7, -0.1, 0.0, -5.412],
            },
        ],
    },
)

filter_test_db = (
    "filter-test",
    {
        "table1": [
            {"name": "a", "type": "int", "vals": [5, 4, 3, 6, 0, -3]},
            {
                "name": "b",
                "type": "string",
                "vals": [
                    "bulbasaur",
                    "ivysaur",
                    "venosaur",
                    "charmander",
                    "charmeleon",
                    "charizard",
                ],
            },
            {"name": "c", "type": "int", "vals": [7, 4, 3, 5, 6, 0]},
            {
                "name": "d",
                "type": "float",
                "vals": [
                    0.29881039,
                    0.4852774,
                    0.70230872,
                    0.88748193,
                    0.65813114,
                    0.31778557,
                ],
            },
            {"name": "e", "type": "int", "vals": [9, 2, 1, 3, 1, 8]},
            {
                "name": "f",
                "type": "string",
                "vals": [
                    "squirtle",
                    "wartortle",
                    "blastoise",
                    "caterpie",
                    "metapod",
                    "butterfree",
                ],
            },
        ],
        "table2": [
            {"name": "a", "type": "int", "vals": [9, 6, 7, 0, 9, 9, 2, 5, 4, 1]},
            {"name": "b", "type": "int", "vals": [7, 2, 4, 5, 3, 1, 4, 8, 5, 4]},
            {"name": "c", "type": "int", "vals": [8, 3, 1, 5, 0, 8, 9, 0, 3, 1]},
            {"name": "d", "type": "int", "vals": [4, 4, 5, 3, 9, 8, 2, 4, 3, 0]},
        ],
    },
)


def gen_test_data(db_name: str, data: dict[str, list[dict]]) -> None:
    db = DB.from_dict(db_name, data)
    db.write(TEST_DATA_DIR)


def main() -> None:
    gen_test_data(join_test_db[0], join_test_db[1])
    gen_test_data(filter_test_db[0], filter_test_db[1])


if __name__ == "__main__":
    main()
