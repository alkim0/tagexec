# Introduction
This repository contains the code for Basilisk, a column-oriented database system capable of performing tagged execution as outlined in "Optimizing Disjunctive Queries with Tagged Execution" (SIGMOD 2024).
A link to the technical report can be found [here](https://arxiv.org/abs/2404.09109).
To cite, please use:
```
@article{kimOptimizingDisjunctiveQueries2024
  author       = {Albert Kim and
                  Samuel Madden},
  title        = {Optimizing Disjunctive Queries with Tagged Execution},
  journal      = {Proc. {ACM} Manag. Data},
  volume       = {2},
  number       = {3},
  pages        = {158:1--158:25},
  year         = {2024},
  url          = {https://doi.org/10.1145/3654961},
  doi          = {10.1145/3654961},
}
```

# Running Experiments

## `drop_caches`
Before, running any experimental code, we must first create a supplemental script for dropping Linux's file system cache.
Go into the `drop_caches` directory, run make, update owner and set the sticky bit:
```
cd drop_caches
make
sudo chown root drop_caches
sudo chmod u+s drop_caches
```

## Other Scripts
Other scripts are in the `scripts` directory and managed by poetry. Make sure to run install dependencies with:
```
cd scripts
poetry install
```

## Generating Data
### Generating IMDB Data
To generate the data for the join order benchmark, download the csvs from http://homepages.cwi.nl/~boncz/job/imdb.tgz
Then go to the script directory and use the `imdb2db` script to generate data in a compatible format:
```
cd scripts
poetry run imdb2db <imdb-csvs-directory>
```
The output directory may be changed with the `-o` flag.

### Generating Synthetic Data
To generate the data for the synthetic experiments, use the `gen-synth-data` script:
```
cd scripts
poetry run gen-synth-data
```
The output directory may be changed with the `-o` flag.

## Running Experiments
Run experiments with the `job-exp` and `synth-exp` binaries:
```
cargo build --release
./target/release/job-exp
./target/release/synth-exp
```
The database directory may be changed with the `--db-path` flag.

## Running Tests
To run tests, we must first create the test data. Go into the `scripts` directory and run:
```
cd scripts
poetry run gen-test-data
```

Then in the main directory, run:
```
cargo test
```
