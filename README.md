# README

Code and simulation data for *Beyond power laws: broad functional advantages of log-normal brain networks*.

## Installation

Use Python 3.12. Brian2 may require a C++ compiler. Run from the repository root:

```bash
python -m pip install -r requirements.txt
```

## Usage

Analyze the supplied simulation data and plot the results:

```bash
python reproduce.py fit all
python reproduce.py figures
```

To calculate bootstrap intervals, run `python reproduce.py fit all --bootstrap 1000 --seed 42`, then rerun `figures`.

Generate new simulations with `python reproduce.py simulate FUNCTION --profile full`, replacing `FUNCTION` with `stability`, `kuramoto`, `navigation`, `stdp` or `hopfield`. Omitting `--profile full` runs a small execution check.

Outputs are saved under `result/`. Use `python reproduce.py COMMAND --help` for options.

## Data

`data/functional/` contains the five simulation observation tables. Variables and preprocessing are defined in `code/src/connectome_dynamics/reproduction.py`.

Biological connectomes must be obtained separately:

| Dataset                  | Source                                     |
| ------------------------ | ------------------------------------------ |
| Adult C. elegans         | https://doi.org/10.1038/s41586-019-1352-7  |
| Developmental C. elegans | https://doi.org/10.1038/s41586-021-03778-8 |
| Platynereis larva        | https://doi.org/10.7554/eLife.97964.2      |
| Drosophila larva         | https://doi.org/10.1126/science.add9330    |
| Drosophila adult         | https://doi.org/10.1038/s41586-024-07968-y |
| Ciona tadpole            | https://doi.org/10.7554/eLife.16962        |

Prepare trusted `.pkl` files containing `networkx.DiGraph` objects: neurons are nodes, edges point from presynaptic to postsynaptic neurons, and `weight` stores the synapse count for each ordered neuron pair. Save graphs in `result/processed_connectomes/`, using the filenames listed in `network_statistics.py`.

Input-table requirements and conversion rules are in `graph_construction.py`; the adult Drosophila graph requires manual preparation. After preparing the graphs, run:

```bash
python reproduce.py summary
python reproduce.py degree-fits --degree-mode total
python reproduce.py thresholds
```

## Licenses

Code: MIT (`LICENSE`). Simulation data: CC BY 4.0 (`data/LICENSE`). Third-party data retain their original terms. Please cite the associated paper and biological data sources used.
