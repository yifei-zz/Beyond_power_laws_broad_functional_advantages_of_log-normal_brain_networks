"""Construct processed directed connectomes from the raw source files."""
from __future__ import annotations

import argparse
import networkx as nx
import numpy as np
import pandas as pd

from .io import save_pickle
from .paths import DATA_DIR, RESULT_DIR

OUTPUT_DIR = RESULT_DIR / "processed_connectomes"
INTERMEDIATE_DIR = OUTPUT_DIR / "intermediate"

def matrix_to_digraph(matrix: pd.DataFrame, normalize_node_ids: bool = True) -> nx.DiGraph:
    matrix = matrix.fillna(0)
    if normalize_node_ids:
        matrix.index = matrix.index.astype(str)
        matrix.columns = matrix.columns.astype(str)
    if list(matrix.index) != list(matrix.columns):
        raise ValueError("Adjacency-matrix row and column identifiers must match in the same order.")
    values = matrix.to_numpy(dtype=float)
    source_idx, target_idx = np.nonzero(values > 0)
    graph = nx.DiGraph()
    labels = matrix.index.tolist()
    graph.add_nodes_from(labels)
    graph.add_weighted_edges_from(
        (labels[i], labels[j], float(values[i, j])) for i, j in zip(source_idx, target_idx)
    )
    return graph

def build_drosophila_larva() -> nx.DiGraph:
    path = DATA_DIR / "drosophila_larva/drosophila_larva_connectivity_matrix.csv"
    matrix = pd.read_csv(path, index_col=0)
    graph = matrix_to_digraph(matrix, normalize_node_ids=True)
    save_pickle(graph, OUTPUT_DIR / "drosophila_larva_connectome.pkl")
    return graph

def build_platynereis_larva() -> nx.DiGraph:
    path = DATA_DIR / "platynereis_larva/platynereis_larva_connectivity_matrix.csv"
    graph = matrix_to_digraph(pd.read_csv(path, index_col=0))
    save_pickle(graph, OUTPUT_DIR / "platynereis_larva_connectome.pkl")
    return graph

def build_ciona_tadpole() -> nx.DiGraph:
    path = DATA_DIR / "ciona_tadpole/ciona_tadpole_connectivity_fractional.csv"
    matrix = pd.read_csv(path, index_col=0).fillna(0)
    integer_matrix = (matrix / 0.06).round().astype(int)
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    integer_matrix.to_csv(INTERMEDIATE_DIR / "ciona_tadpole_integer_connectivity_matrix.csv")
    graph = matrix_to_digraph(integer_matrix)
    save_pickle(graph, OUTPUT_DIR / "ciona_tadpole_connectome.pkl")
    return graph

def build_celegans_development() -> list[nx.DiGraph]:
    workbook = DATA_DIR / "celegans_development/celegans_developmental_connectivity_matrices.xlsx"
    excel = pd.ExcelFile(workbook)
    graphs: list[nx.DiGraph] = []
    for stage, sheet_name in enumerate(excel.sheet_names[1:], start=1):
        matrix = excel.parse(sheet_name, index_col=0).fillna(0)
        graph = nx.DiGraph()
        for target in matrix.index:
            for source in matrix.columns:
                weight = float(matrix.loc[target, source])
                if weight > 0:
                    graph.add_edge(source, target, weight=weight)
        save_pickle(graph, OUTPUT_DIR / f"celegans_development_stage_{stage:02d}_connectome.pkl")
        graphs.append(graph)
    return graphs

def build_celegans_chemical(sex: str) -> nx.DiGraph:
    if sex not in {"male", "hermaphrodite"}:
        raise ValueError("sex must be 'male' or 'hermaphrodite'")
    folder = DATA_DIR / f"celegans_{sex}"
    nodes = pd.read_csv(folder / f"celegans_{sex}_chemical_nodes.csv")
    edges = pd.read_csv(folder / f"celegans_{sex}_chemical_edges.csv")
    graph = nx.DiGraph()
    name_to_id: dict[str, str] = {}
    for _, row in nodes.iterrows():
        node_id = f"chem_{row['index']}"
        name = str(row["name"])
        canonical_id = name_to_id.get(name, node_id)
        if canonical_id == node_id:
            graph.add_node(node_id, name=name, synapse_type="chemical")
            name_to_id[name] = node_id
        name_to_id[node_id] = canonical_id
    for _, row in edges.iterrows():
        source = name_to_id.get(f"chem_{row['source']}", f"chem_{row['source']}")
        target = name_to_id.get(f"chem_{row['target']}", f"chem_{row['target']}")
        weight = float(row["connectivity"])
        if graph.has_edge(source, target):
            graph[source][target]["weight"] += weight
        else:
            graph.add_edge(source, target, weight=weight, synapse_type="chemical")
    save_pickle(graph, OUTPUT_DIR / f"celegans_{sex}_chemical_connectome.pkl")
    return graph

BUILDERS = {
    "drosophila_larva": build_drosophila_larva,
    "platynereis_larva": build_platynereis_larva,
    "ciona_tadpole": build_ciona_tadpole,
    "celegans_development": build_celegans_development,
    "celegans_male": lambda: build_celegans_chemical("male"),
    "celegans_hermaphrodite": lambda: build_celegans_chemical("hermaphrodite"),
}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["all", *BUILDERS], nargs="?", default="all")
    args = parser.parse_args()
    selected = BUILDERS if args.dataset == "all" else {args.dataset: BUILDERS[args.dataset]}
    for name, builder in selected.items():
        result = builder()
        if isinstance(result, list):
            print(f"{name}: built {len(result)} graphs")
        else:
            print(f"{name}: {result.number_of_nodes()} nodes, {result.number_of_edges()} edges")

if __name__ == "__main__":
    main()
