"""Compute descriptive statistics for processed empirical connectomes."""
from __future__ import annotations
import pandas as pd
import networkx as nx
from .io import load_pickle
from .paths import RESULT_DIR

GRAPH_DIR = RESULT_DIR / "processed_connectomes"
OUTPUT_DIR = RESULT_DIR / "empirical_analysis/dataset_summary"
DATASETS = {
    **{f"celegans_development_{i:02d}": ("C. elegans", GRAPH_DIR / f"celegans_development_stage_{i:02d}_connectome.pkl") for i in range(1, 9)},
    "celegans_male": ("C. elegans", GRAPH_DIR / "celegans_male_chemical_connectome.pkl"),
    "celegans_hermaphrodite": ("C. elegans", GRAPH_DIR / "celegans_hermaphrodite_chemical_connectome.pkl"),
    "platynereis_larva": ("Platynereis", GRAPH_DIR / "platynereis_larva_connectome.pkl"),
    "drosophila_larva": ("Drosophila", GRAPH_DIR / "drosophila_larva_connectome.pkl"),
    "drosophila_adult": ("Drosophila", GRAPH_DIR / "drosophila_adult_connectome.pkl"),
    "ciona_tadpole": ("Ciona intestinalis", GRAPH_DIR / "ciona_tadpole_connectome.pkl"),
}

def summarize_connectomes() -> pd.DataFrame:
    rows = []
    for dataset_id, (species, path) in DATASETS.items():
        graph = load_pickle(path)
        nodes = graph.number_of_nodes()
        edges = graph.number_of_edges()
        rows.append({
            "dataset_id": dataset_id,
            "species": species,
            "nodes": nodes,
            "edges": edges,
            "average_total_degree": sum(dict(graph.degree()).values()) / nodes if nodes else 0.0,
            "density": edges / (nodes * (nodes - 1)) if nodes > 1 else 0.0,
            "reciprocity": nx.reciprocity(graph) if edges else 0.0,
        })
    frame = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_DIR / "connectome_dataset_summary.csv", index=False)
    return frame

def main() -> None:
    frame = summarize_connectomes()
    print(frame.to_string(index=False))

if __name__ == "__main__":
    main()
