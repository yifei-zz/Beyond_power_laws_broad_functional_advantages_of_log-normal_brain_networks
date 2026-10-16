
"""Fit candidate heavy-tailed degree distributions to every empirical connectome."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import powerlaw
from scipy.stats import ksone

from .paths import RESULT_DIR
from .io import load_pickle

GRAPH_DIR = RESULT_DIR / "processed_connectomes"
OUT_DIR = RESULT_DIR / "empirical_analysis" / "degree_distribution"
CANDIDATES = {
    "power_law": 2,
    "exponential": 1,
    "lognormal": 2,
    "stretched_exponential": 2,
    "truncated_power_law": 2,
}


def degree_sequence(graph, mode: str) -> list[int]:
    if mode == "in":
        values = [degree for _, degree in graph.in_degree()]
    elif mode == "out":
        values = [degree for _, degree in graph.out_degree()]
    else:
        values = [degree for _, degree in graph.degree()]
    return [int(value) for value in values if value > 0]


def analyze(path: Path, mode: str) -> list[dict]:
    graph = load_pickle(path)
    values = np.asarray(degree_sequence(graph, mode))
    fit = powerlaw.Fit(values, xmin=None, discrete=True, verbose=False)
    n_tail = int(np.sum(values >= fit.xmin))
    rows = []
    for name, parameter_count in CANDIDATES.items():
        distribution = getattr(fit, name)
        ks = float(distribution.KS())
        p_value = float(ksone.sf(ks, n_tail))
        log_likelihood = float(np.sum(distribution.loglikelihoods(values)))
        rows.append({
            "distribution": name,
            "xmin": float(fit.xmin),
            "power_law_alpha": float(fit.power_law.alpha),
            "ks_statistic": ks,
            "ks_p_value": p_value,
            "log_likelihood": log_likelihood,
            "aic": 2 * parameter_count - 2 * log_likelihood,
            "bic": parameter_count * np.log(fit.n) - 2 * log_likelihood,
            "n_tail": n_tail,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree-mode", choices=["total", "in", "out"], default="total")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for path in sorted(GRAPH_DIR.glob("*_connectome.pkl")):
        for row in analyze(path, args.degree_mode):
            row["dataset_id"] = path.stem.removesuffix("_connectome")
            row["degree_mode"] = args.degree_mode
            all_rows.append(row)
    pd.DataFrame(all_rows).to_csv(OUT_DIR / f"degree_distribution_fits_{args.degree_mode}.csv", index=False)


if __name__ == "__main__":
    main()
