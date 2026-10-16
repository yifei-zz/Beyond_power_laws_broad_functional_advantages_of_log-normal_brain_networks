"""Generate directed null networks with the five degree-distribution families used in the paper."""
from __future__ import annotations

import argparse
import pickle
import random as python_random
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.integrate import quad
from scipy.stats import kstest

from .paths import RESULT_DIR

DISTRIBUTION_PARAMETERS = {
    "Exponential": [{"lambda": 0.1}, {"lambda": 0.25}, {"lambda": 0.5}],
    "Power_law": [{"alpha": 2.2}, {"alpha": 2.5}, {"alpha": 3.0}],
    "Truncated_Power_law": [
        {"alpha": 1.8, "lambda": 30}, {"alpha": 2.0, "lambda": 50}, {"alpha": 2.3, "lambda": 50}
    ],
    "Log_normal": [{"mu": 2.0, "sigma": 0.4}, {"mu": 2.8, "sigma": 0.7}, {"mu": 3.5, "sigma": 1.0}],
    "Stretched_Exponential": [
        {"alpha": 0.4, "beta": 25}, {"alpha": 0.6, "beta": 18}, {"alpha": 0.8, "beta": 12}
    ],
}

class DirectedNullNetworkExperiment:
    def __init__(self, n_nodes: int = 500, n_samples: int = 3, avg_degree: float | None = None):
        self.n_nodes = n_nodes
        self.n_samples = n_samples
        self.avg_degree = avg_degree
        self.max_degree = n_nodes
        self.k_min = 1
        self.distributions = {name: {"param_list": values} for name, values in DISTRIBUTION_PARAMETERS.items()}
        self.networks = {}
        self.results = {}
        self.biological_metrics = {}
        self.distribution_fit = {}

    def truncated_power_law_pdf(self, x, alpha, lambda_):
        return x ** (-alpha) * np.exp(-x / lambda_)

    def truncated_power_law_norm_const(self, alpha, lambda_):
        integral, _ = quad(lambda x: x ** (-alpha) * np.exp(-x / lambda_), self.k_min, np.inf)
        return 1 / integral

    def generate_truncated_power_law(self, alpha, lambda_):
        constant = self.truncated_power_law_norm_const(alpha, lambda_)
        k_max = min(self.max_degree, int(10 * lambda_))
        k_values = np.arange(self.k_min, k_max + 1)
        pmf = np.array([constant * self.truncated_power_law_pdf(k, alpha, lambda_) for k in k_values])
        pmf /= pmf.sum()
        return np.random.choice(k_values, size=self.n_nodes, p=pmf)

    def generate_degrees(self, dist, **params):
        if dist == "Exponential":
            degrees = np.random.exponential(1 / params["lambda"], size=self.n_nodes)
        elif dist == "Power_law":
            x = np.arange(1, self.max_degree)
            pmf = x ** (-params["alpha"])
            pmf /= pmf.sum()
            degrees = np.random.choice(x, size=self.n_nodes, p=pmf)
        elif dist == "Truncated_Power_law":
            degrees = self.generate_truncated_power_law(params["alpha"], params["lambda"])
        elif dist == "Log_normal":
            degrees = np.random.lognormal(params["mu"], params["sigma"], self.n_nodes)
        elif dist == "Stretched_Exponential":
            u = np.random.uniform(0, 1, size=self.n_nodes)
            degrees = params["beta"] * (-np.log(1 - u)) ** (1 / params["alpha"])
        else:
            raise ValueError(f"Unsupported distribution: {dist}")
        degrees = np.clip(np.round(degrees).astype(int), 1, self.max_degree)
        self.evaluate_distribution_fit(dist, degrees, params)
        return degrees

    def evaluate_distribution_fit(self, dist, degrees, params):
        if len(np.unique(degrees)) < 2:
            return
        sorted_deg = np.sort(degrees)
        if dist == "Exponential":
            theoretical_cdf = 1 - np.exp(-params["lambda"] * sorted_deg)
        elif dist == "Power_law":
            x = np.arange(1, sorted_deg.max() + 1)
            pmf = x ** (-params["alpha"]); pmf /= pmf.sum()
            theoretical_cdf = np.interp(sorted_deg, x, np.cumsum(pmf), left=0, right=1)
        elif dist == "Truncated_Power_law":
            constant = self.truncated_power_law_norm_const(params["alpha"], params["lambda"])
            x = np.arange(1, sorted_deg.max() + 1)
            pdf = np.array([constant * self.truncated_power_law_pdf(k, params["alpha"], params["lambda"]) for k in x])
            theoretical_cdf = np.interp(sorted_deg, x, np.cumsum(pdf), left=0, right=1)
        elif dist == "Log_normal":
            from scipy.stats import norm
            theoretical_cdf = norm.cdf((np.log(sorted_deg + 1e-10) - params["mu"]) / params["sigma"])
        else:
            theoretical_cdf = 1 - np.exp(-(sorted_deg / params["beta"]) ** params["alpha"])
        ks_stat, p_value = kstest(degrees, lambda x: np.interp(x, sorted_deg, theoretical_cdf))
        key = f"{dist}_{'_'.join(f'{k}{v}' for k, v in params.items())}"
        self.distribution_fit[key] = {"ks_stat": ks_stat, "p_value": p_value, "mean": np.mean(degrees)}

    def create_undirected_graph(self, degrees):
        graph = nx.Graph()
        graph.add_nodes_from(range(len(degrees)))
        added_edges = set()
        for node, degree in enumerate(degrees):
            available = [target for target in range(len(degrees)) if target != node and not graph.has_edge(node, target)]
            remaining = min(max(int(degree) - graph.degree(node), 0), len(available))
            if remaining:
                for target in np.random.choice(available, size=remaining, replace=False):
                    edge = tuple(sorted((node, int(target))))
                    if edge not in added_edges:
                        graph.add_edge(*edge); added_edges.add(edge)
        return graph

    def create_directed_graph(self, graph, p):
        directed = nx.DiGraph(); directed.add_nodes_from(graph.nodes)
        for source, target in graph.edges():
            if source == target:
                directed.add_edge(source, target)
            elif source < target:
                if python_random.random() < p:
                    directed.add_edge(source, target)
                    if python_random.random() < p:
                        directed.add_edge(target, source)
                else:
                    directed.add_edge(target, source)
        return directed

    def generate_network(self, distribution: str, parameters: dict, p: float) -> nx.DiGraph:
        degrees = self.generate_degrees(distribution, **parameters)
        return self.create_directed_graph(self.create_undirected_graph(degrees), p)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=300)
    parser.add_argument("--distribution", choices=DISTRIBUTION_PARAMETERS, default="Power_law")
    parser.add_argument("--parameter-index", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--direction-probability", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    experiment = DirectedNullNetworkExperiment(args.nodes)
    parameters = DISTRIBUTION_PARAMETERS[args.distribution][args.parameter_index]
    graph = experiment.generate_network(args.distribution, parameters, args.direction_probability)
    output = args.output or RESULT_DIR / "null_networks" / "example_directed_null_network.pkl"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {output}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

if __name__ == "__main__":
    main()
