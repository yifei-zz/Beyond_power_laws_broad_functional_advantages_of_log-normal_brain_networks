"""Sparse Hopfield associative-memory dynamics on manuscript null networks.

The null-network generator is reused from :mod:`connectome_dynamics.null_networks`.
This module adds only the associative-memory layer and calculates three candidate
network-level responses from a shared set of retrieval experiments:

1. robust memory capacity (critical P/N at fixed cue corruption),
2. retrieval overlap at a fixed load and cue corruption,
3. critical cue-corruption level at a fixed load.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import random
from typing import Iterable

import networkx as nx
import numpy as np
from scipy import sparse
from ..null_networks import DirectedNullNetworkExperiment


class HopfieldExperiment(DirectedNullNetworkExperiment):
    """Run load/noise conditions using the shared five-family network generator."""

    def run_experiment(self, p, *, settings, load_values, noise_values, base_seed):
        rows = []
        noise_by_load = {float(load): sorted(set(float(x) for x in noise_values))
                         for load in sorted(load_values)}
        for family, info in self.distributions.items():
            for parameter_index, parameters in enumerate(info['param_list']):
                for replicate in range(self.n_samples):
                    task_id = (f'n{self.n_nodes}_p{float(p):.1f}_'
                               f'{family}_par{parameter_index}_rep{replicate:02d}')
                    digest = hashlib.sha256(f'{base_seed}|{task_id}'.encode('utf-8')).digest()
                    seed = int.from_bytes(digest[:4], 'little', signed=False)
                    np.random.seed(seed)
                    random.seed(seed)
                    rng = np.random.default_rng(seed)
                    graph = self.generate_network(family, parameters, p)
                    connectivity = effective_connectivity(graph, settings.connectivity_mode)
                    reciprocity = nx.reciprocity(graph)
                    metadata = {
                        'task_id': task_id, 'n_nodes': int(self.n_nodes), 'p': float(p),
                        'distribution_type': family, 'parameter_index': parameter_index,
                        'distribution_parameters': json.dumps(parameters, sort_keys=True, separators=(',', ':')),
                        'replicate': replicate, 'seed': seed,
                        'node_count': graph.number_of_nodes(), 'edge_count': graph.number_of_edges(),
                        'avg_degree': float(np.mean([degree for _, degree in graph.degree()])),
                        'effective_avg_degree': float(connectivity.sum(axis=1).mean()),
                        'reciprocity': float(reciprocity) if reciprocity is not None else float('nan'),
                    }
                    for result in evaluate_load_grid(connectivity, noise_by_load=noise_by_load,
                                                     rng=rng, settings=settings):
                        rows.append({**asdict(result), **metadata})
        return rows


@dataclass(frozen=True)
class HopfieldSettings:
    connectivity_mode: str = "symmetrized_mean"
    hebbian_normalization: str = "n_nodes"
    zero_field_policy: str = "positive"
    max_update_steps: int = 25
    convergence_patience: int = 1
    max_test_patterns: int = 5
    cue_repeats: int = 1
    success_overlap_threshold: float = 0.95
    success_rate_threshold: float = 0.80
    edge_chunk_size: int = 200_000


@dataclass
class RetrievalSummary:
    memory_load: float
    pattern_count: int
    noise_level: float
    n_cues: int
    mean_final_overlap: float
    median_final_overlap: float
    success_rate: float
    mean_update_steps: float
    converged_fraction: float


def effective_connectivity(graph: nx.DiGraph, mode: str) -> sparse.csr_matrix:
    """Return the sparse coupling mask used by the Hopfield layer.

    ``symmetrized_mean`` preserves the directed-network parameter through edge
    weights: a one-way pair contributes 0.5 in both directions and a reciprocal
    pair contributes 1.0. ``as_generated`` uses the directed adjacency exactly.
    """
    nodes = list(graph.nodes())
    adjacency = nx.to_scipy_sparse_array(
        graph, nodelist=nodes, dtype=np.float32, format="csr"
    )
    adjacency.setdiag(0.0)
    adjacency.eliminate_zeros()

    if mode == "symmetrized_mean":
        adjacency = ((adjacency + adjacency.T) * np.float32(0.5)).tocsr()
    elif mode == "reciprocal_only":
        adjacency = adjacency.minimum(adjacency.T).tocsr()
    elif mode != "as_generated":
        raise ValueError(
            "connectivity_mode must be 'symmetrized_mean', "
            "'reciprocal_only', or 'as_generated'"
        )

    adjacency.setdiag(0.0)
    adjacency.eliminate_zeros()
    return adjacency


def hebbian_sparse_weights(
    connectivity: sparse.csr_matrix,
    patterns: np.ndarray,
    *,
    normalization: str,
    edge_chunk_size: int = 200_000,
) -> sparse.csr_matrix:
    """Calculate Hebbian weights only on existing edges.

    This avoids constructing the dense ``N × N`` Hebbian matrix and therefore
    supports the manuscript's largest networks on a 32 GB workstation.
    """
    if patterns.ndim != 2:
        raise ValueError("patterns must have shape (n_patterns, n_nodes)")
    n_patterns, n_nodes = patterns.shape
    coo = connectivity.tocoo(copy=False)
    rows = np.asarray(coo.row, dtype=np.int64)
    cols = np.asarray(coo.col, dtype=np.int64)
    mask_values = np.asarray(coo.data, dtype=np.float32)
    values = np.empty(len(rows), dtype=np.float32)

    if normalization == "n_nodes":
        denominator = float(n_nodes)
    elif normalization == "n_patterns":
        denominator = float(max(n_patterns, 1))
    else:
        raise ValueError("hebbian_normalization must be 'n_nodes' or 'n_patterns'")

    int_patterns = np.asarray(patterns, dtype=np.int8)
    for start in range(0, len(rows), edge_chunk_size):
        stop = min(start + edge_chunk_size, len(rows))
        # Temporary shape is (P, chunk), kept bounded by edge_chunk_size.
        edge_products = (
            int_patterns[:, rows[start:stop]] * int_patterns[:, cols[start:stop]]
        )
        correlations = edge_products.sum(axis=0, dtype=np.int32) / denominator
        values[start:stop] = mask_values[start:stop] * correlations.astype(np.float32)

    weights = sparse.csr_matrix(
        (values, (rows, cols)), shape=connectivity.shape, dtype=np.float32
    )
    weights.setdiag(0.0)
    weights.eliminate_zeros()
    return weights


def retrieve_synchronous_batch(
    weights: sparse.csr_matrix,
    initial_states: np.ndarray,
    *,
    max_steps: int,
    convergence_patience: int,
    zero_field_policy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retrieve a batch of cues with synchronous sparse updates."""
    states = np.asarray(initial_states, dtype=np.int8).copy()
    n_states = states.shape[0]
    stable_counts = np.zeros(n_states, dtype=np.int16)
    converged = np.zeros(n_states, dtype=bool)
    convergence_steps = np.full(n_states, max_steps, dtype=np.int16)

    for step in range(1, max_steps + 1):
        fields = weights.dot(states.T.astype(np.float32)).T
        if zero_field_policy == "positive":
            new_states = np.where(fields >= 0.0, 1, -1).astype(np.int8)
        elif zero_field_policy == "retain":
            new_states = np.where(
                fields > 0.0, 1, np.where(fields < 0.0, -1, states)
            ).astype(np.int8)
        else:
            raise ValueError("zero_field_policy must be 'positive' or 'retain'")

        unchanged = np.all(new_states == states, axis=1)
        stable_counts = np.where(unchanged, stable_counts + 1, 0)
        newly_converged = (~converged) & (stable_counts >= convergence_patience)
        convergence_steps[newly_converged] = step
        converged |= newly_converged
        states = new_states
        if np.all(converged):
            break

    return states, converged, convergence_steps


def random_patterns(rng: np.random.Generator, count: int, n_nodes: int) -> np.ndarray:
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=(count, n_nodes))


def corrupted_cue(
    pattern: np.ndarray, noise_level: float, rng: np.random.Generator
) -> np.ndarray:
    cue = np.asarray(pattern, dtype=np.int8).copy()
    n_flip = int(round(float(noise_level) * cue.size))
    if n_flip > 0:
        positions = rng.choice(cue.size, size=min(n_flip, cue.size), replace=False)
        cue[positions] *= -1
    return cue


def evaluate_load(
    connectivity: sparse.csr_matrix,
    *,
    memory_load: float,
    noise_levels: Iterable[float],
    rng: np.random.Generator,
    settings: HopfieldSettings,
) -> list[RetrievalSummary]:
    """Store one pattern set and evaluate all requested cue-noise levels."""
    n_nodes = connectivity.shape[0]
    pattern_count = max(1, int(round(memory_load * n_nodes)))
    patterns = random_patterns(rng, pattern_count, n_nodes)
    weights = hebbian_sparse_weights(
        connectivity,
        patterns,
        normalization=settings.hebbian_normalization,
        edge_chunk_size=settings.edge_chunk_size,
    )

    n_targets = min(pattern_count, settings.max_test_patterns)
    if n_targets == pattern_count:
        target_indices = np.arange(pattern_count, dtype=int)
    else:
        target_indices = np.sort(
            rng.choice(pattern_count, size=n_targets, replace=False)
        )

    cues: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    cue_noise: list[float] = []
    for noise in sorted(set(float(x) for x in noise_levels)):
        for target_index in target_indices:
            for _ in range(settings.cue_repeats):
                target = patterns[target_index]
                cues.append(corrupted_cue(target, noise, rng))
                targets.append(target)
                cue_noise.append(noise)

    cue_matrix = np.stack(cues, axis=0)
    target_matrix = np.stack(targets, axis=0)
    final_states, converged, steps = retrieve_synchronous_batch(
        weights,
        cue_matrix,
        max_steps=settings.max_update_steps,
        convergence_patience=settings.convergence_patience,
        zero_field_policy=settings.zero_field_policy,
    )
    overlaps = np.mean(final_states * target_matrix, axis=1, dtype=np.float64)
    successes = overlaps >= settings.success_overlap_threshold
    cue_noise_array = np.asarray(cue_noise, dtype=float)

    summaries: list[RetrievalSummary] = []
    for noise in sorted(set(cue_noise)):
        selected = np.isclose(cue_noise_array, noise)
        summaries.append(
            RetrievalSummary(
                memory_load=float(memory_load),
                pattern_count=int(pattern_count),
                noise_level=float(noise),
                n_cues=int(selected.sum()),
                mean_final_overlap=float(np.mean(overlaps[selected])),
                median_final_overlap=float(np.median(overlaps[selected])),
                success_rate=float(np.mean(successes[selected])),
                mean_update_steps=float(np.mean(steps[selected])),
                converged_fraction=float(np.mean(converged[selected])),
            )
        )
    return summaries


def derive_candidate_metrics(
    detailed_rows: list[dict],
    *,
    load_values: list[float],
    noise_values: list[float],
    capacity_noise_level: float,
    fixed_overlap_load: float,
    fixed_overlap_noise: float,
    critical_noise_load: float,
    success_overlap_threshold: float,
    success_rate_threshold: float,
) -> dict[str, float | int]:
    """Derive the three candidate network-level responses."""
    def qualifies(row: dict) -> bool:
        return (
            row["mean_final_overlap"] >= success_overlap_threshold
            and row["success_rate"] >= success_rate_threshold
        )

    capacity_rows = [
        row
        for row in detailed_rows
        if np.isclose(row["noise_level"], capacity_noise_level)
        and any(np.isclose(row["memory_load"], x) for x in load_values)
    ]
    successful_loads = [row["memory_load"] for row in capacity_rows if qualifies(row)]
    memory_capacity = max(successful_loads, default=0.0)
    capacity_left_censored = int(memory_capacity == 0.0)
    capacity_right_censored = int(
        bool(successful_loads) and np.isclose(memory_capacity, max(load_values))
    )

    overlap_candidates = [
        row
        for row in detailed_rows
        if np.isclose(row["memory_load"], fixed_overlap_load)
        and np.isclose(row["noise_level"], fixed_overlap_noise)
    ]
    retrieval_overlap = (
        float(overlap_candidates[0]["mean_final_overlap"])
        if overlap_candidates
        else float("nan")
    )

    noise_rows = [
        row
        for row in detailed_rows
        if np.isclose(row["memory_load"], critical_noise_load)
        and any(np.isclose(row["noise_level"], x) for x in noise_values)
    ]
    successful_noise = [row["noise_level"] for row in noise_rows if qualifies(row)]
    critical_noise = max(successful_noise, default=0.0)
    critical_noise_left_censored = int(not successful_noise)
    critical_noise_right_censored = int(
        bool(successful_noise) and np.isclose(critical_noise, max(noise_values))
    )

    return {
        "memory_capacity": float(memory_capacity),
        "retrieval_overlap": float(retrieval_overlap),
        "critical_noise_level": float(critical_noise),
        "capacity_left_censored": capacity_left_censored,
        "capacity_right_censored": capacity_right_censored,
        "critical_noise_left_censored": critical_noise_left_censored,
        "critical_noise_right_censored": critical_noise_right_censored,
    }


def evaluate_load_grid(
    connectivity: sparse.csr_matrix,
    *,
    noise_by_load: dict[float, list[float]],
    rng: np.random.Generator,
    settings: HopfieldSettings,
) -> list[RetrievalSummary]:
    """Evaluate all load/noise conditions while incrementally building weights.

    Patterns are nested across load levels and edge correlations are accumulated
    only for newly added patterns. This gives the same Hebbian weights for each
    requested load while avoiding repeated work across the load sweep.
    """
    n_nodes = connectivity.shape[0]
    load_to_count = {
        float(load): max(1, int(round(float(load) * n_nodes)))
        for load in noise_by_load
    }
    max_count = max(load_to_count.values())
    patterns = random_patterns(rng, max_count, n_nodes)

    coo = connectivity.tocoo(copy=False)
    rows = np.asarray(coo.row, dtype=np.int64)
    cols = np.asarray(coo.col, dtype=np.int64)
    mask_values = np.asarray(coo.data, dtype=np.float32)
    cumulative = np.zeros(len(rows), dtype=np.int32)
    previous_count = 0
    output: list[RetrievalSummary] = []

    count_to_loads: dict[int, list[float]] = {}
    for load, count in load_to_count.items():
        count_to_loads.setdefault(count, []).append(load)

    if settings.hebbian_normalization == "n_nodes":
        denominator = float(n_nodes)
    elif settings.hebbian_normalization == "n_patterns":
        denominator = None
    else:
        raise ValueError("hebbian_normalization must be 'n_nodes' or 'n_patterns'")

    for pattern_count in sorted(count_to_loads):
        new_patterns = patterns[previous_count:pattern_count]
        for start in range(0, len(rows), settings.edge_chunk_size):
            stop = min(start + settings.edge_chunk_size, len(rows))
            if len(new_patterns):
                products = (
                    new_patterns[:, rows[start:stop]]
                    * new_patterns[:, cols[start:stop]]
                )
                cumulative[start:stop] += products.sum(axis=0, dtype=np.int32)
        current_denominator = denominator or float(pattern_count)
        values = mask_values * (cumulative.astype(np.float32) / current_denominator)
        weights = sparse.csr_matrix(
            (values, (rows, cols)), shape=connectivity.shape, dtype=np.float32
        )
        weights.setdiag(0.0)
        weights.eliminate_zeros()

        loads = sorted(count_to_loads[pattern_count])
        union_noises = sorted(
            {noise for load in loads for noise in noise_by_load[load]}
        )
        n_targets = min(pattern_count, settings.max_test_patterns)
        if n_targets == pattern_count:
            target_indices = np.arange(pattern_count, dtype=int)
        else:
            target_indices = np.sort(
                rng.choice(pattern_count, size=n_targets, replace=False)
            )

        cues: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        cue_noise: list[float] = []
        for noise in union_noises:
            for target_index in target_indices:
                for _ in range(settings.cue_repeats):
                    target = patterns[target_index]
                    cues.append(corrupted_cue(target, noise, rng))
                    targets.append(target)
                    cue_noise.append(float(noise))

        final_states, converged, steps = retrieve_synchronous_batch(
            weights,
            np.stack(cues, axis=0),
            max_steps=settings.max_update_steps,
            convergence_patience=settings.convergence_patience,
            zero_field_policy=settings.zero_field_policy,
        )
        target_matrix = np.stack(targets, axis=0)
        overlaps = np.mean(final_states * target_matrix, axis=1, dtype=np.float64)
        successes = overlaps >= settings.success_overlap_threshold
        cue_noise_array = np.asarray(cue_noise, dtype=float)

        noise_summary: dict[float, dict[str, float | int]] = {}
        for noise in union_noises:
            selected = np.isclose(cue_noise_array, noise)
            noise_summary[noise] = {
                "n_cues": int(selected.sum()),
                "mean_final_overlap": float(np.mean(overlaps[selected])),
                "median_final_overlap": float(np.median(overlaps[selected])),
                "success_rate": float(np.mean(successes[selected])),
                "mean_update_steps": float(np.mean(steps[selected])),
                "converged_fraction": float(np.mean(converged[selected])),
            }

        for load in loads:
            for noise in noise_by_load[load]:
                item = noise_summary[float(noise)]
                output.append(
                    RetrievalSummary(
                        memory_load=float(load),
                        pattern_count=int(pattern_count),
                        noise_level=float(noise),
                        n_cues=int(item["n_cues"]),
                        mean_final_overlap=float(item["mean_final_overlap"]),
                        median_final_overlap=float(item["median_final_overlap"]),
                        success_rate=float(item["success_rate"]),
                        mean_update_steps=float(item["mean_update_steps"]),
                        converged_fraction=float(item["converged_fraction"]),
                    )
                )
        previous_count = pattern_count
    return output
