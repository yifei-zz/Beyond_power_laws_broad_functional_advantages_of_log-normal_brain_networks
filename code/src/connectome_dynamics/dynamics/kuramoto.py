# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
from scipy.special import gamma
from scipy.stats import kstest
from collections import defaultdict
from scipy.integrate import quad
import random
import os
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

from ..null_networks import DirectedNullNetworkExperiment
from ..paths import PROJECT_ROOT


class KuramotoExperiment(DirectedNullNetworkExperiment):
    def assign_natural_frequencies(self, G, mean=0, std=0.1):
        frequencies = {}
        for node in G.nodes():
            frequencies[node] = np.random.normal(mean, std)
        return frequencies

    def calculate_order_parameter(self, phases, nodes):
        if not nodes:
            return 0.0
        complex_sum = np.mean([np.exp(1j * phases[node]) for node in nodes])
        return np.abs(complex_sum)

    def simulate_kuramoto(self, G, frequencies, lambda_coupling, dt=0.05, max_steps=None):
        if max_steps is not None and max_steps < 1:
            raise ValueError('max_steps must be positive')
        nodes = list(G.nodes())
        N = len(nodes)
        phases = {node: np.random.uniform(0, 2 * np.pi) for node in nodes}
        R_t = []
        while True:
            new_phases = phases.copy()
            for i in nodes:
                interaction = 0
                for j in G.predecessors(i):
                    phase_diff = phases[j] - phases[i]
                    phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi
                    interaction += np.sin(phase_diff)
                interaction /= N
                dphase = dt * (frequencies[i] + lambda_coupling * interaction)
                new_phases[i] = (phases[i] + dphase) % (2 * np.pi)
            phases = new_phases
            R = self.calculate_order_parameter(phases, nodes)
            R_t.append(float(R))
            if len(R_t) >= 10 and R - R_t[-10] < 0.001:
                break
            if max_steps is not None and len(R_t) >= max_steps:
                break
        return R_t

    def run_multiple_simulations(self, G, lambda_values, std, dt=0.05):
        all_results = defaultdict(lambda: {'R_value': -1, 'R_t': []})
        for lambda_coupling in lambda_values:
            R_t = []
            frequencies = self.assign_natural_frequencies(G, mean=0, std=std)
            R_t = self.simulate_kuramoto(G, frequencies, lambda_coupling, dt,
                                        max_steps=getattr(self, 'max_steps', None))
            all_results[lambda_coupling]['R_value'] = R_t[-1]
            all_results[lambda_coupling]['R_t'] = R_t
        return all_results

    def run_experiment(self, p, n_nodes, lambda_values, std, dt=0.05):
        self.results = {}
        self.networks = {}
        self.distribution_fit = {}
        all_samples_data = []
        for dist_name, dist_info in self.distributions.items():
            param_list = dist_info['param_list']
            for param_idx, params in enumerate(param_list):
                param_str = '_'.join((f'{k}{v}' for k, v in params.items()))
                network_type = f'{dist_name}_{param_str}'
                for sample in range(self.n_samples):
                    degrees = self.generate_degrees(dist_name, **params)
                    G_undirected = self.create_undirected_graph(degrees)
                    G_directed = self.create_directed_graph(G_undirected, p)
                    sample_results = self.run_multiple_simulations(G_directed, lambda_values, std, dt=dt)
                    for lambda_coupling, R in sample_results.items():
                        sample_data = {'n_nodes': n_nodes, 'p': p, 'avg_degree': np.mean(degrees), 'distribution_type': dist_name, 'params': param_str, 'full_network_type': network_type, 'sample_id': sample, 'std': std, 'lambda': lambda_coupling, 'R_value': R['R_value'], 'R_t': R['R_t']}
                        all_samples_data.append(sample_data)
        return all_samples_data

# # 完整并行仿真

import multiprocessing
from functools import partial

def run_single_experiment(params):
    """运行单个参数组合的实验，返回结果DataFrame"""
    lambda_values = [100, 800, 3000]
    dt = 0.1

    n_nodes, p, std = params
    base_filename = f"n{n_nodes}-p{int(p*10)}-std{std*10}"
    result_dir = str(PROJECT_ROOT / "result" / "dynamics" / "kuramoto")
    os.makedirs(result_dir, exist_ok=True)
    
    csv_filepath = os.path.join(result_dir, f"{base_filename}.csv")
    
    # 如果文件已存在，直接读取并返回
    if os.path.exists(csv_filepath):
        print(f"文件已存在，直接读取: {base_filename}")
        return pd.read_csv(csv_filepath)
    
    experiment = KuramotoExperiment(
        n_nodes=n_nodes,
        n_samples=5
    )
    
    # 运行实验，传递参数信息
    results = experiment.run_experiment(p, n_nodes, lambda_values, std, dt)
    
    samples_df = pd.DataFrame(results)
    samples_df.to_csv(csv_filepath, index=False)
    
    print(f"5次实验合并CSV已保存到: {csv_filepath}")
    
    return samples_df

def run_parameter_sensitivity_parallel():
    """并行运行参数灵敏度分析实验并合并所有结果"""
    n_nodes_list = [300, 500, 1000, 2000, 3000, 5000, 8000, 10000]
    p_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    std_list = [0.1, 0.5, 1.0]

    param_combinations = [
        (n, p, std)
        for n in n_nodes_list
        for p in p_list
        for std in std_list
    ]
    
    # 创建进程池，使用最大31个进程
    with multiprocessing.Pool(processes=min(31, multiprocessing.cpu_count())) as pool:
        all_results = pool.map(run_single_experiment, param_combinations)
    
    combined_df = pd.concat(all_results, ignore_index=True)
    
    combined_csv_path = str(PROJECT_ROOT / "result" / "dynamics" / "kuramoto" / "kuramoto_simulation_results.csv")
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"\n{'='*50}")
    print(f"所有实验结果已合并保存到: {combined_csv_path}")
    print(f"总记录数: {len(combined_df)}")
    print(f"参数组合数: {len(param_combinations)}")
    print(f"总样本数: {len(combined_df)}")
    print(f"{'='*50}")
    
    return combined_df

if __name__ == "__main__":
    final_df = run_parameter_sensitivity_parallel()
