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
import scipy.sparse as sp

from ..null_networks import DirectedNullNetworkExperiment
from ..paths import PROJECT_ROOT


class SpectralAbscissaExperiment(DirectedNullNetworkExperiment):
    def spectral_abscissa(self, G):
        n = G.number_of_nodes()
        W = nx.adjacency_matrix(G)
        I = sp.eye(n, format='csc')
        A = -I + W
        if n <= 1000:
            A_dense = A.toarray()
            eigenvalues = np.linalg.eigvals(A_dense)
            Re_lambda_max = max(eigenvalues.real)
        else:
            try:
                eigenvalues = sp.linalg.eigs(A, k=1, which='LR', return_eigenvectors=False)
                Re_lambda_max = eigenvalues[0].real
            except (sp.linalg.ArpackError, ValueError):
                k = min(n, 100)
                eigenvalues = sp.linalg.eigs(A, k=k, which='LM', return_eigenvectors=False)
                Re_lambda_max = max(eigenvalues.real)
        return Re_lambda_max

    def run_experiment(self, p, n_nodes):
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
                    sample_results = self.spectral_abscissa(G_directed)
                    sample_data = {'n_nodes': n_nodes, 'p': p, 'avg_degree': np.mean(degrees), 'distribution_type': dist_name, 'params': param_str, 'full_network_type': network_type, 'sample_id': sample, 'Re_lambda_max': sample_results}
                    all_samples_data.append(sample_data)
        return all_samples_data

# # 完整并行仿真

import multiprocessing
from functools import partial

def run_single_experiment(params):
    """运行单个参数组合的实验，返回结果DataFrame"""
    n_nodes, p = params
    base_filename = f"n{n_nodes}-p{int(p*10)}"
    result_dir = str(PROJECT_ROOT / "result" / "dynamics" / "spectral_abscissa")
    os.makedirs(result_dir, exist_ok=True)
    
    csv_filepath = os.path.join(result_dir, f"{base_filename}.csv")
    
    # 如果文件已存在，直接读取并返回
    if os.path.exists(csv_filepath):
        print(f"文件已存在，直接读取: {base_filename}")
        return pd.read_csv(csv_filepath)
    
    experiment = SpectralAbscissaExperiment(
        n_nodes=n_nodes,
        n_samples=10
    )
    
    # 运行实验，传递参数信息
    results = experiment.run_experiment(p, n_nodes)
    
    samples_df = pd.DataFrame(results)
    samples_df.to_csv(csv_filepath, index=False)
    
    print(f"10次实验合并CSV已保存到: {csv_filepath}")
    
    return samples_df

def run_parameter_sensitivity_parallel():
    """并行运行参数灵敏度分析实验并合并所有结果"""
    n_nodes_list = [10000, 8000, 5000, 3000, 2000, 1000, 500, 300]
    p_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    param_combinations = [
        (n, p)
        for n in n_nodes_list
        for p in p_list
    ]
    
    # 创建进程池，使用最大31个进程
    with multiprocessing.Pool(processes=min(31, multiprocessing.cpu_count())) as pool:
        all_results = pool.map(run_single_experiment, param_combinations)
    
    combined_df = pd.concat(all_results, ignore_index=True)
    
    combined_csv_path = str(PROJECT_ROOT / "result" / "dynamics" / "spectral_abscissa" / "spectral_abscissa_simulation_results.csv")
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
