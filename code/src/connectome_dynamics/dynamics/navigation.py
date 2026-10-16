# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.stats import kstest
from collections import defaultdict
import re
from scipy.integrate import quad
import random
import dill
import os
import pandas as pd
from pathlib import Path


from ..null_networks import DirectedNullNetworkExperiment
from ..paths import PROJECT_ROOT


class NavigationExperiment(DirectedNullNetworkExperiment):
    def greedy_navigation(self, G, source, target):
        """使用贪婪策略进行导航（基于拓扑距离）"""
        if source == target:
            return ([source], 0, 1.0)
        try:
            length_dict = dict(nx.single_target_shortest_path_length(G, target))
        except nx.NetworkXNoPath:
            return (None, float('inf'), 0.0)
        shortest_path_length = length_dict.get(source, float('inf'))
        if shortest_path_length == float('inf'):
            return (None, float('inf'), 0.0)
        current = source
        path = [current]
        steps = 0
        while current != target:
            neighbors = list(G.successors(current))
            if not neighbors:
                return (None, float('inf'), 0.0)
            neighbor_distances = [length_dict.get(n, float('inf')) for n in neighbors]
            min_distance = min(neighbor_distances)
            best_neighbors = [n for i, n in enumerate(neighbors) if neighbor_distances[i] == min_distance]
            next_node = np.random.choice(best_neighbors)
            path.append(next_node)
            current = next_node
            steps += 1
            if steps > self.n_nodes * 2:
                return (None, float('inf'), 0.0)
        actual_length = len(path) - 1
        efficiency = shortest_path_length / actual_length if actual_length > 0 else 1.0
        return (path, actual_length, efficiency)

    def run_navigation_for_network(self, G, network_type, sample_idx):
        results = {'path_lengths': [], 'efficiency': [], 'success_count': 0, 'total_pairs': 0}
        nodes = list(G.nodes())
        n_pairs = 500
        for _ in range(n_pairs):
            source, target = np.random.choice(nodes, size=2, replace=False)
            _, path_length, eff = self.greedy_navigation(G, source, target)
            if path_length < float('inf'):
                results['path_lengths'].append(path_length)
                results['efficiency'].append(eff)
                results['success_count'] += 1
            results['total_pairs'] += 1
        results['success_rate'] = results['success_count'] / results['total_pairs'] if results['total_pairs'] > 0 else 0.0
        if sample_idx == self.n_samples - 1:
            self.networks[network_type] = G.copy()
        return (results, G)

    def calculate_biological_metrics(self):
        """计算具有生物学可解释性的评价指标（适应有向网络）"""
        metrics = {}
        for network_type, result in self.results.items():
            g = self.networks.get(network_type)
            if g is None:
                continue
            avg_path_length = np.mean(result['path_lengths']) if result['path_lengths'] else 0
            if len(result['path_lengths']) > 1 and avg_path_length > 0:
                path_length_cv = np.std(result['path_lengths']) / avg_path_length
            else:
                path_length_cv = 0
            total_degrees = [d for n, d in g.degree()]
            if total_degrees:
                degree_counts = np.bincount(total_degrees)
                degree_probs = degree_counts / sum(degree_counts)
                degree_entropy = -np.sum(degree_probs * np.log2(degree_probs + 1e-10))
            else:
                degree_entropy = 0
            metrics[network_type] = {'success_rate': result['success_rate'], 'avg_path_length': avg_path_length, 'path_length_cv': path_length_cv, 'degree_entropy': degree_entropy}
        self.biological_metrics = metrics
        return metrics

    def plot_degree_distributions(self):
        """绘制不同有向网络的出度分布"""
        if not self.networks:
            print('没有网络数据可供绘图')
            return
        plt.figure(figsize=(15, 12))
        param_groups = defaultdict(list)
        for net_type in self.networks:
            base_type = net_type.split('_')[0]
            param_groups[base_type].append(net_type)
        row = 0
        for base_type, net_types in param_groups.items():
            for col, net_type in enumerate(net_types):
                g = self.networks[net_type]
                out_degrees = [d for n, d in g.out_degree()]
                plt.subplot(len(param_groups), len(net_types), row * len(net_types) + col + 1)
                plt.hist(out_degrees, bins=30, alpha=0.7)
                plt.title(f"{net_type.replace('_', ' ')}")
                plt.xlabel('Out-Degree')
                plt.ylabel('Frequency')
            row += 1
        plt.tight_layout()
        plt.show()
        plt.figure(figsize=(10, 8))
        for net_type, g in self.networks.items():
            out_degrees = [d for n, d in g.out_degree()]
            if not out_degrees:
                continue
            degree_counts = np.bincount(out_degrees)
            degrees_unique = np.arange(len(degree_counts))
            probs = degree_counts / sum(degree_counts)
            non_zero_mask = probs > 0
            degrees_unique = degrees_unique[non_zero_mask]
            probs = probs[non_zero_mask]
            if len(degrees_unique) > 0 and len(probs) > 0:
                plt.loglog(degrees_unique, probs, 'o', markersize=4, label=net_type.replace('_', ' '))
        plt.xlabel('Out-Degree (log scale)')
        plt.ylabel('Probability (log scale)')
        plt.title('Out-Degree Distributions in Log-Log Scale')
        plt.legend(ncol=2)
        plt.grid(True, which='both', ls='-', alpha=0.2)
        plt.show()

    def plot_parameter_sensitivity(self):
        """绘制参数敏感性分析图"""
        if not self.biological_metrics:
            self.calculate_biological_metrics()
        metrics = self.biological_metrics
        if not metrics:
            print('没有生物学指标可供绘图')
            return
        net_data = defaultdict(list)
        for net_type, values in metrics.items():
            base_type = net_type.split('_')[0]
            params = {}
            if 'alpha' in net_type:
                match = re.search('alpha([\\d\\.]+)', net_type)
                if match:
                    params['alpha'] = float(match.group(1))
            if 'beta' in net_type:
                match = re.search('beta([\\d\\.]+)', net_type)
                if match:
                    params['beta'] = float(match.group(1))
            if 'sigma' in net_type:
                match = re.search('sigma([\\d\\.]+)', net_type)
                if match:
                    params['sigma'] = float(match.group(1))
            if 'lambda' in net_type:
                match = re.search('lambda([\\d\\.]+)', net_type)
                if match:
                    params['lambda'] = float(match.group(1))
            net_data[base_type].append((params, values))
        if not net_data:
            return
        metric_names = list(metrics.values())[0].keys()
        n_metrics = len(metric_names)
        plt.figure(figsize=(15, 5 * n_metrics))
        for metric_idx, metric_name in enumerate(metric_names):
            ax = plt.subplot(n_metrics, 1, metric_idx + 1)
            for base_type, data in net_data.items():
                param_values = []
                metric_values = []
                for params, metrics_dict in data:
                    if params:
                        param_val = list(params.values())[0]
                        param_values.append(param_val)
                        metric_values.append(metrics_dict[metric_name])
                    else:
                        param_values.append(0)
                        metric_values.append(metrics_dict[metric_name])
                sorted_indices = np.argsort(param_values)
                sorted_params = np.array(param_values)[sorted_indices]
                sorted_metrics = np.array(metric_values)[sorted_indices]
                ax.plot(sorted_params, sorted_metrics, 'o-', label=base_type)
            ax.set_title(f"{metric_name.replace('_', ' ').title()} Sensitivity")
            ax.set_xlabel('Parameter Value')
            ax.set_ylabel(metric_name)
            ax.grid(True, alpha=0.3)
            ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_distribution_fit(self):
        """绘制分布拟合度评估结果"""
        if not self.distribution_fit:
            print('没有分布拟合数据可供绘图')
            return
        ks_stats = []
        p_values = []
        mean_diffs = []
        labels = []
        for net_type, fit_data in self.distribution_fit.items():
            ks_stats.append(fit_data['ks_stat'])
            p_values.append(fit_data['p_value'])
            mean_diffs.append(abs(fit_data['mean'] - fit_data['target_mean']))
            labels.append(net_type)
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))
        axes[0].bar(labels, ks_stats, color='skyblue')
        axes[0].set_title('Kolmogorov-Smirnov Statistic (Lower is Better)')
        axes[0].set_ylabel('KS Statistic')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(axis='y', alpha=0.3)
        axes[1].bar(labels, p_values, color='lightgreen')
        axes[1].axhline(0.05, color='r', linestyle='--', label='Significance Threshold (0.05)')
        axes[1].set_title('P-value (Higher is Better)')
        axes[1].set_ylabel('P-value')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
        axes[2].bar(labels, mean_diffs, color='salmon')
        axes[2].set_title('Absolute Difference from Target Mean (Lower is Better)')
        axes[2].set_ylabel('|Mean - Target|')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_path_length_distributions(self):
        """绘制不同网络的路径长度分布并标注平均值"""
        if not self.results:
            print('没有导航结果可供绘图')
            return
        param_groups = defaultdict(list)
        for net_type in self.results:
            base_type = net_type.split('_')[0]
            param_groups[base_type].append(net_type)
        plt.figure(figsize=(15, 10))
        n_rows = len(param_groups)
        n_cols = max((len(group) for group in param_groups.values()))
        for row, (base_type, net_types) in enumerate(param_groups.items()):
            for col, net_type in enumerate(net_types):
                result = self.results[net_type]
                if not result['path_lengths']:
                    continue
                ax = plt.subplot(n_rows, n_cols, row * n_cols + col + 1)
                n, bins, patches = ax.hist(result['path_lengths'], bins=30, alpha=0.7)
                mean_val = np.mean(result['path_lengths'])
                ax.axvline(mean_val, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
                ax.set_title(f"{net_type.replace('_', ' ')}")
                ax.set_xlabel('Path Length')
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def run_experiment(self, p, experiment_params):
        """运行整个实验，返回所有样本数据"""
        self.results = {}
        self.networks = {}
        self.distribution_fit = {}
        all_samples_data = []
        n_nodes, avg_degree = experiment_params
        for dist_name, dist_info in self.distributions.items():
            param_list = dist_info['param_list']
            for param_idx, params in enumerate(param_list):
                param_str = '_'.join((f'{k}{v}' for k, v in params.items()))
                network_type = f'{dist_name}_{param_str}'
                dist_results = {'path_lengths': [], 'efficiency': [], 'success_count': 0, 'total_pairs': 0}
                for sample in range(self.n_samples):
                    degrees = self.generate_degrees(dist_name, **params)
                    G_undirected = self.create_undirected_graph(degrees)
                    G_directed = self.create_directed_graph(G_undirected, p)
                    sample_results, G = self.run_navigation_for_network(G_directed, network_type, sample)
                    success_rate = sample_results['success_rate']
                    if sample_results['path_lengths']:
                        avg_pl = np.mean(sample_results['path_lengths'])
                        path_length_cv = np.std(sample_results['path_lengths']) / avg_pl if avg_pl > 0 else 0
                    else:
                        avg_pl = 0.0
                        path_length_cv = 0.0
                    total_degrees = [d for n, d in G.degree()]
                    if total_degrees:
                        degree_counts = np.bincount(total_degrees)
                        degree_probs = degree_counts / sum(degree_counts)
                        degree_entropy = -np.sum(degree_probs * np.log2(degree_probs + 1e-10))
                    else:
                        degree_entropy = 0.0
                    sample_data = {'n_nodes': n_nodes, 'avg_degree': avg_degree, 'p': p, 'distribution_type': dist_name, 'params': param_str, 'full_network_type': network_type, 'sample_id': sample, 'success_rate': success_rate, 'avg_path_length': avg_pl, 'path_length_cv': path_length_cv, 'degree_entropy': degree_entropy, 'avg_degree': np.mean(degrees)}
                    all_samples_data.append(sample_data)
                    dist_results['path_lengths'].extend(sample_results['path_lengths'])
                    dist_results['efficiency'].extend(sample_results['efficiency'])
                    dist_results['success_count'] += sample_results['success_count']
                    dist_results['total_pairs'] += sample_results['total_pairs']
                dist_results['success_rate'] = dist_results['success_count'] / dist_results['total_pairs'] if dist_results['total_pairs'] > 0 else 0.0
                self.results[network_type] = dist_results
        self.calculate_biological_metrics()
        return {'biological_metrics': self.biological_metrics, 'distribution_fit': self.distribution_fit, 'all_samples_data': all_samples_data}

import multiprocessing
from functools import partial

def run_single_experiment(params):
    """运行单个参数组合的实验，返回结果DataFrame"""
    n_nodes, avg_degree, p = params
    base_filename = f"n{n_nodes}-ad{avg_degree}-p{int(p*10)}"
    result_dir = str(PROJECT_ROOT / "result" / "dynamics" / "navigation" / "raw_batches")
    os.makedirs(result_dir, exist_ok=True)
    
    pkl_filepath = os.path.join(result_dir, f"{base_filename}.pkl")
    csv_filepath = os.path.join(result_dir, f"{base_filename}.csv")
    
    # 如果文件已存在，直接读取并返回
    if os.path.exists(csv_filepath):
        print(f"文件已存在，直接读取: {base_filename}")
        return pd.read_csv(csv_filepath)
    
    experiment = NavigationExperiment(
        n_nodes=n_nodes,
        avg_degree=avg_degree,
        n_samples=10  # 10次重复实验
    )
    
    # 运行实验，传递参数信息
    results = experiment.run_experiment(p, (n_nodes, avg_degree))
    
    with open(pkl_filepath, "wb") as f:
        dill.dump(results["biological_metrics"], f)
    
    samples_df = pd.DataFrame(results["all_samples_data"])
    samples_df.to_csv(csv_filepath, index=False)
    
    print(f"结果已保存到: {pkl_filepath}")

    return samples_df

def run_parameter_sensitivity_parallel():
    """并行运行参数灵敏度分析实验并合并所有结果"""
    n_nodes_list = [300, 500, 1000, 2000, 3000, 5000, 8000, 10000]
    avg_degree_list = [10]
    p_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    param_combinations = [
        (n, ad, p)
        for n in n_nodes_list
        for ad in avg_degree_list
        for p in p_list
    ]
    
    # 创建进程池，使用最大31个进程
    with multiprocessing.Pool(processes=min(31, multiprocessing.cpu_count())) as pool:
        all_results = pool.map(run_single_experiment, param_combinations)
    
    combined_df = pd.concat(all_results, ignore_index=True)
    
    combined_csv_path = str(PROJECT_ROOT / "result" / "dynamics" / "navigation" / "navigation_simulation_dataset.csv")
    combined_df.to_csv(combined_csv_path, index=False)

    return combined_df

if __name__ == "__main__":
    final_df = run_parameter_sensitivity_parallel()
