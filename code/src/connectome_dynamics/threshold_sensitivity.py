# -*- coding: utf-8 -*-
# Source notebook: 05_连接阈值敏感性分析/假阳性连接.ipynb
# Cell order is preserved; Markdown cells are retained as comments.

import networkx as nx
import numpy as np
import pandas as pd
import powerlaw
from scipy.stats import ksone
import matplotlib.pyplot as plt
import seaborn as sns
import dill

DEGREE_MODE = 'total'

import warnings

warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = ["SimHei"]

pd.set_option("max_colwidth", None)

def load_network(file_path):
    with open(file_path, "rb") as file:
        G = dill.load(file)
    return G

# 网络预处理：根据权重阈值筛选边
def preprocess_network(G, weight_threshold):
    """
    预处理网络，保留权重大于等于阈值的边
    
    参数:
    G -- 原始网络
    weight_threshold -- 权重阈值
    
    返回:
    new_G -- 预处理后的网络
    """
    new_G = G.copy()
    edges_to_remove = [(u, v) for u, v, data in G.edges(data=True) if data.get('weight', 1) < weight_threshold]
    new_G.remove_edges_from(edges_to_remove)
    return new_G

def get_degree_distribution(G):
    """获取网络的度数分布"""
    degrees = [d for _, d in G.degree()]
    return degrees

def get_out_degree_sequence(G):
    """获取网络的出度序列"""
    out_degrees = [d for _, d in G.out_degree()]
    return out_degrees

def get_in_degree_sequence(G):
    """获取网络的入度序列"""
    in_degrees = [d for _, d in G.in_degree()]
    return in_degrees

def fit_power_law(data):
    """拟合幂律分布并搜索最优kmin"""
    fit = powerlaw.Fit(data, xmin=None, discrete=True)
    kmin = fit.xmin
    alpha = fit.power_law.alpha
    return fit, kmin, alpha

def calculate_ks_pvalues(fit, data):
    """计算五种分布的KS检验p值"""
    pvalues = {}
    distributions = ['power_law', 'exponential', 'truncated_power_law', 'lognormal', 'stretched_exponential']
    
    for dist in distributions:
        try:
            if dist == 'power_law':
                ks_stat = fit.power_law.KS()
            elif dist == 'exponential':
                ks_stat = fit.exponential.KS()
            elif dist == 'truncated_power_law':
                ks_stat = fit.truncated_power_law.KS()
            elif dist == 'lognormal':
                ks_stat = fit.lognormal.KS()
            elif dist == 'stretched_exponential':
                ks_stat = fit.stretched_exponential.KS()
            else:
                continue
                
            n = len(fit.data[fit.data >= fit.xmin])
            pvalue = ksone.sf(ks_stat, n)
            pvalues[dist] = pvalue
        except Exception as e:
            print(f"计算{dist}的KS p值时出错: {e}")
            pvalues[dist] = np.nan
    return pvalues

def calculate_aic_bic(fit, data, distribution, params, pvalues):
    """计算指定分布的AIC和BIC，若p值<0.1则返回-1"""
    if pvalues.get(distribution, 1) < 0.1:
        return -1, -1
    
    try:
        if distribution == 'power_law':
            loglikelihoods = fit.power_law.loglikelihoods(data)
        elif distribution == 'exponential':
            loglikelihoods = fit.exponential.loglikelihoods(data)
        elif distribution == 'truncated_power_law':
            loglikelihoods = fit.truncated_power_law.loglikelihoods(data)
        elif distribution == 'lognormal':
            loglikelihoods = fit.lognormal.loglikelihoods(data)
        elif distribution == 'stretched_exponential':
            loglikelihoods = fit.stretched_exponential.loglikelihoods(data)
        else:
            return None, None
        
        loglikelihood = np.sum(loglikelihoods)
        k = params[distribution]
        n = fit.n
        
        aic = 2 * k - 2 * loglikelihood
        bic = k * np.log(n) - 2 * loglikelihood
        
        return aic, bic
    except Exception as e:
        print(f"计算{distribution}的AIC/BIC时出错: {e}")
        return -1, -1

# 主函数：执行完整分析流程
def analyze_network_with_threshold(weight_threshold, G):
    """
    对指定权重阈值的网络进行完整分析
    
    参数:
    weight_threshold -- 权重阈值
    G -- 待分析网络
    
    返回:
    results -- 包含p值、AIC和BIC的结果字典
    """
    preprocessed_G = preprocess_network(G, weight_threshold)
    
    if preprocessed_G.number_of_edges() == 0:
        print(f"警告：权重阈值{weight_threshold}下网络无边，无法分析")
        return None, None, None
    degree_extractors = {
        'total': get_degree_distribution,
        'in': get_in_degree_sequence,
        'out': get_out_degree_sequence,
    }
    degrees = degree_extractors[DEGREE_MODE](preprocessed_G)
    
    if len(degrees) < 10:
        print(f"警告：权重阈值{weight_threshold}下节点数过少，无法分析")
        return None, None, None
    
    fit, kmin, alpha = fit_power_law(degrees)
    
    # 计算各分布的KS p值
    ks_pvalues = calculate_ks_pvalues(fit, degrees)
    
    distributions = {
        'power_law': 2,        # alpha, xmin
        'exponential': 1,      # lambda
        'truncated_power_law': 2, # alpha, truncation
        'lognormal': 2,        # mu, sigma
        'stretched_exponential': 2 # lambda, beta
    }
    
    results = {
        'ks_pvalues': ks_pvalues,
        'AIC': {},
        'BIC': {}
    }
    
    for dist, params in distributions.items():
        aic, bic = calculate_aic_bic(fit, degrees, dist, distributions, ks_pvalues)
        if aic is not None and bic is not None:
            results['AIC'][dist] = aic
            results['BIC'][dist] = bic
        
    return results, alpha, fit.n

def run_simulation(thresholds, file_path):
    """
    对多个权重阈值执行仿真实验
    
    参数:
    thresholds -- 权重阈值列表
    file_path -- 数据文件地址
    
    返回:
    all_results -- 包含所有阈值结果的DataFrame
    """
    G = load_network(file_path)
    all_results = []
    alphas = []
    ns = []
    
    for threshold in thresholds:
        print(f"正在分析权重阈值 >= {threshold} 的网络...")
        results, alpha, n = analyze_network_with_threshold(threshold, G)
        alphas.append(alpha)
        ns.append(n)
        
        if results is not None:         
            # 提取KS p值
            pvalue_series = pd.Series(results['ks_pvalues'], name=f'KS_pvalue_threshold_{threshold}')
            
            aic_series = pd.Series(results['AIC'], name=f'AIC_threshold_{threshold}')
            
            bic_series = pd.Series(results['BIC'], name=f'BIC_threshold_{threshold}')
            
            threshold_results = pd.concat([pvalue_series, aic_series, bic_series], axis=1)
            all_results.append(threshold_results)
    
    if all_results:
        final_df = pd.concat(all_results, axis=1)
        return final_df, alphas, ns
    else:
        return pd.DataFrame(), alphas, ns

def add_delta_columns(results_df):
    """
    添加ΔAIC和ΔBIC列，以幂律模型为基准，并保持列的顺序
    
    参数:
    results_df -- 包含AIC和BIC的结果DataFrame
    
    返回:
    results_df -- 添加了差值列的DataFrame
    """
    thresholds = set()
    for col in results_df.columns:
        if 'AIC' in col or 'BIC' in col:
            threshold = int(col.split('_')[-1])
            thresholds.add(threshold)
    
    # 按阈值顺序处理（确保与原始列顺序一致）
    for threshold in sorted(thresholds):
        aic_col = f"AIC_threshold_{threshold}"
        delta_aic_col = f"ΔAIC_threshold_{threshold}"
        
        if aic_col in results_df.columns:
            powerlaw_pval_col = f"KS_pvalue_threshold_{threshold}"
            powerlaw_pval = results_df.loc['power_law', powerlaw_pval_col] if powerlaw_pval_col in results_df.columns else None
            
            if powerlaw_pval >= 0.1:
                powerlaw_aic = results_df.loc['power_law', aic_col]
                delta_aic_values = results_df[aic_col] - powerlaw_aic
                for i, model in enumerate(results_df.index):
                    if model != 'power_law':
                        model_aic = results_df.loc[model, aic_col]
                        if model_aic == -1:
                            delta_aic_values[i] = -1
            else:
                delta_aic_values = [-1 for _ in results_df.index]

            aic_col_idx = results_df.columns.get_loc(aic_col)
            results_df.insert(aic_col_idx + 1, delta_aic_col, delta_aic_values)
        
        bic_col = f"BIC_threshold_{threshold}"
        delta_bic_col = f"ΔBIC_threshold_{threshold}"
        
        if bic_col in results_df.columns:
            powerlaw_pval_col = f"KS_pvalue_threshold_{threshold}"
            powerlaw_pval = results_df.loc['power_law', powerlaw_pval_col] if powerlaw_pval_col in results_df.columns else None
            
            if powerlaw_pval >= 0.1:
                powerlaw_bic = results_df.loc['power_law', bic_col]
                delta_bic_values = results_df[bic_col] - powerlaw_bic
                for i, model in enumerate(results_df.index):
                    if model != 'power_law':
                        model_bic = results_df.loc[model, bic_col]
                        if model_bic == -1:
                            delta_bic_values[i] = -1
            else:
                delta_bic_values = [-1 for _ in results_df.index]
            
            bic_col_idx = results_df.columns.get_loc(bic_col)
            results_df.insert(bic_col_idx + 1, delta_bic_col, delta_bic_values)
    
    return results_df

# 比较AIC和BIC，找出显著模型
def find_significant_models(results_df, reference_model='power_law'):
    """
    找出AIC和BIC显著优于参考模型的分布
    
    参数:
    results_df -- 包含AIC和BIC的结果DataFrame
    reference_model -- 参考模型（默认：幂律分布）
    
    返回:
    significant_models -- 显著模型的DataFrame
    """
    significant_models = {}
    
    for col in results_df.columns:
        if ('AIC' in col or 'BIC' in col) and ('Δ' not in col):
            threshold = int(col.split('_')[-1])
            model_type = 'AIC' if 'AIC' in col else 'BIC'
            
            if threshold not in significant_models:
                significant_models[threshold] = {
                    'AIC': [],
                    'BIC': [],
                    'AIC最优': None,
                    'BIC最优': None
                }
            
            powerlaw_pval_col = f"KS_pvalue_threshold_{threshold}"
            powerlaw_pval = results_df.loc[reference_model, powerlaw_pval_col] if powerlaw_pval_col in results_df.columns else None
                        
            if powerlaw_pval >= 0.1:
                flag = False
                
                # 幂律模型有效，使用传统Δ>2判断
                ref_col = f"{model_type}_threshold_{threshold}"
                ref_value = results_df.loc[reference_model, ref_col]

                # 存储所有模型的Delta值，用于找出最优模型
                delta_values = {}
                
                for model in results_df.index:
                    if model != reference_model:
                        model_value = results_df.loc[model, ref_col]
                        if model_value != -1:
                            delta = ref_value - model_value
                            delta_values[model] = delta
                            if delta > 2:
                                significant_models[threshold][model_type].append(model)
                            if delta > -2:
                                flag = True
                
                # 处理无显著模型的情况（幂律有效时输出"无显著模型"）
                if not significant_models[threshold][model_type]:
                    if flag:
                        significant_models[threshold][model_type] = ["无显著优于幂律模型"]
                    else:
                        significant_models[threshold][model_type] = ["幂律最优"]
                    
            else:
                # 幂律模型无效，检查其他模型p值
                pval_col = f"KS_pvalue_threshold_{threshold}"
                valid_models = []
                
                for model in results_df.index:
                    if model != reference_model:
                        model_pval = results_df.loc[model, pval_col] if pval_col in results_df.columns else 1
                        if model_pval >= 0.1:
                            valid_models.append(model)
                
                if valid_models:
                    significant_models[threshold][model_type] = valid_models
                else:
                    # 所有模型p值<0.1时输出"-1"
                    significant_models[threshold][model_type] = ["-1"]

            # 找出AIC/BIC最优的模型（与其他所有模型相比，Δ均>2）
            metric_col = f"{model_type}_threshold_{threshold}"
            all_models = results_df.index.tolist()
            optional_models = [m for m in all_models if results_df.loc[m, metric_col] != -1]
            
            best_model = None
            for model in optional_models:
                is_best = True
                model_value = results_df.loc[model, metric_col]
                
                for other_model in optional_models:
                    if other_model != model:
                        other_value = results_df.loc[other_model, metric_col]
                        delta = other_value - model_value  # 注意这里是其他模型值减当前模型值
                        if delta <= 2:
                            is_best = False
                            break
                
                if is_best:
                    best_model = model
                    break
            
            significant_models[threshold][f'{model_type}最优'] = best_model if best_model else '-1'                    
                    
    if significant_models:
        rows = []
        for threshold, models in significant_models.items():
            rows.append({
                '阈值': threshold,
                'AIC显著': ', '.join(models['AIC']) if models['AIC'] else '无',
                'BIC显著': ', '.join(models['BIC']) if models['BIC'] else '无',
                'AIC最优': models['AIC最优'],
                'BIC最优': models['BIC最优']
            })
        
        return pd.DataFrame(rows)
    else:
        return pd.DataFrame()

# # 批量阈值敏感性分析


from pathlib import Path
import sys

from .paths import RESULT_DIR
GRAPH_DIR = RESULT_DIR / "processed_connectomes"
OUTPUT_DIR = RESULT_DIR / "empirical_analysis" / "threshold_sensitivity"

FILE_LIST = [
    (GRAPH_DIR / "drosophila_larva_connectome.pkl", "drosophila_larva"),
    (GRAPH_DIR / "drosophila_adult_connectome.pkl", "drosophila_adult"),
    (GRAPH_DIR / "celegans_male_chemical_connectome.pkl", "celegans_male"),
    (GRAPH_DIR / "celegans_hermaphrodite_chemical_connectome.pkl", "celegans_hermaphrodite"),
    *[(GRAPH_DIR / f"celegans_development_stage_{i:02d}_connectome.pkl", f"celegans_stage_{i:02d}") for i in range(1, 9)],
    (GRAPH_DIR / "platynereis_larva_connectome.pkl", "platynereis_larva"),
    (GRAPH_DIR / "ciona_tadpole_connectome.pkl", "ciona_tadpole"),
]


def run_all_degree_modes() -> None:
    global DEGREE_MODE
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_files = {
        "total": OUTPUT_DIR / "threshold_sensitivity_total_degree.xlsx",
        "in": OUTPUT_DIR / "threshold_sensitivity_in_degree.xlsx",
        "out": OUTPUT_DIR / "threshold_sensitivity_out_degree.xlsx",
    }
    for mode, output_path in output_files.items():
        DEGREE_MODE = mode
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            for file_path, sheet_name in FILE_LIST:
                simulation_results, alphas, ns = run_simulation([0, 1, 2, 3, 4, 5], str(file_path))
                if simulation_results.empty:
                    continue
                simulation_results = add_delta_columns(simulation_results)
                significant_models = find_significant_models(simulation_results)
                significant_models["power_law_alpha"] = alphas
                significant_models["fitted_node_count"] = ns
                significant_models.set_index("阈值", inplace=True)
                simulation_results.to_excel(writer, sheet_name=sheet_name, startrow=0)
                significant_models.to_excel(writer, sheet_name=sheet_name, startrow=len(simulation_results) + 2)


if __name__ == "__main__":
    run_all_degree_modes()
