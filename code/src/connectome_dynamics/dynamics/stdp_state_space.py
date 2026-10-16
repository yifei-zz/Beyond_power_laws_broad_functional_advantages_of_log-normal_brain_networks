# -*- coding: utf-8 -*-

#

import numpy as np
import networkx as nx
import pandas as pd
import os
from collections import defaultdict
import scipy.sparse as sp
from scipy.spatial.distance import cosine
from scipy.stats import kstest
from scipy.integrate import quad
from brian2 import *
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from scipy.special import gamma

set_device('cpp_standalone', build_on_run=False)  # 使用C++ standalone模式

# ============== 网络生成类 ==============

from ..null_networks import DirectedNullNetworkExperiment
from ..paths import PROJECT_ROOT


# ============== 神经动力学仿真类 ==============
class Brian2PatternSTDPNetwork:
    """
    基于Brian2的STDP网络动力学仿真类
    """
    
    def __init__(self, G, params=None, spatial_temporal_sum_factor=15):
        """
        初始化Brian2 STDP网络
        
        参数:
            G: NetworkX有向图对象
            params: 仿真参数字典
        """
        self.G = G
        self.n_nodes = G.number_of_nodes()
        self.spatial_temporal_sum_factor = spatial_temporal_sum_factor  # 模拟n个突触前神经元同时激活的空间总和效应 或 单个神经元在短时间内发放n个脉冲的时间总和效应
        
        default_params = {
            'v_rest': -65.0 * mV,
            'v_reset': -70.0 * mV,
            'v_th': -50.0 * mV,
            'tau_m': 20.0 * ms,  # 膜时间常数（决定电位衰减速度）
            'tau_ref': 2.0 * ms,  # 不应期（发放后短暂无法再次发放）

            'Rm': 100.0 * Mohm,  # 膜电阻（典型值50-200 MΩ）
            'Cm': 0.2 * nF,  # 膜电容（典型值0.1-0.5 nF）（τ_m = Rm * Cm）
            
            'A_plus': 0.01,  # LTP（长时程增强）幅度
            'A_minus': 0.012,  # LTD（长时程抑制）幅度（A- > A+避免权重爆炸）
            'tau_plus': 20.0 * ms,  # 前突触迹衰减时间常数
            'tau_minus': 20.0 * ms,  # 后突触迹衰减时间常数
            'w_min': 0.01,
            'w_max': 1.0,
            
            'dt': 0.1 * ms,
            'trial_duration': 1000.0 * ms,  # 每次试次持续时间
            'intertrial_interval': 100.0 * ms,  # 试次间隔（仅背景噪声）
            'n_training_trials': 20,  # 训练试次数
            'n_testing_trials': 5,  # 测试试次数
            
            'pattern_fraction': 0.1,  # 模式神经元占比（10%）
            'n_patterns': 5,  # 输入模式数量
            'rate_pattern': 80.0 * Hz,  # 模式神经元发放率（高于背景）
            'rate_background': 10.0 * Hz,  # 背景噪声发放率

            'I_syn_scale': 50.0 * pA,  # 突触电流尺度
            'I_background_scale': 10.0 * pA,  # 背景电流尺度
            'I_pattern_scale': 400.0 * pA,  # 模式电流尺度
        }
        
        if params is not None:
            default_params.update(params)
        self.params = default_params
        
        defaultclock.dt = self.params['dt']
        
        # 设置模式神经元（输入刺激的目标神经元）
        self._setup_pattern_neurons()
        
        # 初始化网络（延迟到实际运行时，节省内存）
        self.network_initialized = False

        self.initial_weights_numpy = None
        
    def _setup_pattern_neurons(self):
        """设置模式神经元分组（模拟不同输入模式的目标神经元）"""
        n_pattern_neurons = int(self.n_nodes * self.params['pattern_fraction'])
        all_neurons = np.arange(self.n_nodes)
        self.pattern_neurons = np.random.choice(all_neurons, n_pattern_neurons, replace=False)
        
        # 将模式神经元划分为n_patterns组（每组对应一种输入模式）
        self.pattern_groups = []
        neurons_per_group = n_pattern_neurons // self.params['n_patterns']
        
        for i in range(self.params['n_patterns']):
            start_idx = i * neurons_per_group
            end_idx = (i + 1) * neurons_per_group
            self.pattern_groups.append(self.pattern_neurons[start_idx:end_idx])
    
    def _setup_brian2_network(self):
        """构建Brian2神经网络"""
        if self.network_initialized:
            return
            
        n = self.n_nodes
        
        # LIF神经元方程（unless refractory：不应期内不更新）
        eqs_neurons = '''
        dv/dt = (v_rest - v + Rm * (I_ext + I_syn)) / tau_m : volt (unless refractory)
        I_ext : amp
        I_syn : amp
        '''
        
        self.neurons = NeuronGroup(
            n, eqs_neurons,
            threshold='v > v_th',
            reset='v = v_reset',
            refractory=self.params['tau_ref'],
            method='euler',
            namespace={
                'v_rest': self.params['v_rest'],
                'v_reset': self.params['v_reset'],
                'v_th': self.params['v_th'],
                'tau_m': self.params['tau_m'],
                'Rm': self.params['Rm']
            }
        )
        
        self.neurons.v = self.params['v_rest']
        self.neurons.I_ext = 0 * pA
        self.neurons.I_syn = 0 * pA
        
        edges = list(self.G.edges())
        
        if len(edges) == 0:
            # 空图处理：创建空突触对象
            self.synapses = Synapses(self.neurons, self.neurons, '')
            self.initial_weights_numpy  = np.array([])
        else:
            sources, targets = zip(*edges)
            
            # STDP突触动力学方程（突触权重变化规则）
            eqs_synapses = '''
            w : 1
            dapre/dt = -apre / tau_plus : 1 (event-driven)
            dapost/dt = -apost / tau_minus : 1 (event-driven)
            '''
            
            # 突触前事件（LTP）（前神经元发放时）
            on_pre = '''
            I_syn_post += w * I_syn_scale
            apre += A_plus
            w = clip(w + apost, w_min, w_max)
            '''
            
            # 突触后事件（LTD）（后神经元发放时）
            on_post = '''
            apost += -A_minus
            w = clip(w + apre, w_min, w_max)
            '''
            
            self.synapses = Synapses(
                self.neurons, self.neurons,
                model=eqs_synapses,
                on_pre=on_pre,
                on_post=on_post,
                method='euler',
                namespace={
                    'A_plus': self.params['A_plus'],
                    'A_minus': self.params['A_minus'],
                    'tau_plus': self.params['tau_plus'],
                    'tau_minus': self.params['tau_minus'],
                    'w_min': self.params['w_min'],
                    'w_max': self.params['w_max'],
                    'I_syn_scale': self.params['I_syn_scale']
                }
            )
            
            # 连接突触（按网络的有向边）
            self.synapses.connect(i=list(sources), j=list(targets))
            
            # 初始化突触权重（在0.3w_max ~ 0.7w_max之间随机，避免极端值）
            w_init_low = 0.3 * self.params['w_max']
            w_init_high = 0.7 * self.params['w_max']

            # 保存初始权重到numpy数组，而不是从Brian2变量获取
            initial_weights = np.random.uniform(w_init_low, w_init_high, len(self.synapses))
            self.synapses.w = initial_weights
            self.initial_weights_numpy = initial_weights.copy()  # 保存初始权重
        
        # 创建监视器（记录神经元活动）
        self.spike_mon = SpikeMonitor(self.neurons)  # 记录所有神经元的发放时间
        self.state_mon = StateMonitor(self.neurons, 'v', record=False)  # 暂时不记录所有膜电位状态以节省内存
        
        self.network_initialized = True
    
    def _apply_pattern_stimulation(self, pattern_id):
        """应用模式刺激（外部输入）"""
        # 背景噪声：所有神经元接收泊松分布的随机输入（模拟大脑自发活动）
        background_input = np.random.poisson(
            # 泊松率 = 背景发放率 * 时间步长（转换为单位时间内的事件数）
            float(self.params['rate_background'] / Hz * defaultclock.dt / second),
            self.n_nodes
        )
        I_background = background_input * self.params['I_background_scale']
        self.neurons.I_ext = I_background
        
        # 模式刺激：若指定pattern_id，对应组的神经元接收更强输入
        if pattern_id is not None and pattern_id < len(self.pattern_groups):
            pattern_neurons = self.pattern_groups[pattern_id]
            pattern_input = np.random.poisson(
                # 模式输入：泊松分布，强度是背景的5倍（突出模式特异性）
                float(self.params['rate_pattern'] / Hz * defaultclock.dt / second),
                len(pattern_neurons)
            ) * self.spatial_temporal_sum_factor 
            I_pattern = pattern_input * self.params['I_pattern_scale']
            current_I_ext = self.neurons.I_ext[:]  # 获取当前电流值
            current_I_ext[pattern_neurons] += I_pattern
            self.neurons.I_ext = current_I_ext
    
    def run_training_phase(self):
        """运行训练阶段"""

        device.reinit()
        set_device('cpp_standalone', build_on_run=False, directory='../../data/03_stdp/output_train')  # 重新设置确保生效
        device.activate()
        defaultclock.dt = self.params['dt']
        
        # 初始化网络（创建神经元、突触、监视器）
        self._setup_brian2_network()
        
        trial_duration = self.params['trial_duration']  # 每个试次持续时间
        intertrial_interval = self.params['intertrial_interval']  # 试次间隔（100ms）
        n_trials = self.params['n_training_trials']
        
        current_state = {'trial_idx': 0, 'pattern_id': 0}
        
        # 使用NetworkOperation实现自定义刺激：实时更新刺激（每个时间步执行）
        @network_operation(dt=defaultclock.dt)
        def update_stimulation():
            current_time = defaultclock.t  # 当前仿真时间
            trial_total_time = trial_duration + intertrial_interval  # 一个试次+间隔的总时间
            trial_idx = int(current_time / trial_total_time)  # 当前试次索引
            
            if trial_idx >= n_trials:
                return
            
            current_state['trial_idx'] = trial_idx
            time_in_trial = current_time - trial_idx * trial_total_time  # 试次内的时间
            
            if time_in_trial < trial_duration:
                # 训练阶段（试次阶段）：施加模式刺激（循环5种模式）
                pattern_id = trial_idx % self.params['n_patterns']
                current_state['pattern_id'] = pattern_id
                self._apply_pattern_stimulation(pattern_id)
            else:
                # 间隔期：仅背景噪声
                self._apply_pattern_stimulation(None)
        
        # 构建仿真网络（包含神经元、突触、监视器、刺激更新操作）
        net_objects = [self.neurons, self.spike_mon, self.state_mon, update_stimulation]
        if len(self.synapses) > 0:
            net_objects.insert(1, self.synapses)
        
        net = Network(net_objects)  # 创建Brian2网络对象
        
        # 运行训练（总时间=试次数×（试次持续时间+间隔））
        total_time = n_trials * (trial_duration + intertrial_interval)

        net.run(total_time, report=None)  # 运行并输出进度

        if len(self.synapses) > 0:
            self.final_weights = self.synapses.w[:].copy()
        else:
            self.final_weights = np.array([])
        
    def run_testing_phase(self):
        """运行测试阶段（不更新权重，评估模式响应）"""

        device.reinit()
        set_device('runtime')  # 切换到runtime模式进行测试
        
        trial_duration = self.params['trial_duration']
        n_trials = self.params['n_testing_trials']
        n_patterns = self.params['n_patterns']
        n_nodes = self.n_nodes  # 网络节点数
        
        if hasattr(self, 'final_weights'):
            final_weights = self.final_weights
        else:
            final_weights = np.array([])
            
        # 保存网络结构信息（边列表）
        if hasattr(self, 'G') and self.G is not None:
            edges = list(self.G.edges())
            if len(edges) > 0:
                sources, targets = zip(*edges)
                sources = np.array(sources)
                targets = np.array(targets)
            else:
                sources = np.array([])
                targets = np.array([])
        else:
            sources = np.array([])
            targets = np.array([])
        
        pattern_responses = []  # 存储每种模式的平均响应（发放率向量）
        
        for pattern_id in range(n_patterns):
            pattern_firing_rates = []  # 存储该模式多次测试的发放率
            
            for trial in range(n_trials):
                defaultclock.dt = self.params['dt']

                test_neurons = NeuronGroup(
                    n_nodes, 
                    '''
                    dv/dt = (v_rest - v + Rm * (I_ext + I_syn)) / tau_m : volt (unless refractory)
                    I_ext : amp
                    I_syn : amp
                    ''',
                    threshold='v > v_th',
                    reset='v = v_reset',
                    refractory=self.params['tau_ref'],
                    method='euler',
                    namespace={
                        'v_rest': self.params['v_rest'],
                        'v_reset': self.params['v_reset'],
                        'v_th': self.params['v_th'],
                        'tau_m': self.params['tau_m'],
                        'Rm': self.params['Rm']
                    }
                )
                # 初始化神经元状态（每次试次重置，确保公平性）
                test_neurons.v = self.params['v_rest']
                test_neurons.I_ext = 0 * pA
                test_neurons.I_syn = 0 * pA
                
                test_spike_mon = SpikeMonitor(test_neurons)
                net_objects = [test_neurons, test_spike_mon]
                
                # 添加突触（固定训练后的权重，不更新）
                if len(final_weights) > 0 and len(sources) > 0:
                    test_synapses = Synapses(
                        test_neurons, test_neurons,
                        model='w : 1',  # 仅保留权重参数（无STDP更新）
                        on_pre='I_syn_post += w * I_syn_scale',  # 固定突触电流，仅传递电流，不更新权重
                        method='euler',
                        namespace={'I_syn_scale': self.params['I_syn_scale']}  
                    )
                    test_synapses.connect(i=sources, j=targets)
                    test_synapses.w = final_weights
                    net_objects.insert(1, test_synapses)
                
                # 定义当前trial的刺激逻辑（与训练一致）
                @network_operation(dt=defaultclock.dt)
                def update_test_stimulation():
                    background_input = np.random.poisson(
                        float(self.params['rate_background'] / Hz * defaultclock.dt / second),
                        n_nodes
                    )
                    I_background = background_input * self.params['I_background_scale']
                    test_neurons.I_ext = I_background
                    
                    # 模式刺激（仅当前pattern_id的组）
                    if pattern_id < len(self.pattern_groups):
                        pattern_neurons = self.pattern_groups[pattern_id]
                        pattern_input = np.random.poisson(
                            float(self.params['rate_pattern'] / Hz * defaultclock.dt / second),
                            len(pattern_neurons)
                        ) * self.spatial_temporal_sum_factor 
                        I_pattern = pattern_input * self.params['I_pattern_scale']
                        current_I_ext = test_neurons.I_ext[:]
                        current_I_ext[pattern_neurons] += I_pattern
                        test_neurons.I_ext = current_I_ext
                
                net_objects.append(update_test_stimulation)  # 添加刺激操作
                
                # 运行当前trial的测试网络（仅使用当前新建的神经元组）
                test_net = Network(net_objects)
                test_net.run(trial_duration, report=None)
                
                # 计算当前trial的发放率（每个神经元的发放次数/时间）
                spike_counts = np.zeros(n_nodes)
                spike_trains = test_spike_mon.spike_trains()  # 神经元发放时间列表
                for neuron_id in range(n_nodes):
                    spike_counts[neuron_id] = len(spike_trains[neuron_id])  # 发放次数
                
                # 发放率 = 发放次数 / 试次持续时间（单位：Hz）
                firing_rates = spike_counts / float(trial_duration / second)
                pattern_firing_rates.append(firing_rates)
                
                test_net.remove(update_test_stimulation)
                if len(final_weights) > 0 and len(sources) > 0:
                    test_net.remove(test_synapses)
                test_net.remove(test_neurons)
                test_net.remove(test_spike_mon)                

                # 清理当前trial的对象（加速内存释放）
                del test_net
            
            avg_response = np.mean(pattern_firing_rates, axis=0)
            pattern_responses.append(avg_response)

        device.reinit()
        set_device('cpp_standalone', build_on_run=False, directory='../../data/03_stdp/output_train')

        return pattern_responses
    
    def calculate_pattern_separation(self, pattern_responses):
        """计算模式分离度（衡量网络区分不同模式的能力）"""
        n_patterns = len(pattern_responses)
        
        if n_patterns < 2:  # 模式数不足，无法计算分离度
            return 0.0
        
        similarities = []
        for i in range(n_patterns):
            for j in range(i + 1, n_patterns):
                # 余弦相似度：越接近1，响应越相似（分离度差）
                norm_i = np.linalg.norm(pattern_responses[i])
                norm_j = np.linalg.norm(pattern_responses[j])
                if norm_i > 0 and norm_j > 0:
                    similarity = 1 - cosine(pattern_responses[i], pattern_responses[j])
                    similarities.append(similarity)
                else:
                    similarities.append(0.0)

        mean_similarity = np.mean(similarities) if similarities else 0.0
        pattern_separation = 1.0 - mean_similarity
        
        return max(0.0, min(1.0, pattern_separation))
    
    def calculate_weight_consistency(self):
        """计算权重变化一致性（用标准差衡量，值越小网络学习越稳定）"""
        if not hasattr(self, 'final_weights') or len(self.final_weights) == 0:
            return 0.0
            
        if self.initial_weights_numpy is None or len(self.initial_weights_numpy) == 0:
            return 0.0
        
        weight_changes = self.final_weights - self.initial_weights_numpy
        return float(np.std(weight_changes))
    
    def calculate_pca_convexhull_volume_and_min_enclosing_ball_volume(self, pattern_responses):
        """PCA降维+凸包体积（推荐优先使用）"""
        # 数据预处理：单位归一化（消除发放率量级影响）
        norm_responses = pattern_responses / np.maximum(np.linalg.norm(pattern_responses, axis=1, keepdims=True), 1e-10)
        
        # PCA降维：保留90%方差，且维度不超过模式数-1（避免无法构建凸包）
        n_patterns = norm_responses.shape[0]
        pca = PCA(n_components=min(0.9, n_patterns-1))
        reduced_responses = pca.fit_transform(norm_responses)

        # 计算并打印保留的信息比例（方差解释率之和）
        explained_variance = np.sum(pca.explained_variance_ratio_)

        try:
            hull = ConvexHull(reduced_responses)
            convex_hull_volume = hull.volume
        except:
            convex_hull_volume = 0.0  # 无法构建时返回0（如点共线/共面）
        
        center = np.mean(reduced_responses, axis=0)
        radius = np.max(np.linalg.norm(reduced_responses - center, axis=1))
        
        max_iter = 50
        tol = 1e-6
        prev_radius = np.inf
        for _ in range(max_iter):
            dists = np.linalg.norm(reduced_responses - center, axis=1)
            farthest_point = reduced_responses[np.argmax(dists)]
            center += 0.1 * (farthest_point - center)
            radius = np.max(np.linalg.norm(reduced_responses - center, axis=1))
            if abs(radius - prev_radius) < tol:
                break
            prev_radius = radius
        
        # 计算低维球体体积（pca_dim维）
        d = pca.n_components_
        if d == 0:
            min_ball_volume = 0.0
        else:
            pi_power = np.pi ** (d / 2)
            gamma_term = gamma(d / 2 + 1)
            min_ball_volume = (pi_power / gamma_term) * (radius ** d)
        return convex_hull_volume, min_ball_volume, explained_variance

    def calculate_cov_det_volume(self, pattern_responses):
        """协方差矩阵行列式（最快计算速度）"""
        # 数据预处理：单位归一化
        norm_responses = pattern_responses / np.maximum(np.linalg.norm(pattern_responses, axis=1, keepdims=True), 1e-10)
        responses_T = norm_responses.T  # 转置为(神经元数, 模式数)，适配协方差计算
        
        # 计算协方差矩阵（添加微小值避免奇异矩阵）
        cov_matrix = np.cov(responses_T, rowvar=False) + 1e-3 * np.eye(responses_T.shape[1])
        
        # 计算行列式（确保非负）
        det_value = np.linalg.det(cov_matrix)
        return max(det_value, 1e-20)
    
    def run_full_simulation(self):
        """运行完整仿真（训练→测试→计算指标）"""
        self.run_training_phase()
        
        if hasattr(self, 'spike_mon'):
            spike_trains = self.spike_mon.spike_trains()  # 训练阶段的发放记录
            total_time = float(self.params['n_training_trials'] * (
                self.params['trial_duration'] + self.params['intertrial_interval']
            ) / second)  # 训练总时间（秒）
            
            firing_rates = []
            for neuron_id in range(self.n_nodes):
                rate = len(spike_trains[neuron_id]) / total_time
                firing_rates.append(rate)
            
            mean_firing_rate = float(np.mean(firing_rates))  # 平均发放率
            # 发放率变异系数（CV）= 标准差/均值（衡量发放率的离散程度）
            firing_rate_cv = float(np.std(firing_rates) / (mean_firing_rate + 1e-10))
        else:
            mean_firing_rate = 0.0
            firing_rate_cv = 0.0

        pattern_responses = self.run_testing_phase()
        
        pattern_separation = self.calculate_pattern_separation(pattern_responses)  # 计算模式分离度
        weight_consistency = self.calculate_weight_consistency()  # 计算权重变化一致性
        
        pattern_responses_np = np.array(pattern_responses)
        pca_hull_volume, min_ball_volume, explained_variance = self.calculate_pca_convexhull_volume_and_min_enclosing_ball_volume(pattern_responses_np)
        cov_det_volume = self.calculate_cov_det_volume(pattern_responses_np)  

        results = {
            'pattern_separation': pattern_separation,  # 模式分离度
            'weight_consistency': weight_consistency,  # 权重变化一致性
            'mean_firing_rate': mean_firing_rate,  # 训练阶段平均发放率
            'firing_rate_cv': firing_rate_cv,  # 训练阶段发放率cv
            'pattern_responses': pattern_responses,  # 测试阶段每种模式的平均发放率
            'final_weights': self.final_weights if hasattr(self, 'final_weights') else np.array([]),
            'pca_hull_volume': pca_hull_volume,
            'pca_hull_volume_explained_variance': explained_variance,
            'cov_det_volume': cov_det_volume,
            'min_ball_volume': min_ball_volume
        }

        return results

# ============== 实验主类 ==============
class PatternSTDPExperiment:
    """
    模式STDP实验主类：整合网络生成和Brian2仿真
    """
    
    def __init__(self, base_experiment):
        self.base_exp = base_experiment  # 关联网络生成类实例
        self.stdp_results = {}  # 存储STDP仿真结果
        
    def run_pattern_stdp_analysis(self, n_nodes_list, p_list, spatial_temporal_sum_factor=15, n_samples=1, n_repeats=1):
        """对所有网络类型和参数组合运行模式STDP分析"""
        all_results = []
        
        for n_nodes in n_nodes_list:
            for p in p_list:
                print(f"\n{'='*60}")
                print(f"运行实验: n_nodes={n_nodes}, p={p}")
                print(f"{'='*60}")
                
                # 重新创建基础实验对象，确保所有属性正确初始化
                self.base_exp = DirectedNullNetworkExperiment(n_nodes=n_nodes, n_samples=n_samples)
                
                for dist_name, dist_info in self.base_exp.distributions.items():
                    param_list = dist_info["param_list"]
                    
                    for param_idx, params in enumerate(param_list):
                        param_str = "_".join(f"{k}{v}" for k, v in params.items())
                        network_type = f"{dist_name}_{param_str}"
                        
                        print(f"网络类型: {network_type}")
                        
                        for sample_id in range(n_samples):

                            degrees = self.base_exp.generate_degrees(dist_name, **params)
                            G_undirected = self.base_exp.create_undirected_graph(degrees)
                            G_directed = self.base_exp.create_directed_graph(G_undirected, p)
                            
                            for repeat_id in range(n_repeats):

                                brian2_sim = Brian2PatternSTDPNetwork(
                                    G_directed, 
                                    spatial_temporal_sum_factor=spatial_temporal_sum_factor
                                )
                                
                                metrics = brian2_sim.run_full_simulation()
                                
                                # 整理结果（保存每一次的完整结果）
                                result_entry = {
                                    'n_nodes': n_nodes,
                                    'p': p,
                                    'spatial_temporal_sum_factor': spatial_temporal_sum_factor,
                                    'distribution_type': dist_name,
                                    'params': param_str,
                                    'network_type': network_type,
                                    'sample_id': sample_id,  # 网络样本ID
                                    'repeat_id': repeat_id,  # 重复实验ID
                                    'combined_id': f"n{n_nodes}_p{p}_s{sample_id}_r{repeat_id}",  # 组合ID
                                    'initial_avg_degree': float(np.mean(degrees)),
                                    'pattern_separation': metrics['pattern_separation'],
                                    'weight_consistency': metrics['weight_consistency'],
                                    'mean_firing_rate': metrics['mean_firing_rate'],
                                    'firing_rate_cv': metrics['firing_rate_cv'],
                                    'pca_hull_volume': metrics['pca_hull_volume'],
                                    'pca_hull_volume_explained_variance': metrics['pca_hull_volume_explained_variance'],
                                    'cov_det_volume': metrics['cov_det_volume'],
                                    'min_ball_volume': metrics['min_ball_volume'],
                                    'pattern_responses_str': str(metrics['pattern_responses'])
                                }
                                
                                for pattern_idx, response in enumerate(metrics['pattern_responses']):
                                    result_entry[f'pattern_{pattern_idx}_avg_response'] = float(np.mean(response))
                                    result_entry[f'pattern_{pattern_idx}_std_response'] = float(np.std(response))
                                
                                all_results.append(result_entry)
                                
                                key = f"n{n_nodes}_p{p}_{network_type}_s{sample_id}_r{repeat_id}"
                                self.stdp_results[key] = metrics
                                
                                del brian2_sim
        
        return all_results
    
    def save_results(self, results, output_dir="../../data/03_stdp"):
        """保存实验结果为CSV文件"""
        os.makedirs(output_dir, exist_ok=True)
        
        summary_data = []
        for r in results:
            summary_data.append({
                'n_nodes': r['n_nodes'],
                'p': r['p'],
                'spatial_temporal_sum_factor': r['spatial_temporal_sum_factor'],
                'distribution_type': r['distribution_type'],
                'params': r['params'],
                'network_type': r['network_type'],
                'sample_id': r['sample_id'],
                'repeat_id': r['repeat_id'],
                'combined_id': r['combined_id'],
                'initial_avg_degree': r['initial_avg_degree'],
                'pattern_separation': r['pattern_separation'],
                'weight_consistency': r['weight_consistency'],
                'mean_firing_rate': r['mean_firing_rate'],
                'firing_rate_cv': r['firing_rate_cv'],
                'pca_hull_volume': r['pca_hull_volume'],
                'pca_hull_volume_explained_variance': r['pca_hull_volume_explained_variance'],
                'cov_det_volume': r['cov_det_volume'],
                'min_ball_volume': r['min_ball_volume']
            })
        
        df_detailed = pd.DataFrame(summary_data)
        detailed_path = os.path.join(output_dir, "brian2_pattern_stdp_detailed.csv")
        df_detailed.to_csv(detailed_path, index=False)
        print(f"\n详细结果已保存至: {detailed_path}")
        
        # 计算并保存统计汇总（按网络类型分组的均值和标准差）
        summary_stats = df_detailed.groupby(['n_nodes', 'p', 'spatial_temporal_sum_factor', 'distribution_type', 'network_type']).agg({
            'pattern_separation': ['mean', 'std', 'min', 'max'],
            'weight_consistency': ['mean', 'std', 'min', 'max'],
            'mean_firing_rate': ['mean', 'std', 'min', 'max'],
            'firing_rate_cv': ['mean', 'std', 'min', 'max'],
            'pca_hull_volume': ['mean', 'std', 'min', 'max'],
            'cov_det_volume': ['mean', 'std', 'min', 'max'],
            'min_ball_volume': ['mean', 'std', 'min', 'max']
        }).round(4)
        
        summary_stats.columns = [f"{col[0]}_{col[1]}" for col in summary_stats.columns]
        summary_stats.reset_index(inplace=True)
        
        stats_path = os.path.join(output_dir, "brian2_pattern_stdp_statistics.csv")
        summary_stats.to_csv(stats_path, index=False)
        print(f"统计汇总已保存至: {stats_path}")
        
        return df_detailed, summary_stats

def run_brian2_pattern_stdp_experiment(p_list, spatial_temporal_sum_factor, output_dir):
    """运行完整的Brian2模式STDP实验"""
    print("开始Brian2模式STDP网络动力学实验...")
    print("="*60)
    
    n_nodes_list = [10000, 8000, 5000, 3000, 2000, 1000, 500, 300]
    
    # 创建基础实验对象（初始节点数不重要，会在运行时更新）
    base_exp = DirectedNullNetworkExperiment(
        n_nodes=100,  # 这个值会被覆盖
        n_samples=5
    )
    
    pattern_stdp_exp = PatternSTDPExperiment(base_exp)
    
    # 运行分析，遍历所有参数组合
    results = pattern_stdp_exp.run_pattern_stdp_analysis(
        n_nodes_list=n_nodes_list,
        p_list=p_list,
        spatial_temporal_sum_factor=spatial_temporal_sum_factor,
        n_samples=5,  # 每个参数组合的网络样本数
        n_repeats=1   # 每个网络上的重复实验次数
    )
    
    df_detailed, df_stats = pattern_stdp_exp.save_results(results, output_dir=str(output_dir))
    
    print("\n" + "="*60)
    print("实验完成！关键结果汇总：")
    print("="*60)
    print(f"总实验次数: {len(results)}")
    print(f"参数组合: {len(n_nodes_list)}个n_nodes × {len(p_list)}个p = {len(n_nodes_list)*len(p_list)}种组合")

    print("\n各分布类型的平均指标（基于所有实验）：")
    dist_summary = df_detailed.groupby('distribution_type').agg({
        'pattern_separation': ['mean', 'std'],
        'weight_consistency': ['mean', 'std'],
        'mean_firing_rate': ['mean', 'std'],
        'pca_hull_volume': ['mean', 'std'],
        'cov_det_volume': ['mean', 'std'],
        'min_ball_volume': ['mean', 'std'],
    }).round(4)
    print(dist_summary)
    
    return df_detailed, df_stats




def parse_float_list(values):
    return [float(value) for value in values.split(",") if value.strip()]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the STDP state-space simulation in configurable batches.")
    parser.add_argument("--p-values", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--spatial-temporal-factors", default="10,15,20")
    args = parser.parse_args()

    project_root = PROJECT_ROOT
    p_values = parse_float_list(args.p_values)
    factors = parse_float_list(args.spatial_temporal_factors)
    for factor in factors:
        output_dir = project_root / "result" / "dynamics" / "stdp_state_space" / "raw_batches" / f"phi_{factor:g}"
        run_brian2_pattern_stdp_experiment(p_values, factor, output_dir)
