"""Read and validate the supplied analysis data."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from .paths import PROJECT_ROOT

QUANTILES = [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
ANALYSES = {
    'stability': {'response': 'Re_lambda_max', 'continuous': ['n_nodes', 'p', 'avg_degree'], 'standardize_response': True},
    'kuramoto': {'response': 'R_value', 'continuous': ['n_nodes', 'p', 'avg_degree', 'std', 'lambda'], 'standardize_response': True},
    'navigation': {'response': 'success_rate', 'continuous': ['n_nodes', 'p', 'avg_degree'], 'standardize_response': True},
    'stdp': {'response': 'min_ball_volume', 'continuous': ['n_nodes', 'p', 'initial_avg_degree', 'spatial_temporal_sum_factor'], 'standardize_response': True},
    'hopfield': {'response': 'mean_final_overlap', 'continuous': ['n_nodes', 'p', 'avg_degree', 'memory_load', 'noise_level'], 'standardize_response': False},
}


def prepare_analysis(name: str, input_path: Path | None = None):
    config = ANALYSES[name]
    path = input_path or PROJECT_ROOT / 'data' / 'functional' / f'{name}.csv.gz'
    columns = config['continuous'] + ['distribution_type', config['response']]
    full = pd.read_csv(path)
    missing = set(columns) - set(full.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    frame = full[columns].dropna().copy()
    if frame.empty:
        raise ValueError(f'No complete analysis rows in {path}')
    X_cont = StandardScaler().fit_transform(frame[config['continuous']])
    encoder = LabelEncoder()
    codes = encoder.fit_transform(frame['distribution_type'])
    X = np.column_stack([X_cont, codes])
    y = frame[config['response']].to_numpy(dtype=float)
    y_scaler = StandardScaler().fit(y[:, None])
    if config['standardize_response']:
        y = y_scaler.transform(y[:, None]).ravel()
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError('Analysis inputs contain infinite values')
    metadata = {'analysis': name, 'rows': len(frame), 'dropped_rows': len(full) - len(frame),
                'response': config['response'], 'continuous': config['continuous'],
                'standardize_response': config['standardize_response'],
                'response_mean': float(y_scaler.mean_[0]), 'response_scale': float(y_scaler.scale_[0])}
    return X, y, list(encoder.classes_), metadata


def select_hopfield_conditions(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame.memory_load.isin([.005, .01, .02]) & frame.noise_level.isin([.05, .1, .2])].copy()

