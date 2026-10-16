"""Fit quantile-indexed weighted GAMs and export family predictions."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pygam import LinearGAM, f, s
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from .paths import PROJECT_ROOT
from .reproduction import ANALYSES, QUANTILES, prepare_analysis


def make_terms(count):
    terms = s(0, n_splines=10)
    for index in range(1, count):
        terms += s(index, n_splines=10)
    return terms + f(count)


def fit_analysis(name, output_dir, quantiles=QUANTILES, bootstrap=0, seed=42):
    X, y, names, metadata = prepare_analysis(name)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=seed)
    point = np.zeros((len(names), X.shape[1])); point[:, -1] = np.arange(len(names))
    count = X.shape[1] - 1
    rng = np.random.default_rng(seed)
    metrics, effects, intervals = [], [], []
    for tau in quantiles:
        weights = np.where(ytr >= np.quantile(ytr, tau), tau, 1 - tau)
        model = LinearGAM(make_terms(count)).fit(Xtr, ytr, weights=weights)
        predicted = model.predict(Xte)
        metrics.append({'quantile': tau, 'r2': r2_score(yte, predicted), 'mse': mean_squared_error(yte, predicted)})
        effects.extend({'quantile': tau, 'distribution_type': family, 'effect': float(value)}
                       for family, value in zip(names, model.predict(point)))
        samples = []
        for _ in range(bootstrap):
            indices = rng.integers(0, len(ytr), len(ytr))
            xb, yb = Xtr[indices], ytr[indices]
            wb = np.where(yb >= np.quantile(yb, tau), tau, 1 - tau)
            fitted = LinearGAM(make_terms(count)).fit(xb, yb, weights=wb)
            samples.append(fitted.predict(point))
        if samples:
            samples = np.asarray(samples)
            for code, family in enumerate(names):
                intervals.append({'quantile': tau, 'distribution_type': family,
                                  'mean_effect': float(samples[:, code].mean()),
                                  'ci_lower': float(np.quantile(samples[:, code], .025)),
                                  'ci_upper': float(np.quantile(samples[:, code], .975))})
        print(f'{name}: tau={tau:g} fitted', flush=True)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(output_dir / 'model_performance.csv', index=False)
    pd.DataFrame(effects).to_csv(output_dir / 'distribution_effects.csv', index=False)
    if intervals:
        pd.DataFrame(intervals).to_csv(output_dir / 'distribution_effect_confidence_intervals.csv', index=False)
    else:
        (output_dir / 'distribution_effect_confidence_intervals.csv').unlink(missing_ok=True)
    ranks = pd.DataFrame(intervals if intervals else effects)
    value_column = 'mean_effect' if intervals else 'effect'
    ranks['rank'] = ranks.groupby('quantile')[value_column].rank(ascending=name == 'stability', method='min')
    ranks['ranking_basis'] = 'bootstrap_mean' if intervals else 'single_fit'
    ranks[['quantile', 'distribution_type', 'rank', 'ranking_basis']].to_csv(output_dir / 'family_ranks.csv', index=False)
    metadata.update({'split_seed': seed, 'bootstrap_seed': seed, 'bootstrap_iterations': bootstrap, 'quantiles': list(quantiles)})
    (output_dir / 'analysis_metadata.json').write_text(json.dumps(metadata, indent=2))
    return {'metrics': metrics, 'effects': effects, 'metadata': metadata}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('analysis', choices=[*ANALYSES, 'all'])
    parser.add_argument('--bootstrap', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quantiles', type=float, nargs='+', default=QUANTILES)
    parser.add_argument('--outdir', type=Path, default=PROJECT_ROOT / 'result/statistical_analysis')
    args = parser.parse_args()
    if args.bootstrap < 0 or any(not 0 < q < 1 for q in args.quantiles):
        parser.error('bootstrap must be nonnegative and quantiles must lie strictly between 0 and 1')
    for name in ANALYSES if args.analysis == 'all' else [args.analysis]:
        fit_analysis(name, args.outdir / name, args.quantiles, args.bootstrap, args.seed)


if __name__ == '__main__':
    main()
