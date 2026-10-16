"""Run explicit, reproducible simulation grids without overwriting supplied inputs."""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
from .paths import PROJECT_ROOT
from .reproduction import select_hopfield_conditions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('function', choices=['stability', 'kuramoto', 'navigation', 'hopfield', 'stdp'])
    parser.add_argument('--profile', choices=['smoke', 'full'], default='smoke')
    parser.add_argument('--nodes', type=int, nargs='+')
    parser.add_argument('--probabilities', type=float, nargs='+')
    parser.add_argument('--replicates', type=int)
    parser.add_argument('--seed', type=int, default=20260806)
    parser.add_argument('--outdir', type=Path, default=PROJECT_ROOT / 'result/simulations')
    args = parser.parse_args()
    full = args.profile == 'full'
    nodes = args.nodes or ([300,500,1000,2000,3000,5000,8000,10000] if full else [50])
    probabilities = args.probabilities or (list(np.round(np.arange(11) / 10, 1)) if full else [.3])
    default_repeats = (5 if args.function in ['kuramoto', 'stdp'] else 10) if full else 1
    repeats = default_repeats if args.replicates is None else args.replicates
    if min(nodes) < 2 or repeats < 1 or any(p < 0 or p > 1 for p in probabilities):
        parser.error('nodes >= 2, replicates >= 1 and probabilities in [0,1] are required')
    args.outdir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed); random.seed(args.seed)
    rows = []
    if args.function == 'stdp':
        from .null_networks import DirectedNullNetworkExperiment
        from .dynamics.stdp_state_space import Brian2PatternSTDPNetwork
        import brian2 as brian
        brian.seed(args.seed)
        for n in nodes:
            if n < 50:
                parser.error('STDP requires at least 50 nodes for five pattern groups')
            for p in probabilities:
                generator = DirectedNullNetworkExperiment(n_nodes=n)
                for family, info in generator.distributions.items():
                    for index, params in enumerate(info['param_list'] if full else info['param_list'][:1]):
                        for replicate in range(repeats):
                            for factor in [10,15,20] if full else [10]:
                                degrees = generator.generate_degrees(family, **params)
                                graph = generator.create_directed_graph(
                                    generator.create_undirected_graph(degrees), p)
                                settings = None if full else {'trial_duration': 1*brian.ms,
                                    'intertrial_interval': .1*brian.ms, 'n_training_trials': 1, 'n_testing_trials': 1}
                                output = Brian2PatternSTDPNetwork(graph, settings, factor).run_full_simulation()
                                rows.append({'n_nodes': n, 'p': p, 'distribution_type': family,
                                    'parameter_index': index, 'replicate': replicate,
                                    'spatial_temporal_sum_factor': factor,
                                    'initial_avg_degree': float(np.mean(degrees)),
                                    **{k: float(v) for k,v in output.items() if np.isscalar(v)}})
    else:
        from .dynamics.navigation import NavigationExperiment
        from .dynamics.spectral_abscissa import SpectralAbscissaExperiment
        from .dynamics.kuramoto import KuramotoExperiment
        from .dynamics.hopfield import HopfieldExperiment, HopfieldSettings
        cls = {'navigation': NavigationExperiment, 'stability': SpectralAbscissaExperiment,
               'kuramoto': KuramotoExperiment, 'hopfield': HopfieldExperiment}[args.function]
        for n in nodes:
            for p in probabilities:
                experiment = cls(n_nodes=n, n_samples=repeats)
                if not full:
                    experiment.distributions = {k: {'param_list': v['param_list'][:1]} for k,v in experiment.distributions.items()}
                if args.function == 'hopfield':
                    rows.extend(experiment.run_experiment(
                        p, base_seed=args.seed,
                        load_values=[.002, .005, .01, .02, .04],
                        noise_values=[0., .05, .1, .15, .2, .3, .4],
                        settings=HopfieldSettings(
                            connectivity_mode='symmetrized_mean', hebbian_normalization='n_nodes',
                            zero_field_policy='positive', max_update_steps=15, convergence_patience=1,
                            max_test_patterns=3, cue_repeats=1,
                            success_overlap_threshold=.95, success_rate_threshold=.80)))
                elif args.function == 'kuramoto':
                    experiment.max_steps = None if full else 1000
                    for std in [0.1, .5, 1.] if full else [.1]:
                        rows.extend(experiment.run_experiment(p, n, [100,800,3000] if full else [100], std, dt=.1))
                elif args.function == 'navigation':
                    rows.extend(experiment.run_experiment(p, (n, experiment.avg_degree))['all_samples_data'])
                else:
                    rows.extend(experiment.run_experiment(p, n))
    if args.function == 'hopfield':
        frame = pd.DataFrame(rows)
        frame.to_csv(args.outdir / 'hopfield_full_grid.csv', index=False)
        select_hopfield_conditions(frame).to_csv(args.outdir / 'hopfield_analysis_input.csv', index=False)
    else:
        pd.DataFrame(rows).to_csv(args.outdir / f'{args.function}_results.csv', index=False)
    metadata = {'function': args.function, 'profile': args.profile, 'nodes': nodes,
                'probabilities': probabilities, 'replicates': repeats, 'seed': args.seed, 'rows': len(rows)}
    if args.function == 'kuramoto':
        metadata['max_steps_per_simulation'] = None if full else 1000
    (args.outdir / f'{args.function}_run.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata))


if __name__ == '__main__':
    main()
