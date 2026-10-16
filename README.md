# POWERLAW: connectome structure and network function

Research code and author-generated simulation observations for **Beyond power laws: broad functional advantages of log-normal brain networks**.

Yifei Zhang, Xiaoyu Zhang, Xiangrui Ma, Xi Chen, Aoyu Wang, Yuyan Zhang, Chaomin Ou, Gang Yan and Xin Lu.

The repository includes the five-family network generator, five functional models and a shared weighted GAM/bootstrap pipeline. It distributes only code and five simulation observation tables. Biological connectomes must be obtained separately. Statistical estimates, diagnostics, rankings and figures are generated locally rather than bundled.

## Install and run

Use Python 3.12. Run from the repository root:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python reproduce.py --help
python reproduce.py fit all
python reproduce.py figures
```

`fit all` fits 35 weighted GAMs at seven indices and writes 175 single-fit family predictions, diagnostics, ranks and metadata to `result/statistical_analysis/`. The default does not run a bootstrap. `figures` reads these newly generated files and draws five functional panels in PNG, PDF and SVG, plus correlation tables, under `result/figures/`.

Use the same bootstrap command for every function:

```bash
python reproduce.py fit all --bootstrap 1000 --seed 42
python reproduce.py figures
# Or process one function:
python reproduce.py fit hopfield --bootstrap 1000 --seed 42
python reproduce.py figures --analysis hopfield
```

Each function requires 7,000 bootstrap fits at the full settings, for 35,000 across all five functions. Each fit predicts all five families from the same resample. With bootstrap results present, figures show bootstrap means and percentile intervals; otherwise they show single-fit predictions without interval bands. Rankings state whether they use bootstrap means or single-fit predictions. A refit without bootstrap removes a previous interval file in that output directory.

Custom output locations:

```bash
python reproduce.py fit all --outdir result/custom_fits
python reproduce.py figures --fits-dir result/custom_fits --outdir result/custom_figures
```

Use `python reproduce.py COMMAND --help` for command-specific arguments. The `summary` and `thresholds` commands run fixed empirical workflows without additional options. No calculation overwrites the distributed observation tables.

## Files

| Path | Purpose |
|---|---|
| `reproduce.py` | Shared command entry point |
| `requirements.txt` | Pinned direct dependencies, including Brian2 |
| `code/src/connectome_dynamics/` | Graph construction, network generation, simulation orchestration, analysis and plotting |
| `code/src/connectome_dynamics/dynamics/` | The five functional models |
| `simulation.py` within the source package | Simulation grids and function-specific experimental settings, including Hopfield |
| `analysis.py` within the source package | Shared GAM, bootstrap, interval and ranking implementation for all five functions |
| `data/functional/` | Five compressed simulation observation tables with recorded generation identifiers |
| `LICENSE`, `data/LICENSE` | MIT code license and CC BY 4.0 simulation data license |

## Simulation observations

Read compressed tables directly with `pandas.read_csv('data/functional/stability.csv.gz')`. A row is a simulation condition, not an independent biological sample. The tables contain 224,400 rows in total. Formal input values and row order are retained.

| File | Rows | Response | Continuous GAM predictors |
|---|---:|---|---|
| `stability.csv.gz` | 13,200 | `Re_lambda_max` | `n_nodes`, `p`, `avg_degree` |
| `kuramoto.csv.gz` | 59,400 | `R_value` | `n_nodes`, `p`, `avg_degree`, `std`, `lambda` |
| `navigation.csv.gz` | 13,200 | `success_rate` | `n_nodes`, `p`, `avg_degree` |
| `stdp.csv.gz` | 19,800 | `min_ball_volume` | `n_nodes`, `p`, `initial_avg_degree`, `spatial_temporal_sum_factor` |
| `hopfield.csv.gz` | 118,800 | `mean_final_overlap` | `n_nodes`, `p`, `avg_degree`, `memory_load`, `noise_level` |

`distribution_type` identifies Exponential, Power_law, Truncated_Power_law, Log_normal or Stretched_Exponential. `n_nodes` is network size and `p` is the direction probability. The three degree-distribution parameter sets per family and the network construction procedure are in `null_networks.py`.

`params`/`distribution_parameters` identify the degree-distribution parameters. `sample_id`, `replicate` and `repeat_id` are within-grid repetition identifiers, not globally unique network IDs; combine them with size, direction probability, family, parameter set and other relevant conditions. Historical seeds that were not recorded are not retrospectively assigned.

- Stability retains `params`, `full_network_type` and `sample_id`.
- Kuramoto retains `std`, `lambda`, `n_steps`, and `terminal_R_window` (a JSON array of the final ten order parameters). The formal response and original trajectory diagnostics can differ slightly in decimal precision.
- Navigation retains recorded parameter/sample identifiers and navigation diagnostics. `avg_degree` is the supplied analysis covariate.
- STDP retains parameter/repetition identifiers and recorded activity and weight diagnostics. `initial_avg_degree` is the mean sampled target degree before graph construction. Missing exploratory PCA outputs do not invalidate otherwise complete formal analysis rows.
- Hopfield retains deterministic task IDs/seeds, network statistics and retrieval diagnostics. Each of 13,200 networks contributes nine selected conditions; this is not the complete 35-condition grid.

## Shared GAM and bootstrap

`reproduction.py` handles input preparation for all five functions. Continuous predictors are standardized on the complete input with `StandardScaler`, and family names use `LabelEncoder`. The split is 80:20 by row, with default seed 42. Required analysis columns must be complete.

The retained response-scale setting differs: stability, Kuramoto, navigation and STDP are standardized; Hopfield stays on its original overlap scale. A shared implementation uses this per-function setting; rearranging the code does not change the analysis units.

`analysis.py` fits `pygam.LinearGAM` with ten spline bases per continuous predictor and a categorical family term. For each weighting index τ in `[.01, .10, .25, .50, .75, .90, .99]`, weights are τ at or above the training-response quantile and 1−τ below it. This is quantile-indexed weighted squared error, not a quantile-loss fit.

The common bootstrap samples training rows with replacement, recalculates weights, refits and predicts all five families at zero standardized continuous covariates. It reports bootstrap means and the 2.5th/97.5th percentiles. Training/test splitting and resampling do not group conditions from a shared network. Rank 1 means a smaller prediction for stability and a larger one for the other functions. Family predictions are not pairwise causal effects.

The split and bootstrap each start from the specified seed. The bootstrap random stream proceeds through the indices in command-line order; changing that order or the number of draws changes later samples. Fresh seeded results are not a promise of exact historical bootstrap draws, particularly where original random seeds were not recorded. Numerical-library and BLAS environments can also affect floating-point values.

## Generate new simulations

```bash
python reproduce.py simulate stability
python reproduce.py simulate kuramoto
python reproduce.py simulate navigation
python reproduce.py simulate stdp
python reproduce.py simulate hopfield
python reproduce.py simulate hopfield --profile full --nodes 300 --probabilities .3 --replicates 1
```

All functions use `simulation.py` for the command, shared network grid, repeat counts, seed option and output directory, and use `null_networks.py` for graph construction. Model-specific parameters remain explicit in that common simulation entry point or the corresponding dynamics model. There is no separate Hopfield YAML, experiment driver or bootstrap script.

The default `smoke` profile uses 50 nodes, direction probability .3, the first parameter set per family and one replicate. STDP trials are shortened. These runs test execution, not the full experimental design. Brian2 may require a working C++ toolchain.

`--profile full` uses sizes `[300,500,1000,2000,3000,5000,8000,10000]`, direction probabilities 0.0–1.0 in increments of .1, and all three parameter sets per family. There are ten repeats for stability/navigation/Hopfield and five for Kuramoto/STDP. Full runs can require substantial time and memory. `--nodes`, `--probabilities`, `--replicates`, `--seed` and `--outdir` override the corresponding defaults. New-run seed defaults to 20260806; effective settings and row counts are recorded in run JSON files.

- Stability and navigation each produce 13,200 rows in the full grid.
- Kuramoto uses frequency standard deviations `[.1,.5,1.]`, couplings `[100,800,3000]` and `dt=.1`, producing 59,400 rows. Its retained stop is `R_t[-1] - R_t[-10] < .001` after ten values; this can stop on a decrease and does not prove convergence. Smoke runs have a 1,000-step cap; full runs have no additional cap.
- STDP uses factors `[10,15,20]`, each with a separate network realization, producing 19,800 rows. Smoke uses one training/testing trial per pattern, a 1 ms trial duration and a .1 ms interval. Execution of the standalone training phase and a learning effect have not been independently established by the release checks; finite response volumes alone do not establish learning.
- Hopfield evaluates loads `[.002,.005,.01,.02,.04]` and noises `[0,.05,.1,.15,.2,.3,.4]` on each of 13,200 networks, producing 462,000 rows. It selects loads `[.005,.01,.02]` and noises `[.05,.1,.2]` to produce 118,800 analysis rows. The complete grid is evaluated before selection to retain the random-number sequence. Connectivity is `symmetrized_mean`, Hebbian normalization uses node count, updates are synchronous with at most 15 steps, zero fields choose positive state, and at most three stored patterns are tested per load with one cue repeat. Each task retains its recorded seed derivation: the first four SHA-256 bytes, little-endian, of `base_seed|task_id`. The smoke profile also evaluates all 35 conditions per network.

For full Hopfield runs, `hopfield_full_grid.csv` and the selected `hopfield_analysis_input.csv` are separate generated files. Other functions export their results and run metadata to the chosen output directory. Generating new simulations does not replace the distributed observations automatically.

## Biological data availability and graph format


Obtain the biological connectomes from their original publications and follow the providers' access and reuse terms:

| Dataset | Original source |
|---|---|
| Adult male/hermaphrodite C. elegans | https://doi.org/10.1038/s41586-019-1352-7 |
| Developmental C. elegans | https://doi.org/10.1038/s41586-021-03778-8 |
| Platynereis larva | https://doi.org/10.7554/eLife.97964.2 |
| Drosophila larva | https://doi.org/10.1126/science.add9330 |
| Drosophila adult | https://doi.org/10.1038/s41586-024-07968-y |
| Ciona tadpole | https://doi.org/10.7554/eLife.16962 |

The analysis consumes a trusted Python pickle containing a `networkx.DiGraph`, with neurons as nodes, presynaptic → postsynaptic edges, and a numeric synapse-count `weight` attribute. A directed edge aggregates the weight for that ordered neuron pair. Pickle files should only be loaded from trusted sources. Put locally constructed graphs in `result/processed_connectomes/`.

The construction adapters expect the following normalized source tables. Download and any source-specific extraction into these formats are prerequisites; filenames here are local adapter paths, not promised provider filenames.

| Adapter | Required local input and conversion |
|---|---|
| `drosophila_larva` | `data/drosophila_larva/drosophila_larva_connectivity_matrix.csv`: first column contains row neuron IDs; other columns contain the same IDs in the same order. Rows are sources, columns targets; positive entries become weighted edges. Row/column IDs are compared as strings. |
| `platynereis_larva` | `data/platynereis_larva/platynereis_larva_connectivity_matrix.csv`: same source-row/target-column format. |
| `ciona_tadpole` | `data/ciona_tadpole/ciona_tadpole_connectivity_fractional.csv`: same adjacency format. Divide fractional weights by 0.06 and round to integers before adding positive edges. |
| `celegans_development` | `data/celegans_development/celegans_developmental_connectivity_matrices.xlsx`: skip the first sheet; subsequent sheets are stages 01–08. First column contains target neuron IDs; column headers contain source IDs. Add positive source-to-target edges. |
| `celegans_male`, `celegans_hermaphrodite` | `data/celegans_{sex}/celegans_{sex}_chemical_nodes.csv` with `index,name`, and `celegans_{sex}_chemical_edges.csv` with `source,target,connectivity`. Merge duplicate neuron names and sum weights for duplicate ordered edges. |
| Adult fly (manual preparation) | Convert original chemical synapse records to the directed weighted graph above and save `result/processed_connectomes/drosophila_adult_connectome.pkl`. No automatic adult-fly source adapter is supplied. |

The graph filenames emitted by the adapters are `drosophila_larva_connectome.pkl`, `platynereis_larva_connectome.pkl`, `ciona_tadpole_connectome.pkl`, `celegans_development_stage_01_connectome.pkl` through `..._08_connectome.pkl`, and `celegans_{sex}_chemical_connectome.pkl`. Obtain and construct all 14 graphs before running the complete empirical summary or threshold workflow.

```bash
python reproduce.py graphs drosophila_larva
python reproduce.py summary
python reproduce.py degree-fits --degree-mode total
python reproduce.py thresholds
```

Degree fitting excludes nonpositive degrees, uses discrete `powerlaw.Fit` with automatic `xmin`, and fits power law, exponential, truncated power law, log-normal and stretched exponential families. The retained KS p calculation is `scipy.stats.ksone.sf(D, n_tail)`. Threshold analysis retains edges with weights at least the threshold and refits. All empirical summaries and workbooks are calculated from the locally prepared graphs and written under `result/empirical_analysis/`.

After running `summary`, optionally include the freshly computed size relationship:

```bash
python reproduce.py figures --empirical-summary result/empirical_analysis/dataset_summary/connectome_dataset_summary.csv
```

The size relationship is a log-log regression of edges on nodes. The plotting command produces functional panels and, when explicitly supplied, the empirical size panel. It does not recreate every manuscript schematic or degree-distribution plot. Successful installation and small checks do not establish full regeneration of every manuscript result. Full simulation grids and all five 1,000-draw analyses have not all been rerun for this release.

## Citation and licenses

Please cite the associated paper by title and authors above and each original biological data source used. Code is MIT licensed; author-generated simulation observations are CC BY 4.0 as specified in `data/LICENSE`. Third-party biological data remain subject to their original terms.
