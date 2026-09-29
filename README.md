# Machine Learning Meta-Learning Study — Soft Materials (Granular Hydrogels)

This repository contains the data, processing pipeline, and analysis notebooks for the study:

> Practical insights for applying machine learning to soft materials science: an empirical study of model optimization and evaluation on granular hydrogel datasets\*\*
> C. A. Verheyen, J. A. Lewis, E. T. Roche

It is the analysis companion to a two-part submission to *Frontiers in Soft Matter* (Research Topic on data-driven modeling for soft matter). The companion FAIR2 data-resource article describes the underlying experimental datasets in full.

\---

## What this study is about

Machine learning is increasingly applied to soft matter, hydrogels, and biomaterials, where datasets are typically **small, heterogeneous, and experimentally expensive**. In this regime, common modeling heuristics carried over from large-data domains may not hold. Using a meta-learning dataset of **\~19.6 million trained model instances** spanning **31 prediction tasks** (14 classification, 17 regression) derived from granular-hydrogel experiments, this study measures how individual modeling-pipeline decisions — algorithm choice, hyperparameter tuning, model selection strategy, cross-validation protocol, and data handling — actually affect reported performance.

Rather than trying to find the single best model for any one task, the study is designed to **characterize the modeling process itself**. Key findings include: tree-ensemble methods are robust default choices while several widely used algorithms are fragile; extensive search does not reliably outperform simpler selection on held-out data; single (non-nested) cross-validation systematically overstates performance, more so for regression and for larger search spaces; and regression is consistently less stable than classification. These are framed as empirical demonstrations within this granular-hydrogel task suite rather than as universal rules.

\---

## How the analysis works

All processed data is **rebuilt from the raw datasets at runtime** — the repository ships the raw meta-learning datasets and the processing code, and each notebook regenerates the processed and aggregated dataframes from scratch. There are no pre-computed intermediate files required to run the notebooks.

* **`utils.py`** — the shared processing module. Its central function, `process\_all\_tasks()`, loads the raw per-task datasets and produces the full set of processed and aggregated dataframes used by every figure notebook (per-task inner/outer split and aggregate frames, and the merged frames used for the nested-vs-single and search-strategy analyses).Note: *correct\_source\_data.py* is a one-time source-data correction (documented for provenance; the committed raw data is already corrected).
* **`data/`** — the raw meta-learning datasets (committed).
* **`notebooks/`** — one notebook per main-text figure; each imports `utils`, calls `process\_all\_tasks()`, and produces the corresponding figure.
* **`figures/`** — exported versions of each main-text figure produced via the figure notebooks.
* **`supplementary/`** — saved tables and figures from the supplemental analyses (statistical tests, metric-dependence, seed-sensitivity by algorithm, per-task metric summaries, regression baselines, solver comparison).

### Running the notebooks

1. Ensure the dependencies below are installed.
2. Open any figure notebook and run all cells. The first cell builds all processed data from the raw datasets via `process\_all\_tasks()`.

   * **Note:** this build takes roughly **15–20 minutes**, as it reconstructs the full processed dataset from the raw model-instance data. This is intentional — the notebooks reproduce every result from raw data with no hidden pre-computed state.
3. Each notebook then produces its figure (and, where applicable, saves outputs).

### Dependencies

Python 3.x with:

* `scikit-learn` 
* `pandas`, `numpy`, `scipy`
* `matplotlib`, `seaborn`
* `statsmodels` 
* `xgboost` 

\---

## Supplemental notebooks

Two supplemental notebooks accompany the main analysis:

* **Supplemental analyses** (post-hoc, on the processed frames): formal statistical tests (Wilcoxon signed-rank with Benjamini-Hochberg correction), metric-dependence of model selection (MAE vs. median absolute error), seed-sensitivity decomposed by algorithm, per-task summaries across all evaluation metrics, and per-task regression trivial baselines. This notebook builds from the raw datasets like the main notebooks.
* **Model re-runs** (targeted comparison on the base experimental data): compares SGD-based linear models against deterministic solvers (lbfgs LogisticRegression, Ridge, Huber) and a modern gradient-boosting implementation (XGBoost), to isolate optimizer-specific instability and characterize modern-method robustness.

> *Note on base data: the model-re-run notebook operates on the base experimental datasets (raw features and targets per task), which are the source data for the companion FAIR2 data-resource article and are not duplicated in this repository. The code and all resulting outputs are provided here; regenerating those specific results requires the base datasets (available via the companion article). All other notebooks are fully reproducible from the data committed to this repository.*

\---

## Citation

If you use this code or data, please cite the study above (and the companion FAIR2 data-resource article for the underlying experimental datasets). Full citation details will be finalized upon publication.

