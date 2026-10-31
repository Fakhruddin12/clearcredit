> Historical v2 study. The [post-review challenge](../v3/README.md) adds matched-capacity baselines and history ablations, recommends a model for further validation, and corrects the interpretation of global-policy recall as a general early-warning conclusion. Original results remain preserved below.

# ClearCredit research extension

**Measured question:** Does modelling six months of repayment history improve behavioural default-risk ranking, and where does the model fail?

Read [RESEARCH_REVIEW.md](RESEARCH_REVIEW.md) for the literature review, professional modelling framework, code audit, experiment and limitations. This directory contains the second measured research study. The root CLI/API belongs to the earlier tabular workflow; the new study has its own commands and results below.

## Measured result

The development-selected three-seed GRU achieved **0.7772 AUC** and **0.5521 average precision** on the existing 6,000-account reporting sample. Ranking its highest-risk 20% captured **50.6% of defaults**, compared with **51.1% for LightGBM**. Its AUC advantage over LightGBM is small, with a paired interval including zero. The blend had the highest reporting AUC, but was not selected because its development result was lower. No reporting-driven retuning was performed.

The strongest finding is diagnostic: the GRU misses **85.2% of defaults in accounts without a latest status code of at least 1**, under the fixed development review policy. It relies heavily on existing distress. Missing the latest month's data changes **18.05% of review flags**. These findings matter more than claiming a neural-network breakthrough.

## Reproduce

Python 3.12 was used. A CPU install is sufficient.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python experiment.py --data data/source.xls --out results_reproduced
python -m pytest -q tests
python verify_results.py --data data/source.xls --results results_reproduced
```

If the workbook is absent, the experiment downloads it from UCI and requires the saved checksum. The full fitting/analysis run completed in approximately 96 seconds in the development environment; runtime and numerical reproducibility depend on hardware and package versions. The saved environment records the exact versions used. Summary metrics, protocols, tuning histories and figures are checked in; datasets, model weights and per-case predictions are generated locally and ignored by Git. The checked-in summary results include the subsequent correction to a **diagnostic prior-shift weighting error**; a regression check was added. Model fits and selection were unchanged.

The old reporting sample is already known from v1. Reproducing it does not produce a new untouched or prospective validation sample. Do not continue architecture selection against these reporting results.

## Files

- `experiment.py`: checks, partitioning, tabular and sequence models, development selection, calibration and diagnostics.
- `tests/test_risk_contracts.py`: seven numerical and data-handling contract checks.
- `verify_results.py`: restores the saved GRU ensemble and checks all 6,000 predictions and headline metric denominators without fitting again.
- `results/protocol.json`: protocol written before model fitting.
- `results/data_manifest.json`: source checksum, partition hashes and duplicate-pattern exclusions. Running the experiment also generates split_indices.npz locally.
- `results/tuning.json`: all development searches, neural training histories and seed results.
- `results/selection_locked_before_reporting.json`: selected model and fixed development thresholds.
- `results/results.json`: raw/calibrated results for every model, bootstrap intervals, deciles, subgroups and sensitivity checks.
- Generated `results_reproduced/reporting_predictions.csv`: per-case predictions; created by the experiment and excluded from Git.
- Generated `results_reproduced/models/`: preprocessing, calibrators and weights; created by the experiment and excluded from Git. Load only trusted locally generated joblib files.
- `results/*.png`: calibration/capture and risk-decile figures.
- `MODEL_CARD.md`: scope, limitations, appropriate interpretation and unresolved validation needs.
- `RESEARCH_REVIEW.md`: analysis and primary sources.

The original Home Credit SQL extension is not included or credited with these results. Source data is not bundled. This package is not approved or validated for actual lending, pricing, credit-limit or collections decisions.

