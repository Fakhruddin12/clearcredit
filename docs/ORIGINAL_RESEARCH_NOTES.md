# ClearCredit: credit-risk model comparison and decision support

ClearCredit is an end-to-end research project for comparing credit-default models and examining how probability calibration and decision thresholds affect a review workflow. It includes a measured benchmark on the UCI Default of Credit Card Clients dataset and a separate DuckDB/SQL feature-engineering extension for Home Credit data.

The project asks two practical questions: **does a boosted-tree model improve risk ranking over a logistic-regression baseline, and how do calibration and a stated error-cost scenario change the resulting decisions?** The work is intended for reproducible analysis and model-risk discussion. It is not a lending product and is not validated for real credit decisions.

## Results at a glance: UCI benchmark

The benchmark uses **30,000 historical cases from Taiwan**, with 23 input features and next-month default as the target. Model selection used five-fold stratified cross-validation on the development split. A separate stratified 20% holdout was used once for final evaluation.

| Model | Mean 5-fold PR-AUC | Fold SD | Mean 5-fold ROC-AUC | Mean Brier score |
|---|---:|---:|---:|---:|
| Logistic regression | 0.504 | 0.014 | 0.726 | 0.206 |
| **LightGBM (selected)** | **0.555** | **0.009** | **0.780** | **0.171** |

On the locked holdout, the selected calibrated LightGBM achieved **0.547 PR-AUC**, **0.774 ROC-AUC**, and a **0.136 Brier score**. Its calibration slope was 0.999. A decision threshold was selected using out-of-fold development predictions under an explicitly assumed false-approval:false-decline cost ratio of 5:1; at that threshold, holdout precision was 0.371 and recall was 0.744.

The cost ratio is a scenario assumption, not a measured lender cost. These historical benchmark scores do not establish business savings or likely performance in a current UK, European, UAE, Singaporean, or Hong Kong applicant population. Detailed metrics and diagnostics are included under [`reports/uci/`](../reports/uci/).

## Dataset and reproducibility

- **Source:** UCI Machine Learning Repository, [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).
- **Population:** 30,000 historical Taiwan credit-card clients; target is default payment next month.
- **Data handling:** the preparation command downloads the public workbook, validates the schema and target, records a checksum, and creates a processed feature table.
- **Split:** 80% development and 20% stratified holdout. Five-fold stratified cross-validation is applied within development data.
- **Leakage control:** imputation and transformations are fit within each fold. The holdout is reserved for final reporting.

Raw and processed data are not included in this repository. Running the preparation step downloads the public UCI file into the ignored `data/` directory.

## Quick start

Python 3.10 or newer is required. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,api]"
python -m clearcredit.cli prepare --config configs/uci.yaml
python -m clearcredit.cli train --config configs/uci.yaml
python -m clearcredit.cli report --config configs/uci.yaml
```

The commands download and prepare the dataset, compare logistic regression and LightGBM, save the fitted pipeline and evaluation outputs, and generate a report. Model artifacts and downloaded data remain local and are excluded from Git.

Run the tests with:

```bash
pytest
```

## Home Credit SQL extension

The repository also contains `sql/aggregate_features.sql`, a multi-table DuckDB feature pipeline for Home Credit application, bureau, and prior-application tables. It aggregates historical records to applicant level and keeps information observed after the current application out of the feature set unless a safe as-of join is available.

To run this extension, download the competition files after accepting the dataset terms, place `application_train.csv`, `bureau.csv`, and `previous_application.csv` under `data/raw/`, and run:

```bash
python -m clearcredit.cli prepare --config configs/default.yaml
python -m clearcredit.cli train --config configs/default.yaml
```

The Home Credit extension has **not** been run on the full competition dataset, so no performance claim is made for it. Its features and model comparison are separate from the measured UCI benchmark above.

## Modeling and evaluation

- DuckDB/SQL feature aggregation with schema, target, and data-quality checks.
- Logistic regression baseline and LightGBM comparison; CatBoost is an optional extension.
- Five-fold stratified cross-validation; preprocessing is fit within each fold.
- Locked stratified holdout for final reporting.
- Calibration diagnostics, threshold analysis, and explicit scenario costs.
- PR-AUC, ROC-AUC, log loss, Brier score, calibration slope/intercept, precision, and recall.
- Permutation importance, EDA summaries, model card, scoring API, Dockerfile, CI workflow, and tests.

## Local research API

After fitting a model, launch the demonstration API:

```bash
uvicorn clearcredit.api:app --host 127.0.0.1 --port 8000
```

`GET /health` reports whether a model artifact is loaded. `POST /score` accepts a JSON object containing the model features and returns a calibrated risk score plus a research-threshold flag. The endpoint includes a warning and must not be used to make actual lending decisions.

## Repository layout

```text
configs/                   UCI benchmark and Home Credit settings
reports/uci/               Included benchmark metrics, model card, and diagnostics
sql/                       Multi-table Home Credit feature aggregation
src/clearcredit/           Ingestion, checks, modeling, evaluation, CLI, API
tests/                     Feature, evaluation, and API contract tests
.github/workflows/         Continuous integration test workflow
Dockerfile                 Container definition for the local API
```

## Responsible-use limitations

- The UCI sample is historical and geographically narrow; it is not a prospective validation set for another lender or market.
- A random stratified split does not test temporal or customer-level drift.
- Calibration and the chosen threshold do not establish fairness, legal compliance, or suitability for credit decisions.
- The cost ratio is illustrative. A real decision policy would need validated business costs, policy constraints, explainability, governance, and jurisdiction-specific legal review.
- Descriptive group checks in this dataset are not proof that a model treats people fairly.

## CV-ready summary

> Built a leakage-aware credit-risk model comparison on 30,000 UCI cases; selected calibrated LightGBM over logistic regression using five-fold PR-AUC (0.555 ± 0.009), then achieved 0.547 PR-AUC and 0.774 ROC-AUC on a locked holdout, with calibration and threshold trade-offs documented.

State that this is a historical Taiwan benchmark. Do not imply lender impact, production approval, or transferability to another market.
