# Independent validation: specified, not yet run

The current UCI cohort cannot answer this stage. This document is a concrete design for a later study; no new data, results or validation are claimed.

## Question and population

For existing credit-card accounts with six months of history, does ordered history improve next-month default capture beyond latest-month information at a fixed review workload? Retain the provider's documented default definition. Do not silently substitute a regulatory or lifetime default definition.

## Data acceptance and freezing

Require legally usable monthly account histories, stable account identifiers, real observation dates, documented units/status codes and fully matured next-month outcomes. Require at least 24 monthly observation dates to support more than one economic window. Document missingness, closures, eligibility and source selection before evaluating a model. Freeze the source checksum, as-of feature specification and outcome-maturity cutoff.

For an initial customer-disjoint time benchmark, choose one eligible observation per customer using an outcome-independent deterministic rule fixed in the manifest. This sacrifices repeated snapshots but prevents the same customer's labelled rows crossing partitions. State that this estimates performance on held-out existing customers, not longitudinal repeat scoring. A separate longitudinal benchmark can later assess repeat scoring with explicit temporal purging and account-clustered inference.

## Splits and controls

Use the earliest 60% of eligible observation dates for development, the next 20% for independent calibration and the last 20% for reporting. Choose date boundaries from the date inventory and maturity metadata before inspecting model results. Within development, use at least three expanding-window tuning folds. Purge training rows whose outcome windows extend into the next evaluation window. Do not split by arbitrary account IDs or treat six predictor months as six outcome cohorts.

Preserve a final untouched calendar window. If reporting support is insufficient for useful uncertainty estimates, report that limitation instead of silently changing boundaries. Freeze eligibility and subgroup definitions before reporting labels are used for evaluation.

## Models and common budget

Include the binary latest-delay rule, severity rule, one-status logistic model, latest-month logistic/LightGBM, full-history LightGBM, flattened monthly MLP and ordered GRU. Include an independently permuted monthly-order GRU and a no-status GRU as diagnostic controls. Use the same raw monthly channels for neural representation comparisons. Give latest/full tree models the same search budget and use five predetermined seeds for neural models. Select hyperparameters and epochs inside development only. Fit probability recalibration on the later independent calibration window.

## Objective, uncertainty and stopping

Primary objective: default capture at exactly 20% review capacity. Secondary: AUC, average precision, proper probability scores, calibration by risk band, inference time and subgroup capture at a separately defined workload. Fix label-independent tie handling.

Carry forward the review-stage practical threshold: at least a two-percentage-point absolute capture advantage over LightGBM, positive lower bound of a paired 95% interval, and median inference time no more than twice LightGBM. This is a proposed research decision rule, not a validated business value threshold. Refine it only with genuine review-cost and intervention evidence before the independent study is frozen.

Report paired account-cluster bootstrap intervals and seed variation separately. Make the primary architecture comparison explicit; mark secondary ablations exploratory rather than claiming multiplicity-controlled significance. End the experiment after the fixed run. If the criterion fails, retain the negative result and prefer the simpler comparator for further validation.

## Deliverable

A versioned source manifest; frozen protocol and dates; source code and meaningful tests; development search records; generated predictions and fitted artifacts for authorized local reproduction; all model results and uncertainty; a model recommendation; and a plain research report. Release only permitted aggregates, never confidential customer records. Independent review and operational approval remain separate from this developer study.
