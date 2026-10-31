> Historical v2 study. The [post-review challenge](../v3/README.md) adds matched-capacity baselines and history ablations, recommends a model for further validation, and corrects the interpretation of global-policy recall as a general early-warning conclusion. Original results remain preserved below.

# Model card: ClearCredit behavioural GRU

**Status:** research demonstration, developer-tested. No independent validation or production approval. No deployment performed.

**Intended question:** Can six monthly financial-behaviour snapshots improve next-month source-defined default-risk ranking for existing credit-card accounts?

**Research use:** retrospective ranking and analysis of a hypothetical account-review queue. A flag is not a credit decision or a recommendation to restrict a person.

**Population:** 30,000 historical Taiwan credit-card accounts from UCI dataset 350. Six months of history, source label `default payment next month`. Do not substitute a one-year regulatory PD or a standard 90-day default definition.

**Information:** credit limit; six repayment-status, balance and payment histories. Sex, age, education, marriage, ID and the target are excluded. Demographic exclusion does not prove fairness.

**Architecture:** one monthly GRU layer, hidden width 24, static credit-limit branch, a 32-unit nonlinear head, dropout 0.2. Three independently seeded networks are averaged. Unweighted binary log loss, AdamW, gradient clipping. Separate sigmoid probability calibration.

**Comparators:** properly transformed logistic regression, LightGBM, residual neural network and a predeclared fixed tree/neural blend. These use the same available financial information, but the tabular families receive engineered summaries and the GRU receives the raw ordered channels and credit limit. This is not a controlled isolation of architecture alone.

**Selection:** best development average precision. Two modest neural presets, three tree presets and three logistic penalties. Selected neural presets use seeds 42/137/2026. Final neural epochs are fixed from development histories. Not an exhaustive architecture search.

**Partition controls:** 6,000 original reporting accounts; 608 development rows with matching reporting financial profiles excluded. Development financial groups do not cross fit/tune/calibration boundaries. Final base fitting 18,714, calibration 4,678. The already inspected reporting set is not pristine or out-of-time.

**Results:** GRU reporting AUC 0.7772; AP 0.5521; highest-risk 20% capture 50.57%. LightGBM AUC 0.7753 and capture 51.09%. No clear neural advantage over LightGBM in the paired AUC interval. GRU calibration mean 21.56% predicted vs22.12% observed; slope 0.987. Full precision, denominators and intervals are in results.json.

**Known failure modes:** heavy reliance on already recorded repayment delays; 85.2% missed defaults in the other latest-status group at the fixed threshold; score instability under missing recent history; some subgroup underprediction; calibration failure under changed prevalence; historical market/population restrictions; selection bias from existing account holders; unobserved income and economic shocks; no measured causal benefit from intervention.

**Explanations:** grouped permutation is supplied as a global predictive-dependence diagnostic. It is not a causal attribution, robust individual explanation or legally validated adverse-action reason system. No explanation-fidelity or individual-reason test was completed.

**Operational controls needed before any genuine use:** agreed input lineage and observation cut-offs; rejection/fallback for missing required history; compatible matured outcomes; independent review; local policy and legal assessment; prospective/out-of-time validation; relevant loss/exposure data where monetary claims are sought; model-specific monitoring and escalation criteria.

**Monitoring specification, not a deployed monitor:** schema/units/ranges and missingness; account-history completeness; score and input distribution changes; matured-outcome AUC/AP/capture, risk-band observed/expected rates and workload; subgroup error and calibration changes. An input-drift statistic does not alone prove model failure. No universal numeric drift/approval thresholds have been validated.

**Reproducibility:** pinned workbook SHA-256, partition hashes, locked selection, all tuning histories, environment and seven contract checks included. The experiment generates three-seed weights, preprocessing, calibrators, split indices and predictions locally; those generated files are excluded from Git. Sample bootstrap intervals exclude retraining and model-selection uncertainty.

**Remaining work:** external or genuinely later outcome-labelled data; earlier-deterioration study; latest-month and order-destroyed ablations; broader fair tuning; explanation fidelity; realistic intervention/workload economics; independent challenge. No claim these steps have already been completed.
