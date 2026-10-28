# ClearCredit: behavioural credit-risk research and model challenge

Research date: 28 October 2026. Prepared for Fakhruddin Hakim Hussain.

## Assessment

The original project was a useful end-to-end classification demonstration, but its headline AUC, a calibrated score and an API did not establish a distinctive credit-risk contribution. Adding a neural network alone would not change that. The stronger project asks whether repayment trajectories add useful information, compares them fairly with simpler models, and identifies when a good-looking portfolio score hides a weak decision process.

I implemented and ran that extension. A three-seed monthly GRU was selected using development average precision. On the previously used 6,000-account reporting set, it achieved **0.777 ROC-AUC**, compared with **0.775 for LightGBM** and **0.765 for an improved logistic baseline**. The neural advantage over LightGBM is small and its paired confidence interval includes zero. At an exact 20% review capacity, LightGBM captures slightly more defaults. This is evidence for a more rigorous research project, not a claim of a breakthrough model.

The most useful new finding is a limitation: at the fixed development review threshold, the GRU misses **85.2% of defaults in accounts whose latest repayment-status code is below 1**. It recognises already visible distress much better than it identifies deterioration before that signal appears. That distinction should drive the next research stage.

This directory contains the measured second research study. The root API remains a demonstration of the earlier tabular workflow; the Home Credit SQL extension has not been benchmarked.

## 1. What a professional risk model must define first

**Population and use.** These are existing credit-card accounts in a historical Taiwan sample, not new applicants. The defensible use case is a hypothetical queue for account-risk review. This is behavioural scoring. It cannot establish approval performance on previously rejected applicants, collection-treatment effectiveness or loan profitability. The UCI source describes six monthly histories and a next-month default-payment label [1].

**Observation and outcome.** Use only information available at the latest statement. Keep all six months of a person together. Preserve the dataset's target definition: it is not automatically a 90-day delinquency definition, a one-year regulatory PD, or an IFRS 9 lifetime PD. A production specification would also define exclusions, incomplete outcome windows, cures and repeat default events.

**Rank and probability are different outputs.** A model may order accounts well while systematically underestimating their risks. The Basel validation research treats discriminatory power and calibration separately [2]. Here, ranking is evaluated with AUC, average precision and default capture at a fixed workload; probabilities are evaluated with risk-group observed rates, calibration slope/intercept, observed-to-expected default ratio and proper scoring rules.

**Risk estimates and actions are also different.** Predicting default does not show that reviewing or restricting an account prevents default. Expected credit loss would additionally require a compatible loss-given-default and exposure-at-default definition, recovery and exposure data, and an appropriate horizon. Those are absent. The project therefore reports hypothetical review workload, not dollar savings, expected-loss reductions or optimised credit limits.

**Validation must challenge the model's intended use.** The revised US interagency model-risk guidance, SR 26-2, replaced SR 11-7 in April 2026. It discusses purpose, data, conceptual soundness, outcomes analysis, monitoring and effective challenge, with effort proportionate to model risk [3]. These are useful organising principles here, not a claim that this student project satisfies a bank's governance or independent validation requirements.

## 2. Literature review and decisions taken

This is a focused review of primary research and supervisory sources relevant to this dataset, not an exhaustive systematic review. Sources were chosen for model comparison, sequential account behaviour, probability calibration, interpretability and validation. No result from another paper is treated as a performance promise for ClearCredit.

| Source | Relevant evidence or argument | Decision for this project |
|---|---|---|
| Lessmann et al. (2015), credit-scoring benchmark [4] | Evaluates 41 classifiers on eight credit datasets and examines the link between prediction measures and managerial value. | Keep credible simple and tree baselines; report a workload-dependent result as well as discrimination. |
| Gorishniy et al. (NeurIPS 2021), tabular ResNet and FT-Transformer [5] | Establishes strong neural baselines under shared comparisons and finds no universally best family against boosted trees. | Implement a small residual network as a neural control. The implementation is inspired by that family, not an exact reproduction. |
| Grinsztajn et al. (NeurIPS 2022) [6] | Shows that trees remain strong on medium-sized tabular benchmarks and investigates their different inductive biases. | Do not discard LightGBM merely to put deep learning on the CV. |
| Clements et al. (2020), American Express/Rutgers [7] | Studies recurrent and temporal-convolution models for credit monitoring using much longer transaction histories. | Test an ordered-history GRU plus a static credit-limit branch. Six UCI monthly snapshots are much less information than the paper's sequences; its financial benefits cannot be transferred. |
| Gorishniy et al. (ICLR 2025), TabM [8] | Shows the value of parameter-efficient MLP ensembling in tabular deep learning. | Use three independent seeds to reduce neural run-to-run variability in this experiment. This is conventional ensembling, **not an implementation or replication of TabM**. TabM remains a later challenger. |
| Guo et al. (ICML 2017) [9] | Finds that neural classification accuracy and probability calibration can differ, and studies post-processing calibration. | Fit sigmoid recalibration on a separate calibration partition. The paper's image/text findings are motivation to check, not evidence that a particular credit calibrator will win. |
| Rudin (2019), high-stakes interpretability [10] | Argues that post-hoc explanation is not equivalent to an inherently interpretable decision model. | Retain logistic regression and do not label permutation importance as a complete account-level reason system. |
| EBA discussion paper (2021) [11] | Discusses the performance/explainability trade-off and avoiding complexity without useful predictive gains in IRB work. | Assess whether the neural gain justifies its complexity; do not infer regulatory acceptability from a higher AUC. The document is a discussion paper, not a certification standard. |

**Why GRU rather than an arbitrary Transformer?** The available structure is a short ordered repayment history. A small recurrent encoder directly tests whether ordering and joint balance/payment trajectories add information. A residual network tests whether a generic nonlinear tabular model is enough. FT-Transformer and TabM are credible future comparators, but adding every architecture after seeing reporting results would make the comparison less trustworthy. Neither has been evaluated here.

## 3. Audit of the original implementation

The existing repository was inspected at master commit `905c1b28a016e5e1f3dd5e343aa4d01d931e0f9d` [12].

1. **Demographic exclusion and audit did not match the UCI schema.** The code drops `CODE_GENDER` and `DAYS_BIRTH` and looks for them during auditing. UCI supplies `SEX` and `AGE`, so those fields remain predictors and the saved group audit reports only `All`. The extension explicitly excludes sex, age, education, marriage and ID and uses sex and age only for descriptive audits. Removing them does not remove proxy discrimination or prove fairness.
2. **Class weights complicate PD estimation.** The original balanced logistic and tree losses change the effective class prior. Calibration can compensate, but should not be assumed to solve every resulting distortion. The extension uses unweighted log loss and evaluates minority-outcome performance using average precision and capture metrics rather than resampling or SMOTE.
3. **The original probability comparison was uneven.** Cross-validation compares uncalibrated, class-weighted probability scores; only the selected model receives the final calibration step. The extension fits an independent calibrator for every candidate family and reports raw and calibrated results separately.
4. **A fresh test set cannot be created by renaming an old one.** The original 6,000 cases have already been inspected. They remain a descriptive internal reporting set. No future-market, prospective or untouched-test claim is made.
5. **A multi-table pipeline is not evidence of a measured multi-table model.** The original Home Credit SQL extension has not been fully benchmarked. It stays separate from all UCI performance claims.

## 4. Implemented experiment

### Data integrity and partitions

The public workbook is downloaded from UCI, schema checked and pinned by SHA-256: `30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933`.

After removing demographic inputs, **817 rows share their 19 financial inputs with an earlier row**. There are **85 identical-input groups containing different labels**. These are not necessarily duplicate people: many accounts genuinely share low-information histories. Different outcomes for identical available inputs are also a direct reminder that individual defaults cannot be inferred with certainty.

The original seed-42 stratified reporting split is retained. **608 development rows** with financial-input patterns also present in reporting are excluded from fitting. The remaining development set is split by financial-input groups, preventing identical available profiles from crossing its fit, tuning and calibration partitions. This is deliberately conservative and changes the original protocol.

| Partition | Accounts | Use |
|---|---:|---|
| Initial fitting | 14,036 | Fit candidate parameters and preprocessing |
| Tuning | 4,678 | Select configuration, early-stopping epochs, family and review cut-off |
| Calibration | 4,678 | Fit probability recalibration only |
| Reporting | 6,000 | One scheduled internal evaluation and descriptive diagnostics |

After selection, base models are refit on the 18,714 fitting+tuning accounts. Calibration accounts stay excluded. Split indices and hashes are saved. The same account's months are never treated as six independent training examples.

### Predictors and models

The common information set is credit limit plus six monthly repayment-status, billed-balance and payment-amount histories. The tabular models also receive utilisation, payments relative to limit and simple historical summaries. Signed-log transforms preserve negative billed balances; they are not blindly deleted as errors. Payment/limit features are not described as legally due repayment ratios.

Status values `-2`, `-1`, `0` and positive delay codes are preserved as distinct categories in neural/logistic preprocessing. Unclear codes are not silently renamed “good credit.” The sequence runs from the oldest month to the latest, with status, balance and payment channels aligned. Scaling and imputation are fit only on the relevant fitting data.

| Family | Implemented search |
|---|---|
| Logistic regression | Three regularisation values with nonlinear numeric preprocessing and status indicators |
| LightGBM | Three leaf-size/regularisation presets, 450 trees at learning rate 0.03 |
| Residual neural network | Two widths, two residual blocks, dropout, early stopping; selected preset uses three seeds |
| Monthly GRU | Two hidden widths, six ordered monthly observations, static credit-limit branch; selected preset uses three seeds |
| Blend | Predeclared 50:50 average of LightGBM and the strongest development neural family |

Tuning is intentionally modest and is not a claim of an exhaustive or compute-matched architecture benchmark. Neural fits use unweighted binary log loss, AdamW, gradient clipping, at most 60 epochs and patience eight on tuning log loss. Family selection uses tuning average precision. Final neural refits use the median selected epoch count: five for the residual network and 26 for the GRU.

The GRU ensemble wins development average precision (**0.5723**) by a small margin over the blend (**0.5717**), so it remains the selected research model. The blend is **not retrospectively promoted** because it later has the best reporting AUC.

## 5. Results and what they mean

All entries below use the same 6,000 reporting accounts, with 1,327 observed defaults (**22.12%**). Probabilities are calibrated on separate development accounts.

| Model | ROC-AUC | Average precision | Defaults captured in highest-risk 20% | Defaults captured, count |
|---|---:|---:|---:|---:|
| Logistic regression | 0.7647 | 0.5204 | 50.04% | 664 |
| LightGBM | 0.7753 | 0.5495 | 51.09% | 678 |
| Residual network, three seeds | 0.7732 | 0.5416 | 49.89% | 662 |
| **GRU, selected on development** | **0.7772** | **0.5521** | **50.57%** | **671** |
| Fixed tree/GRU blend, challenger | 0.7789 | 0.5557 | 50.41% | 669 |

**Readable interpretation:** ranking 1,200 of the 6,000 accounts as highest risk places 671 of the 1,327 eventual defaults in that group. That is **50.6% of defaults at a 20% workload**, or **2.53 times random selection's expected concentration**. It also means 529 reviewed accounts do not default and 656 defaults remain outside that exact-capacity group. A high model score is not a definitive diagnosis.

The selected GRU's ROC-AUC is **0.7772, with a 95% cluster-bootstrap interval of 0.7610–0.7918**. Its AUC difference over LightGBM is about **0.0020**, with a paired interval of **−0.0031 to +0.0071**. The average-precision difference also includes zero. It would be misleading to claim that deep learning clearly outperforms boosting. The modest search does not establish that the two families are equivalent either.

Intervals use 500 paired bootstrap samples of identical-financial-input groups. They describe reporting-sample variability conditional on these fitted models. They do not incorporate training, tuning, prior researcher choices, future economic drift or multiplicity across every diagnostic.

### Probability reliability, in understandable terms

The GRU predicts a mean risk of **21.56%** against **22.12% observed**. Its calibration slope is **0.987** and observed-to-expected defaults **1.026**. These are reasonably close at portfolio level, but do not establish accuracy in every group.

For example, the lowest predicted-risk tenth has **4.82% predicted** against **4.17% observed**, while the highest tenth has **70.34% predicted** against **69.83% observed**. Some middle groups are underestimated: decile five has **13.17% predicted** against **15.67% observed**. The complete decile table includes counts and Wilson intervals; intervals assume independent outcomes and should be read alongside the clustered aggregate analysis.

The technical Brier score is retained in `results.json`: it is the mean squared error of predicted probabilities against 0/1 outcomes, and lower is better. Relative to assigning everyone the refit-sample default rate, the GRU reduces this probability error by **21.3%**. That is a comprehensible benchmark, not a 21.3% reduction in defaults or credit losses. Calibration improves this score only slightly here; it should not be advertised as a large model gain.

## 6. Where the model works and where it fails

### It mostly identifies observable distress

At the fixed development threshold, reporting review volume is **20.28%**. Of 1,324 accounts with a latest status code of at least 1, the observed default rate is **49.85%** and the model misses **12.9% of their defaults**. Among the other 4,676 accounts, the observed default rate is **14.26%**, but it misses **85.2% of defaults**. Their within-group AUC is only **0.683**, compared with **0.736** in the delay-code group.

This is a more revealing challenge than the overall AUC. The model's usefulness as an *early* warning tool is limited, because many captured cases already have a visible repayment-status warning. A sensible next experiment would train and validate a dedicated early-deterioration model among accounts without that warning, using genuinely earlier observations and a clearly defined future outcome. That experiment has **not** been run here and must not reuse this reporting set for design and final proof.

Grouped permutation supports this dependence: moving repayment-status histories between accounts reduces average precision by about **0.275**, much more than moving balances (**0.037**), payment amounts (**0.044**) or credit limits (**0.030**). These perturbations can create unusual combinations. The numbers show predictive dependence, not causal effects or an adequate adverse-action explanation.

### Missing recent data is a material operational risk

Masking the latest month's status, balance and payment changes **18.05% of review flags**; median absolute score change is **9.47 percentage points**, with a 95th percentile of **24.19 points**. The missingness mask prevents confusing a missing status with a zero code, but the model was not trained on real missing-history patterns. Its fallback is therefore not validated. A production scoring service should route missing required history to a controlled exception/fallback process rather than silently treating the imputed score as equally reliable.

Increasing nonnegative latest-month delay codes by one changes **5.73% of flags**. This is a synthetic input-sensitivity check. It does not estimate how a real customer would behave or how a recession would change default probabilities.

### Good aggregate calibration hides subgroup errors

Sex and age are not model inputs. Nevertheless, for accounts aged 45 or above, mean predicted risk is **22.29%** against **24.49% observed**. For those under 30 it is **23.28%** against **22.33% observed**. Review rates also differ. These differences warrant investigation of sampling, omitted variables, calibration and policy consequences. They are descriptive findings, not proof of unlawful discrimination, its absence, or a justification for group-specific lending thresholds.

### Base-rate change breaks a static probability estimate

Under an illustrative **pure label-shift** scenario with a 35% default rate and unchanged feature distributions conditional on outcome, the unadjusted model's weighted mean prediction is only **24.33%**. A prior-odds adjustment moves it to **34.64%**. This demonstrates calibration's dependence on population assumptions. The true future prior would need estimation; pure label shift does not capture changes in income, underwriting, recoveries or the relationship between behaviour and default.

### What the dataset cannot establish

The source has one historical observation cohort, no long panel of separately labelled customer snapshots and no usable independent future reporting cohort. Six feature months do **not** create six prospective outcome cohorts. Sorting arbitrary IDs would not fix that. There is no evidence here for geographic transportability, economic-cycle stability, external lender performance, actual treatment benefit or a capital/ECL application.

## 7. Professional next-stage design

The first priority is more appropriate validation data, not a larger network. Obtain legally usable account snapshots with observation dates, fully matured future outcomes, multiple vintages and a stable account key. Freeze an earlier-period development sample, a subsequent calibration window and an untouched later reporting window. Keep customers together, purge overlapping outcome windows where relevant, and define how repeat account observations match the deployment population. Add several rolling-origin evaluations rather than presenting a single fortunate period.

An American Express competition dataset is a potential behavioural-sequence resource [13], subject to its access terms. Its label is different: the host describes an 18-month performance window and a 120-day non-payment condition. It must not be treated as interchangeable with UCI next-month default or assumed to supply every required out-of-time cohort. Access, actual timestamp coverage, sample selection and permitted uses would need inspection before building that study. No competition data was accessed in this task.

For that richer study, compare an interpretable scorecard/GAM, a tuned tree model, TabM, and a temporal model such as a GRU or causal temporal convolution. Include a latest-month-only baseline and an order-destroyed history ablation to test whether apparent sequence gains really depend on temporal information. Predefine a primary workload or cost objective, minimum practically meaningful gain, calibration tolerances and subgroup reporting requirements. Evaluate on more than one suitable dataset before making a broad architecture claim.

A professional model-development package should additionally specify feature lineage and as-of rules, outcome maturity, exclusions and missingness controls, account-level explanation fidelity, an accountable owner, validation findings, versioned approval and a monitoring response. Monitoring should track missing required inputs and schema failures, score/input drift, outcome-matured ranking, observed-versus-expected defaults by risk band, workload, and subgroup errors. Input drift alone does not prove performance deterioration, and universal PSI cut-offs are not a substitute for a model-specific escalation policy.

This extension is developer testing. Independent challenge, prospective evidence, policy approval and production operation remain separate requirements.

## 8. Reproducibility and evidence

`experiment.py` downloads and validates the workbook, records a protocol before fitting, constructs grouped partitions, tunes and refits candidates, calibrates them, and produces metrics and charts. `results/selection_locked_before_reporting.json` records the development-selected model. `results/results.json` contains every reported model, the clustered intervals, subgroup checks, sensitivity scenarios and timing. The generated `results_reproduced/reporting_predictions.csv` enables metric reproduction after running the experiment. Running the experiment generates model parameters, preprocessing, split arrays and per-case predictions for local analysis. These generated binary artifacts and row-level predictions are excluded from Git. Only load trusted locally generated joblib files, which are executable Python serialisations.

Seven contract checks passed: exclusion of demographics/target/ID from predictors; monthly channel ordering; capture denominators and review counts; prior-shift algebra; separation of calibration and reporting priors in stress weights; explicit missing-status encoding; and bounded Wilson intervals. Cross-partition financial-group separation is asserted in the experiment itself. A separate restore check reproduced all 6,000 calibrated GRU predictions from the saved weights and verified AUC, average precision, capture counts, total defaults and the stress denominator. Research charts were visually inspected.

The first diagnostic run contained a stress-weight denominator error when reporting and calibration prevalences differed. It was corrected to weight the reporting sample by its own empirical prevalence; a regression check was added. No base models, selection choices or reported candidate performance were retuned after reporting results were inspected.

## References

All sources accessed 28 October 2026 unless noted. The official 2023 EBA follow-up was identified but its full text was blocked, so the substantive EBA discussion above relies on the accessible 2021 discussion-paper material rather than presenting the unread follow-up as reviewed.

1. UCI. *Default of Credit Card Clients.* Dataset metadata, variable definitions and workbook. https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
2. Basel Committee research group (2005). *Studies on the Validation of Internal Rating Systems*, revised Working Paper 14. Historical research on validation, not a current compliance checklist. https://www.bis.org/publ/bcbs_wp14.pdf
3. Federal Reserve/OCC/FDIC (2026). *Revised Guidance on Model Risk Management*, SR 26-2, 17 April 2026. https://www.federalreserve.gov/supervisionreg/srletters/sr2602.htm and https://www.federalreserve.gov/frrs/guidance/supervisory-guidance-on-model-risk-management.htm
4. Lessmann, S., Baesens, B., Seow, H.-V. and Thomas, L.C. (2015). *Benchmarking state-of-the-art classification algorithms for credit scoring: An update of research.* EJOR 247(1), 124–136. Publisher abstract/introduction reviewed. https://doi.org/10.1016/j.ejor.2015.05.030
5. Gorishniy, Y., Rubachev, I., Khrulkov, V. and Babenko, A. (2021). *Revisiting Deep Learning Models for Tabular Data.* NeurIPS. https://arxiv.org/abs/2106.11959 and https://arxiv.org/html/2106.11959v5
6. Grinsztajn, L., Oyallon, E. and Varoquaux, G. (2022). *Why do tree-based models still outperform deep learning on tabular data?* NeurIPS. https://arxiv.org/abs/2207.08815
7. Clements, J.M., Xu, D., Yousefi, N. and Efimov, D. (2020). *Sequential Deep Learning for Credit Risk Monitoring with Tabular Financial Data.* Research preprint, including American Express AI Research authors. https://arxiv.org/abs/2012.15330 and https://arxiv.org/html/2012.15330v1
8. Gorishniy, Y., Kotelnikov, A. and Babenko, A. (2025). *TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling.* ICLR. https://arxiv.org/abs/2410.24210 and https://arxiv.org/html/2410.24210v3
9. Guo, C., Pleiss, G., Sun, Y. and Weinberger, K.Q. (2017). *On Calibration of Modern Neural Networks.* ICML, PMLR 70, 1321–1330. https://proceedings.mlr.press/v70/guo17a.html
10. Rudin, C. (2019). *Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead.* Nature Machine Intelligence 1, 206–215. Perspective. https://www.nature.com/articles/s42256-019-0048-x
11. EBA (2021). *Discussion paper on machine learning for IRB models*, EBA/DP/2021/04, 11 November 2021. Consultation document, not binding guidance. https://www.eba.europa.eu/publications-and-media/press-releases/eba-consults-machine-learning-internal-ratings-based-models ; official PDF https://www.eba.europa.eu/sites/default/files/document_library/Publications/Discussions/2022/Discussion%2520on%2520machine%2520learning%2520for%2520IRB%2520models/1023883/Discussion%2520paper%2520on%2520machine%2520learning%2520for%2520IRB%2520models.pdf
12. Fakhruddin12/ClearCredit. Existing source and saved benchmark inspected, commit `905c1b28a016e5e1f3dd5e343aa4d01d931e0f9d`. https://github.com/Fakhruddin12/clearcredit
13. American Express/Kaggle (2022). *American Express – Default Prediction*, host dataset description. Potential future data source only. https://www.kaggle.com/competitions/amex-default-prediction/data
