# ClearCredit — LinkedIn Project Entry

Saved: 31 October 2026

**Project name:** ClearCredit | Credit Risk Modelling and Validation

**Description:**

Investigated how much repayment history adds beyond an account's latest status when predicting next-month credit-card default.

Compared simple review rules, latest-month models, full-history LightGBM and neural representations using 30,000 historical accounts. At the same 1,200-account review capacity, full-history LightGBM captured 678 defaults versus 637 using latest-month inputs: 41 additional cases at the same workload. An ordered GRU captured 671 versus 648 after month order was disrupted, but had no clear capture advantage over the tree model or a flattened monthly network.

Built separate fitting, tuning and calibration stages, controls for repeated financial profiles, paired bootstrap comparisons and five-seed sensitivity checks. Also separated global-policy recall from within-group ranking when investigating accounts without a latest recorded delay.

The report explains the experiments, model recommendation and limitations. Results are retrospective; independent later-outcome validation remains the next step.

**Skills:** Python · Machine Learning · Credit Risk · Statistical Modelling · Model Validation

**Project URL:** https://github.com/Fakhruddin12/clearcredit

**Media title:** ClearCredit — Repayment History, Model Comparisons and Validation

**Media description:** Research report with capacity-matched baselines, history ablations, seed sensitivity, paired uncertainty, subgroup analysis and an explicit model recommendation.

**PDF to attach:** ClearCredit_Project_Report.pdf


**Public report:** https://github.com/Fakhruddin12/clearcredit/blob/master/research/v3/ClearCredit_Project_Report.pdf

**Project dates:** Use the actual dates from your records; no start date has been assumed.

## Public report link
https://github.com/Fakhruddin12/clearcredit/blob/master/reports/PROJECT_REPORT.pdf

## October review clarification — evidence note
At exactly 1,200 reviews, full-history LightGBM captures 678 observed defaults versus 637 for latest-month LightGBM: 41 additional cases, or +3.09 capture percentage points (paired 95% bootstrap interval 1.27–4.63). The GRU captures 671 and has no clear capacity advantage over LightGBM.

The 6,000-account reporting cohort was already examined. The dataset describes Taiwan accounts in 2005. Ordered-history evidence in one fitted GRU does not establish that recurrence is necessary, and the no-latest-delay subgroup is not a validated first-deterioration event study.
