# Model card: ClearCredit post-review comparison

**Status:** developer-run exploratory research; no independent validation or deployment.

**Population/outcome:** UCI 350 historical Taiwan existing credit-card accounts; six months of 2005 financial inputs and the source-defined next-month default label. No regulatory/lifetime default reinterpretation.

**Objective:** capture observed defaults within an exactly 20% hypothetical review queue. It does not measure prevented defaults, actual treatment effect or savings.

**Evidence:** full-history LightGBM captures 678/1,327 defaults versus 637 for latest-month LightGBM. Ordered GRU captures 671, flattened MLP 665, permuted-order GRU 648 and no-status GRU 559. The history-tree contrast is 3.09 percentage points, paired 95% interval 1.27–4.63. All these results are retrospective.

**Recommendation:** advance full-history LightGBM for independent validation under the specified review objective. Preserve the original development-selected GRU; do not label a reporting winner as newly development-selected. The GRU fails the declared requirement for at least 2 points of capture improvement and a positive paired lower bound over LightGBM. Its local CPU batch inference was faster; latency is not the failing criterion.

**Inputs:** financial only; exclude ID and demographics. Status-only and latest-month models have explicit restricted feature lists. The no-status GRU removes every status and status-missingness channel. The flattened MLP uses the same 103 monthly/static values as the ordered GRU.

**Controls:** financial-input groups do not cross the original four partitions. Final fitting 18,714; calibration 4,678; reporting 6,000. Source checksum and partition hashes match v2. Latest models use development-only selection. Primary neural ensembles use three original seeds; five individual fits assess seed sensitivity.

**Early-warning correction:** no latest delay does not mean no earlier distress.641 of 4,676 such accounts have an earlier delay code. A separate 936-account subgroup queue captures 37.63% of 667 defaults with GRU, versus 37.93% with the history tree. It changes review allocation. No genuine future first-deterioration endpoint or validation cohort was added.

**Uncertainty:**500 paired financial-group bootstrap draws conditional on fitted models. No retraining uncertainty, full research-selection correction or multiplicity-controlled architecture claims. One fixed order-permutation realisation; inherited GRU training budget may disadvantage another architecture.

**Other failure modes:** original masking changes 18.05% of original-policy review flags; natural missing-history fallback unvalidated. Age calibration gaps and pure-label-shift sensitivity remain descriptive. Old global-threshold diagnostics are not silently reused as measurements of the new calibration-threshold policy.

**Reproducibility:**16 tests pass; restore verification matches all eleven score vectors and all aggregate/subgroup headline metrics. Source data, weights and per-account predictions generated locally; summary results committed. The original API and Home Credit SQL extension remain separate unbenchmarked/developer demonstrations.

**Next requirement:** independently later, matured outcomes and a frozen chronological study as specified in NEXT_STUDY_PROTOCOL.md. Independent challenge, actual policy economics, input lineage, exception handling and operational approval remain outstanding.
