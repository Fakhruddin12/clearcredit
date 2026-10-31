# Results and model choice

This retrospective study asks whether six months of repayment history add useful information beyond the latest account record, and whether a recurrent model is needed to use that information.

## Read the numbers correctly

The reporting sample contains **6,000 accounts and 1,327 observed defaults**. Each main queue reviews exactly **1,200 accounts**. Capture divides selected defaults by all 1,327 defaults; it is different from the fraction of reviewed accounts that default.

Full-history LightGBM selects **678 defaults and 522 non-defaults**: **51.09% default capture** and **56.50% queue precision**. Another **649 defaults remain outside the queue**. These are observed historical outcomes, not prevented defaults or realised savings.

## What the comparisons show

| Comparison | Additional defaults in the same-size queue | Capture difference | Paired 95% interval, percentage points |
|---|---:|---:|---:|
| Full-history vs latest-month LightGBM | +41 | +3.09 percentage points | [1.27, 4.63] |
| Ordered vs permuted-order GRU | +23 | +1.73 percentage points | [0.61, 3.07] |
| Ordered GRU vs flattened monthly MLP | +6 | +0.45 percentage points | [-0.44, 1.15] |
| Ordered GRU vs full-history LightGBM | -7 | -0.53 percentage points | [-2.05, 1.03] |
| Ordered GRU vs GRU without status channels | +112 | +8.44 percentage points | [6.43, 10.23] |

Historical features help this tree workflow. Consistent order helps the tested GRU, and repayment-status inputs contribute substantially. Recurrence has no clear capture advantage over the flattened network or tree comparator. The history-tree comparison includes engineered historical summaries; it does not isolate one lag or feature.

## Which model should advance?

**Advance full-history LightGBM for independent validation under the stated 20% review objective.** The original development-selected model remains the GRU. This is a retrospective research recommendation, not a new model selected on an untouched test set or a deployment approval.

The GRU was faster in the recorded CPU batch measurement: 37.6 ms versus 56.0 ms per 6,000 accounts. The choice is therefore not justified by claiming the tree is faster. The GRU has not demonstrated the required capture improvement. Neither model has measured intervention value or production operating costs.

## What remains uncertain?

The reporting cohort is reused and comes from one historical 2005 population. The 500 paired group-bootstrap draws condition on fitted models. Five training seeds are a separate sensitivity check. These do not establish stability across future populations or account for all earlier research choices. Secondary intervals are exploratory and not multiplicity-adjusted; controls share a budget inherited from the original GRU and use one order-permutation realisation.

Among accounts without a latest delay, a separately allocated subgroup fifth captures 37.63% of defaults with the GRU and 37.93% with LightGBM. Some accounts have earlier delays. This is within-group ranking under a different allocation policy, not proof of first-deterioration detection.

The 20% workload and two-point improvement criterion are illustrative research choices, not validated economic optima. The next substantive step is the [independent later-outcome study](NEXT_STUDY_PROTOCOL.md).

[Full results](results/results.json) · [Protocol](results/protocol.json) · [Report](ClearCredit_Project_Report.pdf) · [Reproduction](README.md#reproduce)
