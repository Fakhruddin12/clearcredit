# ClearCredit

**How much does repayment history add beyond an account's latest status?**

I studied next-month credit-card default using 30,000 historical Taiwan accounts. The project compares simple review rules, latest-month models and six-month history models, then tests whether monthly order and repayment-status information explain the results.

At the same 1,200-account review workload, full-history LightGBM captures **678 defaults**, compared with **637** using latest-month inputs. An ordered GRU captures **671**, versus **648** after monthly order is disrupted. History helps in these comparisons, but the GRU has no clear capture advantage over the tree model or a flattened monthly network.

[**Results in brief**](research/v3/RESULTS_SUMMARY.md) · [**Current research and reproduction**](research/v3/README.md) · [**Project report**](research/v3/ClearCredit_Project_Report.pdf) · [**Model card**](research/v3/MODEL_CARD.md) · [**Implemented changes**](research/v3/CHANGES.md)

## Results at a common workload

The reused reporting sample contains 6,000 accounts and 1,327 defaults. Every row below reviews exactly 1,200 accounts. Neural results use the original three-seed ensemble; additional seeds are sensitivity checks.

| Approach | AUC | Defaults captured | Capture rate |
|---|---:|---:|---:|
| Binary latest-delay rule | 0.678 | 593 | 44.69% |
| Latest-delay severity rule | 0.691 | 616 | 46.42% |
| Latest-month LightGBM | 0.759 | 637 | 48.00% |
| **Full-history LightGBM** | **0.775** | **678** | **51.09%** |
| Ordered GRU | 0.777 | 671 | 50.57% |
| Same monthly inputs, flattened MLP | 0.773 | 665 | 50.11% |
| Independently permuted monthly order, GRU | 0.768 | 648 | 48.83% |
| Repayment-status channels removed, GRU | 0.731 | 559 | 42.13% |

Rules use a fixed, label-independent tie key. Uniform tie handling gives expected capture of 45.08% for the binary rule and 46.73% for severity. Tie sensitivity and the complete eleven-approach comparison are in [saved results](research/v3/results/results.json).

The full-history tree's capture advantage over the latest-month tree is **3.09 percentage points**, with a paired 95% cluster-bootstrap interval of **1.27–4.63 points**. The ordered versus permuted GRU contrast is **1.73 points**, interval **0.61–3.07**. These are conditional exploratory intervals; the reporting cohort was already examined and no multiplicity adjustment is claimed.

![Neural history controls with conditional uncertainty](research/v3/results/history_controls.png)

## What changed after challenging the original study

- Added simple status rules and latest-month controls at exactly the same review capacity.
- Compared ordered, flattened, permuted-order and no-status neural representations under a fixed training budget.
- Reported five-seed sensitivity separately from fitted-model bootstrap uncertainty.
- Corrected an early-warning overstatement: low recall under a global threshold does not establish an inability to rank accounts within a lower-risk subgroup.
- Made an explicit research recommendation: advance full-history LightGBM for independent validation under this review objective. The original development-selected GRU is preserved, not retrospectively renamed.

Among accounts without a latest recorded delay, a separately allocated highest-risk fifth captures **37.63%** of defaults with the GRU and **37.93%** with LightGBM. This group can contain earlier delays. It is not a verified first-deterioration cohort, and the dedicated subgroup queue is a different allocation policy from the global queue.

## Evidence and limits

The source is one historical 2005 cohort, not new applicants or a modern prospective portfolio. Financial-input groups are separated across fitting, tuning, calibration and reporting partitions. New latest-month searches use development labels only. Follow-up neural controls inherit the original GRU budget rather than receiving independent exhaustive tuning.

The reporting cohort was used in earlier research. These results remain **exploratory retrospective evaluation**, not fresh out-of-time validation. No lending intervention, financial savings, production readiness or credit-loss reduction is claimed. [The next independent study is specified but has not been run.](research/v3/NEXT_STUDY_PROTOCOL.md)

## Reproduce

Python 3.12 and a CPU are sufficient. From the repository root:

```bash
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r research/v3/requirements.txt
cd research/v3
python challenge.py --data ../v2/data/source.xls --out results_reproduced
python -m pytest -q tests ../v2/tests
python verify_challenge.py --data ../v2/data/source.xls --results results_reproduced
```

If original fitted artifacts are absent, the runner first reproduces v2 and downloads the checksum-verified UCI workbook. The follow-up itself completed in about 123 seconds using restored baseline artifacts in the recorded environment; a clean run also incurs baseline fitting. Runtime varies by environment.

Sixteen numerical/data-handling tests pass. The restore verifier checks all eleven score vectors for 6,000 accounts and recomputes aggregate and subgroup metrics. Model weights, source data and per-account predictions are generated locally and excluded from Git.

## Earlier versions and engineering scope

[The original v2 experiment](research/v2/) remains as research history. The root `src/clearcredit` CLI/API belongs to an earlier tabular demonstration. A [Home Credit SQL extension](sql/aggregate_features.sql) has synthetic join tests but has not been benchmarked on the full dataset. Neither extension inherits the current benchmark or production approval.

**Author:** Fakhruddin Hakim Hussain · [Portfolio](https://github.com/Fakhruddin12)
