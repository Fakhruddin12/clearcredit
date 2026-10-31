# ClearCredit: research decision

Reviewed 31 October 2026. This note separates the supported result from the next research decision.

## Decision

Full-history LightGBM is the model to advance for the fixed review-capacity objective.

## Strongest relevant comparison

At exactly 1,200 reviews, full-history LightGBM captures 678 observed defaults versus 637 for latest-month LightGBM: 41 additional cases, or +3.09 capture percentage points (paired 95% bootstrap interval 1.27–4.63). The GRU captures 671 and has no clear capacity advantage over LightGBM.

## Evidence limits

The 6,000-account reporting cohort was already examined. The dataset describes Taiwan accounts in 2005. Ordered-history evidence in one fitted GRU does not establish that recurrence is necessary, and the no-latest-delay subgroup is not a validated first-deterioration event study.

## Checks and changes

The existing v3 results and earlier verification remain unchanged. This review adds the practical 41-case interpretation and an explicit future-cohort decision; it does not refit or select another model.

## Next independent milestone

Freeze the next-study protocol and evaluate the full-history and latest-month controls on an independent later cohort, retaining exact capacity and subgroup definitions.

[Project report](../reports/PROJECT_REPORT.pdf) · [Prepared LinkedIn Projects entry](../reports/LINKEDIN_PROJECT_ENTRY.md)

The report and entry are portfolio summaries. Original experimental source version: `5cafe91cc65ba3a8aac2ad154d4aeedd40bf6dce`. The report records verification scope and distinguishes earlier numerical replays from this presentation/code review.
