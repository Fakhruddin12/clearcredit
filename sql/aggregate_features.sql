-- Build only attributes observable at the current application decision.
-- Paths are supplied by the Python runner; {{application_csv}}, {{bureau_csv}},
-- and {{previous_application_csv}} are safely substituted after path quoting.
WITH app AS (
    SELECT * FROM read_csv_auto('{{application_csv}}', nullstr=['', 'NA'])
), bureau AS (
    SELECT SK_ID_CURR,
           COUNT(*) FILTER (WHERE DAYS_CREDIT < 0) AS bureau_n_prior_accounts,
           AVG(AMT_CREDIT_SUM) FILTER (WHERE DAYS_CREDIT < 0) AS bureau_mean_credit_sum,
           SUM(AMT_CREDIT_SUM) FILTER (WHERE DAYS_CREDIT < 0) AS bureau_total_credit_sum,
           MAX(DAYS_CREDIT) FILTER (WHERE DAYS_CREDIT < 0) AS bureau_most_recent_days,
           AVG(CREDIT_DAY_OVERDUE) FILTER (WHERE DAYS_CREDIT < 0) AS bureau_mean_days_overdue,
           SUM(AMT_CREDIT_SUM_DEBT) FILTER (WHERE DAYS_CREDIT < 0) AS bureau_total_debt
    FROM read_csv_auto('{{bureau_csv}}', nullstr=['', 'NA'])
    GROUP BY SK_ID_CURR
), prior AS (
    SELECT SK_ID_CURR,
           COUNT(*) FILTER (WHERE DAYS_DECISION < 0) AS prior_n_applications,
           AVG(AMT_APPLICATION) FILTER (WHERE DAYS_DECISION < 0) AS prior_mean_requested,
           AVG(AMT_CREDIT) FILTER (WHERE DAYS_DECISION < 0) AS prior_mean_granted,
           SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Refused' AND DAYS_DECISION < 0 THEN 1 ELSE 0 END) AS prior_refusals,
           AVG(AMT_CREDIT / NULLIF(AMT_APPLICATION, 0)) FILTER (WHERE DAYS_DECISION < 0) AS prior_mean_grant_ratio
    FROM read_csv_auto('{{previous_application_csv}}', nullstr=['', 'NA'])
    GROUP BY SK_ID_CURR
)
SELECT app.* EXCLUDE (SK_ID_CURR), app.SK_ID_CURR,
       bureau.* EXCLUDE (SK_ID_CURR), prior.* EXCLUDE (SK_ID_CURR)
FROM app
LEFT JOIN bureau USING (SK_ID_CURR)
LEFT JOIN prior USING (SK_ID_CURR)
