
-- Check row count
SELECT COUNT(*) AS total_records
FROM credit_records;

-- Inspect default distribution
SELECT
    default_next_month,
    COUNT(*) AS customers,
    ROUND(
        100.0 * COUNT(*) /
        (SELECT COUNT(*) FROM credit_records),
        2
    ) AS percentage
FROM credit_records
GROUP BY default_next_month;

-- Check credit limits and ages
SELECT
    MIN(AGE) AS min_age,
    MAX(AGE) AS max_age,
    MIN(LIMIT_BAL) AS min_credit_limit,
    MAX(LIMIT_BAL) AS max_credit_limit
FROM credit_records;

-- Average credit limit by target
SELECT
    default_next_month,
    ROUND(AVG(LIMIT_BAL), 2) AS avg_credit_limit
FROM credit_records
GROUP BY default_next_month;

-- Check missing values in key fields
SELECT
    SUM(CASE WHEN AGE IS NULL THEN 1 ELSE 0 END)
        AS missing_age,
    SUM(CASE WHEN LIMIT_BAL IS NULL THEN 1 ELSE 0 END)
        AS missing_limit,
    SUM(CASE WHEN default_next_month IS NULL
             THEN 1 ELSE 0 END) AS missing_target
FROM credit_records;
