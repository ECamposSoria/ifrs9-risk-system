DECLARE analytics_dataset STRING DEFAULT @@dataset_id;
DECLARE env_suffix STRING DEFAULT SPLIT(analytics_dataset, '_')[OFFSET(2)];
DECLARE processed_dataset STRING DEFAULT CONCAT('ifrs9_processed_', env_suffix);
DECLARE target_table STRING DEFAULT FORMAT('`%s.daily_stage_metrics`', analytics_dataset);
DECLARE source_table STRING DEFAULT FORMAT('`%s.ifrs9_staging_results`', processed_dataset);

EXECUTE IMMEDIATE FORMAT(
  """
  CREATE OR REPLACE TABLE %s AS
  SELECT
    calculation_date,
    provision_stage,
    COUNT(1) AS loan_count,
    SUM(expected_credit_loss) AS total_ecl,
    SUM(exposure_at_default) AS total_ead,
    CURRENT_TIMESTAMP() AS refreshed_at
  FROM %s
  GROUP BY calculation_date, provision_stage
  """,
  target_table,
  source_table
);

