-- Stored procedure to (re)build core analytics views
DECLARE analytics_dataset STRING DEFAULT @@dataset_id;
DECLARE env_suffix STRING DEFAULT SPLIT(analytics_dataset, '_')[OFFSET(2)];
DECLARE project_id STRING DEFAULT @@project_id;
DECLARE raw_dataset STRING DEFAULT CONCAT('ifrs9_raw_', env_suffix);
DECLARE processed_dataset STRING DEFAULT CONCAT('ifrs9_processed_', env_suffix);
DECLARE ml_dataset STRING DEFAULT CONCAT('ifrs9_ml_', env_suffix);

EXECUTE IMMEDIATE FORMAT(
  """
  CREATE OR REPLACE VIEW `%1$s.loan_stage_summary` AS
  SELECT
    provision_stage,
    COUNT(1) AS loan_count,
    SUM(expected_credit_loss) AS total_ecl,
    SUM(exposure_at_default) AS total_ead,
    CURRENT_TIMESTAMP() AS refreshed_at
  FROM `%2$s.ifrs9_staging_results`
  GROUP BY provision_stage
  """,
  analytics_dataset,
  processed_dataset
);

EXECUTE IMMEDIATE FORMAT(
  """
  CREATE OR REPLACE VIEW `%1$s.portfolio_region_summary` AS
  SELECT
    region,
    provision_stage,
    COUNT(1) AS loan_count,
    SUM(loan_amount) AS total_exposure,
    AVG(credit_score) AS average_credit_score
  FROM `%2$s.loan_portfolio`
  GROUP BY region, provision_stage
  """,
  analytics_dataset,
  raw_dataset
);

EXECUTE IMMEDIATE FORMAT(
  """
  CREATE OR REPLACE VIEW `%1$s.model_performance_snapshot` AS
  SELECT
    p.prediction_date,
    p.model_name,
    p.model_version,
    p.probability_default,
    p.predicted_stage,
    f.behaviour_score,
    f.utilisation_rate,
    f.delinquency_days
  FROM `%2$s.model_predictions` AS p
  LEFT JOIN `%3$s.ml_features` AS f
  USING (loan_id, feature_set_version)
  """,
  analytics_dataset,
  ml_dataset,
  ml_dataset
);
