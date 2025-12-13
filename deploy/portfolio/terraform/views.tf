locals {
  loan_portfolio_table = format("`%s.%s.loan_portfolio`", var.project_id, google_bigquery_dataset.raw.dataset_id)
  stage_label_expr = <<-SQL
  CASE provision_stage
    WHEN 1 THEN 'Stage 1'
    WHEN 2 THEN 'Stage 2'
    WHEN 3 THEN 'Stage 3'
    ELSE CAST(provision_stage AS STRING)
  END
  SQL
}

resource "google_bigquery_table" "ecl_by_stage" {
  project             = var.project_id
  dataset_id          = google_bigquery_dataset.analytics.dataset_id
  table_id            = "ecl_by_stage"
  deletion_protection = false

  view {
    use_legacy_sql = false
    query          = <<-SQL
    SELECT
      provision_stage,
      ANY_VALUE(${trimspace(local.stage_label_expr)}) AS provision_stage_label,
      COUNT(1) AS loan_count,
      SUM(loan_amount) AS total_exposure,
      AVG(loan_amount) AS avg_loan_amount,
      AVG(credit_score) AS avg_credit_score,
      AVG(days_past_due) AS avg_days_past_due,
      SUM(ead) AS total_ead,
      SUM(ecl) AS total_ecl
    FROM ${local.loan_portfolio_table}
    GROUP BY provision_stage
    SQL
  }
}

resource "google_bigquery_table" "geographic_distribution" {
  project             = var.project_id
  dataset_id          = google_bigquery_dataset.analytics.dataset_id
  table_id            = "geographic_distribution"
  deletion_protection = false

  view {
    use_legacy_sql = false
    query          = <<-SQL
    SELECT
      region,
      provision_stage,
      ANY_VALUE(${trimspace(local.stage_label_expr)}) AS provision_stage_label,
      COUNT(1) AS loan_count,
      SUM(loan_amount) AS total_exposure,
      AVG(credit_score) AS avg_credit_score
    FROM ${local.loan_portfolio_table}
    GROUP BY region, provision_stage
    SQL
  }
}

resource "google_bigquery_table" "product_analysis" {
  project             = var.project_id
  dataset_id          = google_bigquery_dataset.analytics.dataset_id
  table_id            = "product_analysis"
  deletion_protection = false

  view {
    use_legacy_sql = false
    query          = <<-SQL
    SELECT
      loan_type,
      provision_stage,
      ANY_VALUE(${trimspace(local.stage_label_expr)}) AS provision_stage_label,
      COUNT(1) AS loan_count,
      SUM(loan_amount) AS total_exposure,
      AVG(interest_rate) AS avg_interest_rate,
      AVG(term_months) AS avg_term_months
    FROM ${local.loan_portfolio_table}
    GROUP BY loan_type, provision_stage
    SQL
  }
}

resource "google_bigquery_table" "risk_metrics" {
  project             = var.project_id
  dataset_id          = google_bigquery_dataset.analytics.dataset_id
  table_id            = "risk_metrics"
  deletion_protection = false

  view {
    use_legacy_sql = false
    query          = <<-SQL
    SELECT
      COUNT(1) AS loan_count,
      SUM(loan_amount) AS total_exposure,
      AVG(credit_score) AS avg_credit_score,
      AVG(dti_ratio) AS avg_dti,
      AVG(ltv_ratio) AS avg_ltv,
      AVG(days_past_due) AS avg_days_past_due,
      SAFE_DIVIDE(SUM(CASE WHEN provision_stage = 3 THEN 1 ELSE 0 END), COUNT(1)) * 100.0 AS default_rate_pct,
      SUM(ead) AS total_ead,
      SUM(ecl) AS total_ecl
    FROM ${local.loan_portfolio_table}
    SQL
  }
}

resource "google_bigquery_table" "credit_score_bands" {
  project             = var.project_id
  dataset_id          = google_bigquery_dataset.analytics.dataset_id
  table_id            = "credit_score_bands"
  deletion_protection = false

  view {
    use_legacy_sql = false
    query          = <<-SQL
    SELECT
      CASE
        WHEN credit_score >= 900 THEN 'Excellent (900+)'
        WHEN credit_score >= 800 THEN 'Very Good (800-899)'
        WHEN credit_score >= 700 THEN 'Good (700-799)'
        WHEN credit_score >= 600 THEN 'Fair (600-699)'
        WHEN credit_score >= 500 THEN 'Poor (500-599)'
        ELSE 'Very Poor (<500)'
      END AS credit_band,
      provision_stage,
      ANY_VALUE(${trimspace(local.stage_label_expr)}) AS provision_stage_label,
      COUNT(1) AS loan_count,
      SUM(loan_amount) AS total_exposure,
      SUM(ecl) AS total_ecl
    FROM ${local.loan_portfolio_table}
    GROUP BY credit_band, provision_stage
    SQL
  }
}

