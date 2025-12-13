resource "google_bigquery_dataset" "raw" {
  project    = var.project_id
  dataset_id = "ifrs9_raw_${var.environment}"
  location   = var.bigquery_location

  delete_contents_on_destroy = true

  labels = merge(local.common_labels, {
    data_classification = "raw"
  })

  depends_on = [google_project_service.required_apis["bigquery"]]
}

resource "google_bigquery_dataset" "analytics" {
  project    = var.project_id
  dataset_id = "ifrs9_analytics_${var.environment}"
  location   = var.bigquery_location

  delete_contents_on_destroy = true

  labels = merge(local.common_labels, {
    data_classification = "analytics"
  })

  depends_on = [google_project_service.required_apis["bigquery"]]
}

resource "google_bigquery_dataset" "ml" {
  project    = var.project_id
  dataset_id = "ifrs9_ml_${var.environment}"
  location   = var.bigquery_location

  delete_contents_on_destroy = true

  labels = merge(local.common_labels, {
    data_classification = "ml"
  })

  depends_on = [google_project_service.required_apis["bigquery"]]
}

# Placeholder table - Cloud Run job will truncate and reload data
resource "google_bigquery_table" "loan_portfolio" {
  project             = var.project_id
  dataset_id          = google_bigquery_dataset.raw.dataset_id
  table_id            = "loan_portfolio"
  deletion_protection = false

  schema = jsonencode([
    { name = "loan_id", type = "STRING", mode = "REQUIRED" },
    { name = "loan_amount", type = "FLOAT64", mode = "NULLABLE" },
    { name = "interest_rate", type = "FLOAT64", mode = "NULLABLE" },
    { name = "term_months", type = "INT64", mode = "NULLABLE" },
    { name = "credit_score", type = "INT64", mode = "NULLABLE" },
    { name = "dti_ratio", type = "FLOAT64", mode = "NULLABLE" },
    { name = "ltv_ratio", type = "FLOAT64", mode = "NULLABLE" },
    { name = "days_past_due", type = "INT64", mode = "NULLABLE" },
    { name = "loan_type", type = "STRING", mode = "NULLABLE" },
    { name = "region", type = "STRING", mode = "NULLABLE" },
    { name = "provision_stage", type = "INT64", mode = "NULLABLE" },
    { name = "pd", type = "FLOAT64", mode = "NULLABLE" },
    { name = "lgd", type = "FLOAT64", mode = "NULLABLE" },
    { name = "ead", type = "FLOAT64", mode = "NULLABLE" },
    { name = "ecl", type = "FLOAT64", mode = "NULLABLE" },
    { name = "created_at", type = "TIMESTAMP", mode = "NULLABLE" }
  ])

  labels = local.common_labels
}

