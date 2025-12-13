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

