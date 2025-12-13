locals {
  sa_prefix         = substr(replace("ifrs9-${var.environment}-pf", "_", "-"), 0, 24)
  cloud_run_sa_id   = substr("${local.sa_prefix}-runjob", 0, 30)
  vertex_train_sa_id = substr("${local.sa_prefix}-vertex", 0, 30)
}

resource "google_service_account" "cloud_run_job" {
  project      = var.project_id
  account_id   = local.cloud_run_sa_id
  display_name = "IFRS9 portfolio Cloud Run job"

  depends_on = [google_project_service.required_apis["iam"]]
}

resource "google_project_iam_member" "cloud_run_log_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.cloud_run_job.email}"
}

resource "google_project_iam_member" "cloud_run_bigquery_job_user" {
  project = var.project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.cloud_run_job.email}"
}

resource "google_bigquery_dataset_iam_member" "cloud_run_raw_editor" {
  project    = var.project_id
  dataset_id = google_bigquery_dataset.raw.dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = "serviceAccount:${google_service_account.cloud_run_job.email}"
}

resource "google_storage_bucket_iam_member" "cloud_run_artifacts_admin" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.cloud_run_job.email}"
}

resource "google_service_account" "vertex_training" {
  count = var.enable_vertex_training_iam ? 1 : 0

  project      = var.project_id
  account_id   = local.vertex_train_sa_id
  display_name = "IFRS9 portfolio Vertex training"

  depends_on = [google_project_service.required_apis["iam"]]
}

resource "google_project_iam_member" "vertex_aiplatform_user" {
  count = var.enable_vertex_training_iam ? 1 : 0

  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.vertex_training[0].email}"

  depends_on = [google_project_service.required_apis["aiplatform"]]
}

resource "google_project_iam_member" "vertex_bigquery_job_user" {
  count = var.enable_vertex_training_iam ? 1 : 0

  project = var.project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.vertex_training[0].email}"
}

resource "google_bigquery_dataset_iam_member" "vertex_raw_viewer" {
  count = var.enable_vertex_training_iam ? 1 : 0

  project    = var.project_id
  dataset_id = google_bigquery_dataset.raw.dataset_id
  role       = "roles/bigquery.dataViewer"
  member     = "serviceAccount:${google_service_account.vertex_training[0].email}"
}

resource "google_storage_bucket_iam_member" "vertex_artifacts_admin" {
  count = var.enable_vertex_training_iam ? 1 : 0

  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.vertex_training[0].email}"
}

