resource "google_service_account" "ifrs9_batch_runner" {
  project      = var.project_id
  account_id   = "ifrs9-staging-job"
  display_name = "IFRS9 staging batch runner"
}

resource "google_project_iam_member" "ifrs9_batch_runner_token_creator" {
  project = var.project_id
  role    = "roles/iam.serviceAccountTokenCreator"
  member  = "serviceAccount:${google_service_account.ifrs9_batch_runner.email}"
}

resource "google_cloud_run_v2_job" "ifrs9_batch" {
  name     = "ifrs9-batch-job"
  location = var.serverless_region
  project  = var.project_id

  template {
    template {
      service_account = google_service_account.ifrs9_batch_runner.email
      max_retries     = 0
      timeout         = "1800s"

      containers {
        image = var.serverless_cloud_run_image
        env {
          name  = "IFRS9_RECORD_COUNT"
          value = tostring(var.ifrs9_batch_job.record_count)
        }
        env {
          name  = "IFRS9_SEED"
          value = tostring(var.ifrs9_batch_job.seed)
        }
        env {
          name  = "GCS_OUTPUT_BUCKET"
          value = var.ifrs9_batch_job.gcs_bucket
        }
        env {
          name  = "GCS_OBJECT_PREFIX"
          value = var.ifrs9_batch_job.gcs_prefix
        }
        resources {
          limits = {
            cpu    = "1"
            memory = "512Mi"
          }
        }
      }
    }
  }
}
