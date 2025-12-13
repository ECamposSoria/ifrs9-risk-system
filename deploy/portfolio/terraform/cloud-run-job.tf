resource "google_cloud_run_v2_job" "portfolio_batch" {
  name     = "ifrs9-portfolio-batch"
  location = var.cloud_run_region
  project  = var.project_id

  deletion_protection = false

  template {
    template {
      service_account = google_service_account.cloud_run_job.email
      max_retries     = 0
      timeout         = "1800s"

      containers {
        image = var.cloud_run_job_image

        env {
          name  = "IFRS9_RECORD_COUNT"
          value = tostring(var.ifrs9_record_count)
        }
        env {
          name  = "IFRS9_SEED"
          value = tostring(var.ifrs9_seed)
        }
        env {
          name  = "GCS_OUTPUT_BUCKET"
          value = google_storage_bucket.artifacts.name
        }
        env {
          name  = "GCS_OBJECT_PREFIX"
          value = var.gcs_object_prefix
        }
        env {
          name  = "BQ_PROJECT_ID"
          value = var.project_id
        }
        env {
          name  = "BQ_DATASET"
          value = google_bigquery_dataset.raw.dataset_id
        }
        env {
          name  = "BQ_TABLE"
          value = "loan_portfolio"
        }
        env {
          name  = "BQ_LOCATION"
          value = var.bigquery_location
        }
        env {
          name  = "BQ_WRITE_DISPOSITION"
          value = "WRITE_TRUNCATE"
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

  depends_on = [google_project_service.required_apis["run"]]
}

