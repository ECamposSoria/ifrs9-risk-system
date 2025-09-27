resource "google_cloud_run_v2_job" "ifrs9_batch" {
  count    = var.enable_serverless_orchestration ? 1 : 0
  name     = "ifrs9-batch-job"
  location = var.serverless_region
  project  = var.project_id

  template {
    template {
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

      service_account = module.serverless_orchestration[0].service_accounts.workflow_executor
      max_retries     = 0
      timeout         = "1800s"
    }
  }

  depends_on = [module.serverless_orchestration]
}
