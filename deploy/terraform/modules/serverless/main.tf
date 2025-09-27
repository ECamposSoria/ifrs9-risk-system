locals {
  name_prefix          = substr(replace("ifrs9-${var.environment}", "_", "-"), 0, 20)
  cloud_run_service_id = "${local.name_prefix}-orchestrator"
  workflow_name        = "${local.name_prefix}-workflow"
  scheduler_job_name   = "${local.name_prefix}-workflow-job"
  run_account_id       = substr("${local.name_prefix}-run", 0, 30)
  workflow_account_id  = substr("${local.name_prefix}-wf", 0, 30)
  scheduler_account_id = substr("${local.name_prefix}-sch", 0, 30)
  workflow_labels      = merge(var.labels, { component = "serverless-orchestration" })
}

resource "google_service_account" "run_executor" {
  project      = var.project_id
  account_id   = local.run_account_id
  display_name = "${title(var.environment)} serverless runner"
}

resource "google_service_account" "workflow_executor" {
  project      = var.project_id
  account_id   = local.workflow_account_id
  display_name = "${title(var.environment)} workflow orchestrator"
}

resource "google_service_account" "scheduler_invoker" {
  project      = var.project_id
  account_id   = local.scheduler_account_id
  display_name = "${title(var.environment)} workflow scheduler"
}

resource "google_cloud_run_service" "orchestrator" {
  name     = local.cloud_run_service_id
  location = var.region
  project  = var.project_id

  metadata {
    labels = merge(var.labels, {
      component = "serverless-orchestration"
    })
  }

  template {
    metadata {
      annotations = {
        "autoscaling.knative.dev/minScale" = "0"
        "run.googleapis.com/launch-stage"  = "GA"
      }
      labels = var.labels
    }
    spec {
      service_account_name = google_service_account.run_executor.email
      timeout_seconds      = var.cloud_run_timeout_seconds

      containers {
        image = var.cloud_run_image
        resources {
          limits = {
            cpu    = "1"
            memory = "512Mi"
          }
        }
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}

resource "google_cloud_run_service_iam_member" "workflow_invoker" {
  location = var.region
  project  = var.project_id
  service  = google_cloud_run_service.orchestrator.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.workflow_executor.email}"
}

resource "google_project_iam_member" "workflow_token_creator" {
  project = var.project_id
  role    = "roles/iam.serviceAccountTokenCreator"
  member  = "serviceAccount:${google_service_account.workflow_executor.email}"
}

resource "google_workflows_workflow" "orchestration" {
  project       = var.project_id
  name          = local.workflow_name
  region        = var.region
  description   = "Serverless orchestration workflow for IFRS9 tasks"
  service_account = google_service_account.workflow_executor.id
  call_log_level = "LOG_ERRORS_ONLY"
  labels         = local.workflow_labels
  user_env_vars = {
    RUN_SERVICE_URL = google_cloud_run_service.orchestrator.status[0].url
  }
  source_contents = <<-EOT
  main:
    params: [event]
    steps:
      - callCloudRun:
          call: http.post
          args:
            url: $${sys.get_env("RUN_SERVICE_URL")}
            auth:
              type: OIDC
            body:
              triggerPayload: $${event}
      - returnResult:
          return: $${callCloudRun.body}
  EOT
}

resource "google_project_iam_member" "scheduler_token_creator" {
  project = var.project_id
  role    = "roles/iam.serviceAccountTokenCreator"
  member  = "serviceAccount:${google_service_account.scheduler_invoker.email}"
}

resource "google_project_iam_member" "workflow_invoker" {
  project = var.project_id
  role    = "roles/workflows.invoker"
  member  = "serviceAccount:${google_service_account.scheduler_invoker.email}"
}

resource "google_cloud_scheduler_job" "workflow_trigger" {
  project = var.project_id
  region  = var.region
  name    = local.scheduler_job_name
  schedule    = var.schedule_cron
  time_zone   = "Etc/UTC"
  description = "Trigger IFRS9 workflow via Cloud Scheduler"
  attempt_deadline = "600s"

  http_target {
    http_method = "POST"
    uri         = "https://workflowexecutions.googleapis.com/v1/projects/${var.project_id}/locations/${var.region}/workflows/${google_workflows_workflow.orchestration.name}:execute"
    body        = base64encode(jsonencode({ argument = "{}" }))
    headers = {
      "Content-Type" = "application/json"
    }
    oidc_token {
      service_account_email = google_service_account.scheduler_invoker.email
    }
  }

  depends_on = [google_project_iam_member.workflow_invoker]
}
