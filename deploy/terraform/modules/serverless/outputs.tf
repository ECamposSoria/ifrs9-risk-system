output "cloud_run_service_url" {
  description = "URL of the Cloud Run service"
  value       = google_cloud_run_service.orchestrator.status[0].url
}

output "workflow_name" {
  description = "Name of the deployed Workflow"
  value       = google_workflows_workflow.orchestration.name
}

output "scheduler_job_name" {
  description = "Name of the Cloud Scheduler job"
  value       = google_cloud_scheduler_job.workflow_trigger.name
}

output "service_accounts" {
  description = "Service accounts used by the serverless orchestration stack"
  value = {
    run_executor       = google_service_account.run_executor.email
    workflow_executor  = google_service_account.workflow_executor.email
    scheduler_invoker  = google_service_account.scheduler_invoker.email
  }
}
