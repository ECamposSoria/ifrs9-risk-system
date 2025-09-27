# IFRS9 staging environment configuration
project_id             = "academic-ocean-472500-j4"
region                 = "southamerica-east1"
vertex_region          = "southamerica-east1"
vertex_zone            = "southamerica-east1-b"
environment            = "staging"
terraform_state_bucket = "ifrs9-terraform-state-staging"

# Dataproc worker bounds aligned to staging defaults
dataproc_min_workers = 2
dataproc_max_workers = 6

# GKE node scaling tuned for staging footprint
gke_node_config = {
  machine_type   = "e2-standard-2"
  disk_size_gb   = 80
  disk_type      = "pd-standard"
  min_node_count = 1
  max_node_count = 2
  preemptible    = false
}

# Disable always-on services for personal staging usage
enable_gke                      = false
enable_composer                 = false
enable_dataproc                 = false
enable_backup_services          = false
enable_serverless_orchestration = true
serverless_region               = "us-central1"
enable_bigquery_tables          = true
serverless_cloud_run_image      = "gcr.io/academic-ocean-472500-j4/ifrs9-cloud-run-job:0.1.0"
ifrs9_batch_job = {
  record_count = 5000
  seed         = 42
  gcs_bucket   = "ifrs9-batch-artifacts"
  gcs_prefix   = "ifrs9-batch"
}

# Keep optional overrides minimal until staging review completes
monitoring_notification_emails = []
ssl_certificate_domains        = []
