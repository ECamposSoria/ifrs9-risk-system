# IFRS9 portfolio deployment defaults (academic-ocean-472500-j4)

project_id = "academic-ocean-472500-j4"

region            = "southamerica-east1"
bigquery_location = "southamerica-east1"

cloud_run_region = "us-central1"
environment      = "staging"

# Build + push this image first (pinned tag; no :latest)
cloud_run_job_image = "gcr.io/academic-ocean-472500-j4/ifrs9-cloud-run-job:0.2.0"

ifrs9_record_count = 5000
ifrs9_seed         = 42

artifact_bucket_retention_days = 30
enable_vertex_training_iam     = true

