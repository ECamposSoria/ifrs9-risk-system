# BigQuery Module - Data Warehouse Setup
# Creates BigQuery datasets and tables for IFRS9 data

terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

# Raw data dataset
resource "google_bigquery_dataset" "raw_dataset" {
  dataset_id    = "ifrs9_raw_${var.environment}"
  friendly_name = "IFRS9 Raw Data - ${title(var.environment)}"
  description   = "Raw loan and customer data for IFRS9 processing"
  location      = var.region
  
  default_encryption_configuration {
    kms_key_name = var.kms_key_id
  }
  
  labels = merge(var.labels, {
    data_classification = "raw"
    retention_policy   = "long_term"
  })
  
  # Data retention
  default_table_expiration_ms = var.data_retention_policies[var.environment].raw_data_days * 24 * 60 * 60 * 1000
  
  # Access control
  dynamic "access" {
    for_each = var.service_accounts
    content {
      role          = "WRITER"
      user_by_email = access.value
    }
  }
  
  access {
    role   = "OWNER"
    user_by_email = "serviceAccount:${var.project_id}@appspot.gserviceaccount.com"
  }
}

# Processed/staged data dataset
resource "google_bigquery_dataset" "processed_dataset" {
  dataset_id    = "ifrs9_processed_${var.environment}"
  friendly_name = "IFRS9 Processed Data - ${title(var.environment)}"
  description   = "Processed and validated loan data for IFRS9 calculations"
  location      = var.region
  
  default_encryption_configuration {
    kms_key_name = var.kms_key_id
  }
  
  labels = merge(var.labels, {
    data_classification = "processed"
    retention_policy   = "regulatory"
  })
  
  default_table_expiration_ms = var.data_retention_policies[var.environment].processed_data_days * 24 * 60 * 60 * 1000
  
  dynamic "access" {
    for_each = var.service_accounts
    content {
      role          = "WRITER"
      user_by_email = access.value
    }
  }
}

# Analytics dataset for dashboards
resource "google_bigquery_dataset" "analytics_dataset" {
  dataset_id    = "ifrs9_analytics_${var.environment}"
  friendly_name = "IFRS9 Analytics - ${title(var.environment)}"
  description   = "Analytics views and aggregated data for dashboards and reporting"
  location      = var.region
  
  default_encryption_configuration {
    kms_key_name = var.kms_key_id
  }
  
  labels = merge(var.labels, {
    data_classification = "analytics"
    retention_policy   = "business"
  })
  
  dynamic "access" {
    for_each = var.service_accounts
    content {
      role          = "READER"
      user_by_email = access.value
    }
  }
  
  # Additional access for dashboards
  dynamic "access" {
    for_each = length(var.dashboard_viewers_group) > 0 ? [var.dashboard_viewers_group] : []
    content {
      role           = "READER"
      group_by_email = access.value
    }
  }
}

# ML dataset for model training and inference
resource "google_bigquery_dataset" "ml_dataset" {
  dataset_id    = "ifrs9_ml_${var.environment}"
  friendly_name = "IFRS9 Machine Learning - ${title(var.environment)}"
  description   = "ML features, training data, and model artifacts"
  location      = var.region
  
  default_encryption_configuration {
    kms_key_name = var.kms_key_id
  }
  
  labels = merge(var.labels, {
    data_classification = "ml"
    retention_policy   = "model_lifecycle"
  })
  
  default_table_expiration_ms = var.data_retention_policies[var.environment].model_data_days * 24 * 60 * 60 * 1000
  
  dynamic "access" {
    for_each = var.service_accounts
    content {
      role          = "WRITER"
      user_by_email = access.value
    }
  }
}

# Raw loan data table
resource "google_bigquery_table" "loan_portfolio" {
  count     = var.enable_tables ? 1 : 0
  dataset_id = google_bigquery_dataset.raw_dataset.dataset_id
  table_id   = "loan_portfolio"
  
  description = "Raw loan portfolio data"
  
  # Partitioning for performance
  time_partitioning {
    type  = "DAY"
    field = "created_date"
    require_partition_filter = true
  }
  
  # Clustering for query optimization
  clustering = ["region", "producto_tipo", "provision_stage"]
  
  schema = var.enable_tables ? file("${path.module}/schemas/loan_portfolio.json") : null
  
  labels = var.labels
}

# Customer data table
resource "google_bigquery_table" "customers" {
  count     = var.enable_tables ? 1 : 0
  dataset_id = google_bigquery_dataset.raw_dataset.dataset_id
  table_id   = "customers"
  
  description = "Customer information and demographics"
  
  clustering = ["region", "customer_segment"]
  
  schema = var.enable_tables ? file("${path.module}/schemas/customers.json") : null
  
  labels = var.labels
}

# Economic indicators table
resource "google_bigquery_table" "economic_indicators" {
  count     = var.enable_tables ? 1 : 0
  dataset_id = google_bigquery_dataset.raw_dataset.dataset_id
  table_id   = "economic_indicators"
  
  description = "Macroeconomic indicators for scenario modeling"
  
  time_partitioning {
    type  = "MONTH"
    field = "indicator_date"
  }
  
  schema = var.enable_tables ? file("${path.module}/schemas/economic_indicators.json") : null
  
  labels = var.labels
}

# IFRS9 staging results table
resource "google_bigquery_table" "ifrs9_staging" {
  count     = var.enable_tables ? 1 : 0
  dataset_id = google_bigquery_dataset.processed_dataset.dataset_id
  table_id   = "ifrs9_staging_results"
  
  description = "IFRS9 staging classification results"
  
  time_partitioning {
    type  = "DAY"
    field = "calculation_date"
    require_partition_filter = true
  }
  
  clustering = ["provision_stage", "region"]
  
  schema = var.enable_tables ? file("${path.module}/schemas/ifrs9_staging.json") : null
  
  labels = var.labels
}

# ECL calculations table
resource "google_bigquery_table" "ecl_calculations" {
  count     = var.enable_tables ? 1 : 0
  dataset_id = google_bigquery_dataset.processed_dataset.dataset_id
  table_id   = "ecl_calculations"
  
  description = "Expected Credit Loss calculations"
  
  time_partitioning {
    type  = "DAY"
    field = "calculation_date"
    require_partition_filter = true
  }
  
  clustering = ["provision_stage", "calculation_method"]
  
  schema = var.enable_tables ? file("${path.module}/schemas/ecl_calculations.json") : null
  
  labels = var.labels
}

# ML features table
resource "google_bigquery_table" "ml_features" {
  count     = var.enable_tables ? 1 : 0
  dataset_id = google_bigquery_dataset.ml_dataset.dataset_id
  table_id   = "ml_features"
  
  description = "Feature engineering results for ML models"
  
  time_partitioning {
    type  = "DAY"
    field = "feature_date"
    require_partition_filter = true
  }
  
  clustering = ["feature_set_version", "model_type"]
  
  schema = var.enable_tables ? file("${path.module}/schemas/ml_features.json") : null
  
  labels = var.labels
}

# Model predictions table
resource "google_bigquery_table" "model_predictions" {
  count     = var.enable_tables ? 1 : 0
  dataset_id = google_bigquery_dataset.ml_dataset.dataset_id
  table_id   = "model_predictions"
  
  description = "ML model predictions and scores"
  
  time_partitioning {
    type  = "DAY"
    field = "prediction_date"
    require_partition_filter = true
  }
  
  clustering = ["model_name", "model_version"]
  
  schema = var.enable_tables ? file("${path.module}/schemas/model_predictions.json") : null
  
  labels = var.labels
}

# Create analytical views
resource "google_bigquery_routine" "create_analytics_views" {
  count          = var.enable_tables ? 1 : 0
  dataset_id      = google_bigquery_dataset.analytics_dataset.dataset_id
  routine_id      = "create_ifrs9_views"
  routine_type    = "PROCEDURE"
  language        = "SQL"
  
  definition_body = var.enable_tables ? file("${path.module}/sql/create_analytics_views.sql") : null
  
  description = "Creates all IFRS9 analytical views for dashboards"
}

# Scheduled queries for data processing
resource "google_bigquery_data_transfer_config" "daily_aggregation" {
  count          = var.enable_tables ? 1 : 0
  display_name   = "IFRS9 Daily Aggregation - ${title(var.environment)}"
  location       = var.region
  data_source_id = "scheduled_query"
  
  schedule = "every day 02:00"
  
  destination_dataset_id = google_bigquery_dataset.analytics_dataset.dataset_id
  
  params = {
    query            = var.enable_tables ? file("${path.module}/sql/daily_aggregation.sql") : ""
    write_disposition = "WRITE_TRUNCATE"
    use_legacy_sql    = false
  }
}

# Data quality monitoring table
resource "google_bigquery_table" "data_quality_metrics" {
  count     = var.enable_tables ? 1 : 0
  dataset_id = google_bigquery_dataset.analytics_dataset.dataset_id
  table_id   = "data_quality_metrics"
  
  description = "Data quality monitoring and validation results"
  
  time_partitioning {
    type  = "DAY"
    field = "check_date"
    require_partition_filter = true
  }
  
  schema = var.enable_tables ? file("${path.module}/schemas/data_quality_metrics.json") : null
  
  labels = var.labels
}
