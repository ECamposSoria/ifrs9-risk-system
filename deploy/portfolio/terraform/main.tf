terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.40"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

locals {
  required_api_services = {
    bigquery        = "bigquery.googleapis.com"
    iam             = "iam.googleapis.com"
    run             = "run.googleapis.com"
    storage         = "storage.googleapis.com"
    aiplatform      = "aiplatform.googleapis.com"
    artifactregistry = "artifactregistry.googleapis.com"
  }

  common_labels = merge(var.labels, {
    environment = var.environment
    project     = "ifrs9-risk-system"
    managed_by  = "terraform"
    stack       = "portfolio"
  })
}

resource "google_project_service" "required_apis" {
  for_each = local.required_api_services

  project = var.project_id
  service = each.value

  disable_on_destroy         = false
  disable_dependent_services = false
}

