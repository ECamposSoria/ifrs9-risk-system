locals {
  environment_name = lower(replace("ifrs9-${var.environment}-composer", "_", "-"))
}

resource "google_composer_environment" "primary" {
  project = var.project_id
  name    = local.environment_name
  region  = var.region
  labels  = var.labels

  config {
    environment_size = var.environment_size

    node_config {
      network    = var.network_name
      subnetwork = var.subnet_name

      service_account = var.service_account_email
      tags            = var.node_tags
    }

    software_config {
      image_version            = var.image_version
      airflow_config_overrides = var.airflow_config
      env_variables            = var.env_variables
      pypi_packages            = var.pypi_packages
    }

    private_environment_config {
      connection_type                        = "VPC_PEERING"
      enable_private_endpoint                = var.enable_private_endpoint
      master_ipv4_cidr_block                 = var.master_ipv4_cidr
      cloud_sql_ipv4_cidr_block              = var.cloud_sql_ipv4_cidr
      cloud_composer_network_ipv4_cidr_block = var.composer_network_ipv4_cidr
      cloud_composer_connection_subnetwork   = var.connection_subnetwork
    }

    resilience_mode = var.resilience_mode

    workloads_config {
      scheduler {
        cpu        = var.scheduler_resources.cpu
        memory_gb  = var.scheduler_resources.memory_gb
        storage_gb = var.scheduler_resources.storage_gb
      }
      web_server {
        cpu        = var.web_server_resources.cpu
        memory_gb  = var.web_server_resources.memory_gb
        storage_gb = var.web_server_resources.storage_gb
      }
      worker {
        cpu        = var.worker_resources.cpu
        memory_gb  = var.worker_resources.memory_gb
        storage_gb = var.worker_resources.storage_gb
        min_count  = var.worker_resources.min_count
        max_count  = var.worker_resources.max_count
      }
    }
  }

  timeouts {
    create = "90m"
    update = "90m"
    delete = "60m"
  }
}
