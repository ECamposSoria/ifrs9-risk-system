locals {
  cluster_name = lower(replace("ifrs9-${var.environment}-dp", "_", "-"))
}

resource "google_dataproc_autoscaling_policy" "primary" {
  count     = var.enable_autoscaling ? 1 : 0
  policy_id = "${local.cluster_name}-policy"
  project   = var.project_id
  location  = var.region

  basic_algorithm {
    yarn_config {
      graceful_decommission_timeout  = "3600s"
      scale_up_factor                = 0.6
      scale_down_factor              = 0.4
      scale_up_min_worker_fraction   = 0.2
      scale_down_min_worker_fraction = 0.2
    }
  }

  worker_config {
    min_instances = var.min_workers
    max_instances = var.max_workers
  }

  secondary_worker_config {
    min_instances = 0
    max_instances = max(0, var.max_workers - var.min_workers)
  }
}

resource "google_dataproc_cluster" "primary" {
  name    = local.cluster_name
  project = var.project_id
  region  = var.region

  labels = var.labels

  cluster_config {
    gce_cluster_config {
      subnetwork       = var.subnet_name
      service_account  = var.service_account_email
      internal_ip_only = true
    }

    master_config {
      num_instances = 1
      machine_type  = "e2-standard-4"

      disk_config {
        boot_disk_size_gb = var.master_disk_size_gb
        boot_disk_type    = "pd-standard"
      }
    }

    worker_config {
      num_instances = var.min_workers
      machine_type  = "e2-standard-4"

      disk_config {
        boot_disk_size_gb = var.worker_disk_size_gb
        boot_disk_type    = "pd-standard"
      }
    }

    lifecycle_config {
      idle_delete_ttl = "604800s"
    }

    dynamic "autoscaling_config" {
      for_each = var.enable_autoscaling ? [google_dataproc_autoscaling_policy.primary[0].id] : []
      content {
        policy_uri = autoscaling_config.value
      }
    }
  }
}
