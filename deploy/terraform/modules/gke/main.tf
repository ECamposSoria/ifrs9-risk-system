locals {
  cluster_name   = lower(replace("ifrs9-${var.environment}-gke", "_", "-"))
  node_pool_name = "${local.cluster_name}-primary"
  wi_pool        = "${var.project_id}.svc.id.goog"
}

resource "google_container_cluster" "primary" {
  name     = local.cluster_name
  location = var.region
  project  = var.project_id

  description = "IFRS9 ${var.environment} GKE cluster"

  remove_default_node_pool = true
  initial_node_count       = 1

  network    = var.network_name
  subnetwork = var.subnet_name

  enable_shielded_nodes = true

  release_channel {
    channel = var.release_channel
  }

  workload_identity_config {
    workload_pool = local.wi_pool
  }

  ip_allocation_policy {
    cluster_secondary_range_name  = var.pod_secondary_range_name
    services_secondary_range_name = var.service_secondary_range_name
  }

  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = var.enable_private_endpoint
    master_ipv4_cidr_block  = var.master_ipv4_cidr
  }

  dynamic "master_authorized_networks_config" {
    for_each = length(var.authorized_networks) > 0 ? [var.authorized_networks] : []
    content {
      dynamic "cidr_blocks" {
        for_each = master_authorized_networks_config.value
        content {
          cidr_block   = cidr_blocks.value.cidr_block
          display_name = coalesce(cidr_blocks.value.display_name, "authorized-${replace(cidr_blocks.value.cidr_block, "/", "-")}")
        }
      }
    }
  }

  addons_config {
    gke_backup_agent_config {
      enabled = var.enable_backup_agent
    }

    http_load_balancing {
      disabled = var.disable_http_load_balancing
    }

    horizontal_pod_autoscaling {
      disabled = false
    }
  }

  network_policy {
    enabled  = var.enable_network_policy
    provider = "CALICO"
  }

  resource_labels = var.labels

  maintenance_policy {
    daily_maintenance_window {
      start_time = var.maintenance_window_start
    }
  }

  logging_config {
    enable_components = var.logging_components
  }

  monitoring_config {
    enable_components = var.monitoring_components
  }

  deletion_protection = var.enable_deletion_protection

  lifecycle {
    ignore_changes = [maintenance_policy]
  }
}

resource "google_container_node_pool" "primary" {
  name     = local.node_pool_name
  project  = var.project_id
  location = var.region
  cluster  = google_container_cluster.primary.name

  autoscaling {
    min_node_count = var.node_config.min_node_count
    max_node_count = var.node_config.max_node_count
  }

  node_config {
    machine_type = var.node_config.machine_type
    disk_size_gb = var.node_config.disk_size_gb
    disk_type    = var.node_config.disk_type
    preemptible  = var.node_config.preemptible

    image_type = coalesce(try(var.node_config.image_type, null), "COS_CONTAINERD")

    service_account = var.node_service_account_email
    oauth_scopes    = []

    labels = merge(var.node_labels, {
      environment = var.environment
    })

    tags = distinct(concat(var.node_tags, ["ifrs9", var.environment]))

    metadata = merge({
      disable-legacy-endpoints = "true"
    }, var.node_metadata)

    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  upgrade_settings {
    max_surge       = var.upgrade_max_surge
    max_unavailable = var.upgrade_max_unavailable
  }

  timeouts {
    create = "40m"
    update = "60m"
    delete = "40m"
  }
}
