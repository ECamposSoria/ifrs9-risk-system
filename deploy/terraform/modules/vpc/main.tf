locals {
  network_name    = lower(replace("ifrs9-${var.environment}-core", "_", "-"))
  subnet_name     = "${local.network_name}-${var.region}-subnet"
  nat_router_name = "${local.network_name}-${var.region}-nat-router"
  nat_name        = "${local.network_name}-${var.region}-nat"
  firewall_tags   = distinct(concat(["ifrs9", var.environment], var.firewall_target_tags))
  egress_allow_rules = [
    {
      name               = "${local.network_name}-allow-google-apis"
      description        = "Permit Private Google Access VIP for Google APIs"
      destination_ranges = ["199.36.153.8/30"]
      priority           = 900
      ports = [
        {
          protocol = "tcp"
          ports    = ["443"]
        }
      ]
    },
    {
      name               = "${local.network_name}-allow-egress-https"
      description        = "Allow HTTPS egress for registries (gcr.io, *.pkg.dev, storage.googleapis.com, pypi.org)"
      destination_ranges = ["0.0.0.0/0"]
      priority           = 910
      ports = [
        {
          protocol = "tcp"
          ports    = ["443"]
        }
      ]
    },
    {
      name               = "${local.network_name}-allow-egress-http"
      description        = "Allow HTTP egress for package mirrors"
      destination_ranges = ["0.0.0.0/0"]
      priority           = 920
      ports = [
        {
          protocol = "tcp"
          ports    = ["80"]
        }
      ]
    }
  ]
}

resource "google_compute_network" "primary" {
  name                    = local.network_name
  project                 = var.project_id
  auto_create_subnetworks = false
  description             = "IFRS9 core VPC for ${var.environment}"
}

resource "google_compute_subnetwork" "primary" {
  name                     = local.subnet_name
  project                  = var.project_id
  region                   = var.region
  ip_cidr_range            = var.subnet_cidr
  network                  = google_compute_network.primary.id
  description              = "Primary subnet for IFRS9 ${var.environment}"
  private_ip_google_access = true

  secondary_ip_range {
    range_name    = "${local.network_name}-pods"
    ip_cidr_range = var.pod_ip_cidr
  }

  secondary_ip_range {
    range_name    = "${local.network_name}-services"
    ip_cidr_range = var.service_ip_cidr
  }

  log_config {
    aggregation_interval = var.subnet_flow_aggregation_interval
    flow_sampling        = var.subnet_flow_sampling
    metadata             = "INCLUDE_ALL_METADATA"
  }

  lifecycle {
    ignore_changes = [secondary_ip_range]

    precondition {
      condition     = var.enforce_autopilot_drain ? var.autopilot_active_nodes == 0 : true
      error_message = length(var.autopilot_active_node_names) > 0 ? format("Autopilot node drain still in progress for: %s", join(", ", var.autopilot_active_node_names)) : "Autopilot node drain still in progress."
    }
  }
}

resource "google_compute_router" "nat" {
  name    = local.nat_router_name
  project = var.project_id
  region  = var.region
  network = google_compute_network.primary.id

  description = "Router for Cloud NAT handling private egress"
}

resource "google_compute_router_nat" "nat" {
  name                                = local.nat_name
  project                             = var.project_id
  region                              = var.region
  router                              = google_compute_router.nat.name
  nat_ip_allocate_option              = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat  = "ALL_SUBNETWORKS_ALL_IP_RANGES"
  enable_endpoint_independent_mapping = true

  dynamic "log_config" {
    for_each = var.enable_nat_logging ? [1] : []
    content {
      enable = true
      filter = var.nat_log_filter
    }
  }
}

resource "google_compute_firewall" "egress_default_deny" {
  name               = "${local.network_name}-egress-deny"
  project            = var.project_id
  network            = google_compute_network.primary.name
  direction          = "EGRESS"
  priority           = 65534
  destination_ranges = ["0.0.0.0/0"]
  target_tags        = local.firewall_tags

  deny {
    protocol = "all"
  }
}

resource "google_compute_firewall" "egress_allow" {
  for_each = { for rule in local.egress_allow_rules : rule.name => rule }

  name               = each.key
  description        = each.value.description
  project            = var.project_id
  network            = google_compute_network.primary.name
  direction          = "EGRESS"
  priority           = lookup(each.value, "priority", 1000)
  destination_ranges = each.value.destination_ranges
  target_tags        = local.firewall_tags

  dynamic "allow" {
    for_each = each.value.ports
    content {
      protocol = allow.value.protocol
      ports    = lookup(allow.value, "ports", null)
    }
  }
}
