locals {
  lb_name               = lower(replace("ifrs9-${var.environment}-lb", "_", "-"))
  default_backend_key   = var.default_backend != null ? var.default_backend : element(keys(var.backend_services), 0)
  path_backends         = { for key, config in var.backend_services : key => config if length(config.paths == null ? [] : config.paths) > 0 }
  has_https_certificate = length(var.ssl_certificate_domains) > 0
}

resource "google_compute_global_address" "external" {
  name         = "${local.lb_name}-ip"
  project      = var.project_id
  description  = "Global IP for IFRS9 ${var.environment} external entrypoint"
  ip_version   = "IPV4"
  address_type = "EXTERNAL"
}

resource "google_compute_managed_ssl_certificate" "managed" {
  count = local.has_https_certificate ? 1 : 0

  name    = "${local.lb_name}-cert"
  project = var.project_id

  managed {
    domains = var.ssl_certificate_domains
  }
}

resource "google_compute_backend_bucket" "backends" {
  for_each = var.backend_services

  name        = "${local.lb_name}-${each.key}"
  bucket_name = each.value.bucket_name
  enable_cdn  = lookup(each.value, "enable_cdn", false)
  description = lookup(each.value, "description", "IFRS9 backend bucket ${each.key}")
  project     = var.project_id
}

resource "google_compute_url_map" "default" {
  name    = "${local.lb_name}-urlmap"
  project = var.project_id

  default_service = google_compute_backend_bucket.backends[local.default_backend_key].self_link

  dynamic "path_matcher" {
    for_each = length(local.path_backends) > 0 ? [1] : []
    content {
      name            = "ifrs9-paths"
      default_service = google_compute_backend_bucket.backends[local.default_backend_key].self_link

      dynamic "path_rule" {
        for_each = local.path_backends
        content {
          paths   = path_rule.value.paths == null ? ["/*"] : path_rule.value.paths
          service = google_compute_backend_bucket.backends[path_rule.key].self_link
        }
      }
    }
  }
}

resource "google_compute_url_map" "http_redirect" {
  name    = "${local.lb_name}-http-redirect"
  project = var.project_id

  default_url_redirect {
    https_redirect = true
    strip_query    = false
  }
}

resource "google_compute_target_http_proxy" "http" {
  name    = "${local.lb_name}-http-proxy"
  project = var.project_id
  url_map = google_compute_url_map.http_redirect.id
}

resource "google_compute_global_forwarding_rule" "http" {
  name       = "${local.lb_name}-http-rule"
  project    = var.project_id
  target     = google_compute_target_http_proxy.http.id
  port_range = "80"
  ip_address = google_compute_global_address.external.address
}

resource "google_compute_target_https_proxy" "https" {
  count = local.has_https_certificate ? 1 : 0

  name             = "${local.lb_name}-https-proxy"
  project          = var.project_id
  url_map          = google_compute_url_map.default.id
  ssl_certificates = [google_compute_managed_ssl_certificate.managed[0].id]
}

resource "google_compute_global_forwarding_rule" "https" {
  count = local.has_https_certificate ? 1 : 0

  name       = "${local.lb_name}-https-rule"
  project    = var.project_id
  target     = google_compute_target_https_proxy.https[0].id
  port_range = "443"
  ip_address = google_compute_global_address.external.address
}
