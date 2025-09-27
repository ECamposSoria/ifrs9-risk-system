locals {
  name_prefix = substr(replace("ifrs9-${var.environment}", "_", "-"), 0, 28)
  accounts = {
    dataproc = "${local.name_prefix}-dataproc"
    composer = "${local.name_prefix}-composer"
    vertex   = "${local.name_prefix}-vertex"
    gke      = "${local.name_prefix}-gke"
  }
  role_map = {
    dataproc = [
      "roles/dataproc.editor",
      "roles/dataproc.worker",
      "roles/storage.objectAdmin",
      "roles/logging.logWriter",
      "roles/monitoring.metricWriter"
    ]
    composer = [
      "roles/composer.worker",
      "roles/iam.serviceAccountUser"
    ]
    vertex = [
      "roles/aiplatform.user",
      "roles/notebooks.admin"
    ]
    gke = [
      "roles/logging.logWriter",
      "roles/monitoring.metricWriter",
      "roles/container.nodeServiceAccount",
      "roles/artifactregistry.reader"
    ]
  }
}

resource "google_service_account" "service_accounts" {
  for_each     = local.accounts
  account_id   = each.value
  project      = var.project_id
  display_name = "IFRS9 ${var.environment} ${each.key} service account"
}

locals {
  flattened_role_bindings = flatten([
    for key, roles in local.role_map : [
      for role in roles : {
        service_key = key
        role        = role
        member      = "serviceAccount:${google_service_account.service_accounts[key].email}"
      }
    ]
  ])
}

resource "google_project_iam_member" "service_account_roles" {
  for_each = { for binding in local.flattened_role_bindings :
  "${binding.service_key}-${replace(binding.role, "/", "-")}" => binding }

  project = var.project_id
  role    = each.value.role
  member  = each.value.member
}
