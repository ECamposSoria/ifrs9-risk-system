locals {
  workbench_instance_name = lower(replace("ifrs9-${var.environment}-wb", "_", "-"))
}

resource "google_workbench_instance" "primary" {
  provider = google-beta

  project  = var.project_id
  location = var.zone
  name     = local.workbench_instance_name
  labels   = var.labels

  desired_state = var.desired_state

  gce_setup {
    machine_type = var.machine_type

    boot_disk {
      disk_type    = var.boot_disk.disk_type
      disk_size_gb = var.boot_disk.disk_size_gb
      kms_key      = var.boot_disk.kms_key
    }

    vm_image {
      project = var.vm_image.project
      family  = var.vm_image.family
    }

    network_interfaces {
      network  = var.network_self_link
      subnet   = var.subnet_self_link
      nic_type = "GVNIC"
    }

    service_accounts {
      email = var.service_account_email
    }

    shielded_instance_config {
      enable_secure_boot          = true
      enable_vtpm                 = true
      enable_integrity_monitoring = true
    }
  }

  instance_owners = var.instance_owners

  disable_proxy_access = var.disable_proxy_access

  timeouts {
    create = "120m"
    update = "120m"
    delete = "60m"
  }
}
