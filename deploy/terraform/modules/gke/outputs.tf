output "cluster_name" {
  description = "Name of the GKE cluster"
  value       = google_container_cluster.primary.name
}

output "cluster_id" {
  description = "Fully qualified cluster ID"
  value       = google_container_cluster.primary.id
}

output "cluster_endpoint" {
  description = "Cluster control plane endpoint"
  value       = google_container_cluster.primary.endpoint
  sensitive   = true
}

output "cluster_ca_certificate" {
  description = "Base64-encoded cluster CA certificate"
  value       = google_container_cluster.primary.master_auth[0].cluster_ca_certificate
  sensitive   = true
}

output "workload_identity_pool" {
  description = "Workload Identity pool configured for the cluster"
  value       = "${var.project_id}.svc.id.goog"
}

output "node_pool_name" {
  description = "Primary node pool name"
  value       = google_container_node_pool.primary.name
}

output "node_service_account" {
  description = "Service account email used by node pool"
  value       = var.node_service_account_email
}

output "service_catalog" {
  description = "Canonical Kubernetes service DNS entries used by downstream modules"
  value = {
    ifrs9_api    = "ifrs9-api.ifrs9.svc.cluster.local"
    ml_service   = "ifrs9-ml.ifrs9.svc.cluster.local"
    data_service = "ifrs9-data.ifrs9.svc.cluster.local"
  }
}
