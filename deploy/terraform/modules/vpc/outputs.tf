output "network_name" {
  description = "Name of the VPC network"
  value       = google_compute_network.primary.name
}

output "network_self_link" {
  description = "Self link for the VPC network"
  value       = google_compute_network.primary.self_link
}

output "subnet_name" {
  description = "Name of the primary subnet"
  value       = google_compute_subnetwork.primary.name
}

output "subnet_cidr" {
  description = "CIDR range allocated to the primary subnet"
  value       = google_compute_subnetwork.primary.ip_cidr_range
}

output "pod_secondary_range_name" {
  description = "Secondary range name reserved for GKE pods"
  value       = "${google_compute_network.primary.name}-pods"
}

output "service_secondary_range_name" {
  description = "Secondary range name reserved for GKE services"
  value       = "${google_compute_network.primary.name}-services"
}

output "subnet_self_link" {
  description = "Self link for the primary subnet"
  value       = google_compute_subnetwork.primary.self_link
}

output "nat_name" {
  description = "Cloud NAT resource name"
  value       = google_compute_router_nat.nat.name
}

output "nat_region" {
  description = "Region where Cloud NAT is deployed"
  value       = google_compute_router_nat.nat.region
}

output "nat_logging_enabled" {
  description = "Indicates whether NAT logging is enabled"
  value       = length(google_compute_router_nat.nat.log_config) > 0
}
