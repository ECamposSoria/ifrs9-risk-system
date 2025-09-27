output "instance_name" {
  description = "Cloud SQL instance name"
  value       = local.instance_name
}

output "connection_name" {
  description = "Cloud SQL connection string"
  value       = local.connection_name
}

output "private_ip_address" {
  description = "Private IP placeholder"
  value       = "10.20.0.5"
}

output "database_names" {
  description = "Default databases created"
  value       = ["ifrs9_core"]
}
