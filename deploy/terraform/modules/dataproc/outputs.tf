output "cluster_name" {
  description = "Name of the Dataproc cluster"
  value       = google_dataproc_cluster.primary.name
}

output "master_instance_name" {
  description = "Name of the Dataproc master instance"
  value       = try(google_dataproc_cluster.primary.cluster_config[0].master_config[0].instance_names[0], null)
}

output "spark_history_server_url" {
  description = "Spark history server URL"
  value       = try(google_dataproc_cluster.primary.cluster_config[0].endpoint_config[0].http_ports["Spark History Server"], null)
}
