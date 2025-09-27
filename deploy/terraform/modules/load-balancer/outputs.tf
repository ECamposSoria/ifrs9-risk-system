output "external_ip" {
  description = "External IP address reserved for the load balancer"
  value       = google_compute_global_address.external.address
}

output "certificate_name" {
  description = "Managed SSL certificate name"
  value       = length(google_compute_managed_ssl_certificate.managed) > 0 ? google_compute_managed_ssl_certificate.managed[0].name : null
}

output "certificate_expire_time" {
  description = "Expiration timestamp of the managed certificate"
  value       = length(google_compute_managed_ssl_certificate.managed) > 0 ? google_compute_managed_ssl_certificate.managed[0].expire_time : null
}

output "backend_bucket_names" {
  description = "Names of backend buckets associated with the load balancer"
  value       = { for key, bucket in google_compute_backend_bucket.backends : key => bucket.name }
}

output "http_forwarding_rule" {
  description = "Global forwarding rule for HTTP"
  value       = google_compute_global_forwarding_rule.http.id
}

output "https_forwarding_rule" {
  description = "Global forwarding rule for HTTPS"
  value       = length(google_compute_global_forwarding_rule.https) > 0 ? google_compute_global_forwarding_rule.https[0].id : null
}
