output "key_ring_id" {
  description = "Identifier for the key ring"
  value       = google_kms_key_ring.core.id
}

output "storage_key_id" {
  description = "Crypto key ID for Cloud Storage encryption"
  value       = google_kms_crypto_key.storage.id
}

output "bigquery_key_id" {
  description = "Crypto key ID for BigQuery encryption"
  value       = google_kms_crypto_key.bigquery.id
}

output "compute_key_id" {
  description = "Crypto key ID for compute resources"
  value       = google_kms_crypto_key.compute.id
}
