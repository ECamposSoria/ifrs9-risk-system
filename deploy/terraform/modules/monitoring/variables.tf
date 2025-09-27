variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "environment" {
  description = "Deployment environment"
  type        = string
}

variable "labels" {
  description = "Labels for monitoring resources"
  type        = map(string)
  default     = {}
}

variable "notification_emails" {
  description = "Email addresses for alert notifications"
  type        = list(string)
  default     = []
}

variable "audit_dataset_id" {
  description = "BigQuery dataset ID receiving audit log exports"
  type        = string
  default     = ""
}

variable "billing_account_id" {
  description = "Billing account ID used for project budgets"
  type        = string
  default     = ""
}

variable "budget_monthly_amount" {
  description = "Monthly budget amount"
  type        = number
  default     = 0
}

variable "budget_alert_spend_threshold" {
  description = "Spend threshold for budget alerts"
  type        = number
  default     = 0.8
}

variable "budget_currency_code" {
  description = "Currency code for the budget amount"
  type        = string
  default     = "USD"
}
