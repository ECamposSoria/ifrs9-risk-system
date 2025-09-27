locals {
  instance_name   = lower(replace("ifrs9-${var.environment}-sql", "_", "-"))
  connection_name = "${var.project_id}:${var.region}:${local.instance_name}"
}

resource "null_resource" "cloud_sql_placeholder" {
  triggers = {
    project  = var.project_id
    instance = local.instance_name
    network  = var.network_name
    db_tier  = var.tier
  }
}
