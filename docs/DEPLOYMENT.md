# IFRS9 Risk System - Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying the IFRS9 Risk System to Google Cloud Platform. The deployment supports multiple environments (dev, staging, prod) with automated infrastructure provisioning and application deployment.

## Prerequisites

### 1. Required Tools

Install the following tools on your local machine:

```bash
# Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init

# Terraform
wget https://releases.hashicorp.com/terraform/1.6.0/terraform_1.6.0_linux_amd64.zip
unzip terraform_1.6.0_linux_amd64.zip
sudo mv terraform /usr/local/bin/

# kubectl
gcloud components install kubectl

# Docker
sudo apt-get update
sudo apt-get install docker.io
sudo usermod -aG docker $USER

# Poetry (Python dependency management)
curl -sSL https://install.python-poetry.org | python3 -
```

### 2. GCP Project Setup

```bash
# Set project variables
export PROJECT_ID="your-ifrs9-project-id"
export BILLING_ACCOUNT_ID="your-billing-account-id"
export REGION="europe-west1"
export ENVIRONMENT="prod"  # or "dev", "staging"

# Create GCP project
gcloud projects create $PROJECT_ID
gcloud config set project $PROJECT_ID

# Link billing account
gcloud beta billing projects link $PROJECT_ID \
  --billing-account $BILLING_ACCOUNT_ID

# Enable required APIs
gcloud services enable \
  compute.googleapis.com \
  container.googleapis.com \
  bigquery.googleapis.com \
  storage.googleapis.com \
  dataproc.googleapis.com \
  aiplatform.googleapis.com \
  composer.googleapis.com \
  monitoring.googleapis.com \
  logging.googleapis.com \
  secretmanager.googleapis.com \
  cloudkms.googleapis.com \
  iam.googleapis.com
```

### 3. Authentication Setup

```bash
# Create service account for Terraform
gcloud iam service-accounts create terraform-sa \
  --display-name "Terraform Service Account"

# Grant necessary permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member serviceAccount:terraform-sa@$PROJECT_ID.iam.gserviceaccount.com \
  --role roles/owner

# Download service account key
gcloud iam service-accounts keys create terraform-sa-key.json \
  --iam-account terraform-sa@$PROJECT_ID.iam.gserviceaccount.com

# Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/terraform-sa-key.json"
```

## Infrastructure Deployment

### 1. Terraform State Backend

```bash
# Create bucket for Terraform state
export TF_STATE_BUCKET="${PROJECT_ID}-terraform-state"

gsutil mb -l $REGION gs://$TF_STATE_BUCKET
gsutil versioning set on gs://$TF_STATE_BUCKET

# Enable object lifecycle management
cat > lifecycle.json << EOF
{
  "rule": [
    {
      "action": {"type": "Delete"},
      "condition": {"numNewerVersions": 5}
    }
  ]
}
EOF

gsutil lifecycle set lifecycle.json gs://$TF_STATE_BUCKET
```

### 2. Environment Configuration

Create environment-specific variable files:

```bash
# Create terraform.tfvars for your environment
cat > deploy/terraform/environments/${ENVIRONMENT}.tfvars << EOF
# Core project configuration
project_id = "$PROJECT_ID"
region = "$REGION"
environment = "$ENVIRONMENT"
terraform_state_bucket = "$TF_STATE_BUCKET"

# Resource sizing for $ENVIRONMENT
dataproc_min_workers = 2
dataproc_max_workers = 8

gke_node_config = {
  machine_type   = "e2-standard-4"
  disk_size_gb   = 100
  disk_type      = "pd-ssd"
  min_node_count = 2
  max_node_count = 6
  preemptible    = false
}

# Monitoring configuration
monitoring_notification_emails = [
  "alerts@yourbank.com",
  "devops@yourbank.com"
]

# SSL certificate domains (update with your domains)
ssl_certificate_domains = [
  "ifrs9-api.yourbank.com",
  "ifrs9-dashboards.yourbank.com"
]

# Network security
network_security = {
  allowed_ip_ranges     = ["10.0.0.0/8", "172.16.0.0/12"]
  enable_private_nodes  = true
  enable_network_policy = true
  authorized_networks   = [
    {
      cidr_block   = "203.0.113.0/24"
      display_name = "Corporate Network"
    }
  ]
}
EOF
```

### 3. Deploy Infrastructure

```bash
cd deploy/terraform

# Initialize Terraform
terraform init \
  -backend-config="bucket=$TF_STATE_BUCKET" \
  -backend-config="prefix=terraform/state/$ENVIRONMENT"

# Plan deployment
terraform plan \
  -var-file="environments/${ENVIRONMENT}.tfvars" \
  -out=tfplan

# Review the plan and apply
terraform apply tfplan
```

### 4. Verify Infrastructure

```bash
# Check cluster access
gcloud container clusters get-credentials \
  ifrs9-gke-cluster-$ENVIRONMENT \
  --region $REGION \
  --project $PROJECT_ID

kubectl get nodes
kubectl get namespaces

# Verify BigQuery datasets
bq ls --project_id=$PROJECT_ID

# Check Cloud Storage buckets
gsutil ls -p $PROJECT_ID
```

## Application Deployment

### 1. Build Docker Images

```bash
# Set image variables
export DOCKER_REGISTRY="gcr.io/$PROJECT_ID"
export IMAGE_TAG="v$(date +%Y%m%d)-$(git rev-parse --short HEAD)"

# Build API image
docker build \
  -f docker/production/Dockerfile.api \
  -t $DOCKER_REGISTRY/ifrs9-api:$IMAGE_TAG \
  --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
  --build-arg VERSION="1.0.0" \
  --build-arg COMMIT_SHA="$(git rev-parse HEAD)" \
  .

# Build ML service image
docker build \
  -f docker/production/Dockerfile.ml \
  -t $DOCKER_REGISTRY/ifrs9-ml:$IMAGE_TAG \
  .

# Build data processing image
docker build \
  -f docker/production/Dockerfile.spark \
  -t $DOCKER_REGISTRY/ifrs9-spark:$IMAGE_TAG \
  .

# Push images to registry
docker push $DOCKER_REGISTRY/ifrs9-api:$IMAGE_TAG
docker push $DOCKER_REGISTRY/ifrs9-ml:$IMAGE_TAG
docker push $DOCKER_REGISTRY/ifrs9-spark:$IMAGE_TAG
```

### 2. Configure Kubernetes Secrets

```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Create secrets for database and API keys
kubectl create secret generic ifrs9-secrets \
  --namespace=ifrs9-risk-system \
  --from-literal=database-url="postgresql://user:password@host:port/db" \
  --from-literal=api-secret-key="$(openssl rand -base64 32)" \
  --from-file=bigquery-credentials=./terraform-sa-key.json \
  --from-file=vertex-ai-credentials=./terraform-sa-key.json

# Create TLS certificates (if using custom domains)
kubectl create secret tls ifrs9-tls-secret \
  --namespace=ifrs9-risk-system \
  --cert=path/to/tls.crt \
  --key=path/to/tls.key
```

### 3. Deploy Application Configuration

```bash
# Update ConfigMap with environment-specific values
envsubst < k8s/configmaps/ifrs9-config.yaml | kubectl apply -f -

# Deploy persistent volumes
kubectl apply -f k8s/storage/

# Deploy services
kubectl apply -f k8s/services/

# Deploy applications
envsubst < k8s/deployments/ifrs9-api.yaml | kubectl apply -f -
envsubst < k8s/deployments/ifrs9-ml.yaml | kubectl apply -f -
envsubst < k8s/deployments/ifrs9-worker.yaml | kubectl apply -f -

# Deploy ingress
envsubst < k8s/ingress/ifrs9-ingress.yaml | kubectl apply -f -
```

### 4. Verify Deployment

```bash
# Check pod status
kubectl get pods -n ifrs9-risk-system

# Check service endpoints
kubectl get services -n ifrs9-risk-system

# Check ingress
kubectl get ingress -n ifrs9-risk-system

# View logs
kubectl logs -f deployment/ifrs9-api -n ifrs9-risk-system

# Test API health
curl -k https://your-domain.com/health
```

## Data Pipeline Deployment

### 1. Upload Airflow DAGs

```bash
# Get Composer environment details
export COMPOSER_BUCKET=$(terraform output -raw composer_gcs_bucket)

# Upload DAGs
gsutil -m cp -r airflow/dags/* gs://$COMPOSER_BUCKET/dags/

# Upload plugins
gsutil -m cp -r airflow/plugins/* gs://$COMPOSER_BUCKET/plugins/

# Set Airflow variables
gcloud composer environments run ifrs9-composer-$ENVIRONMENT \
  --location $REGION \
  variables set -- \
  PROJECT_ID $PROJECT_ID \
  ENVIRONMENT $ENVIRONMENT \
  BIGQUERY_DATASET ifrs9_processed_$ENVIRONMENT
```

### 2. Configure Data Sources

```bash
# Create BigQuery views
bq query --use_legacy_sql=false --project_id=$PROJECT_ID \
  < dashboards/bigquery_views.sql

# Upload sample data for testing
gsutil cp data/sample/* gs://$PROJECT_ID-raw-data-$ENVIRONMENT/loans/

# Test data processing pipeline
gcloud composer environments run ifrs9-composer-$ENVIRONMENT \
  --location $REGION \
  dags trigger -- daily_ifrs9_processing
```

## Monitoring Deployment

### 1. Deploy Prometheus and Grafana

```bash
# Add Helm repositories
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

# Create monitoring namespace
kubectl create namespace monitoring

# Install Prometheus
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --values k8s/monitoring/prometheus-values.yaml

# Install Grafana dashboards
kubectl apply -f k8s/monitoring/grafana-dashboards/
```

### 2. Configure Alerting

```bash
# Create alert rules
kubectl apply -f k8s/monitoring/alert-rules.yaml

# Configure notification channels
kubectl create secret generic alertmanager-config \
  --namespace monitoring \
  --from-file=alertmanager.yml=k8s/monitoring/alertmanager.yml

# Restart Alertmanager
kubectl rollout restart deployment/prometheus-alertmanager \
  --namespace monitoring
```

## Security Hardening

### 1. Network Policies

```bash
# Apply network policies
kubectl apply -f k8s/security/network-policies.yaml

# Verify network policies
kubectl get networkpolicies -n ifrs9-risk-system
```

### 2. Pod Security Policies

```bash
# Apply pod security policies
kubectl apply -f k8s/security/pod-security-policies.yaml

# Verify PSP
kubectl get podsecuritypolicy
```

### 3. RBAC Configuration

```bash
# Apply RBAC rules
kubectl apply -f k8s/security/rbac.yaml

# Verify service accounts
kubectl get serviceaccounts -n ifrs9-risk-system
```

## Environment-Specific Deployments

### Development Environment

```bash
export ENVIRONMENT="dev"

# Deploy with development settings
terraform apply \
  -var-file="environments/dev.tfvars" \
  -var="enable_deletion_protection=false" \
  -var="enable_binary_logging=false"

# Use smaller resource allocations
export DOCKER_REGISTRY="gcr.io/$PROJECT_ID"
export IMAGE_TAG="dev-latest"
```

### Staging Environment

```bash
export ENVIRONMENT="staging"

# Deploy with staging settings
terraform apply \
  -var-file="environments/staging.tfvars" \
  -var="enable_deletion_protection=true"

# Enable additional monitoring
kubectl apply -f k8s/monitoring/staging-alerts.yaml
```

### Production Environment

```bash
export ENVIRONMENT="prod"

# Deploy with production settings
terraform apply \
  -var-file="environments/prod.tfvars" \
  -var="enable_deletion_protection=true" \
  -var="enable_binary_logging=true"

# Enable all security features
kubectl apply -f k8s/security/
kubectl apply -f k8s/monitoring/production-alerts.yaml
```

## Post-Deployment Verification

### 1. Functional Testing

```bash
# Run API tests
cd tests/integration
python -m pytest test_api_endpoints.py -v

# Run data pipeline tests
python -m pytest test_data_pipeline.py -v

# Run ML model tests
python -m pytest test_ml_models.py -v
```

### 2. Performance Testing

```bash
# Run load tests
cd tests/performance
python -m pytest test_load_performance.py -v

# Monitor resource usage
kubectl top pods -n ifrs9-risk-system
kubectl top nodes
```

### 3. Security Validation

```bash
# Run security scans
trivy image $DOCKER_REGISTRY/ifrs9-api:$IMAGE_TAG

# Check network policies
kubectl get networkpolicies -n ifrs9-risk-system -o yaml

# Verify encryption at rest
gcloud kms keys list --location=$REGION --keyring=ifrs9-keyring
```

## Maintenance and Updates

### 1. Rolling Updates

```bash
# Update application images
kubectl set image deployment/ifrs9-api \
  ifrs9-api=$DOCKER_REGISTRY/ifrs9-api:$NEW_IMAGE_TAG \
  -n ifrs9-risk-system

# Monitor rollout
kubectl rollout status deployment/ifrs9-api -n ifrs9-risk-system

# Rollback if necessary
kubectl rollout undo deployment/ifrs9-api -n ifrs9-risk-system
```

### 2. Infrastructure Updates

```bash
# Update Terraform configuration
terraform plan -var-file="environments/${ENVIRONMENT}.tfvars"
terraform apply -var-file="environments/${ENVIRONMENT}.tfvars"

# Update GKE cluster
gcloud container clusters upgrade ifrs9-gke-cluster-$ENVIRONMENT \
  --region $REGION
```

### 3. Backup and Restore

```bash
# Backup BigQuery datasets
bq extract --destination_format=AVRO \
  ifrs9_processed_$ENVIRONMENT.loan_portfolio \
  gs://$PROJECT_ID-backups-$ENVIRONMENT/bigquery/loan_portfolio_$(date +%Y%m%d).avro

# Backup Kubernetes configurations
kubectl get all -n ifrs9-risk-system -o yaml > k8s-backup-$(date +%Y%m%d).yaml

# Create etcd snapshot (GKE managed)
gcloud container clusters create-snapshot ifrs9-gke-cluster-$ENVIRONMENT \
  --region $REGION
```

## Troubleshooting

### 1. Common Issues

**Pod Startup Issues**:
```bash
kubectl describe pod <pod-name> -n ifrs9-risk-system
kubectl logs <pod-name> -n ifrs9-risk-system --previous
```

**Service Discovery Issues**:
```bash
kubectl get endpoints -n ifrs9-risk-system
kubectl get services -n ifrs9-risk-system
nslookup ifrs9-api.ifrs9-risk-system.svc.cluster.local
```

**Storage Issues**:
```bash
kubectl get pvc -n ifrs9-risk-system
kubectl describe pvc <pvc-name> -n ifrs9-risk-system
```

### 2. Performance Issues

**High CPU Usage**:
```bash
kubectl top pods -n ifrs9-risk-system
kubectl get hpa -n ifrs9-risk-system
```

**Memory Issues**:
```bash
kubectl describe node <node-name>
kubectl get events --sort-by=.metadata.creationTimestamp
```

### 3. Network Issues

**Connectivity Problems**:
```bash
kubectl exec -it <pod-name> -n ifrs9-risk-system -- nslookup google.com
kubectl get networkpolicies -n ifrs9-risk-system
```

## Cleanup

### 1. Remove Applications

```bash
kubectl delete namespace ifrs9-risk-system
kubectl delete namespace monitoring
```

### 2. Destroy Infrastructure

```bash
cd deploy/terraform
terraform destroy -var-file="environments/${ENVIRONMENT}.tfvars"
```

### 3. Delete GCP Project

```bash
gcloud projects delete $PROJECT_ID
```

This deployment guide ensures a reliable, secure, and scalable deployment of the IFRS9 Risk System across multiple environments.