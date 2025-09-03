IFRS9 Kubernetes Deployment Guide

Overview
- 8 agents deployed via Helm chart `charts/ifrs9-agents`
- Observability stack via Helm chart `charts/ifrs9-ops` (Prometheus Operator, Grafana, Jaeger, EFK)
- Service mesh (Istio) enabled per-namespace for mTLS and traffic policies
- Config and Secrets via ConfigMaps, Secrets, and optional GCP Secret Manager CSI
- GKE-ready with StorageClass alias `standard-rwo` and Workload Identity annotations

Prerequisites
- Kubernetes v1.25+
- Helm v3.11+
- Istio or ASM installed cluster-wide (enable sidecar injection)
- ArgoCD installed for GitOps flow (optional but recommended)
- GKE: enable Workload Identity; provision GSA and bind to KSA

Build & Push Agent Images
- Use GitHub Actions pipeline `.github/workflows/ci-cd.yaml` or locally:
  - gcloud auth configure-docker
  - docker build -t REGION-docker.pkg.dev/PROJECT/ifrs9/<agent>:tag -f docker/agents/<agent>/Dockerfile .
  - docker push REGION-docker.pkg.dev/PROJECT/ifrs9/<agent>:tag

Install Agents
- helm upgrade --install ifrs9-agents charts/ifrs9-agents \
    --namespace ifrs9 --create-namespace \
    --set image.registry=REGION-docker.pkg.dev/PROJECT/ifrs9 \
    --set image.tag=latest

Install Observability
- helm dependency update charts/ifrs9-ops
- helm upgrade --install ifrs9-ops charts/ifrs9-ops \
    --namespace ifrs9

GKE Integration
- Apply StorageClass: kubectl apply -f k8s/gke/storageclass-standard-rwo.yaml
- Create KSA per agent with GSA annotation (edit `k8s/gke/serviceaccounts.yaml`):
  - kubectl apply -f k8s/gke/serviceaccounts.yaml
- Bind GSA to roles for GCS/BigQuery via Terraform (deploy/terraform)

Secrets via CSI (optional)
- Enable Secret Manager CSI Driver on GKE
- Update `k8s/gke/secretmanager-csi-example.yaml` with your project and secret names
- kubectl apply -f k8s/gke/secretmanager-csi-example.yaml

ArgoCD (GitOps)
- Apply AppProject and Applications:
  - kubectl apply -f argo/appproject-ifrs9.yaml
  - kubectl apply -f argo/application-agents.yaml
  - kubectl apply -f argo/application-ops.yaml

Security
- JWT enforced by services (`API_JWT_SECRET` and `API_JWT_AUDIENCE`)
- Basic rate limiting and security headers enabled by default
- Istio PeerAuthentication (STRICT mTLS) and namespace-level AuthorizationPolicy

Scaling & Health
- HPAs per agent (autoscaling/v2) with CPU utilization target
- Probes: /readyz and /healthz; Prometheus scraping via /metrics

