#!/usr/bin/env bash
set -euo pipefail
PROJECT="academic-ocean-472500-j4"

check() {
  local desc="$1"; shift
  echo "------"  
  echo "$desc"
  "$@" --project="$PROJECT"
}

check "Compute networks:" gcloud compute networks list
check "Compute subnetworks:" gcloud compute networks subnets list
check "Compute instances:" gcloud compute instances list
check "GKE clusters:" gcloud container clusters list
check "Dataproc clusters:" gcloud dataproc clusters list --region=global
check "Cloud SQL instances:" gcloud sql instances list
check "Cloud Run services (us-central1):" gcloud run services list --platform=managed --region=us-central1
check "Cloud Run jobs (us-central1):" gcloud run jobs list --region=us-central1
check "Cloud Functions (gen1):" gcloud functions list
check "Cloud Functions (gen2):" gcloud functions deploy --help > /dev/null # placeholder (no list command? skipping) 
