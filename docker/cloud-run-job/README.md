# IFRS9 Cloud Run Batch Job

This container image packages a lightweight IFRS9 batch task that:

1. Generates a synthetic loan portfolio using the existing `DataGenerator` utilities.
2. Builds a stage-level summary (loan counts and exposure) using pandas.
3. Writes Parquet/JSON artefacts locally and optionally uploads them to Cloud Storage.

## Build

```bash
docker build -t gcr.io/PROJECT_ID/ifrs9-cloud-run-job:0.1.0 \
  -f docker/cloud-run-job/Dockerfile .
```

## Publish

```bash
docker push gcr.io/PROJECT_ID/ifrs9-cloud-run-job:0.1.0
```

## Runtime Configuration

| Variable | Description | Default |
| --- | --- | --- |
| `IFRS9_RECORD_COUNT` | Number of synthetic loans to generate | `5000` |
| `IFRS9_SEED` | Random seed for reproducibility | `42` |
| `OUTPUT_DIR` | Local directory for artefacts | `/app/output` |
| `GCS_OUTPUT_BUCKET` | Optional Cloud Storage bucket for artefacts | *(unset)* |
| `GCS_OBJECT_PREFIX` | Prefix for uploaded objects | `ifrs9-batch` |

Deploy as a Cloud Run Job:

```bash
gcloud run jobs create ifrs9-batch-job \
  --image=gcr.io/PROJECT_ID/ifrs9-cloud-run-job:0.1.0 \
  --region=us-central1 \
  --set-env-vars=GCS_OUTPUT_BUCKET=ifrs9-job-artifacts

gcloud run jobs execute ifrs9-batch-job --region=us-central1
```
