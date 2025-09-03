"""
GCP Integration Module for IFRS9 Risk System

This module provides comprehensive integration with Google Cloud Platform services
including BigQuery, Cloud Storage, Dataproc, and authentication management.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
import pandas as pd

from google.cloud import bigquery, storage, dataproc_v1
from google.cloud.exceptions import NotFound, Conflict
from google.oauth2 import service_account
from google.api_core import retry
import pyarrow as pa
import pyarrow.parquet as pq

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GCPIntegration:
    """Main class for GCP service integrations"""
    
    def __init__(self, 
                 project_id: str,
                 credentials_path: Optional[str] = None,
                 location: str = "us-central1"):
        """
        Initialize GCP Integration
        
        Args:
            project_id: GCP project ID
            credentials_path: Path to service account JSON file
            location: GCP region for resources
        """
        self.project_id = project_id
        self.location = location
        self.credentials = self._load_credentials(credentials_path)
        
        # Initialize clients
        self.bigquery_client = None
        self.storage_client = None
        self.dataproc_client = None
        
        # Environment detection
        self.is_local = self._detect_environment()
        
        logger.info(f"GCP Integration initialized for project: {project_id}")
        logger.info(f"Running in {'local' if self.is_local else 'cloud'} environment")
    
    def _load_credentials(self, credentials_path: Optional[str]) -> Optional[service_account.Credentials]:
        """Load service account credentials"""
        if credentials_path and os.path.exists(credentials_path):
            return service_account.Credentials.from_service_account_file(credentials_path)
        elif os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
            cred_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            if os.path.exists(cred_path):
                return service_account.Credentials.from_service_account_file(cred_path)
        return None
    
    def _detect_environment(self) -> bool:
        """Detect if running in local or cloud environment"""
        # Check for common local indicators
        local_indicators = [
            os.path.exists('/proc/version'),  # Linux systems
            os.getenv('COMPUTERNAME'),        # Windows
            os.getenv('USER') == 'root',      # Docker containers
            not os.getenv('GOOGLE_CLOUD_PROJECT')  # No automatic GCP env vars
        ]
        return any(local_indicators)

class BigQueryManager(GCPIntegration):
    """BigQuery operations manager"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bigquery_client = bigquery.Client(
            project=self.project_id,
            credentials=self.credentials
        )
    
    def create_dataset(self, 
                      dataset_id: str, 
                      description: str = "IFRS9 Risk System Dataset",
                      location: Optional[str] = None) -> bool:
        """
        Create BigQuery dataset
        
        Args:
            dataset_id: Dataset identifier
            description: Dataset description
            location: Dataset location (defaults to class location)
            
        Returns:
            bool: Success status
        """
        try:
            dataset_location = location or self.location
            dataset_ref = f"{self.project_id}.{dataset_id}"
            
            # Check if dataset exists
            try:
                self.bigquery_client.get_dataset(dataset_ref)
                logger.info(f"Dataset {dataset_ref} already exists")
                return True
            except NotFound:
                pass
            
            # Create dataset
            dataset = bigquery.Dataset(dataset_ref)
            dataset.description = description
            dataset.location = dataset_location
            
            dataset = self.bigquery_client.create_dataset(dataset, timeout=30)
            logger.info(f"Created dataset {dataset_ref}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create dataset {dataset_id}: {str(e)}")
            return False
    
    def create_table_from_schema(self, 
                                dataset_id: str, 
                                table_id: str, 
                                schema: List[bigquery.SchemaField],
                                description: str = "") -> bool:
        """
        Create BigQuery table with schema
        
        Args:
            dataset_id: Dataset identifier
            table_id: Table identifier
            schema: Table schema
            description: Table description
            
        Returns:
            bool: Success status
        """
        try:
            table_ref = f"{self.project_id}.{dataset_id}.{table_id}"
            
            # Check if table exists
            try:
                self.bigquery_client.get_table(table_ref)
                logger.info(f"Table {table_ref} already exists")
                return True
            except NotFound:
                pass
            
            # Create table
            table = bigquery.Table(table_ref, schema=schema)
            table.description = description
            
            table = self.bigquery_client.create_table(table)
            logger.info(f"Created table {table_ref}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create table {table_id}: {str(e)}")
            return False
    
    def upload_dataframe(self, 
                        df: pd.DataFrame,
                        dataset_id: str,
                        table_id: str,
                        write_disposition: str = "WRITE_TRUNCATE") -> bool:
        """
        Upload pandas DataFrame to BigQuery
        
        Args:
            df: Pandas DataFrame
            dataset_id: Dataset identifier
            table_id: Table identifier
            write_disposition: Write mode (WRITE_TRUNCATE, WRITE_APPEND, WRITE_EMPTY)
            
        Returns:
            bool: Success status
        """
        try:
            table_ref = f"{self.project_id}.{dataset_id}.{table_id}"
            
            job_config = bigquery.LoadJobConfig(
                write_disposition=write_disposition,
                autodetect=True
            )
            
            job = self.bigquery_client.load_table_from_dataframe(
                df, table_ref, job_config=job_config
            )
            
            job.result()  # Wait for job completion
            
            logger.info(f"Uploaded {len(df)} rows to {table_ref}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload data to {table_id}: {str(e)}")
            return False
    
    def query_to_dataframe(self, query: str) -> Optional[pd.DataFrame]:
        """
        Execute BigQuery query and return as DataFrame
        
        Args:
            query: SQL query string
            
        Returns:
            pd.DataFrame or None: Query results
        """
        try:
            df = self.bigquery_client.query(query).to_dataframe()
            logger.info(f"Query returned {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            return None
    
    def create_ifrs9_dataset_schema(self) -> Dict[str, List[bigquery.SchemaField]]:
        """
        Define IFRS9 dataset table schemas
        
        Returns:
            Dict mapping table names to schemas
        """
        schemas = {
            "loan_data": [
                bigquery.SchemaField("loan_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("customer_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("loan_amount", "FLOAT64", mode="REQUIRED"),
                bigquery.SchemaField("interest_rate", "FLOAT64", mode="REQUIRED"),
                bigquery.SchemaField("term_months", "INTEGER", mode="REQUIRED"),
                bigquery.SchemaField("credit_score", "INTEGER", mode="REQUIRED"),
                bigquery.SchemaField("dti_ratio", "FLOAT64", mode="REQUIRED"),
                bigquery.SchemaField("employment_length", "INTEGER", mode="REQUIRED"),
                bigquery.SchemaField("loan_purpose", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("origination_date", "DATE", mode="REQUIRED"),
                bigquery.SchemaField("maturity_date", "DATE", mode="REQUIRED"),
                bigquery.SchemaField("historial_de_pagos", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("region", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("producto_tipo", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
            ],
            "ifrs9_results": [
                bigquery.SchemaField("loan_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("calculation_date", "DATE", mode="REQUIRED"),
                bigquery.SchemaField("ifrs9_stage", "INTEGER", mode="REQUIRED"),
                bigquery.SchemaField("pd_12m", "FLOAT64", mode="REQUIRED"),
                bigquery.SchemaField("pd_lifetime", "FLOAT64", mode="REQUIRED"),
                bigquery.SchemaField("lgd", "FLOAT64", mode="REQUIRED"),
                bigquery.SchemaField("ead", "FLOAT64", mode="REQUIRED"),
                bigquery.SchemaField("ecl", "FLOAT64", mode="REQUIRED"),
                bigquery.SchemaField("days_past_due", "INTEGER", mode="REQUIRED"),
                bigquery.SchemaField("significant_increase", "BOOLEAN", mode="REQUIRED"),
                bigquery.SchemaField("default_indicator", "BOOLEAN", mode="REQUIRED"),
                bigquery.SchemaField("model_version", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
            ],
            "macro_economic_data": [
                bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
                bigquery.SchemaField("gdp_growth", "FLOAT64", mode="NULLABLE"),
                bigquery.SchemaField("unemployment_rate", "FLOAT64", mode="NULLABLE"),
                bigquery.SchemaField("inflation_rate", "FLOAT64", mode="NULLABLE"),
                bigquery.SchemaField("interest_rate", "FLOAT64", mode="NULLABLE"),
                bigquery.SchemaField("currency_exchange_rate", "FLOAT64", mode="NULLABLE"),
                bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
            ]
        }
        return schemas

class CloudStorageManager(GCPIntegration):
    """Cloud Storage operations manager"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.storage_client = storage.Client(
            project=self.project_id,
            credentials=self.credentials
        )
    
    def create_bucket(self, bucket_name: str, storage_class: str = "STANDARD") -> bool:
        """
        Create Cloud Storage bucket
        
        Args:
            bucket_name: Bucket name
            storage_class: Storage class (STANDARD, NEARLINE, COLDLINE, ARCHIVE)
            
        Returns:
            bool: Success status
        """
        try:
            # Check if bucket exists
            try:
                self.storage_client.get_bucket(bucket_name)
                logger.info(f"Bucket {bucket_name} already exists")
                return True
            except NotFound:
                pass
            
            # Create bucket
            bucket = self.storage_client.bucket(bucket_name)
            bucket.storage_class = storage_class
            bucket.location = self.location
            
            bucket = self.storage_client.create_bucket(bucket)
            logger.info(f"Created bucket {bucket_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create bucket {bucket_name}: {str(e)}")
            return False
    
    def upload_to_gcs(self, 
                     local_path: Union[str, Path], 
                     bucket_name: str, 
                     blob_name: str,
                     content_type: Optional[str] = None) -> bool:
        """
        Upload file to Google Cloud Storage
        
        Args:
            local_path: Path to local file
            bucket_name: GCS bucket name
            blob_name: Name for the blob in GCS
            content_type: MIME type of the file
            
        Returns:
            bool: Success status
        """
        try:
            local_path = Path(local_path)
            if not local_path.exists():
                logger.error(f"Local file not found: {local_path}")
                return False
            
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            
            if content_type:
                blob.content_type = content_type
            
            blob.upload_from_filename(str(local_path))
            
            logger.info(f"Uploaded {local_path} to gs://{bucket_name}/{blob_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload {local_path}: {str(e)}")
            return False
    
    def download_from_gcs(self, 
                         bucket_name: str, 
                         blob_name: str, 
                         local_path: Union[str, Path]) -> bool:
        """
        Download file from Google Cloud Storage
        
        Args:
            bucket_name: GCS bucket name
            blob_name: Name of the blob in GCS
            local_path: Local path to save the file
            
        Returns:
            bool: Success status
        """
        try:
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            
            local_path = Path(local_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            blob.download_to_filename(str(local_path))
            
            logger.info(f"Downloaded gs://{bucket_name}/{blob_name} to {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {blob_name}: {str(e)}")
            return False
    
    def upload_dataframe_as_parquet(self, 
                                   df: pd.DataFrame,
                                   bucket_name: str,
                                   blob_name: str) -> bool:
        """
        Upload DataFrame as Parquet to GCS
        
        Args:
            df: Pandas DataFrame
            bucket_name: GCS bucket name
            blob_name: Name for the parquet file
            
        Returns:
            bool: Success status
        """
        try:
            # Convert DataFrame to Parquet bytes
            table = pa.Table.from_pandas(df)
            parquet_buffer = pa.BufferOutputStream()
            pq.write_table(table, parquet_buffer)
            parquet_bytes = parquet_buffer.getvalue().to_pybytes()
            
            # Upload to GCS
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.content_type = 'application/octet-stream'
            
            blob.upload_from_string(parquet_bytes)
            
            logger.info(f"Uploaded DataFrame ({len(df)} rows) as Parquet to gs://{bucket_name}/{blob_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload DataFrame as Parquet: {str(e)}")
            return False
    
    def migrate_local_to_cloud(self, 
                              source_dir: Union[str, Path], 
                              target_bucket: str,
                              target_prefix: str = "") -> bool:
        """
        Migrate local directory to Cloud Storage
        
        Args:
            source_dir: Local source directory
            target_bucket: Target GCS bucket
            target_prefix: Prefix for uploaded files
            
        Returns:
            bool: Success status
        """
        try:
            source_dir = Path(source_dir)
            if not source_dir.exists():
                logger.error(f"Source directory not found: {source_dir}")
                return False
            
            success_count = 0
            total_files = sum(1 for _ in source_dir.rglob("*") if _.is_file())
            
            for file_path in source_dir.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(source_dir)
                    blob_name = f"{target_prefix}/{relative_path}".lstrip("/")
                    
                    if self.upload_to_gcs(file_path, target_bucket, blob_name):
                        success_count += 1
            
            logger.info(f"Migrated {success_count}/{total_files} files to gs://{target_bucket}")
            return success_count == total_files
            
        except Exception as e:
            logger.error(f"Failed to migrate directory: {str(e)}")
            return False

class DataprocManager(GCPIntegration):
    """Dataproc cluster management"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataproc_client = dataproc_v1.ClusterControllerClient(
            credentials=self.credentials
        )
    
    def provision_dataproc_cluster(self, 
                                  cluster_name: str,
                                  worker_count: int = 2,
                                  machine_type: str = "n1-standard-2",
                                  disk_size: int = 100,
                                  preemptible_workers: int = 0) -> bool:
        """
        Provision Dataproc cluster for Spark processing
        
        Args:
            cluster_name: Name of the cluster
            worker_count: Number of worker nodes
            machine_type: GCE machine type
            disk_size: Boot disk size in GB
            preemptible_workers: Number of preemptible worker nodes
            
        Returns:
            bool: Success status
        """
        try:
            # Check if cluster exists
            try:
                cluster = self.dataproc_client.get_cluster(
                    request={
                        "project_id": self.project_id,
                        "region": self.location,
                        "cluster_name": cluster_name
                    }
                )
                logger.info(f"Cluster {cluster_name} already exists")
                return True
            except NotFound:
                pass
            
            # Cluster configuration
            cluster_config = {
                "project_id": self.project_id,
                "cluster_name": cluster_name,
                "config": {
                    "master_config": {
                        "num_instances": 1,
                        "machine_type_uri": machine_type,
                        "disk_config": {
                            "boot_disk_type": "pd-standard",
                            "boot_disk_size_gb": disk_size
                        }
                    },
                    "worker_config": {
                        "num_instances": worker_count,
                        "machine_type_uri": machine_type,
                        "disk_config": {
                            "boot_disk_type": "pd-standard",
                            "boot_disk_size_gb": disk_size
                        }
                    },
                    "software_config": {
                        "image_version": "2.0-debian10"
                    },
                    "gce_cluster_config": {
                        "zone_uri": f"{self.location}-a"
                    }
                }
            }
            
            # Add preemptible workers if specified
            if preemptible_workers > 0:
                cluster_config["config"]["preemptible_worker_config"] = {
                    "num_instances": preemptible_workers,
                    "machine_type_uri": machine_type,
                    "disk_config": {
                        "boot_disk_type": "pd-standard",
                        "boot_disk_size_gb": disk_size
                    }
                }
            
            # Create cluster
            operation = self.dataproc_client.create_cluster(
                request={
                    "project_id": self.project_id,
                    "region": self.location,
                    "cluster": cluster_config
                }
            )
            
            result = operation.result(timeout=900)  # 15 minutes timeout
            logger.info(f"Created Dataproc cluster: {cluster_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create Dataproc cluster: {str(e)}")
            return False
    
    def delete_dataproc_cluster(self, cluster_name: str) -> bool:
        """
        Delete Dataproc cluster
        
        Args:
            cluster_name: Name of the cluster to delete
            
        Returns:
            bool: Success status
        """
        try:
            operation = self.dataproc_client.delete_cluster(
                request={
                    "project_id": self.project_id,
                    "region": self.location,
                    "cluster_name": cluster_name
                }
            )
            
            operation.result(timeout=300)  # 5 minutes timeout
            logger.info(f"Deleted Dataproc cluster: {cluster_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete Dataproc cluster: {str(e)}")
            return False

class IFRS9GCPIntegration:
    """High-level IFRS9-specific GCP integration"""
    
    def __init__(self, 
                 project_id: str,
                 credentials_path: Optional[str] = None,
                 location: str = "us-central1"):
        """Initialize IFRS9 GCP Integration"""
        self.project_id = project_id
        self.location = location
        
        # Initialize managers
        self.bigquery = BigQueryManager(project_id, credentials_path, location)
        self.storage = CloudStorageManager(project_id, credentials_path, location)
        self.dataproc = DataprocManager(project_id, credentials_path, location)
        
        # IFRS9 specific configurations
        self.dataset_id = "ifrs9_risk_system"
        self.bucket_name = f"ifrs9-data-{project_id}"
    
    def setup_ifrs9_infrastructure(self) -> bool:
        """
        Set up complete IFRS9 GCP infrastructure
        
        Returns:
            bool: Success status
        """
        logger.info("Setting up IFRS9 GCP infrastructure...")
        
        try:
            # Create Cloud Storage bucket
            if not self.storage.create_bucket(self.bucket_name):
                return False
            
            # Create BigQuery dataset
            if not self.bigquery.create_dataset(
                self.dataset_id, 
                "IFRS9 Risk Management System Dataset"
            ):
                return False
            
            # Create tables with IFRS9 schemas
            schemas = self.bigquery.create_ifrs9_dataset_schema()
            for table_name, schema in schemas.items():
                if not self.bigquery.create_table_from_schema(
                    self.dataset_id, 
                    table_name, 
                    schema,
                    f"IFRS9 {table_name.replace('_', ' ').title()} table"
                ):
                    return False
            
            logger.info("IFRS9 GCP infrastructure setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup IFRS9 infrastructure: {str(e)}")
            return False
    
    def upload_ifrs9_data(self, 
                         data_dict: Dict[str, pd.DataFrame],
                         upload_to_bigquery: bool = True,
                         upload_to_gcs: bool = True) -> bool:
        """
        Upload IFRS9 data to GCP services
        
        Args:
            data_dict: Dictionary mapping table names to DataFrames
            upload_to_bigquery: Whether to upload to BigQuery
            upload_to_gcs: Whether to upload to Cloud Storage
            
        Returns:
            bool: Success status
        """
        try:
            success = True
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for table_name, df in data_dict.items():
                logger.info(f"Processing {table_name} ({len(df)} rows)")
                
                # Upload to BigQuery
                if upload_to_bigquery:
                    if not self.bigquery.upload_dataframe(
                        df, self.dataset_id, table_name
                    ):
                        success = False
                
                # Upload to Cloud Storage as Parquet
                if upload_to_gcs:
                    blob_name = f"data/{table_name}/{timestamp}/{table_name}.parquet"
                    if not self.storage.upload_dataframe_as_parquet(
                        df, self.bucket_name, blob_name
                    ):
                        success = False
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to upload IFRS9 data: {str(e)}")
            return False

# Utility functions for easy integration
def get_gcp_config() -> Dict[str, Any]:
    """Get GCP configuration from environment variables"""
    return {
        'project_id': os.getenv('GCP_PROJECT_ID'),
        'credentials_path': os.getenv('GOOGLE_APPLICATION_CREDENTIALS'),
        'location': os.getenv('GCP_LOCATION', 'us-central1'),
        'dataset_id': os.getenv('BIGQUERY_DATASET_ID', 'ifrs9_risk_system'),
        'bucket_name': os.getenv('GCS_BUCKET_NAME')
    }

def initialize_ifrs9_gcp(project_id: str, 
                        credentials_path: Optional[str] = None) -> IFRS9GCPIntegration:
    """
    Initialize IFRS9 GCP integration with default settings
    
    Args:
        project_id: GCP project ID
        credentials_path: Path to service account credentials
        
    Returns:
        IFRS9GCPIntegration instance
    """
    return IFRS9GCPIntegration(project_id, credentials_path)

if __name__ == "__main__":
    # Example usage
    config = get_gcp_config()
    
    if config['project_id']:
        gcp_integration = initialize_ifrs9_gcp(
            config['project_id'], 
            config['credentials_path']
        )
        
        # Setup infrastructure
        if gcp_integration.setup_ifrs9_infrastructure():
            logger.info("IFRS9 GCP infrastructure is ready!")
        else:
            logger.error("Failed to setup IFRS9 infrastructure")
    else:
        logger.error("GCP_PROJECT_ID environment variable not set")