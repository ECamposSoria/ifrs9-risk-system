#!/usr/bin/env python3
"""
IFRS9 External Systems Integration Validator
Comprehensive validation for BigQuery, GCS, PostgreSQL, and Airflow integrations
"""

import os
import sys
import asyncio
import time
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import uuid
import tempfile
import random
import pandas as pd
import polars as pl
from pathlib import Path

# Google Cloud imports
from google.cloud import bigquery, storage
from google.cloud.exceptions import GoogleCloudError
import google.auth
from google.auth.exceptions import GoogleAuthError

# PostgreSQL imports
import asyncpg
import psycopg2
from psycopg2 import pool

# Airflow imports
import requests
from requests.auth import HTTPBasicAuth

# Monitoring imports
from prometheus_client import Counter, Histogram, Gauge
import aiohttp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ValidationConfig:
    """Configuration for external system validation"""
    # BigQuery settings
    bigquery_project: str = "ifrs9-production"
    bigquery_dataset: str = "ifrs9_data"
    bigquery_test_table: str = "validation_test"
    
    # GCS settings
    gcs_bucket: str = "ifrs9-data-bucket"
    gcs_test_prefix: str = "validation_tests"
    
    # PostgreSQL settings
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_database: str = "ifrs9"
    postgres_user: str = "airflow"
    postgres_password: str = "airflow"
    postgres_pool_size: int = 20
    
    # Airflow settings
    airflow_host: str = "localhost"
    airflow_port: int = 8080
    airflow_username: str = "admin"
    airflow_password: str = "admin"
    
    # Test parameters
    test_data_size: int = 10000
    concurrent_connections: int = 10
    timeout_seconds: int = 300
    retry_attempts: int = 3

@dataclass
class ValidationResult:
    """Result of a validation test"""
    test_name: str
    system: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    success: bool
    error_message: Optional[str]
    performance_metrics: Dict[str, Any]
    details: Dict[str, Any]

class ExternalSystemMetrics:
    """Prometheus metrics for external system validation"""
    
    def __init__(self):
        self.validation_duration = Histogram(
            'ifrs9_external_validation_duration_seconds',
            'Duration of external system validation tests',
            ['system', 'test_type', 'status']
        )
        
        self.validation_success_rate = Gauge(
            'ifrs9_external_validation_success_rate',
            'Success rate of external system validations',
            ['system']
        )
        
        self.connection_pool_utilization = Gauge(
            'ifrs9_external_connection_pool_utilization',
            'Connection pool utilization percentage',
            ['system']
        )
        
        self.data_transfer_rate_mbps = Gauge(
            'ifrs9_external_data_transfer_rate_mbps',
            'Data transfer rate in MB/s',
            ['system', 'direction']
        )
        
        self.query_performance = Histogram(
            'ifrs9_external_query_duration_seconds',
            'Query execution duration',
            ['system', 'query_type']
        )

class BigQueryValidator:
    """Validator for BigQuery integration"""
    
    def __init__(self, config: ValidationConfig, metrics: ExternalSystemMetrics):
        self.config = config
        self.metrics = metrics
        self.client = None
        
    async def initialize(self):
        """Initialize BigQuery client"""
        try:
            self.client = bigquery.Client(project=self.config.bigquery_project)
            logger.info("BigQuery client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize BigQuery client: {e}")
            raise
    
    async def validate_connectivity(self) -> ValidationResult:
        """Test basic BigQuery connectivity"""
        test_name = "bigquery_connectivity"
        start_time = datetime.now()
        
        try:
            # Test basic connection
            datasets = list(self.client.list_datasets())
            
            # Test query execution
            query = "SELECT 1 as test_value"
            query_job = self.client.query(query)
            results = list(query_job)
            
            if len(results) > 0 and results[0]['test_value'] == 1:
                success = True
                error_message = None
            else:
                success = False
                error_message = "Unexpected query result"
                
        except Exception as e:
            success = False
            error_message = str(e)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Update metrics
        status = 'success' if success else 'failure'
        self.metrics.validation_duration.labels(
            system='bigquery',
            test_type='connectivity',
            status=status
        ).observe(duration)
        
        return ValidationResult(
            test_name=test_name,
            system='bigquery',
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            success=success,
            error_message=error_message,
            performance_metrics={'query_duration': duration},
            details={'datasets_count': len(datasets) if success else 0}
        )
    
    async def validate_data_upload_performance(self) -> ValidationResult:
        """Test data upload performance to BigQuery"""
        test_name = "bigquery_upload_performance"
        start_time = datetime.now()
        
        try:
            # Create test dataset if not exists
            dataset_id = f"{self.config.bigquery_project}.{self.config.bigquery_dataset}"
            dataset = bigquery.Dataset(dataset_id)
            
            try:
                dataset = self.client.create_dataset(dataset, exists_ok=True)
            except Exception:
                pass  # Dataset might already exist
            
            # Generate test data
            test_data = self._generate_test_data(self.config.test_data_size)
            
            # Create table reference
            table_id = f"{dataset_id}.{self.config.bigquery_test_table}_{int(time.time())}"
            
            # Define schema
            schema = [
                bigquery.SchemaField("id", "INTEGER", mode="REQUIRED"),
                bigquery.SchemaField("customer_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("loan_amount", "FLOAT", mode="REQUIRED"),
                bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
            ]
            
            # Create table
            table = bigquery.Table(table_id, schema=schema)
            table = self.client.create_table(table)
            
            # Upload data
            upload_start = time.time()
            job_config = bigquery.LoadJobConfig(
                source_format=bigquery.SourceFormat.CSV,
                skip_leading_rows=1,
                autodetect=False,
                schema=schema
            )
            
            # Convert to CSV for upload
            csv_data = test_data.to_csv(index=False)
            job = self.client.load_table_from_file(
                file_obj=tempfile.NamedTemporaryFile(mode='w+', suffix='.csv'),
                destination=table,
                job_config=job_config
            )
            
            job.result()  # Wait for completion
            upload_duration = time.time() - upload_start
            
            # Verify upload
            query = f"SELECT COUNT(*) as row_count FROM `{table_id}`"
            query_job = self.client.query(query)
            result = list(query_job)[0]
            uploaded_rows = result['row_count']
            
            # Calculate performance metrics
            data_size_mb = len(csv_data) / 1024 / 1024
            transfer_rate = data_size_mb / upload_duration if upload_duration > 0 else 0
            
            # Update metrics
            self.metrics.data_transfer_rate_mbps.labels(
                system='bigquery',
                direction='upload'
            ).set(transfer_rate)
            
            success = uploaded_rows == len(test_data)
            error_message = None if success else f"Uploaded {uploaded_rows} rows, expected {len(test_data)}"
            
            # Cleanup
            self.client.delete_table(table)
            
        except Exception as e:
            success = False
            error_message = str(e)
            transfer_rate = 0
            data_size_mb = 0
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Update metrics
        status = 'success' if success else 'failure'
        self.metrics.validation_duration.labels(
            system='bigquery',
            test_type='upload_performance',
            status=status
        ).observe(duration)
        
        return ValidationResult(
            test_name=test_name,
            system='bigquery',
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            success=success,
            error_message=error_message,
            performance_metrics={
                'upload_duration': upload_duration if success else 0,
                'transfer_rate_mbps': transfer_rate,
                'data_size_mb': data_size_mb
            },
            details={
                'records_uploaded': uploaded_rows if success else 0,
                'expected_records': len(test_data) if success else 0
            }
        )
    
    async def validate_query_performance(self) -> ValidationResult:
        """Test BigQuery query performance"""
        test_name = "bigquery_query_performance"
        start_time = datetime.now()
        
        performance_queries = [
            {
                'name': 'simple_aggregation',
                'query': f"""
                    SELECT 
                        COUNT(*) as total_loans,
                        SUM(outstanding_balance) as total_balance,
                        AVG(interest_rate) as avg_rate
                    FROM `{self.config.bigquery_project}.{self.config.bigquery_dataset}.loan_portfolio`
                    WHERE origination_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 1 YEAR)
                """
            },
            {
                'name': 'complex_join',
                'query': f"""
                    SELECT 
                        l.ifrs9_stage,
                        COUNT(*) as loan_count,
                        AVG(c.annual_income) as avg_income,
                        SUM(l.outstanding_balance) as total_exposure
                    FROM `{self.config.bigquery_project}.{self.config.bigquery_dataset}.loan_portfolio` l
                    JOIN `{self.config.bigquery_project}.{self.config.bigquery_dataset}.customers` c 
                        ON l.customer_id = c.customer_id
                    GROUP BY l.ifrs9_stage
                    ORDER BY total_exposure DESC
                """
            }
        ]
        
        query_results = []
        overall_success = True
        
        try:
            for query_info in performance_queries:
                query_start = time.time()
                
                try:
                    query_job = self.client.query(query_info['query'])
                    results = list(query_job)
                    query_duration = time.time() - query_start
                    
                    self.metrics.query_performance.labels(
                        system='bigquery',
                        query_type=query_info['name']
                    ).observe(query_duration)
                    
                    query_results.append({
                        'query_name': query_info['name'],
                        'duration': query_duration,
                        'rows_returned': len(results),
                        'success': True
                    })
                    
                except Exception as e:
                    query_duration = time.time() - query_start
                    overall_success = False
                    
                    query_results.append({
                        'query_name': query_info['name'],
                        'duration': query_duration,
                        'error': str(e),
                        'success': False
                    })
            
            error_message = None if overall_success else "Some queries failed"
            
        except Exception as e:
            overall_success = False
            error_message = str(e)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Update metrics
        status = 'success' if overall_success else 'failure'
        self.metrics.validation_duration.labels(
            system='bigquery',
            test_type='query_performance',
            status=status
        ).observe(duration)
        
        return ValidationResult(
            test_name=test_name,
            system='bigquery',
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            success=overall_success,
            error_message=error_message,
            performance_metrics={
                'avg_query_duration': sum(q['duration'] for q in query_results) / len(query_results) if query_results else 0,
                'total_queries': len(query_results)
            },
            details={'query_results': query_results}
        )
    
    def _generate_test_data(self, size: int) -> pd.DataFrame:
        """Generate test data for upload testing"""
        return pd.DataFrame({
            'id': range(1, size + 1),
            'customer_id': [f"CUST_{i:06d}" for i in range(1, size + 1)],
            'loan_amount': [random.uniform(1000, 100000) for _ in range(size)],
            'created_at': [datetime.now() - timedelta(days=random.randint(0, 365)) for _ in range(size)]
        })

class GCSValidator:
    """Validator for Google Cloud Storage integration"""
    
    def __init__(self, config: ValidationConfig, metrics: ExternalSystemMetrics):
        self.config = config
        self.metrics = metrics
        self.client = None
        self.bucket = None
    
    async def initialize(self):
        """Initialize GCS client"""
        try:
            self.client = storage.Client()
            self.bucket = self.client.bucket(self.config.gcs_bucket)
            logger.info("GCS client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize GCS client: {e}")
            raise
    
    async def validate_connectivity(self) -> ValidationResult:
        """Test basic GCS connectivity"""
        test_name = "gcs_connectivity"
        start_time = datetime.now()
        
        try:
            # Test bucket access
            if self.bucket.exists():
                # List some objects
                blobs = list(self.bucket.list_blobs(max_results=10))
                success = True
                error_message = None
            else:
                success = False
                error_message = f"Bucket {self.config.gcs_bucket} does not exist"
                
        except Exception as e:
            success = False
            error_message = str(e)
            blobs = []
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Update metrics
        status = 'success' if success else 'failure'
        self.metrics.validation_duration.labels(
            system='gcs',
            test_type='connectivity',
            status=status
        ).observe(duration)
        
        return ValidationResult(
            test_name=test_name,
            system='gcs',
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            success=success,
            error_message=error_message,
            performance_metrics={'list_duration': duration},
            details={'objects_found': len(blobs) if success else 0}
        )
    
    async def validate_upload_download_performance(self) -> ValidationResult:
        """Test GCS upload and download performance"""
        test_name = "gcs_upload_download_performance"
        start_time = datetime.now()
        
        try:
            # Generate test data
            test_data_size = 10 * 1024 * 1024  # 10MB
            test_data = os.urandom(test_data_size)
            
            # Create test blob name
            blob_name = f"{self.config.gcs_test_prefix}/test_{uuid.uuid4()}.bin"
            blob = self.bucket.blob(blob_name)
            
            # Upload test
            upload_start = time.time()
            blob.upload_from_string(test_data)
            upload_duration = time.time() - upload_start
            upload_rate = (test_data_size / 1024 / 1024) / upload_duration  # MB/s
            
            # Update upload metrics
            self.metrics.data_transfer_rate_mbps.labels(
                system='gcs',
                direction='upload'
            ).set(upload_rate)
            
            # Download test
            download_start = time.time()
            downloaded_data = blob.download_as_bytes()
            download_duration = time.time() - download_start
            download_rate = (len(downloaded_data) / 1024 / 1024) / download_duration  # MB/s
            
            # Update download metrics
            self.metrics.data_transfer_rate_mbps.labels(
                system='gcs',
                direction='download'
            ).set(download_rate)
            
            # Verify data integrity
            data_integrity_ok = len(downloaded_data) == len(test_data)
            
            # Cleanup
            blob.delete()
            
            success = data_integrity_ok
            error_message = None if success else "Data integrity check failed"
            
        except Exception as e:
            success = False
            error_message = str(e)
            upload_duration = download_duration = 0
            upload_rate = download_rate = 0
            test_data_size = 0
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Update metrics
        status = 'success' if success else 'failure'
        self.metrics.validation_duration.labels(
            system='gcs',
            test_type='upload_download_performance',
            status=status
        ).observe(duration)
        
        return ValidationResult(
            test_name=test_name,
            system='gcs',
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            success=success,
            error_message=error_message,
            performance_metrics={
                'upload_duration': upload_duration,
                'download_duration': download_duration,
                'upload_rate_mbps': upload_rate,
                'download_rate_mbps': download_rate,
                'data_size_mb': test_data_size / 1024 / 1024
            },
            details={'data_integrity': success}
        )

class PostgreSQLValidator:
    """Validator for PostgreSQL integration"""
    
    def __init__(self, config: ValidationConfig, metrics: ExternalSystemMetrics):
        self.config = config
        self.metrics = metrics
        self.connection_pool = None
    
    async def initialize(self):
        """Initialize PostgreSQL connection pool"""
        try:
            # Create asyncpg connection pool
            dsn = f"postgresql://{self.config.postgres_user}:{self.config.postgres_password}@{self.config.postgres_host}:{self.config.postgres_port}/{self.config.postgres_database}"
            
            self.connection_pool = await asyncpg.create_pool(
                dsn,
                min_size=5,
                max_size=self.config.postgres_pool_size,
                command_timeout=self.config.timeout_seconds
            )
            
            logger.info("PostgreSQL connection pool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL pool: {e}")
            raise
    
    async def validate_connectivity(self) -> ValidationResult:
        """Test PostgreSQL connectivity and basic operations"""
        test_name = "postgresql_connectivity"
        start_time = datetime.now()
        
        try:
            async with self.connection_pool.acquire() as conn:
                # Test basic query
                result = await conn.fetchval("SELECT 1")
                
                # Test database info
                db_version = await conn.fetchval("SELECT version()")
                
                success = result == 1
                error_message = None
                
        except Exception as e:
            success = False
            error_message = str(e)
            db_version = None
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Update metrics
        status = 'success' if success else 'failure'
        self.metrics.validation_duration.labels(
            system='postgresql',
            test_type='connectivity',
            status=status
        ).observe(duration)
        
        # Update pool utilization
        if self.connection_pool:
            pool_utilization = (self.connection_pool.get_size() - self.connection_pool.get_idle_size()) / self.connection_pool.get_max_size() * 100
            self.metrics.connection_pool_utilization.labels(system='postgresql').set(pool_utilization)
        
        return ValidationResult(
            test_name=test_name,
            system='postgresql',
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            success=success,
            error_message=error_message,
            performance_metrics={'connection_duration': duration},
            details={'database_version': db_version if success else None}
        )
    
    async def validate_connection_pool_performance(self) -> ValidationResult:
        """Test connection pool performance under load"""
        test_name = "postgresql_connection_pool_performance"
        start_time = datetime.now()
        
        async def test_connection():
            """Test a single connection"""
            try:
                async with self.connection_pool.acquire() as conn:
                    # Simulate typical IFRS9 query
                    await conn.execute("SELECT pg_sleep(0.1)")  # Simulate 100ms query
                    return True
            except Exception:
                return False
        
        try:
            # Run concurrent connections
            tasks = [test_connection() for _ in range(self.config.concurrent_connections)]
            results = await asyncio.gather(*tasks)
            
            successful_connections = sum(results)
            success_rate = successful_connections / len(results) * 100
            
            success = success_rate >= 95  # 95% success rate threshold
            error_message = None if success else f"Success rate {success_rate:.1f}% below threshold"
            
        except Exception as e:
            success = False
            error_message = str(e)
            success_rate = 0
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Update metrics
        status = 'success' if success else 'failure'
        self.metrics.validation_duration.labels(
            system='postgresql',
            test_type='connection_pool_performance',
            status=status
        ).observe(duration)
        
        return ValidationResult(
            test_name=test_name,
            system='postgresql',
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            success=success,
            error_message=error_message,
            performance_metrics={
                'success_rate': success_rate,
                'concurrent_connections': self.config.concurrent_connections,
                'avg_connection_time': duration / self.config.concurrent_connections
            },
            details={'successful_connections': successful_connections if success else 0}
        )

class AirflowValidator:
    """Validator for Apache Airflow integration"""
    
    def __init__(self, config: ValidationConfig, metrics: ExternalSystemMetrics):
        self.config = config
        self.metrics = metrics
        self.base_url = f"http://{self.config.airflow_host}:{self.config.airflow_port}"
        self.auth = HTTPBasicAuth(self.config.airflow_username, self.config.airflow_password)
    
    async def validate_connectivity(self) -> ValidationResult:
        """Test Airflow API connectivity"""
        test_name = "airflow_connectivity"
        start_time = datetime.now()
        
        try:
            # Test health endpoint
            health_url = f"{self.base_url}/health"
            response = requests.get(health_url, auth=self.auth, timeout=30)
            
            if response.status_code == 200:
                health_data = response.json()
                
                # Test API version endpoint
                version_url = f"{self.base_url}/api/v1/version"
                version_response = requests.get(version_url, auth=self.auth, timeout=30)
                
                if version_response.status_code == 200:
                    version_data = version_response.json()
                    success = True
                    error_message = None
                else:
                    success = False
                    error_message = f"API version check failed: {version_response.status_code}"
            else:
                success = False
                error_message = f"Health check failed: {response.status_code}"
                health_data = None
                version_data = None
                
        except Exception as e:
            success = False
            error_message = str(e)
            health_data = None
            version_data = None
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Update metrics
        status = 'success' if success else 'failure'
        self.metrics.validation_duration.labels(
            system='airflow',
            test_type='connectivity',
            status=status
        ).observe(duration)
        
        return ValidationResult(
            test_name=test_name,
            system='airflow',
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            success=success,
            error_message=error_message,
            performance_metrics={'api_response_time': duration},
            details={
                'health_status': health_data if success else None,
                'version_info': version_data if success else None
            }
        )
    
    async def validate_dag_operations(self) -> ValidationResult:
        """Test DAG operations"""
        test_name = "airflow_dag_operations"
        start_time = datetime.now()
        
        try:
            # List DAGs
            dags_url = f"{self.base_url}/api/v1/dags"
            response = requests.get(dags_url, auth=self.auth, timeout=30)
            
            if response.status_code == 200:
                dags_data = response.json()
                dag_count = dags_data.get('total_entries', 0)
                
                # Check for IFRS9 DAG specifically
                ifrs9_dag_found = False
                for dag in dags_data.get('dags', []):
                    if 'ifrs9' in dag.get('dag_id', '').lower():
                        ifrs9_dag_found = True
                        break
                
                success = dag_count > 0 and ifrs9_dag_found
                error_message = None if success else "IFRS9 DAG not found"
                
            else:
                success = False
                error_message = f"DAG list failed: {response.status_code}"
                dag_count = 0
                ifrs9_dag_found = False
                
        except Exception as e:
            success = False
            error_message = str(e)
            dag_count = 0
            ifrs9_dag_found = False
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Update metrics
        status = 'success' if success else 'failure'
        self.metrics.validation_duration.labels(
            system='airflow',
            test_type='dag_operations',
            status=status
        ).observe(duration)
        
        return ValidationResult(
            test_name=test_name,
            system='airflow',
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            success=success,
            error_message=error_message,
            performance_metrics={'dag_list_time': duration},
            details={
                'total_dags': dag_count,
                'ifrs9_dag_found': ifrs9_dag_found
            }
        )

class ExternalSystemsValidator:
    """Main validator orchestrator for all external systems"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.metrics = ExternalSystemMetrics()
        
        # Initialize validators
        self.bigquery_validator = BigQueryValidator(config, self.metrics)
        self.gcs_validator = GCSValidator(config, self.metrics)
        self.postgresql_validator = PostgreSQLValidator(config, self.metrics)
        self.airflow_validator = AirflowValidator(config, self.metrics)
        
        self.results: List[ValidationResult] = []
    
    async def initialize_all(self):
        """Initialize all validators"""
        logger.info("Initializing external system validators...")
        
        try:
            await self.bigquery_validator.initialize()
            await self.gcs_validator.initialize()
            await self.postgresql_validator.initialize()
            # Airflow doesn't need async initialization
            
            logger.info("All validators initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize validators: {e}")
            raise
    
    async def run_all_validations(self) -> List[ValidationResult]:
        """Run all validation tests"""
        logger.info("Starting comprehensive external systems validation")
        
        all_results = []
        
        # BigQuery validations
        logger.info("Running BigQuery validations...")
        bq_results = await self._run_bigquery_validations()
        all_results.extend(bq_results)
        
        # GCS validations
        logger.info("Running GCS validations...")
        gcs_results = await self._run_gcs_validations()
        all_results.extend(gcs_results)
        
        # PostgreSQL validations
        logger.info("Running PostgreSQL validations...")
        pg_results = await self._run_postgresql_validations()
        all_results.extend(pg_results)
        
        # Airflow validations
        logger.info("Running Airflow validations...")
        af_results = await self._run_airflow_validations()
        all_results.extend(af_results)
        
        self.results = all_results
        
        # Update overall success metrics
        self._update_overall_metrics()
        
        # Save results
        self._save_results()
        
        logger.info(f"Completed all validations. Total: {len(all_results)} tests")
        return all_results
    
    async def _run_bigquery_validations(self) -> List[ValidationResult]:
        """Run all BigQuery validation tests"""
        results = []
        
        results.append(await self.bigquery_validator.validate_connectivity())
        results.append(await self.bigquery_validator.validate_data_upload_performance())
        results.append(await self.bigquery_validator.validate_query_performance())
        
        return results
    
    async def _run_gcs_validations(self) -> List[ValidationResult]:
        """Run all GCS validation tests"""
        results = []
        
        results.append(await self.gcs_validator.validate_connectivity())
        results.append(await self.gcs_validator.validate_upload_download_performance())
        
        return results
    
    async def _run_postgresql_validations(self) -> List[ValidationResult]:
        """Run all PostgreSQL validation tests"""
        results = []
        
        results.append(await self.postgresql_validator.validate_connectivity())
        results.append(await self.postgresql_validator.validate_connection_pool_performance())
        
        return results
    
    async def _run_airflow_validations(self) -> List[ValidationResult]:
        """Run all Airflow validation tests"""
        results = []
        
        results.append(await self.airflow_validator.validate_connectivity())
        results.append(await self.airflow_validator.validate_dag_operations())
        
        return results
    
    def _update_overall_metrics(self):
        """Update overall system success rate metrics"""
        systems = ['bigquery', 'gcs', 'postgresql', 'airflow']
        
        for system in systems:
            system_results = [r for r in self.results if r.system == system]
            if system_results:
                success_rate = sum(1 for r in system_results if r.success) / len(system_results) * 100
                self.metrics.validation_success_rate.labels(system=system).set(success_rate)
    
    def _save_results(self):
        """Save validation results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"external_systems_validation_{timestamp}.json"
        
        results_data = []
        for result in self.results:
            result_dict = {
                'test_name': result.test_name,
                'system': result.system,
                'start_time': result.start_time.isoformat(),
                'end_time': result.end_time.isoformat(),
                'duration_seconds': result.duration_seconds,
                'success': result.success,
                'error_message': result.error_message,
                'performance_metrics': result.performance_metrics,
                'details': result.details
            }
            results_data.append(result_dict)
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
    
    def print_summary(self):
        """Print validation summary"""
        logger.info("=== EXTERNAL SYSTEMS VALIDATION SUMMARY ===")
        
        systems = ['bigquery', 'gcs', 'postgresql', 'airflow']
        
        for system in systems:
            system_results = [r for r in self.results if r.system == system]
            successful = sum(1 for r in system_results if r.success)
            total = len(system_results)
            success_rate = (successful / total * 100) if total > 0 else 0
            
            logger.info(f"{system.upper()}:")
            logger.info(f"  Tests: {successful}/{total} successful ({success_rate:.1f}%)")
            
            for result in system_results:
                status = "✓" if result.success else "✗"
                logger.info(f"  {status} {result.test_name}: {result.duration_seconds:.2f}s")
                if not result.success and result.error_message:
                    logger.info(f"    Error: {result.error_message}")
        
        # Overall summary
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        overall_success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        logger.info(f"\nOVERALL: {successful_tests}/{total_tests} tests passed ({overall_success_rate:.1f}%)")

async def main():
    """Main execution function"""
    # Configuration
    config = ValidationConfig()
    
    # Create validator
    validator = ExternalSystemsValidator(config)
    
    try:
        # Initialize all systems
        await validator.initialize_all()
        
        # Run all validations
        results = await validator.run_all_validations()
        
        # Print summary
        validator.print_summary()
        
        # Return success if all critical tests pass
        critical_failures = [r for r in results if not r.success and 'connectivity' in r.test_name]
        if critical_failures:
            logger.error("Critical connectivity tests failed!")
            sys.exit(1)
        else:
            logger.info("All external systems validation completed successfully!")
            
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())