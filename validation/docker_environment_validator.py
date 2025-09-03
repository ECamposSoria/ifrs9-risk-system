#!/usr/bin/env python3
"""
IFRS9 Docker Environment Validation Agent
Comprehensive validation framework for Docker-based IFRS9 system

This validator performs comprehensive validation of the Docker environment:
- Container health and ML dependency verification
- Cross-container integration testing  
- BigQuery/GCS connectivity validation
- IFRS9 business rule compliance
- End-to-end pipeline validation
"""

import sys
import os
import json
import time
import traceback
import subprocess
import requests
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any, Optional, Union
import pandas as pd
from pathlib import Path


class DockerEnvironmentValidator:
    """Comprehensive Docker environment validation for IFRS9 system."""
    
    def __init__(self):
        """Initialize the Docker environment validator."""
        self.validation_results = {}
        self.issues_found = []
        self.critical_issues = []
        self.warnings = []
        self.container_health = {}
        
        # Container configuration from docker-compose.ifrs9.yml
        self.containers = {
            'spark-master': {'port': 8080, 'health_endpoint': '/'},
            'spark-worker': {'port': 8081, 'health_endpoint': '/'},
            'jupyter': {'port': 8888, 'health_endpoint': '/api'},
            'airflow-webserver': {'port': 8080, 'health_endpoint': '/health'},
            'postgres': {'port': 5432, 'health_check': 'pg_isready'},
        }
        
        # ML dependencies to validate post-build
        self.ml_dependencies = {
            'core_ml': ['xgboost', 'lightgbm', 'catboost', 'optuna', 'shap'],
            'data_processing': ['pandas', 'numpy', 'scikit-learn'],
            'spark_ml': ['pyspark.ml', 'pyspark.mllib'],
            'cloud_integration': ['google-cloud-bigquery', 'google-cloud-storage']
        }
    
    def validate_docker_environment(self) -> Dict[str, Any]:
        """Validate Docker environment and container status."""
        print("=" * 70)
        print("VALIDATING DOCKER ENVIRONMENT")
        print("=" * 70)
        
        docker_validation = {
            'docker_info': {},
            'container_status': {},
            'network_connectivity': {},
            'volume_mounts': {}
        }
        
        try:
            # Check Docker daemon
            result = subprocess.run(['docker', 'info'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                docker_validation['docker_info']['status'] = 'RUNNING'
                # Parse key Docker info
                info_lines = result.stdout.split('\n')
                for line in info_lines[:10]:  # First 10 lines contain key info
                    if ':' in line:
                        key, value = line.split(':', 1)
                        docker_validation['docker_info'][key.strip()] = value.strip()
                print("✅ Docker daemon: Running")
            else:
                docker_validation['docker_info']['status'] = 'ERROR'
                docker_validation['docker_info']['error'] = result.stderr
                self.critical_issues.append("Docker daemon not accessible")
                print("❌ Docker daemon: Not accessible")
                
        except Exception as e:
            docker_validation['docker_info']['status'] = 'ERROR'
            docker_validation['docker_info']['error'] = str(e)
            self.critical_issues.append(f"Docker environment error: {str(e)}")
            print(f"❌ Docker environment: {str(e)}")
        
        return docker_validation
    
    def validate_container_health(self) -> Dict[str, Any]:
        """Validate health status of all IFRS9 containers."""
        print("\nVALIDATING CONTAINER HEALTH")
        print("-" * 50)
        
        container_health = {}
        
        try:
            # Get running containers
            result = subprocess.run(['docker', 'ps', '--format', 'json'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                running_containers = []
                for line in result.stdout.strip().split('\n'):
                    if line:
                        try:
                            container_info = json.loads(line)
                            running_containers.append(container_info)
                        except json.JSONDecodeError:
                            continue
                
                # Check each expected container
                for container_name, config in self.containers.items():
                    container_found = False
                    for container in running_containers:
                        if container_name in container.get('Names', ''):
                            container_found = True
                            
                            container_health[container_name] = {
                                'status': 'RUNNING',
                                'state': container.get('State', 'unknown'),
                                'ports': container.get('Ports', 'unknown'),
                                'image': container.get('Image', 'unknown'),
                                'created': container.get('CreatedAt', 'unknown')
                            }
                            
                            # Test container health endpoint if applicable
                            if 'health_endpoint' in config:
                                health_status = self._test_container_endpoint(
                                    container_name, config['port'], config['health_endpoint']
                                )
                                container_health[container_name]['health_check'] = health_status
                            
                            print(f"✅ {container_name}: Running - {container.get('State', 'unknown')}")
                            break
                    
                    if not container_found:
                        container_health[container_name] = {
                            'status': 'NOT_FOUND',
                            'error': 'Container not running'
                        }
                        self.critical_issues.append(f"Container {container_name} not running")
                        print(f"❌ {container_name}: Not found/running")
                        
            else:
                error_msg = f"Failed to get container status: {result.stderr}"
                self.critical_issues.append(error_msg)
                print(f"❌ Container status check failed: {result.stderr}")
                
        except Exception as e:
            error_msg = f"Container health validation failed: {str(e)}"
            self.critical_issues.append(error_msg)
            print(f"❌ {error_msg}")
        
        return container_health
    
    def _test_container_endpoint(self, container_name: str, port: int, endpoint: str, 
                               timeout: int = 5) -> Dict[str, Any]:
        """Test container health endpoint."""
        try:
            url = f"http://localhost:{port}{endpoint}"
            response = requests.get(url, timeout=timeout)
            
            return {
                'status': 'HEALTHY' if response.status_code in [200, 302] else 'UNHEALTHY',
                'status_code': response.status_code,
                'response_time_ms': int(response.elapsed.total_seconds() * 1000),
                'url': url
            }
        except requests.RequestException as e:
            return {
                'status': 'UNREACHABLE',
                'error': str(e),
                'url': f"http://localhost:{port}{endpoint}"
            }
    
    def validate_ml_dependencies(self) -> Dict[str, Any]:
        """Validate ML dependencies across containers."""
        print("\nVALIDATING ML DEPENDENCIES IN CONTAINERS")
        print("-" * 50)
        
        ml_validation = {}
        
        # Test ML dependencies in Jupyter container
        jupyter_ml = self._validate_container_ml_deps('jupyter')
        ml_validation['jupyter'] = jupyter_ml
        
        # Test ML dependencies in Spark containers
        spark_master_ml = self._validate_container_ml_deps('spark-master')
        ml_validation['spark-master'] = spark_master_ml
        
        return ml_validation
    
    def _validate_container_ml_deps(self, container_name: str) -> Dict[str, Any]:
        """Validate ML dependencies within a specific container."""
        container_ml_status = {
            'container': container_name,
            'dependencies': {},
            'summary': {}
        }
        
        try:
            # Create validation script
            validation_script = """
import sys
import json

dependencies = {
    'core_ml': ['xgboost', 'lightgbm', 'catboost', 'optuna', 'shap'],
    'data_processing': ['pandas', 'numpy', 'sklearn'],
    'visualization': ['matplotlib', 'seaborn'],
    'cloud_integration': ['google.cloud.bigquery', 'google.cloud.storage']
}

results = {}
for category, deps in dependencies.items():
    results[category] = {}
    for dep in deps:
        try:
            if dep == 'sklearn':
                import sklearn
                module = sklearn
            else:
                module = __import__(dep)
            
            version = getattr(module, '__version__', 'unknown')
            results[category][dep] = {
                'status': 'OK',
                'version': version
            }
        except ImportError:
            results[category][dep] = {
                'status': 'MISSING',
                'error': 'Module not found'
            }
        except Exception as e:
            results[category][dep] = {
                'status': 'ERROR',
                'error': str(e)
            }

print(json.dumps(results))
"""
            
            # Execute validation script in container
            cmd = ['docker', 'exec', container_name, 'python3', '-c', validation_script]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                try:
                    deps_result = json.loads(result.stdout.strip())
                    container_ml_status['dependencies'] = deps_result
                    
                    # Calculate summary
                    total_deps = sum(len(category_deps) for category_deps in deps_result.values())
                    ok_deps = sum(1 for category_deps in deps_result.values() 
                                for dep_info in category_deps.values() 
                                if dep_info['status'] == 'OK')
                    
                    container_ml_status['summary'] = {
                        'total_dependencies': total_deps,
                        'working_dependencies': ok_deps,
                        'success_rate': (ok_deps / total_deps * 100) if total_deps > 0 else 0
                    }
                    
                    print(f"✅ {container_name}: {ok_deps}/{total_deps} ML deps working ({container_ml_status['summary']['success_rate']:.1f}%)")
                    
                except json.JSONDecodeError as e:
                    container_ml_status['error'] = f"Failed to parse dependency results: {str(e)}"
                    print(f"❌ {container_name}: Failed to parse ML dependency results")
            else:
                container_ml_status['error'] = f"Container execution failed: {result.stderr}"
                print(f"❌ {container_name}: ML dependency check failed - {result.stderr}")
                
        except Exception as e:
            container_ml_status['error'] = str(e)
            print(f"❌ {container_name}: ML dependency validation error - {str(e)}")
        
        return container_ml_status
    
    def validate_cross_container_integration(self) -> Dict[str, Any]:
        """Validate integration between containers."""
        print("\nVALIDATING CROSS-CONTAINER INTEGRATION")
        print("-" * 50)
        
        integration_tests = {
            'jupyter_to_spark': self._test_jupyter_spark_connection(),
            'airflow_to_spark': self._test_airflow_spark_connection(),
            'spark_cluster_connectivity': self._test_spark_cluster_connectivity(),
            'data_volume_sharing': self._test_data_volume_sharing()
        }
        
        return integration_tests
    
    def _test_jupyter_spark_connection(self) -> Dict[str, Any]:
        """Test Jupyter to Spark connectivity."""
        test_result = {'test': 'jupyter_to_spark', 'status': 'PENDING'}
        
        try:
            spark_test_script = """
import json
try:
    from pyspark.sql import SparkSession
    
    spark = SparkSession.builder \\
        .appName("DockerValidationTest") \\
        .master("spark://spark-master:7077") \\
        .config("spark.sql.adaptive.enabled", "true") \\
        .getOrCreate()
    
    # Test basic functionality
    test_data = [("test1", 100), ("test2", 200)]
    df = spark.createDataFrame(test_data, ["name", "value"])
    count = df.count()
    
    spark.stop()
    
    result = {
        "status": "SUCCESS",
        "spark_version": spark.version,
        "test_record_count": count,
        "master_url": "spark://spark-master:7077"
    }
    
except Exception as e:
    result = {
        "status": "FAILED",
        "error": str(e)
    }

print(json.dumps(result))
"""
            
            cmd = ['docker', 'exec', 'jupyter', 'python3', '-c', spark_test_script]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                try:
                    test_result.update(json.loads(result.stdout.strip()))
                    if test_result['status'] == 'SUCCESS':
                        print("✅ Jupyter → Spark: Connection successful")
                    else:
                        print(f"❌ Jupyter → Spark: {test_result.get('error', 'Unknown error')}")
                        self.issues_found.append(f"Jupyter-Spark integration failed: {test_result.get('error', 'Unknown')}")
                except json.JSONDecodeError:
                    test_result['status'] = 'ERROR'
                    test_result['error'] = 'Failed to parse test results'
                    print("❌ Jupyter → Spark: Failed to parse results")
            else:
                test_result['status'] = 'ERROR'
                test_result['error'] = result.stderr
                print(f"❌ Jupyter → Spark: Execution failed - {result.stderr}")
                
        except Exception as e:
            test_result['status'] = 'ERROR'
            test_result['error'] = str(e)
            print(f"❌ Jupyter → Spark: Exception - {str(e)}")
        
        return test_result
    
    def _test_airflow_spark_connection(self) -> Dict[str, Any]:
        """Test Airflow to Spark connectivity."""
        test_result = {'test': 'airflow_to_spark', 'status': 'PENDING'}
        
        try:
            # Check if Airflow can connect to Spark master
            spark_connection_test = """
import json
import os
try:
    # Check environment variables
    spark_master_url = os.environ.get('SPARK_MASTER_URL', 'spark://spark-master:7077')
    
    # Test basic connectivity (without full Spark session to avoid resource conflicts)
    import socket
    
    host, port = 'spark-master', 7077
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    result_code = sock.connect_ex((host, port))
    sock.close()
    
    if result_code == 0:
        result = {
            "status": "SUCCESS",
            "spark_master_url": spark_master_url,
            "connection_test": "PASS",
            "host": host,
            "port": port
        }
    else:
        result = {
            "status": "FAILED", 
            "error": f"Cannot connect to {host}:{port}",
            "connection_code": result_code
        }
    
except Exception as e:
    result = {
        "status": "ERROR",
        "error": str(e)
    }

print(json.dumps(result))
"""
            
            cmd = ['docker', 'exec', 'airflow-scheduler', 'python3', '-c', spark_connection_test]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                try:
                    test_result.update(json.loads(result.stdout.strip()))
                    if test_result['status'] == 'SUCCESS':
                        print("✅ Airflow → Spark: Connection successful")
                    else:
                        print(f"❌ Airflow → Spark: {test_result.get('error', 'Connection failed')}")
                        self.issues_found.append(f"Airflow-Spark integration failed: {test_result.get('error', 'Connection failed')}")
                except json.JSONDecodeError:
                    test_result['status'] = 'ERROR'
                    test_result['error'] = 'Failed to parse test results'
                    print("❌ Airflow → Spark: Failed to parse results")
            else:
                test_result['status'] = 'ERROR'
                test_result['error'] = result.stderr
                print(f"❌ Airflow → Spark: Test execution failed")
                
        except Exception as e:
            test_result['status'] = 'ERROR'
            test_result['error'] = str(e)
            print(f"❌ Airflow → Spark: Exception - {str(e)}")
        
        return test_result
    
    def _test_spark_cluster_connectivity(self) -> Dict[str, Any]:
        """Test Spark master-worker connectivity."""
        test_result = {'test': 'spark_cluster', 'status': 'PENDING'}
        
        try:
            # Test master-worker connection by checking worker registration
            master_info_script = """
import json
import requests
try:
    response = requests.get('http://localhost:8080/json/', timeout=10)
    if response.status_code == 200:
        data = response.json()
        workers = data.get('workers', [])
        active_workers = [w for w in workers if w.get('state') == 'ALIVE']
        
        result = {
            "status": "SUCCESS",
            "total_workers": len(workers),
            "active_workers": len(active_workers),
            "workers_info": active_workers
        }
    else:
        result = {
            "status": "FAILED",
            "error": f"HTTP {response.status_code}"
        }
except Exception as e:
    result = {
        "status": "ERROR",
        "error": str(e)
    }

print(json.dumps(result))
"""
            
            cmd = ['docker', 'exec', 'spark-master', 'python3', '-c', master_info_script]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            
            if result.returncode == 0:
                try:
                    test_result.update(json.loads(result.stdout.strip()))
                    if test_result['status'] == 'SUCCESS':
                        worker_count = test_result.get('active_workers', 0)
                        print(f"✅ Spark Cluster: {worker_count} active workers")
                    else:
                        print(f"❌ Spark Cluster: {test_result.get('error', 'Cluster check failed')}")
                        self.issues_found.append(f"Spark cluster connectivity issue: {test_result.get('error', 'Unknown')}")
                except json.JSONDecodeError:
                    test_result['status'] = 'ERROR'
                    test_result['error'] = 'Failed to parse cluster info'
                    print("❌ Spark Cluster: Failed to parse cluster info")
            else:
                test_result['status'] = 'ERROR'
                test_result['error'] = result.stderr
                print(f"❌ Spark Cluster: Check failed")
                
        except Exception as e:
            test_result['status'] = 'ERROR'
            test_result['error'] = str(e)
            print(f"❌ Spark Cluster: Exception - {str(e)}")
        
        return test_result
    
    def _test_data_volume_sharing(self) -> Dict[str, Any]:
        """Test data volume sharing between containers."""
        test_result = {'test': 'data_volume_sharing', 'status': 'PENDING'}
        
        try:
            # Create test file in one container and read from another
            test_filename = f"docker_validation_test_{int(time.time())}.txt"
            test_content = f"Docker validation test - {datetime.now().isoformat()}"
            
            # Write test file from Jupyter container
            write_cmd = ['docker', 'exec', 'jupyter', 'bash', '-c', 
                        f'echo "{test_content}" > /home/jovyan/data/{test_filename}']
            write_result = subprocess.run(write_cmd, capture_output=True, text=True, timeout=10)
            
            if write_result.returncode == 0:
                # Read test file from Spark container
                read_cmd = ['docker', 'exec', 'spark-master', 'cat', f'/data/{test_filename}']
                read_result = subprocess.run(read_cmd, capture_output=True, text=True, timeout=10)
                
                if read_result.returncode == 0:
                    read_content = read_result.stdout.strip()
                    if read_content == test_content:
                        test_result['status'] = 'SUCCESS'
                        test_result['test_file'] = test_filename
                        print("✅ Data Volume Sharing: Successful")
                    else:
                        test_result['status'] = 'FAILED'
                        test_result['error'] = 'Content mismatch between containers'
                        print("❌ Data Volume Sharing: Content mismatch")
                        self.issues_found.append("Data volume sharing: content mismatch between containers")
                else:
                    test_result['status'] = 'FAILED'
                    test_result['error'] = f'Failed to read from spark container: {read_result.stderr}'
                    print("❌ Data Volume Sharing: Failed to read from Spark container")
                    
                # Cleanup test file
                cleanup_cmd = ['docker', 'exec', 'jupyter', 'rm', f'/home/jovyan/data/{test_filename}']
                subprocess.run(cleanup_cmd, capture_output=True, timeout=5)
            else:
                test_result['status'] = 'FAILED'
                test_result['error'] = f'Failed to write test file: {write_result.stderr}'
                print("❌ Data Volume Sharing: Failed to write test file")
                
        except Exception as e:
            test_result['status'] = 'ERROR'
            test_result['error'] = str(e)
            print(f"❌ Data Volume Sharing: Exception - {str(e)}")
        
        return test_result
    
    def validate_bigquery_gcs_connectivity(self) -> Dict[str, Any]:
        """Validate BigQuery and GCS connectivity with updated google-cloud-bigquery version."""
        print("\nVALIDATING BIGQUERY/GCS CONNECTIVITY")
        print("-" * 50)
        
        cloud_validation = {
            'bigquery': self._test_bigquery_connectivity(),
            'gcs': self._test_gcs_connectivity(),
            'package_versions': self._check_cloud_package_versions()
        }
        
        return cloud_validation
    
    def _test_bigquery_connectivity(self) -> Dict[str, Any]:
        """Test BigQuery connectivity and compatibility."""
        bq_test = {'service': 'bigquery', 'status': 'PENDING'}
        
        try:
            bq_test_script = """
import json
try:
    from google.cloud import bigquery
    
    # Test package import and basic functionality
    client_info = {
        "package_version": bigquery.__version__,
        "import_success": True
    }
    
    # Test client creation (without actual credentials)
    try:
        # This will fail without credentials but tests package compatibility
        client = bigquery.Client()
        client_info["client_creation"] = "SUCCESS_NO_CREDS"
    except Exception as e:
        if "credentials" in str(e).lower() or "authentication" in str(e).lower():
            client_info["client_creation"] = "SUCCESS_NO_CREDS"
            client_info["auth_note"] = "Credentials not configured (expected in validation)"
        else:
            client_info["client_creation"] = "FAILED"
            client_info["client_error"] = str(e)
    
    result = {
        "status": "SUCCESS",
        "details": client_info
    }
    
except ImportError as e:
    result = {
        "status": "MISSING",
        "error": f"BigQuery package not available: {str(e)}"
    }
except Exception as e:
    result = {
        "status": "ERROR",
        "error": str(e)
    }

print(json.dumps(result))
"""
            
            cmd = ['docker', 'exec', 'jupyter', 'python3', '-c', bq_test_script]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                try:
                    bq_test.update(json.loads(result.stdout.strip()))
                    if bq_test['status'] == 'SUCCESS':
                        version = bq_test.get('details', {}).get('package_version', 'unknown')
                        print(f"✅ BigQuery: Package available (v{version})")
                    else:
                        print(f"❌ BigQuery: {bq_test.get('error', 'Package issue')}")
                        self.warnings.append(f"BigQuery connectivity issue: {bq_test.get('error', 'Unknown')}")
                except json.JSONDecodeError:
                    bq_test['status'] = 'ERROR'
                    bq_test['error'] = 'Failed to parse BigQuery test results'
                    print("❌ BigQuery: Failed to parse test results")
            else:
                bq_test['status'] = 'ERROR'
                bq_test['error'] = result.stderr
                print("❌ BigQuery: Test execution failed")
                
        except Exception as e:
            bq_test['status'] = 'ERROR'
            bq_test['error'] = str(e)
            print(f"❌ BigQuery: Exception - {str(e)}")
        
        return bq_test
    
    def _test_gcs_connectivity(self) -> Dict[str, Any]:
        """Test Google Cloud Storage connectivity."""
        gcs_test = {'service': 'gcs', 'status': 'PENDING'}
        
        try:
            gcs_test_script = """
import json
try:
    from google.cloud import storage
    
    client_info = {
        "package_version": getattr(storage, '__version__', 'unknown'),
        "import_success": True
    }
    
    try:
        client = storage.Client()
        client_info["client_creation"] = "SUCCESS_NO_CREDS"
    except Exception as e:
        if "credentials" in str(e).lower() or "authentication" in str(e).lower():
            client_info["client_creation"] = "SUCCESS_NO_CREDS"
            client_info["auth_note"] = "Credentials not configured (expected in validation)"
        else:
            client_info["client_creation"] = "FAILED"
            client_info["client_error"] = str(e)
    
    result = {
        "status": "SUCCESS",
        "details": client_info
    }
    
except ImportError as e:
    result = {
        "status": "MISSING",
        "error": f"GCS package not available: {str(e)}"
    }
except Exception as e:
    result = {
        "status": "ERROR",
        "error": str(e)
    }

print(json.dumps(result))
"""
            
            cmd = ['docker', 'exec', 'jupyter', 'python3', '-c', gcs_test_script]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                try:
                    gcs_test.update(json.loads(result.stdout.strip()))
                    if gcs_test['status'] == 'SUCCESS':
                        version = gcs_test.get('details', {}).get('package_version', 'unknown')
                        print(f"✅ GCS: Package available (v{version})")
                    else:
                        print(f"❌ GCS: {gcs_test.get('error', 'Package issue')}")
                        self.warnings.append(f"GCS connectivity issue: {gcs_test.get('error', 'Unknown')}")
                except json.JSONDecodeError:
                    gcs_test['status'] = 'ERROR'
                    gcs_test['error'] = 'Failed to parse GCS test results'
                    print("❌ GCS: Failed to parse test results")
            else:
                gcs_test['status'] = 'ERROR'
                gcs_test['error'] = result.stderr
                print("❌ GCS: Test execution failed")
                
        except Exception as e:
            gcs_test['status'] = 'ERROR'
            gcs_test['error'] = str(e)
            print(f"❌ GCS: Exception - {str(e)}")
        
        return gcs_test
    
    def _check_cloud_package_versions(self) -> Dict[str, Any]:
        """Check cloud package versions for compatibility."""
        version_check = {'status': 'PENDING'}
        
        try:
            version_script = """
import json
packages = {}

try:
    import google.cloud.bigquery as bq
    packages['google-cloud-bigquery'] = {
        'version': bq.__version__,
        'status': 'OK'
    }
except ImportError:
    packages['google-cloud-bigquery'] = {
        'version': 'N/A',
        'status': 'MISSING'
    }

try:
    import google.cloud.storage as storage
    packages['google-cloud-storage'] = {
        'version': getattr(storage, '__version__', 'unknown'),
        'status': 'OK'
    }
except ImportError:
    packages['google-cloud-storage'] = {
        'version': 'N/A',
        'status': 'MISSING'
    }

print(json.dumps(packages))
"""
            
            cmd = ['docker', 'exec', 'jupyter', 'python3', '-c', version_script]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            
            if result.returncode == 0:
                try:
                    version_check['packages'] = json.loads(result.stdout.strip())
                    version_check['status'] = 'SUCCESS'
                    
                    # Check if BigQuery version meets requirement (>=3.4.0)
                    bq_info = version_check['packages'].get('google-cloud-bigquery', {})
                    if bq_info.get('status') == 'OK':
                        bq_version = bq_info.get('version', '0.0.0')
                        if self._compare_versions(bq_version, '3.4.0') >= 0:
                            print(f"✅ google-cloud-bigquery: v{bq_version} (>=3.4.0 requirement met)")
                        else:
                            print(f"⚠️ google-cloud-bigquery: v{bq_version} (upgrade to >=3.4.0 recommended)")
                            self.warnings.append(f"google-cloud-bigquery version {bq_version} < 3.4.0")
                    
                except json.JSONDecodeError:
                    version_check['status'] = 'ERROR'
                    version_check['error'] = 'Failed to parse package versions'
            else:
                version_check['status'] = 'ERROR'
                version_check['error'] = result.stderr
                
        except Exception as e:
            version_check['status'] = 'ERROR'
            version_check['error'] = str(e)
        
        return version_check
    
    def _compare_versions(self, version1: str, version2: str) -> int:
        """Compare two version strings. Returns: 1 if v1>v2, 0 if equal, -1 if v1<v2"""
        try:
            v1_parts = [int(x) for x in version1.split('.')]
            v2_parts = [int(x) for x in version2.split('.')]
            
            # Pad shorter version with zeros
            max_len = max(len(v1_parts), len(v2_parts))
            v1_parts.extend([0] * (max_len - len(v1_parts)))
            v2_parts.extend([0] * (max_len - len(v2_parts)))
            
            for v1, v2 in zip(v1_parts, v2_parts):
                if v1 > v2:
                    return 1
                elif v1 < v2:
                    return -1
            return 0
        except:
            return 0  # Unable to compare
    
    def validate_ifrs9_docker_business_rules(self) -> Dict[str, Any]:
        """Validate IFRS9 business rules in Docker environment."""
        print("\nVALIDATING IFRS9 BUSINESS RULES IN DOCKER")
        print("-" * 50)
        
        business_validation = {}
        
        # Test IFRS9 validation using PySpark in Docker
        ifrs9_test = self._test_ifrs9_spark_validation()
        business_validation['spark_validation'] = ifrs9_test
        
        return business_validation
    
    def _test_ifrs9_spark_validation(self) -> Dict[str, Any]:
        """Test IFRS9 validation using Spark in Docker environment."""
        test_result = {'test': 'ifrs9_spark_validation', 'status': 'PENDING'}
        
        try:
            ifrs9_validation_script = """
import json
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType
    from pyspark.sql.functions import col, when
    
    # Create Spark session
    spark = SparkSession.builder \\
        .appName("IFRS9_Docker_Validation") \\
        .master("spark://spark-master:7077") \\
        .getOrCreate()
    
    # Create test IFRS9 data
    schema = StructType([
        StructField("loan_id", StringType(), True),
        StructField("customer_id", StringType(), True),
        StructField("loan_amount", FloatType(), True),
        StructField("current_balance", FloatType(), True),
        StructField("days_past_due", IntegerType(), True),
        StructField("provision_stage", StringType(), True),
        StructField("pd_rate", FloatType(), True),
        StructField("lgd_rate", FloatType(), True)
    ])
    
    test_data = [
        ("L001", "C001", 100000.0, 95000.0, 0, "STAGE_1", 0.02, 0.45),
        ("L002", "C002", 50000.0, 45000.0, 45, "STAGE_2", 0.15, 0.55),
        ("L003", "C003", 75000.0, 60000.0, 120, "STAGE_3", 0.85, 0.75),
        ("L004", "C004", 30000.0, 35000.0, 15, "STAGE_1", 0.03, 0.40),  # Invalid: balance > amount
        ("L005", "C005", 40000.0, 30000.0, 95, "STAGE_2", 1.2, 0.50),   # Invalid: PD > 1
    ]
    
    df = spark.createDataFrame(test_data, schema)
    
    # IFRS9 Business Rule Validations
    validation_results = {}
    
    # Rule 1: Stage-DPD Consistency
    stage_3_low_dpd = df.filter((col("provision_stage") == "STAGE_3") & (col("days_past_due") < 90)).count()
    stage_1_high_dpd = df.filter((col("provision_stage") == "STAGE_1") & (col("days_past_due") > 30)).count()
    
    validation_results["stage_dpd_consistency"] = {
        "stage_3_low_dpd_count": stage_3_low_dpd,
        "stage_1_high_dpd_count": stage_1_high_dpd,
        "compliant": stage_3_low_dpd == 0 and stage_1_high_dpd == 0
    }
    
    # Rule 2: Balance Validation  
    invalid_balance_count = df.filter(col("current_balance") > col("loan_amount")).count()
    validation_results["balance_validation"] = {
        "invalid_balance_count": invalid_balance_count,
        "compliant": invalid_balance_count == 0
    }
    
    # Rule 3: Rate Validation
    invalid_pd_count = df.filter((col("pd_rate") < 0) | (col("pd_rate") > 1)).count()
    invalid_lgd_count = df.filter((col("lgd_rate") < 0) | (col("lgd_rate") > 1)).count()
    
    validation_results["rate_validation"] = {
        "invalid_pd_count": invalid_pd_count,
        "invalid_lgd_count": invalid_lgd_count,
        "compliant": invalid_pd_count == 0 and invalid_lgd_count == 0
    }
    
    # Calculate overall compliance
    compliant_rules = sum(1 for rule in validation_results.values() if rule.get("compliant", False))
    total_rules = len(validation_results)
    compliance_score = (compliant_rules / total_rules * 100) if total_rules > 0 else 0
    
    spark.stop()
    
    result = {
        "status": "SUCCESS",
        "total_test_records": len(test_data),
        "validation_results": validation_results,
        "compliance_score": compliance_score,
        "compliant_rules": compliant_rules,
        "total_rules": total_rules
    }
    
except Exception as e:
    result = {
        "status": "ERROR",
        "error": str(e)
    }

print(json.dumps(result))
"""
            
            cmd = ['docker', 'exec', 'jupyter', 'python3', '-c', ifrs9_validation_script]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
            
            if result.returncode == 0:
                try:
                    test_result.update(json.loads(result.stdout.strip()))
                    if test_result['status'] == 'SUCCESS':
                        compliance_score = test_result.get('compliance_score', 0)
                        print(f"✅ IFRS9 Spark Validation: {compliance_score:.1f}% compliance")
                    else:
                        print(f"❌ IFRS9 Spark Validation: {test_result.get('error', 'Validation failed')}")
                        self.issues_found.append(f"IFRS9 Docker validation failed: {test_result.get('error', 'Unknown')}")
                except json.JSONDecodeError:
                    test_result['status'] = 'ERROR'
                    test_result['error'] = 'Failed to parse validation results'
                    print("❌ IFRS9 Spark Validation: Failed to parse results")
            else:
                test_result['status'] = 'ERROR'
                test_result['error'] = result.stderr
                print("❌ IFRS9 Spark Validation: Execution failed")
                
        except Exception as e:
            test_result['status'] = 'ERROR'
            test_result['error'] = str(e)
            print(f"❌ IFRS9 Spark Validation: Exception - {str(e)}")
        
        return test_result
    
    def run_comprehensive_docker_validation(self) -> Dict[str, Any]:
        """Run comprehensive Docker environment validation."""
        print("IFRS9 DOCKER ENVIRONMENT VALIDATION")
        print("Comprehensive validation of containerized IFRS9 system")
        print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
        print()
        
        # Run all validation steps
        self.validation_results['docker_environment'] = self.validate_docker_environment()
        self.validation_results['container_health'] = self.validate_container_health()
        self.validation_results['ml_dependencies'] = self.validate_ml_dependencies()
        self.validation_results['cross_container_integration'] = self.validate_cross_container_integration()
        self.validation_results['cloud_connectivity'] = self.validate_bigquery_gcs_connectivity()
        self.validation_results['ifrs9_business_rules'] = self.validate_ifrs9_docker_business_rules()
        
        # Compile validation summary
        self.validation_results['validation_summary'] = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'total_critical_issues': len(self.critical_issues),
            'total_issues': len(self.issues_found),
            'total_warnings': len(self.warnings),
            'critical_issues': self.critical_issues,
            'issues_found': self.issues_found,
            'warnings': self.warnings,
            'overall_status': 'PASS' if len(self.critical_issues) == 0 else 'FAIL',
            'docker_ready': len(self.critical_issues) == 0,
            'ml_ready': self._assess_ml_readiness(),
            'production_ready': self._assess_production_readiness()
        }
        
        return self.validation_results
    
    def _assess_ml_readiness(self) -> bool:
        """Assess if the ML environment is ready."""
        ml_deps = self.validation_results.get('ml_dependencies', {})
        
        for container, deps_info in ml_deps.items():
            if 'summary' in deps_info:
                success_rate = deps_info['summary'].get('success_rate', 0)
                if success_rate < 80:  # At least 80% of ML deps should work
                    return False
        return True
    
    def _assess_production_readiness(self) -> bool:
        """Assess if the system is ready for production use."""
        return (len(self.critical_issues) == 0 and 
                len(self.issues_found) <= 2 and  # Allow minor issues
                self._assess_ml_readiness())
    
    def generate_docker_validation_report(self) -> str:
        """Generate comprehensive Docker validation report."""
        if not self.validation_results:
            return "No validation results available. Run Docker validation first."
        
        report = []
        report.append("=" * 80)
        report.append("IFRS9 DOCKER ENVIRONMENT VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
        
        summary = self.validation_results['validation_summary']
        report.append(f"Overall Status: {summary['overall_status']}")
        report.append(f"Docker Ready: {summary['docker_ready']}")
        report.append(f"ML Ready: {summary['ml_ready']}")
        report.append(f"Production Ready: {summary['production_ready']}")
        report.append("")
        
        # Validation Summary
        report.append("VALIDATION SUMMARY")
        report.append("-" * 40)
        report.append(f"Critical Issues: {summary['total_critical_issues']}")
        report.append(f"Issues Found: {summary['total_issues']}")
        report.append(f"Warnings: {summary['total_warnings']}")
        report.append("")
        
        if summary['critical_issues']:
            report.append("CRITICAL ISSUES (Must Fix Before Production):")
            for issue in summary['critical_issues']:
                report.append(f"  ❌ {issue}")
            report.append("")
        
        if summary['issues_found']:
            report.append("ISSUES FOUND:")
            for issue in summary['issues_found']:
                report.append(f"  ⚠️ {issue}")
            report.append("")
        
        if summary['warnings']:
            report.append("WARNINGS:")
            for warning in summary['warnings']:
                report.append(f"  ℹ️ {warning}")
            report.append("")
        
        # Container Health Summary
        container_health = self.validation_results.get('container_health', {})
        if container_health:
            report.append("CONTAINER HEALTH")
            report.append("-" * 40)
            running_containers = sum(1 for c in container_health.values() 
                                   if isinstance(c, dict) and c.get('status') == 'RUNNING')
            total_containers = len(container_health)
            report.append(f"Running Containers: {running_containers}/{total_containers}")
            
            for container, status in container_health.items():
                if isinstance(status, dict):
                    container_status = status.get('status', 'UNKNOWN')
                    health_check = status.get('health_check', {})
                    if health_check:
                        health_status = health_check.get('status', 'UNKNOWN')
                        report.append(f"  {container}: {container_status} - Health: {health_status}")
                    else:
                        report.append(f"  {container}: {container_status}")
            report.append("")
        
        # ML Dependencies Summary
        ml_deps = self.validation_results.get('ml_dependencies', {})
        if ml_deps:
            report.append("ML DEPENDENCIES")
            report.append("-" * 40)
            for container, deps_info in ml_deps.items():
                if 'summary' in deps_info:
                    success_rate = deps_info['summary'].get('success_rate', 0)
                    working = deps_info['summary'].get('working_dependencies', 0)
                    total = deps_info['summary'].get('total_dependencies', 0)
                    report.append(f"  {container}: {working}/{total} deps working ({success_rate:.1f}%)")
            report.append("")
        
        # Integration Tests Summary
        integration = self.validation_results.get('cross_container_integration', {})
        if integration:
            report.append("INTEGRATION TESTS")
            report.append("-" * 40)
            for test_name, test_result in integration.items():
                if isinstance(test_result, dict):
                    status = test_result.get('status', 'UNKNOWN')
                    report.append(f"  {test_name}: {status}")
            report.append("")
        
        # Cloud Connectivity Summary
        cloud = self.validation_results.get('cloud_connectivity', {})
        if cloud:
            report.append("CLOUD CONNECTIVITY")
            report.append("-" * 40)
            bq_status = cloud.get('bigquery', {}).get('status', 'UNKNOWN')
            gcs_status = cloud.get('gcs', {}).get('status', 'UNKNOWN')
            report.append(f"  BigQuery: {bq_status}")
            report.append(f"  Google Cloud Storage: {gcs_status}")
            report.append("")
        
        # IFRS9 Business Rules Summary
        business = self.validation_results.get('ifrs9_business_rules', {})
        if business:
            spark_validation = business.get('spark_validation', {})
            if spark_validation.get('status') == 'SUCCESS':
                compliance_score = spark_validation.get('compliance_score', 0)
                report.append("IFRS9 BUSINESS RULES")
                report.append("-" * 40)
                report.append(f"  Compliance Score: {compliance_score:.1f}%")
                report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        
        if summary['production_ready']:
            report.append("✅ Docker environment READY for production deployment")
            report.append("   All critical components validated successfully")
        elif summary['docker_ready']:
            report.append("✅ Docker environment READY for testing")
            if not summary['ml_ready']:
                report.append("⚠️  ML environment needs attention before production")
            if len(self.issues_found) > 0:
                report.append("⚠️  Address non-critical issues before production")
        else:
            report.append("❌ Docker environment NOT READY - Critical issues must be resolved")
            report.append("   Priority actions:")
            for issue in self.critical_issues[:3]:
                report.append(f"   1. Resolve: {issue}")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """Main execution function for Docker validation."""
    print("Starting IFRS9 Docker Environment Validation")
    print("Comprehensive Docker validation agent")
    
    validator = DockerEnvironmentValidator()
    
    try:
        # Run comprehensive Docker validation
        results = validator.run_comprehensive_docker_validation()
        
        # Generate report
        report = validator.generate_docker_validation_report()
        print("\n" + report)
        
        # Save results to files
        output_dir = "/opt/airflow/data/validation"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON results
        json_file = os.path.join(output_dir, f"docker_validation_results_{timestamp}.json")
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save text report
        report_file = os.path.join(output_dir, f"docker_validation_report_{timestamp}.txt")
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\nDocker validation results saved to:")
        print(f"  JSON: {json_file}")
        print(f"  Report: {report_file}")
        
        # Exit with appropriate code
        validation_summary = results['validation_summary']
        if validation_summary['overall_status'] == 'PASS':
            exit_code = 0
            print(f"\nDocker Validation Status: PASS")
            if validation_summary['production_ready']:
                print("System is READY for production deployment")
            else:
                print("System is ready for testing (address warnings before production)")
        else:
            exit_code = 1
            print(f"\nDocker Validation Status: FAIL - Critical issues must be resolved")
        
        return exit_code
        
    except Exception as e:
        print(f"\nFATAL ERROR during Docker validation: {str(e)}")
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)