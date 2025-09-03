"""
IFRS9 Containerized Test Orchestrator
=====================================
Orchestrates validation testing across the entire containerized IFRS9 infrastructure.

This orchestrator manages:
1. Container health and readiness validation
2. Kubernetes resource testing 
3. Service mesh and networking validation
4. Monitoring stack integration testing
5. CI/CD pipeline validation
6. Cloud platform integration testing

Author: IFRS9 Risk System Team  
Version: 1.0.0
Date: 2025-09-03
"""

import asyncio
import yaml
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
import uuid
from kubernetes import client as k8s_client, config as k8s_config
from kubernetes.client.rest import ApiException
import httpx
import psutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass  
class TestResult:
    """Test execution result"""
    test_name: str
    component: str
    status: str  # PASS, FAIL, SKIP, ERROR
    duration_seconds: float
    timestamp: datetime
    details: Dict[str, Any]
    error_message: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None

@dataclass
class ValidationSuite:
    """Validation test suite configuration"""
    name: str
    description: str
    tests: List[str]
    parallel_execution: bool
    timeout_seconds: int
    retry_count: int
    dependencies: List[str]

class ContainerizedTestOrchestrator:
    """Main orchestrator for containerized validation testing"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or "/home/eze/projects/ifrs9-risk-system/validation/test_config.yaml"
        self.results_dir = Path("/home/eze/projects/ifrs9-risk-system/validation/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir = Path("/home/eze/projects/ifrs9-risk-system/validation/datasets")
        
        # Test execution tracking
        self.test_results: List[TestResult] = []
        self.execution_start_time = None
        self.k8s_client = None
        
        # Load test configuration
        self.load_test_configuration()
        
    def load_test_configuration(self):
        """Load test configuration from YAML"""
        config_data = {
            'validation_suites': {
                'container_orchestration': {
                    'name': 'Container Orchestration Validation',
                    'description': 'Validates Kubernetes deployments, services, and HPA',
                    'tests': [
                        'test_pod_health_checks',
                        'test_deployment_rollouts', 
                        'test_service_discovery',
                        'test_hpa_scaling',
                        'test_network_policies',
                        'test_resource_quotas'
                    ],
                    'parallel_execution': True,
                    'timeout_seconds': 300,
                    'retry_count': 2,
                    'dependencies': []
                },
                'infrastructure': {
                    'name': 'Infrastructure Validation',
                    'description': 'Validates ConfigMaps, Secrets, PVCs, and storage',
                    'tests': [
                        'test_configmap_validation',
                        'test_secrets_validation',
                        'test_pvc_operations',
                        'test_storage_performance',
                        'test_backup_restore'
                    ],
                    'parallel_execution': True,
                    'timeout_seconds': 600,
                    'retry_count': 1,
                    'dependencies': ['container_orchestration']
                },
                'monitoring': {
                    'name': 'Monitoring Stack Validation', 
                    'description': 'Validates Prometheus, Grafana, Jaeger, and ELK',
                    'tests': [
                        'test_prometheus_scraping',
                        'test_grafana_dashboards',
                        'test_jaeger_tracing',
                        'test_elk_logging',
                        'test_alerting_rules'
                    ],
                    'parallel_execution': True,
                    'timeout_seconds': 450,
                    'retry_count': 2,
                    'dependencies': ['container_orchestration']
                },
                'cicd': {
                    'name': 'CI/CD Pipeline Validation',
                    'description': 'Validates Docker builds, Helm deployments, ArgoCD',
                    'tests': [
                        'test_docker_builds',
                        'test_helm_deployments', 
                        'test_argocd_sync',
                        'test_gitops_workflow',
                        'test_security_scanning'
                    ],
                    'parallel_execution': False,
                    'timeout_seconds': 900,
                    'retry_count': 1,
                    'dependencies': ['infrastructure']
                },
                'cloud_integration': {
                    'name': 'Cloud Integration Validation',
                    'description': 'Validates GKE, BigQuery, Cloud Storage integration',
                    'tests': [
                        'test_gke_cluster_health',
                        'test_bigquery_connectivity',
                        'test_cloud_storage_operations',
                        'test_iam_workload_identity',
                        'test_cloud_sql_proxy'
                    ],
                    'parallel_execution': True,
                    'timeout_seconds': 300,
                    'retry_count': 2,
                    'dependencies': ['container_orchestration', 'infrastructure']
                }
            },
            'global_settings': {
                'max_parallel_suites': 3,
                'result_retention_days': 30,
                'detailed_logging': True,
                'export_prometheus_metrics': True,
                'notification_webhook': 'https://hooks.slack.com/services/...'  # Configure as needed
            }
        }
        
        # Save configuration to file
        config_path = Path(self.config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
            
        self.config = config_data
        logger.info(f"Test configuration loaded with {len(config_data['validation_suites'])} test suites")

    async def initialize_k8s_client(self):
        """Initialize Kubernetes client"""
        try:
            # Try in-cluster config first, then local config
            try:
                k8s_config.load_incluster_config()
                logger.info("Using in-cluster Kubernetes configuration")
            except k8s_config.ConfigException:
                k8s_config.load_kube_config()
                logger.info("Using local kubeconfig")
                
            self.k8s_client = k8s_client.ApiClient()
            
        except Exception as e:
            logger.error(f"Failed to initialize Kubernetes client: {e}")
            self.k8s_client = None

    async def run_validation_suite(self, suite_name: str) -> List[TestResult]:
        """Run a complete validation suite"""
        suite_config = self.config['validation_suites'][suite_name]
        logger.info(f"Starting validation suite: {suite_config['name']}")
        
        suite_results = []
        start_time = time.time()
        
        # Check dependencies
        if not await self.check_dependencies(suite_config.get('dependencies', [])):
            error_result = TestResult(
                test_name=f"{suite_name}_dependencies",
                component=suite_name,
                status="FAIL",
                duration_seconds=0,
                timestamp=datetime.now(),
                details={'error': 'Dependencies not met'},
                error_message="Required dependencies not satisfied"
            )
            return [error_result]
        
        tests = suite_config['tests']
        
        if suite_config['parallel_execution']:
            # Run tests in parallel
            with ThreadPoolExecutor(max_workers=min(len(tests), 5)) as executor:
                futures = {
                    executor.submit(
                        self.run_single_test, 
                        test_name, 
                        suite_name,
                        suite_config['timeout_seconds'],
                        suite_config['retry_count']
                    ): test_name for test_name in tests
                }
                
                for future in as_completed(futures):
                    test_name = futures[future]
                    try:
                        result = future.result()
                        suite_results.append(result)
                        logger.info(f"Test {test_name} completed: {result.status}")
                    except Exception as e:
                        error_result = TestResult(
                            test_name=test_name,
                            component=suite_name,
                            status="ERROR",
                            duration_seconds=0,
                            timestamp=datetime.now(),
                            details={'exception': str(e)},
                            error_message=f"Test execution failed: {e}"
                        )
                        suite_results.append(error_result)
        else:
            # Run tests sequentially
            for test_name in tests:
                result = self.run_single_test(
                    test_name,
                    suite_name, 
                    suite_config['timeout_seconds'],
                    suite_config['retry_count']
                )
                suite_results.append(result)
                logger.info(f"Test {test_name} completed: {result.status}")
                
                # Stop if critical test fails
                if result.status == "FAIL" and "critical" in test_name.lower():
                    logger.warning(f"Critical test {test_name} failed, stopping suite")
                    break
        
        duration = time.time() - start_time
        logger.info(f"Validation suite {suite_name} completed in {duration:.2f}s")
        
        return suite_results

    def run_single_test(self, test_name: str, component: str, timeout: int, retry_count: int) -> TestResult:
        """Run a single validation test"""
        start_time = time.time()
        
        for attempt in range(retry_count + 1):
            try:
                # Route to appropriate test method
                if hasattr(self, test_name):
                    test_method = getattr(self, test_name)
                    result = test_method()
                    
                    if result['status'] == 'PASS':
                        return TestResult(
                            test_name=test_name,
                            component=component,
                            status="PASS",
                            duration_seconds=time.time() - start_time,
                            timestamp=datetime.now(),
                            details=result['details'],
                            metrics=result.get('metrics')
                        )
                    elif attempt < retry_count:
                        logger.warning(f"Test {test_name} failed, retrying ({attempt + 1}/{retry_count})")
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        return TestResult(
                            test_name=test_name,
                            component=component,
                            status="FAIL",
                            duration_seconds=time.time() - start_time,
                            timestamp=datetime.now(),
                            details=result['details'],
                            error_message=result.get('error_message')
                        )
                else:
                    return TestResult(
                        test_name=test_name,
                        component=component,
                        status="SKIP",
                        duration_seconds=time.time() - start_time,
                        timestamp=datetime.now(),
                        details={'reason': 'Test method not implemented'},
                        error_message=f"Test method {test_name} not found"
                    )
                    
            except Exception as e:
                if attempt < retry_count:
                    logger.warning(f"Test {test_name} error, retrying ({attempt + 1}/{retry_count}): {e}")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    return TestResult(
                        test_name=test_name,
                        component=component,
                        status="ERROR",
                        duration_seconds=time.time() - start_time,
                        timestamp=datetime.now(),
                        details={'exception': str(e)},
                        error_message=f"Test execution failed: {e}"
                    )
        
        # Should never reach here
        return TestResult(
            test_name=test_name,
            component=component,
            status="ERROR",
            duration_seconds=time.time() - start_time,
            timestamp=datetime.now(),
            details={'error': 'Unexpected test completion'},
            error_message="Test completed unexpectedly"
        )

    async def check_dependencies(self, dependencies: List[str]) -> bool:
        """Check if test suite dependencies are satisfied"""
        if not dependencies:
            return True
            
        for dep in dependencies:
            # Check if dependency suite has been run successfully
            dep_results = [r for r in self.test_results if r.component == dep]
            if not dep_results:
                logger.error(f"Dependency {dep} has not been run")
                return False
                
            # Check if any critical tests failed
            failed_critical = [r for r in dep_results if r.status == "FAIL" and "critical" in r.test_name.lower()]
            if failed_critical:
                logger.error(f"Dependency {dep} has critical test failures")
                return False
                
        return True

    # Container Orchestration Tests
    def test_pod_health_checks(self) -> Dict[str, Any]:
        """Test pod health checks using validation dataset"""
        try:
            health_data = pd.read_csv(self.datasets_dir / "container_health_checks.csv")
            
            # Analyze health check data
            total_checks = len(health_data)
            healthy_pods = len(health_data[health_data['ready'] == True])
            avg_response_time = health_data['response_time_ms'].mean()
            high_memory_pods = len(health_data[health_data['memory_usage_mb'] > 800])
            
            # Validate against thresholds
            health_ratio = healthy_pods / total_checks
            success = (
                health_ratio >= 0.95 and  # 95% pods should be healthy
                avg_response_time <= 200 and  # Average response time under 200ms
                high_memory_pods / total_checks <= 0.1  # Less than 10% high memory usage
            )
            
            return {
                'status': 'PASS' if success else 'FAIL',
                'details': {
                    'total_checks': total_checks,
                    'healthy_pods': healthy_pods,
                    'health_ratio': health_ratio,
                    'avg_response_time_ms': avg_response_time,
                    'high_memory_pods': high_memory_pods
                },
                'metrics': {
                    'health_ratio': health_ratio,
                    'avg_response_time': avg_response_time,
                    'memory_usage_p95': health_data['memory_usage_mb'].quantile(0.95)
                }
            }
        except Exception as e:
            return {
                'status': 'ERROR',
                'details': {'error': str(e)},
                'error_message': f"Health check test failed: {e}"
            }

    def test_deployment_rollouts(self) -> Dict[str, Any]:
        """Test deployment rollout validation"""
        try:
            k8s_data = pd.read_csv(self.datasets_dir / "k8s_resources_validation.csv")
            
            # Filter deployment data
            deployments = k8s_data[k8s_data['resource_type'] == 'Deployment']
            
            ready_deployments = len(deployments[deployments['status'] == 'Ready'])
            total_deployments = len(deployments)
            
            # Check replica consistency
            replica_issues = len(deployments[deployments['replicas_desired'] != deployments['replicas_ready']])
            
            success = (
                ready_deployments / total_deployments >= 0.9 and  # 90% deployments ready
                replica_issues / total_deployments <= 0.05  # Less than 5% replica issues
            )
            
            return {
                'status': 'PASS' if success else 'FAIL',
                'details': {
                    'total_deployments': total_deployments,
                    'ready_deployments': ready_deployments,
                    'replica_issues': replica_issues,
                    'ready_ratio': ready_deployments / total_deployments
                },
                'metrics': {
                    'deployment_ready_ratio': ready_deployments / total_deployments,
                    'replica_consistency': 1 - (replica_issues / total_deployments)
                }
            }
        except Exception as e:
            return {
                'status': 'ERROR',
                'details': {'error': str(e)},
                'error_message': f"Deployment rollout test failed: {e}"
            }

    def test_service_discovery(self) -> Dict[str, Any]:
        """Test service discovery validation"""
        try:
            sd_data = pd.read_csv(self.datasets_dir / "service_discovery_validation.csv")
            
            successful_discoveries = len(sd_data[sd_data['success'] == True])
            total_discoveries = len(sd_data)
            avg_response_time = sd_data['response_time_ms'].mean()
            
            # Check TLS usage
            tls_enabled = len(sd_data[sd_data['tls_enabled'] == True])
            
            success = (
                successful_discoveries / total_discoveries >= 0.98 and  # 98% success rate
                avg_response_time <= 50  # Fast service discovery
            )
            
            return {
                'status': 'PASS' if success else 'FAIL',
                'details': {
                    'total_discoveries': total_discoveries,
                    'successful_discoveries': successful_discoveries,
                    'success_rate': successful_discoveries / total_discoveries,
                    'avg_response_time_ms': avg_response_time,
                    'tls_enabled_ratio': tls_enabled / total_discoveries
                },
                'metrics': {
                    'discovery_success_rate': successful_discoveries / total_discoveries,
                    'avg_discovery_time': avg_response_time,
                    'tls_adoption': tls_enabled / total_discoveries
                }
            }
        except Exception as e:
            return {
                'status': 'ERROR',
                'details': {'error': str(e)},
                'error_message': f"Service discovery test failed: {e}"
            }

    def test_hpa_scaling(self) -> Dict[str, Any]:
        """Test HPA scaling validation"""
        try:
            hpa_data = pd.read_csv(self.datasets_dir / "hpa_scaling_validation.csv")
            
            successful_scaling = len(hpa_data[hpa_data['success'] == True])
            total_scaling_events = len(hpa_data)
            avg_scaling_duration = hpa_data['scaling_duration_seconds'].mean()
            
            # Check scaling responsiveness
            fast_scaling = len(hpa_data[hpa_data['scaling_duration_seconds'] <= 60])
            
            success = (
                successful_scaling / total_scaling_events >= 0.95 and  # 95% scaling success
                avg_scaling_duration <= 120  # Average scaling under 2 minutes
            )
            
            return {
                'status': 'PASS' if success else 'FAIL',
                'details': {
                    'total_scaling_events': total_scaling_events,
                    'successful_scaling': successful_scaling,
                    'success_rate': successful_scaling / total_scaling_events,
                    'avg_scaling_duration_seconds': avg_scaling_duration,
                    'fast_scaling_events': fast_scaling
                },
                'metrics': {
                    'scaling_success_rate': successful_scaling / total_scaling_events,
                    'avg_scaling_time': avg_scaling_duration,
                    'fast_scaling_ratio': fast_scaling / total_scaling_events
                }
            }
        except Exception as e:
            return {
                'status': 'ERROR',
                'details': {'error': str(e)},
                'error_message': f"HPA scaling test failed: {e}"
            }

    # Infrastructure Tests
    def test_configmap_validation(self) -> Dict[str, Any]:
        """Test ConfigMap validation"""
        try:
            cm_data = pd.read_csv(self.datasets_dir / "configmaps_validation.csv")
            
            valid_configs = len(cm_data[cm_data['validation_status'] == 'VALID'])
            total_configs = len(cm_data)
            
            # Check encryption and auto-reload
            encrypted_configs = len(cm_data[cm_data['encryption_enabled'] == True])
            auto_reload_configs = len(cm_data[cm_data['auto_reload'] == True])
            
            success = valid_configs / total_configs >= 0.95  # 95% valid configs
            
            return {
                'status': 'PASS' if success else 'FAIL',
                'details': {
                    'total_configs': total_configs,
                    'valid_configs': valid_configs,
                    'validation_rate': valid_configs / total_configs,
                    'encrypted_configs': encrypted_configs,
                    'auto_reload_configs': auto_reload_configs
                },
                'metrics': {
                    'config_validity_rate': valid_configs / total_configs,
                    'encryption_adoption': encrypted_configs / total_configs,
                    'auto_reload_adoption': auto_reload_configs / total_configs
                }
            }
        except Exception as e:
            return {
                'status': 'ERROR',
                'details': {'error': str(e)},
                'error_message': f"ConfigMap validation test failed: {e}"
            }

    def test_secrets_validation(self) -> Dict[str, Any]:
        """Test Secrets validation"""
        try:
            secrets_data = pd.read_csv(self.datasets_dir / "secrets_validation.csv")
            
            valid_secrets = len(secrets_data[secrets_data['validation_status'] == 'VALID'])
            total_secrets = len(secrets_data)
            
            # Check rotation requirements
            rotation_needed = len(secrets_data[secrets_data['rotation_required'] == True])
            expired_certs = len(secrets_data[secrets_data['validation_status'] == 'EXPIRED'])
            
            success = (
                valid_secrets / total_secrets >= 0.98 and  # 98% valid secrets
                expired_certs == 0  # No expired certificates
            )
            
            return {
                'status': 'PASS' if success else 'FAIL',
                'details': {
                    'total_secrets': total_secrets,
                    'valid_secrets': valid_secrets,
                    'validation_rate': valid_secrets / total_secrets,
                    'rotation_needed': rotation_needed,
                    'expired_certs': expired_certs
                },
                'metrics': {
                    'secret_validity_rate': valid_secrets / total_secrets,
                    'rotation_backlog': rotation_needed / total_secrets
                }
            }
        except Exception as e:
            return {
                'status': 'ERROR',
                'details': {'error': str(e)},
                'error_message': f"Secrets validation test failed: {e}"
            }

    # Monitoring Tests
    def test_prometheus_scraping(self) -> Dict[str, Any]:
        """Test Prometheus metrics scraping"""
        try:
            prom_data = pd.read_csv(self.datasets_dir / "prometheus_validation.csv")
            
            successful_scrapes = len(prom_data[prom_data['scrape_success'] == True])
            total_scrapes = len(prom_data)
            avg_scrape_duration = prom_data['scrape_duration_ms'].mean()
            
            # Check alert thresholds
            alerts_triggered = len(prom_data[prom_data['alert_threshold_breached'] == True])
            
            success = (
                successful_scrapes / total_scrapes >= 0.99 and  # 99% successful scrapes
                avg_scrape_duration <= 100  # Fast scraping
            )
            
            return {
                'status': 'PASS' if success else 'FAIL',
                'details': {
                    'total_scrapes': total_scrapes,
                    'successful_scrapes': successful_scrapes,
                    'scrape_success_rate': successful_scrapes / total_scrapes,
                    'avg_scrape_duration_ms': avg_scrape_duration,
                    'alerts_triggered': alerts_triggered
                },
                'metrics': {
                    'scrape_success_rate': successful_scrapes / total_scrapes,
                    'avg_scrape_time': avg_scrape_duration,
                    'alert_rate': alerts_triggered / total_scrapes
                }
            }
        except Exception as e:
            return {
                'status': 'ERROR',
                'details': {'error': str(e)},
                'error_message': f"Prometheus scraping test failed: {e}"
            }

    # Additional test methods would be implemented here for all other test cases...

    async def run_all_validation_suites(self) -> Dict[str, Any]:
        """Run all validation suites"""
        logger.info("Starting comprehensive containerized validation testing")
        self.execution_start_time = datetime.now()
        
        await self.initialize_k8s_client()
        
        all_results = {}
        suite_execution_order = [
            'container_orchestration',
            'infrastructure', 
            'monitoring',
            'cicd',
            'cloud_integration'
        ]
        
        for suite_name in suite_execution_order:
            logger.info(f"Executing validation suite: {suite_name}")
            suite_results = await self.run_validation_suite(suite_name)
            all_results[suite_name] = suite_results
            self.test_results.extend(suite_results)
            
            # Check if we should continue based on results
            failed_critical = [r for r in suite_results if r.status == "FAIL" and "critical" in r.test_name]
            if failed_critical:
                logger.error(f"Critical failures in {suite_name}, stopping execution")
                break
        
        # Generate comprehensive report
        report = await self.generate_validation_report(all_results)
        
        logger.info("Containerized validation testing completed")
        return report

    async def generate_validation_report(self, all_results: Dict[str, List[TestResult]]) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        total_duration = (datetime.now() - self.execution_start_time).total_seconds()
        
        # Calculate overall statistics
        total_tests = sum(len(results) for results in all_results.values())
        passed_tests = sum(len([r for r in results if r.status == "PASS"]) for results in all_results.values())
        failed_tests = sum(len([r for r in results if r.status == "FAIL"]) for results in all_results.values())
        error_tests = sum(len([r for r in results if r.status == "ERROR"]) for results in all_results.values())
        
        report = {
            'execution_summary': {
                'start_time': self.execution_start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_duration_seconds': total_duration,
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'error_tests': error_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0
            },
            'suite_results': {},
            'detailed_results': []
        }
        
        # Per-suite analysis
        for suite_name, results in all_results.items():
            suite_passed = len([r for r in results if r.status == "PASS"])
            suite_total = len(results)
            
            report['suite_results'][suite_name] = {
                'total_tests': suite_total,
                'passed_tests': suite_passed,
                'success_rate': suite_passed / suite_total if suite_total > 0 else 0,
                'avg_duration': sum(r.duration_seconds for r in results) / len(results) if results else 0,
                'failed_tests': [r.test_name for r in results if r.status in ["FAIL", "ERROR"]]
            }
        
        # Detailed results
        for result in self.test_results:
            report['detailed_results'].append(asdict(result))
        
        # Save report
        report_path = self.results_dir / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate summary markdown
        await self.generate_markdown_report(report, report_path.with_suffix('.md'))
        
        logger.info(f"Validation report generated: {report_path}")
        return report

    async def generate_markdown_report(self, report: Dict[str, Any], output_path: Path):
        """Generate human-readable markdown report"""
        
        md_content = f"""# IFRS9 Containerized Validation Report

**Generated:** {report['execution_summary']['end_time']}
**Duration:** {report['execution_summary']['total_duration_seconds']:.2f} seconds
**Overall Success Rate:** {report['execution_summary']['success_rate']:.1%}

## Executive Summary
- **Total Tests:** {report['execution_summary']['total_tests']}
- **Passed:** {report['execution_summary']['passed_tests']} ✅
- **Failed:** {report['execution_summary']['failed_tests']} ❌ 
- **Errors:** {report['execution_summary']['error_tests']} ⚠️

## Suite Results

"""
        
        for suite_name, suite_data in report['suite_results'].items():
            status_emoji = "✅" if suite_data['success_rate'] >= 0.9 else "❌" if suite_data['success_rate'] < 0.7 else "⚠️"
            
            md_content += f"""### {suite_name.replace('_', ' ').title()} {status_emoji}
- **Success Rate:** {suite_data['success_rate']:.1%}
- **Tests:** {suite_data['passed_tests']}/{suite_data['total_tests']} passed
- **Average Duration:** {suite_data['avg_duration']:.2f}s
"""
            
            if suite_data['failed_tests']:
                md_content += f"- **Failed Tests:** {', '.join(suite_data['failed_tests'])}\n"
            
            md_content += "\n"
        
        # Recommendations section
        overall_success = report['execution_summary']['success_rate']
        md_content += "## Recommendations\n\n"
        
        if overall_success >= 0.95:
            md_content += "🎉 **Excellent!** Your containerized infrastructure is performing exceptionally well.\n"
        elif overall_success >= 0.85:
            md_content += "👍 **Good!** Your infrastructure is mostly healthy with minor issues to address.\n"
        else:
            md_content += "⚠️ **Action Required!** Several critical issues need immediate attention.\n"
        
        with open(output_path, 'w') as f:
            f.write(md_content)

# Main execution
async def main():
    """Main execution function"""
    orchestrator = ContainerizedTestOrchestrator()
    results = await orchestrator.run_all_validation_suites()
    
    print(f"Validation completed. Success rate: {results['execution_summary']['success_rate']:.1%}")
    return results

if __name__ == "__main__":
    asyncio.run(main())