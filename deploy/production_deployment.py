#!/usr/bin/env python3
"""
IFRS9 Production Deployment Orchestrator
Comprehensive deployment script for production-ready infrastructure
"""

import os
import sys
import asyncio
import json
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProductionDeployer:
    """Main orchestrator for IFRS9 production deployment"""
    
    def __init__(self):
        self.deployment_start = datetime.now()
        self.deployment_status = {}
        
    async def deploy_monitoring_stack(self):
        """Deploy Prometheus/Grafana monitoring stack"""
        logger.info("Deploying monitoring infrastructure...")
        
        try:
            # Create monitoring directories
            monitoring_dirs = [
                '/opt/prometheus/data',
                '/opt/grafana/data',
                '/opt/prometheus/config',
                '/opt/grafana/config'
            ]
            
            for dir_path in monitoring_dirs:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
            
            # Copy monitoring configurations
            config_files = [
                ('monitoring/prometheus.yml', '/opt/prometheus/config/prometheus.yml'),
                ('monitoring/ifrs9_alerting_rules.yml', '/opt/prometheus/config/ifrs9_alerting_rules.yml'),
                ('monitoring/ifrs9_recording_rules.yml', '/opt/prometheus/config/ifrs9_recording_rules.yml'),
                ('monitoring/grafana-dashboards-agents.json', '/opt/grafana/config/dashboards/')
            ]
            
            for src, dst in config_files:
                src_path = Path(src)
                dst_path = Path(dst)
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                
                if src_path.exists():
                    import shutil
                    shutil.copy2(src_path, dst_path)
                    logger.info(f"Copied {src} to {dst}")
            
            # Deploy monitoring containers
            await self._deploy_monitoring_containers()
            
            # Start Polars metrics exporter
            await self._start_polars_metrics_exporter()
            
            self.deployment_status['monitoring'] = 'SUCCESS'
            logger.info("✓ Monitoring stack deployed successfully")
            
        except Exception as e:
            self.deployment_status['monitoring'] = f'FAILED: {str(e)}'
            logger.error(f"✗ Monitoring stack deployment failed: {e}")
            raise
    
    async def _deploy_monitoring_containers(self):
        """Deploy monitoring containers using docker-compose"""
        monitoring_compose = {
            'version': '3.8',
            'services': {
                'prometheus': {
                    'image': 'prom/prometheus:latest',
                    'container_name': 'ifrs9-prometheus',
                    'ports': ['9090:9090'],
                    'volumes': [
                        '/opt/prometheus/config/prometheus.yml:/etc/prometheus/prometheus.yml',
                        '/opt/prometheus/config/ifrs9_alerting_rules.yml:/etc/prometheus/ifrs9_alerting_rules.yml',
                        '/opt/prometheus/config/ifrs9_recording_rules.yml:/etc/prometheus/ifrs9_recording_rules.yml',
                        '/opt/prometheus/data:/prometheus'
                    ],
                    'command': [
                        '--config.file=/etc/prometheus/prometheus.yml',
                        '--storage.tsdb.path=/prometheus',
                        '--web.console.libraries=/etc/prometheus/console_libraries',
                        '--web.console.templates=/etc/prometheus/consoles',
                        '--storage.tsdb.retention.time=90d',
                        '--storage.tsdb.retention.size=50GB',
                        '--web.enable-lifecycle'
                    ],
                    'restart': 'unless-stopped',
                    'networks': ['ifrs9-monitoring']
                },
                'grafana': {
                    'image': 'grafana/grafana:latest',
                    'container_name': 'ifrs9-grafana',
                    'ports': ['3000:3000'],
                    'volumes': [
                        '/opt/grafana/data:/var/lib/grafana',
                        '/opt/grafana/config/dashboards:/etc/grafana/provisioning/dashboards'
                    ],
                    'environment': [
                        'GF_SECURITY_ADMIN_PASSWORD=ifrs9admin123',
                        'GF_USERS_ALLOW_SIGN_UP=false'
                    ],
                    'restart': 'unless-stopped',
                    'networks': ['ifrs9-monitoring']
                },
                'alertmanager': {
                    'image': 'prom/alertmanager:latest',
                    'container_name': 'ifrs9-alertmanager',
                    'ports': ['9093:9093'],
                    'restart': 'unless-stopped',
                    'networks': ['ifrs9-monitoring']
                }
            },
            'networks': {
                'ifrs9-monitoring': {
                    'driver': 'bridge'
                }
            }
        }
        
        # Write compose file
        compose_file = Path('docker-compose.monitoring.yml')
        with open(compose_file, 'w') as f:
            yaml.dump(monitoring_compose, f, default_flow_style=False)
        
        # Deploy using docker-compose
        cmd = ['docker-compose', '-f', str(compose_file), 'up', '-d']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Docker-compose failed: {result.stderr}")
        
        logger.info("Monitoring containers deployed successfully")
    
    async def _start_polars_metrics_exporter(self):
        """Start Polars performance metrics exporter"""
        try:
            # Start Polars metrics exporter as a background service
            cmd = [
                'python3',
                'monitoring/polars-performance-exporter.py'
            ]
            
            # Start in background
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=Path.cwd()
            )
            
            logger.info(f"Polars metrics exporter started with PID: {process.pid}")
            
        except Exception as e:
            logger.error(f"Failed to start Polars metrics exporter: {e}")
    
    async def setup_load_testing_framework(self):
        """Set up the load testing framework"""
        logger.info("Setting up load testing framework...")
        
        try:
            # Create testing directories
            test_dirs = [
                '/opt/ifrs9/testing/results',
                '/opt/ifrs9/testing/data',
                '/opt/ifrs9/testing/logs'
            ]
            
            for dir_path in test_dirs:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
            
            # Copy testing scripts
            test_files = [
                'testing/load_testing_framework.py',
                'testing/external_systems_validator.py',
                'testing/docker_orchestration_validator.py'
            ]
            
            for test_file in test_files:
                src_path = Path(test_file)
                dst_path = Path(f'/opt/ifrs9/testing/{src_path.name}')
                
                if src_path.exists():
                    import shutil
                    shutil.copy2(src_path, dst_path)
                    logger.info(f"Copied {test_file} to testing directory")
            
            # Create testing cron jobs
            await self._setup_testing_cron_jobs()
            
            self.deployment_status['load_testing'] = 'SUCCESS'
            logger.info("✓ Load testing framework deployed successfully")
            
        except Exception as e:
            self.deployment_status['load_testing'] = f'FAILED: {str(e)}'
            logger.error(f"✗ Load testing framework deployment failed: {e}")
    
    async def _setup_testing_cron_jobs(self):
        """Set up cron jobs for automated testing"""
        cron_entries = [
            # Daily load testing at 3 AM
            "0 3 * * * cd /opt/ifrs9/testing && python3 load_testing_framework.py >> /opt/ifrs9/testing/logs/load_test.log 2>&1",
            # Weekly external systems validation on Sundays at 4 AM
            "0 4 * * 0 cd /opt/ifrs9/testing && python3 external_systems_validator.py >> /opt/ifrs9/testing/logs/external_validation.log 2>&1",
            # Daily Docker validation at 2 AM
            "0 2 * * * cd /opt/ifrs9/testing && python3 docker_orchestration_validator.py >> /opt/ifrs9/testing/logs/docker_validation.log 2>&1"
        ]
        
        # Add to crontab
        for entry in cron_entries:
            cmd = f'(crontab -l 2>/dev/null; echo "{entry}") | crontab -'
            subprocess.run(cmd, shell=True)
        
        logger.info("Testing cron jobs configured")
    
    async def deploy_backup_recovery_system(self):
        """Deploy backup and recovery system"""
        logger.info("Deploying backup and recovery system...")
        
        try:
            # Create backup directories
            backup_dirs = [
                '/opt/ifrs9/backups/local',
                '/opt/ifrs9/backups/staging',
                '/opt/ifrs9/backups/logs'
            ]
            
            for dir_path in backup_dirs:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
            
            # Copy backup scripts
            backup_files = [
                'backup/backup_recovery_orchestrator.py'
            ]
            
            for backup_file in backup_files:
                src_path = Path(backup_file)
                dst_path = Path(f'/opt/ifrs9/backups/{src_path.name}')
                
                if src_path.exists():
                    import shutil
                    shutil.copy2(src_path, dst_path)
                    logger.info(f"Copied {backup_file} to backup directory")
            
            # Set up backup cron jobs
            await self._setup_backup_cron_jobs()
            
            # Start backup metrics server
            await self._start_backup_metrics_server()
            
            self.deployment_status['backup_recovery'] = 'SUCCESS'
            logger.info("✓ Backup and recovery system deployed successfully")
            
        except Exception as e:
            self.deployment_status['backup_recovery'] = f'FAILED: {str(e)}'
            logger.error(f"✗ Backup and recovery system deployment failed: {e}")
    
    async def _setup_backup_cron_jobs(self):
        """Set up backup cron jobs"""
        backup_cron_entries = [
            # Full backup every Sunday at 2 AM
            "0 2 * * 0 cd /opt/ifrs9/backups && python3 backup_recovery_orchestrator.py --mode=full >> /opt/ifrs9/backups/logs/backup.log 2>&1",
            # Incremental backup Monday-Saturday at 2 AM
            "0 2 * * 1-6 cd /opt/ifrs9/backups && python3 backup_recovery_orchestrator.py --mode=incremental >> /opt/ifrs9/backups/logs/backup.log 2>&1",
            # Backup verification daily at 5 AM
            "0 5 * * * cd /opt/ifrs9/backups && python3 backup_recovery_orchestrator.py --mode=verify >> /opt/ifrs9/backups/logs/verify.log 2>&1"
        ]
        
        for entry in backup_cron_entries:
            cmd = f'(crontab -l 2>/dev/null; echo "{entry}") | crontab -'
            subprocess.run(cmd, shell=True)
        
        logger.info("Backup cron jobs configured")
    
    async def _start_backup_metrics_server(self):
        """Start backup metrics server"""
        try:
            cmd = [
                'python3',
                '/opt/ifrs9/backups/backup_recovery_orchestrator.py',
                '--metrics-only'
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            logger.info(f"Backup metrics server started with PID: {process.pid}")
            
        except Exception as e:
            logger.error(f"Failed to start backup metrics server: {e}")
    
    async def validate_production_deployment(self):
        """Validate the entire production deployment"""
        logger.info("Validating production deployment...")
        
        validation_results = {}
        
        try:
            # Run Docker orchestration validation
            docker_validator_cmd = ['python3', 'testing/docker_orchestration_validator.py']
            docker_result = subprocess.run(docker_validator_cmd, capture_output=True, text=True)
            validation_results['docker_orchestration'] = docker_result.returncode == 0
            
            if docker_result.returncode != 0:
                logger.error(f"Docker validation failed: {docker_result.stderr}")
            
            # Run external systems validation
            external_validator_cmd = ['python3', 'testing/external_systems_validator.py']
            external_result = subprocess.run(external_validator_cmd, capture_output=True, text=True)
            validation_results['external_systems'] = external_result.returncode == 0
            
            if external_result.returncode != 0:
                logger.error(f"External systems validation failed: {external_result.stderr}")
            
            # Check monitoring endpoints
            monitoring_health = await self._check_monitoring_health()
            validation_results['monitoring'] = monitoring_health
            
            # Overall validation status
            all_passed = all(validation_results.values())
            
            if all_passed:
                self.deployment_status['validation'] = 'SUCCESS'
                logger.info("✓ Production deployment validation PASSED")
            else:
                self.deployment_status['validation'] = f'FAILED: {validation_results}'
                logger.error(f"✗ Production deployment validation FAILED: {validation_results}")
            
            return all_passed
            
        except Exception as e:
            self.deployment_status['validation'] = f'ERROR: {str(e)}'
            logger.error(f"Production validation error: {e}")
            return False
    
    async def _check_monitoring_health(self) -> bool:
        """Check monitoring endpoints health"""
        try:
            import requests
            
            endpoints = [
                ('Prometheus', 'http://localhost:9090/-/healthy'),
                ('Grafana', 'http://localhost:3000/api/health'),
                ('Polars Metrics', 'http://localhost:9092/metrics')
            ]
            
            for name, url in endpoints:
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        logger.info(f"✓ {name} is healthy")
                    else:
                        logger.error(f"✗ {name} health check failed: {response.status_code}")
                        return False
                except requests.exceptions.RequestException as e:
                    logger.error(f"✗ {name} is unreachable: {e}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Monitoring health check error: {e}")
            return False
    
    def generate_deployment_report(self):
        """Generate comprehensive deployment report"""
        deployment_end = datetime.now()
        deployment_duration = (deployment_end - self.deployment_start).total_seconds()
        
        report = {
            'deployment_summary': {
                'start_time': self.deployment_start.isoformat(),
                'end_time': deployment_end.isoformat(),
                'duration_seconds': deployment_duration,
                'overall_status': 'SUCCESS' if all(
                    'SUCCESS' in status for status in self.deployment_status.values()
                ) else 'PARTIAL' if any(
                    'SUCCESS' in status for status in self.deployment_status.values()
                ) else 'FAILED'
            },
            'component_status': self.deployment_status,
            'infrastructure_components': {
                'monitoring': {
                    'prometheus': 'http://localhost:9090',
                    'grafana': 'http://localhost:3000',
                    'alertmanager': 'http://localhost:9093',
                    'polars_metrics': 'http://localhost:9092'
                },
                'testing': {
                    'load_testing': '/opt/ifrs9/testing/load_testing_framework.py',
                    'external_validation': '/opt/ifrs9/testing/external_systems_validator.py',
                    'docker_validation': '/opt/ifrs9/testing/docker_orchestration_validator.py'
                },
                'backup': {
                    'orchestrator': '/opt/ifrs9/backups/backup_recovery_orchestrator.py',
                    'backup_metrics': 'http://localhost:9095'
                }
            },
            'scheduled_jobs': {
                'backups': [
                    'Full backup: Sunday 2 AM',
                    'Incremental backup: Monday-Saturday 2 AM',
                    'Backup verification: Daily 5 AM'
                ],
                'testing': [
                    'Load testing: Daily 3 AM',
                    'External validation: Sunday 4 AM',
                    'Docker validation: Daily 2 AM'
                ]
            },
            'next_steps': [
                'Monitor Grafana dashboards for system health',
                'Review backup reports in /opt/ifrs9/backups/logs/',
                'Check testing results in /opt/ifrs9/testing/results/',
                'Configure alerting notifications',
                'Set up log aggregation and analysis'
            ]
        }
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"ifrs9_production_deployment_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Deployment report saved to: {report_file}")
        
        # Print summary
        logger.info("=== IFRS9 PRODUCTION DEPLOYMENT SUMMARY ===")
        logger.info(f"Overall Status: {report['deployment_summary']['overall_status']}")
        logger.info(f"Duration: {deployment_duration:.1f} seconds")
        logger.info("\nComponent Status:")
        for component, status in self.deployment_status.items():
            logger.info(f"  {component}: {status}")
        
        return report
    
    async def deploy_all(self):
        """Deploy all production infrastructure components"""
        logger.info("Starting IFRS9 production deployment...")
        logger.info("="*50)
        
        try:
            # 1. Deploy monitoring stack
            await self.deploy_monitoring_stack()
            
            # 2. Set up load testing framework
            await self.setup_load_testing_framework()
            
            # 3. Deploy backup and recovery system
            await self.deploy_backup_recovery_system()
            
            # 4. Validate entire deployment
            validation_passed = await self.validate_production_deployment()
            
            # 5. Generate deployment report
            report = self.generate_deployment_report()
            
            if validation_passed:
                logger.info("🎉 IFRS9 production deployment completed successfully!")
                logger.info("The system is now ready for production workloads.")
            else:
                logger.error("⚠️  IFRS9 production deployment completed with issues.")
                logger.error("Please review the deployment report and address any failed components.")
            
            return report
            
        except Exception as e:
            logger.error(f"Production deployment failed: {e}")
            self.deployment_status['overall'] = f'FAILED: {str(e)}'
            raise

async def main():
    """Main deployment function"""
    deployer = ProductionDeployer()
    
    try:
        report = await deployer.deploy_all()
        
        # Exit with appropriate code
        if report['deployment_summary']['overall_status'] == 'SUCCESS':
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())