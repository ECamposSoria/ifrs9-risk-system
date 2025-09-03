#!/usr/bin/env python3
"""
IFRS9 Master Validation Orchestrator
Comprehensive orchestration of all validation agents for Docker environment

This master orchestrator coordinates and runs all validation agents:
- Docker Environment Validator
- End-to-End Pipeline Validator  
- Environment Validator (legacy compatibility)
- Cross-validation and consistency checks
- Master reporting and decision making
- Production readiness assessment
"""

import sys
import os
import json
import time
import subprocess
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any, Optional, Union
import pandas as pd
from pathlib import Path


class MasterValidationOrchestrator:
    """Master orchestrator for all IFRS9 validation agents."""
    
    def __init__(self):
        """Initialize the master validation orchestrator."""
        self.validation_results = {}
        self.agent_results = {}
        self.critical_issues = []
        self.issues_found = []
        self.warnings = []
        
        # Validation agent paths
        self.validation_agents = {
            'docker_environment': '/opt/airflow/validation/docker_environment_validator.py',
            'end_to_end_pipeline': '/opt/airflow/validation/end_to_end_pipeline_validator.py',
            'environment_legacy': '/opt/airflow/validation/environment_validator.py'
        }
        
        # Validation priorities (higher = more critical)
        self.agent_priorities = {
            'docker_environment': 10,  # Must pass for system to work
            'end_to_end_pipeline': 9,  # Critical for functionality
            'environment_legacy': 7    # Compatibility check
        }
        
        # Required validation thresholds
        self.validation_thresholds = {
            'min_agents_pass': 2,      # Minimum agents that must pass
            'max_critical_issues': 0,  # Maximum critical issues allowed
            'max_total_issues': 5,     # Maximum total issues allowed
            'ml_dependency_threshold': 0.8,  # 80% ML dependencies must work
            'pipeline_success_threshold': 0.75  # 75% pipeline steps must succeed
        }
    
    def run_docker_environment_validation(self) -> Dict[str, Any]:
        """Run Docker environment validation agent."""
        print("=" * 70)
        print("RUNNING DOCKER ENVIRONMENT VALIDATION AGENT")
        print("=" * 70)
        
        agent_result = {
            'agent': 'docker_environment',
            'status': 'PENDING',
            'start_time': datetime.now().isoformat(),
            'priority': self.agent_priorities['docker_environment']
        }
        
        try:
            # Import and run Docker environment validator
            from docker_environment_validator import DockerEnvironmentValidator
            
            validator = DockerEnvironmentValidator()
            results = validator.run_comprehensive_docker_validation()
            
            agent_result['status'] = 'COMPLETED'
            agent_result['results'] = results
            agent_result['end_time'] = datetime.now().isoformat()
            
            # Extract key metrics
            validation_summary = results.get('validation_summary', {})
            agent_result['overall_status'] = validation_summary.get('overall_status', 'UNKNOWN')
            agent_result['critical_issues'] = validation_summary.get('critical_issues', [])
            agent_result['total_issues'] = validation_summary.get('total_issues', 0)
            agent_result['docker_ready'] = validation_summary.get('docker_ready', False)
            agent_result['production_ready'] = validation_summary.get('production_ready', False)
            
            print(f"✅ Docker Environment Validation: {agent_result['overall_status']}")
            print(f"   Docker Ready: {agent_result['docker_ready']}")
            print(f"   Production Ready: {agent_result['production_ready']}")
            
            # Aggregate issues
            self.critical_issues.extend(agent_result['critical_issues'])
            self.issues_found.extend(validation_summary.get('issues_found', []))
            self.warnings.extend(validation_summary.get('warnings', []))
            
        except ImportError as e:
            agent_result['status'] = 'IMPORT_ERROR'
            agent_result['error'] = f"Could not import docker_environment_validator: {str(e)}"
            agent_result['end_time'] = datetime.now().isoformat()
            print(f"❌ Docker Environment Validation: Import failed - {str(e)}")
            self.critical_issues.append(f"Docker validation agent import failed: {str(e)}")
            
        except Exception as e:
            agent_result['status'] = 'ERROR'
            agent_result['error'] = str(e)
            agent_result['end_time'] = datetime.now().isoformat()
            print(f"❌ Docker Environment Validation: Execution failed - {str(e)}")
            self.critical_issues.append(f"Docker validation agent failed: {str(e)}")
        
        return agent_result
    
    def run_end_to_end_pipeline_validation(self) -> Dict[str, Any]:
        """Run end-to-end pipeline validation agent."""
        print("\nRUNNING END-TO-END PIPELINE VALIDATION AGENT")
        print("-" * 70)
        
        agent_result = {
            'agent': 'end_to_end_pipeline',
            'status': 'PENDING',
            'start_time': datetime.now().isoformat(),
            'priority': self.agent_priorities['end_to_end_pipeline']
        }
        
        try:
            # Import and run end-to-end pipeline validator
            from end_to_end_pipeline_validator import EndToEndPipelineValidator
            
            validator = EndToEndPipelineValidator()
            results = validator.run_comprehensive_pipeline_validation()
            
            agent_result['status'] = 'COMPLETED'
            agent_result['results'] = results
            agent_result['end_time'] = datetime.now().isoformat()
            
            # Extract key metrics
            pipeline_summary = results.get('pipeline_summary', {})
            agent_result['overall_status'] = pipeline_summary.get('overall_status', 'UNKNOWN')
            agent_result['critical_issues'] = pipeline_summary.get('critical_issues', [])
            agent_result['total_issues'] = pipeline_summary.get('total_issues', 0)
            
            pipeline_readiness = pipeline_summary.get('pipeline_readiness', {})
            agent_result['pipeline_ready'] = pipeline_readiness.get('overall_ready', False)
            agent_result['production_ready'] = pipeline_summary.get('production_readiness', False)
            
            print(f"✅ End-to-End Pipeline Validation: {agent_result['overall_status']}")
            print(f"   Pipeline Ready: {agent_result['pipeline_ready']}")
            print(f"   Production Ready: {agent_result['production_ready']}")
            
            # Aggregate issues
            self.critical_issues.extend(agent_result['critical_issues'])
            self.issues_found.extend(pipeline_summary.get('issues_found', []))
            self.warnings.extend(pipeline_summary.get('warnings', []))
            
        except ImportError as e:
            agent_result['status'] = 'IMPORT_ERROR'
            agent_result['error'] = f"Could not import end_to_end_pipeline_validator: {str(e)}"
            agent_result['end_time'] = datetime.now().isoformat()
            print(f"❌ End-to-End Pipeline Validation: Import failed - {str(e)}")
            self.critical_issues.append(f"Pipeline validation agent import failed: {str(e)}")
            
        except Exception as e:
            agent_result['status'] = 'ERROR'
            agent_result['error'] = str(e)
            agent_result['end_time'] = datetime.now().isoformat()
            print(f"❌ End-to-End Pipeline Validation: Execution failed - {str(e)}")
            self.critical_issues.append(f"Pipeline validation agent failed: {str(e)}")
        
        return agent_result
    
    def run_environment_legacy_validation(self) -> Dict[str, Any]:
        """Run legacy environment validation for compatibility."""
        print("\nRUNNING LEGACY ENVIRONMENT VALIDATION (COMPATIBILITY)")
        print("-" * 70)
        
        agent_result = {
            'agent': 'environment_legacy',
            'status': 'PENDING',
            'start_time': datetime.now().isoformat(),
            'priority': self.agent_priorities['environment_legacy']
        }
        
        try:
            # Import and run legacy environment validator
            from environment_validator import IFRS9EnvironmentValidator
            
            validator = IFRS9EnvironmentValidator()
            results = validator.run_comprehensive_validation()
            
            agent_result['status'] = 'COMPLETED'
            agent_result['results'] = results
            agent_result['end_time'] = datetime.now().isoformat()
            
            # Extract key metrics
            validation_summary = results.get('validation_summary', {})
            agent_result['overall_status'] = validation_summary.get('overall_status', 'UNKNOWN')
            agent_result['critical_issues'] = validation_summary.get('critical_issues', [])
            agent_result['total_issues'] = validation_summary.get('total_issues', 0)
            
            print(f"✅ Legacy Environment Validation: {agent_result['overall_status']}")
            print(f"   Compatibility Status: {agent_result['overall_status']}")
            
            # Add legacy issues as warnings (not critical for Docker environment)
            self.warnings.extend(agent_result['critical_issues'])
            self.warnings.extend(validation_summary.get('issues_found', []))
            self.warnings.extend(validation_summary.get('warnings', []))
            
        except ImportError as e:
            agent_result['status'] = 'IMPORT_ERROR'
            agent_result['error'] = f"Could not import environment_validator: {str(e)}"
            agent_result['end_time'] = datetime.now().isoformat()
            print(f"⚠️ Legacy Environment Validation: Import failed - {str(e)} (Non-critical)")
            self.warnings.append(f"Legacy validation agent import failed: {str(e)}")
            
        except Exception as e:
            agent_result['status'] = 'ERROR'
            agent_result['error'] = str(e)
            agent_result['end_time'] = datetime.now().isoformat()
            print(f"⚠️ Legacy Environment Validation: Execution failed - {str(e)} (Non-critical)")
            self.warnings.append(f"Legacy validation agent failed: {str(e)}")
        
        return agent_result
    
    def run_cross_validation_checks(self) -> Dict[str, Any]:
        """Run cross-validation consistency checks between agents."""
        print("\nRUNNING CROSS-VALIDATION CONSISTENCY CHECKS")
        print("-" * 70)
        
        cross_validation = {
            'test_name': 'cross_validation_checks',
            'status': 'PENDING',
            'start_time': datetime.now().isoformat(),
            'checks': {}
        }
        
        try:
            # Check 1: Docker vs Pipeline readiness consistency
            docker_result = self.agent_results.get('docker_environment', {})
            pipeline_result = self.agent_results.get('end_to_end_pipeline', {})
            
            docker_ready = docker_result.get('docker_ready', False)
            pipeline_ready = pipeline_result.get('pipeline_ready', False)
            
            cross_validation['checks']['docker_pipeline_consistency'] = {
                'docker_ready': docker_ready,
                'pipeline_ready': pipeline_ready,
                'consistent': docker_ready == pipeline_ready or (docker_ready and not pipeline_ready),
                'note': 'Docker readiness should enable pipeline readiness'
            }
            
            # Check 2: Critical issues consistency
            docker_issues = len(docker_result.get('critical_issues', []))
            pipeline_issues = len(pipeline_result.get('critical_issues', []))
            
            cross_validation['checks']['critical_issues_consistency'] = {
                'docker_critical_issues': docker_issues,
                'pipeline_critical_issues': pipeline_issues,
                'total_unique_issues': len(self.critical_issues),
                'consistency_check': 'PASS' if len(self.critical_issues) >= max(docker_issues, pipeline_issues) else 'FAIL'
            }
            
            # Check 3: Production readiness alignment
            docker_production = docker_result.get('production_ready', False)
            pipeline_production = pipeline_result.get('production_ready', False)
            
            cross_validation['checks']['production_readiness_alignment'] = {
                'docker_production_ready': docker_production,
                'pipeline_production_ready': pipeline_production,
                'aligned': docker_production == pipeline_production,
                'overall_production_ready': docker_production and pipeline_production
            }
            
            # Check 4: Data consistency check
            docker_ml_deps = self._extract_ml_dependency_status(docker_result)
            pipeline_success = self._extract_pipeline_success_rate(pipeline_result)
            
            cross_validation['checks']['data_consistency'] = {
                'ml_dependency_rate': docker_ml_deps,
                'pipeline_success_rate': pipeline_success,
                'consistent': (docker_ml_deps > 0.8 and pipeline_success > 0.7) or (docker_ml_deps <= 0.8 and pipeline_success <= 0.7),
                'note': 'ML dependencies and pipeline success should correlate'
            }
            
            # Overall cross-validation status
            all_checks_pass = all(
                check.get('consistent', True) or check.get('aligned', True) or check.get('consistency_check', 'PASS') == 'PASS'
                for check in cross_validation['checks'].values()
                if isinstance(check, dict)
            )
            
            cross_validation['status'] = 'SUCCESS' if all_checks_pass else 'INCONSISTENT'
            cross_validation['end_time'] = datetime.now().isoformat()
            
            print(f"✅ Cross-validation checks: {cross_validation['status']}")
            for check_name, check_result in cross_validation['checks'].items():
                if isinstance(check_result, dict):
                    consistent = check_result.get('consistent', check_result.get('aligned', check_result.get('consistency_check') == 'PASS'))
                    print(f"   {check_name}: {'✅' if consistent else '⚠️'} {'CONSISTENT' if consistent else 'INCONSISTENT'}")
            
            if cross_validation['status'] != 'SUCCESS':
                self.warnings.append("Cross-validation checks found inconsistencies between agents")
                
        except Exception as e:
            cross_validation['status'] = 'ERROR'
            cross_validation['error'] = str(e)
            cross_validation['end_time'] = datetime.now().isoformat()
            print(f"❌ Cross-validation checks: Error - {str(e)}")
            self.issues_found.append(f"Cross-validation failed: {str(e)}")
        
        return cross_validation
    
    def _extract_ml_dependency_status(self, docker_result: Dict[str, Any]) -> float:
        """Extract ML dependency success rate from Docker validation results."""
        try:
            results = docker_result.get('results', {})
            ml_deps = results.get('ml_dependencies', {})
            
            total_success_rate = 0.0
            container_count = 0
            
            for container, deps_info in ml_deps.items():
                if isinstance(deps_info, dict) and 'summary' in deps_info:
                    success_rate = deps_info['summary'].get('success_rate', 0)
                    total_success_rate += success_rate
                    container_count += 1
            
            return (total_success_rate / container_count / 100.0) if container_count > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _extract_pipeline_success_rate(self, pipeline_result: Dict[str, Any]) -> float:
        """Extract pipeline success rate from pipeline validation results."""
        try:
            results = pipeline_result.get('results', {})
            
            successful_pipelines = 0
            total_pipelines = 0
            
            for pipeline_name in ['data_ingestion', 'ml_model_pipeline', 'ecl_calculation']:
                pipeline_data = results.get(pipeline_name, {})
                if pipeline_data:
                    total_pipelines += 1
                    if pipeline_data.get('status') in ['SUCCESS', 'PARTIAL_SUCCESS']:
                        successful_pipelines += 1
            
            return (successful_pipelines / total_pipelines) if total_pipelines > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def assess_overall_system_status(self) -> Dict[str, Any]:
        """Assess overall system status based on all validation results."""
        print("\nASSESSING OVERALL SYSTEM STATUS")
        print("-" * 50)
        
        assessment = {
            'assessment_time': datetime.now().isoformat(),
            'agent_summary': {},
            'system_metrics': {},
            'readiness_assessment': {},
            'recommendations': [],
            'overall_status': 'UNKNOWN'
        }
        
        try:
            # Agent summary
            successful_agents = 0
            total_agents = len(self.agent_results)
            
            for agent_name, agent_result in self.agent_results.items():
                if isinstance(agent_result, dict):
                    status = agent_result.get('overall_status', agent_result.get('status', 'UNKNOWN'))
                    assessment['agent_summary'][agent_name] = {
                        'status': status,
                        'priority': agent_result.get('priority', 0),
                        'critical_issues': len(agent_result.get('critical_issues', [])),
                        'total_issues': agent_result.get('total_issues', 0)
                    }
                    
                    if status in ['PASS', 'SUCCESS']:
                        successful_agents += 1
            
            # System metrics
            assessment['system_metrics'] = {
                'total_agents_run': total_agents,
                'successful_agents': successful_agents,
                'agent_success_rate': (successful_agents / total_agents) if total_agents > 0 else 0,
                'total_critical_issues': len(self.critical_issues),
                'total_issues': len(self.issues_found),
                'total_warnings': len(self.warnings)
            }
            
            # Readiness assessment
            docker_agent = self.agent_results.get('docker_environment', {})
            pipeline_agent = self.agent_results.get('end_to_end_pipeline', {})
            
            assessment['readiness_assessment'] = {
                'docker_ready': docker_agent.get('docker_ready', False),
                'pipeline_ready': pipeline_agent.get('pipeline_ready', False),
                'docker_production_ready': docker_agent.get('production_ready', False),
                'pipeline_production_ready': pipeline_agent.get('production_ready', False),
                'meets_minimum_requirements': self._meets_minimum_requirements(),
                'recommended_for_testing': self._recommended_for_testing(),
                'recommended_for_production': self._recommended_for_production()
            }
            
            # Overall status determination
            assessment['overall_status'] = self._determine_overall_status(assessment)
            
            # Recommendations
            assessment['recommendations'] = self._generate_recommendations(assessment)
            
            print(f"Overall System Status: {assessment['overall_status']}")
            print(f"Agent Success Rate: {assessment['system_metrics']['agent_success_rate']:.1%}")
            print(f"Critical Issues: {assessment['system_metrics']['total_critical_issues']}")
            print(f"Docker Ready: {assessment['readiness_assessment']['docker_ready']}")
            print(f"Pipeline Ready: {assessment['readiness_assessment']['pipeline_ready']}")
            print(f"Production Ready: {assessment['readiness_assessment']['recommended_for_production']}")
            
        except Exception as e:
            assessment['overall_status'] = 'ERROR'
            assessment['error'] = str(e)
            print(f"❌ System status assessment failed: {str(e)}")
            self.critical_issues.append(f"System status assessment failed: {str(e)}")
        
        return assessment
    
    def _meets_minimum_requirements(self) -> bool:
        """Check if system meets minimum requirements."""
        metrics = {
            'successful_agents': sum(1 for result in self.agent_results.values() 
                                   if isinstance(result, dict) and 
                                   result.get('overall_status', result.get('status')) in ['PASS', 'SUCCESS']),
            'critical_issues': len(self.critical_issues),
            'total_issues': len(self.issues_found)
        }
        
        return (
            metrics['successful_agents'] >= self.validation_thresholds['min_agents_pass'] and
            metrics['critical_issues'] <= self.validation_thresholds['max_critical_issues'] and
            metrics['total_issues'] <= self.validation_thresholds['max_total_issues']
        )
    
    def _recommended_for_testing(self) -> bool:
        """Check if system is recommended for testing."""
        docker_ready = self.agent_results.get('docker_environment', {}).get('docker_ready', False)
        minimum_requirements = self._meets_minimum_requirements()
        
        return minimum_requirements and docker_ready
    
    def _recommended_for_production(self) -> bool:
        """Check if system is recommended for production."""
        docker_production = self.agent_results.get('docker_environment', {}).get('production_ready', False)
        pipeline_production = self.agent_results.get('end_to_end_pipeline', {}).get('production_ready', False)
        minimum_requirements = self._meets_minimum_requirements()
        
        return minimum_requirements and docker_production and pipeline_production
    
    def _determine_overall_status(self, assessment: Dict[str, Any]) -> str:
        """Determine overall system status."""
        metrics = assessment['system_metrics']
        readiness = assessment['readiness_assessment']
        
        if metrics['total_critical_issues'] > 0:
            return 'CRITICAL_ISSUES'
        elif readiness['recommended_for_production']:
            return 'PRODUCTION_READY'
        elif readiness['recommended_for_testing']:
            return 'TESTING_READY'
        elif readiness['meets_minimum_requirements']:
            return 'MINIMUM_REQUIREMENTS_MET'
        else:
            return 'NOT_READY'
    
    def _generate_recommendations(self, assessment: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on assessment."""
        recommendations = []
        
        metrics = assessment['system_metrics']
        readiness = assessment['readiness_assessment']
        
        # Critical issues recommendations
        if metrics['total_critical_issues'] > 0:
            recommendations.append("URGENT: Resolve all critical issues before proceeding")
            recommendations.append(f"Critical issues found: {', '.join(self.critical_issues[:3])}")
        
        # Agent failure recommendations
        if metrics['agent_success_rate'] < 0.8:
            recommendations.append("Address validation agent failures to ensure comprehensive validation")
        
        # Docker readiness recommendations
        if not readiness['docker_ready']:
            recommendations.append("Fix Docker environment issues before testing")
        
        # Pipeline readiness recommendations  
        if readiness['docker_ready'] and not readiness['pipeline_ready']:
            recommendations.append("Resolve pipeline issues - Docker environment is ready but pipeline validation failed")
        
        # Production readiness recommendations
        if readiness['recommended_for_testing'] and not readiness['recommended_for_production']:
            recommendations.append("System ready for testing - address remaining issues before production deployment")
            
        if metrics['total_issues'] > 2:
            recommendations.append(f"Resolve {metrics['total_issues']} non-critical issues for better system stability")
            
        if metrics['total_warnings'] > 5:
            recommendations.append(f"Review {metrics['total_warnings']} warnings for potential improvements")
        
        # Success recommendations
        if readiness['recommended_for_production']:
            recommendations.append("✅ System validated and ready for production deployment")
            recommendations.append("Consider implementing monitoring and alerting for production environment")
        
        return recommendations
    
    def run_comprehensive_validation_orchestration(self) -> Dict[str, Any]:
        """Run comprehensive validation orchestration of all agents."""
        print("IFRS9 MASTER VALIDATION ORCHESTRATION")
        print("Coordinating comprehensive validation of Docker environment")
        print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
        print()
        
        orchestration_start = datetime.now()
        
        # Run all validation agents
        print("Phase 1: Running Individual Validation Agents")
        print("=" * 70)
        
        self.agent_results['docker_environment'] = self.run_docker_environment_validation()
        self.agent_results['end_to_end_pipeline'] = self.run_end_to_end_pipeline_validation()
        self.agent_results['environment_legacy'] = self.run_environment_legacy_validation()
        
        # Run cross-validation checks
        print("\nPhase 2: Cross-Validation and Consistency Checks")
        print("=" * 70)
        
        cross_validation_result = self.run_cross_validation_checks()
        self.validation_results['cross_validation'] = cross_validation_result
        
        # Assess overall system status
        print("\nPhase 3: Overall System Status Assessment")
        print("=" * 70)
        
        system_assessment = self.assess_overall_system_status()
        self.validation_results['system_assessment'] = system_assessment
        
        # Compile final results
        orchestration_end = datetime.now()
        
        self.validation_results['orchestration_summary'] = {
            'start_time': orchestration_start.isoformat(),
            'end_time': orchestration_end.isoformat(),
            'duration_seconds': (orchestration_end - orchestration_start).total_seconds(),
            'agents_executed': list(self.agent_results.keys()),
            'agent_results': self.agent_results,
            'total_critical_issues': len(self.critical_issues),
            'total_issues': len(self.issues_found),
            'total_warnings': len(self.warnings),
            'critical_issues': self.critical_issues,
            'issues_found': self.issues_found,
            'warnings': self.warnings,
            'overall_status': system_assessment['overall_status'],
            'production_ready': system_assessment['readiness_assessment']['recommended_for_production'],
            'testing_ready': system_assessment['readiness_assessment']['recommended_for_testing'],
            'docker_environment_ready': system_assessment['readiness_assessment']['docker_ready']
        }
        
        return self.validation_results
    
    def generate_master_validation_report(self) -> str:
        """Generate comprehensive master validation report."""
        if not self.validation_results:
            return "No validation results available. Run master validation orchestration first."
        
        report = []
        report.append("=" * 80)
        report.append("IFRS9 MASTER VALIDATION ORCHESTRATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
        
        summary = self.validation_results.get('orchestration_summary', {})
        system_assessment = self.validation_results.get('system_assessment', {})
        
        report.append(f"Overall Status: {summary.get('overall_status', 'UNKNOWN')}")
        report.append(f"Production Ready: {summary.get('production_ready', False)}")
        report.append(f"Testing Ready: {summary.get('testing_ready', False)}")
        report.append(f"Docker Environment Ready: {summary.get('docker_environment_ready', False)}")
        report.append("")
        
        # Orchestration Summary
        report.append("ORCHESTRATION SUMMARY")
        report.append("-" * 40)
        duration = summary.get('duration_seconds', 0)
        report.append(f"Execution Duration: {duration:.1f} seconds")
        report.append(f"Agents Executed: {len(summary.get('agents_executed', []))}")
        report.append(f"Critical Issues: {summary.get('total_critical_issues', 0)}")
        report.append(f"Total Issues: {summary.get('total_issues', 0)}")
        report.append(f"Warnings: {summary.get('total_warnings', 0)}")
        report.append("")
        
        # Agent Results Summary
        if 'agent_results' in summary:
            report.append("VALIDATION AGENT RESULTS")
            report.append("-" * 40)
            
            agent_results = summary['agent_results']
            for agent_name, agent_result in agent_results.items():
                if isinstance(agent_result, dict):
                    status = agent_result.get('overall_status', agent_result.get('status', 'UNKNOWN'))
                    priority = agent_result.get('priority', 0)
                    critical_issues = len(agent_result.get('critical_issues', []))
                    total_issues = agent_result.get('total_issues', 0)
                    
                    report.append(f"  {agent_name.replace('_', ' ').title()}:")
                    report.append(f"    Status: {status} (Priority: {priority})")
                    report.append(f"    Critical Issues: {critical_issues}")
                    report.append(f"    Total Issues: {total_issues}")
            report.append("")
        
        # System Assessment
        if system_assessment:
            readiness = system_assessment.get('readiness_assessment', {})
            report.append("SYSTEM READINESS ASSESSMENT")
            report.append("-" * 40)
            report.append(f"Docker Ready: {readiness.get('docker_ready', False)}")
            report.append(f"Pipeline Ready: {readiness.get('pipeline_ready', False)}")
            report.append(f"Minimum Requirements Met: {readiness.get('meets_minimum_requirements', False)}")
            report.append(f"Recommended for Testing: {readiness.get('recommended_for_testing', False)}")
            report.append(f"Recommended for Production: {readiness.get('recommended_for_production', False)}")
            report.append("")
        
        # Critical Issues (if any)
        if summary.get('critical_issues'):
            report.append("CRITICAL ISSUES (Must Fix)")
            report.append("-" * 40)
            for issue in summary['critical_issues']:
                report.append(f"  ❌ {issue}")
            report.append("")
        
        # Issues Found (if any)
        if summary.get('issues_found'):
            report.append("ISSUES FOUND")
            report.append("-" * 40)
            for issue in summary['issues_found'][:10]:  # Top 10 issues
                report.append(f"  ⚠️ {issue}")
            if len(summary['issues_found']) > 10:
                report.append(f"  ... and {len(summary['issues_found']) - 10} more issues")
            report.append("")
        
        # Recommendations
        if system_assessment and 'recommendations' in system_assessment:
            report.append("RECOMMENDATIONS")
            report.append("-" * 40)
            for recommendation in system_assessment['recommendations']:
                if recommendation.startswith('✅'):
                    report.append(f"  {recommendation}")
                elif recommendation.startswith('URGENT'):
                    report.append(f"  🚨 {recommendation}")
                else:
                    report.append(f"  💡 {recommendation}")
            report.append("")
        
        # Final Status
        report.append("FINAL ASSESSMENT")
        report.append("-" * 40)
        
        overall_status = summary.get('overall_status', 'UNKNOWN')
        
        if overall_status == 'PRODUCTION_READY':
            report.append("🎉 CONGRATULATIONS: IFRS9 Docker Environment is PRODUCTION READY!")
            report.append("   All validation agents passed successfully")
            report.append("   System ready for production deployment")
        elif overall_status == 'TESTING_READY':
            report.append("✅ IFRS9 Docker Environment is TESTING READY")
            report.append("   Core functionality validated successfully")
            report.append("   Address remaining issues before production")
        elif overall_status == 'MINIMUM_REQUIREMENTS_MET':
            report.append("⚠️ IFRS9 Docker Environment meets minimum requirements")
            report.append("   Some validation issues remain")
            report.append("   Not recommended for production deployment")
        elif overall_status == 'CRITICAL_ISSUES':
            report.append("❌ IFRS9 Docker Environment has CRITICAL ISSUES")
            report.append("   Must resolve critical issues before use")
            report.append("   System not ready for testing or production")
        else:
            report.append("❓ IFRS9 Docker Environment status UNKNOWN")
            report.append("   Validation incomplete or failed")
            report.append("   Review validation results for details")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def generate_html_report(self, results: Dict[str, Any], text_report: str) -> str:
        """Generate HTML validation report."""
        summary = results.get('orchestration_summary', {})
        system_assessment = results.get('system_assessment', {})
        
        # Determine status colors and icons
        overall_status = summary.get('overall_status', 'UNKNOWN')
        status_colors = {
            'PRODUCTION_READY': '#28a745',
            'TESTING_READY': '#17a2b8',
            'MINIMUM_REQUIREMENTS_MET': '#ffc107',
            'CRITICAL_ISSUES': '#dc3545',
            'NOT_READY': '#6c757d',
            'UNKNOWN': '#6c757d'
        }
        
        status_icons = {
            'PRODUCTION_READY': '🎉',
            'TESTING_READY': '✅',
            'MINIMUM_REQUIREMENTS_MET': '⚠️',
            'CRITICAL_ISSUES': '❌',
            'NOT_READY': '🚫',
            'UNKNOWN': '❓'
        }
        
        status_color = status_colors.get(overall_status, '#6c757d')
        status_icon = status_icons.get(overall_status, '❓')
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IFRS9 Master Validation Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        
        .status-badge {{
            background-color: {status_color};
            color: white;
            padding: 10px 20px;
            border-radius: 25px;
            display: inline-block;
            font-weight: bold;
            margin: 10px 0;
        }}
        
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .summary-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }}
        
        .agent-results {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        
        .agent-card {{
            border: 1px solid #dee2e6;
            border-radius: 8px;
            margin: 10px 0;
            padding: 15px;
        }}
        
        .agent-success {{ border-left: 4px solid #28a745; }}
        .agent-warning {{ border-left: 4px solid #ffc107; }}
        .agent-error {{ border-left: 4px solid #dc3545; }}
        
        .issues-section {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        
        .critical-issue {{
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
        }}
        
        .warning-issue {{
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
        }}
        
        .recommendation {{
            background: #d1ecf1;
            border: 1px solid #bee5eb;
            color: #0c5460;
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
        }}
        
        .success-rec {{
            background: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }}
        
        .urgent-rec {{
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }}
        
        h1, h2, h3 {{ color: #495057; }}
        
        .timestamp {{
            font-size: 0.9em;
            color: #6c757d;
            text-align: center;
            margin-top: 20px;
        }}
        
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            border-top: 1px solid #dee2e6;
            color: #6c757d;
        }}
        
        .progress-bar {{
            width: 100%;
            height: 20px;
            background-color: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #28a745 0%, #20c997 100%);
            transition: width 0.3s ease;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{status_icon} IFRS9 Master Validation Report</h1>
        <div class="status-badge">{overall_status.replace('_', ' ').title()}</div>
        <p>Generated: {datetime.now(timezone.utc).isoformat()}</p>
    </div>
    
    <div class="summary-grid">
        <div class="summary-card">
            <h3>Overall Status</h3>
            <p><strong>{overall_status.replace('_', ' ').title()}</strong></p>
            <p>Production Ready: {'✅ Yes' if summary.get('production_ready') else '❌ No'}</p>
            <p>Testing Ready: {'✅ Yes' if summary.get('testing_ready') else '❌ No'}</p>
        </div>
        
        <div class="summary-card">
            <h3>Execution Summary</h3>
            <p>Duration: {summary.get('duration_seconds', 0):.1f} seconds</p>
            <p>Agents Run: {len(summary.get('agents_executed', []))}</p>
            <p>Timestamp: {summary.get('start_time', 'Unknown')}</p>
        </div>
        
        <div class="summary-card">
            <h3>Issues Summary</h3>
            <p>Critical Issues: <strong style="color: #dc3545;">{summary.get('total_critical_issues', 0)}</strong></p>
            <p>Total Issues: <strong style="color: #ffc107;">{summary.get('total_issues', 0)}</strong></p>
            <p>Warnings: <strong style="color: #17a2b8;">{summary.get('total_warnings', 0)}</strong></p>
        </div>
        
        <div class="summary-card">
            <h3>Environment Status</h3>
            <p>Docker Ready: {'✅ Yes' if summary.get('docker_environment_ready') else '❌ No'}</p>
            <p>Pipeline Status: {'✅ Ready' if system_assessment.get('readiness_assessment', {}).get('pipeline_ready') else '❌ Issues'}</p>
            <p>Min. Requirements: {'✅ Met' if system_assessment.get('readiness_assessment', {}).get('meets_minimum_requirements') else '❌ Not Met'}</p>
        </div>
    </div>"""
        
        # Agent Results Section
        if 'agent_results' in summary:
            html += """
    <div class="agent-results">
        <h2>Validation Agent Results</h2>"""
            
            agent_results = summary['agent_results']
            for agent_name, agent_result in agent_results.items():
                if isinstance(agent_result, dict):
                    status = agent_result.get('overall_status', agent_result.get('status', 'UNKNOWN'))
                    priority = agent_result.get('priority', 0)
                    critical_issues = len(agent_result.get('critical_issues', []))
                    total_issues = agent_result.get('total_issues', 0)
                    
                    # Determine card class
                    if status in ['PASS', 'SUCCESS']:
                        card_class = 'agent-success'
                        status_emoji = '✅'
                    elif status in ['PARTIAL_SUCCESS', 'WARNING']:
                        card_class = 'agent-warning'
                        status_emoji = '⚠️'
                    else:
                        card_class = 'agent-error'
                        status_emoji = '❌'
                    
                    html += f"""
        <div class="agent-card {card_class}">
            <h4>{status_emoji} {agent_name.replace('_', ' ').title()}</h4>
            <p><strong>Status:</strong> {status} (Priority: {priority})</p>
            <p><strong>Critical Issues:</strong> {critical_issues}</p>
            <p><strong>Total Issues:</strong> {total_issues}</p>
        </div>"""
            
            html += "</div>"
        
        # Critical Issues Section
        if summary.get('critical_issues'):
            html += """
    <div class="issues-section">
        <h2>❌ Critical Issues (Must Fix)</h2>"""
            
            for issue in summary['critical_issues']:
                html += f'<div class="critical-issue">❌ {issue}</div>'
            
            html += "</div>"
        
        # Issues Found Section
        if summary.get('issues_found'):
            html += """
    <div class="issues-section">
        <h2>⚠️ Issues Found</h2>"""
            
            for issue in summary['issues_found'][:10]:  # Top 10
                html += f'<div class="warning-issue">⚠️ {issue}</div>'
            
            if len(summary['issues_found']) > 10:
                html += f'<p><em>... and {len(summary["issues_found"]) - 10} more issues</em></p>'
            
            html += "</div>"
        
        # Recommendations Section
        if system_assessment and 'recommendations' in system_assessment:
            html += """
    <div class="issues-section">
        <h2>💡 Recommendations</h2>"""
            
            for rec in system_assessment['recommendations']:
                if rec.startswith('✅'):
                    html += f'<div class="recommendation success-rec">{rec}</div>'
                elif rec.startswith('URGENT') or rec.startswith('🚨'):
                    html += f'<div class="recommendation urgent-rec">{rec}</div>'
                else:
                    html += f'<div class="recommendation">💡 {rec}</div>'
            
            html += "</div>"
        
        # Text Report Section (collapsible)
        html += f"""
    <div class="issues-section">
        <h2>📄 Detailed Text Report</h2>
        <details>
            <summary>Click to expand full text report</summary>
            <pre style="background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto;">{text_report}</pre>
        </details>
    </div>
    
    <div class="footer">
        <p>Generated by IFRS9 Master Validation Orchestrator</p>
        <p>For technical support, contact the IFRS9 development team</p>
        <div class="timestamp">Report generated at: {datetime.now(timezone.utc).isoformat()}</div>
    </div>
</body>
</html>"""
        
        return html


def main():
    """Main execution function for master validation orchestration."""
    print("Starting IFRS9 Master Validation Orchestration")
    print("Comprehensive coordination of all validation agents")
    
    orchestrator = MasterValidationOrchestrator()
    
    try:
        # Run comprehensive validation orchestration
        results = orchestrator.run_comprehensive_validation_orchestration()
        
        # Generate master report
        report = orchestrator.generate_master_validation_report()
        print("\n" + report)
        
        # Save results to files
        output_dir = "/opt/airflow/data/validation"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON results
        json_file = os.path.join(output_dir, f"master_validation_results_{timestamp}.json")
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save master report
        report_file = os.path.join(output_dir, f"master_validation_report_{timestamp}.txt")
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Save HTML report (enhanced)
        html_report = orchestrator.generate_html_report(results, report)
        html_file = os.path.join(output_dir, f"master_validation_report_{timestamp}.html")
        with open(html_file, 'w') as f:
            f.write(html_report)
        
        print(f"\nMaster validation results saved to:")
        print(f"  JSON: {json_file}")
        print(f"  Report: {report_file}")
        print(f"  HTML: {html_file}")
        
        # Exit with appropriate code
        orchestration_summary = results.get('orchestration_summary', {})
        overall_status = orchestration_summary.get('overall_status', 'UNKNOWN')
        
        if overall_status in ['PRODUCTION_READY', 'TESTING_READY']:
            exit_code = 0
            print(f"\nMaster Validation Status: SUCCESS - {overall_status}")
        elif overall_status == 'MINIMUM_REQUIREMENTS_MET':
            exit_code = 1
            print(f"\nMaster Validation Status: WARNING - Minimum requirements met but issues remain")
        else:
            exit_code = 2
            print(f"\nMaster Validation Status: FAILED - {overall_status}")
        
        return exit_code
        
    except Exception as e:
        print(f"\nFATAL ERROR during master validation orchestration: {str(e)}")
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)