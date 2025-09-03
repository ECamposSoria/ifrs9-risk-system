#!/usr/bin/env python3
"""
IFRS9 Containerized Validation Runner
=====================================
Main script to execute comprehensive containerized validation testing.

This script orchestrates:
1. Dataset generation for all containerized components
2. Validation testing across the entire infrastructure
3. Report generation and notifications
4. Integration with monitoring and alerting

Usage:
    python run_containerized_validation.py [--generate-datasets] [--run-tests] [--suite SUITE_NAME]

Author: IFRS9 Risk System Team
Version: 1.0.0
Date: 2025-09-03
"""

import asyncio
import argparse
import sys
import os
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from validation.containerized_validation_framework import ContainerizedValidationFramework
from validation.containerized_test_orchestrator import ContainerizedTestOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'/tmp/ifrs9_containerized_validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

class ContainerizedValidationRunner:
    """Main runner for containerized validation process"""
    
    def __init__(self):
        self.framework = ContainerizedValidationFramework()
        self.orchestrator = ContainerizedTestOrchestrator()
        self.start_time = datetime.now()
        
    async def generate_validation_datasets(self) -> Dict[str, str]:
        """Generate all validation datasets"""
        logger.info("🚀 Starting validation dataset generation for containerized infrastructure")
        
        try:
            results = await self.framework.generate_all_validation_datasets()
            
            # Verify dataset generation
            datasets_dir = Path("/home/eze/projects/ifrs9-risk-system/validation/datasets")
            generated_files = list(datasets_dir.glob("*.csv"))
            
            logger.info(f"✅ Successfully generated {len(generated_files)} validation dataset files")
            for file_path in sorted(generated_files):
                logger.info(f"   📊 {file_path.name}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Dataset generation failed: {e}")
            raise

    async def run_validation_tests(self, suite_name: str = None) -> Dict[str, Any]:
        """Run validation tests"""
        logger.info("🧪 Starting containerized validation testing")
        
        try:
            if suite_name:
                logger.info(f"Running specific validation suite: {suite_name}")
                results = await self.orchestrator.run_validation_suite(suite_name)
                
                # Create summary for single suite
                passed = len([r for r in results if r.status == "PASS"])
                total = len(results)
                
                summary = {
                    'execution_summary': {
                        'suite_name': suite_name,
                        'total_tests': total,
                        'passed_tests': passed,
                        'success_rate': passed / total if total > 0 else 0
                    },
                    'detailed_results': results
                }
            else:
                logger.info("Running all validation suites")
                summary = await self.orchestrator.run_all_validation_suites()
            
            success_rate = summary['execution_summary']['success_rate']
            logger.info(f"✅ Validation testing completed - Success rate: {success_rate:.1%}")
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Validation testing failed: {e}")
            raise

    async def run_complete_validation(self) -> Dict[str, Any]:
        """Run complete validation process (datasets + tests)"""
        logger.info("🎯 Starting complete containerized validation process")
        
        complete_results = {
            'start_time': self.start_time.isoformat(),
            'dataset_generation': {},
            'validation_testing': {},
            'overall_status': 'RUNNING'
        }
        
        try:
            # Step 1: Generate validation datasets
            logger.info("Step 1/2: Generating validation datasets")
            dataset_results = await self.generate_validation_datasets()
            complete_results['dataset_generation'] = dataset_results
            
            # Step 2: Run validation tests
            logger.info("Step 2/2: Running validation tests")
            test_results = await self.run_validation_tests()
            complete_results['validation_testing'] = test_results
            
            # Determine overall status
            success_rate = test_results['execution_summary']['success_rate']
            if success_rate >= 0.95:
                complete_results['overall_status'] = 'EXCELLENT'
            elif success_rate >= 0.85:
                complete_results['overall_status'] = 'GOOD'
            elif success_rate >= 0.70:
                complete_results['overall_status'] = 'WARNING'
            else:
                complete_results['overall_status'] = 'CRITICAL'
            
            duration = (datetime.now() - self.start_time).total_seconds()
            complete_results['end_time'] = datetime.now().isoformat()
            complete_results['total_duration_seconds'] = duration
            
            # Save complete results
            results_file = f"/tmp/ifrs9_complete_validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump(complete_results, f, indent=2, default=str)
            
            logger.info(f"🎉 Complete validation finished - Status: {complete_results['overall_status']}")
            logger.info(f"📋 Results saved to: {results_file}")
            
            return complete_results
            
        except Exception as e:
            complete_results['overall_status'] = 'FAILED'
            complete_results['error'] = str(e)
            logger.error(f"❌ Complete validation failed: {e}")
            raise

    def print_results_summary(self, results: Dict[str, Any]):
        """Print human-readable results summary"""
        print("\n" + "="*80)
        print("🏗️  IFRS9 CONTAINERIZED VALIDATION RESULTS")
        print("="*80)
        
        if 'validation_testing' in results:
            testing_results = results['validation_testing']
            summary = testing_results['execution_summary']
            
            print(f"📊 EXECUTION SUMMARY:")
            print(f"   • Total Tests: {summary['total_tests']}")
            print(f"   • Passed: {summary['passed_tests']} ✅")
            print(f"   • Failed: {summary['failed_tests']} ❌")
            print(f"   • Errors: {summary['error_tests']} ⚠️")
            print(f"   • Success Rate: {summary['success_rate']:.1%}")
            print(f"   • Duration: {summary['total_duration_seconds']:.2f}s")
            
            if 'suite_results' in testing_results:
                print(f"\n📋 SUITE BREAKDOWN:")
                for suite_name, suite_data in testing_results['suite_results'].items():
                    status_icon = "✅" if suite_data['success_rate'] >= 0.9 else "❌" if suite_data['success_rate'] < 0.7 else "⚠️"
                    suite_display = suite_name.replace('_', ' ').title()
                    print(f"   • {suite_display}: {suite_data['success_rate']:.1%} {status_icon} ({suite_data['passed_tests']}/{suite_data['total_tests']})")
        
        print(f"\n🎯 OVERALL STATUS: {results.get('overall_status', 'UNKNOWN')}")
        
        if results.get('overall_status') == 'EXCELLENT':
            print("🎉 Outstanding! Your containerized infrastructure is performing exceptionally well.")
        elif results.get('overall_status') == 'GOOD':
            print("👍 Great job! Minor issues detected but overall infrastructure is healthy.")
        elif results.get('overall_status') == 'WARNING':
            print("⚠️  Attention needed! Several issues require investigation.")
        elif results.get('overall_status') == 'CRITICAL':
            print("🚨 Critical issues detected! Immediate action required.")
        
        print("="*80 + "\n")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='IFRS9 Containerized Validation Runner')
    parser.add_argument('--generate-datasets', action='store_true', 
                       help='Generate validation datasets only')
    parser.add_argument('--run-tests', action='store_true',
                       help='Run validation tests only (requires existing datasets)')
    parser.add_argument('--suite', type=str, choices=[
        'container_orchestration', 'infrastructure', 'monitoring', 'cicd', 'cloud_integration'
    ], help='Run specific validation suite only')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # If no specific action, run complete validation
    if not args.generate_datasets and not args.run_tests:
        args.generate_datasets = True
        args.run_tests = True
    
    runner = ContainerizedValidationRunner()
    
    try:
        if args.generate_datasets and args.run_tests:
            # Complete validation
            results = asyncio.run(runner.run_complete_validation())
        elif args.generate_datasets:
            # Dataset generation only
            results = asyncio.run(runner.generate_validation_datasets())
            results = {'dataset_generation': results, 'overall_status': 'DATASETS_GENERATED'}
        elif args.run_tests:
            # Testing only
            results = asyncio.run(runner.run_validation_tests(args.suite))
            results = {'validation_testing': results, 'overall_status': 'TESTS_COMPLETED'}
        
        # Print summary
        runner.print_results_summary(results)
        
        # Exit codes for CI/CD integration
        if results.get('overall_status') in ['EXCELLENT', 'GOOD', 'DATASETS_GENERATED', 'TESTS_COMPLETED']:
            sys.exit(0)
        elif results.get('overall_status') == 'WARNING':
            sys.exit(1)
        else:  # CRITICAL or FAILED
            sys.exit(2)
            
    except KeyboardInterrupt:
        logger.info("⏹️ Validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()