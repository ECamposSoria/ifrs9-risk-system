#!/usr/bin/env python3
"""Optimized test runner for IFRS9 system.

This script provides comprehensive test execution with coverage collection,
performance monitoring, and detailed reporting optimized for containerized environments.
"""

import os
import sys
import time
import argparse
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / 'logs' / 'test_execution.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


class OptimizedTestRunner:
    """Optimized test runner for IFRS9 system with comprehensive reporting."""
    
    def __init__(self, project_root: Path):
        """Initialize the test runner.
        
        Args:
            project_root: Path to the project root directory
        """
        self.project_root = project_root
        self.test_results: Dict[str, Any] = {}
        self.coverage_threshold = 80
        
    def setup_environment(self) -> bool:
        """Setup test environment and validate dependencies.
        
        Returns:
            True if setup successful, False otherwise
        """
        logger.info("Setting up test environment...")
        
        try:
            # Create necessary directories
            (self.project_root / 'logs').mkdir(exist_ok=True)
            (self.project_root / 'test-results').mkdir(exist_ok=True)
            (self.project_root / 'htmlcov').mkdir(exist_ok=True)
            
            # Verify Python dependencies
            required_packages = [
                'pytest', 'pytest-cov', 'pyspark', 'pandas', 'numpy'
            ]
            
            for package in required_packages:
                try:
                    __import__(package.replace('-', '_'))
                    logger.debug(f"✓ {package} available")
                except ImportError:
                    logger.error(f"✗ {package} not available")
                    return False
            
            # Check Spark environment
            spark_home = os.environ.get('SPARK_HOME')
            if not spark_home:
                logger.warning("SPARK_HOME not set, using bundled Spark")
            
            # Verify Java availability
            java_result = subprocess.run(['java', '-version'], 
                                       capture_output=True, text=True)
            if java_result.returncode != 0:
                logger.error("Java not available - required for Spark")
                return False
            
            logger.info("✓ Test environment setup successful")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup test environment: {e}")
            return False
    
    def run_test_suite(self, 
                      test_pattern: str = "tests/",
                      markers: Optional[List[str]] = None,
                      parallel: bool = False,
                      verbose: bool = True) -> Dict[str, Any]:
        """Run the complete test suite with optimized configuration.
        
        Args:
            test_pattern: Pattern for test discovery
            markers: Pytest markers to include/exclude
            parallel: Whether to run tests in parallel
            verbose: Whether to use verbose output
            
        Returns:
            Dictionary with test execution results
        """
        logger.info(f"Starting test execution: {test_pattern}")
        start_time = time.time()
        
        # Build pytest command
        cmd = [
            'python', '-m', 'pytest',
            test_pattern,
            '--cov=src',
            '--cov-report=term-missing',
            '--cov-report=html:htmlcov',
            '--cov-report=xml:coverage.xml',
            f'--cov-fail-under={self.coverage_threshold}',
            '--junitxml=test-results/junit.xml',
            '--tb=short',
            '--strict-markers',
            '--strict-config'
        ]
        
        if verbose:
            cmd.append('--verbose')
        
        if markers:
            for marker in markers:
                cmd.extend(['-m', marker])
        
        if parallel:
            cmd.extend(['-n', 'auto'])
        
        # Add duration reporting
        cmd.append('--durations=10')
        
        # Set environment variables for optimal performance
        env = os.environ.copy()
        env.update({
            'PYTHONPATH': str(self.project_root),
            'SPARK_LOCAL_IP': '127.0.0.1',
            'PYSPARK_PYTHON': sys.executable,
            'PYSPARK_DRIVER_PYTHON': sys.executable
        })
        
        # Run tests
        try:
            logger.info(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )
            
            execution_time = time.time() - start_time
            
            # Parse results
            test_results = {
                'command': ' '.join(cmd),
                'exit_code': result.returncode,
                'execution_time': execution_time,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'success': result.returncode == 0
            }
            
            # Log results
            if result.returncode == 0:
                logger.info(f"✓ Tests passed in {execution_time:.2f}s")
            else:
                logger.error(f"✗ Tests failed with exit code {result.returncode}")
                logger.error(f"STDERR: {result.stderr}")
            
            return test_results
            
        except subprocess.TimeoutExpired:
            logger.error("Test execution timed out (30 minutes)")
            return {
                'command': ' '.join(cmd),
                'exit_code': -1,
                'execution_time': 1800,
                'error': 'Timeout',
                'success': False
            }
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return {
                'command': ' '.join(cmd),
                'exit_code': -1,
                'execution_time': time.time() - start_time,
                'error': str(e),
                'success': False
            }
    
    def run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests with fast execution."""
        logger.info("Running unit tests...")
        return self.run_test_suite(
            test_pattern="tests/",
            markers=["unit", "not slow", "not integration"],
            parallel=True,
            verbose=True
        )
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests."""
        logger.info("Running integration tests...")
        return self.run_test_suite(
            test_pattern="tests/",
            markers=["integration"],
            parallel=False,
            verbose=True
        )
    
    def run_spark_tests(self) -> Dict[str, Any]:
        """Run Spark-specific tests."""
        logger.info("Running Spark tests...")
        return self.run_test_suite(
            test_pattern="tests/",
            markers=["spark"],
            parallel=False,
            verbose=True
        )
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests."""
        logger.info("Running performance tests...")
        return self.run_test_suite(
            test_pattern="tests/",
            markers=["slow", "ml"],
            parallel=False,
            verbose=True
        )
    
    def analyze_coverage(self) -> Dict[str, Any]:
        """Analyze test coverage and generate detailed report.
        
        Returns:
            Dictionary with coverage analysis results
        """
        logger.info("Analyzing test coverage...")
        
        coverage_file = self.project_root / 'coverage.xml'
        if not coverage_file.exists():
            logger.warning("Coverage file not found")
            return {'error': 'Coverage file not available'}
        
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(coverage_file)
            root = tree.getroot()
            
            # Extract overall coverage
            overall_coverage = float(root.get('line-rate', 0)) * 100
            
            # Extract package coverage
            packages = {}
            for package in root.findall('.//package'):
                package_name = package.get('name', 'unknown')
                package_coverage = float(package.get('line-rate', 0)) * 100
                packages[package_name] = package_coverage
            
            # Extract class coverage
            classes = {}
            for cls in root.findall('.//class'):
                class_name = cls.get('name', 'unknown')
                class_coverage = float(cls.get('line-rate', 0)) * 100
                classes[class_name] = class_coverage
            
            coverage_report = {
                'overall_coverage': overall_coverage,
                'coverage_threshold': self.coverage_threshold,
                'passes_threshold': overall_coverage >= self.coverage_threshold,
                'packages': packages,
                'classes': classes,
                'timestamp': time.time()
            }
            
            logger.info(f"Overall coverage: {overall_coverage:.1f}% "
                       f"(threshold: {self.coverage_threshold}%)")
            
            return coverage_report
            
        except Exception as e:
            logger.error(f"Failed to analyze coverage: {e}")
            return {'error': str(e)}
    
    def generate_comprehensive_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive test execution report.
        
        Args:
            results: Test execution results
            
        Returns:
            Comprehensive report dictionary
        """
        logger.info("Generating comprehensive test report...")
        
        # Analyze coverage
        coverage_analysis = self.analyze_coverage()
        
        # Parse junit results if available
        junit_file = self.project_root / 'test-results' / 'junit.xml'
        junit_analysis = {}
        
        if junit_file.exists():
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(junit_file)
                root = tree.getroot()
                
                junit_analysis = {
                    'total_tests': int(root.get('tests', 0)),
                    'failures': int(root.get('failures', 0)),
                    'errors': int(root.get('errors', 0)),
                    'skipped': int(root.get('skipped', 0)),
                    'time': float(root.get('time', 0)),
                    'success_rate': 0
                }
                
                if junit_analysis['total_tests'] > 0:
                    passed = (junit_analysis['total_tests'] - 
                             junit_analysis['failures'] - 
                             junit_analysis['errors'])
                    junit_analysis['success_rate'] = passed / junit_analysis['total_tests'] * 100
                
                logger.info(f"Test results: {junit_analysis['total_tests']} tests, "
                           f"{junit_analysis['success_rate']:.1f}% success rate")
                
            except Exception as e:
                logger.warning(f"Failed to parse junit results: {e}")
        
        # Compile comprehensive report
        report = {
            'timestamp': time.time(),
            'project_root': str(self.project_root),
            'test_execution': results,
            'coverage_analysis': coverage_analysis,
            'junit_analysis': junit_analysis,
            'summary': {
                'overall_success': (
                    results.get('success', False) and 
                    coverage_analysis.get('passes_threshold', False)
                ),
                'execution_time': results.get('execution_time', 0),
                'coverage_percentage': coverage_analysis.get('overall_coverage', 0),
                'test_count': junit_analysis.get('total_tests', 0),
                'success_rate': junit_analysis.get('success_rate', 0)
            }
        }
        
        # Save report
        report_file = self.project_root / 'test-results' / 'comprehensive_report.json'
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Report saved to: {report_file}")
        except Exception as e:
            logger.warning(f"Failed to save report: {e}")
        
        return report
    
    def run_full_test_suite(self) -> Dict[str, Any]:
        """Run the complete test suite with all optimizations.
        
        Returns:
            Comprehensive test execution results
        """
        logger.info("Starting full test suite execution...")
        
        # Setup environment
        if not self.setup_environment():
            return {'error': 'Environment setup failed'}
        
        # Run all tests
        results = self.run_test_suite(
            test_pattern="tests/",
            markers=None,
            parallel=False,  # Disabled for Spark compatibility
            verbose=True
        )
        
        # Generate comprehensive report
        report = self.generate_comprehensive_report(results)
        
        # Log summary
        summary = report['summary']
        logger.info("=== TEST EXECUTION SUMMARY ===")
        logger.info(f"Overall Success: {'✓' if summary['overall_success'] else '✗'}")
        logger.info(f"Execution Time: {summary['execution_time']:.2f}s")
        logger.info(f"Coverage: {summary['coverage_percentage']:.1f}%")
        logger.info(f"Test Count: {summary['test_count']}")
        logger.info(f"Success Rate: {summary['success_rate']:.1f}%")
        logger.info("================================")
        
        return report


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(description="Optimized IFRS9 Test Runner")
    parser.add_argument(
        '--test-type',
        choices=['unit', 'integration', 'spark', 'performance', 'full'],
        default='full',
        help='Type of tests to run'
    )
    parser.add_argument(
        '--coverage-threshold',
        type=int,
        default=80,
        help='Coverage threshold percentage'
    )
    parser.add_argument(
        '--markers',
        nargs='*',
        help='Pytest markers to include'
    )
    parser.add_argument(
        '--pattern',
        default='tests/',
        help='Test file pattern'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run tests in parallel'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = OptimizedTestRunner(project_root)
    runner.coverage_threshold = args.coverage_threshold
    
    # Run specified tests
    if args.test_type == 'unit':
        results = runner.run_unit_tests()
    elif args.test_type == 'integration':
        results = runner.run_integration_tests()
    elif args.test_type == 'spark':
        results = runner.run_spark_tests()
    elif args.test_type == 'performance':
        results = runner.run_performance_tests()
    elif args.test_type == 'full':
        results = runner.run_full_test_suite()
    else:
        # Custom test run
        results = runner.run_test_suite(
            test_pattern=args.pattern,
            markers=args.markers,
            parallel=args.parallel,
            verbose=args.verbose
        )
        results = runner.generate_comprehensive_report(results)
    
    # Exit with appropriate code
    if results.get('summary', {}).get('overall_success', False):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()