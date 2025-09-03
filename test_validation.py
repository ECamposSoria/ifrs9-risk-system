#!/usr/bin/env python3
"""Validation script for optimized test configuration.

This script validates the test optimization improvements without requiring
full PySpark setup, focusing on the structure and fixture design.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tests.test_data_factory import TestDataFactory


def validate_test_data_factory():
    """Validate the test data factory functionality."""
    print("🔍 Validating Test Data Factory...")
    
    factory = TestDataFactory(seed=42)
    
    # Test basic portfolio generation
    loans = factory.generate_loan_portfolio(n_loans=10)
    assert len(loans) > 10, "Should include edge cases"
    
    # Test required columns
    required_cols = [
        'loan_id', 'customer_id', 'loan_amount', 'current_balance',
        'days_past_due', 'credit_score', 'loan_type', 'provision_stage'
    ]
    for col in required_cols:
        assert col in loans.columns, f"Missing required column: {col}"
    
    # Test stage distribution
    stages = loans['provision_stage'].unique()
    assert 'STAGE_1' in stages, "Missing STAGE_1"
    assert 'STAGE_2' in stages, "Missing STAGE_2"
    assert 'STAGE_3' in stages, "Missing STAGE_3"
    
    # Test data quality
    assert loans['loan_amount'].min() > 0, "Loan amounts should be positive"
    assert loans['credit_score'].between(300, 850).all(), "Credit scores out of range"
    assert loans['days_past_due'].min() >= 0, "DPD should be non-negative"
    
    # Test edge cases
    edge_case_ids = loans[loans['loan_id'].str.startswith('E')]['loan_id'].tolist()
    assert len(edge_case_ids) > 0, "Should have edge cases"
    
    print("✅ Test Data Factory validation passed")
    return True


def validate_validation_test_data():
    """Validate the validation test data generation."""
    print("🔍 Validating Validation Test Data...")
    
    factory = TestDataFactory(seed=42)
    invalid_data = factory.generate_validation_test_data()
    
    # Check that invalid data contains expected issues
    assert len(invalid_data) > 0, "Should have validation test cases"
    assert 'test_issue' in invalid_data.columns, "Should have test_issue column"
    
    issues = invalid_data['test_issue'].unique()
    expected_issues = ['invalid_loan_id', 'negative_amount', 'credit_score_too_high']
    for issue in expected_issues:
        assert issue in issues, f"Missing test issue: {issue}"
    
    print("✅ Validation Test Data validation passed")
    return True


def validate_performance_test_data():
    """Validate performance test data generation."""
    print("🔍 Validating Performance Test Data...")
    
    factory = TestDataFactory(seed=42)
    perf_data = factory.generate_performance_test_data(n_loans=100)
    
    assert len(perf_data) == 100, f"Expected 100 loans, got {len(perf_data)}"
    assert not perf_data.empty, "Performance data should not be empty"
    
    # Test data distribution
    stage_dist = perf_data['provision_stage'].value_counts(normalize=True)
    assert stage_dist['STAGE_1'] > 0.5, "Should have majority Stage 1 loans"
    
    print("✅ Performance Test Data validation passed")
    return True


def validate_pytest_configuration():
    """Validate pytest configuration."""
    print("🔍 Validating Pytest Configuration...")
    
    # Check pytest.ini exists and has required sections
    pytest_ini = project_root / 'pytest.ini'
    assert pytest_ini.exists(), "pytest.ini should exist"
    
    with open(pytest_ini, 'r') as f:
        content = f.read()
        
    # Check required configurations
    required_configs = [
        'testpaths = tests',
        '--cov=src',
        '--cov-fail-under=80',
        'markers =',
        'spark: marks tests as requiring Spark session'
    ]
    
    for config in required_configs:
        assert config in content, f"Missing pytest config: {config}"
    
    print("✅ Pytest Configuration validation passed")
    return True


def validate_conftest_structure():
    """Validate conftest.py structure without importing PySpark."""
    print("🔍 Validating Conftest Structure...")
    
    conftest_file = project_root / 'tests' / 'conftest.py'
    assert conftest_file.exists(), "conftest.py should exist"
    
    with open(conftest_file, 'r') as f:
        content = f.read()
    
    # Check for required fixtures
    required_fixtures = [
        'def spark_session',
        'def datetime_converter',
        'def sample_loan_data',
        'def test_data_manager',
        'def ml_test_data'
    ]
    
    for fixture in required_fixtures:
        assert fixture in content, f"Missing fixture: {fixture}"
    
    # Check for proper imports
    required_imports = [
        'import pytest',
        'from datetime_converter import DateTimeConverter',
        'from pyspark.sql import SparkSession'
    ]
    
    for import_stmt in required_imports:
        assert import_stmt in content, f"Missing import: {import_stmt}"
    
    print("✅ Conftest Structure validation passed")
    return True


def validate_test_files_structure():
    """Validate test file structure."""
    print("🔍 Validating Test Files Structure...")
    
    test_files = [
        'tests/test_rules.py',
        'tests/test_rules_optimized.py',
        'tests/conftest.py',
        'tests/test_data_factory.py',
        'tests/test_cleanup_manager.py'
    ]
    
    for test_file in test_files:
        file_path = project_root / test_file
        assert file_path.exists(), f"Missing test file: {test_file}"
    
    # Check test_rules_optimized.py structure
    optimized_tests = project_root / 'tests' / 'test_rules_optimized.py'
    with open(optimized_tests, 'r') as f:
        content = f.read()
    
    required_elements = [
        '@pytest.mark.spark',
        '@pytest.mark.unit',
        'def test_staging_classification_basic',
        'spark_session',
        'datetime_converter'
    ]
    
    for element in required_elements:
        assert element in content, f"Missing element in optimized tests: {element}"
    
    print("✅ Test Files Structure validation passed")
    return True


def validate_scripts_structure():
    """Validate scripts structure."""
    print("🔍 Validating Scripts Structure...")
    
    script_files = [
        'scripts/run_optimized_tests.py'
    ]
    
    for script_file in script_files:
        file_path = project_root / script_file
        assert file_path.exists(), f"Missing script file: {script_file}"
        assert file_path.stat().st_mode & 0o111, f"Script not executable: {script_file}"
    
    print("✅ Scripts Structure validation passed")
    return True


def generate_validation_report():
    """Generate comprehensive validation report."""
    print("📊 Generating Validation Report...")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'validation_results': {
            'test_data_factory': False,
            'validation_test_data': False,
            'performance_test_data': False,
            'pytest_configuration': False,
            'conftest_structure': False,
            'test_files_structure': False,
            'scripts_structure': False
        },
        'overall_success': False
    }
    
    try:
        # Run all validations
        validations = [
            ('test_data_factory', validate_test_data_factory),
            ('validation_test_data', validate_validation_test_data),
            ('performance_test_data', validate_performance_test_data),
            ('pytest_configuration', validate_pytest_configuration),
            ('conftest_structure', validate_conftest_structure),
            ('test_files_structure', validate_test_files_structure),
            ('scripts_structure', validate_scripts_structure)
        ]
        
        for name, validation_func in validations:
            try:
                result = validation_func()
                report['validation_results'][name] = result
            except Exception as e:
                print(f"❌ {name} validation failed: {e}")
                report['validation_results'][name] = False
        
        # Calculate overall success
        report['overall_success'] = all(report['validation_results'].values())
        
        # Save report
        report_file = project_root / 'test_validation_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Report saved to: {report_file}")
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        report['error'] = str(e)
    
    return report


def main():
    """Main validation execution."""
    print("🚀 Starting IFRS9 Test Optimization Validation")
    print("=" * 50)
    
    report = generate_validation_report()
    
    print("\n" + "=" * 50)
    print("📋 VALIDATION SUMMARY")
    print("=" * 50)
    
    for name, result in report['validation_results'].items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:25} {status}")
    
    print("-" * 50)
    overall_status = "✅ SUCCESS" if report['overall_success'] else "❌ FAILED"
    print(f"{'Overall Status':25} {overall_status}")
    
    if report['overall_success']:
        print("\n🎉 All test optimizations validated successfully!")
        print("The IFRS9 test suite is ready for containerized execution.")
    else:
        print("\n⚠️  Some validations failed. Review the issues above.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)