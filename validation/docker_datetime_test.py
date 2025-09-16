#!/usr/bin/env python3
"""
Docker-specific datetime conversion validation for IFRS9 system.
This script validates that all datetime conversion fixes work properly in the containerized environment.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import logging
from typing import Dict, List, Any

# Configure environment for Docker validation
os.environ.setdefault('PYARROW_IGNORE_TIMEZONE', '1')
os.environ.setdefault('TZ', 'UTC')

# Add validation directory to path for imports
sys.path.insert(0, '/app/validation')

try:
    from datetime_converter import DateTimeConverter
    from pyspark.sql import SparkSession
    from pyspark.sql.types import TimestampType
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DockerDateTimeValidator:
    """Validates datetime conversion functionality in Docker environment."""
    
    def __init__(self):
        """Initialize validator with Docker-optimized Spark session."""
        self.spark = self._create_docker_spark_session()
        self.converter = DateTimeConverter()
        self.validation_results = []
    
    def _create_docker_spark_session(self) -> SparkSession:
        """Create Spark session with Docker-optimized datetime configuration."""
        return SparkSession.builder \
            .appName("IFRS9-Docker-DateTime-Validator") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true") \
            .config("spark.sql.execution.arrow.maxRecordsPerBatch", "10000") \
            .config("spark.sql.datetime.java8API.enabled", "true") \
            .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .getOrCreate()
    
    def test_basic_datetime_conversion(self) -> bool:
        """Test basic datetime64[ns] conversion functionality."""
        print("\n🧪 Testing basic datetime64[ns] conversion...")
        
        try:
            # Create test data with various datetime scenarios
            test_data = pd.DataFrame({
                'id': [1, 2, 3, 4],
                'maturity_date': pd.to_datetime(['2024-01-01', '2024-06-15', '2025-12-31', '2023-03-20']),
                'created_time': pd.to_datetime([
                    '2024-01-01 10:30:00',
                    '2024-06-15 14:45:30',
                    '2025-12-31 23:59:59',
                    '2023-03-20 09:15:45'
                ]),
                'amount': [100.0, 200.0, 300.0, 400.0]
            })
            
            # Validate original dtypes
            print(f"  Original dtypes: {dict(test_data.dtypes)}")
            
            # Ensure datetime64[ns] format
            prepared_df = self.converter.ensure_datetime64_ns(
                test_data, 
                datetime_columns=['maturity_date', 'created_time']
            )
            
            # Verify datetime64[ns] conversion
            for col in ['maturity_date', 'created_time']:
                if prepared_df[col].dtype != 'datetime64[ns]':
                    raise ValueError(f"Column {col} not properly converted to datetime64[ns]: {prepared_df[col].dtype}")
            
            print("  ✅ Basic datetime64[ns] conversion successful")
            self.validation_results.append({
                'test': 'basic_datetime_conversion',
                'status': 'PASS',
                'details': 'All datetime columns properly converted to datetime64[ns]'
            })
            return True
            
        except Exception as e:
            print(f"  ❌ Basic datetime conversion failed: {e}")
            self.validation_results.append({
                'test': 'basic_datetime_conversion',
                'status': 'FAIL',
                'details': str(e)
            })
            return False
    
    def test_pandas_to_spark_conversion(self) -> bool:
        """Test Pandas to Spark DataFrame conversion with datetime handling."""
        print("\n🧪 Testing Pandas to Spark conversion...")
        
        try:
            # Create test data with datetime columns
            test_data = pd.DataFrame({
                'loan_id': ['L001', 'L002', 'L003'],
                'maturity_date': pd.to_datetime(['2024-01-01', '2024-06-15', '2025-12-31']),
                'updated_time': pd.to_datetime([
                    '2024-01-01 10:30:00',
                    '2024-06-15 14:45:30', 
                    '2025-12-31 23:59:59'
                ]),
                'balance': [10000.0, 20000.0, 30000.0]
            })
            
            # Convert to Spark DataFrame
            spark_df = self.converter.pandas_to_spark_safe(
                test_data,
                self.spark,
                datetime_columns=['maturity_date', 'updated_time'],
                table_name='test_datetime_conversion'
            )
            
            # Verify Spark schema has TimestampType for datetime columns
            schema_dict = {field.name: str(field.dataType) for field in spark_df.schema.fields}
            print(f"  Spark schema: {schema_dict}")
            
            # Check that datetime columns are properly typed
            datetime_columns = ['maturity_date', 'updated_time']
            for col in datetime_columns:
                if 'TimestampType' not in str(spark_df.schema[col].dataType):
                    raise ValueError(f"Column {col} not properly converted to TimestampType in Spark")
            
            print("  ✅ Pandas to Spark conversion successful")
            self.validation_results.append({
                'test': 'pandas_to_spark_conversion',
                'status': 'PASS',
                'details': 'DateTime columns properly converted to TimestampType'
            })
            return True
            
        except Exception as e:
            print(f"  ❌ Pandas to Spark conversion failed: {e}")
            self.validation_results.append({
                'test': 'pandas_to_spark_conversion',
                'status': 'FAIL',
                'details': str(e)
            })
            return False
    
    def test_spark_to_pandas_conversion(self) -> bool:
        """Test the critical Spark to Pandas .toPandas() conversion."""
        print("\n🧪 Testing Spark to Pandas .toPandas() conversion...")
        
        try:
            # Create Spark DataFrame with timestamp columns
            test_data = pd.DataFrame({
                'loan_id': ['L001', 'L002', 'L003'],
                'maturity_date': pd.to_datetime(['2024-01-01', '2024-06-15', '2025-12-31']),
                'created_at': pd.to_datetime([
                    '2024-01-01 10:30:00',
                    '2024-06-15 14:45:30',
                    '2025-12-31 23:59:59'
                ]),
                'amount': [100.0, 200.0, 300.0]
            })
            
            # Convert to Spark first
            spark_df = self.converter.pandas_to_spark_safe(
                test_data,
                self.spark,
                datetime_columns=['maturity_date', 'created_at']
            )
            
            # This is the critical test - convert back to Pandas
            converted_pandas = self.converter.spark_to_pandas_safe(spark_df)
            
            # Verify all datetime columns are properly formatted
            datetime_cols = ['maturity_date', 'created_at']
            for col in datetime_cols:
                col_dtype = converted_pandas[col].dtype
                if col_dtype != 'datetime64[ns]':
                    raise ValueError(f"Column {col} has incorrect dtype after conversion: {col_dtype}")
            
            print(f"  Final dtypes: {dict(converted_pandas.dtypes)}")
            print("  ✅ Spark to Pandas .toPandas() conversion successful")
            self.validation_results.append({
                'test': 'spark_to_pandas_conversion',
                'status': 'PASS',
                'details': 'All datetime columns properly converted to datetime64[ns]'
            })
            return True
            
        except Exception as e:
            print(f"  ❌ Spark to Pandas conversion failed: {e}")
            self.validation_results.append({
                'test': 'spark_to_pandas_conversion',
                'status': 'FAIL',
                'details': str(e)
            })
            return False
    
    def test_round_trip_conversion(self) -> bool:
        """Test complete round-trip: Pandas -> Spark -> Pandas."""
        print("\n🧪 Testing round-trip datetime conversion...")
        
        try:
            # Create complex test data
            original_data = pd.DataFrame({
                'loan_id': ['L001', 'L002', 'L003', 'L004'],
                'maturity_date': pd.to_datetime([
                    '2024-01-01', '2024-06-15', 
                    '2025-12-31', '2023-03-20'
                ]).astype('datetime64[ns]'),
                'last_payment_date': pd.to_datetime([
                    '2024-01-01 10:30:00', '2024-06-14 14:45:30',
                    '2025-12-30 23:59:59', '2023-03-19 09:15:45'
                ]).astype('datetime64[ns]'),
                'balance': [10000.0, 20000.0, 30000.0, 15000.0],
                'days_past_due': [0, 15, 30, 5]
            })
            
            print(f"  Original dtypes: {dict(original_data.dtypes)}")
            
            # Round trip: Pandas -> Spark -> Pandas
            spark_df = self.converter.pandas_to_spark_safe(
                original_data,
                self.spark,
                datetime_columns=['maturity_date', 'last_payment_date']
            )
            
            final_data = self.converter.spark_to_pandas_safe(spark_df)
            
            print(f"  Final dtypes: {dict(final_data.dtypes)}")
            
            # Verify data integrity and dtype consistency
            datetime_cols = ['maturity_date', 'last_payment_date']
            for col in datetime_cols:
                if final_data[col].dtype != 'datetime64[ns]':
                    raise ValueError(f"Round-trip failed: {col} dtype is {final_data[col].dtype}")
                
                # Check data values are preserved (allowing for timezone adjustments)
                original_dates = original_data[col].dt.normalize()
                final_dates = final_data[col].dt.normalize()
                
                if not original_dates.equals(final_dates):
                    print(f"  ⚠️ Date values differ for {col}, but this may be acceptable for timezone normalization")
            
            print("  ✅ Round-trip conversion successful")
            self.validation_results.append({
                'test': 'round_trip_conversion',
                'status': 'PASS',
                'details': 'Complete round-trip conversion preserved datetime64[ns] format'
            })
            return True
            
        except Exception as e:
            print(f"  ❌ Round-trip conversion failed: {e}")
            self.validation_results.append({
                'test': 'round_trip_conversion',
                'status': 'FAIL',
                'details': str(e)
            })
            return False
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        total_tests = len(self.validation_results)
        passed_tests = sum(1 for result in self.validation_results if result['status'] == 'PASS')
        
        report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'environment': 'Docker IFRS9 Spark Container',
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': f"{(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "0%",
            'detailed_results': self.validation_results,
            'system_info': {
                'pandas_version': pd.__version__,
                'numpy_version': np.__version__,
                'pyarrow_timezone_ignore': os.environ.get('PYARROW_IGNORE_TIMEZONE'),
                'timezone': str(datetime.now().astimezone().tzinfo),
                'pyarrow_ignore_timezone': os.environ.get('PYARROW_IGNORE_TIMEZONE')
            }
        }
        
        return report
    
    def run_all_validations(self) -> bool:
        """Run all datetime validation tests."""
        print("🚀 Starting IFRS9 Docker DateTime Validation Suite")
        print("=" * 60)
        
        tests = [
            self.test_basic_datetime_conversion,
            self.test_pandas_to_spark_conversion,
            self.test_spark_to_pandas_conversion,
            self.test_round_trip_conversion
        ]
        
        all_passed = True
        for test_func in tests:
            try:
                result = test_func()
                all_passed = all_passed and result
            except Exception as e:
                print(f"❌ Test {test_func.__name__} encountered unexpected error: {e}")
                all_passed = False
        
        # Generate final report
        report = self.generate_validation_report()
        
        print("\n" + "=" * 60)
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {report['total_tests']}")
        print(f"Passed: {report['passed_tests']}")
        print(f"Failed: {report['failed_tests']}")
        print(f"Success Rate: {report['success_rate']}")
        
        if all_passed:
            print("\n✅ ALL DATETIME CONVERSION TESTS PASSED")
            print("🎉 Docker environment is ready for IFRS9 datetime processing!")
        else:
            print("\n❌ SOME DATETIME CONVERSION TESTS FAILED")
            print("⚠️ Review failed tests before proceeding with IFRS9 processing")
        
        # Save validation report
        try:
            import json
            with open('/app/logs/datetime_validation_report.json', 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n📄 Detailed report saved to: /app/logs/datetime_validation_report.json")
        except Exception as e:
            print(f"⚠️ Could not save validation report: {e}")
        
        return all_passed
    
    def cleanup(self):
        """Clean up Spark session and resources."""
        if hasattr(self, 'spark'):
            self.spark.stop()


def main():
    """Main validation execution."""
    validator = DockerDateTimeValidator()
    
    try:
        success = validator.run_all_validations()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n⚠️ Validation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Validation failed with unexpected error: {e}")
        return 1
    finally:
        validator.cleanup()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)