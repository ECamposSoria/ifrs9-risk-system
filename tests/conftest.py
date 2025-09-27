"""Pytest configuration and fixtures for IFRS9 system tests."""

import os
import sys
import warnings

import pytest
import pandas as pd
from datetime import datetime, timedelta
from typing import Generator, Dict, Any

from sklearn.exceptions import UndefinedMetricWarning
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

# Import datetime converter from validation directory
from validation.datetime_converter import DateTimeConverter


warnings.filterwarnings(
    "ignore",
    message="Precision and F-score are ill-defined.*",
    category=UndefinedMetricWarning,
)


@pytest.fixture(scope="session")
def spark_config() -> Dict[str, str]:
    """Provide optimized Spark configuration for testing."""
    return {
        "spark.app.name": "IFRS9TestSuite",
        "spark.master": "local[2]",  # Use 2 cores for better performance
        "spark.sql.shuffle.partitions": "4",  # Reduced for test data
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.sql.warehouse.dir": "/tmp/spark-warehouse",
        "spark.sql.execution.arrow.pyspark.enabled": "false",
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        "spark.sql.adaptive.localShuffleReader.enabled": "true"
    }


@pytest.fixture(scope="function")
def spark_session(spark_config) -> Generator[SparkSession, None, None]:
    """Create a fresh Spark session for each test function.
    
    This ensures complete test isolation by providing a clean Spark context
    for each test, preventing interference between tests.
    """
    # Build Spark session with optimized configuration
    builder = SparkSession.builder
    for key, value in spark_config.items():
        builder = builder.config(key, value)
    
    spark = builder.getOrCreate()
    
    # Set log level to reduce noise during testing
    spark.sparkContext.setLogLevel("WARN")
    
    yield spark
    
    # Cleanup: drop all temporary views and stop the session
    try:
        # Drop all temporary views created during the test
        for table_name in spark.catalog.listTables():
            if table_name.isTemporary:
                spark.catalog.dropTempView(table_name.name)
    except Exception as e:
        # Log warning but don't fail the test
        print(f"Warning: Could not clean up temporary views: {e}")
    
    # Stop the Spark session to ensure clean state
    spark.stop()


@pytest.fixture(scope="function")
def datetime_converter() -> DateTimeConverter:
    """Provide datetime converter instance for safe Pandas/PySpark conversions."""
    return DateTimeConverter()


@pytest.fixture(scope="function")
def sample_loan_data() -> pd.DataFrame:
    """Generate comprehensive sample loan data for testing.
    
    Returns a realistic loan portfolio with various scenarios including:
    - Different loan types and stages
    - Various DPD scenarios
    - Edge cases for testing boundary conditions
    """
    # Base test data with realistic scenarios
    loans_data = {
        "loan_id": [
            "L000001", "L000002", "L000003", "L000004", "L000005",
            "L000006", "L000007", "L000008", "L000009", "L000010"
        ],
        "customer_id": [
            "C000001", "C000002", "C000003", "C000004", "C000005",
            "C000006", "C000007", "C000008", "C000009", "C000010"
        ],
        "loan_amount": [
            100000.0, 50000.0, 25000.0, 150000.0, 75000.0,
            300000.0, 15000.0, 200000.0, 80000.0, 120000.0
        ],
        "current_balance": [
            80000.0, 45000.0, 24000.0, 140000.0, 70000.0,
            280000.0, 14000.0, 180000.0, 78000.0, 110000.0
        ],
        "days_past_due": [
            0, 35, 95, 15, 45,  # Stage 1, 2, 3, 1, 2
            120, 5, 60, 0, 30   # Stage 3, 1, 2, 1, 2 (boundary)
        ],
        "credit_score": [
            750, 650, 500, 700, 600,  # Good, Fair, Poor, Good, Fair
            480, 800, 550, 720, 680   # Poor, Excellent, Poor, Good, Fair
        ],
        "loan_type": [
            "MORTGAGE", "AUTO", "PERSONAL", "MORTGAGE", "AUTO",
            "MORTGAGE", "PERSONAL", "MORTGAGE", "AUTO", "PERSONAL"
        ],
        "collateral_value": [
            120000.0, 55000.0, 0.0, 180000.0, 85000.0,
            350000.0, 0.0, 220000.0, 90000.0, 0.0
        ],
        "ltv_ratio": [
            0.83, 0.91, None, 0.83, 0.88,
            0.86, None, 0.91, 0.89, None
        ],
        "interest_rate": [
            4.5, 6.2, 12.5, 3.8, 5.9,
            4.1, 15.2, 4.3, 6.8, 11.9
        ],
        "term_months": [
            360, 60, 36, 360, 72,
            360, 24, 360, 60, 48
        ]
    }
    
    # Generate maturity dates based on loan terms
    maturity_dates = []
    base_date = datetime(2024, 1, 1)  # Fixed base date for reproducible tests
    
    for i, term in enumerate(loans_data["term_months"]):
        maturity_date = base_date + timedelta(days=term * 30)  # Approximate months to days
        maturity_dates.append(maturity_date.strftime("%Y-%m-%d"))
    
    loans_data["maturity_date"] = maturity_dates
    
    # Add some additional derived fields for testing
    loans_data["provision_stage"] = [
        "STAGE_1", "STAGE_2", "STAGE_3", "STAGE_1", "STAGE_2",
        "STAGE_3", "STAGE_1", "STAGE_2", "STAGE_1", "STAGE_2"
    ]
    
    return pd.DataFrame(loans_data)


@pytest.fixture(scope="function")
def sample_spark_df(spark_session: SparkSession, sample_loan_data: pd.DataFrame, 
                   datetime_converter: DateTimeConverter) -> DataFrame:
    """Convert sample loan data to Spark DataFrame with proper datetime handling."""
    # Use datetime converter for safe conversion
    return datetime_converter.pandas_to_spark_safe(
        sample_loan_data, 
        spark_session, 
        datetime_columns=["maturity_date"],
        table_name="test_loans"
    )


@pytest.fixture(scope="function")
def ifrs9_rules_engine(spark_session: SparkSession):
    """Provide IFRS9RulesEngine instance with test-specific configuration."""
    from src.rules_engine import IFRS9RulesEngine
    return IFRS9RulesEngine(spark=spark_session)


@pytest.fixture(scope="function")
def validation_test_data() -> pd.DataFrame:
    """Generate data specifically for validation testing."""
    return pd.DataFrame({
        "loan_id": ["L000001", "L000002", "INVALID_ID", "L000004"],
        "customer_id": ["C000001", "C000002", "C000003", "C000004"],
        "loan_amount": [100000.0, 50000.0, -25000.0, 150000.0],  # Negative amount for testing
        "interest_rate": [5.5, 6.2, 12.5, 25.0],  # High rate for testing
        "term_months": [360, 60, 36, 0],  # Zero term for testing
        "loan_type": ["MORTGAGE", "AUTO", "INVALID_TYPE", "MORTGAGE"],  # Invalid type
        "credit_score": [750, 650, 1000, 700],  # Out of range score
        "days_past_due": [0, 35, -5, 15],  # Negative DPD for testing
        "current_balance": [95000.0, 45000.0, 24000.0, 140000.0],
        "provision_stage": ["STAGE_1", "STAGE_2", "INVALID_STAGE", "STAGE_1"],
        "pd_rate": [0.02, 0.05, 1.5, 0.03],  # Out of range PD
        "lgd_rate": [0.45, 0.55, -0.1, 0.40],  # Negative LGD for testing
    })


@pytest.fixture(scope="function")
def ml_test_data() -> pd.DataFrame:
    """Generate larger dataset for ML model testing."""
    import numpy as np
    
    np.random.seed(42)  # For reproducible results
    n_loans = 1000
    
    return pd.DataFrame({
        "loan_id": [f"L{i:06d}" for i in range(1, n_loans + 1)],
        "customer_id": [f"C{i:06d}" for i in range(1, n_loans + 1)],
        "loan_amount": np.random.lognormal(10, 1, n_loans),
        "current_balance": np.random.lognormal(9.5, 1, n_loans),
        "days_past_due": np.random.poisson(10, n_loans),
        "credit_score": np.random.normal(650, 100, n_loans).clip(300, 850),
        "loan_type": np.random.choice(["MORTGAGE", "AUTO", "PERSONAL"], n_loans),
        "collateral_value": np.random.lognormal(10.5, 1.2, n_loans),
        "ltv_ratio": np.random.beta(2, 2, n_loans),
        "interest_rate": np.random.gamma(2, 2, n_loans),
        "term_months": np.random.choice([24, 36, 48, 60, 72, 240, 360], n_loans),
        "provision_stage": np.random.choice(["STAGE_1", "STAGE_2", "STAGE_3"], n_loans, p=[0.7, 0.2, 0.1])
    })


class TestDataManager:
    """Utility class for managing test data and cleanup."""
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self._temp_views = set()
        self._temp_tables = set()
    
    def register_temp_view(self, df: DataFrame, view_name: str) -> DataFrame:
        """Register a temporary view and track it for cleanup."""
        df.createOrReplaceTempView(view_name)
        self._temp_views.add(view_name)
        return df
    
    def cleanup(self):
        """Clean up all registered temporary views and tables."""
        for view_name in self._temp_views:
            try:
                self.spark.catalog.dropTempView(view_name)
            except Exception:
                pass  # View might not exist
        
        self._temp_views.clear()
        self._temp_tables.clear()


@pytest.fixture(scope="function")
def test_data_manager(spark_session: SparkSession) -> Generator[TestDataManager, None, None]:
    """Provide a test data manager for tracking and cleaning up test resources."""
    manager = TestDataManager(spark_session)
    yield manager
    manager.cleanup()


# Test execution hooks
def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (deselect with '-m \"not integration\"')"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "spark: marks tests that require Spark (deselect with '-m \"not spark\"')"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test characteristics."""
    for item in items:
        # Mark tests that use spark_session fixture as requiring Spark
        if "spark_session" in item.fixturenames:
            item.add_marker(pytest.mark.spark)
        
        # Mark tests with large datasets as slow
        if "ml_test_data" in item.fixturenames:
            item.add_marker(pytest.mark.slow)
        
        # Mark tests that test multiple components as integration tests
        if "integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)
