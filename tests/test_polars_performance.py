"""
Performance benchmark tests for Polars integration in IFRS9 system.

Tests performance comparisons between Polars and pandas operations,
memory usage optimization, and ML pipeline efficiency.
"""

import pytest
import time
import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime
import psutil
import os
import sys
from typing import Dict, Any

# Import the Polars integration module
sys.path.append('/app/src')
try:
    from polars_ml_integration import PolarsEnhancedCreditRiskClassifier, create_synthetic_ifrs9_data_polars
    POLARS_INTEGRATION_AVAILABLE = True
except ImportError as e:
    POLARS_INTEGRATION_AVAILABLE = False
    print(f"Polars integration not available: {e}")


@pytest.mark.skipif(not POLARS_INTEGRATION_AVAILABLE, reason="Polars integration not available")
class TestPolarsPerformanceBenchmarks:
    """Performance benchmark tests for Polars operations."""

    @pytest.fixture
    def small_dataset(self):
        """Small dataset for quick tests."""
        return create_synthetic_ifrs9_data_polars(10000)

    @pytest.fixture
    def medium_dataset(self):
        """Medium dataset for performance testing."""
        return create_synthetic_ifrs9_data_polars(100000)

    @pytest.fixture
    def large_dataset(self):
        """Large dataset for stress testing."""
        return create_synthetic_ifrs9_data_polars(500000)

    def measure_memory_usage(self, func, *args, **kwargs):
        """Measure memory usage of a function."""
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        result = func(*args, **kwargs)
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        return result, mem_after - mem_before

    def test_dataframe_creation_performance(self):
        """Test DataFrame creation performance."""
        n_rows = 100000
        data = {
            'id': [f'ID{i:06d}' for i in range(n_rows)],
            'amount': np.random.lognormal(10, 1, n_rows),
            'rate': np.random.uniform(0.02, 0.12, n_rows),
            'score': np.random.normal(700, 100, n_rows),
            'date': [datetime(2024, 1, 1) for _ in range(n_rows)]
        }

        # Benchmark Polars DataFrame creation
        start_time = time.time()
        pl_df = pl.DataFrame(data)
        polars_creation_time = time.time() - start_time

        # Benchmark pandas DataFrame creation
        start_time = time.time()
        pd_df = pd.DataFrame(data)
        pandas_creation_time = time.time() - start_time

        print(f"\nDataFrame Creation Benchmark ({n_rows:,} rows):")
        print(f"Polars: {polars_creation_time:.4f}s")
        print(f"Pandas: {pandas_creation_time:.4f}s")
        print(f"Speedup: {pandas_creation_time/polars_creation_time:.2f}x")

        # Verify same shape
        assert pl_df.shape == pd_df.shape

        # Both should complete in reasonable time
        assert polars_creation_time < 5.0
        assert pandas_creation_time < 5.0


if __name__ == "__main__":
    # Run performance benchmarks when called directly
    pytest.main([__file__, "-v", "-s"])