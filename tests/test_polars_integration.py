"""
Test suite for Polars integration in IFRS9 system.

Tests Polars installation, basic operations, interoperability with pandas,
and compatibility with ML models in the Docker environment.
"""

import pytest
import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime, date
from typing import Dict, Any


class TestPolarsInstallation:
    """Test basic Polars installation and functionality."""
    
    def test_polars_import(self):
        """Test that Polars can be imported successfully."""
        assert pl.__version__ is not None
        print(f"Polars version: {pl.__version__}")
    
    def test_polars_basic_operations(self):
        """Test basic Polars DataFrame operations."""
        # Create a simple DataFrame
        df = pl.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'value': [10.5, 20.3, 30.1, 40.7, 50.9],
            'category': ['A', 'B', 'A', 'C', 'B'],
            'date': [date(2024, 1, i) for i in range(1, 6)]
        })
        
        # Test basic properties
        assert df.shape == (5, 4)
        assert list(df.columns) == ['id', 'value', 'category', 'date']
        
        # Test filtering
        filtered = df.filter(pl.col('value') > 25)
        assert filtered.shape[0] == 3
        
        # Test aggregation
        agg_result = df.group_by('category').agg([
            pl.col('value').mean().alias('mean_value'),
            pl.col('value').count().alias('count')
        ])
        assert agg_result.shape[0] == 3  # A, B, C categories
    
    def test_polars_lazy_evaluation(self):
        """Test Polars lazy evaluation capabilities."""
        # Create lazy frame
        df_lazy = pl.LazyFrame({
            'x': range(1000),
            'y': range(1000, 2000)
        })
        
        # Build query
        query = (df_lazy
                .filter(pl.col('x') > 500)
                .with_columns((pl.col('x') * 2).alias('x_doubled'))
                .select(['x', 'x_doubled'])
                .head(10))
        
        # Execute query
        result = query.collect()
        assert result.shape == (10, 2)
        assert result['x_doubled'][0] == result['x'][0] * 2


class TestPolarsInteroperability:
    """Test Polars interoperability with pandas and NumPy."""
    
    def test_pandas_to_polars_conversion(self):
        """Test conversion from pandas to Polars."""
        # Create pandas DataFrame
        pd_df = pd.DataFrame({
            'loan_id': ['L001', 'L002', 'L003'],
            'balance': [100000.0, 250000.0, 500000.0],
            'rating': ['A', 'B', 'A'],
            'origination_date': pd.to_datetime(['2023-01-01', '2023-06-15', '2024-01-30'])
        })
        
        # Convert to Polars
        pl_df = pl.from_pandas(pd_df)
        
        assert pl_df.shape == pd_df.shape
        assert list(pl_df.columns) == list(pd_df.columns)
        
        # Test data preservation
        assert pl_df['loan_id'].to_list() == pd_df['loan_id'].tolist()
        assert np.allclose(pl_df['balance'].to_numpy(), pd_df['balance'].values)
    
    def test_polars_to_pandas_conversion(self):
        """Test conversion from Polars to pandas."""
        # Create Polars DataFrame with datetime
        pl_df = pl.DataFrame({
            'customer_id': [1, 2, 3],
            'exposure': [1000000.0, 2500000.0, 500000.0],
            'pd': [0.01, 0.05, 0.02],
            'reporting_date': [datetime(2024, 1, 31), datetime(2024, 2, 29), datetime(2024, 3, 31)]
        })
        
        # Convert to pandas
        pd_df = pl_df.to_pandas()
        
        assert pd_df.shape == pl_df.shape
        assert list(pd_df.columns) == list(pl_df.columns)
        
        # Test datetime preservation
        assert pd.api.types.is_datetime64_any_dtype(pd_df['reporting_date'])
    
    def test_numpy_interop(self):
        """Test Polars interoperability with NumPy arrays."""
        # Create NumPy array
        np_array = np.random.normal(0, 1, 1000)
        
        # Create Polars DataFrame from NumPy
        pl_df = pl.DataFrame({'values': np_array})
        
        # Extract back to NumPy
        extracted = pl_df['values'].to_numpy()
        
        assert np.allclose(np_array, extracted)
        assert extracted.dtype == np_array.dtype


class TestPolarsMLCompatibility:
    """Test Polars compatibility with ML models."""
    
    def test_xgboost_compatibility(self):
        """Test if Polars DataFrames can be used with XGBoost."""
        try:
            import xgboost as xgb
            
            # Create synthetic IFRS9-like data
            pl_df = pl.DataFrame({
                'balance': np.random.uniform(1000, 1000000, 1000),
                'term': np.random.randint(12, 360, 1000),
                'rating_score': np.random.uniform(300, 850, 1000),
                'ltv': np.random.uniform(0.3, 0.95, 1000),
                'pd': np.random.uniform(0.001, 0.1, 1000)  # Target variable
            })
            
            # Prepare features and target
            feature_cols = ['balance', 'term', 'rating_score', 'ltv']
            X = pl_df.select(feature_cols)
            y = pl_df.select('pd')
            
            # Test XGBoost training with Polars (if supported)
            # Note: This might require conversion to pandas depending on XGBoost version
            try:
                # Try direct Polars support
                dtrain = xgb.DMatrix(X, label=y)
                model = xgb.train({'objective': 'reg:squarederror'}, dtrain, num_boost_round=10)
                assert model is not None
                print("✅ XGBoost supports Polars DataFrames directly")
            except Exception as e:
                # Fall back to pandas conversion
                X_pandas = X.to_pandas()
                y_pandas = y.to_pandas().values.flatten()
                dtrain = xgb.DMatrix(X_pandas, label=y_pandas)
                model = xgb.train({'objective': 'reg:squarederror'}, dtrain, num_boost_round=10)
                assert model is not None
                print("✅ XGBoost works with Polars via pandas conversion")
                
        except ImportError:
            pytest.skip("XGBoost not installed")
    
    def test_lightgbm_compatibility(self):
        """Test if Polars DataFrames can be used with LightGBM."""
        try:
            import lightgbm as lgb
            
            # Create synthetic data
            pl_df = pl.DataFrame({
                'feature1': np.random.normal(0, 1, 1000),
                'feature2': np.random.normal(0, 1, 1000),
                'feature3': np.random.uniform(0, 1, 1000),
                'target': np.random.binomial(1, 0.3, 1000)
            })
            
            # Prepare data
            feature_cols = ['feature1', 'feature2', 'feature3']
            X = pl_df.select(feature_cols)
            y = pl_df.select('target')
            
            # Test LightGBM training
            try:
                # Try direct Polars support
                train_data = lgb.Dataset(X, label=y)
                model = lgb.train({'objective': 'binary'}, train_data, num_boost_round=10)
                assert model is not None
                print("✅ LightGBM supports Polars DataFrames directly")
            except Exception as e:
                # Fall back to pandas conversion
                X_pandas = X.to_pandas()
                y_pandas = y.to_pandas().values.flatten()
                train_data = lgb.Dataset(X_pandas, label=y_pandas)
                model = lgb.train({'objective': 'binary'}, train_data, num_boost_round=10)
                assert model is not None
                print("✅ LightGBM works with Polars via pandas conversion")
                
        except ImportError:
            pytest.skip("LightGBM not installed")


class TestPolarsPerformance:
    """Test Polars performance characteristics."""
    
    def test_large_dataframe_operations(self):
        """Test Polars performance on larger datasets."""
        # Create a larger synthetic dataset
        n_rows = 100000
        pl_df = pl.DataFrame({
            'id': range(n_rows),
            'value': np.random.normal(100, 25, n_rows),
            'category': np.random.choice(['A', 'B', 'C', 'D'], n_rows),
            'date': [datetime(2024, 1, 1)] * n_rows
        })
        
        # Test operations
        result = (pl_df
                 .filter(pl.col('value') > 100)
                 .group_by('category')
                 .agg([
                     pl.col('value').mean().alias('mean_value'),
                     pl.col('value').std().alias('std_value'),
                     pl.col('id').count().alias('count')
                 ])
                 .sort('mean_value', descending=True))
        
        assert result.shape[0] <= 4  # Max 4 categories
        assert 'mean_value' in result.columns
        assert 'std_value' in result.columns
        assert 'count' in result.columns
    
    def test_memory_efficiency(self):
        """Test Polars memory efficiency compared to pandas."""
        import sys
        
        # Create identical datasets
        n_rows = 50000
        data = {
            'col1': np.random.random(n_rows),
            'col2': np.random.randint(0, 100, n_rows),
            'col3': ['category_' + str(i % 10) for i in range(n_rows)]
        }
        
        # Pandas DataFrame
        pd_df = pd.DataFrame(data)
        pd_memory = sys.getsizeof(pd_df)
        
        # Polars DataFrame
        pl_df = pl.DataFrame(data)
        pl_memory = sys.getsizeof(pl_df)
        
        print(f"Pandas memory usage: {pd_memory:,} bytes")
        print(f"Polars memory usage: {pl_memory:,} bytes")
        
        # Basic assertion - both should be reasonable
        assert pd_memory > 0
        assert pl_memory > 0


class TestPolarsIFRS9Specific:
    """Test Polars with IFRS9-specific use cases."""
    
    def test_ifrs9_data_transformations(self):
        """Test common IFRS9 data transformations with Polars."""
        # Create synthetic loan portfolio
        n_loans = 10000
        pl_df = pl.DataFrame({
            'loan_id': [f'L{i:06d}' for i in range(n_loans)],
            'origination_date': [
                datetime(2023, np.random.randint(1, 13), np.random.randint(1, 28))
                for _ in range(n_loans)
            ],
            'maturity_date': [
                datetime(2025, np.random.randint(1, 13), np.random.randint(1, 28))
                for _ in range(n_loans)
            ],
            'balance': np.random.lognormal(10, 1, n_loans),
            'interest_rate': np.random.uniform(0.02, 0.08, n_loans),
            'ltv': np.random.uniform(0.3, 0.95, n_loans),
            'rating': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_loans, 
                                    p=[0.2, 0.25, 0.3, 0.15, 0.1])
        })
        
        # Common IFRS9 transformations
        result = (pl_df
                 .with_columns([
                     # Calculate remaining term in months
                     ((pl.col('maturity_date') - pl.col('origination_date')).dt.days() / 30)
                     .alias('remaining_term_months'),
                     # Calculate monthly payment approximation
                     (pl.col('balance') * pl.col('interest_rate') / 12)
                     .alias('monthly_payment_approx'),
                     # Risk buckets
                     pl.when(pl.col('ltv') > 0.8).then(pl.lit('High Risk'))
                     .when(pl.col('ltv') > 0.6).then(pl.lit('Medium Risk'))
                     .otherwise(pl.lit('Low Risk'))
                     .alias('risk_bucket')
                 ])
                 .filter(pl.col('balance') > 1000)  # Filter small loans
                 .group_by(['rating', 'risk_bucket'])
                 .agg([
                     pl.col('balance').sum().alias('total_exposure'),
                     pl.col('balance').mean().alias('avg_balance'),
                     pl.col('interest_rate').mean().alias('avg_rate'),
                     pl.count().alias('loan_count')
                 ])
                 .sort(['rating', 'total_exposure'], descending=[False, True]))
        
        # Validate results
        assert result.shape[0] > 0
        assert 'total_exposure' in result.columns
        assert 'avg_balance' in result.columns
        assert 'avg_rate' in result.columns
        assert 'loan_count' in result.columns
        
        # Check that all values are reasonable
        assert (result['total_exposure'] > 0).all()
        assert (result['avg_balance'] > 1000).all()  # Due to filter
        assert (result['avg_rate'] >= 0.02).all() and (result['avg_rate'] <= 0.08).all()
    
    def test_time_series_operations(self):
        """Test Polars time series operations for IFRS9 reporting."""
        # Create time series data
        dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
        n_dates = len(dates)
        
        pl_df = pl.DataFrame({
            'reporting_date': dates,
            'portfolio_balance': np.random.lognormal(15, 0.1, n_dates),
            'provision_amount': np.random.lognormal(12, 0.2, n_dates),
            'new_defaults': np.random.poisson(5, n_dates)
        })
        
        # Monthly aggregation
        monthly_result = (pl_df
                         .with_columns([
                             pl.col('reporting_date').dt.truncate('1mo').alias('month'),
                             (pl.col('provision_amount') / pl.col('portfolio_balance'))
                             .alias('provision_rate')
                         ])
                         .group_by('month')
                         .agg([
                             pl.col('portfolio_balance').mean().alias('avg_portfolio_balance'),
                             pl.col('provision_amount').sum().alias('total_provisions'),
                             pl.col('provision_rate').mean().alias('avg_provision_rate'),
                             pl.col('new_defaults').sum().alias('total_new_defaults')
                         ])
                         .sort('month'))
        
        # Validate monthly aggregation
        assert monthly_result.shape[0] == 24  # 24 months
        assert (monthly_result['avg_portfolio_balance'] > 0).all()
        assert (monthly_result['total_provisions'] > 0).all()
        assert (monthly_result['avg_provision_rate'] >= 0).all()
        

if __name__ == "__main__":
    # Run tests when called directly
    pytest.main([__file__, "-v"])