#!/usr/bin/env python3
"""
Datetime conversion utility for IFRS9 system
Handles safe conversion between PySpark and Pandas DataFrames
"""

import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame, SparkSession
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, DoubleType, IntegerType
from typing import Union, List, Optional
import logging
import os
import warnings
import pytz
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Configure PyArrow for enhanced datetime compatibility
os.environ.setdefault('PYARROW_IGNORE_TIMEZONE', '1')

# Suppress PyArrow timezone warnings for clean conversion
warnings.filterwarnings('ignore', category=UserWarning, module='pyarrow')


class DateTimeConverter:
    """Utility class for safe datetime conversions between PySpark and Pandas."""
    
    @staticmethod
    def ensure_datetime64_ns(df: pd.DataFrame, datetime_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Ensure datetime columns are properly formatted as datetime64[ns] for PySpark compatibility.
        Enhanced with timezone handling and unit-less dtype protection.
        
        Args:
            df: Pandas DataFrame to process
            datetime_columns: Optional list of column names to convert. If None, auto-detect.
            
        Returns:
            Pandas DataFrame with properly formatted datetime columns
        """
        df_copy = df.copy()
        
        if datetime_columns is None:
            # Enhanced auto-detection with more datetime patterns
            datetime_columns = []
            datetime_keywords = ['date', 'time', 'created', 'updated', 'modified', 
                               'timestamp', 'maturity', 'effective', 'expiry', 'due']
            
            for col in df_copy.columns:
                # Check column name patterns
                if any(keyword in col.lower() for keyword in datetime_keywords):
                    datetime_columns.append(col)
                # Check existing datetime-like dtypes
                elif 'datetime' in str(df_copy[col].dtype).lower():
                    datetime_columns.append(col)
                # Check object columns for datetime-like strings
                elif df_copy[col].dtype == 'object' and not df_copy[col].dropna().empty:
                    sample_val = df_copy[col].dropna().iloc[0]
                    if sample_val and isinstance(sample_val, str):
                        try:
                            pd.to_datetime(sample_val)
                            datetime_columns.append(col)
                        except (ValueError, TypeError):
                            continue
        
        # Convert identified columns with enhanced error handling
        for col in datetime_columns:
            if col in df_copy.columns:
                try:
                    original_dtype = df_copy[col].dtype
                    
                    # Handle timezone-aware datetimes first
                    if hasattr(df_copy[col].dtype, 'tz') and df_copy[col].dtype.tz is not None:
                        # Convert to UTC then localize to None for compatibility
                        df_copy[col] = df_copy[col].dt.tz_convert('UTC').dt.tz_localize(None)
                    
                    # Convert to datetime with comprehensive error handling
                    df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce', utc=False)
                    
                    # Critical: Ensure dtype is specifically datetime64[ns] to prevent unit-less error
                    if str(df_copy[col].dtype) == 'datetime64' or 'datetime64' in str(df_copy[col].dtype):
                        # Force conversion to nanosecond precision
                        df_copy[col] = df_copy[col].astype('datetime64[ns]')
                    
                    # Validate the conversion was successful
                    if df_copy[col].dtype != 'datetime64[ns]':
                        logger.warning(f"Column '{col}' dtype is {df_copy[col].dtype}, forcing to datetime64[ns]")
                        df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce').astype('datetime64[ns]')
                    
                    logger.info(f"Successfully converted column '{col}' from {original_dtype} to datetime64[ns]")
                    
                except Exception as e:
                    logger.error(f"Failed to convert column '{col}' to datetime64[ns]: {e}")
                    # Keep original column but log the failure for debugging
                    continue
        
        return df_copy
    
    @staticmethod
    def pandas_to_spark_safe(df: pd.DataFrame, spark: SparkSession, 
                           datetime_columns: Optional[List[str]] = None,
                           table_name: str = "temp_table") -> SparkDataFrame:
        """
        Safely convert Pandas DataFrame to Spark DataFrame with proper datetime handling.
        
        Args:
            df: Pandas DataFrame to convert
            spark: Spark session
            datetime_columns: Optional list of datetime column names
            table_name: Name for temporary table registration
            
        Returns:
            Spark DataFrame with properly handled datetime columns
        """
        # Ensure proper datetime formatting
        df_prepared = DateTimeConverter.ensure_datetime64_ns(df, datetime_columns)
        
        # Create Spark DataFrame
        spark_df = spark.createDataFrame(df_prepared)
        
        # Explicitly cast datetime columns to TimestampType if needed
        if datetime_columns:
            for col in datetime_columns:
                if col in spark_df.columns:
                    spark_df = spark_df.withColumn(col, spark_df[col].cast(TimestampType()))
        
        # Register as temporary view for SQL access
        spark_df.createOrReplaceTempView(table_name)
        
        return spark_df
    
    @staticmethod
    def spark_to_pandas_safe(spark_df: SparkDataFrame) -> pd.DataFrame:
        """
        Safely convert Spark DataFrame to Pandas DataFrame with enhanced datetime preservation.
        Enhanced to handle all timezone scenarios and prevent unit-less datetime64 errors.
        
        Args:
            spark_df: Spark DataFrame to convert
            
        Returns:
            Pandas DataFrame with properly formatted datetime columns
        """
        try:
            # Use Arrow-optimized conversion with fallback
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pandas_df = spark_df.toPandas()
                
        except Exception as e:
            logger.warning(f"Arrow conversion failed, falling back to non-Arrow: {e}")
            # Disable Arrow for this conversion and retry
            spark_df.sql_ctx.setConf("spark.sql.execution.arrow.pyspark.enabled", "false")
            pandas_df = spark_df.toPandas()
            # Re-enable Arrow for future conversions
            spark_df.sql_ctx.setConf("spark.sql.execution.arrow.pyspark.enabled", "true")
        
        # Enhanced datetime column processing
        for col in pandas_df.columns:
            col_dtype = str(pandas_df[col].dtype)
            
            # Handle various timezone-aware datetime formats
            if 'datetime64[ns,' in col_dtype:
                # Timezone-aware datetime - convert to naive UTC
                try:
                    pandas_df[col] = pandas_df[col].dt.tz_convert('UTC').dt.tz_localize(None)
                    logger.debug(f"Converted timezone-aware column '{col}' to naive datetime64[ns]")
                except Exception as e:
                    logger.warning(f"Timezone conversion failed for '{col}': {e}")
                    pandas_df[col] = pandas_df[col].dt.tz_localize(None)
                    
            elif col_dtype == 'datetime64' or 'datetime64[' not in col_dtype and 'datetime' in col_dtype:
                # Unit-less or improperly formatted datetime - fix it
                try:
                    pandas_df[col] = pd.to_datetime(pandas_df[col], errors='coerce').astype('datetime64[ns]')
                    logger.debug(f"Fixed unit-less datetime column '{col}' to datetime64[ns]")
                except Exception as e:
                    logger.error(f"Failed to fix datetime column '{col}': {e}")
                    
            elif 'datetime' in col_dtype and pandas_df[col].dtype != 'datetime64[ns]':
                # Any other datetime format - standardize to datetime64[ns]
                try:
                    pandas_df[col] = pd.to_datetime(pandas_df[col], errors='coerce').astype('datetime64[ns]')
                    logger.debug(f"Standardized datetime column '{col}' to datetime64[ns]")
                except Exception as e:
                    logger.warning(f"Could not standardize datetime column '{col}': {e}")
        
        return pandas_df
    
    @staticmethod
    def validate_datetime_conversion(df: pd.DataFrame, spark: SparkSession) -> dict:
        """
        Validate that datetime conversion will work without errors.
        
        Args:
            df: Pandas DataFrame to test
            spark: Spark session
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'original_dtypes': dict(df.dtypes),
            'datetime_columns': [],
            'conversion_safe': False,
            'issues': [],
            'recommendations': []
        }
        
        # Identify datetime columns
        for col in df.columns:
            if 'datetime' in str(df[col].dtype):
                validation_results['datetime_columns'].append(col)
                
                # Check if dtype is properly specified
                if df[col].dtype == 'datetime64' or str(df[col].dtype) == 'datetime64':
                    validation_results['issues'].append(f"Column '{col}' has unit-less datetime64 dtype")
                    validation_results['recommendations'].append(f"Convert '{col}' to datetime64[ns]")
        
        # Test conversion
        try:
            test_df = DateTimeConverter.ensure_datetime64_ns(df.head(1))
            test_spark_df = spark.createDataFrame(test_df)
            validation_results['conversion_safe'] = True
        except Exception as e:
            validation_results['conversion_safe'] = False
            validation_results['issues'].append(f"Conversion test failed: {str(e)}")
        
        return validation_results


def test_datetime_conversion():
    """Test datetime conversion functionality."""
    from pyspark.sql import SparkSession
    
    # Create Spark session
    spark = SparkSession.builder.appName("DateTimeConversionTest").getOrCreate()
    
    # Create test data with various datetime formats
    test_data = pd.DataFrame({
        'id': [1, 2, 3],
        'created_date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
        'updated_time': pd.to_datetime(['2023-01-01 10:00:00', '2023-01-02 11:00:00', '2023-01-03 12:00:00']),
        'amount': [100.0, 200.0, 300.0]
    })
    
    print("Original dtypes:")
    print(test_data.dtypes)
    print()
    
    # Validate conversion
    converter = DateTimeConverter()
    validation = converter.validate_datetime_conversion(test_data, spark)
    
    print("Validation results:")
    for key, value in validation.items():
        print(f"{key}: {value}")
    print()
    
    # Test safe conversion
    try:
        spark_df = converter.pandas_to_spark_safe(test_data, spark, ['created_date', 'updated_time'])
        print("✅ Conversion successful!")
        print("Spark DataFrame schema:")
        spark_df.printSchema()
        
        # Convert back to Pandas
        converted_back = converter.spark_to_pandas_safe(spark_df)
        print("✅ Round-trip conversion successful!")
        print("Final dtypes:")
        print(converted_back.dtypes)
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
    
    spark.stop()


if __name__ == "__main__":
    test_datetime_conversion()