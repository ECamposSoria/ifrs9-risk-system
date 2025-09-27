"""Test cleanup manager for IFRS9 system testing.

This module provides utilities for managing test resources, cleanup operations,
and temporary view management in PySpark test environments.
"""

import logging
import os
import shutil
from contextlib import contextmanager
from typing import List, Set, Optional, Dict, Any
from pathlib import Path

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.utils import AnalysisException


logger = logging.getLogger(__name__)


class CleanupManager:
    """Manager for cleaning up test resources and temporary objects."""
    
    def __init__(self, spark: SparkSession, test_name: str = "unknown"):
        """Initialize the cleanup manager.
        
        Args:
            spark: SparkSession instance
            test_name: Name of the test for logging purposes
        """
        self.spark = spark
        self.test_name = test_name
        self._temp_views: Set[str] = set()
        self._temp_tables: Set[str] = set()
        self._temp_files: List[Path] = []
        self._temp_directories: List[Path] = []
        self._cached_dataframes: List[DataFrame] = []
        
        logger.debug(f"Initialized cleanup manager for test: {test_name}")
    
    def register_temp_view(self, df: DataFrame, view_name: str, replace_existing: bool = True) -> DataFrame:
        """Register a temporary view and track it for cleanup.
        
        Args:
            df: DataFrame to create view from
            view_name: Name of the temporary view
            replace_existing: Whether to replace existing view
            
        Returns:
            The original DataFrame for chaining
        """
        try:
            if replace_existing:
                df.createOrReplaceTempView(view_name)
            else:
                df.createTempView(view_name)
            
            self._temp_views.add(view_name)
            logger.debug(f"Registered temp view: {view_name} for test: {self.test_name}")
            
        except Exception as e:
            logger.warning(f"Failed to create temp view {view_name}: {e}")
            
        return df
    
    def register_temp_table(self, df: DataFrame, table_name: str) -> DataFrame:
        """Register a temporary table and track it for cleanup.
        
        Args:
            df: DataFrame to register as table
            table_name: Name of the temporary table
            
        Returns:
            The original DataFrame for chaining
        """
        try:
            df.write.mode("overwrite").saveAsTable(table_name)
            self._temp_tables.add(table_name)
            logger.debug(f"Registered temp table: {table_name} for test: {self.test_name}")
            
        except Exception as e:
            logger.warning(f"Failed to create temp table {table_name}: {e}")
            
        return df
    
    def register_temp_file(self, file_path: str) -> Path:
        """Register a temporary file for cleanup.
        
        Args:
            file_path: Path to the temporary file
            
        Returns:
            Path object for the file
        """
        path = Path(file_path)
        self._temp_files.append(path)
        logger.debug(f"Registered temp file: {file_path} for test: {self.test_name}")
        return path
    
    def register_temp_directory(self, dir_path: str) -> Path:
        """Register a temporary directory for cleanup.
        
        Args:
            dir_path: Path to the temporary directory
            
        Returns:
            Path object for the directory
        """
        path = Path(dir_path)
        self._temp_directories.append(path)
        logger.debug(f"Registered temp directory: {dir_path} for test: {self.test_name}")
        return path
    
    def cache_dataframe(self, df: DataFrame, storage_level: str = "MEMORY_ONLY") -> DataFrame:
        """Cache a DataFrame and track it for cleanup.
        
        Args:
            df: DataFrame to cache
            storage_level: Spark storage level
            
        Returns:
            The cached DataFrame
        """
        try:
            cached_df = df.cache()
            self._cached_dataframes.append(cached_df)
            logger.debug(f"Cached DataFrame for test: {self.test_name}")
            return cached_df
        except Exception as e:
            logger.warning(f"Failed to cache DataFrame: {e}")
            return df
    
    def cleanup_temp_views(self) -> None:
        """Clean up all registered temporary views."""
        for view_name in self._temp_views:
            try:
                self.spark.catalog.dropTempView(view_name)
                logger.debug(f"Dropped temp view: {view_name}")
            except AnalysisException:
                # View might not exist, which is fine
                logger.debug(f"Temp view {view_name} was already dropped or didn't exist")
            except Exception as e:
                logger.warning(f"Failed to drop temp view {view_name}: {e}")
        
        self._temp_views.clear()
    
    def cleanup_temp_tables(self) -> None:
        """Clean up all registered temporary tables."""
        for table_name in self._temp_tables:
            try:
                self.spark.sql(f"DROP TABLE IF EXISTS {table_name}")
                logger.debug(f"Dropped temp table: {table_name}")
            except Exception as e:
                logger.warning(f"Failed to drop temp table {table_name}: {e}")
        
        self._temp_tables.clear()
    
    def cleanup_temp_files(self) -> None:
        """Clean up all registered temporary files."""
        for file_path in self._temp_files:
            try:
                if file_path.exists():
                    file_path.unlink()
                    logger.debug(f"Removed temp file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temp file {file_path}: {e}")
        
        self._temp_files.clear()
    
    def cleanup_temp_directories(self) -> None:
        """Clean up all registered temporary directories."""
        for dir_path in self._temp_directories:
            try:
                if dir_path.exists():
                    shutil.rmtree(dir_path)
                    logger.debug(f"Removed temp directory: {dir_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temp directory {dir_path}: {e}")
        
        self._temp_directories.clear()
    
    def cleanup_cached_dataframes(self) -> None:
        """Clean up all cached DataFrames."""
        for df in self._cached_dataframes:
            try:
                df.unpersist()
                logger.debug(f"Unpersisted cached DataFrame")
            except Exception as e:
                logger.warning(f"Failed to unpersist DataFrame: {e}")
        
        self._cached_dataframes.clear()
    
    def cleanup_all(self) -> None:
        """Clean up all registered resources."""
        logger.debug(f"Starting cleanup for test: {self.test_name}")
        
        # Clean up in reverse order of typical creation
        self.cleanup_cached_dataframes()
        self.cleanup_temp_views()
        self.cleanup_temp_tables()
        self.cleanup_temp_files()
        self.cleanup_temp_directories()
        
        logger.debug(f"Completed cleanup for test: {self.test_name}")
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get a summary of currently tracked resources.
        
        Returns:
            Dictionary with resource counts and details
        """
        return {
            "test_name": self.test_name,
            "temp_views_count": len(self._temp_views),
            "temp_views": list(self._temp_views),
            "temp_tables_count": len(self._temp_tables),
            "temp_tables": list(self._temp_tables),
            "temp_files_count": len(self._temp_files),
            "temp_files": [str(p) for p in self._temp_files],
            "temp_directories_count": len(self._temp_directories),
            "temp_directories": [str(p) for p in self._temp_directories],
            "cached_dataframes_count": len(self._cached_dataframes)
        }
    
    @contextmanager
    def managed_temp_view(self, df: DataFrame, view_name: str):
        """Context manager for temporary views with automatic cleanup.
        
        Args:
            df: DataFrame to create view from
            view_name: Name of the temporary view
            
        Yields:
            The view name for use in SQL queries
        """
        try:
            self.register_temp_view(df, view_name)
            yield view_name
        finally:
            try:
                self.spark.catalog.dropTempView(view_name)
                self._temp_views.discard(view_name)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp view {view_name}: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        self.cleanup_all()


class SparkSessionManager:
    """Manager for Spark session lifecycle in tests."""
    
    @staticmethod
    def create_test_session(app_name: str, 
                          config: Optional[Dict[str, str]] = None,
                          master: str = "local[2]") -> SparkSession:
        """Create an optimized Spark session for testing.
        
        Args:
            app_name: Application name for the Spark session
            config: Additional Spark configuration
            master: Spark master URL
            
        Returns:
            Configured SparkSession
        """
        builder = SparkSession.builder.appName(app_name).master(master)
        
        # Default test optimizations
        default_config = {
            "spark.sql.shuffle.partitions": "4",
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.adaptive.coalescePartitions.enabled": "true",
            "spark.sql.warehouse.dir": "/tmp/spark-warehouse",
            "spark.sql.execution.arrow.pyspark.enabled": "true",
            "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
            "spark.sql.adaptive.localShuffleReader.enabled": "true",
            "spark.sql.adaptive.skewJoin.enabled": "true",
            "spark.sql.adaptive.broadcastTimeout": "300s"
        }
        
        # Merge with provided config
        if config:
            default_config.update(config)
        
        # Apply configuration
        for key, value in default_config.items():
            builder = builder.config(key, value)
        
        spark = builder.getOrCreate()
        
        # Set log level to reduce noise
        spark.sparkContext.setLogLevel("WARN")
        
        return spark
    
    @staticmethod
    @contextmanager
    def managed_session(app_name: str, 
                       config: Optional[Dict[str, str]] = None,
                       master: str = "local[2]"):
        """Context manager for Spark sessions with automatic cleanup.
        
        Args:
            app_name: Application name for the Spark session
            config: Additional Spark configuration
            master: Spark master URL
            
        Yields:
            Configured SparkSession
        """
        spark = None
        try:
            spark = SparkSessionManager.create_test_session(app_name, config, master)
            yield spark
        finally:
            if spark:
                try:
                    spark.stop()
                except Exception as e:
                    logger.warning(f"Failed to stop Spark session: {e}")


def create_temp_directory(base_path: str = "/tmp", prefix: str = "ifrs9_test") -> Path:
    """Create a temporary directory for test data.
    
    Args:
        base_path: Base path for temporary directory
        prefix: Prefix for directory name
        
    Returns:
        Path to the created directory
    """
    import tempfile
    temp_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=base_path))
    return temp_dir


def verify_spark_session_health(spark: SparkSession) -> Dict[str, Any]:
    """Verify that the Spark session is healthy and responsive.
    
    Args:
        spark: SparkSession to check
        
    Returns:
        Dictionary with health check results
    """
    health_check = {
        "session_active": False,
        "can_create_df": False,
        "can_run_sql": False,
        "catalog_accessible": False,
        "errors": []
    }
    
    try:
        # Check if session is active
        if spark.sparkContext and not spark.sparkContext._jsc.sc().isStopped():
            health_check["session_active"] = True
        
        # Test DataFrame creation
        test_df = spark.range(1)
        if test_df.count() == 1:
            health_check["can_create_df"] = True
        
        # Test SQL execution
        result = spark.sql("SELECT 1 as test_col").collect()
        if len(result) == 1 and result[0]["test_col"] == 1:
            health_check["can_run_sql"] = True
        
        # Test catalog access
        tables = spark.catalog.listTables()
        health_check["catalog_accessible"] = True
        
    except Exception as e:
        health_check["errors"].append(str(e))
    
    return health_check
