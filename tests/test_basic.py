"""Basic test to verify Spark container functionality."""

import unittest
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession


class TestBasicSparkFunctionality(unittest.TestCase):
    """Test basic PySpark functionality in the container."""
    
    @classmethod
    def setUpClass(cls):
        """Set up Spark session for testing."""
        cls.spark = SparkSession.builder \
            .appName("TestBasicSpark") \
            .master("local[*]") \
            .config("spark.sql.shuffle.partitions", "2") \
            .getOrCreate()
    
    @classmethod
    def tearDownClass(cls):
        """Stop Spark session."""
        cls.spark.stop()
    
    def test_spark_session_creation(self):
        """Test that Spark session can be created."""
        self.assertIsNotNone(self.spark)
        self.assertEqual(self.spark.version, "3.4.1")
    
    def test_dataframe_creation(self):
        """Test basic DataFrame operations."""
        data = [("Alice", 25), ("Bob", 30), ("Charlie", 35)]
        columns = ["name", "age"]
        
        df = self.spark.createDataFrame(data, columns)
        self.assertEqual(df.count(), 3)
        
        # Test conversion to Pandas
        pandas_df = df.toPandas()
        self.assertEqual(len(pandas_df), 3)
        self.assertIn("name", pandas_df.columns)
        self.assertIn("age", pandas_df.columns)
    
    def test_sql_operations(self):
        """Test SQL operations on DataFrames."""
        data = [("Alice", 25, "Engineering"),
                ("Bob", 30, "Marketing"),
                ("Charlie", 35, "Engineering")]
        columns = ["name", "age", "department"]
        
        df = self.spark.createDataFrame(data, columns)
        df.createOrReplaceTempView("employees")
        
        result = self.spark.sql("SELECT department, COUNT(*) as count FROM employees GROUP BY department").collect()
        
        self.assertEqual(len(result), 2)
        dept_counts = {row.department: row.count for row in result}
        self.assertEqual(dept_counts["Engineering"], 2)
        self.assertEqual(dept_counts["Marketing"], 1)


if __name__ == "__main__":
    unittest.main()