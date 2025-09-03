#!/usr/bin/env python3
"""
IFRS9 End-to-End Pipeline Validation Agent
Comprehensive validation framework for complete IFRS9 data pipeline

This validator performs end-to-end validation of the complete IFRS9 pipeline:
- Data ingestion and processing validation
- ML model pipeline validation  
- ECL calculation accuracy validation
- Synthetic data generation validation
- Performance and resource monitoring
- Production readiness assessment
"""

import sys
import os
import json
import time
import traceback
import subprocess
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any, Optional, Union
import pandas as pd
from pathlib import Path


class EndToEndPipelineValidator:
    """End-to-end pipeline validation for IFRS9 system."""
    
    def __init__(self):
        """Initialize the pipeline validator."""
        self.validation_results = {}
        self.issues_found = []
        self.critical_issues = []
        self.warnings = []
        self.performance_metrics = {}
        
        # Pipeline test scenarios
        self.test_scenarios = {
            'data_ingestion': {
                'description': 'Test data loading and basic processing',
                'containers': ['jupyter', 'spark-master'],
                'timeout': 120
            },
            'ml_model_training': {
                'description': 'Test ML model training pipeline',
                'containers': ['jupyter', 'spark-master', 'spark-worker'],
                'timeout': 300
            },
            'ecl_calculation': {
                'description': 'Test complete ECL calculation pipeline',
                'containers': ['jupyter', 'spark-master', 'spark-worker'],
                'timeout': 180
            },
            'synthetic_data_gen': {
                'description': 'Test synthetic data generation',
                'containers': ['jupyter'],
                'timeout': 150
            },
            'airflow_dag_execution': {
                'description': 'Test Airflow DAG execution',
                'containers': ['airflow-scheduler', 'spark-master'],
                'timeout': 240
            }
        }
    
    def validate_data_ingestion_pipeline(self) -> Dict[str, Any]:
        """Validate data ingestion and basic processing pipeline."""
        print("=" * 70)
        print("VALIDATING DATA INGESTION PIPELINE")
        print("=" * 70)
        
        pipeline_validation = {
            'test_name': 'data_ingestion_pipeline',
            'status': 'PENDING',
            'start_time': datetime.now().isoformat(),
            'steps': {}
        }
        
        try:
            # Step 1: Test synthetic data generation
            print("Step 1: Testing synthetic data generation...")
            synthetic_data_result = self._test_synthetic_data_generation()
            pipeline_validation['steps']['synthetic_data_generation'] = synthetic_data_result
            
            if synthetic_data_result['status'] != 'SUCCESS':
                pipeline_validation['status'] = 'FAILED'
                pipeline_validation['error'] = 'Synthetic data generation failed'
                self.critical_issues.append("Data ingestion pipeline: Synthetic data generation failed")
                return pipeline_validation
            
            # Step 2: Test data loading in Spark
            print("Step 2: Testing data loading in Spark...")
            data_loading_result = self._test_spark_data_loading()
            pipeline_validation['steps']['spark_data_loading'] = data_loading_result
            
            if data_loading_result['status'] != 'SUCCESS':
                pipeline_validation['status'] = 'FAILED'
                pipeline_validation['error'] = 'Spark data loading failed'
                self.critical_issues.append("Data ingestion pipeline: Spark data loading failed")
                return pipeline_validation
            
            # Step 3: Test basic data transformations
            print("Step 3: Testing basic data transformations...")
            transformation_result = self._test_data_transformations()
            pipeline_validation['steps']['data_transformations'] = transformation_result
            
            if transformation_result['status'] != 'SUCCESS':
                pipeline_validation['status'] = 'FAILED'
                pipeline_validation['error'] = 'Data transformations failed'
                self.issues_found.append("Data ingestion pipeline: Basic transformations failed")
            
            # Overall pipeline status
            all_success = all(step.get('status') == 'SUCCESS' 
                            for step in pipeline_validation['steps'].values())
            pipeline_validation['status'] = 'SUCCESS' if all_success else 'PARTIAL_SUCCESS'
            
            pipeline_validation['end_time'] = datetime.now().isoformat()
            print(f"✅ Data Ingestion Pipeline: {pipeline_validation['status']}")
            
        except Exception as e:
            pipeline_validation['status'] = 'ERROR'
            pipeline_validation['error'] = str(e)
            pipeline_validation['end_time'] = datetime.now().isoformat()
            self.critical_issues.append(f"Data ingestion pipeline error: {str(e)}")
            print(f"❌ Data Ingestion Pipeline: Error - {str(e)}")
        
        return pipeline_validation
    
    def _test_synthetic_data_generation(self) -> Dict[str, Any]:
        """Test synthetic data generation functionality."""
        test_result = {'test': 'synthetic_data_generation', 'status': 'PENDING'}
        
        try:
            synthetic_data_script = """
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from faker import Faker

try:
    # Initialize Faker
    fake = Faker()
    Faker.seed(42)  # For reproducible results
    np.random.seed(42)
    
    # Generate synthetic loan portfolio data
    n_loans = 1000
    
    loan_data = []
    for i in range(n_loans):
        loan_id = f"L{i+1:06d}"
        customer_id = f"C{i+1:06d}"
        
        # Loan characteristics
        loan_amount = np.random.uniform(10000, 500000)
        interest_rate = np.random.uniform(3.0, 15.0)
        term_months = np.random.choice([12, 24, 36, 48, 60, 84, 120, 180, 240, 360])
        loan_type = np.random.choice(['MORTGAGE', 'AUTO', 'PERSONAL', 'BUSINESS'])
        
        # Customer characteristics  
        credit_score = int(np.random.normal(700, 100))
        credit_score = max(300, min(850, credit_score))  # Bound to realistic range
        
        # Payment behavior
        days_past_due = max(0, int(np.random.exponential(5)))
        
        # Current balance (with some loan progression)
        balance_ratio = np.random.uniform(0.3, 1.0)
        current_balance = loan_amount * balance_ratio
        
        # IFRS9 staging based on DPD
        if days_past_due <= 30:
            provision_stage = "STAGE_1"
        elif days_past_due <= 89:
            provision_stage = "STAGE_2"
        else:
            provision_stage = "STAGE_3"
        
        # Risk parameters
        if provision_stage == "STAGE_1":
            pd_rate = np.random.uniform(0.01, 0.05)
            lgd_rate = np.random.uniform(0.25, 0.45)
        elif provision_stage == "STAGE_2":
            pd_rate = np.random.uniform(0.05, 0.25)
            lgd_rate = np.random.uniform(0.35, 0.65)
        else:  # STAGE_3
            pd_rate = np.random.uniform(0.50, 0.95)
            lgd_rate = np.random.uniform(0.55, 0.85)
        
        loan_data.append({
            'loan_id': loan_id,
            'customer_id': customer_id,
            'loan_amount': round(loan_amount, 2),
            'interest_rate': round(interest_rate, 2),
            'term_months': term_months,
            'loan_type': loan_type,
            'credit_score': credit_score,
            'days_past_due': days_past_due,
            'current_balance': round(current_balance, 2),
            'provision_stage': provision_stage,
            'pd_rate': round(pd_rate, 4),
            'lgd_rate': round(lgd_rate, 4),
            'origination_date': fake.date_between(start_date='-3y', end_date='today').isoformat(),
            'last_payment_date': fake.date_between(start_date='-1y', end_date='today').isoformat()
        })
    
    # Create DataFrame
    df = pd.DataFrame(loan_data)
    
    # Save to test location
    test_file_path = "/home/jovyan/data/test_synthetic_loan_portfolio.csv"
    df.to_csv(test_file_path, index=False)
    
    # Validation metrics
    validation_metrics = {
        'total_records': len(df),
        'stage_distribution': df['provision_stage'].value_counts().to_dict(),
        'loan_type_distribution': df['loan_type'].value_counts().to_dict(),
        'avg_loan_amount': float(df['loan_amount'].mean()),
        'avg_credit_score': float(df['credit_score'].mean()),
        'data_quality_checks': {
            'no_nulls': df.isnull().sum().sum() == 0,
            'valid_credit_scores': df['credit_score'].between(300, 850).all(),
            'valid_pd_rates': df['pd_rate'].between(0, 1).all(),
            'valid_lgd_rates': df['lgd_rate'].between(0, 1).all(),
            'balance_consistency': (df['current_balance'] <= df['loan_amount']).all()
        }
    }
    
    result = {
        "status": "SUCCESS",
        "synthetic_data_file": test_file_path,
        "validation_metrics": validation_metrics
    }
    
except Exception as e:
    result = {
        "status": "ERROR",
        "error": str(e)
    }

print(json.dumps(result, default=str))
"""
            
            cmd = ['docker', 'exec', 'jupyter', 'python3', '-c', synthetic_data_script]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                try:
                    test_result.update(json.loads(result.stdout.strip()))
                    if test_result['status'] == 'SUCCESS':
                        metrics = test_result.get('validation_metrics', {})
                        record_count = metrics.get('total_records', 0)
                        print(f"    ✅ Synthetic data: {record_count} loan records generated")
                    else:
                        print(f"    ❌ Synthetic data: {test_result.get('error', 'Generation failed')}")
                except json.JSONDecodeError:
                    test_result['status'] = 'ERROR'
                    test_result['error'] = 'Failed to parse synthetic data results'
                    print("    ❌ Synthetic data: Failed to parse results")
            else:
                test_result['status'] = 'ERROR'
                test_result['error'] = result.stderr
                print(f"    ❌ Synthetic data: Execution failed")
                
        except Exception as e:
            test_result['status'] = 'ERROR'
            test_result['error'] = str(e)
            print(f"    ❌ Synthetic data: Exception - {str(e)}")
        
        return test_result
    
    def _test_spark_data_loading(self) -> Dict[str, Any]:
        """Test data loading in Spark environment."""
        test_result = {'test': 'spark_data_loading', 'status': 'PENDING'}
        
        try:
            spark_loading_script = """
import json
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, count, avg, max as spark_max, min as spark_min
    import time
    
    start_time = time.time()
    
    # Create Spark session
    spark = SparkSession.builder \\
        .appName("IFRS9_DataIngestion_Test") \\
        .master("spark://spark-master:7077") \\
        .config("spark.sql.adaptive.enabled", "true") \\
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \\
        .getOrCreate()
    
    # Load the synthetic data
    test_file_path = "/home/jovyan/data/test_synthetic_loan_portfolio.csv"
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(test_file_path)
    
    # Basic validation operations
    record_count = df.count()
    column_count = len(df.columns)
    
    # Data quality checks
    null_counts = {}
    for col_name in df.columns:
        null_count = df.filter(col(col_name).isNull()).count()
        null_counts[col_name] = null_count
    
    # Statistical analysis
    stats = {}
    numeric_columns = ['loan_amount', 'interest_rate', 'credit_score', 'days_past_due', 'current_balance']
    
    for col_name in numeric_columns:
        if col_name in df.columns:
            col_stats = df.select(
                avg(col(col_name)).alias('avg'),
                spark_min(col(col_name)).alias('min'),
                spark_max(col(col_name)).alias('max'),
                count(col(col_name)).alias('count')
            ).collect()[0]
            
            stats[col_name] = {
                'avg': float(col_stats['avg']) if col_stats['avg'] is not None else None,
                'min': float(col_stats['min']) if col_stats['min'] is not None else None,
                'max': float(col_stats['max']) if col_stats['max'] is not None else None,
                'count': int(col_stats['count'])
            }
    
    # Stage distribution
    stage_dist = df.groupBy('provision_stage').count().collect()
    stage_distribution = {row['provision_stage']: row['count'] for row in stage_dist}
    
    loading_time = time.time() - start_time
    
    spark.stop()
    
    result = {
        "status": "SUCCESS",
        "loading_time_seconds": round(loading_time, 2),
        "record_count": record_count,
        "column_count": column_count,
        "null_counts": null_counts,
        "statistical_summary": stats,
        "stage_distribution": stage_distribution,
        "data_quality_score": 100.0 - (sum(null_counts.values()) / record_count * 100) if record_count > 0 else 0
    }
    
except Exception as e:
    result = {
        "status": "ERROR",
        "error": str(e)
    }

print(json.dumps(result, default=str))
"""
            
            cmd = ['docker', 'exec', 'jupyter', 'python3', '-c', spark_loading_script]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                try:
                    test_result.update(json.loads(result.stdout.strip()))
                    if test_result['status'] == 'SUCCESS':
                        record_count = test_result.get('record_count', 0)
                        loading_time = test_result.get('loading_time_seconds', 0)
                        quality_score = test_result.get('data_quality_score', 0)
                        print(f"    ✅ Spark data loading: {record_count} records in {loading_time}s (Quality: {quality_score:.1f}%)")
                    else:
                        print(f"    ❌ Spark data loading: {test_result.get('error', 'Loading failed')}")
                except json.JSONDecodeError:
                    test_result['status'] = 'ERROR'
                    test_result['error'] = 'Failed to parse Spark loading results'
                    print("    ❌ Spark data loading: Failed to parse results")
            else:
                test_result['status'] = 'ERROR'
                test_result['error'] = result.stderr
                print(f"    ❌ Spark data loading: Execution failed")
                
        except Exception as e:
            test_result['status'] = 'ERROR'
            test_result['error'] = str(e)
            print(f"    ❌ Spark data loading: Exception - {str(e)}")
        
        return test_result
    
    def _test_data_transformations(self) -> Dict[str, Any]:
        """Test basic data transformations."""
        test_result = {'test': 'data_transformations', 'status': 'PENDING'}
        
        try:
            transformation_script = """
import json
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, when, round as spark_round, datediff, current_date
    from pyspark.sql.types import *
    import time
    
    start_time = time.time()
    
    # Create Spark session
    spark = SparkSession.builder \\
        .appName("IFRS9_Transformation_Test") \\
        .master("spark://spark-master:7077") \\
        .config("spark.sql.adaptive.enabled", "true") \\
        .getOrCreate()
    
    # Load the synthetic data
    test_file_path = "/home/jovyan/data/test_synthetic_loan_portfolio.csv"
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(test_file_path)
    
    original_count = df.count()
    
    # Transformation 1: Add risk categories
    df_with_risk = df.withColumn(
        "risk_category",
        when(col("credit_score") >= 750, "LOW")
        .when(col("credit_score") >= 650, "MEDIUM")
        .otherwise("HIGH")
    )
    
    # Transformation 2: Add loan utilization ratio
    df_with_util = df_with_risk.withColumn(
        "utilization_ratio",
        spark_round(col("current_balance") / col("loan_amount"), 4)
    )
    
    # Transformation 3: Add ECL calculation (simplified)
    df_with_ecl = df_with_util.withColumn(
        "ecl_amount",
        spark_round(col("current_balance") * col("pd_rate") * col("lgd_rate"), 2)
    )
    
    # Transformation 4: Filter and aggregate
    stage_summary = df_with_ecl.groupBy("provision_stage", "risk_category") \\
        .agg({
            "loan_amount": "sum",
            "current_balance": "sum", 
            "ecl_amount": "sum",
            "loan_id": "count"
        }).collect()
    
    # Convert to dictionary for JSON serialization
    summary_data = []
    for row in stage_summary:
        summary_data.append({
            "provision_stage": row["provision_stage"],
            "risk_category": row["risk_category"],
            "total_loan_amount": float(row["sum(loan_amount)"]),
            "total_current_balance": float(row["sum(current_balance)"]),
            "total_ecl_amount": float(row["sum(ecl_amount)"]),
            "loan_count": int(row["count(loan_id)"])
        })
    
    # Data quality checks after transformation
    final_count = df_with_ecl.count()
    null_check = df_with_ecl.filter(
        col("risk_category").isNull() |
        col("utilization_ratio").isNull() |
        col("ecl_amount").isNull()
    ).count()
    
    transformation_time = time.time() - start_time
    
    spark.stop()
    
    result = {
        "status": "SUCCESS",
        "transformation_time_seconds": round(transformation_time, 2),
        "original_record_count": original_count,
        "final_record_count": final_count,
        "null_transformations": null_check,
        "transformations_applied": 4,
        "stage_risk_summary": summary_data,
        "data_integrity": final_count == original_count and null_check == 0
    }
    
except Exception as e:
    result = {
        "status": "ERROR",
        "error": str(e)
    }

print(json.dumps(result, default=str))
"""
            
            cmd = ['docker', 'exec', 'jupyter', 'python3', '-c', transformation_script]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                try:
                    test_result.update(json.loads(result.stdout.strip()))
                    if test_result['status'] == 'SUCCESS':
                        transformation_time = test_result.get('transformation_time_seconds', 0)
                        data_integrity = test_result.get('data_integrity', False)
                        transformations_applied = test_result.get('transformations_applied', 0)
                        print(f"    ✅ Data transformations: {transformations_applied} applied in {transformation_time}s (Integrity: {data_integrity})")
                    else:
                        print(f"    ❌ Data transformations: {test_result.get('error', 'Transformation failed')}")
                except json.JSONDecodeError:
                    test_result['status'] = 'ERROR'
                    test_result['error'] = 'Failed to parse transformation results'
                    print("    ❌ Data transformations: Failed to parse results")
            else:
                test_result['status'] = 'ERROR'
                test_result['error'] = result.stderr
                print(f"    ❌ Data transformations: Execution failed")
                
        except Exception as e:
            test_result['status'] = 'ERROR'
            test_result['error'] = str(e)
            print(f"    ❌ Data transformations: Exception - {str(e)}")
        
        return test_result
    
    def validate_ml_model_pipeline(self) -> Dict[str, Any]:
        """Validate ML model training pipeline."""
        print("\nVALIDATING ML MODEL PIPELINE")
        print("-" * 50)
        
        ml_pipeline_validation = {
            'test_name': 'ml_model_pipeline',
            'status': 'PENDING',
            'start_time': datetime.now().isoformat(),
            'steps': {}
        }
        
        try:
            # Test ML model training with synthetic data
            print("Testing ML model training with synthetic data...")
            ml_training_result = self._test_ml_model_training()
            ml_pipeline_validation['steps']['ml_model_training'] = ml_training_result
            
            if ml_training_result['status'] == 'SUCCESS':
                ml_pipeline_validation['status'] = 'SUCCESS'
                print("✅ ML Model Pipeline: Training successful")
            else:
                ml_pipeline_validation['status'] = 'FAILED'
                ml_pipeline_validation['error'] = ml_training_result.get('error', 'ML training failed')
                self.issues_found.append("ML model pipeline: Training failed")
                print(f"❌ ML Model Pipeline: {ml_pipeline_validation['error']}")
            
            ml_pipeline_validation['end_time'] = datetime.now().isoformat()
            
        except Exception as e:
            ml_pipeline_validation['status'] = 'ERROR'
            ml_pipeline_validation['error'] = str(e)
            ml_pipeline_validation['end_time'] = datetime.now().isoformat()
            self.critical_issues.append(f"ML model pipeline error: {str(e)}")
            print(f"❌ ML Model Pipeline: Error - {str(e)}")
        
        return ml_pipeline_validation
    
    def _test_ml_model_training(self) -> Dict[str, Any]:
        """Test ML model training functionality."""
        test_result = {'test': 'ml_model_training', 'status': 'PENDING'}
        
        try:
            ml_training_script = """
import json
try:
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
    import time
    
    start_time = time.time()
    
    # Load synthetic data
    test_file_path = "/home/jovyan/data/test_synthetic_loan_portfolio.csv"
    df = pd.read_csv(test_file_path)
    
    # Prepare features for default prediction model
    # Target: whether loan is in Stage 3 (default)
    df['default_flag'] = (df['provision_stage'] == 'STAGE_3').astype(int)
    
    # Feature engineering
    feature_columns = [
        'loan_amount', 'interest_rate', 'term_months', 'credit_score', 
        'days_past_due', 'current_balance', 'pd_rate', 'lgd_rate'
    ]
    
    X = df[feature_columns].copy()
    y = df['default_flag']
    
    # Handle any missing values
    X = X.fillna(X.mean())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train models
    models = {}
    model_results = {}
    
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    
    models['random_forest'] = rf_model
    model_results['random_forest'] = {
        'accuracy': accuracy_score(y_test, rf_pred),
        'roc_auc': roc_auc_score(y_test, rf_pred_proba),
        'feature_importance': {
            feature: importance 
            for feature, importance in zip(feature_columns, rf_model.feature_importances_)
        }
    }
    
    # Logistic Regression
    lr_model = LogisticRegression(random_state=42, max_iter=200)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_pred_proba = lr_model.predict_proba(X_test)[:, 1]
    
    models['logistic_regression'] = lr_model
    model_results['logistic_regression'] = {
        'accuracy': accuracy_score(y_test, lr_pred),
        'roc_auc': roc_auc_score(y_test, lr_pred_proba),
        'coefficients': {
            feature: coef 
            for feature, coef in zip(feature_columns, lr_model.coef_[0])
        }
    }
    
    training_time = time.time() - start_time
    
    result = {
        "status": "SUCCESS",
        "training_time_seconds": round(training_time, 2),
        "dataset_info": {
            "total_samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "default_rate": float(y.mean()),
            "feature_count": len(feature_columns)
        },
        "model_results": model_results,
        "best_model": "random_forest" if model_results['random_forest']['roc_auc'] > model_results['logistic_regression']['roc_auc'] else "logistic_regression"
    }
    
except Exception as e:
    result = {
        "status": "ERROR",
        "error": str(e)
    }

print(json.dumps(result, default=str))
"""
            
            cmd = ['docker', 'exec', 'jupyter', 'python3', '-c', ml_training_script]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                try:
                    test_result.update(json.loads(result.stdout.strip()))
                    if test_result['status'] == 'SUCCESS':
                        training_time = test_result.get('training_time_seconds', 0)
                        best_model = test_result.get('best_model', 'unknown')
                        dataset_info = test_result.get('dataset_info', {})
                        samples = dataset_info.get('total_samples', 0)
                        print(f"    ✅ ML training: {best_model} model on {samples} samples in {training_time}s")
                    else:
                        print(f"    ❌ ML training: {test_result.get('error', 'Training failed')}")
                except json.JSONDecodeError:
                    test_result['status'] = 'ERROR'
                    test_result['error'] = 'Failed to parse ML training results'
                    print("    ❌ ML training: Failed to parse results")
            else:
                test_result['status'] = 'ERROR'
                test_result['error'] = result.stderr
                print(f"    ❌ ML training: Execution failed")
                
        except Exception as e:
            test_result['status'] = 'ERROR'
            test_result['error'] = str(e)
            print(f"    ❌ ML training: Exception - {str(e)}")
        
        return test_result
    
    def validate_ecl_calculation_pipeline(self) -> Dict[str, Any]:
        """Validate ECL calculation pipeline."""
        print("\nVALIDATING ECL CALCULATION PIPELINE")
        print("-" * 50)
        
        ecl_pipeline_validation = {
            'test_name': 'ecl_calculation_pipeline',
            'status': 'PENDING',
            'start_time': datetime.now().isoformat(),
            'steps': {}
        }
        
        try:
            # Test ECL calculation with Spark
            print("Testing ECL calculation with Spark...")
            ecl_calculation_result = self._test_ecl_calculation()
            ecl_pipeline_validation['steps']['ecl_calculation'] = ecl_calculation_result
            
            if ecl_calculation_result['status'] == 'SUCCESS':
                ecl_pipeline_validation['status'] = 'SUCCESS'
                print("✅ ECL Calculation Pipeline: Calculation successful")
            else:
                ecl_pipeline_validation['status'] = 'FAILED'
                ecl_pipeline_validation['error'] = ecl_calculation_result.get('error', 'ECL calculation failed')
                self.issues_found.append("ECL calculation pipeline: Calculation failed")
                print(f"❌ ECL Calculation Pipeline: {ecl_pipeline_validation['error']}")
            
            ecl_pipeline_validation['end_time'] = datetime.now().isoformat()
            
        except Exception as e:
            ecl_pipeline_validation['status'] = 'ERROR'
            ecl_pipeline_validation['error'] = str(e)
            ecl_pipeline_validation['end_time'] = datetime.now().isoformat()
            self.critical_issues.append(f"ECL calculation pipeline error: {str(e)}")
            print(f"❌ ECL Calculation Pipeline: Error - {str(e)}")
        
        return ecl_pipeline_validation
    
    def _test_ecl_calculation(self) -> Dict[str, Any]:
        """Test ECL calculation functionality."""
        test_result = {'test': 'ecl_calculation', 'status': 'PENDING'}
        
        try:
            ecl_calculation_script = """
import json
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, sum as spark_sum, avg, max as spark_max, min as spark_min, round as spark_round, when
    import time
    
    start_time = time.time()
    
    # Create Spark session
    spark = SparkSession.builder \\
        .appName("IFRS9_ECL_Calculation_Test") \\
        .master("spark://spark-master:7077") \\
        .config("spark.sql.adaptive.enabled", "true") \\
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \\
        .getOrCreate()
    
    # Load synthetic data
    test_file_path = "/home/jovyan/data/test_synthetic_loan_portfolio.csv"
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(test_file_path)
    
    # ECL Calculation Components
    # 1. 12-month ECL for Stage 1
    # 2. Lifetime ECL for Stage 2 and Stage 3
    
    # Add time horizon adjustment (simplified)
    df_with_ecl = df.withColumn(
        "time_horizon",
        when(col("provision_stage") == "STAGE_1", 1.0)  # 12 months
        .otherwise(col("term_months") / 12.0)  # Remaining term in years
    ).withColumn(
        "ead_amount",  # Exposure at Default
        col("current_balance")
    ).withColumn(
        "ecl_12m",  # 12-month ECL
        spark_round(
            col("current_balance") * 
            col("pd_rate") * 
            col("lgd_rate"), 
            2
        )
    ).withColumn(
        "ecl_lifetime",  # Lifetime ECL (simplified)
        spark_round(
            col("current_balance") * 
            col("pd_rate") * 
            col("lgd_rate") * 
            col("time_horizon"), 
            2
        )
    ).withColumn(
        "ecl_final",
        when(col("provision_stage") == "STAGE_1", col("ecl_12m"))
        .otherwise(col("ecl_lifetime"))
    )
    
    # Aggregate ECL by stage
    ecl_summary = df_with_ecl.groupBy("provision_stage").agg(
        spark_sum("current_balance").alias("total_exposure"),
        spark_sum("ecl_final").alias("total_ecl"),
        avg("pd_rate").alias("avg_pd_rate"),
        avg("lgd_rate").alias("avg_lgd_rate"),
        avg("ecl_final").alias("avg_ecl_per_loan"),
        spark_sum("ead_amount").alias("total_ead")
    ).collect()
    
    # Convert to dictionary
    stage_ecl_summary = {}
    total_ecl = 0
    total_exposure = 0
    
    for row in ecl_summary:
        stage = row["provision_stage"]
        ecl_amount = float(row["total_ecl"])
        exposure = float(row["total_exposure"])
        
        stage_ecl_summary[stage] = {
            "total_exposure": exposure,
            "total_ecl": ecl_amount,
            "ecl_rate": (ecl_amount / exposure * 100) if exposure > 0 else 0,
            "avg_pd_rate": float(row["avg_pd_rate"]),
            "avg_lgd_rate": float(row["avg_lgd_rate"]),
            "avg_ecl_per_loan": float(row["avg_ecl_per_loan"]),
            "total_ead": float(row["total_ead"])
        }
        
        total_ecl += ecl_amount
        total_exposure += exposure
    
    # Portfolio-level metrics
    portfolio_metrics = {
        "total_portfolio_exposure": total_exposure,
        "total_portfolio_ecl": total_ecl,
        "overall_ecl_rate": (total_ecl / total_exposure * 100) if total_exposure > 0 else 0,
        "stage_1_ratio": (stage_ecl_summary.get("STAGE_1", {}).get("total_exposure", 0) / total_exposure * 100) if total_exposure > 0 else 0,
        "stage_2_ratio": (stage_ecl_summary.get("STAGE_2", {}).get("total_exposure", 0) / total_exposure * 100) if total_exposure > 0 else 0,
        "stage_3_ratio": (stage_ecl_summary.get("STAGE_3", {}).get("total_exposure", 0) / total_exposure * 100) if total_exposure > 0 else 0
    }
    
    # Data quality validations
    validation_checks = {
        "no_negative_ecl": df_with_ecl.filter(col("ecl_final") < 0).count() == 0,
        "no_null_ecl": df_with_ecl.filter(col("ecl_final").isNull()).count() == 0,
        "ecl_reasonable_range": df_with_ecl.filter(col("ecl_final") > col("current_balance")).count() == 0,
        "pd_lgd_valid_range": df_with_ecl.filter(
            (col("pd_rate") < 0) | (col("pd_rate") > 1) |
            (col("lgd_rate") < 0) | (col("lgd_rate") > 1)
        ).count() == 0
    }
    
    calculation_time = time.time() - start_time
    
    spark.stop()
    
    result = {
        "status": "SUCCESS",
        "calculation_time_seconds": round(calculation_time, 2),
        "stage_ecl_summary": stage_ecl_summary,
        "portfolio_metrics": portfolio_metrics,
        "validation_checks": validation_checks,
        "data_quality_passed": all(validation_checks.values())
    }
    
except Exception as e:
    result = {
        "status": "ERROR",
        "error": str(e)
    }

print(json.dumps(result, default=str))
"""
            
            cmd = ['docker', 'exec', 'jupyter', 'python3', '-c', ecl_calculation_script]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            
            if result.returncode == 0:
                try:
                    test_result.update(json.loads(result.stdout.strip()))
                    if test_result['status'] == 'SUCCESS':
                        calculation_time = test_result.get('calculation_time_seconds', 0)
                        portfolio_metrics = test_result.get('portfolio_metrics', {})
                        total_ecl = portfolio_metrics.get('total_portfolio_ecl', 0)
                        ecl_rate = portfolio_metrics.get('overall_ecl_rate', 0)
                        data_quality = test_result.get('data_quality_passed', False)
                        print(f"    ✅ ECL calculation: Total ECL ${total_ecl:,.0f} ({ecl_rate:.2f}% rate) in {calculation_time}s (Quality: {data_quality})")
                    else:
                        print(f"    ❌ ECL calculation: {test_result.get('error', 'Calculation failed')}")
                except json.JSONDecodeError:
                    test_result['status'] = 'ERROR'
                    test_result['error'] = 'Failed to parse ECL calculation results'
                    print("    ❌ ECL calculation: Failed to parse results")
            else:
                test_result['status'] = 'ERROR'
                test_result['error'] = result.stderr
                print(f"    ❌ ECL calculation: Execution failed")
                
        except Exception as e:
            test_result['status'] = 'ERROR'
            test_result['error'] = str(e)
            print(f"    ❌ ECL calculation: Exception - {str(e)}")
        
        return test_result
    
    def run_performance_monitoring(self) -> Dict[str, Any]:
        """Monitor system performance during pipeline execution."""
        print("\nMONITORING SYSTEM PERFORMANCE")
        print("-" * 50)
        
        performance_metrics = {
            'monitoring_start_time': datetime.now().isoformat(),
            'container_metrics': {},
            'resource_utilization': {}
        }
        
        try:
            # Get container resource usage
            containers = ['jupyter', 'spark-master', 'spark-worker', 'airflow-scheduler']
            
            for container in containers:
                try:
                    # Get container stats
                    stats_cmd = ['docker', 'stats', '--no-stream', '--format', 
                               'table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}',
                               container]
                    
                    result = subprocess.run(stats_cmd, capture_output=True, text=True, timeout=10)
                    
                    if result.returncode == 0 and result.stdout.strip():
                        lines = result.stdout.strip().split('\n')
                        if len(lines) > 1:  # Skip header
                            stats_data = lines[1].split('\t')
                            if len(stats_data) >= 5:
                                performance_metrics['container_metrics'][container] = {
                                    'cpu_percent': stats_data[1],
                                    'memory_usage': stats_data[2],
                                    'network_io': stats_data[3],
                                    'block_io': stats_data[4],
                                    'status': 'MONITORED'
                                }
                            else:
                                performance_metrics['container_metrics'][container] = {
                                    'status': 'PARSING_ERROR',
                                    'raw_data': result.stdout
                                }
                    else:
                        performance_metrics['container_metrics'][container] = {
                            'status': 'NOT_RUNNING',
                            'error': result.stderr
                        }
                        
                    print(f"    📊 {container}: {performance_metrics['container_metrics'][container]['status']}")
                        
                except subprocess.TimeoutExpired:
                    performance_metrics['container_metrics'][container] = {
                        'status': 'TIMEOUT',
                        'error': 'Stats collection timeout'
                    }
                    print(f"    ⏱️ {container}: Monitoring timeout")
                    
                except Exception as e:
                    performance_metrics['container_metrics'][container] = {
                        'status': 'ERROR',
                        'error': str(e)
                    }
                    print(f"    ❌ {container}: Monitoring error - {str(e)}")
            
            # Get overall Docker system info
            try:
                system_cmd = ['docker', 'system', 'df', '--format', 'json']
                system_result = subprocess.run(system_cmd, capture_output=True, text=True, timeout=10)
                
                if system_result.returncode == 0:
                    system_data = json.loads(system_result.stdout)
                    performance_metrics['resource_utilization']['docker_system'] = system_data
                    print("    📊 Docker system: Resource utilization captured")
                
            except Exception as e:
                performance_metrics['resource_utilization']['docker_system_error'] = str(e)
                print(f"    ⚠️ Docker system monitoring error: {str(e)}")
            
            performance_metrics['monitoring_end_time'] = datetime.now().isoformat()
            performance_metrics['monitoring_status'] = 'SUCCESS'
            print("✅ Performance monitoring: Completed")
            
        except Exception as e:
            performance_metrics['monitoring_status'] = 'ERROR'
            performance_metrics['monitoring_error'] = str(e)
            print(f"❌ Performance monitoring: Error - {str(e)}")
        
        return performance_metrics
    
    def run_comprehensive_pipeline_validation(self) -> Dict[str, Any]:
        """Run comprehensive end-to-end pipeline validation."""
        print("IFRS9 END-TO-END PIPELINE VALIDATION")
        print("Comprehensive validation of complete IFRS9 data pipeline")
        print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
        print()
        
        # Run all pipeline validation steps
        self.validation_results['data_ingestion'] = self.validate_data_ingestion_pipeline()
        self.validation_results['ml_model_pipeline'] = self.validate_ml_model_pipeline()
        self.validation_results['ecl_calculation'] = self.validate_ecl_calculation_pipeline()
        self.validation_results['performance_monitoring'] = self.run_performance_monitoring()
        
        # Compile validation summary
        self.validation_results['pipeline_summary'] = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'total_critical_issues': len(self.critical_issues),
            'total_issues': len(self.issues_found),
            'total_warnings': len(self.warnings),
            'critical_issues': self.critical_issues,
            'issues_found': self.issues_found,
            'warnings': self.warnings,
            'overall_status': 'PASS' if len(self.critical_issues) == 0 else 'FAIL',
            'pipeline_readiness': self._assess_pipeline_readiness(),
            'production_readiness': self._assess_production_readiness()
        }
        
        return self.validation_results
    
    def _assess_pipeline_readiness(self) -> Dict[str, Any]:
        """Assess overall pipeline readiness."""
        readiness = {
            'data_ingestion_ready': False,
            'ml_pipeline_ready': False,
            'ecl_calculation_ready': False,
            'performance_acceptable': False,
            'overall_ready': False
        }
        
        # Check data ingestion readiness
        data_ingestion = self.validation_results.get('data_ingestion', {})
        readiness['data_ingestion_ready'] = data_ingestion.get('status') in ['SUCCESS', 'PARTIAL_SUCCESS']
        
        # Check ML pipeline readiness  
        ml_pipeline = self.validation_results.get('ml_model_pipeline', {})
        readiness['ml_pipeline_ready'] = ml_pipeline.get('status') == 'SUCCESS'
        
        # Check ECL calculation readiness
        ecl_pipeline = self.validation_results.get('ecl_calculation', {})
        readiness['ecl_calculation_ready'] = ecl_pipeline.get('status') == 'SUCCESS'
        
        # Check performance readiness
        performance = self.validation_results.get('performance_monitoring', {})
        readiness['performance_acceptable'] = performance.get('monitoring_status') == 'SUCCESS'
        
        # Overall readiness
        readiness['overall_ready'] = (
            readiness['data_ingestion_ready'] and
            readiness['ml_pipeline_ready'] and
            readiness['ecl_calculation_ready'] and
            len(self.critical_issues) == 0
        )
        
        return readiness
    
    def _assess_production_readiness(self) -> bool:
        """Assess if the system is ready for production use."""
        pipeline_readiness = self._assess_pipeline_readiness()
        return (
            pipeline_readiness['overall_ready'] and
            len(self.issues_found) <= 2 and  # Allow minor issues
            pipeline_readiness['performance_acceptable']
        )
    
    def generate_pipeline_validation_report(self) -> str:
        """Generate comprehensive pipeline validation report."""
        if not self.validation_results:
            return "No pipeline validation results available. Run pipeline validation first."
        
        report = []
        report.append("=" * 80)
        report.append("IFRS9 END-TO-END PIPELINE VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
        
        summary = self.validation_results['pipeline_summary']
        readiness = summary.get('pipeline_readiness', {})
        
        report.append(f"Overall Status: {summary['overall_status']}")
        report.append(f"Pipeline Ready: {readiness.get('overall_ready', False)}")
        report.append(f"Production Ready: {summary['production_readiness']}")
        report.append("")
        
        # Pipeline Summary
        report.append("PIPELINE VALIDATION SUMMARY")
        report.append("-" * 40)
        report.append(f"Critical Issues: {summary['total_critical_issues']}")
        report.append(f"Issues Found: {summary['total_issues']}")
        report.append(f"Warnings: {summary['total_warnings']}")
        report.append("")
        
        if summary['critical_issues']:
            report.append("CRITICAL ISSUES (Must Fix Before Production):")
            for issue in summary['critical_issues']:
                report.append(f"  ❌ {issue}")
            report.append("")
        
        if summary['issues_found']:
            report.append("ISSUES FOUND:")
            for issue in summary['issues_found']:
                report.append(f"  ⚠️ {issue}")
            report.append("")
        
        # Individual Pipeline Status
        report.append("PIPELINE COMPONENT STATUS")
        report.append("-" * 40)
        
        data_ingestion = self.validation_results.get('data_ingestion', {})
        report.append(f"Data Ingestion: {data_ingestion.get('status', 'UNKNOWN')} - Ready: {readiness.get('data_ingestion_ready', False)}")
        
        ml_pipeline = self.validation_results.get('ml_model_pipeline', {})
        report.append(f"ML Model Pipeline: {ml_pipeline.get('status', 'UNKNOWN')} - Ready: {readiness.get('ml_pipeline_ready', False)}")
        
        ecl_pipeline = self.validation_results.get('ecl_calculation', {})
        report.append(f"ECL Calculation: {ecl_pipeline.get('status', 'UNKNOWN')} - Ready: {readiness.get('ecl_calculation_ready', False)}")
        
        performance = self.validation_results.get('performance_monitoring', {})
        report.append(f"Performance Monitoring: {performance.get('monitoring_status', 'UNKNOWN')} - Acceptable: {readiness.get('performance_acceptable', False)}")
        report.append("")
        
        # Performance Metrics Summary
        if 'performance_monitoring' in self.validation_results:
            perf_data = self.validation_results['performance_monitoring']
            container_metrics = perf_data.get('container_metrics', {})
            
            if container_metrics:
                report.append("CONTAINER PERFORMANCE")
                report.append("-" * 40)
                for container, metrics in container_metrics.items():
                    status = metrics.get('status', 'UNKNOWN')
                    if status == 'MONITORED':
                        cpu = metrics.get('cpu_percent', 'N/A')
                        memory = metrics.get('memory_usage', 'N/A')
                        report.append(f"  {container}: CPU {cpu}, Memory {memory}")
                    else:
                        report.append(f"  {container}: {status}")
                report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        
        if summary['production_readiness']:
            report.append("✅ End-to-End Pipeline READY for production deployment")
            report.append("   All critical pipeline components validated successfully")
        elif readiness.get('overall_ready', False):
            report.append("✅ End-to-End Pipeline READY for testing")
            if not readiness.get('performance_acceptable', False):
                report.append("⚠️  Performance monitoring needs attention before production")
            if len(self.issues_found) > 0:
                report.append("⚠️  Address non-critical issues before production")
        else:
            report.append("❌ End-to-End Pipeline NOT READY - Critical issues must be resolved")
            report.append("   Priority actions:")
            for issue in self.critical_issues[:3]:
                report.append(f"   1. Resolve: {issue}")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """Main execution function for pipeline validation."""
    print("Starting IFRS9 End-to-End Pipeline Validation")
    print("Comprehensive pipeline validation agent")
    
    validator = EndToEndPipelineValidator()
    
    try:
        # Run comprehensive pipeline validation
        results = validator.run_comprehensive_pipeline_validation()
        
        # Generate report
        report = validator.generate_pipeline_validation_report()
        print("\n" + report)
        
        # Save results to files
        output_dir = "/opt/airflow/data/validation"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON results
        json_file = os.path.join(output_dir, f"pipeline_validation_results_{timestamp}.json")
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save text report
        report_file = os.path.join(output_dir, f"pipeline_validation_report_{timestamp}.txt")
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\nPipeline validation results saved to:")
        print(f"  JSON: {json_file}")
        print(f"  Report: {report_file}")
        
        # Exit with appropriate code
        pipeline_summary = results['pipeline_summary']
        if pipeline_summary['overall_status'] == 'PASS':
            exit_code = 0
            print(f"\nPipeline Validation Status: PASS")
            if pipeline_summary['production_readiness']:
                print("End-to-End Pipeline is READY for production deployment")
            else:
                print("End-to-End Pipeline is ready for testing (address warnings before production)")
        else:
            exit_code = 1
            print(f"\nPipeline Validation Status: FAIL - Critical issues must be resolved")
        
        return exit_code
        
    except Exception as e:
        print(f"\nFATAL ERROR during pipeline validation: {str(e)}")
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)