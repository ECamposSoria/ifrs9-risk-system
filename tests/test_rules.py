"""Unit tests for IFRS9 rules engine.

Tests the core functionality of the IFRS9 rules processing engine.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from pyspark.sql import DataFrame
from src.rules_engine import IFRS9RulesEngine

# Import datetime converter for safe conversions
import sys
sys.path.append('/home/eze/projects/ifrs9-risk-system/validation')
from datetime_converter import DateTimeConverter


@pytest.mark.spark
class TestIFRS9RulesEngine:
    """Test cases for IFRS9 Rules Engine using pytest fixtures."""
    
    
    def test_staging_classification(self, spark_session, sample_spark_df, datetime_converter):
        """Test IFRS9 stage classification logic."""
        engine = IFRS9RulesEngine(spark=spark_session)
        result_df = engine._apply_staging_rules(sample_spark_df)
        result_pd = datetime_converter.spark_to_pandas_safe(result_df)
        
        # Check stage assignments
        stages = result_pd.set_index("loan_id")["calculated_stage"].to_dict()
        
        # Loan 1: DPD=0, good credit -> Stage 1
        assert stages["L000001"] == "STAGE_1"
        
        # Loan 2: DPD=35 -> Stage 2
        assert stages["L000002"] == "STAGE_2"
        
        # Loan 3: DPD=95 -> Stage 3
        assert stages["L000003"] == "STAGE_3"
        
        # Loan 4: DPD=15 -> Stage 1
        assert stages["L000004"] == "STAGE_1"
    
    def test_pd_calculation(self, spark_session, sample_spark_df, datetime_converter):
        """Test Probability of Default calculation."""
        engine = IFRS9RulesEngine(spark=spark_session)
        staged_df = engine._apply_staging_rules(sample_spark_df)
        result_df = engine._calculate_risk_parameters(staged_df)
        result_pd = datetime_converter.spark_to_pandas_safe(result_df)
        
        pd_values = result_pd.set_index("loan_id")["calculated_pd"].to_dict()
        
        # Stage 3 should have PD = 1.0
        assert pd_values["L000003"] == 1.0
        
        # Stage 1 should have low PD
        assert pd_values["L000001"] < 0.05
        
        # Stage 2 should have medium PD
        assert pd_values["L000002"] > 0.05
        assert pd_values["L000002"] < 1.0
    
    def test_lgd_calculation(self, spark_session, sample_spark_df, datetime_converter):
        """Test Loss Given Default calculation."""
        engine = IFRS9RulesEngine(spark=spark_session)
        staged_df = engine._apply_staging_rules(sample_spark_df)
        result_df = engine._calculate_risk_parameters(staged_df)
        result_pd = datetime_converter.spark_to_pandas_safe(result_df)
        
        lgd_values = result_pd.set_index("loan_id")["calculated_lgd"].to_dict()
        
        # All LGD values should be between 0 and 1
        for loan_id, lgd in lgd_values.items():
            assert lgd >= 0.0
            assert lgd <= 1.0
        
        # Unsecured loan should have higher LGD
        assert lgd_values["L000003"] > lgd_values["L000001"]
    
    def test_ecl_calculation(self, spark_session, sample_spark_df, datetime_converter):
        """Test Expected Credit Loss calculation."""
        engine = IFRS9RulesEngine(spark=spark_session)
        processed_df = engine.process_portfolio(sample_spark_df)
        result_pd = datetime_converter.spark_to_pandas_safe(processed_df)
        
        ecl_values = result_pd.set_index("loan_id")["calculated_ecl"].to_dict()
        
        # All ECL values should be non-negative
        for loan_id, ecl in ecl_values.items():
            assert ecl >= 0.0
        
        # Stage 3 loan should have highest ECL
        max_ecl_loan = max(ecl_values, key=ecl_values.get)
        assert max_ecl_loan == "L000003"
    
    def test_validation(self, spark_session, sample_spark_df):
        """Test validation functions."""
        engine = IFRS9RulesEngine(spark=spark_session)
        processed_df = engine.process_portfolio(sample_spark_df)
        validations = engine.validate_calculations(processed_df)
        
        # Check that all validations pass
        for validation in validations:
            assert validation["passed"], \
                f"Validation failed: {validation['check']} - {validation['message']}"
    
    def test_summary_report(self, spark_session, sample_spark_df):
        """Test summary report generation."""
        engine = IFRS9RulesEngine(spark=spark_session)
        processed_df = engine.process_portfolio(sample_spark_df)
        summary = engine.generate_summary_report(processed_df)
        
        # Check report structure
        assert "portfolio_metrics" in summary
        assert "stage_distribution" in summary
        assert "risk_distribution" in summary
        
        # Check portfolio metrics
        metrics = summary["portfolio_metrics"]
        assert "total_loans" in metrics
        assert "total_exposure" in metrics
        assert "total_ecl" in metrics
        assert "coverage_ratio" in metrics
        
        # Check counts (updated to match sample data fixture)
        assert metrics["total_loans"] == 10
        assert metrics["total_exposure"] > 0


@pytest.mark.validation
class TestDataValidation:
    """Test cases for data validation module using pytest fixtures."""
    
    def test_loan_schema_validation(self):
        """Test loan portfolio schema validation."""
        # Use simplified validator to avoid pandera dependency issues
        from src.validation_simple import DataValidator
        validator = DataValidator()
        
        # Create valid loan data
        valid_loans = pd.DataFrame({
            "loan_id": ["L000001"],
            "customer_id": ["C000001"],
            "loan_amount": [100000.0],
            "interest_rate": [5.5],
            "term_months": [360],
            "loan_type": ["MORTGAGE"],
            "credit_score": [750],
            "days_past_due": [0],
            "current_balance": [95000.0],
            "provision_stage": ["STAGE_1"],
            "pd_rate": [0.02],
            "lgd_rate": [0.45],
        })
        
        passed, errors = validator.validate_loan_portfolio(valid_loans)
        assert passed
        assert len(errors) == 0
    
    def test_invalid_loan_data(self, validation_test_data):
        """Test validation with invalid loan data."""
        from src.validation_simple import DataValidator
        validator = DataValidator()
        
        passed, errors = validator.validate_loan_portfolio(validation_test_data)
        assert not passed
        assert len(errors) > 0


@pytest.mark.ml
@pytest.mark.slow
class TestMLModel:
    """Test cases for ML model module using pytest fixtures."""
    
    def test_feature_preparation(self, ml_test_data):
        """Test feature preparation."""
        from src.ml_model import CreditRiskClassifier
        classifier = CreditRiskClassifier(model_type="random_forest")
        
        X, feature_names = classifier.prepare_features(ml_test_data)
        
        # Check that features are created
        assert X is not None
        assert len(feature_names) > 10
        
        # Check for no missing values
        assert X.isnull().sum().sum() == 0
    
    def test_model_training(self, ml_test_data):
        """Test model training."""
        from src.ml_model import CreditRiskClassifier
        classifier = CreditRiskClassifier(model_type="random_forest")
        
        X, _ = classifier.prepare_features(ml_test_data)
        y = ml_test_data["provision_stage"]
        
        metrics = classifier.train_stage_classifier(X, y, test_size=0.3)
        
        # Check metrics are returned
        assert "accuracy" in metrics
        assert "cv_mean" in metrics
        assert "feature_importance" in metrics
        
        # Check accuracy is reasonable
        assert metrics["accuracy"] > 0.5
    
    def test_prediction(self, ml_test_data):
        """Test model prediction."""
        from src.ml_model import CreditRiskClassifier
        classifier = CreditRiskClassifier(model_type="random_forest")
        
        X, _ = classifier.prepare_features(ml_test_data)
        y = ml_test_data["provision_stage"]
        
        # Train model
        classifier.train_stage_classifier(X, y, test_size=0.3)
        
        # Make predictions
        X_test = X.iloc[:10]
        predictions, probabilities = classifier.predict_stage(X_test)
        
        # Check predictions
        assert len(predictions) == 10
        assert probabilities.shape[0] == 10
        
        # Check predictions are valid stages
        valid_stages = ["STAGE_1", "STAGE_2", "STAGE_3"]
        for pred in predictions:
            assert pred in valid_stages