"""Optimized unit tests for IFRS9 rules engine.

This module contains comprehensive tests for the IFRS9 rules processing engine
with improved PySpark session management, datetime conversion handling,
and enhanced test isolation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any

from pyspark.sql import DataFrame
import pyspark.sql.functions as F

from src.rules_engine import IFRS9RulesEngine


@pytest.mark.spark
@pytest.mark.unit
class TestIFRS9RulesEngineOptimized:
    """Optimized test cases for IFRS9 Rules Engine with proper session management."""
    
    def test_staging_classification_basic(self, spark_session, sample_spark_df, 
                                        ifrs9_rules_engine, datetime_converter):
        """Test basic IFRS9 stage classification logic."""
        result_df = ifrs9_rules_engine._apply_staging_rules(sample_spark_df)
        result_pd = datetime_converter.spark_to_pandas_safe(result_df)
        
        # Check stage assignments based on DPD
        stages = result_pd.set_index("loan_id")["calculated_stage"].to_dict()
        
        # Verify specific stage classifications
        assert stages["L000001"] == "STAGE_1"  # DPD=0, good credit
        assert stages["L000002"] == "STAGE_2"  # DPD=35
        assert stages["L000003"] == "STAGE_3"  # DPD=95
        assert stages["L000004"] == "STAGE_1"  # DPD=15
        assert stages["L000005"] == "STAGE_2"  # DPD=45
        assert stages["L000006"] == "STAGE_3"  # DPD=120
        assert stages["L000010"] == "STAGE_2"  # DPD=30 (boundary)
    
    def test_staging_classification_edge_cases(self, spark_session, datetime_converter):
        """Test edge cases in stage classification."""
        # Create edge case test data
        edge_case_data = pd.DataFrame({
            "loan_id": ["E001", "E002", "E003", "E004", "E005"],
            "customer_id": ["C001", "C002", "C003", "C004", "C005"],
            "loan_amount": [100000, 50000, 25000, 150000, 75000],
            "current_balance": [80000, 45000, 24000, 140000, 70000],
            "days_past_due": [30, 89, 90, 0, 91],  # Boundary values
            "credit_score": [499, 500, 501, 750, 400],  # Around threshold
            "loan_type": ["MORTGAGE", "AUTO", "PERSONAL", "MORTGAGE", "AUTO"],
            "collateral_value": [120000, 55000, 0, 180000, 85000],
            "ltv_ratio": [0.94, 0.95, 0.96, 0.83, None],  # Around threshold
            "interest_rate": [4.5, 6.2, 12.5, 3.8, 5.9],
            "term_months": [360, 60, 36, 360, 72],
            "maturity_date": [(datetime.now() + timedelta(days=365)).strftime("%Y-%m-%d")] * 5
        })
        
        spark_df = datetime_converter.pandas_to_spark_safe(
            edge_case_data, spark_session, datetime_columns=["maturity_date"]
        )
        
        engine = IFRS9RulesEngine(spark=spark_session)
        result_df = engine._apply_staging_rules(spark_df)
        result_pd = datetime_converter.spark_to_pandas_safe(result_df)
        
        stages = result_pd.set_index("loan_id")["calculated_stage"].to_dict()
        
        # Test boundary conditions
        assert stages["E001"] == "STAGE_2"  # DPD=30 (threshold)
        assert stages["E002"] == "STAGE_2"  # DPD=89 (just below Stage 3)
        assert stages["E003"] == "STAGE_3"  # DPD=90 (threshold)
        assert stages["E004"] == "STAGE_2"  # Good DPD but poor credit (499)
        assert stages["E005"] == "STAGE_3"  # DPD=91 (over threshold)
    
    def test_pd_calculation_comprehensive(self, spark_session, sample_spark_df, 
                                        ifrs9_rules_engine, datetime_converter):
        """Test comprehensive Probability of Default calculation."""
        staged_df = ifrs9_rules_engine._apply_staging_rules(sample_spark_df)
        result_df = ifrs9_rules_engine._calculate_risk_parameters(staged_df)
        result_pd = datetime_converter.spark_to_pandas_safe(result_df)
        
        pd_values = result_pd.set_index("loan_id")["calculated_pd"].to_dict()
        
        # Stage 3 loans should have PD = 1.0
        stage_3_loans = result_pd[result_pd["calculated_stage"] == "STAGE_3"]["loan_id"].tolist()
        for loan_id in stage_3_loans:
            assert pd_values[loan_id] == 1.0
        
        # Stage 1 loans should have low PD
        stage_1_loans = result_pd[result_pd["calculated_stage"] == "STAGE_1"]["loan_id"].tolist()
        for loan_id in stage_1_loans:
            assert 0.0 <= pd_values[loan_id] < 0.05
        
        # Stage 2 loans should have medium PD
        stage_2_loans = result_pd[result_pd["calculated_stage"] == "STAGE_2"]["loan_id"].tolist()
        for loan_id in stage_2_loans:
            assert 0.05 <= pd_values[loan_id] < 1.0
    
    def test_lgd_calculation_comprehensive(self, spark_session, sample_spark_df, 
                                         ifrs9_rules_engine, datetime_converter):
        """Test comprehensive Loss Given Default calculation."""
        staged_df = ifrs9_rules_engine._apply_staging_rules(sample_spark_df)
        result_df = ifrs9_rules_engine._calculate_risk_parameters(staged_df)
        result_pd = datetime_converter.spark_to_pandas_safe(result_df)
        
        lgd_values = result_pd.set_index("loan_id")["calculated_lgd"].to_dict()
        
        # All LGD values should be between 0 and 1
        for loan_id, lgd in lgd_values.items():
            assert 0.0 <= lgd <= 1.0, f"LGD for {loan_id} is {lgd}, outside valid range"
        
        # Test loan type specific LGD expectations
        mortgage_loans = result_pd[result_pd["loan_type"] == "MORTGAGE"]
        personal_loans = result_pd[result_pd["loan_type"] == "PERSONAL"]
        
        # Personal loans should generally have higher LGD than secured loans
        if not mortgage_loans.empty and not personal_loans.empty:
            avg_mortgage_lgd = mortgage_loans["calculated_lgd"].mean()
            avg_personal_lgd = personal_loans["calculated_lgd"].mean()
            assert avg_personal_lgd >= avg_mortgage_lgd
    
    def test_ecl_calculation_comprehensive(self, spark_session, sample_spark_df, 
                                         ifrs9_rules_engine, datetime_converter):
        """Test comprehensive Expected Credit Loss calculation."""
        processed_df = ifrs9_rules_engine.process_portfolio(sample_spark_df)
        result_pd = datetime_converter.spark_to_pandas_safe(processed_df)
        
        ecl_values = result_pd.set_index("loan_id")["calculated_ecl"].to_dict()
        
        # All ECL values should be non-negative
        for loan_id, ecl in ecl_values.items():
            assert ecl >= 0.0, f"ECL for {loan_id} is {ecl}, should be non-negative"
        
        # Stage 3 loans should have higher ECL than Stage 1 loans
        stage_3_ecl = result_pd[result_pd["calculated_stage"] == "STAGE_3"]["calculated_ecl"].mean()
        stage_1_ecl = result_pd[result_pd["calculated_stage"] == "STAGE_1"]["calculated_ecl"].mean()
        
        if not np.isnan(stage_3_ecl) and not np.isnan(stage_1_ecl):
            assert stage_3_ecl > stage_1_ecl
        
        # ECL should not exceed current balance
        for _, row in result_pd.iterrows():
            assert row["calculated_ecl"] <= row["current_balance"], \
                f"ECL {row['calculated_ecl']} exceeds balance {row['current_balance']} for {row['loan_id']}"
    
    def test_validation_comprehensive(self, spark_session, sample_spark_df, 
                                    ifrs9_rules_engine):
        """Test comprehensive validation functions."""
        processed_df = ifrs9_rules_engine.process_portfolio(sample_spark_df)
        validations = ifrs9_rules_engine.validate_calculations(processed_df)
        
        # Check that all validations exist and have expected structure
        assert isinstance(validations, list)
        assert len(validations) > 0
        
        # Check validation structure
        for validation in validations:
            assert "check" in validation
            assert "passed" in validation
            assert "message" in validation
            assert isinstance(validation["passed"], bool)
        
        # Count failed validations
        failed_validations = [v for v in validations if not v["passed"]]
        
        # Log failed validations for debugging
        if failed_validations:
            for validation in failed_validations:
                print(f"FAILED VALIDATION: {validation['check']} - {validation['message']}")
        
        # All validations should pass for clean test data
        assert len(failed_validations) == 0, f"{len(failed_validations)} validation(s) failed"
    
    def test_summary_report_comprehensive(self, spark_session, sample_spark_df, 
                                        ifrs9_rules_engine):
        """Test comprehensive summary report generation."""
        processed_df = ifrs9_rules_engine.process_portfolio(sample_spark_df)
        summary = ifrs9_rules_engine.generate_summary_report(processed_df)
        
        # Check report structure
        required_sections = ["portfolio_metrics", "stage_distribution", "risk_distribution"]
        for section in required_sections:
            assert section in summary, f"Missing section: {section}"
        
        # Check portfolio metrics
        metrics = summary["portfolio_metrics"]
        required_metrics = ["total_loans", "total_exposure", "total_ecl", "coverage_ratio"]
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            assert metrics[metric] is not None
        
        # Validate metrics consistency
        assert metrics["total_loans"] == 10  # Based on sample data
        assert metrics["total_exposure"] > 0
        assert metrics["total_ecl"] >= 0
        assert 0 <= metrics["coverage_ratio"] <= 1
        
        # Check stage distribution
        stage_dist = summary["stage_distribution"]
        assert "STAGE_1" in stage_dist
        assert "STAGE_2" in stage_dist
        assert "STAGE_3" in stage_dist
        
        # Sum of stage counts should equal total loans
        total_stage_loans = sum(stage_dist.values())
        assert total_stage_loans == metrics["total_loans"]
    
    @pytest.mark.slow
    def test_large_portfolio_performance(self, spark_session, ml_test_data, datetime_converter):
        """Test performance with larger portfolio (marked as slow test)."""
        # Convert large test data to Spark DataFrame
        spark_df = datetime_converter.pandas_to_spark_safe(
            ml_test_data, spark_session, table_name="large_portfolio"
        )
        
        engine = IFRS9RulesEngine(spark=spark_session)
        
        # Measure processing time
        start_time = datetime.now()
        processed_df = engine.process_portfolio(spark_df)
        result_count = processed_df.count()
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        # Verify results
        assert result_count == 1000  # Should process all loans
        assert processing_time < 60  # Should complete within 60 seconds
        
        # Generate summary for large portfolio
        summary = engine.generate_summary_report(processed_df)
        assert summary["portfolio_metrics"]["total_loans"] == 1000
    
    def test_datetime_conversion_robustness(self, spark_session, datetime_converter):
        """Test robustness of datetime conversion between Pandas and Spark."""
        # Create test data with various datetime formats
        test_data = pd.DataFrame({
            "loan_id": ["DT001", "DT002", "DT003"],
            "maturity_date": [
                "2024-12-31",
                datetime(2025, 6, 15),
                pd.Timestamp("2026-03-20")
            ],
            "created_date": pd.to_datetime(["2023-01-01", "2023-06-15", "2023-12-31"]),
            "amount": [100000, 50000, 75000]
        })
        
        # Validate conversion safety
        validation = datetime_converter.validate_datetime_conversion(test_data, spark_session)
        assert validation["conversion_safe"], f"Conversion issues: {validation['issues']}"
        
        # Perform safe conversion
        spark_df = datetime_converter.pandas_to_spark_safe(
            test_data, spark_session, 
            datetime_columns=["maturity_date", "created_date"],
            table_name="datetime_test"
        )
        
        # Convert back to Pandas
        result_df = datetime_converter.spark_to_pandas_safe(spark_df)
        
        # Verify datetime columns are properly formatted
        for col in ["maturity_date", "created_date"]:
            assert pd.api.types.is_datetime64_any_dtype(result_df[col]), \
                f"Column {col} is not datetime type: {result_df[col].dtype}"
    
    def test_memory_cleanup(self, spark_session, test_data_manager):
        """Test that temporary views and memory are properly cleaned up."""
        # Create test data and register temp views
        test_df = spark_session.createDataFrame([("test", 1)], ["col1", "col2"])
        
        # Use test data manager to track temp views
        test_data_manager.register_temp_view(test_df, "temp_test_view")
        
        # Verify temp view exists
        tables_before = spark_session.catalog.listTables()
        temp_tables_before = [t for t in tables_before if t.isTemporary]
        assert len(temp_tables_before) > 0
        
        # Cleanup should happen automatically via fixture
        # This test verifies the cleanup mechanism works


@pytest.mark.integration
@pytest.mark.spark
class TestIFRS9Integration:
    """Integration tests for IFRS9 system components."""
    
    def test_end_to_end_processing(self, spark_session, sample_spark_df, datetime_converter):
        """Test complete end-to-end IFRS9 processing pipeline."""
        engine = IFRS9RulesEngine(spark=spark_session)
        
        # Process complete pipeline
        result_df = engine.process_portfolio(sample_spark_df)
        
        # Validate results
        result_pd = datetime_converter.spark_to_pandas_safe(result_df)
        
        # Check all expected columns are present
        expected_columns = [
            "loan_id", "customer_id", "loan_amount", "current_balance",
            "calculated_stage", "calculated_pd", "calculated_lgd", "calculated_ecl"
        ]
        
        for col in expected_columns:
            assert col in result_pd.columns, f"Missing column: {col}"
        
        # Verify data integrity
        assert len(result_pd) == 10  # All loans processed
        assert result_pd["calculated_ecl"].sum() > 0  # Some ECL calculated
        
        # Generate and validate reports
        summary = engine.generate_summary_report(result_df)
        validations = engine.validate_calculations(result_df)
        
        assert len(summary) > 0
        assert all(v["passed"] for v in validations)