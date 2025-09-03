#!/usr/bin/env python3
"""
IFRS9 Production Readiness Validator
===================================

This module provides comprehensive validation framework for production readiness testing
of the IFRS9 Risk Management System. It validates datasets against business rules,
regulatory requirements, and technical specifications.

Features:
- 6-tier validation framework
- IFRS9 compliance checking
- ML model readiness validation
- Performance benchmarking
- Integration testing support
- Regulatory audit preparation
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
import great_expectations as ge
from pandera import Column, DataFrameSchema, Check
import shap

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


@dataclass
class ValidationResult:
    """Container for validation results."""
    test_name: str
    status: str  # PASSED, FAILED, WARNING
    score: float
    details: Dict[str, Any]
    recommendations: List[str]
    timestamp: datetime


@dataclass
class ValidationConfig:
    """Configuration for validation parameters."""
    completeness_threshold: float = 99.0
    consistency_threshold: float = 95.0
    distribution_tolerance: float = 0.1
    correlation_threshold: float = 0.05
    ifrs9_compliance_threshold: float = 100.0
    ml_readiness_threshold: float = 90.0


class ProductionReadinessValidator:
    """
    Comprehensive validation framework for IFRS9 production readiness.
    
    Implements 6-tier validation:
    1. Completeness - all required fields populated
    2. Consistency - internal logic and relationships maintained
    3. Distribution - realistic statistical properties
    4. Correlation - proper relationships between variables
    5. IFRS9 Compliance - regulatory requirements met
    6. ML-Readiness - proper feature engineering and data quality
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        """Initialize the production readiness validator."""
        self.config = config or ValidationConfig()
        self.logger = self._setup_logging()
        
        # Initialize validation schemas
        self._setup_validation_schemas()
        
        # Validation results storage
        self.validation_results: List[ValidationResult] = []
        self.overall_score = 0.0
        self.compliance_status = "UNKNOWN"
        
        # IFRS9 business rules
        self.ifrs9_rules = self._load_ifrs9_rules()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _setup_validation_schemas(self):
        """Setup Pandera schemas for data validation."""
        # Loan Portfolio Schema
        self.loan_schema = DataFrameSchema({
            "loan_id": Column(str, nullable=False),
            "customer_id": Column(str, nullable=False),
            "loan_amount": Column(float, Check.greater_than(0)),
            "interest_rate": Column(float, Check.in_range(0, 1)),
            "term_months": Column(int, Check.greater_than(0)),
            "origination_date": Column(pd.Timestamp, nullable=False),
            "current_balance": Column(float, Check.greater_than_or_equal_to(0)),
            "days_past_due": Column(int, Check.greater_than_or_equal_to(0)),
            "provision_stage": Column(int, Check.isin([1, 2, 3])),
            "credit_score": Column(int, Check.in_range(300, 999)),
            "pd_12m": Column(float, Check.in_range(0, 1)),
            "pd_lifetime": Column(float, Check.in_range(0, 1)),
            "lgd": Column(float, Check.in_range(0, 1)),
            "ead": Column(float, Check.greater_than_or_equal_to(0)),
            "ecl": Column(float, Check.greater_than_or_equal_to(0))
        })
        
        # Payment History Schema
        self.payment_schema = DataFrameSchema({
            "loan_id": Column(str, nullable=False),
            "payment_date": Column(pd.Timestamp, nullable=False),
            "scheduled_payment": Column(float, Check.greater_than(0)),
            "actual_payment": Column(float, Check.greater_than_or_equal_to(0)),
            "payment_status": Column(str, Check.isin(["PAID", "MISSED", "PARTIAL"])),
            "days_late": Column(int, Check.greater_than_or_equal_to(0))
        })
    
    def _load_ifrs9_rules(self) -> Dict[str, Any]:
        """Load IFRS9 business rules and thresholds."""
        return {
            "staging_rules": {
                "stage_1_dpd_threshold": 0,
                "stage_2_dpd_threshold": 30,
                "stage_3_dpd_threshold": 90,
                "significant_increase_triggers": [
                    "credit_deterioration",
                    "payment_delinquency", 
                    "forbearance",
                    "watchlist_inclusion"
                ]
            },
            "risk_parameters": {
                "pd_ranges": {"min": 0.0, "max": 1.0},
                "lgd_ranges": {"min": 0.0, "max": 1.0},
                "ead_minimum": 0.0,
                "correlation_limits": {
                    "pd_credit_score": {"min": -0.8, "max": -0.3},
                    "lgd_ltv_ratio": {"min": 0.2, "max": 0.8}
                }
            },
            "distribution_requirements": {
                "stage_1_percentage": {"min": 80.0, "max": 90.0},
                "stage_2_percentage": {"min": 8.0, "max": 15.0},
                "stage_3_percentage": {"min": 2.0, "max": 7.0}
            }
        }
    
    def validate_comprehensive_dataset(self, 
                                     loan_portfolio: pd.DataFrame,
                                     payment_history: Optional[pd.DataFrame] = None,
                                     macro_data: Optional[pd.DataFrame] = None,
                                     stage_transitions: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Perform comprehensive validation of production dataset.
        
        Args:
            loan_portfolio: Main loan portfolio dataset
            payment_history: Payment history dataset (optional)
            macro_data: Macroeconomic data (optional)
            stage_transitions: Stage transition history (optional)
            
        Returns:
            Comprehensive validation results
        """
        self.logger.info("🔍 Starting comprehensive dataset validation")
        
        validation_start = datetime.now()
        
        # Tier 1: Completeness Validation
        self.logger.info("📊 Tier 1: Completeness Validation")
        completeness_result = self._validate_completeness(loan_portfolio, payment_history, macro_data)
        self.validation_results.append(completeness_result)
        
        # Tier 2: Consistency Validation
        self.logger.info("🔧 Tier 2: Consistency Validation")
        consistency_result = self._validate_consistency(loan_portfolio, payment_history)
        self.validation_results.append(consistency_result)
        
        # Tier 3: Distribution Validation
        self.logger.info("📈 Tier 3: Distribution Validation")
        distribution_result = self._validate_distributions(loan_portfolio)
        self.validation_results.append(distribution_result)
        
        # Tier 4: Correlation Validation
        self.logger.info("🔗 Tier 4: Correlation Validation")
        correlation_result = self._validate_correlations(loan_portfolio)
        self.validation_results.append(correlation_result)
        
        # Tier 5: IFRS9 Compliance Validation
        self.logger.info("⚖️  Tier 5: IFRS9 Compliance Validation")
        ifrs9_result = self._validate_ifrs9_compliance(loan_portfolio, stage_transitions)
        self.validation_results.append(ifrs9_result)
        
        # Tier 6: ML-Readiness Validation
        self.logger.info("🤖 Tier 6: ML-Readiness Validation")
        ml_readiness_result = self._validate_ml_readiness(loan_portfolio)
        self.validation_results.append(ml_readiness_result)
        
        validation_end = datetime.now()
        validation_time = (validation_end - validation_start).total_seconds()
        
        # Calculate overall results
        self._calculate_overall_results()
        
        # Generate comprehensive report
        report = self._generate_comprehensive_report(validation_time)
        
        self.logger.info(f"✅ Validation completed in {validation_time:.1f} seconds")
        self.logger.info(f"Overall Score: {self.overall_score:.1f}%")
        self.logger.info(f"Compliance Status: {self.compliance_status}")
        
        return report
    
    def _validate_completeness(self, 
                              loan_portfolio: pd.DataFrame,
                              payment_history: Optional[pd.DataFrame],
                              macro_data: Optional[pd.DataFrame]) -> ValidationResult:
        """Tier 1: Validate data completeness."""
        checks = []
        recommendations = []
        
        # Required fields completeness
        required_fields = [
            "loan_id", "customer_id", "loan_amount", "interest_rate", 
            "origination_date", "provision_stage", "credit_score",
            "pd_12m", "lgd", "ead", "ecl"
        ]
        
        missing_fields = [field for field in required_fields if field not in loan_portfolio.columns]
        if missing_fields:
            checks.append({
                "check": "required_fields_present",
                "status": "FAILED",
                "score": 0.0,
                "details": f"Missing required fields: {missing_fields}"
            })
            recommendations.append(f"Add missing required fields: {missing_fields}")
        else:
            checks.append({
                "check": "required_fields_present", 
                "status": "PASSED",
                "score": 100.0,
                "details": "All required fields present"
            })
        
        # Data completeness percentage
        for field in required_fields:
            if field in loan_portfolio.columns:
                completeness_pct = (1 - loan_portfolio[field].isnull().sum() / len(loan_portfolio)) * 100
                
                if completeness_pct >= self.config.completeness_threshold:
                    status = "PASSED"
                elif completeness_pct >= 90.0:
                    status = "WARNING"
                else:
                    status = "FAILED"
                
                checks.append({
                    "check": f"completeness_{field}",
                    "status": status,
                    "score": completeness_pct,
                    "details": f"{field} completeness: {completeness_pct:.1f}%"
                })
                
                if completeness_pct < self.config.completeness_threshold:
                    recommendations.append(f"Improve {field} completeness (currently {completeness_pct:.1f}%)")
        
        # Record count validation
        min_records = 1000
        if len(loan_portfolio) < min_records:
            checks.append({
                "check": "minimum_record_count",
                "status": "FAILED", 
                "score": 0.0,
                "details": f"Insufficient records: {len(loan_portfolio)} < {min_records}"
            })
            recommendations.append(f"Increase dataset size to minimum {min_records} records")
        else:
            checks.append({
                "check": "minimum_record_count",
                "status": "PASSED",
                "score": 100.0,
                "details": f"Sufficient records: {len(loan_portfolio):,}"
            })
        
        # Calculate overall completeness score
        scores = [check["score"] for check in checks]
        overall_score = np.mean(scores)
        
        status = "PASSED" if overall_score >= self.config.completeness_threshold else "FAILED"
        
        return ValidationResult(
            test_name="completeness_validation",
            status=status,
            score=overall_score,
            details={"checks": checks, "record_count": len(loan_portfolio)},
            recommendations=recommendations,
            timestamp=datetime.now()
        )
    
    def _validate_consistency(self, 
                             loan_portfolio: pd.DataFrame,
                             payment_history: Optional[pd.DataFrame]) -> ValidationResult:
        """Tier 2: Validate internal consistency and business logic."""
        checks = []
        recommendations = []
        
        # Schema validation using Pandera
        try:
            self.loan_schema.validate(loan_portfolio)
            checks.append({
                "check": "schema_validation",
                "status": "PASSED",
                "score": 100.0,
                "details": "All schema constraints satisfied"
            })
        except Exception as e:
            checks.append({
                "check": "schema_validation",
                "status": "FAILED",
                "score": 0.0,
                "details": f"Schema validation failed: {str(e)}"
            })
            recommendations.append("Fix schema validation errors")
        
        # Business rule consistency
        # Rule: Current balance <= Original loan amount
        if "current_balance" in loan_portfolio.columns and "loan_amount" in loan_portfolio.columns:
            balance_check = (loan_portfolio["current_balance"] <= loan_portfolio["loan_amount"]).mean() * 100
            checks.append({
                "check": "balance_consistency",
                "status": "PASSED" if balance_check >= 95.0 else "FAILED",
                "score": balance_check,
                "details": f"Current balance <= loan amount: {balance_check:.1f}%"
            })
        
        # Rule: Stage progression logic
        if "provision_stage" in loan_portfolio.columns and "days_past_due" in loan_portfolio.columns:
            stage_logic_violations = 0
            
            # Stage 1 should have DPD = 0 (mostly)
            stage_1_dpd_ok = (loan_portfolio[
                loan_portfolio["provision_stage"] == 1
            ]["days_past_due"] == 0).mean()
            
            # Stage 2 should have DPD > 0 but < 90
            stage_2_loans = loan_portfolio[loan_portfolio["provision_stage"] == 2]
            if len(stage_2_loans) > 0:
                stage_2_dpd_ok = ((stage_2_loans["days_past_due"] > 0) & 
                                 (stage_2_loans["days_past_due"] < 90)).mean()
            else:
                stage_2_dpd_ok = 1.0
            
            # Stage 3 should have DPD >= 90
            stage_3_loans = loan_portfolio[loan_portfolio["provision_stage"] == 3]
            if len(stage_3_loans) > 0:
                stage_3_dpd_ok = (stage_3_loans["days_past_due"] >= 90).mean()
            else:
                stage_3_dpd_ok = 1.0
            
            avg_stage_consistency = np.mean([stage_1_dpd_ok, stage_2_dpd_ok, stage_3_dpd_ok]) * 100
            
            checks.append({
                "check": "stage_dpd_consistency",
                "status": "PASSED" if avg_stage_consistency >= 80.0 else "FAILED",
                "score": avg_stage_consistency,
                "details": f"Stage-DPD consistency: {avg_stage_consistency:.1f}%"
            })
        
        # Rule: PD relationships (12m <= lifetime)
        if "pd_12m" in loan_portfolio.columns and "pd_lifetime" in loan_portfolio.columns:
            pd_relationship_ok = (loan_portfolio["pd_12m"] <= loan_portfolio["pd_lifetime"]).mean() * 100
            checks.append({
                "check": "pd_relationship_consistency",
                "status": "PASSED" if pd_relationship_ok >= 95.0 else "WARNING",
                "score": pd_relationship_ok,
                "details": f"PD 12m <= PD lifetime: {pd_relationship_ok:.1f}%"
            })
        
        # Rule: ECL calculation consistency
        if all(col in loan_portfolio.columns for col in ["ead", "pd_12m", "pd_lifetime", "lgd", "ecl", "provision_stage"]):
            ecl_calculations = []
            
            for _, loan in loan_portfolio.sample(min(1000, len(loan_portfolio))).iterrows():
                if loan["provision_stage"] == 1:
                    expected_ecl = loan["ead"] * loan["pd_12m"] * loan["lgd"]
                else:
                    expected_ecl = loan["ead"] * loan["pd_lifetime"] * loan["lgd"]
                
                actual_ecl = loan["ecl"]
                
                # Allow 5% tolerance for rounding
                if abs(expected_ecl - actual_ecl) / max(expected_ecl, 0.01) <= 0.05:
                    ecl_calculations.append(True)
                else:
                    ecl_calculations.append(False)
            
            ecl_consistency = np.mean(ecl_calculations) * 100
            checks.append({
                "check": "ecl_calculation_consistency",
                "status": "PASSED" if ecl_consistency >= 95.0 else "FAILED",
                "score": ecl_consistency,
                "details": f"ECL calculation consistency: {ecl_consistency:.1f}%"
            })
        
        # Calculate overall consistency score
        scores = [check["score"] for check in checks]
        overall_score = np.mean(scores)
        
        status = "PASSED" if overall_score >= self.config.consistency_threshold else "FAILED"
        
        return ValidationResult(
            test_name="consistency_validation",
            status=status,
            score=overall_score,
            details={"checks": checks},
            recommendations=recommendations,
            timestamp=datetime.now()
        )
    
    def _validate_distributions(self, loan_portfolio: pd.DataFrame) -> ValidationResult:
        """Tier 3: Validate statistical distributions."""
        checks = []
        recommendations = []
        
        # Credit score distribution validation
        if "credit_score" in loan_portfolio.columns:
            credit_scores = loan_portfolio["credit_score"].dropna()
            
            # Should be roughly normal distribution with mean around 650-700
            mean_score = credit_scores.mean()
            std_score = credit_scores.std()
            
            score_check = 100.0 if 600 <= mean_score <= 750 else 60.0
            checks.append({
                "check": "credit_score_distribution",
                "status": "PASSED" if score_check >= 80.0 else "WARNING",
                "score": score_check,
                "details": f"Credit score mean: {mean_score:.0f}, std: {std_score:.0f}"
            })
        
        # Loan amount distribution
        if "loan_amount" in loan_portfolio.columns:
            loan_amounts = loan_portfolio["loan_amount"].dropna()
            
            # Check for realistic ranges
            median_amount = loan_amounts.median()
            q95_amount = loan_amounts.quantile(0.95)
            
            # Reasonable ranges vary by product type, but general check
            amount_check = 100.0 if 10000 <= median_amount <= 500000 else 70.0
            checks.append({
                "check": "loan_amount_distribution",
                "status": "PASSED" if amount_check >= 80.0 else "WARNING",
                "score": amount_check,
                "details": f"Median loan amount: €{median_amount:,.0f}, 95th percentile: €{q95_amount:,.0f}"
            })
        
        # DPD distribution validation
        if "days_past_due" in loan_portfolio.columns:
            dpd_values = loan_portfolio["days_past_due"].dropna()
            
            # Most loans should be current (DPD = 0)
            current_loans_pct = (dpd_values == 0).mean() * 100
            
            dpd_check = 100.0 if current_loans_pct >= 80.0 else 60.0
            checks.append({
                "check": "dpd_distribution", 
                "status": "PASSED" if dpd_check >= 80.0 else "WARNING",
                "score": dpd_check,
                "details": f"Current loans (DPD=0): {current_loans_pct:.1f}%"
            })
        
        # Interest rate distribution
        if "interest_rate" in loan_portfolio.columns:
            interest_rates = loan_portfolio["interest_rate"].dropna()
            
            mean_rate = interest_rates.mean()
            
            # Reasonable rates typically 1% to 15%
            rate_check = 100.0 if 0.01 <= mean_rate <= 0.15 else 50.0
            checks.append({
                "check": "interest_rate_distribution",
                "status": "PASSED" if rate_check >= 80.0 else "WARNING", 
                "score": rate_check,
                "details": f"Mean interest rate: {mean_rate:.1%}"
            })
        
        # Calculate overall distribution score
        scores = [check["score"] for check in checks]
        overall_score = np.mean(scores)
        
        status = "PASSED" if overall_score >= 80.0 else "WARNING"
        
        return ValidationResult(
            test_name="distribution_validation",
            status=status,
            score=overall_score,
            details={"checks": checks},
            recommendations=recommendations,
            timestamp=datetime.now()
        )
    
    def _validate_correlations(self, loan_portfolio: pd.DataFrame) -> ValidationResult:
        """Tier 4: Validate expected correlations between variables."""
        checks = []
        recommendations = []
        
        numeric_columns = loan_portfolio.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) < 2:
            return ValidationResult(
                test_name="correlation_validation",
                status="WARNING",
                score=50.0,
                details={"error": "Insufficient numeric columns for correlation analysis"},
                recommendations=["Ensure numeric variables are properly encoded"],
                timestamp=datetime.now()
            )
        
        # Expected correlations based on banking domain knowledge
        expected_correlations = [
            {
                "var1": "credit_score", 
                "var2": "pd_12m",
                "expected_direction": "negative",
                "min_strength": 0.3,
                "description": "Credit score should negatively correlate with PD"
            },
            {
                "var1": "days_past_due",
                "var2": "provision_stage", 
                "expected_direction": "positive",
                "min_strength": 0.5,
                "description": "DPD should positively correlate with provision stage"
            },
            {
                "var1": "ltv_ratio",
                "var2": "lgd",
                "expected_direction": "positive",
                "min_strength": 0.2,
                "description": "LTV ratio should positively correlate with LGD"
            },
            {
                "var1": "loan_amount",
                "var2": "ead",
                "expected_direction": "positive", 
                "min_strength": 0.7,
                "description": "Loan amount should strongly correlate with EAD"
            }
        ]
        
        correlation_matrix = loan_portfolio[numeric_columns].corr()
        
        for expected_corr in expected_correlations:
            var1, var2 = expected_corr["var1"], expected_corr["var2"]
            
            if var1 in correlation_matrix.columns and var2 in correlation_matrix.columns:
                actual_corr = correlation_matrix.loc[var1, var2]
                expected_direction = expected_corr["expected_direction"]
                min_strength = expected_corr["min_strength"]
                
                # Check direction and strength
                if expected_direction == "positive":
                    is_correct_direction = actual_corr > 0
                    strength_met = abs(actual_corr) >= min_strength
                else:  # negative
                    is_correct_direction = actual_corr < 0
                    strength_met = abs(actual_corr) >= min_strength
                
                if is_correct_direction and strength_met:
                    status = "PASSED"
                    score = 100.0
                elif is_correct_direction:
                    status = "WARNING"
                    score = 70.0
                else:
                    status = "FAILED"
                    score = 30.0
                
                checks.append({
                    "check": f"correlation_{var1}_{var2}",
                    "status": status,
                    "score": score,
                    "details": f"{expected_corr['description']}: {actual_corr:.3f}"
                })
                
                if not (is_correct_direction and strength_met):
                    recommendations.append(f"Review {var1}-{var2} relationship (correlation: {actual_corr:.3f})")
        
        # Check for unrealistic correlations (too high)
        high_correlations = []
        for i, col1 in enumerate(numeric_columns):
            for col2 in numeric_columns[i+1:]:
                corr_value = correlation_matrix.loc[col1, col2]
                if abs(corr_value) > 0.95 and col1 != col2:
                    high_correlations.append((col1, col2, corr_value))
        
        if high_correlations:
            checks.append({
                "check": "multicollinearity_detection",
                "status": "WARNING",
                "score": 60.0,
                "details": f"High correlations detected: {high_correlations[:3]}"
            })
            recommendations.append("Review variables with very high correlations for potential multicollinearity")
        else:
            checks.append({
                "check": "multicollinearity_detection",
                "status": "PASSED",
                "score": 100.0,
                "details": "No extreme multicollinearity detected"
            })
        
        # Calculate overall correlation score
        scores = [check["score"] for check in checks]
        overall_score = np.mean(scores) if scores else 50.0
        
        status = "PASSED" if overall_score >= 80.0 else "WARNING"
        
        return ValidationResult(
            test_name="correlation_validation",
            status=status,
            score=overall_score,
            details={"checks": checks, "correlation_summary": correlation_matrix.describe().to_dict()},
            recommendations=recommendations,
            timestamp=datetime.now()
        )
    
    def _validate_ifrs9_compliance(self, 
                                  loan_portfolio: pd.DataFrame,
                                  stage_transitions: Optional[pd.DataFrame]) -> ValidationResult:
        """Tier 5: Validate IFRS9 regulatory compliance."""
        checks = []
        recommendations = []
        
        # Stage distribution validation
        if "provision_stage" in loan_portfolio.columns:
            stage_dist = loan_portfolio["provision_stage"].value_counts(normalize=True) * 100
            
            rules = self.ifrs9_rules["distribution_requirements"]
            
            for stage in [1, 2, 3]:
                actual_pct = stage_dist.get(stage, 0)
                min_pct = rules[f"stage_{stage}_percentage"]["min"]
                max_pct = rules[f"stage_{stage}_percentage"]["max"]
                
                if min_pct <= actual_pct <= max_pct:
                    status = "PASSED"
                    score = 100.0
                elif actual_pct < min_pct * 0.5 or actual_pct > max_pct * 1.5:
                    status = "FAILED"
                    score = 0.0
                else:
                    status = "WARNING"
                    score = 60.0
                
                checks.append({
                    "check": f"stage_{stage}_distribution",
                    "status": status,
                    "score": score,
                    "details": f"Stage {stage}: {actual_pct:.1f}% (expected: {min_pct}-{max_pct}%)"
                })
        
        # DPD threshold compliance
        if all(col in loan_portfolio.columns for col in ["provision_stage", "days_past_due"]):
            staging_rules = self.ifrs9_rules["staging_rules"]
            
            # Stage 1: DPD should be <= stage_2_threshold
            stage_1_loans = loan_portfolio[loan_portfolio["provision_stage"] == 1]
            stage_1_compliance = (stage_1_loans["days_past_due"] <= staging_rules["stage_2_dpd_threshold"]).mean() * 100
            
            # Stage 2: DPD should be > stage_1_threshold and < stage_3_threshold
            stage_2_loans = loan_portfolio[loan_portfolio["provision_stage"] == 2]
            if len(stage_2_loans) > 0:
                stage_2_compliance = ((stage_2_loans["days_past_due"] > staging_rules["stage_1_dpd_threshold"]) &
                                    (stage_2_loans["days_past_due"] < staging_rules["stage_3_dpd_threshold"])).mean() * 100
            else:
                stage_2_compliance = 100.0
            
            # Stage 3: DPD should be >= stage_3_threshold
            stage_3_loans = loan_portfolio[loan_portfolio["provision_stage"] == 3]
            if len(stage_3_loans) > 0:
                stage_3_compliance = (stage_3_loans["days_past_due"] >= staging_rules["stage_3_dpd_threshold"]).mean() * 100
            else:
                stage_3_compliance = 100.0
            
            avg_dpd_compliance = np.mean([stage_1_compliance, stage_2_compliance, stage_3_compliance])
            
            checks.append({
                "check": "dpd_threshold_compliance",
                "status": "PASSED" if avg_dpd_compliance >= 95.0 else "FAILED",
                "score": avg_dpd_compliance,
                "details": f"DPD threshold compliance: {avg_dpd_compliance:.1f}%"
            })
        
        # Risk parameter validation
        risk_rules = self.ifrs9_rules["risk_parameters"]
        
        for param in ["pd_12m", "pd_lifetime", "lgd"]:
            if param in loan_portfolio.columns:
                param_values = loan_portfolio[param].dropna()
                
                min_val = risk_rules[f"{param.split('_')[0]}_ranges"]["min"]
                max_val = risk_rules[f"{param.split('_')[0]}_ranges"]["max"]
                
                in_range_pct = ((param_values >= min_val) & (param_values <= max_val)).mean() * 100
                
                checks.append({
                    "check": f"{param}_range_compliance",
                    "status": "PASSED" if in_range_pct >= 99.0 else "FAILED",
                    "score": in_range_pct,
                    "details": f"{param} in range [{min_val}, {max_val}]: {in_range_pct:.1f}%"
                })
        
        # ECL non-negative requirement
        if "ecl" in loan_portfolio.columns:
            ecl_non_negative = (loan_portfolio["ecl"] >= 0).mean() * 100
            checks.append({
                "check": "ecl_non_negative",
                "status": "PASSED" if ecl_non_negative == 100.0 else "FAILED",
                "score": ecl_non_negative,
                "details": f"ECL non-negative: {ecl_non_negative:.1f}%"
            })
        
        # Stage transition validation (if available)
        if stage_transitions is not None:
            # Check for forbidden transitions (e.g., Stage 3 to Stage 1 directly)
            forbidden_transitions = 0
            total_transitions = 0
            
            for _, transition in stage_transitions.iterrows():
                from_stage = transition.get("from_stage")
                to_stage = transition.get("to_stage")
                
                if from_stage and to_stage:
                    total_transitions += 1
                    # Direct Stage 3 to Stage 1 should be rare
                    if from_stage == 3 and to_stage == 1:
                        forbidden_transitions += 1
            
            if total_transitions > 0:
                forbidden_pct = (forbidden_transitions / total_transitions) * 100
                transition_compliance = 100.0 - forbidden_pct
                
                checks.append({
                    "check": "stage_transition_logic",
                    "status": "PASSED" if forbidden_pct <= 2.0 else "WARNING",
                    "score": transition_compliance,
                    "details": f"Forbidden transitions: {forbidden_pct:.1f}%"
                })
        
        # Calculate overall IFRS9 compliance score
        scores = [check["score"] for check in checks]
        overall_score = np.mean(scores)
        
        status = "PASSED" if overall_score >= self.config.ifrs9_compliance_threshold else "FAILED"
        
        if overall_score < 100.0:
            recommendations.append("Review IFRS9 staging logic and risk parameter calculations")
        
        return ValidationResult(
            test_name="ifrs9_compliance_validation", 
            status=status,
            score=overall_score,
            details={"checks": checks, "total_checks": len(checks)},
            recommendations=recommendations,
            timestamp=datetime.now()
        )
    
    def _validate_ml_readiness(self, loan_portfolio: pd.DataFrame) -> ValidationResult:
        """Tier 6: Validate ML model readiness."""
        checks = []
        recommendations = []
        
        # Feature completeness for ML
        ml_features = [
            "credit_score", "loan_amount", "interest_rate", "term_months",
            "current_balance", "days_past_due", "ltv_ratio", "dti_ratio"
        ]
        
        available_features = [f for f in ml_features if f in loan_portfolio.columns]
        feature_completeness = (len(available_features) / len(ml_features)) * 100
        
        checks.append({
            "check": "ml_feature_completeness",
            "status": "PASSED" if feature_completeness >= 80.0 else "FAILED",
            "score": feature_completeness,
            "details": f"ML features available: {len(available_features)}/{len(ml_features)}"
        })
        
        # Data type validation for ML
        numeric_features = loan_portfolio[available_features].select_dtypes(include=[np.number]).columns
        numeric_completeness = (len(numeric_features) / len(available_features)) * 100 if available_features else 0
        
        checks.append({
            "check": "numeric_feature_ratio",
            "status": "PASSED" if numeric_completeness >= 70.0 else "WARNING",
            "score": numeric_completeness,
            "details": f"Numeric features: {numeric_completeness:.1f}%"
        })
        
        # Missing value analysis for ML
        if available_features:
            missing_rates = loan_portfolio[available_features].isnull().mean() * 100
            max_missing_rate = missing_rates.max()
            
            ml_missing_score = max(0, 100 - max_missing_rate * 2)  # Penalize high missing rates
            
            checks.append({
                "check": "ml_missing_values",
                "status": "PASSED" if max_missing_rate <= 5.0 else "WARNING",
                "score": ml_missing_score,
                "details": f"Max missing rate: {max_missing_rate:.1f}%"
            })
        
        # Target variable analysis (provision_stage)
        if "provision_stage" in loan_portfolio.columns:
            stage_balance = loan_portfolio["provision_stage"].value_counts(normalize=True)
            min_class_ratio = stage_balance.min()
            
            # Check for severe class imbalance
            balance_score = min(100.0, min_class_ratio * 1000)  # At least 0.1% for each class
            
            checks.append({
                "check": "target_class_balance",
                "status": "PASSED" if min_class_ratio >= 0.01 else "WARNING",
                "score": balance_score,
                "details": f"Minimum class ratio: {min_class_ratio:.3f}"
            })
        
        # Feature scaling readiness
        if len(numeric_features) > 0:
            # Check if features have similar scales
            feature_scales = []
            for feature in numeric_features:
                if feature in loan_portfolio.columns:
                    feature_data = loan_portfolio[feature].dropna()
                    if len(feature_data) > 0:
                        scale = feature_data.std()
                        feature_scales.append(scale)
            
            if feature_scales:
                scale_ratio = max(feature_scales) / min(feature_scales) if min(feature_scales) > 0 else float('inf')
                
                # If scale ratio is very high, features need scaling
                scaling_score = max(0, 100 - np.log10(scale_ratio) * 20)
                
                checks.append({
                    "check": "feature_scaling_readiness",
                    "status": "PASSED" if scale_ratio <= 100 else "WARNING",
                    "score": scaling_score,
                    "details": f"Feature scale ratio: {scale_ratio:.1f}"
                })
                
                if scale_ratio > 100:
                    recommendations.append("Consider feature scaling/normalization for ML models")
        
        # Outlier detection for ML readiness
        outlier_scores = []
        for feature in numeric_features:
            if feature in loan_portfolio.columns:
                feature_data = loan_portfolio[feature].dropna()
                if len(feature_data) > 0:
                    # Use IQR method for outlier detection
                    Q1 = feature_data.quantile(0.25)
                    Q3 = feature_data.quantile(0.75)
                    IQR = Q3 - Q1
                    outlier_threshold = 1.5 * IQR
                    
                    outliers = ((feature_data < (Q1 - outlier_threshold)) | 
                               (feature_data > (Q3 + outlier_threshold))).sum()
                    outlier_rate = (outliers / len(feature_data)) * 100
                    
                    outlier_scores.append(100 - min(outlier_rate * 2, 100))  # Penalize high outlier rates
        
        if outlier_scores:
            avg_outlier_score = np.mean(outlier_scores)
            checks.append({
                "check": "outlier_analysis",
                "status": "PASSED" if avg_outlier_score >= 80.0 else "WARNING",
                "score": avg_outlier_score,
                "details": f"Outlier analysis score: {avg_outlier_score:.1f}"
            })
        
        # Calculate overall ML readiness score
        scores = [check["score"] for check in checks]
        overall_score = np.mean(scores)
        
        status = "PASSED" if overall_score >= self.config.ml_readiness_threshold else "WARNING"
        
        if overall_score < self.config.ml_readiness_threshold:
            recommendations.append("Improve data quality and feature engineering for ML model deployment")
        
        return ValidationResult(
            test_name="ml_readiness_validation",
            status=status,
            score=overall_score,
            details={"checks": checks, "available_features": available_features},
            recommendations=recommendations,
            timestamp=datetime.now()
        )
    
    def _calculate_overall_results(self):
        """Calculate overall validation results."""
        if not self.validation_results:
            self.overall_score = 0.0
            self.compliance_status = "FAILED"
            return
        
        # Weight the different validation tiers
        tier_weights = {
            "completeness_validation": 0.20,
            "consistency_validation": 0.25, 
            "distribution_validation": 0.15,
            "correlation_validation": 0.10,
            "ifrs9_compliance_validation": 0.25,
            "ml_readiness_validation": 0.05
        }
        
        weighted_scores = []
        for result in self.validation_results:
            weight = tier_weights.get(result.test_name, 1.0 / len(self.validation_results))
            weighted_scores.append(result.score * weight)
        
        self.overall_score = sum(weighted_scores)
        
        # Determine compliance status
        if self.overall_score >= 95.0:
            self.compliance_status = "EXCELLENT"
        elif self.overall_score >= 90.0:
            self.compliance_status = "PASSED"
        elif self.overall_score >= 80.0:
            self.compliance_status = "PASSED_WITH_WARNINGS"
        else:
            self.compliance_status = "FAILED"
    
    def _generate_comprehensive_report(self, validation_time: float) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        
        # Aggregate all recommendations
        all_recommendations = []
        for result in self.validation_results:
            all_recommendations.extend(result.recommendations)
        
        # Count status types
        status_counts = {"PASSED": 0, "FAILED": 0, "WARNING": 0}
        for result in self.validation_results:
            status_counts[result.status] = status_counts.get(result.status, 0) + 1
        
        report = {
            "validation_summary": {
                "timestamp": datetime.now().isoformat(),
                "validation_time_seconds": validation_time,
                "overall_score": self.overall_score,
                "compliance_status": self.compliance_status,
                "total_validations": len(self.validation_results),
                "status_breakdown": status_counts
            },
            "tier_results": {
                result.test_name: {
                    "status": result.status,
                    "score": result.score,
                    "details": result.details,
                    "recommendations": result.recommendations
                }
                for result in self.validation_results
            },
            "overall_recommendations": list(set(all_recommendations)),
            "production_readiness": {
                "ready_for_production": self.compliance_status in ["EXCELLENT", "PASSED"],
                "critical_issues": [
                    result.test_name for result in self.validation_results 
                    if result.status == "FAILED"
                ],
                "warnings": [
                    result.test_name for result in self.validation_results
                    if result.status == "WARNING"
                ]
            },
            "next_steps": self._generate_next_steps()
        }
        
        return report
    
    def _generate_next_steps(self) -> List[str]:
        """Generate recommended next steps based on validation results."""
        next_steps = []
        
        if self.compliance_status == "FAILED":
            next_steps.extend([
                "🚨 Address all critical validation failures before proceeding",
                "📊 Review data generation process for quality improvements",
                "⚖️  Ensure IFRS9 business rules are correctly implemented",
                "🔧 Fix data consistency and completeness issues"
            ])
        elif self.compliance_status == "PASSED_WITH_WARNINGS":
            next_steps.extend([
                "⚠️  Review and address validation warnings",
                "📈 Monitor data quality metrics in production",
                "🎯 Consider improvements to achieve excellent rating"
            ])
        else:
            next_steps.extend([
                "✅ Data validation passed - proceed with production deployment",
                "📊 Implement continuous monitoring of data quality metrics",
                "🚀 Execute production readiness testing with confidence"
            ])
        
        # Always include monitoring recommendations
        next_steps.extend([
            "📉 Set up automated data quality monitoring",
            "📋 Document validation results for audit purposes",
            "🔄 Schedule regular validation reviews"
        ])
        
        return next_steps


def main():
    """Main execution function for standalone validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='IFRS9 Production Readiness Validator')
    parser.add_argument('--dataset-path', type=str, required=True,
                       help='Path to loan portfolio dataset')
    parser.add_argument('--payment-history-path', type=str,
                       help='Path to payment history dataset')
    parser.add_argument('--output-dir', type=str, default='validation_results',
                       help='Output directory for validation results')
    
    args = parser.parse_args()
    
    print("🔍 IFRS9 Production Readiness Validator")
    print("=" * 60)
    
    # Initialize validator
    validator = ProductionReadinessValidator()
    
    # Load datasets
    print(f"📂 Loading dataset: {args.dataset_path}")
    if args.dataset_path.endswith('.parquet'):
        loan_portfolio = pd.read_parquet(args.dataset_path)
    else:
        loan_portfolio = pd.read_csv(args.dataset_path)
    
    payment_history = None
    if args.payment_history_path:
        print(f"📂 Loading payment history: {args.payment_history_path}")
        if args.payment_history_path.endswith('.parquet'):
            payment_history = pd.read_parquet(args.payment_history_path)
        else:
            payment_history = pd.read_csv(args.payment_history_path)
    
    # Run validation
    print(f"🚀 Running comprehensive validation on {len(loan_portfolio):,} loans...")
    results = validator.validate_comprehensive_dataset(
        loan_portfolio=loan_portfolio,
        payment_history=payment_history
    )
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "validation_report.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Overall Score: {validator.overall_score:.1f}%")
    print(f"Compliance Status: {validator.compliance_status}")
    print(f"Total Validations: {len(validator.validation_results)}")
    
    for result in validator.validation_results:
        status_emoji = {"PASSED": "✅", "WARNING": "⚠️", "FAILED": "❌"}
        emoji = status_emoji.get(result.status, "❓")
        print(f"{emoji} {result.test_name}: {result.score:.1f}%")
    
    print(f"\n📁 Full report saved to: {output_dir / 'validation_report.json'}")
    
    return 0 if validator.compliance_status in ["EXCELLENT", "PASSED"] else 1


if __name__ == "__main__":
    sys.exit(main())