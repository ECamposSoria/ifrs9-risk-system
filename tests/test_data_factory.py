"""Test data factory for IFRS9 system testing.

This module provides utilities for generating realistic and diverse test data
for comprehensive testing of the IFRS9 rules engine and related components.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class LoanType(Enum):
    """Loan type enumeration."""
    MORTGAGE = "MORTGAGE"
    AUTO = "AUTO"
    PERSONAL = "PERSONAL"
    BUSINESS = "BUSINESS"
    STUDENT = "STUDENT"


class ProvisionStage(Enum):
    """IFRS9 provision stage enumeration."""
    STAGE_1 = "STAGE_1"
    STAGE_2 = "STAGE_2"
    STAGE_3 = "STAGE_3"


@dataclass
class LoanProfile:
    """Loan profile configuration for test data generation."""
    loan_type: LoanType
    min_amount: float
    max_amount: float
    min_term: int
    max_term: int
    min_rate: float
    max_rate: float
    has_collateral: bool
    min_ltv: float = 0.0
    max_ltv: float = 1.0
    credit_score_range: tuple = (300, 850)


class DataFactory:
    """Factory for generating comprehensive test data for IFRS9 system."""
    
    # Loan profiles by type
    LOAN_PROFILES = {
        LoanType.MORTGAGE: LoanProfile(
            loan_type=LoanType.MORTGAGE,
            min_amount=50000.0,
            max_amount=1000000.0,
            min_term=120,
            max_term=360,
            min_rate=3.0,
            max_rate=7.0,
            has_collateral=True,
            min_ltv=0.5,
            max_ltv=0.95,
            credit_score_range=(580, 850)
        ),
        LoanType.AUTO: LoanProfile(
            loan_type=LoanType.AUTO,
            min_amount=15000.0,
            max_amount=100000.0,
            min_term=24,
            max_term=84,
            min_rate=4.0,
            max_rate=12.0,
            has_collateral=True,
            min_ltv=0.7,
            max_ltv=1.2,
            credit_score_range=(550, 850)
        ),
        LoanType.PERSONAL: LoanProfile(
            loan_type=LoanType.PERSONAL,
            min_amount=5000.0,
            max_amount=50000.0,
            min_term=12,
            max_term=72,
            min_rate=8.0,
            max_rate=25.0,
            has_collateral=False,
            credit_score_range=(600, 800)
        ),
        LoanType.BUSINESS: LoanProfile(
            loan_type=LoanType.BUSINESS,
            min_amount=25000.0,
            max_amount=500000.0,
            min_term=24,
            max_term=120,
            min_rate=5.0,
            max_rate=15.0,
            has_collateral=True,
            min_ltv=0.6,
            max_ltv=0.85,
            credit_score_range=(650, 850)
        )
    }
    
    def __init__(self, seed: int = 42):
        """Initialize the test data factory.
        
        Args:
            seed: Random seed for reproducible data generation
        """
        self.rng = np.random.RandomState(seed)
    
    def generate_loan_portfolio(self, 
                              n_loans: int = 100,
                              stage_distribution: Optional[Dict[str, float]] = None,
                              loan_type_distribution: Optional[Dict[str, float]] = None,
                              include_edge_cases: bool = True,
                              base_date: Optional[datetime] = None) -> pd.DataFrame:
        """Generate a comprehensive loan portfolio for testing.
        
        Args:
            n_loans: Number of loans to generate
            stage_distribution: Distribution of loans by IFRS9 stage
            loan_type_distribution: Distribution of loans by type
            include_edge_cases: Whether to include edge cases and boundary conditions
            base_date: Base date for loan origination and maturity calculations
            
        Returns:
            DataFrame with comprehensive loan portfolio data
        """
        if base_date is None:
            base_date = datetime(2024, 1, 1)
        
        if stage_distribution is None:
            stage_distribution = {"STAGE_1": 0.7, "STAGE_2": 0.2, "STAGE_3": 0.1}
        
        if loan_type_distribution is None:
            loan_type_distribution = {
                "MORTGAGE": 0.4, "AUTO": 0.3, "PERSONAL": 0.2, "BUSINESS": 0.1
            }
        
        loans = []
        
        for i in range(n_loans):
            loan = self._generate_single_loan(
                loan_id=f"L{i+1:06d}",
                customer_id=f"C{i+1:06d}",
                stage_distribution=stage_distribution,
                loan_type_distribution=loan_type_distribution,
                base_date=base_date
            )
            loans.append(loan)
        
        # Add edge cases if requested
        if include_edge_cases:
            edge_cases = self._generate_edge_cases(base_date, len(loans))
            loans.extend(edge_cases)
        
        return pd.DataFrame(loans)
    
    def _generate_single_loan(self, 
                            loan_id: str, 
                            customer_id: str,
                            stage_distribution: Dict[str, float],
                            loan_type_distribution: Dict[str, float],
                            base_date: datetime) -> Dict[str, Any]:
        """Generate a single loan record."""
        # Select loan type
        loan_type_name = self.rng.choice(
            list(loan_type_distribution.keys()),
            p=list(loan_type_distribution.values())
        )
        loan_type = LoanType(loan_type_name)
        profile = self.LOAN_PROFILES[loan_type]
        
        # Select provision stage
        stage_name = self.rng.choice(
            list(stage_distribution.keys()),
            p=list(stage_distribution.values())
        )
        stage = ProvisionStage(stage_name)
        
        # Generate loan amount
        loan_amount = self.rng.uniform(profile.min_amount, profile.max_amount)
        
        # Generate term and interest rate
        term_months = self.rng.randint(profile.min_term, profile.max_term + 1)
        interest_rate = self.rng.uniform(profile.min_rate, profile.max_rate)
        
        # Generate credit score within range
        min_score, max_score = profile.credit_score_range
        credit_score = int(self.rng.uniform(min_score, max_score))
        
        # Adjust credit score based on stage (stage correlation)
        if stage == ProvisionStage.STAGE_3:
            credit_score = min(credit_score, 600)  # Poor credit for Stage 3
        elif stage == ProvisionStage.STAGE_2:
            credit_score = max(min(credit_score, 700), 500)  # Medium credit for Stage 2
        else:
            credit_score = max(credit_score, 650)  # Good credit for Stage 1
        
        # Generate DPD based on stage
        days_past_due = self._generate_dpd_for_stage(stage)
        
        # Generate current balance (with some paydown)
        paydown_factor = self.rng.uniform(0.02, 0.4)  # 2% to 40% paydown
        current_balance = loan_amount * (1 - paydown_factor)
        
        # Generate collateral information
        collateral_value = 0.0
        ltv_ratio = None
        
        if profile.has_collateral:
            # Generate collateral value
            collateral_multiplier = self.rng.uniform(1.0, 1.5)
            collateral_value = loan_amount * collateral_multiplier
            
            # Calculate LTV ratio
            ltv_ratio = loan_amount / collateral_value
            # Add some variation within profile limits
            ltv_variation = self.rng.uniform(-0.05, 0.05)
            ltv_ratio = np.clip(ltv_ratio + ltv_variation, profile.min_ltv, profile.max_ltv)
        
        # Generate dates
        origination_date = base_date - timedelta(days=self.rng.randint(30, 1095))  # Up to 3 years ago
        maturity_date = origination_date + timedelta(days=term_months * 30)
        
        return {
            "loan_id": loan_id,
            "customer_id": customer_id,
            "loan_amount": round(loan_amount, 2),
            "current_balance": round(current_balance, 2),
            "days_past_due": days_past_due,
            "credit_score": credit_score,
            "loan_type": loan_type.value,
            "collateral_value": round(collateral_value, 2) if collateral_value > 0 else 0.0,
            "ltv_ratio": round(ltv_ratio, 4) if ltv_ratio is not None else None,
            "interest_rate": round(interest_rate, 2),
            "term_months": term_months,
            "maturity_date": maturity_date.strftime("%Y-%m-%d"),
            "origination_date": origination_date.strftime("%Y-%m-%d"),
            "provision_stage": stage.value,
            # Additional risk parameters
            "pd_rate": self._generate_pd_for_stage(stage),
            "lgd_rate": self._generate_lgd_for_loan_type(loan_type, profile.has_collateral),
        }
    
    def _generate_dpd_for_stage(self, stage: ProvisionStage) -> int:
        """Generate Days Past Due appropriate for the given stage."""
        if stage == ProvisionStage.STAGE_1:
            return max(0, int(self.rng.normal(5, 8)))  # Mostly 0-20 days
        elif stage == ProvisionStage.STAGE_2:
            return int(self.rng.uniform(30, 89))  # 30-89 days (IFRS9 Stage 2)
        else:  # STAGE_3
            return int(self.rng.uniform(90, 365))  # 90+ days (IFRS9 Stage 3)
    
    def _generate_pd_for_stage(self, stage: ProvisionStage) -> float:
        """Generate Probability of Default appropriate for the given stage."""
        if stage == ProvisionStage.STAGE_1:
            return round(max(0.001, self.rng.normal(0.02, 0.01)), 4)  # ~1-3%
        elif stage == ProvisionStage.STAGE_2:
            return round(max(0.05, self.rng.normal(0.15, 0.05)), 4)  # ~10-20%
        else:  # STAGE_3
            return 1.0  # 100% for defaulted loans
    
    def _generate_lgd_for_loan_type(self, loan_type: LoanType, has_collateral: bool) -> float:
        """Generate Loss Given Default appropriate for the loan type."""
        if has_collateral:
            # Secured loans have lower LGD
            base_lgd = self.rng.uniform(0.3, 0.6)
        else:
            # Unsecured loans have higher LGD
            base_lgd = self.rng.uniform(0.6, 0.9)
        
        # Adjust based on loan type
        if loan_type == LoanType.MORTGAGE:
            base_lgd *= 0.8  # Real estate provides good recovery
        elif loan_type == LoanType.AUTO:
            base_lgd *= 0.9  # Vehicle depreciation affects recovery
        elif loan_type == LoanType.PERSONAL:
            base_lgd *= 1.1  # Limited recovery options
        
        return round(min(1.0, max(0.1, base_lgd)), 4)
    
    def _generate_edge_cases(self, base_date: datetime, existing_count: int) -> List[Dict[str, Any]]:
        """Generate edge case loan records for boundary testing."""
        edge_cases = []
        
        # Boundary DPD cases
        boundary_cases = [
            {"days_past_due": 0, "stage": "STAGE_1", "case": "zero_dpd"},
            {"days_past_due": 29, "stage": "STAGE_1", "case": "just_below_stage2"},
            {"days_past_due": 30, "stage": "STAGE_2", "case": "stage2_threshold"},
            {"days_past_due": 89, "stage": "STAGE_2", "case": "just_below_stage3"},
            {"days_past_due": 90, "stage": "STAGE_3", "case": "stage3_threshold"},
            {"days_past_due": 365, "stage": "STAGE_3", "case": "very_late"},
        ]
        
        # Credit score boundaries
        credit_boundaries = [
            {"credit_score": 300, "stage": "STAGE_3", "case": "min_credit"},
            {"credit_score": 500, "stage": "STAGE_2", "case": "poor_credit_boundary"},
            {"credit_score": 850, "stage": "STAGE_1", "case": "max_credit"},
        ]
        
        # LTV boundaries
        ltv_boundaries = [
            {"ltv_ratio": 0.95, "stage": "STAGE_2", "case": "high_ltv"},
            {"ltv_ratio": 1.0, "stage": "STAGE_2", "case": "max_ltv"},
        ]
        
        case_id = existing_count + 1
        
        # Generate boundary DPD cases
        for boundary in boundary_cases:
            edge_case = {
                "loan_id": f"E{case_id:06d}",
                "customer_id": f"EC{case_id:06d}",
                "loan_amount": 100000.0,
                "current_balance": 90000.0,
                "days_past_due": boundary["days_past_due"],
                "credit_score": 650,
                "loan_type": "MORTGAGE",
                "collateral_value": 120000.0,
                "ltv_ratio": 0.83,
                "interest_rate": 5.0,
                "term_months": 360,
                "maturity_date": (base_date + timedelta(days=360*30)).strftime("%Y-%m-%d"),
                "origination_date": base_date.strftime("%Y-%m-%d"),
                "provision_stage": boundary["stage"],
                "pd_rate": 0.02 if boundary["stage"] == "STAGE_1" else 0.15 if boundary["stage"] == "STAGE_2" else 1.0,
                "lgd_rate": 0.45,
                "test_case": boundary["case"]
            }
            edge_cases.append(edge_case)
            case_id += 1
        
        # Generate credit score boundary cases
        for boundary in credit_boundaries:
            edge_case = {
                "loan_id": f"E{case_id:06d}",
                "customer_id": f"EC{case_id:06d}",
                "loan_amount": 100000.0,
                "current_balance": 90000.0,
                "days_past_due": 0 if boundary["stage"] == "STAGE_1" else 45 if boundary["stage"] == "STAGE_2" else 120,
                "credit_score": boundary["credit_score"],
                "loan_type": "MORTGAGE",
                "collateral_value": 120000.0,
                "ltv_ratio": 0.83,
                "interest_rate": 5.0,
                "term_months": 360,
                "maturity_date": (base_date + timedelta(days=360*30)).strftime("%Y-%m-%d"),
                "origination_date": base_date.strftime("%Y-%m-%d"),
                "provision_stage": boundary["stage"],
                "pd_rate": 0.02 if boundary["stage"] == "STAGE_1" else 0.15 if boundary["stage"] == "STAGE_2" else 1.0,
                "lgd_rate": 0.45,
                "test_case": boundary["case"]
            }
            edge_cases.append(edge_case)
            case_id += 1
        
        return edge_cases
    
    def generate_validation_test_data(self) -> pd.DataFrame:
        """Generate data with various validation issues for testing validation logic."""
        invalid_data = [
            # Invalid loan ID format
            {
                "loan_id": "INVALID_ID",
                "customer_id": "C000001",
                "loan_amount": 100000.0,
                "current_balance": 90000.0,
                "days_past_due": 0,
                "credit_score": 750,
                "loan_type": "MORTGAGE",
                "test_issue": "invalid_loan_id"
            },
            # Negative loan amount
            {
                "loan_id": "L000001",
                "customer_id": "C000001",
                "loan_amount": -50000.0,
                "current_balance": 45000.0,
                "days_past_due": 0,
                "credit_score": 700,
                "loan_type": "AUTO",
                "test_issue": "negative_amount"
            },
            # Credit score out of range
            {
                "loan_id": "L000002",
                "customer_id": "C000002",
                "loan_amount": 75000.0,
                "current_balance": 70000.0,
                "days_past_due": 15,
                "credit_score": 1000,
                "loan_type": "PERSONAL",
                "test_issue": "credit_score_too_high"
            },
            # Invalid loan type
            {
                "loan_id": "L000003",
                "customer_id": "C000003",
                "loan_amount": 60000.0,
                "current_balance": 55000.0,
                "days_past_due": 30,
                "credit_score": 650,
                "loan_type": "INVALID_TYPE",
                "test_issue": "invalid_loan_type"
            },
            # Negative DPD
            {
                "loan_id": "L000004",
                "customer_id": "C000004",
                "loan_amount": 80000.0,
                "current_balance": 75000.0,
                "days_past_due": -5,
                "credit_score": 680,
                "loan_type": "MORTGAGE",
                "test_issue": "negative_dpd"
            }
        ]
        
        return pd.DataFrame(invalid_data)
    
    def generate_performance_test_data(self, n_loans: int = 10000) -> pd.DataFrame:
        """Generate large dataset for performance testing."""
        return self.generate_loan_portfolio(
            n_loans=n_loans,
            stage_distribution={"STAGE_1": 0.75, "STAGE_2": 0.2, "STAGE_3": 0.05},
            include_edge_cases=False
        )
