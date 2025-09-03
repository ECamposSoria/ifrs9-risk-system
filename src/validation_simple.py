"""Simplified data validation module for IFRS9 risk system.

This module provides basic data quality checks and validation
without complex dependencies for testing purposes.
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import re


class DataValidator:
    """Simplified data validation for IFRS9 data.
    
    This class provides basic validation without pandera dependency.
    """
    
    def __init__(self):
        """Initialize the data validator."""
        self.validation_results = []
        
    def validate_loan_portfolio(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate loan portfolio data.
        
        Args:
            df: DataFrame with loan data
            
        Returns:
            Tuple of (validation_passed, list_of_errors)
        """
        errors = []
        
        # Check required columns
        required_columns = [
            "loan_id", "customer_id", "loan_amount", "interest_rate",
            "term_months", "loan_type", "credit_score", "days_past_due",
            "current_balance", "provision_stage", "pd_rate", "lgd_rate"
        ]
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            
        if not errors:
            # Validate loan_id format
            invalid_loan_ids = df[~df["loan_id"].str.match(r"^L\d{6}$", na=False)]
            if not invalid_loan_ids.empty:
                errors.append(f"Invalid loan_id format: {invalid_loan_ids['loan_id'].tolist()}")
            
            # Validate customer_id format
            invalid_customer_ids = df[~df["customer_id"].str.match(r"^C\d{6}$", na=False)]
            if not invalid_customer_ids.empty:
                errors.append(f"Invalid customer_id format: {invalid_customer_ids['customer_id'].tolist()}")
            
            # Validate loan_amount
            if (df["loan_amount"] <= 0).any():
                errors.append("Loan amounts must be positive")
            if (df["loan_amount"] > 10000000).any():
                errors.append("Loan amounts exceed maximum limit")
            
            # Validate interest_rate
            if (df["interest_rate"] < 0).any():
                errors.append("Interest rates cannot be negative")
            if (df["interest_rate"] > 30).any():
                errors.append("Interest rates exceed maximum limit")
            
            # Validate credit_score
            if (df["credit_score"] < 300).any():
                errors.append("Credit scores below minimum threshold")
            if (df["credit_score"] > 850).any():
                errors.append("Credit scores exceed maximum limit")
            
            # Validate loan_type
            valid_loan_types = ["MORTGAGE", "AUTO", "PERSONAL", "CREDIT_CARD", "STUDENT", "BUSINESS"]
            invalid_types = df[~df["loan_type"].isin(valid_loan_types)]
            if not invalid_types.empty:
                errors.append(f"Invalid loan types: {invalid_types['loan_type'].unique().tolist()}")
            
            # Validate provision_stage
            valid_stages = ["STAGE_1", "STAGE_2", "STAGE_3"]
            if "provision_stage" in df.columns:
                invalid_stages = df[~df["provision_stage"].isin(valid_stages)]
                if not invalid_stages.empty:
                    errors.append(f"Invalid provision stages: {invalid_stages['provision_stage'].unique().tolist()}")
            
            # Validate PD and LGD rates
            if "pd_rate" in df.columns:
                if (df["pd_rate"] < 0).any() or (df["pd_rate"] > 1).any():
                    errors.append("PD rates must be between 0 and 1")
            
            if "lgd_rate" in df.columns:
                if (df["lgd_rate"] < 0).any() or (df["lgd_rate"] > 1).any():
                    errors.append("LGD rates must be between 0 and 1")
        
        passed = len(errors) == 0
        return passed, errors
    
    def validate_payment_history(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate payment history data.
        
        Args:
            df: DataFrame with payment data
            
        Returns:
            Tuple of (validation_passed, list_of_errors)
        """
        errors = []
        
        # Check required columns
        required_columns = ["loan_id", "payment_date", "amount_due", "amount_paid"]
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        if not errors:
            # Validate amounts
            if (df["amount_due"] < 0).any():
                errors.append("Amount due cannot be negative")
            if (df["amount_paid"] < 0).any():
                errors.append("Amount paid cannot be negative")
        
        passed = len(errors) == 0
        return passed, errors
    
    def validate_macroeconomic_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate macroeconomic data.
        
        Args:
            df: DataFrame with macro data
            
        Returns:
            Tuple of (validation_passed, list_of_errors)
        """
        errors = []
        
        # Check required columns
        required_columns = ["date", "gdp_growth", "unemployment_rate", "inflation_rate"]
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        if not errors:
            # Validate rates are reasonable
            if "gdp_growth" in df.columns:
                if (df["gdp_growth"] < -50).any() or (df["gdp_growth"] > 50).any():
                    errors.append("GDP growth rates outside reasonable range")
            
            if "unemployment_rate" in df.columns:
                if (df["unemployment_rate"] < 0).any() or (df["unemployment_rate"] > 100).any():
                    errors.append("Unemployment rates must be between 0 and 100")
            
            if "inflation_rate" in df.columns:
                if (df["inflation_rate"] < -20).any() or (df["inflation_rate"] > 100).any():
                    errors.append("Inflation rates outside reasonable range")
        
        passed = len(errors) == 0
        return passed, errors
    
    def check_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform data quality checks.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dictionary with quality metrics
        """
        metrics = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "null_counts": df.isnull().sum().to_dict(),
            "duplicate_rows": df.duplicated().sum(),
            "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
            "string_columns": df.select_dtypes(include=['object']).columns.tolist(),
        }
        
        # Add statistics for numeric columns
        numeric_stats = {}
        for col in metrics["numeric_columns"]:
            numeric_stats[col] = {
                "mean": df[col].mean(),
                "std": df[col].std(),
                "min": df[col].min(),
                "max": df[col].max(),
                "nulls": df[col].isnull().sum()
            }
        metrics["numeric_stats"] = numeric_stats
        
        return metrics