"""Data validation module for IFRS9 risk system.

This module provides comprehensive data quality checks and validation
for the IFRS9 credit risk analysis pipeline.
"""

from typing import Any, Dict, List, Optional, Tuple

import great_expectations as ge
import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema


class DataValidator:
    """Data validation and quality checks for IFRS9 data.
    
    This class provides:
    - Schema validation for input data
    - Business rule validation
    - Data quality metrics
    - Anomaly detection
    """
    
    def __init__(self):
        """Initialize the data validator."""
        self.loan_schema = self._create_loan_schema()
        self.payment_schema = self._create_payment_schema()
        self.macro_schema = self._create_macro_schema()
        self.validation_results = []
        
    def _create_loan_schema(self) -> DataFrameSchema:
        """Create schema for loan portfolio data.
        
        Returns:
            Pandera DataFrameSchema for loan data
        """
        return DataFrameSchema({
            "loan_id": Column(str, Check.str_matches(r"^L\d{6}$"), nullable=False),
            "customer_id": Column(str, Check.str_matches(r"^C\d{6}$"), nullable=False),
            "loan_amount": Column(float, [
                Check.greater_than(0),
                Check.less_than_or_equal_to(10000000)
            ], nullable=False),
            "interest_rate": Column(float, [
                Check.greater_than_or_equal_to(0),
                Check.less_than_or_equal_to(30)
            ], nullable=False),
            "term_months": Column(int, [
                Check.greater_than(0),
                Check.less_than_or_equal_to(360)
            ], nullable=False),
            "loan_type": Column(str, Check.isin([
                "MORTGAGE", "AUTO", "PERSONAL", "BUSINESS", "CREDIT_CARD"
            ]), nullable=False),
            "credit_score": Column(int, [
                Check.greater_than_or_equal_to(300),
                Check.less_than_or_equal_to(850)
            ], nullable=False),
            "days_past_due": Column(int, Check.greater_than_or_equal_to(0), nullable=False),
            "current_balance": Column(float, Check.greater_than_or_equal_to(0), nullable=False),
            "provision_stage": Column(str, Check.isin([
                "STAGE_1", "STAGE_2", "STAGE_3"
            ]), nullable=False),
            "pd_rate": Column(float, [
                Check.greater_than_or_equal_to(0),
                Check.less_than_or_equal_to(1)
            ], nullable=True),
            "lgd_rate": Column(float, [
                Check.greater_than_or_equal_to(0),
                Check.less_than_or_equal_to(1)
            ], nullable=True),
        })
    
    def _create_payment_schema(self) -> DataFrameSchema:
        """Create schema for payment history data.
        
        Returns:
            Pandera DataFrameSchema for payment data
        """
        return DataFrameSchema({
            "loan_id": Column(str, Check.str_matches(r"^L\d{6}$"), nullable=False),
            "payment_date": Column("datetime64[ns]", nullable=False),
            "scheduled_payment": Column(float, Check.greater_than_or_equal_to(0), nullable=False),
            "actual_payment": Column(float, Check.greater_than_or_equal_to(0), nullable=False),
            "payment_status": Column(str, Check.isin(["PAID", "MISSED", "PARTIAL"]), nullable=False),
            "days_late": Column(int, Check.greater_than_or_equal_to(0), nullable=False),
        })
    
    def _create_macro_schema(self) -> DataFrameSchema:
        """Create schema for macroeconomic data.
        
        Returns:
            Pandera DataFrameSchema for macro data
        """
        return DataFrameSchema({
            "date": Column("datetime64[ns]", nullable=False),
            "gdp_growth": Column(float, [
                Check.greater_than(-10),
                Check.less_than(10)
            ], nullable=False),
            "unemployment_rate": Column(float, [
                Check.greater_than_or_equal_to(0),
                Check.less_than(30)
            ], nullable=False),
            "inflation_rate": Column(float, [
                Check.greater_than(-5),
                Check.less_than(20)
            ], nullable=False),
            "interest_rate": Column(float, [
                Check.greater_than_or_equal_to(0),
                Check.less_than(25)
            ], nullable=False),
        })
    
    def validate_loan_portfolio(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate loan portfolio data.
        
        Args:
            df: DataFrame with loan portfolio data
            
        Returns:
            Tuple of (validation_passed, list_of_errors)
        """
        errors = []
        
        try:
            # Schema validation
            self.loan_schema.validate(df)
        except pa.errors.SchemaError as e:
            errors.append(f"Schema validation failed: {str(e)}")
        
        # Business rule validations
        
        # Check loan amount vs current balance
        invalid_balance = df[df["current_balance"] > df["loan_amount"]]
        if not invalid_balance.empty:
            errors.append(f"Found {len(invalid_balance)} loans with current_balance > loan_amount")
        
        # Check stage consistency with DPD
        stage_3_low_dpd = df[(df["provision_stage"] == "STAGE_3") & (df["days_past_due"] < 90)]
        if not stage_3_low_dpd.empty:
            errors.append(f"Found {len(stage_3_low_dpd)} Stage 3 loans with DPD < 90")
        
        # Check for duplicate loan IDs
        duplicates = df[df.duplicated(subset=["loan_id"], keep=False)]
        if not duplicates.empty:
            errors.append(f"Found {len(duplicates)} duplicate loan IDs")
        
        # Check ECL calculations if present
        if "ecl_amount" in df.columns:
            negative_ecl = df[df["ecl_amount"] < 0]
            if not negative_ecl.empty:
                errors.append(f"Found {len(negative_ecl)} loans with negative ECL")
        
        validation_passed = len(errors) == 0
        
        # Store results
        self.validation_results.append({
            "dataset": "loan_portfolio",
            "records": len(df),
            "passed": validation_passed,
            "errors": errors
        })
        
        return validation_passed, errors
    
    def validate_payment_history(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate payment history data.
        
        Args:
            df: DataFrame with payment history data
            
        Returns:
            Tuple of (validation_passed, list_of_errors)
        """
        errors = []
        
        try:
            # Schema validation
            self.payment_schema.validate(df)
        except pa.errors.SchemaError as e:
            errors.append(f"Schema validation failed: {str(e)}")
        
        # Business rule validations
        
        # Check actual payment vs scheduled payment consistency
        overpayments = df[df["actual_payment"] > df["scheduled_payment"] * 2]
        if not overpayments.empty:
            errors.append(f"Found {len(overpayments)} payments > 2x scheduled amount")
        
        # Check payment status consistency
        status_mismatch = df[
            ((df["payment_status"] == "PAID") & (df["actual_payment"] == 0)) |
            ((df["payment_status"] == "MISSED") & (df["actual_payment"] > 0))
        ]
        if not status_mismatch.empty:
            errors.append(f"Found {len(status_mismatch)} payments with status mismatch")
        
        validation_passed = len(errors) == 0
        
        # Store results
        self.validation_results.append({
            "dataset": "payment_history",
            "records": len(df),
            "passed": validation_passed,
            "errors": errors
        })
        
        return validation_passed, errors
    
    def validate_macroeconomic_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate macroeconomic data.
        
        Args:
            df: DataFrame with macroeconomic data
            
        Returns:
            Tuple of (validation_passed, list_of_errors)
        """
        errors = []
        
        try:
            # Schema validation
            self.macro_schema.validate(df)
        except pa.errors.SchemaError as e:
            errors.append(f"Schema validation failed: {str(e)}")
        
        # Check for missing dates
        if not df["date"].is_monotonic_increasing:
            errors.append("Dates are not in chronological order")
        
        # Check for gaps in time series
        date_diff = df["date"].diff().dt.days
        if date_diff.max() > 35:  # More than 35 days gap
            errors.append("Found gaps in time series data")
        
        validation_passed = len(errors) == 0
        
        # Store results
        self.validation_results.append({
            "dataset": "macroeconomic_data",
            "records": len(df),
            "passed": validation_passed,
            "errors": errors
        })
        
        return validation_passed, errors
    
    def check_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive data quality checks.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dictionary with data quality metrics
        """
        quality_report = {
            "total_records": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "column_metrics": {},
            "completeness_score": 0,
            "uniqueness_metrics": {},
        }
        
        # Column-level metrics
        for col in df.columns:
            col_metrics = {
                "dtype": str(df[col].dtype),
                "null_count": df[col].isnull().sum(),
                "null_percentage": (df[col].isnull().sum() / len(df)) * 100,
                "unique_values": df[col].nunique(),
                "unique_percentage": (df[col].nunique() / len(df)) * 100,
            }
            
            # Add statistics for numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                col_metrics.update({
                    "mean": df[col].mean(),
                    "std": df[col].std(),
                    "min": df[col].min(),
                    "max": df[col].max(),
                    "q25": df[col].quantile(0.25),
                    "q50": df[col].quantile(0.50),
                    "q75": df[col].quantile(0.75),
                })
            
            quality_report["column_metrics"][col] = col_metrics
        
        # Calculate completeness score
        total_cells = len(df) * len(df.columns)
        non_null_cells = total_cells - df.isnull().sum().sum()
        quality_report["completeness_score"] = (non_null_cells / total_cells) * 100
        
        # Identify potential key columns
        for col in df.columns:
            if df[col].nunique() == len(df):
                quality_report["uniqueness_metrics"][col] = "Potential primary key"
        
        return quality_report
    
    def detect_anomalies(self, df: pd.DataFrame, numeric_columns: List[str]) -> Dict[str, List]:
        """Detect anomalies in numeric columns using statistical methods.
        
        Args:
            df: DataFrame to analyze
            numeric_columns: List of numeric column names to check
            
        Returns:
            Dictionary with anomaly detection results
        """
        anomalies = {}
        
        for col in numeric_columns:
            if col not in df.columns:
                continue
            
            # IQR method for outlier detection
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
            if not outliers.empty:
                anomalies[col] = {
                    "outlier_count": len(outliers),
                    "outlier_percentage": (len(outliers) / len(df)) * 100,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "outlier_indices": outliers.index.tolist()[:10],  # First 10 indices
                }
        
        return anomalies
    
    def generate_validation_report(self) -> str:
        """Generate a summary validation report.
        
        Returns:
            String with formatted validation report
        """
        report = "=" * 60 + "\n"
        report += "IFRS9 DATA VALIDATION REPORT\n"
        report += "=" * 60 + "\n\n"
        
        for result in self.validation_results:
            report += f"Dataset: {result['dataset']}\n"
            report += f"Records: {result['records']:,}\n"
            report += f"Status: {'PASSED' if result['passed'] else 'FAILED'}\n"
            
            if not result['passed']:
                report += "Errors:\n"
                for error in result['errors']:
                    report += f"  - {error}\n"
            
            report += "-" * 40 + "\n"
        
        # Summary
        total_validations = len(self.validation_results)
        passed_validations = sum(1 for r in self.validation_results if r['passed'])
        
        report += f"\nSummary: {passed_validations}/{total_validations} validations passed\n"
        report += "=" * 60 + "\n"
        
        return report


if __name__ == "__main__":
    # Example usage
    validator = DataValidator()
    
    # Validate sample data
    # loans_df = pd.read_csv("data/raw/loan_portfolio.csv")
    # passed, errors = validator.validate_loan_portfolio(loans_df)
    # quality_report = validator.check_data_quality(loans_df)
    # print(validator.generate_validation_report())