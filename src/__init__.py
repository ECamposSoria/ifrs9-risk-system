"""IFRS9 Risk System - Core Package.

This package contains the implementation of an IFRS9-compliant credit risk
analysis system with the following components:

- generate_data: Synthetic data generation for testing
- rules_engine: PySpark-based IFRS9 rules processing
- validation: Data quality and validation checks
- ml_model: Machine learning models for credit risk classification
"""

__version__ = "1.0.0"
__author__ = "IFRS9 Risk Team"

# Package imports
from .generate_data import DataGenerator
from .rules_engine import IFRS9RulesEngine
# Use simplified validator to avoid pandera dependency issues
try:
    from .validation import DataValidator
except ImportError:
    from .validation_simple import DataValidator
from .ml_model import CreditRiskClassifier

__all__ = [
    "DataGenerator",
    "IFRS9RulesEngine",
    "DataValidator",
    "CreditRiskClassifier",
]