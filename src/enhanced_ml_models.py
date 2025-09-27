"""
Enhanced Machine Learning Pipeline for IFRS9 Credit Risk Assessment

This module provides advanced ML models including XGBoost, LightGBM, and ensemble methods
with hyperparameter tuning, cross-validation, and comprehensive model evaluation.
"""

import os
import json
import pickle
import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    RandomizedSearchCV, StratifiedKFold, TimeSeriesSplit
)
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Advanced ML libraries
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor

# Hyperparameter optimization
from optuna import create_study, Trial
import optuna.pruners as pruners
import optuna.samplers as samplers

# Explainability
import shap

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Suppress warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Polars availability check
try:
    import polars as pl
    POLARS_AVAILABLE = True
    logger.info(f"Polars available - Version: {pl.__version__}")
except ImportError:
    POLARS_AVAILABLE = False
    logger.warning("Polars not available - using pandas only")

if POLARS_AVAILABLE:
    from polars.datatypes import NUMERIC_DTYPES
else:
    NUMERIC_DTYPES = set()

class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    """Advanced feature engineering for credit risk modeling with Polars optimization"""
    
    def __init__(self, 
                 create_interactions: bool = True,
                 create_ratios: bool = True,
                 create_temporal_features: bool = True,
                 polynomial_features: bool = False,
                 polynomial_degree: int = 2,
                 use_polars: bool = True,
                 lazy_evaluation: bool = True):
        """
        Initialize feature engineer
        
        Args:
            create_interactions: Create feature interactions
            create_ratios: Create financial ratios
            create_temporal_features: Create time-based features
            polynomial_features: Create polynomial features
            polynomial_degree: Degree for polynomial features
        """
        self.create_interactions = create_interactions
        self.create_ratios = create_ratios
        self.create_temporal_features = create_temporal_features
        self.polynomial_features = polynomial_features
        self.polynomial_degree = polynomial_degree
        self.use_polars = use_polars and POLARS_AVAILABLE
        self.lazy_evaluation = lazy_evaluation
        
        self.feature_names_ = None
        self.interaction_features_ = []
        self.ratio_features_ = []
        self.temporal_features_ = []
        
        if self.use_polars:
            logger.info("AdvancedFeatureEngineer configured with Polars optimization")
        else:
            logger.info("AdvancedFeatureEngineer using pandas implementation")
    
    def fit(self, X: Union[pd.DataFrame, "pl.DataFrame", "pl.LazyFrame"], y=None):
        """Fit the feature engineer for both pandas and Polars inputs."""

        # Handle Polars inputs without forcing conversion when optimization is enabled
        if self.use_polars and POLARS_AVAILABLE and isinstance(X, (pl.DataFrame, pl.LazyFrame)):
            X_pl = X.collect() if isinstance(X, pl.LazyFrame) else X
            self.base_columns_ = list(X_pl.columns)
            self.feature_names_ = list(X_pl.columns)
            numeric_cols = [
                name for name, dtype in zip(X_pl.columns, X_pl.dtypes)
                if dtype in NUMERIC_DTYPES
            ]
            self.numeric_columns_ = numeric_cols
            self.categorical_columns_ = [name for name in X_pl.columns if name not in numeric_cols]
            return self

        # Fallback to pandas-based detection (also handles accidental Polars input)
        if POLARS_AVAILABLE and isinstance(X, pl.DataFrame):
            X = X.to_pandas()
        elif POLARS_AVAILABLE and isinstance(X, pl.LazyFrame):
            X = X.collect().to_pandas()

        self.base_columns_ = list(X.columns)
        self.feature_names_ = list(X.columns)

        self.numeric_columns_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns_ = X.select_dtypes(exclude=[np.number]).columns.tolist()

        return self
    
    def transform(self, X: Union[pd.DataFrame, pl.DataFrame]) -> Union[pd.DataFrame, pl.DataFrame]:
        """Transform features with automatic Polars/pandas optimization"""

        base_cols = set(getattr(self, 'base_columns_', []) or [])
        current_cols = set(X.columns) if hasattr(X, 'columns') else set()
        if base_cols and not base_cols.issubset(current_cols):
            # Assume features already engineered; skip transformation to avoid missing column errors
            return X

        # Auto-detect and use optimal implementation
        if self.use_polars and isinstance(X, (pl.DataFrame, pl.LazyFrame)):
            return self._transform_polars(X)
        elif self.use_polars and isinstance(X, pd.DataFrame):
            # Convert to Polars for optimization, then back to pandas
            X_pl = pl.from_pandas(X)
            X_enhanced_pl = self._transform_polars(X_pl)
            return X_enhanced_pl.to_pandas()
        else:
            # Use pandas implementation
            return self._transform_pandas(X)
    
    def _transform_pandas(self, X: pd.DataFrame) -> pd.DataFrame:
        """Original pandas-based transformation"""
        X_enhanced = X.copy()
        
        # Create financial ratios
        if self.create_ratios:
            X_enhanced = self._create_financial_ratios(X_enhanced)
        
        # Create interactions
        if self.create_interactions:
            X_enhanced = self._create_interactions(X_enhanced)
        
        # Create temporal features
        if self.create_temporal_features:
            X_enhanced = self._create_temporal_features(X_enhanced)
        
        # Create polynomial features
        if self.polynomial_features:
            X_enhanced = self._create_polynomial_features(X_enhanced)
        
        # Fill any NaN values created during transformation
        X_enhanced = X_enhanced.fillna(0)
        
        return X_enhanced
    
    def _transform_polars(self, X: Union[pl.DataFrame, pl.LazyFrame]) -> pl.DataFrame:
        """Polars-optimized feature transformation with lazy evaluation"""

        # Use lazy evaluation if configured
        if self.lazy_evaluation and not isinstance(X, pl.LazyFrame):
            X_lazy = X.lazy()
        elif isinstance(X, pl.LazyFrame):
            X_lazy = X
        else:
            X_lazy = X.lazy()
        
        # Start with base columns
        feature_expressions = [pl.col("*")]
        
        # Create financial ratios using Polars expressions
        if self.create_ratios:
            feature_expressions.extend(self._create_financial_ratios_polars())
        
        # Create interaction features
        if self.create_interactions:
            feature_expressions.extend(self._create_interactions_polars())
        
        # Create temporal features
        if self.create_temporal_features:
            feature_expressions.extend(self._create_temporal_features_polars())
        
        # Create polynomial features
        if self.polynomial_features:
            feature_expressions.extend(self._create_polynomial_features_polars())
        
        # Apply all transformations in one pass
        X_enhanced = X_lazy.with_columns(feature_expressions)
        
        # Handle missing values using Polars
        X_enhanced = X_enhanced.fill_null(strategy="zero")
        
        # Collect if using lazy evaluation
        if isinstance(X_enhanced, pl.LazyFrame):
            X_enhanced = X_enhanced.collect(streaming=True)
        
        # Store feature names
        self.feature_names_ = X_enhanced.columns

        return X_enhanced

    def _has_columns(self, *columns: str) -> bool:
        """Helper to check if required columns are available from fit."""
        base_columns = set(self.feature_names_ or [])
        return all(column in base_columns for column in columns)
    
    def _create_financial_ratios(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create financial ratio features"""
        X_ratios = X.copy()
        
        # Debt service coverage ratio
        if 'customer_income' in X.columns and 'monthly_payment' in X.columns:
            X_ratios['debt_service_coverage'] = (
                X['customer_income'] / 12
            ) / (X['monthly_payment'] + 1e-8)
        
        # Loan utilization ratio
        if 'current_balance' in X.columns and 'loan_amount' in X.columns:
            X_ratios['loan_utilization'] = X['current_balance'] / (X['loan_amount'] + 1e-8)
        
        # Payment-to-income ratio
        if 'monthly_payment' in X.columns and 'customer_income' in X.columns:
            X_ratios['payment_to_income'] = (
                X['monthly_payment'] * 12
            ) / (X['customer_income'] + 1e-8)
        
        # Credit score to loan amount ratio
        if 'credit_score' in X.columns and 'loan_amount' in X.columns:
            X_ratios['credit_score_per_amount'] = X['credit_score'] / (
                np.log1p(X['loan_amount'])
            )
        
        # Interest rate spread (assuming risk-free rate of 2%)
        if 'interest_rate' in X.columns:
            X_ratios['interest_rate_spread'] = X['interest_rate'] - 0.02
        
        # Employment stability score
        if 'employment_length' in X.columns:
            X_ratios['employment_stability'] = np.clip(
                X['employment_length'] / 60, 0, 1
            )  # Normalized to 5 years max
        
        self.ratio_features_ = [col for col in X_ratios.columns if col not in X.columns]
        
        return X_ratios
    
    def _create_interactions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create feature interactions"""
        X_interactions = X.copy()
        
        # Key interactions for credit risk
        interaction_pairs = [
            ('credit_score', 'dti_ratio'),
            ('credit_score', 'days_past_due'),
            ('loan_amount', 'credit_score'),
            ('interest_rate', 'credit_score'),
            ('employment_length', 'dti_ratio'),
            ('current_balance', 'days_past_due')
        ]
        
        for col1, col2 in interaction_pairs:
            if col1 in X.columns and col2 in X.columns:
                # Multiplicative interaction
                X_interactions[f'{col1}_x_{col2}'] = X[col1] * X[col2]
                
                # Ratio interaction
                X_interactions[f'{col1}_div_{col2}'] = X[col1] / (X[col2] + 1e-8)
                
                self.interaction_features_.extend([
                    f'{col1}_x_{col2}', f'{col1}_div_{col2}'
                ])
        
        return X_interactions
    
    def _create_temporal_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features from dates"""
        X_temporal = X.copy()
        
        # Extract features from origination_date if present
        if 'origination_date' in X.columns:
            X_temporal['origination_date'] = pd.to_datetime(X_temporal['origination_date'])
            
            X_temporal['origination_year'] = X_temporal['origination_date'].dt.year
            X_temporal['origination_month'] = X_temporal['origination_date'].dt.month
            X_temporal['origination_quarter'] = X_temporal['origination_date'].dt.quarter
            X_temporal['origination_day_of_year'] = X_temporal['origination_date'].dt.dayofyear
            
            # Loan age in months
            current_date = pd.Timestamp.now()
            X_temporal['loan_age_months'] = (
                current_date - X_temporal['origination_date']
            ).dt.days / 30.44
            
            # Seasonal features
            X_temporal['is_q4_origination'] = (
                X_temporal['origination_quarter'] == 4
            ).astype(int)
            
            self.temporal_features_.extend([
                'origination_year', 'origination_month', 'origination_quarter',
                'origination_day_of_year', 'loan_age_months', 'is_q4_origination'
            ])
        
        return X_temporal
    
    def _create_polynomial_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create polynomial features for key variables"""
        X_poly = X.copy()
        
        # Apply polynomial features to key risk indicators
        poly_features = ['credit_score', 'dti_ratio', 'loan_amount']
        
        for feature in poly_features:
            if feature in X.columns:
                for degree in range(2, self.polynomial_degree + 1):
                    X_poly[f'{feature}_poly_{degree}'] = X[feature] ** degree
        
        return X_poly
    
    def _create_financial_ratios_polars(self) -> List[pl.Expr]:
        """Create financial ratio features using Polars expressions"""
        ratio_expressions = []
        
        # Debt service coverage ratio
        if self._has_columns("customer_income", "monthly_payment"):
            ratio_expressions.append(
                ((pl.col("customer_income") / 12) / (pl.col("monthly_payment") + 1e-8))
                .alias("debt_service_coverage")
            )

        # Loan utilization ratio
        if self._has_columns("current_balance", "loan_amount"):
            ratio_expressions.append(
                (pl.col("current_balance") / (pl.col("loan_amount") + 1e-8))
                .alias("loan_utilization")
            )

        # Payment-to-income ratio
        if self._has_columns("monthly_payment", "customer_income"):
            ratio_expressions.append(
                ((pl.col("monthly_payment") * 12) / (pl.col("customer_income") + 1e-8))
                .alias("payment_to_income")
            )

        # Credit score to loan amount ratio
        if self._has_columns("credit_score", "loan_amount"):
            ratio_expressions.append(
                (pl.col("credit_score") / pl.col("loan_amount").log1p())
                .alias("credit_score_per_amount")
            )

        # Interest rate spread (risk-free rate assumed 2%)
        if self._has_columns("interest_rate"):
            ratio_expressions.append(
                (pl.col("interest_rate") - 0.02)
                .alias("interest_rate_spread")
            )

        # Employment stability score (if employment_length exists)
        if self._has_columns("employment_length"):
            ratio_expressions.append(
                (pl.col("employment_length") / 60).clip(0, 1)
                .alias("employment_stability")
            )

        self.ratio_features_ = [expr.meta.output_name() for expr in ratio_expressions]
        return ratio_expressions
    
    def _create_interactions_polars(self) -> List[pl.Expr]:
        """Create feature interactions using Polars expressions"""
        interaction_expressions = []
        
        # Key interactions for credit risk
        interaction_pairs = [
            ('credit_score', 'dti_ratio'),
            ('credit_score', 'days_past_due'),
            ('loan_amount', 'credit_score'),
            ('interest_rate', 'credit_score'),
            ('employment_length', 'dti_ratio'),
            ('current_balance', 'days_past_due')
        ]
        
        for col1, col2 in interaction_pairs:
            if not self._has_columns(col1, col2):
                continue

            # Multiplicative interaction
            interaction_expressions.append(
                (pl.col(col1) * pl.col(col2))
                .alias(f"{col1}_x_{col2}")
            )

            # Ratio interaction (with safe division)
            interaction_expressions.append(
                (pl.col(col1) / (pl.col(col2) + 1e-8))
                .alias(f"{col1}_div_{col2}")
            )

        # Advanced IFRS9 interactions
        # No additional advanced interactions beyond parity with pandas implementation
        
        self.interaction_features_ = [expr.meta.output_name() for expr in interaction_expressions]
        return interaction_expressions
    
    def _create_temporal_features_polars(self) -> List[pl.Expr]:
        """Create temporal features using Polars expressions"""
        temporal_expressions = []
        
        # Loan age calculation (if origination_date exists)
        if self._has_columns("origination_date"):
            temporal_expressions.extend([
                ((pl.lit(datetime.now()) - pl.col("origination_date")).dt.total_days() / 30.44)
                .alias("loan_age_months"),
                pl.col("origination_date").dt.year().alias("origination_year"),
                pl.col("origination_date").dt.month().alias("origination_month"),
                pl.col("origination_date").dt.quarter().alias("origination_quarter"),
                pl.col("origination_date").dt.ordinal_day().alias("origination_day_of_year"),
                (pl.col("origination_date").dt.quarter() == 4).cast(pl.Int32)
                .alias("is_q4_origination"),
            ])
        else:
            temporal_expressions.append(pl.lit(30).alias("loan_age_months"))

        self.temporal_features_ = [expr.meta.output_name() for expr in temporal_expressions]
        return temporal_expressions
    
    def _create_polynomial_features_polars(self) -> List[pl.Expr]:
        """Create polynomial features using Polars expressions"""
        polynomial_expressions = []
        
        poly_features = ['credit_score', 'dti_ratio', 'loan_amount']

        for feature in poly_features:
            if not self._has_columns(feature):
                continue
            for degree in range(2, self.polynomial_degree + 1):
                polynomial_expressions.append(
                    (pl.col(feature) ** degree)
                    .alias(f"{feature}_poly_{degree}")
                )
        # No additional polynomial combinations beyond pandas parity

        return polynomial_expressions

class OptimizedMLPipeline:
    """Advanced ML pipeline with multiple algorithms and optimization"""
    
    def __init__(self,
                 models_config: Optional[Dict[str, Any]] = None,
                 optimization_method: str = "optuna",
                 cv_folds: int = 5,
                 random_state: int = 42,
                 use_polars: bool = True,
                 enable_streaming: bool = False):
        """
        Initialize ML Pipeline
        
        Args:
            models_config: Configuration for models
            optimization_method: Optimization method ('optuna', 'grid_search', 'random_search')
            cv_folds: Number of CV folds
            random_state: Random state for reproducibility
            use_polars: Enable Polars optimization for feature engineering and data processing
            enable_streaming: Enable streaming processing for large datasets
        """
        self.models_config = models_config or self._get_default_models_config()
        self.optimization_method = optimization_method
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.use_polars = use_polars and POLARS_AVAILABLE
        self.enable_streaming = enable_streaming
        
        # Model storage
        self.trained_models = {}
        self.model_scores = {}
        self.best_model = None
        self.best_model_name = None
        
        # Feature engineering with Polars support
        self.feature_engineer = AdvancedFeatureEngineer(
            use_polars=self.use_polars,
            lazy_evaluation=True
        )
        self.preprocessor = None
        
        # Results storage
        self.training_results = {}
        self.feature_importance = {}
        
        # DataFrame format tracking
        self._input_format = None  # Track if input is polars or pandas

        # Categorical metadata for warning-free preprocessing
        self.categorical_levels_: Dict[str, List[str]] = {}
        self._categorical_features: List[str] = []
        self.processed_feature_names: List[str] = []
        
        if self.use_polars:
            logger.info("OptimizedMLPipeline configured with Polars optimization")
        else:
            logger.info("OptimizedMLPipeline using pandas-only implementation")
        
    def _get_default_models_config(self) -> Dict[str, Any]:
        """Get default configuration for all models"""
        return {
            'random_forest': {
                'model_class': RandomForestClassifier,
                'param_space': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                },
                'fixed_params': {
                    'random_state': 42,
                    'n_jobs': -1,
                    'max_features': 'sqrt',
                    'class_weight': 'balanced'
                }
            }
        }
    
    def prepare_data(self, 
                    df: Union[pd.DataFrame, pl.DataFrame],
                    target_column: str = 'provision_stage',
                    test_size: float = 0.2) -> Tuple[Union[pd.DataFrame, pl.DataFrame], Union[pd.DataFrame, pl.DataFrame], pd.Series, pd.Series]:
        """
        Prepare data for training with Polars optimization
        
        Args:
            df: Input dataframe (Polars or pandas)
            target_column: Target variable column name
            test_size: Test set proportion
            
        Returns:
            X_train, X_test, y_train, y_test (optimized format based on configuration)
        """
        # Track input format
        self._input_format = 'polars' if isinstance(df, (pl.DataFrame, pl.LazyFrame)) else 'pandas'
        
        # Handle Polars DataFrame
        if isinstance(df, (pl.DataFrame, pl.LazyFrame)) and self.use_polars:
            return self._prepare_data_polars(df, target_column, test_size)
        else:
            # Convert Polars to pandas if needed
            if isinstance(df, (pl.DataFrame, pl.LazyFrame)):
                df = df.to_pandas() if hasattr(df, 'to_pandas') else df.collect().to_pandas()
            return self._prepare_data_pandas(df, target_column, test_size)
    
    def _prepare_data_pandas(self, df: pd.DataFrame, target_column: str, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Original pandas-based data preparation"""
        # Separate features and target
        feature_columns = [col for col in df.columns if col != target_column]
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Handle categorical encoding for target
        if y.dtype == 'object':
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
        
        # Train-test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def _prepare_data_polars(self, df: Union[pl.DataFrame, pl.LazyFrame], target_column: str, test_size: float) -> Tuple[pl.DataFrame, pl.DataFrame, pd.Series, pd.Series]:
        """Polars-optimized data preparation"""
        
        # Ensure we have a DataFrame (not LazyFrame) for sklearn operations
        if isinstance(df, pl.LazyFrame):
            df = df.collect(streaming=self.enable_streaming)
        
        # Separate features and target using Polars
        feature_columns = [col for col in df.columns if col != target_column]
        X = df.select(feature_columns)
        y = df.select(target_column).to_series()
        
        # Convert target to pandas for sklearn compatibility and encoding
        y_pandas = y.to_pandas()
        
        # Handle categorical encoding for target
        if y_pandas.dtype == 'object':
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y_pandas)
        else:
            y_encoded = y_pandas.values
        
        # Convert to pandas temporarily for train_test_split (sklearn requirement)
        X_pandas_temp = X.to_pandas()
        
        # Train-test split with stratification
        X_train_pandas, X_test_pandas, y_train, y_test = train_test_split(
            X_pandas_temp, y_encoded,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y_encoded
        )
        
        # Convert back to Polars for feature engineering
        X_train_polars = pl.from_pandas(X_train_pandas)
        X_test_polars = pl.from_pandas(X_test_pandas)
        
        return self._finalize_data_preparation_polars(X_train_polars, X_test_polars, y_train, y_test)
    
    def _finalize_data_preparation_pandas(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: np.ndarray, y_test: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Finalize pandas data preparation with preprocessing"""
        
        # Create preprocessor for numerical and categorical features
        numeric_features = X_train.select_dtypes(include=[np.number]).columns
        categorical_features = X_train.select_dtypes(exclude=[np.number]).columns

        X_train = X_train.copy()
        X_test = X_test.copy()

        self._categorical_features = list(categorical_features)
        levels = self._learn_categorical_levels(X_train, self._categorical_features)
        self.categorical_levels_ = levels
        X_train = self._apply_categorical_levels(X_train)
        X_test = self._apply_categorical_levels(X_test)
        
        from sklearn.preprocessing import OneHotEncoder

        encoder_kwargs: Dict[str, Any] = {
            'drop': 'first',
            'sparse_output': False,
            'handle_unknown': 'ignore'
        }
        if self._categorical_features:
            encoder_kwargs['categories'] = [
                levels[col] for col in self._categorical_features
            ]

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), list(numeric_features)),
                ('cat', OneHotEncoder(**encoder_kwargs), self._categorical_features)
            ]
        )
        
        self.preprocessor = preprocessor
        
        # Fit and transform training data
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Convert back to DataFrames with proper column names
        numeric_feature_names = list(numeric_features)
        
        if len(categorical_features) > 0:
            categorical_feature_names = []
            encoder = preprocessor.named_transformers_['cat']
            if hasattr(encoder, 'get_feature_names_out'):
                cat_names = encoder.get_feature_names_out(categorical_features)
                categorical_feature_names.extend(cat_names)
            else:
                # Fallback for older sklearn versions
                for cat_feature in categorical_features:
                    unique_values = X_train[cat_feature].unique()
                    cat_names = [f"{cat_feature}_{val}" for val in unique_values[1:]]
                    categorical_feature_names.extend(cat_names)
        else:
            categorical_feature_names = []
        
        all_feature_names = numeric_feature_names + categorical_feature_names
        self.processed_feature_names = all_feature_names
        
        X_train_final = pd.DataFrame(X_train_processed, columns=all_feature_names)
        X_test_final = pd.DataFrame(X_test_processed, columns=all_feature_names)
        
        logger.info(f"Pandas data prepared: {len(X_train_final)} training samples, {len(X_test_final)} test samples")
        logger.info(f"Feature engineering created {len(all_feature_names)} total features")
        
        return X_train_final, X_test_final, y_train, y_test
    
    def _finalize_data_preparation_polars(self, X_train: pl.DataFrame, X_test: pl.DataFrame, y_train: np.ndarray, y_test: np.ndarray) -> Tuple[pl.DataFrame, pl.DataFrame, pd.Series, pd.Series]:
        """Finalize Polars data preparation - convert to format suitable for ML models"""
        
        # For now, most ML libraries expect pandas/numpy, so we convert for training
        # but maintain Polars DataFrames for feature engineering efficiency
        X_train_pandas = X_train.to_pandas() if POLARS_AVAILABLE and isinstance(X_train, pl.DataFrame) else X_train
        X_test_pandas = X_test.to_pandas() if POLARS_AVAILABLE and isinstance(X_test, pl.DataFrame) else X_test
        logger.info(f"Polars data prepared (converted to pandas): {len(X_train_pandas)} training samples, {len(X_test_pandas)} test samples")
        return X_train_pandas, X_test_pandas, y_train, y_test

    def _learn_categorical_levels(self, X: pd.DataFrame, categorical_features: List[str]) -> Dict[str, List[str]]:
        levels: Dict[str, List[str]] = {}
        for column in categorical_features:
            if column not in X.columns:
                continue
            series = X[column].fillna("__missing__").astype(str)
            ordered: List[str] = []
            for value in series:
                if value not in ordered:
                    ordered.append(value)
            if "__missing__" not in ordered:
                ordered.append("__missing__")
            ordered = [val for val in ordered if val != "__other__"]
            ordered.append("__other__")
            levels[column] = ordered
        return levels

    def _apply_categorical_levels(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.categorical_levels_:
            return X
        sanitized = X.copy()
        for column, allowed in self.categorical_levels_.items():
            if column not in sanitized.columns:
                continue
            series = sanitized[column].fillna("__missing__").astype(str)
            allowed_set = set(allowed)
            series = series.where(series.isin(allowed_set), "__other__")
            sanitized[column] = series
        return sanitized

    def _to_processed_dataframe(self, matrix: Any) -> pd.DataFrame:
        if isinstance(matrix, pd.DataFrame):
            return matrix
        if self.processed_feature_names:
            return pd.DataFrame(matrix, columns=self.processed_feature_names)
        return pd.DataFrame(matrix)

    def _needs_pandas_conversion(self) -> bool:
        """Check if we need to convert to pandas for model training"""
        # For now, most models need pandas conversion
        # This could be enhanced based on model type and library version
        return True
    
    def optimize_model_optuna(self,
                             model_name: str,
                             X_train: pd.DataFrame,
                             y_train: pd.Series,
                             n_trials: int = 100,
                             timeout: int = 3600) -> Dict[str, Any]:
        """Optimize model using Optuna"""
        
        model_config = self.models_config[model_name]
        
        def objective(trial: Trial) -> float:
            # Sample hyperparameters
            params = {}
            for param, values in model_config['param_space'].items():
                if isinstance(values[0], int):
                    params[param] = trial.suggest_int(param, min(values), max(values))
                elif isinstance(values[0], float):
                    params[param] = trial.suggest_float(param, min(values), max(values))
                else:
                    params[param] = trial.suggest_categorical(param, values)
            
            # Add fixed parameters
            params.update(model_config['fixed_params'])
            
            # Create and train model
            model = model_config['model_class'](**params)
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
                scoring='roc_auc_ovr',
                n_jobs=-1
            )
            
            return cv_scores.mean()
        
        # Create study
        study = create_study(
            direction='maximize',
            sampler=samplers.TPESampler(seed=self.random_state),
            pruner=pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        # Optimize
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        # Get best parameters
        best_params = study.best_params.copy()
        best_params.update(model_config['fixed_params'])
        
        return {
            'best_params': best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials)
        }
    
    def train_all_models(self,
                        X_train: pd.DataFrame,
                        y_train: pd.Series,
                        X_test: pd.DataFrame,
                        y_test: pd.Series,
                        optimize_hyperparams: bool = True) -> Dict[str, Any]:
        """Train all configured models"""
        
        results = {}
        if POLARS_AVAILABLE and isinstance(X_train, (pl.DataFrame, pl.LazyFrame)):
            if isinstance(X_train, pl.LazyFrame):
                X_train = X_train.collect(streaming=self.enable_streaming)
            if isinstance(X_test, pl.LazyFrame):
                X_test = X_test.collect(streaming=self.enable_streaming)
            X_train = X_train.to_pandas()
            X_test = X_test.to_pandas()

        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)
        class_count = len(np.unique(y_train)) if len(y_train) else 0

        # Apply feature engineering and preprocessing pipeline
        self.feature_engineer.fit(X_train)
        X_train_engineered = self.feature_engineer.transform(X_train)
        X_test_engineered = self.feature_engineer.transform(X_test)
        X_train, X_test, y_train, y_test = self._finalize_data_preparation_pandas(
            X_train_engineered, X_test_engineered, y_train, y_test
        )
        
        for model_name in self.models_config.keys():
            logger.info(f"Training {model_name}...")
            
            try:
                if optimize_hyperparams:
                    # Optimize hyperparameters
                    optimization_result = self.optimize_model_optuna(
                        model_name, X_train, y_train, n_trials=50
                    )
                    best_params = optimization_result['best_params']
                    logger.info(f"Best params for {model_name}: {best_params}")
                else:
                    # Use default parameters
                    best_params = self.models_config[model_name]['fixed_params'].copy()
                    if 'param_space' in self.models_config[model_name]:
                        # Use first value from param_space as default
                        for param, values in self.models_config[model_name]['param_space'].items():
                            best_params[param] = values[0]
                
                # Ensure multi-class settings are aligned with encoded targets
                if class_count > 1:
                    if model_name == 'xgboost_classifier':
                        best_params.setdefault('objective', 'multi:softprob')
                        best_params['num_class'] = class_count
                    elif model_name == 'lightgbm_classifier':
                        best_params.setdefault('objective', 'multiclass')
                        best_params['num_class'] = class_count
                    elif model_name == 'catboost_classifier':
                        best_params.setdefault('loss_function', 'MultiClass')
                        best_params.setdefault('verbose', False)

                # Train final model with best parameters
                model = self.models_config[model_name]['model_class'](**best_params)
                model.fit(X_train, y_train)
                
                # Evaluate model
                evaluation_results = self._evaluate_model(
                    model, model_name, X_train, y_train, X_test, y_test
                )
                
                # Store model and results
                self.trained_models[model_name] = model
                results[model_name] = {
                    'model': model,
                    'best_params': best_params,
                    'evaluation': evaluation_results
                }
                
                if optimize_hyperparams:
                    results[model_name]['optimization'] = optimization_result
                
                logger.info(f"{model_name} training completed. AUC: {evaluation_results['test_auc']:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        # Determine best model
        self._select_best_model(results)
        
        self.training_results = results
        return results
    
    def _evaluate_model(self,
                       model: Any,
                       model_name: str,
                       X_train: pd.DataFrame,
                       y_train: pd.Series,
                       X_test: pd.DataFrame,
                       y_test: pd.Series) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        
        # Predictions
        y_train_pred = np.asarray(model.predict(X_train))
        y_test_pred = np.asarray(model.predict(X_test))
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)
        
        # Probabilities (if available)
        try:
            y_train_proba = model.predict_proba(X_train)
            y_test_proba = model.predict_proba(X_test)
            has_proba = True
        except:
            has_proba = False
        
        # Basic metrics
        results = {
            'model_name': model_name,
            'train_accuracy': (y_train_pred == y_train).mean(),
            'test_accuracy': (y_test_pred == y_test).mean(),
        }
        
        # AUC metrics (for multiclass)
        if has_proba:
            try:
                results['train_auc'] = roc_auc_score(
                    y_train, y_train_proba, multi_class='ovr', average='weighted'
                )
                results['test_auc'] = roc_auc_score(
                    y_test, y_test_proba, multi_class='ovr', average='weighted'
                )
            except:
                results['train_auc'] = 0.0
                results['test_auc'] = 0.0
        
        # Classification report
        results['classification_report'] = classification_report(
            y_test,
            y_test_pred,
            output_dict=True,
            zero_division=0,
        )
        
        # Confusion matrix
        results['confusion_matrix'] = confusion_matrix(y_test, y_test_pred).tolist()
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            feature_names = X_train.columns.tolist()
            importance_dict = dict(zip(feature_names, model.feature_importances_))
            # Sort by importance
            sorted_importance = sorted(
                importance_dict.items(), key=lambda x: abs(x[1]), reverse=True
            )
            results['feature_importance'] = dict(sorted_importance[:20])  # Top 20
            
            # Store for later use
            self.feature_importance[model_name] = importance_dict
        
        return results
    
    def _select_best_model(self, results: Dict[str, Any]):
        """Select the best performing model"""
        
        best_score = -1
        best_model = None
        best_name = None
        
        for model_name, result in results.items():
            if 'error' in result:
                continue
            
            # Use test AUC as primary metric
            score = result['evaluation'].get('test_auc', 0)
            
            if score > best_score:
                best_score = score
                best_model = result['model']
                best_name = model_name
        
        self.best_model = best_model
        self.best_model_name = best_name
        self.model_scores = {name: res['evaluation'].get('test_auc', 0) 
                           for name, res in results.items() if 'error' not in res}
        
        if best_name:
            logger.info(f"Best model selected: {best_name} (AUC: {best_score:.4f})")
    
    def generate_shap_explanations(self, 
                                 X_sample: Union[pd.DataFrame, pl.DataFrame],
                                 model_name: Optional[str] = None,
                                 max_samples: int = 100) -> Dict[str, Any]:
        """Generate SHAP explanations for model predictions with Polars optimization"""
        
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            model = self.trained_models.get(model_name)

        if model is None:
            logger.error("generate_shap_explanations called without a trained model")
            return {'error': 'model_not_trained'}
        
        # Handle Polars DataFrame
        if isinstance(X_sample, (pl.DataFrame, pl.LazyFrame)):
            return self._generate_shap_explanations_polars(X_sample, model, model_name, max_samples)
        else:
            return self._generate_shap_explanations_pandas(X_sample, model, model_name, max_samples)
    
    def _generate_shap_explanations_pandas(self, X_sample: pd.DataFrame, model: Any, model_name: str, max_samples: int) -> Dict[str, Any]:
        """Generate SHAP explanations using pandas (original implementation)"""
        X_sample = self._apply_categorical_levels(X_sample)

        # Sample data for explanation
        X_explain = X_sample.sample(min(max_samples, len(X_sample)), random_state=self.random_state)
        
        # Prefer fast tree-based explanation when feature importances are available
        if hasattr(model, 'feature_importances_'):
            importances = np.abs(model.feature_importances_)
            feature_importance_shap = dict(zip(X_explain.columns, importances))
            sorted_features = sorted(feature_importance_shap.items(), key=lambda x: x[1], reverse=True)
            return {
                'model_name': model_name,
                'shap_values': importances,
                'feature_importance_shap': dict(sorted_features[:20]),
                'explanation_samples': len(X_explain),
                'base_values': None,
                'data_format': 'pandas'
            }

        try:
            if hasattr(model, 'predict_proba'):
                explainer = shap.Explainer(model.predict_proba, X_explain.iloc[:20])
            else:
                explainer = shap.Explainer(model.predict, X_explain.iloc[:20])

            shap_values = explainer(X_explain)

            if isinstance(shap_values.values, list):
                mean_shap = np.mean([np.abs(shap_vals).mean(0) for shap_vals in shap_values.values], axis=0)
            else:
                mean_shap = np.abs(shap_values.values).mean(0)

            feature_importance_shap = dict(zip(X_explain.columns, mean_shap))
            sorted_features = sorted(feature_importance_shap.items(), key=lambda x: x[1], reverse=True)

            return {
                'model_name': model_name,
                'shap_values': shap_values,
                'feature_importance_shap': dict(sorted_features[:20]),
                'explanation_samples': len(X_explain),
                'base_values': getattr(explainer, 'expected_value', None),
                'data_format': 'pandas'
            }

        except Exception as e:
            logger.error(f"Error generating SHAP explanations: {str(e)}")
            return {'error': str(e)}
    
    def _generate_shap_explanations_polars(self, X_sample: Union[pl.DataFrame, pl.LazyFrame], model: Any, model_name: str, max_samples: int) -> Dict[str, Any]:
        """Generate SHAP explanations using Polars for efficient data handling"""
        
        try:
            # Ensure we have a DataFrame (not LazyFrame)
            if isinstance(X_sample, pl.LazyFrame):
                X_sample = X_sample.collect(streaming=self.enable_streaming)
            
            # Sample data efficiently using Polars
            sample_size = min(max_samples, len(X_sample))
            X_explain_pl = X_sample.sample(n=sample_size, seed=self.random_state)
            
            # Convert to pandas for SHAP (most SHAP explainers expect pandas/numpy)
            X_explain_pd = X_explain_pl.to_pandas()
            X_explain_pd = self._apply_categorical_levels(X_explain_pd)
            
            # Use pandas SHAP implementation but track that we preprocessed with Polars
            shap_result = self._generate_shap_explanations_pandas(X_explain_pd, model, model_name, sample_size)
            
            # Add Polars-specific metadata
            if 'error' not in shap_result:
                shap_result.update({
                    'data_format': 'polars',
                    'polars_preprocessing': True,
                    'original_rows': len(X_sample),
                    'polars_sampling_time': self._benchmark_polars_sampling(X_sample, sample_size)
                })
            
            return shap_result
            
        except Exception as e:
            logger.error(f"Error generating Polars SHAP explanations: {str(e)}")
            # Fallback to pandas conversion
            try:
                X_sample_pd = X_sample.to_pandas() if hasattr(X_sample, 'to_pandas') else X_sample.collect().to_pandas()
                result = self._generate_shap_explanations_pandas(X_sample_pd, model, model_name, max_samples)
                result['fallback_to_pandas'] = True
                return result
            except Exception as fallback_error:
                return {'error': f"Polars: {str(e)}, Pandas fallback: {str(fallback_error)}"}
    
    def _benchmark_polars_sampling(self, X: pl.DataFrame, sample_size: int) -> float:
        """Benchmark Polars sampling performance"""
        import time
        
        start_time = time.time()
        _ = X.sample(n=sample_size, seed=self.random_state)
        return time.time() - start_time
    
    def save_models(self, save_path: Union[str, Path]) -> Dict[str, str]:
        """Save trained models and preprocessing components"""
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        try:
            # Save trained models
            models_path = save_path / 'models'
            models_path.mkdir(exist_ok=True)
            
            for model_name, model in self.trained_models.items():
                model_file = models_path / f"{model_name}.pkl"
                with open(model_file, 'wb') as f:
                    pickle.dump(model, f)
                saved_files[f'model_{model_name}'] = str(model_file)
            
            # Save preprocessing components
            preprocessing_path = save_path / 'preprocessing'
            preprocessing_path.mkdir(exist_ok=True)
            
            if self.feature_engineer:
                fe_file = preprocessing_path / 'feature_engineer.pkl'
                with open(fe_file, 'wb') as f:
                    pickle.dump(self.feature_engineer, f)
                saved_files['feature_engineer'] = str(fe_file)
            
            if self.preprocessor:
                prep_file = preprocessing_path / 'preprocessor.pkl'
                with open(prep_file, 'wb') as f:
                    pickle.dump(self.preprocessor, f)
                saved_files['preprocessor'] = str(prep_file)
            
            # Save training results and metadata
            metadata = {
                'best_model_name': self.best_model_name,
                'model_scores': self.model_scores,
                'feature_importance': self.feature_importance,
                'training_timestamp': datetime.now().isoformat(),
                'models_config': self.models_config,
                'categorical_levels': self.categorical_levels_,
                'categorical_features': self._categorical_features,
                'processed_feature_names': self.processed_feature_names,
            }
            
            metadata_file = save_path / 'training_metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            saved_files['metadata'] = str(metadata_file)
            
            logger.info(f"Models and components saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            saved_files['error'] = str(e)
        
        return saved_files
    
    def load_models(self, load_path: Union[str, Path]) -> bool:
        """Load trained models and preprocessing components"""
        
        load_path = Path(load_path)
        
        try:
            # Load metadata
            metadata_file = load_path / 'training_metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                self.best_model_name = metadata.get('best_model_name')
                self.model_scores = metadata.get('model_scores', {})
                self.feature_importance = metadata.get('feature_importance', {})
                self.models_config = metadata.get('models_config', self.models_config)
                self.categorical_levels_ = metadata.get('categorical_levels', {})
                self._categorical_features = metadata.get('categorical_features', [])
                self.processed_feature_names = metadata.get('processed_feature_names', [])
            
            # Load trained models
            models_path = load_path / 'models'
            if models_path.exists():
                for model_file in models_path.glob('*.pkl'):
                    model_name = model_file.stem
                    with open(model_file, 'rb') as f:
                        model = pickle.load(f)
                    self.trained_models[model_name] = model
                    
                    if model_name == self.best_model_name:
                        self.best_model = model
            
            # Load preprocessing components
            preprocessing_path = load_path / 'preprocessing'
            if preprocessing_path.exists():
                
                fe_file = preprocessing_path / 'feature_engineer.pkl'
                if fe_file.exists():
                    with open(fe_file, 'rb') as f:
                        self.feature_engineer = pickle.load(f)
                
                prep_file = preprocessing_path / 'preprocessor.pkl'
                if prep_file.exists():
                    with open(prep_file, 'rb') as f:
                        self.preprocessor = pickle.load(f)
            
            logger.info(f"Models and components loaded from {load_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
    
    def predict_batch(self, X: Union[pd.DataFrame, pl.DataFrame], return_probabilities: bool = True, batch_size: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Make batch predictions using the best model with streaming support for large datasets"""
        
        if self.best_model is None:
            raise ValueError("No trained model available. Please train models first.")
        
        # Handle large datasets with streaming if enabled and batch_size specified
        if batch_size is not None and len(X) > batch_size:
            return self._predict_batch_streaming(X, return_probabilities, batch_size)
        
        # Standard batch prediction
        if isinstance(X, (pl.DataFrame, pl.LazyFrame)):
            return self._predict_batch_polars(X, return_probabilities)
        else:
            return self._predict_batch_pandas(X, return_probabilities)
    
    def _predict_batch_pandas(self, X: pd.DataFrame, return_probabilities: bool) -> Dict[str, np.ndarray]:
        """Standard pandas batch prediction"""
        
        # Apply feature engineering and preprocessing
        X_engineered = self.feature_engineer.transform(X)
        if hasattr(X_engineered, "to_pandas") and not isinstance(X_engineered, pd.DataFrame):
            X_engineered = X_engineered.to_pandas()
        X_engineered = self._apply_categorical_levels(X_engineered)
        X_processed = self.preprocessor.transform(X_engineered)
        X_processed = self._to_processed_dataframe(X_processed)
        
        # Make predictions
        predictions = self.best_model.predict(X_processed)
        
        result = {
            'predictions': predictions,
            'model_used': self.best_model_name,
            'num_samples': len(X),
            'processing_method': 'pandas_standard'
        }
        
        if return_probabilities and hasattr(self.best_model, 'predict_proba'):
            probabilities = self.best_model.predict_proba(X_processed)
            result['probabilities'] = probabilities
        
        return result
    
    def _predict_batch_polars(self, X: Union[pl.DataFrame, pl.LazyFrame], return_probabilities: bool) -> Dict[str, np.ndarray]:
        """Polars-optimized batch prediction"""
        
        # Collect LazyFrame if needed
        if isinstance(X, pl.LazyFrame):
            X = X.collect(streaming=self.enable_streaming)
        
        # Apply feature engineering with Polars optimization
        X_engineered = self.feature_engineer.transform(X)
        
        # Convert to pandas for model prediction (most models expect this)
        if self._needs_pandas_conversion():
            X_pandas = X_engineered.to_pandas()
            X_pandas = self._apply_categorical_levels(X_pandas)
            X_processed = self.preprocessor.transform(X_pandas)
            X_processed = self._to_processed_dataframe(X_processed)
        else:
            X_processed = X_engineered

        # Make predictions
        predictions = self.best_model.predict(X_processed)
        
        result = {
            'predictions': predictions,
            'model_used': self.best_model_name,
            'num_samples': len(X),
            'processing_method': 'polars_optimized'
        }
        
        if return_probabilities and hasattr(self.best_model, 'predict_proba'):
            probabilities = self.best_model.predict_proba(X_processed)
            result['probabilities'] = probabilities
        
        return result
    
    def _predict_batch_streaming(self, X: Union[pd.DataFrame, pl.DataFrame], return_probabilities: bool, batch_size: int) -> Dict[str, np.ndarray]:
        """Streaming batch prediction for large datasets"""
        
        logger.info(f"Starting streaming prediction for {len(X)} samples with batch size {batch_size}")
        
        # Initialize results storage
        all_predictions = []
        all_probabilities = [] if return_probabilities else None
        total_batches = (len(X) + batch_size - 1) // batch_size
        
        # Process in batches
        for i in range(total_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X))
            
            # Get batch
            if isinstance(X, (pl.DataFrame, pl.LazyFrame)):
                batch = X.slice(start_idx, end_idx - start_idx)
                batch_result = self._predict_batch_polars(batch, return_probabilities)
            else:
                batch = X.iloc[start_idx:end_idx]
                batch_result = self._predict_batch_pandas(batch, return_probabilities)
            
            # Collect results
            all_predictions.append(batch_result['predictions'])
            if return_probabilities and 'probabilities' in batch_result:
                all_probabilities.append(batch_result['probabilities'])
            
            # Progress logging
            if (i + 1) % 10 == 0:
                logger.info(f"Processed batch {i + 1}/{total_batches}")
        
        # Combine all results
        combined_predictions = np.concatenate(all_predictions)
        
        result = {
            'predictions': combined_predictions,
            'model_used': self.best_model_name,
            'num_samples': len(X),
            'processing_method': 'streaming',
            'batch_size': batch_size,
            'num_batches': total_batches
        }
        
        if return_probabilities and all_probabilities:
            combined_probabilities = np.concatenate(all_probabilities)
            result['probabilities'] = combined_probabilities
        
        logger.info(f"Streaming prediction completed: {len(combined_predictions)} predictions")
        return result

def create_ifrs9_ml_pipeline(random_state: int = 42) -> OptimizedMLPipeline:
    """Create a pre-configured ML pipeline for IFRS9 credit risk"""
    
    # Enhanced configuration for IFRS9 specific models
    ifrs9_models_config = {
        'xgboost_ifrs9': {
            'model_class': xgb.XGBClassifier,
            'param_space': {
                'n_estimators': [200, 300, 500],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9],
                'reg_alpha': [0.1, 1.0, 2.0],
                'reg_lambda': [0.1, 1.0, 2.0],
                'min_child_weight': [1, 3, 5]
            },
            'fixed_params': {
                'objective': 'multi:softprob',
                'random_state': random_state,
                'n_jobs': -1,
                'eval_metric': 'mlogloss'
            }
        },
        'lightgbm_ifrs9': {
            'model_class': lgb.LGBMClassifier,
            'param_space': {
                'n_estimators': [200, 300, 500],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.15],
                'num_leaves': [31, 63, 127],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9],
                'reg_alpha': [0.1, 1.0],
                'reg_lambda': [0.1, 1.0],
                'min_child_samples': [20, 30, 40]
            },
            'fixed_params': {
                'objective': 'multiclass',
                'random_state': random_state,
                'n_jobs': -1,
                'verbosity': -1,
                'boosting_type': 'gbdt'
            }
        },
        'catboost_ifrs9': {
            'model_class': CatBoostClassifier,
            'param_space': {
                'iterations': [300, 500, 700],
                'depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.15],
                'l2_leaf_reg': [1, 3, 5],
                'border_count': [128, 254],
                'bagging_temperature': [0.5, 1.0]
            },
            'fixed_params': {
                'loss_function': 'MultiClass',
                'random_state': random_state,
                'verbose': False,
                'thread_count': -1
            }
        }
    }
    
    return OptimizedMLPipeline(
        models_config=ifrs9_models_config,
        optimization_method="optuna",
        cv_folds=5,
        random_state=random_state
    )

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced IFRS9 ML Pipeline')
    parser.add_argument('--data-path', required=True, help='Path to training data CSV')
    parser.add_argument('--target-column', default='provision_stage', help='Target column name')
    parser.add_argument('--output-dir', default='models/', help='Output directory for models')
    parser.add_argument('--optimize', action='store_true', help='Perform hyperparameter optimization')
    parser.add_argument('--generate-shap', action='store_true', help='Generate SHAP explanations')
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.data_path}")
    df = pd.read_csv(args.data_path)
    
    # Create pipeline
    pipeline = create_ifrs9_ml_pipeline()
    
    # Prepare data
    X_train, X_test, y_train, y_test = pipeline.prepare_data(
        df, target_column=args.target_column
    )
    
    # Train models
    logger.info("Training models...")
    results = pipeline.train_all_models(
        X_train, y_train, X_test, y_test,
        optimize_hyperparams=args.optimize
    )
    
    # Print results summary
    print("\n" + "="*50)
    print("TRAINING RESULTS SUMMARY")
    print("="*50)
    
    for model_name, result in results.items():
        if 'error' not in result:
            auc = result['evaluation']['test_auc']
            acc = result['evaluation']['test_accuracy']
            print(f"{model_name:20s}: AUC={auc:.4f}, Accuracy={acc:.4f}")
        else:
            print(f"{model_name:20s}: ERROR - {result['error']}")
    
    if pipeline.best_model_name:
        print(f"\nBest Model: {pipeline.best_model_name}")
    
    # Generate SHAP explanations
    if args.generate_shap and pipeline.best_model:
        logger.info("Generating SHAP explanations...")
        shap_results = pipeline.generate_shap_explanations(X_test.head(50))
        
        if 'error' not in shap_results:
            print("\nTop 10 Most Important Features (SHAP):")
            print("-" * 40)
            for feature, importance in list(shap_results['feature_importance_shap'].items())[:10]:
                print(f"{feature:25s}: {importance:.4f}")
    
    # Save models
    save_results = pipeline.save_models(args.output_dir)
    logger.info(f"Models saved: {list(save_results.keys())}")
    
    print(f"\nTraining completed! Models saved to: {args.output_dir}")
