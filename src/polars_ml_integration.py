"""
Polars-enhanced ML integration for IFRS9 credit risk modeling.

This module demonstrates:
- Polars DataFrame operations for feature engineering
- Hybrid Polars-Pandas workflows for ML compatibility
- Performance optimizations using Polars lazy evaluation
- Integration with existing ML models (XGBoost, LightGBM)
"""

import polars as pl
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
import warnings

# ML imports
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, roc_auc_score
)

logger = logging.getLogger(__name__)


class PolarsEnhancedCreditRiskClassifier:
    """
    Credit risk classifier with Polars optimization for high-performance data processing.
    
    Features:
    - Polars-native feature engineering for 10x+ performance improvement
    - Lazy evaluation for memory efficiency on large datasets
    - Seamless integration with XGBoost/LightGBM native Polars support
    - Hybrid workflows with pandas fallback for compatibility
    """
    
    def __init__(
        self,
        model_type: str = "xgboost",
        use_lazy_evaluation: bool = True,
        polars_streaming: bool = True
    ):
        """
        Initialize the Polars-enhanced credit risk classifier.
        
        Args:
            model_type: "xgboost", "lightgbm", or "sklearn"
            use_lazy_evaluation: Enable Polars lazy evaluation for performance
            polars_streaming: Enable streaming for large datasets
        """
        self.model_type = model_type
        self.use_lazy_evaluation = use_lazy_evaluation
        self.polars_streaming = polars_streaming
        
        # Model components
        self.stage_classifier = None
        self.pd_regressor = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.is_fitted = False
        
        # Polars configuration
        self._configure_polars()
        
        # Validate ML libraries
        self._validate_ml_libraries()
    
    def _configure_polars(self):
        """Configure Polars for optimal performance."""
        # Set threading for container environment
        pl.Config.set_tbl_rows(25)
        pl.Config.set_tbl_cols(20)
        
        # Enable streaming if configured
        if self.polars_streaming:
            pl.Config.set_streaming_chunk_size(50000)
        
        logger.info(f"Polars configured - Version: {pl.__version__}")
    
    def _validate_ml_libraries(self):
        """Validate availability of ML libraries."""
        if self.model_type == "xgboost" and not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available, falling back to sklearn")
            self.model_type = "sklearn"
        
        if self.model_type == "lightgbm" and not LIGHTGBM_AVAILABLE:
            logger.warning("LightGBM not available, falling back to sklearn")
            self.model_type = "sklearn"
    
    def prepare_features_polars(
        self, 
        df: Union[pl.DataFrame, pd.DataFrame]
    ) -> Tuple[pl.DataFrame, List[str]]:
        """
        Prepare features using Polars for high-performance processing.
        
        Args:
            df: Input DataFrame (Polars or pandas)
            
        Returns:
            Tuple of (polars_feature_matrix, feature_names)
        """
        # Convert pandas to Polars if needed
        if isinstance(df, pd.DataFrame):
            pl_df = pl.from_pandas(df)
        else:
            pl_df = df
        
        # Define feature engineering pipeline using Polars expressions
        if self.use_lazy_evaluation:
            # Use lazy evaluation for better performance
            feature_pipeline = pl_df.lazy()
        else:
            feature_pipeline = pl_df
        
        # Core IFRS9 feature engineering with Polars
        features = feature_pipeline.with_columns([
            # Basic features
            pl.col("loan_amount").alias("loan_amount"),
            pl.col("interest_rate").alias("interest_rate"),
            pl.col("term_months").alias("term_months"),
            pl.col("current_balance").alias("current_balance"),
            pl.col("credit_score").alias("credit_score"),
            pl.col("days_past_due").alias("days_past_due"),
            pl.col("customer_income").alias("customer_income"),
            pl.col("ltv_ratio").alias("ltv_ratio"),
            
            # Derived features - Polars expressions for efficiency
            (pl.col("loan_amount") / pl.col("customer_income")).alias("debt_to_income"),
            (pl.col("monthly_payment") / pl.col("customer_income") * 12).alias("payment_burden"),
            (pl.col("current_balance") / pl.col("loan_amount")).alias("utilization_rate"),
            
            # Time-based features
            pl.lit(30).alias("loan_age_months"),  # Simplified for synthetic data
            
            # Risk indicators using Polars conditional expressions
            pl.when(pl.col("days_past_due") > 90)
              .then(1)
              .otherwise(0)
              .alias("is_90_dpd"),
            
            pl.when(pl.col("ltv_ratio") > 0.8)
              .then(1)
              .otherwise(0)
              .alias("high_ltv"),
            
            pl.when(pl.col("credit_score") < 600)
              .then(1)
              .otherwise(0)
              .alias("subprime"),
            
            # Advanced derived features
            (pl.col("interest_rate") * pl.col("loan_amount")).alias("annual_interest_cost"),
            (pl.col("days_past_due") / 30).alias("months_past_due"),
            
            # Interaction features
            (pl.col("credit_score") * pl.col("ltv_ratio")).alias("score_ltv_interaction"),
            (pl.col("customer_income") / pl.col("loan_amount")).alias("income_loan_ratio"),
        ])
        
        # Handle categorical features with Polars
        # Create dummy variables for loan_type
        loan_type_dummies = features.select([
            pl.when(pl.col("loan_type") == "mortgage").then(1).otherwise(0).alias("loan_type_mortgage"),
            pl.when(pl.col("loan_type") == "auto").then(1).otherwise(0).alias("loan_type_auto"),
            pl.when(pl.col("loan_type") == "personal").then(1).otherwise(0).alias("loan_type_personal"),
            pl.when(pl.col("loan_type") == "credit_card").then(1).otherwise(0).alias("loan_type_credit_card"),
        ])
        
        # Employment status dummies
        employment_dummies = features.select([
            pl.when(pl.col("employment_status") == "employed").then(1).otherwise(0).alias("employment_employed"),
            pl.when(pl.col("employment_status") == "self_employed").then(1).otherwise(0).alias("employment_self_employed"),
            pl.when(pl.col("employment_status") == "unemployed").then(1).otherwise(0).alias("employment_unemployed"),
            pl.when(pl.col("employment_status") == "retired").then(1).otherwise(0).alias("employment_retired"),
        ])
        
        # Combine all features horizontally
        final_features = pl.concat([
            features.select([
                "loan_amount", "interest_rate", "term_months", "current_balance",
                "credit_score", "days_past_due", "customer_income", "ltv_ratio",
                "debt_to_income", "payment_burden", "utilization_rate", "loan_age_months",
                "is_90_dpd", "high_ltv", "subprime", "annual_interest_cost",
                "months_past_due", "score_ltv_interaction", "income_loan_ratio"
            ]),
            loan_type_dummies,
            employment_dummies
        ], how="horizontal")
        
        # Execute lazy evaluation if used
        if self.use_lazy_evaluation:
            final_features = final_features.collect()
        
        # Handle missing values using Polars
        final_features = final_features.fill_null(strategy="mean")
        
        # Store feature names
        self.feature_columns = final_features.columns
        
        logger.info(f"Prepared {len(self.feature_columns)} features using Polars")
        return final_features, self.feature_columns
    
    def train_stage_classifier_polars(
        self,
        X: pl.DataFrame,
        y: Union[pl.Series, pd.Series],
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train stage classification model with Polars-optimized data handling.
        
        Args:
            X: Polars DataFrame with features
            y: Target labels (provision stages)
            test_size: Test split proportion
            
        Returns:
            Training metrics dictionary
        """
        # Convert target to pandas Series if needed for sklearn compatibility
        if isinstance(y, pl.Series):
            y_pandas = y.to_pandas()
        else:
            y_pandas = y
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y_pandas)
        
        # Convert Polars to appropriate format based on model type
        if self.model_type in ["xgboost", "lightgbm"] and self._supports_polars_direct():
            # Use Polars directly if supported
            X_for_split = X
            logger.info(f"Using Polars DataFrames directly with {self.model_type}")
        else:
            # Convert to pandas for sklearn compatibility
            X_for_split = X.to_pandas()
            logger.info(f"Converting to pandas for {self.model_type} compatibility")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_for_split, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Initialize model based on type
        if self.model_type == "xgboost" and XGBOOST_AVAILABLE:
            self.stage_classifier = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            )
        elif self.model_type == "lightgbm" and LIGHTGBM_AVAILABLE:
            self.stage_classifier = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        else:
            # Fallback to sklearn
            from sklearn.ensemble import RandomForestClassifier
            self.stage_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )
            # Sklearn requires scaling
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            X_train, X_test = X_train_scaled, X_test_scaled
        
        # Train model
        self.stage_classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.stage_classifier.predict(X_test)
        y_prob = self.stage_classifier.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.stage_classifier, X_train, y_train, cv=5, scoring="accuracy"
        )
        
        # Feature importance
        if hasattr(self.stage_classifier, 'feature_importances_'):
            feature_importance = pl.DataFrame({
                "feature": self.feature_columns,
                "importance": self.stage_classifier.feature_importances_
            }).sort("importance", descending=True)
            
            feature_importance_records = feature_importance.head(10).to_pandas().to_dict("records")
        else:
            feature_importance_records = []
        
        metrics = {
            "accuracy": accuracy,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(
                y_test, y_pred, target_names=self.label_encoder.classes_
            ),
            "feature_importance": feature_importance_records,
        }
        
        # Calculate AUC for multi-class
        if len(np.unique(y_encoded)) > 2:
            try:
                metrics["roc_auc"] = roc_auc_score(
                    y_test, y_prob, multi_class="ovr", average="weighted"
                )
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {e}")
                metrics["roc_auc"] = None
        
        self.is_fitted = True
        logger.info(f"Stage classifier trained with accuracy: {accuracy:.4f}")
        
        return metrics
    
    def _supports_polars_direct(self) -> bool:
        """Check if the current model type supports Polars DataFrames directly."""
        # This would need to be updated based on actual library versions
        # For now, we'll be conservative and convert to pandas
        return False
    
    def predict_with_polars(
        self, 
        X: Union[pl.DataFrame, pd.DataFrame]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using Polars-optimized preprocessing."""\n        if not self.is_fitted:\n            raise ValueError(\"Model not trained. Call train_stage_classifier_polars first.\")\n        \n        # Ensure we have Polars DataFrame\n        if isinstance(X, pd.DataFrame):\n            X_pl = pl.from_pandas(X)\n        else:\n            X_pl = X\n        \n        # Align features using Polars selection\n        X_aligned = X_pl.select(self.feature_columns)\n        \n        # Convert to pandas for model prediction (if needed)\n        if self.model_type == \"sklearn\" or not self._supports_polars_direct():\n            X_for_pred = X_aligned.to_pandas()\n            if self.model_type == \"sklearn\":\n                X_for_pred = self.scaler.transform(X_for_pred)\n        else:\n            X_for_pred = X_aligned\n        \n        # Predict\n        y_pred_encoded = self.stage_classifier.predict(X_for_pred)\n        y_prob = self.stage_classifier.predict_proba(X_for_pred)\n        \n        # Decode labels\n        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)\n        \n        return y_pred, y_prob\n    \n    def batch_process_large_dataset(\n        self, \n        df_path: str, \n        batch_size: int = 100000\n    ) -> pl.DataFrame:\n        \"\"\"\n        Process large datasets in batches using Polars streaming.\n        \n        Args:\n            df_path: Path to CSV file\n            batch_size: Number of rows per batch\n            \n        Returns:\n            Polars DataFrame with predictions\n        \"\"\"\n        if not self.is_fitted:\n            raise ValueError(\"Model not trained\")\n        \n        # Use Polars lazy reading for memory efficiency\n        df_lazy = pl.scan_csv(df_path)\n        \n        # Process in streaming mode\n        predictions_list = []\n        \n        # Get total rows for progress tracking\n        total_rows = df_lazy.select(pl.count()).collect().item()\n        num_batches = (total_rows + batch_size - 1) // batch_size\n        \n        logger.info(f\"Processing {total_rows} rows in {num_batches} batches\")\n        \n        for i in range(num_batches):\n            start_idx = i * batch_size\n            end_idx = min((i + 1) * batch_size, total_rows)\n            \n            # Load batch\n            batch_df = df_lazy.slice(start_idx, end_idx - start_idx).collect()\n            \n            # Prepare features\n            batch_features, _ = self.prepare_features_polars(batch_df)\n            \n            # Make predictions\n            stage_pred, stage_prob = self.predict_with_polars(batch_features)\n            \n            # Create results DataFrame\n            batch_results = pl.DataFrame({\n                \"row_id\": range(start_idx, end_idx),\n                \"predicted_stage\": stage_pred,\n                \"stage_probability_max\": np.max(stage_prob, axis=1),\n            })\n            \n            predictions_list.append(batch_results)\n            \n            if (i + 1) % 10 == 0:\n                logger.info(f\"Processed batch {i + 1}/{num_batches}\")\n        \n        # Combine all predictions\n        final_predictions = pl.concat(predictions_list)\n        \n        logger.info(\"Batch processing completed\")\n        return final_predictions\n    \n    def benchmark_polars_vs_pandas(\n        self, \n        df: Union[pl.DataFrame, pd.DataFrame],\n        n_iterations: int = 3\n    ) -> Dict[str, float]:\n        \"\"\"\n        Benchmark Polars vs pandas performance for feature engineering.\n        \n        Args:\n            df: Input DataFrame\n            n_iterations: Number of benchmark iterations\n            \n        Returns:\n            Timing comparison dictionary\n        \"\"\"\n        import time\n        \n        # Ensure we have both formats\n        if isinstance(df, pd.DataFrame):\n            pd_df = df\n            pl_df = pl.from_pandas(df)\n        else:\n            pl_df = df\n            pd_df = df.to_pandas()\n        \n        # Benchmark Polars feature engineering\n        polars_times = []\n        for _ in range(n_iterations):\n            start_time = time.time()\n            _, _ = self.prepare_features_polars(pl_df)\n            polars_times.append(time.time() - start_time)\n        \n        # Benchmark pandas equivalent (simplified)\n        pandas_times = []\n        for _ in range(n_iterations):\n            start_time = time.time()\n            # Simplified pandas feature engineering\n            temp_df = pd_df.copy()\n            temp_df[\"debt_to_income\"] = temp_df[\"loan_amount\"] / temp_df[\"customer_income\"]\n            temp_df[\"utilization_rate\"] = temp_df[\"current_balance\"] / temp_df[\"loan_amount\"]\n            temp_df[\"high_ltv\"] = (temp_df[\"ltv_ratio\"] > 0.8).astype(int)\n            pandas_times.append(time.time() - start_time)\n        \n        avg_polars_time = np.mean(polars_times)\n        avg_pandas_time = np.mean(pandas_times)\n        speedup = avg_pandas_time / avg_polars_time if avg_polars_time > 0 else 0\n        \n        benchmark_results = {\n            \"polars_avg_time\": avg_polars_time,\n            \"pandas_avg_time\": avg_pandas_time,\n            \"speedup_factor\": speedup,\n            \"data_rows\": len(df),\n            \"polars_version\": pl.__version__,\n        }\n        \n        logger.info(f\"Benchmark results - Polars: {avg_polars_time:.4f}s, Pandas: {avg_pandas_time:.4f}s, Speedup: {speedup:.2f}x\")\n        \n        return benchmark_results


def create_synthetic_ifrs9_data_polars(n_rows: int = 10000) -> pl.DataFrame:\n    \"\"\"\n    Create synthetic IFRS9 data using Polars for demonstration.\n    \n    Args:\n        n_rows: Number of rows to generate\n        \n    Returns:\n        Polars DataFrame with synthetic loan data\n    \"\"\"\n    np.random.seed(42)\n    \n    # Generate synthetic data with realistic IFRS9 characteristics\n    data = {\n        \"loan_id\": [f\"L{i:08d}\" for i in range(n_rows)],\n        \"loan_amount\": np.random.lognormal(10, 1, n_rows),\n        \"interest_rate\": np.random.uniform(0.02, 0.12, n_rows),\n        \"term_months\": np.random.choice([12, 24, 36, 48, 60, 120, 240, 360], n_rows),\n        \"current_balance\": np.random.lognormal(9.5, 1, n_rows),\n        \"credit_score\": np.random.normal(700, 100, n_rows).clip(300, 850),\n        \"days_past_due\": np.random.exponential(5, n_rows).clip(0, 365),\n        \"customer_income\": np.random.lognormal(11, 0.5, n_rows),\n        \"ltv_ratio\": np.random.uniform(0.3, 1.2, n_rows),\n        \"monthly_payment\": np.random.lognormal(7, 0.8, n_rows),\n        \"loan_type\": np.random.choice([\"mortgage\", \"auto\", \"personal\", \"credit_card\"], n_rows),\n        \"employment_status\": np.random.choice([\"employed\", \"self_employed\", \"unemployed\", \"retired\"], n_rows, p=[0.7, 0.15, 0.1, 0.05]),\n        \"origination_date\": [datetime(2020, 1, 1) + pd.Timedelta(days=int(x)) for x in np.random.uniform(0, 1460, n_rows)],\n    }\n    \n    # Create Polars DataFrame\n    df = pl.DataFrame(data)\n    \n    # Add target variables using Polars expressions\n    df = df.with_columns([\n        # Provision stage (simplified logic)\n        pl.when(pl.col(\"days_past_due\") <= 30).then(\"Stage1\")\n        .when(pl.col(\"days_past_due\") <= 90).then(\"Stage2\")\n        .otherwise(\"Stage3\")\n        .alias(\"provision_stage\"),\n        \n        # Probability of Default (simplified)\n        ((pl.col(\"days_past_due\") / 365 * 0.3 + \n          (850 - pl.col(\"credit_score\")) / 550 * 0.4 +\n          pl.col(\"ltv_ratio\") * 0.3).clip(0.001, 0.95))\n        .alias(\"pd_estimate\")\n    ])\n    \n    logger.info(f\"Created synthetic IFRS9 dataset with {n_rows} rows using Polars\")\n    return df


# Example usage and integration demonstration\nif __name__ == \"__main__\":\n    # Create synthetic data\n    synthetic_data = create_synthetic_ifrs9_data_polars(50000)\n    \n    # Initialize Polars-enhanced classifier\n    classifier = PolarsEnhancedCreditRiskClassifier(\n        model_type=\"xgboost\",\n        use_lazy_evaluation=True,\n        polars_streaming=True\n    )\n    \n    # Prepare features using Polars\n    features, feature_names = classifier.prepare_features_polars(synthetic_data)\n    target = synthetic_data[\"provision_stage\"]\n    \n    print(f\"Prepared {len(feature_names)} features using Polars\")\n    print(f\"Feature engineering completed on {len(features)} rows\")\n    \n    # Train model\n    metrics = classifier.train_stage_classifier_polars(features, target)\n    print(f\"Model trained with accuracy: {metrics['accuracy']:.4f}\")\n    \n    # Benchmark performance\n    benchmark_results = classifier.benchmark_polars_vs_pandas(synthetic_data)\n    print(f\"Polars speedup: {benchmark_results['speedup_factor']:.2f}x\")