"""Machine Learning models for credit risk classification.

This module implements ML models for predicting credit risk stages
and probability of default for the IFRS9 system.
"""

import pickle
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Set up logging
logger = logging.getLogger(__name__)


class CreditRiskClassifier:
    """Machine Learning models for credit risk assessment.
    
    This class provides:
    - Stage classification model
    - Probability of Default (PD) prediction
    - Model training and evaluation
    - Feature importance analysis
    """
    
    def __init__(self, model_type: str = "random_forest"):
        """Initialize the credit risk classifier.
        
        Args:
            model_type: Type of model to use ("random_forest" or "gradient_boosting")
        """
        self.model_type = model_type
        self.stage_classifier = None
        self.pd_regressor = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.is_fitted = False
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare features for model training.
        
        Args:
            df: Raw DataFrame with loan data
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        # Select relevant features
        feature_cols = [
            "loan_amount",
            "interest_rate",
            "term_months",
            "current_balance",
            "credit_score",
            "days_past_due",
            "customer_income",
            "ltv_ratio",
        ]
        
        # Create derived features
        df_features = df.copy()
        
        # Debt-to-income ratio
        df_features["debt_to_income"] = (
            df_features["loan_amount"] / df_features["customer_income"]
        )
        
        # Payment burden
        df_features["payment_burden"] = (
            df_features["monthly_payment"] / df_features["customer_income"] * 12
        )
        
        # Loan age (simplified as random for synthetic data)
        df_features["loan_age_months"] = np.random.randint(0, 60, len(df))
        
        # Utilization rate (for credit cards)
        df_features["utilization_rate"] = (
            df_features["current_balance"] / df_features["loan_amount"]
        )
        
        # Add categorical features as dummies
        loan_type_dummies = pd.get_dummies(df["loan_type"], prefix="loan_type")
        employment_dummies = pd.get_dummies(df["employment_status"], prefix="employment")
        
        # Combine features
        feature_cols.extend(["debt_to_income", "payment_burden", "loan_age_months", "utilization_rate"])
        
        X = pd.concat([
            df_features[feature_cols],
            loan_type_dummies,
            employment_dummies
        ], axis=1)
        
        # Handle missing values
        X = X.fillna(X.median())
        
        self.feature_columns = X.columns.tolist()
        
        return X, self.feature_columns
    
    def train_stage_classifier(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """Train the stage classification model.
        
        Args:
            X: Feature matrix
            y: Target labels (provision stages)
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with training metrics
        """
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize and train model
        if self.model_type == "random_forest":
            self.stage_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Train model
        self.stage_classifier.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.stage_classifier.predict(X_test_scaled)
        y_prob = self.stage_classifier.predict_proba(X_test_scaled)
        
        # Calculate metrics
        accuracy = (y_pred == y_test).mean()
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.stage_classifier, X_train_scaled, y_train, cv=5, scoring="accuracy"
        )
        
        # Feature importance
        feature_importance = pd.DataFrame({
            "feature": self.feature_columns,
            "importance": self.stage_classifier.feature_importances_
        }).sort_values("importance", ascending=False)
        
        metrics = {
            "accuracy": accuracy,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(
                y_test, y_pred,
                target_names=self.label_encoder.classes_
            ),
            "feature_importance": feature_importance.head(10).to_dict("records"),
        }
        
        # Calculate AUC for multi-class
        if len(np.unique(y_encoded)) > 2:
            try:
                metrics["roc_auc"] = roc_auc_score(
                    y_test, y_prob, multi_class="ovr", average="weighted"
                )
            except:
                metrics["roc_auc"] = None
        
        return metrics
    
    def train_pd_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """Train the Probability of Default regression model.
        
        Args:
            X: Feature matrix
            y: Target values (PD rates)
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with training metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features (use existing scaler if available)
        if not self.is_fitted:
            X_train_scaled = self.scaler.fit_transform(X_train)
        else:
            X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize and train model
        self.pd_regressor = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        )
        
        # Train model
        self.pd_regressor.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.pd_regressor.predict(X_test_scaled)
        
        # Ensure predictions are in [0, 1] range
        y_pred = np.clip(y_pred, 0, 1)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.pd_regressor, X_train_scaled, y_train,
            cv=5, scoring="neg_mean_absolute_error"
        )
        
        # Feature importance
        feature_importance = pd.DataFrame({
            "feature": self.feature_columns,
            "importance": self.pd_regressor.feature_importances_
        }).sort_values("importance", ascending=False)
        
        metrics = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "cv_mae_mean": -cv_scores.mean(),
            "cv_mae_std": cv_scores.std(),
            "feature_importance": feature_importance.head(10).to_dict("records"),
            "r2_score": self.pd_regressor.score(X_test_scaled, y_test),
        }
        
        self.is_fitted = True
        
        return metrics
    
    def predict_stage(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict provision stage for new loans.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            Tuple of (predicted_stages, probabilities)
        """
        if self.stage_classifier is None:
            raise ValueError("Stage classifier not trained. Call train_stage_classifier first.")
        
        # Ensure features match training
        X_aligned = X[self.feature_columns]
        X_scaled = self.scaler.transform(X_aligned)
        
        # Predict
        y_pred_encoded = self.stage_classifier.predict(X_scaled)
        y_prob = self.stage_classifier.predict_proba(X_scaled)
        
        # Decode labels
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        return y_pred, y_prob
    
    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        """Predict Probability of Default for new loans.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            Array of PD predictions
        """
        if self.pd_regressor is None:
            raise ValueError("PD regressor not trained. Call train_pd_model first.")
        
        # Ensure features match training
        X_aligned = X[self.feature_columns]
        X_scaled = self.scaler.transform(X_aligned)
        
        # Predict and clip to [0, 1]
        y_pred = self.pd_regressor.predict(X_scaled)
        y_pred = np.clip(y_pred, 0, 1)
        
        return y_pred
    
    def save_models(self, path: str = "models/"):
        """Save trained models to disk.
        
        Args:
            path: Directory path to save models
        """
        import os
        os.makedirs(path, exist_ok=True)
        
        if self.stage_classifier:
            with open(f"{path}/stage_classifier.pkl", "wb") as f:
                pickle.dump(self.stage_classifier, f)
        
        if self.pd_regressor:
            with open(f"{path}/pd_regressor.pkl", "wb") as f:
                pickle.dump(self.pd_regressor, f)
        
        # Save preprocessors
        with open(f"{path}/scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        
        with open(f"{path}/label_encoder.pkl", "wb") as f:
            pickle.dump(self.label_encoder, f)
        
        # Save feature columns
        with open(f"{path}/feature_columns.pkl", "wb") as f:
            pickle.dump(self.feature_columns, f)
        
        print(f"Models saved to {path}")
    
    def load_models(self, path: str = "models/"):
        """Load trained models from disk.
        
        Args:
            path: Directory path containing saved models
        """
        try:
            with open(f"{path}/stage_classifier.pkl", "rb") as f:
                self.stage_classifier = pickle.load(f)
        except FileNotFoundError:
            print("Stage classifier not found")
        
        try:
            with open(f"{path}/pd_regressor.pkl", "rb") as f:
                self.pd_regressor = pickle.load(f)
        except FileNotFoundError:
            print("PD regressor not found")
        
        # Load preprocessors
        with open(f"{path}/scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        
        with open(f"{path}/label_encoder.pkl", "rb") as f:
            self.label_encoder = pickle.load(f)
        
        with open(f"{path}/feature_columns.pkl", "rb") as f:
            self.feature_columns = pickle.load(f)
        
        self.is_fitted = True
        print(f"Models loaded from {path}")
    
    def explain_prediction(self, X: pd.DataFrame, index: int = 0) -> Dict[str, Any]:
        """Explain model prediction for a specific loan.
        
        Args:
            X: Feature matrix
            index: Index of the loan to explain
            
        Returns:
            Dictionary with explanation details
        """
        if self.stage_classifier is None:
            raise ValueError("Models not trained")
        
        # Get single instance
        X_single = X.iloc[[index]]
        X_aligned = X_single[self.feature_columns]
        X_scaled = self.scaler.transform(X_aligned)
        
        # Get predictions
        stage_pred = self.label_encoder.inverse_transform(
            self.stage_classifier.predict(X_scaled)
        )[0]
        stage_prob = self.stage_classifier.predict_proba(X_scaled)[0]
        
        # Get feature contributions (simplified)
        feature_values = X_aligned.iloc[0].to_dict()
        
        # Sort features by importance
        feature_importance = pd.DataFrame({
            "feature": self.feature_columns,
            "importance": self.stage_classifier.feature_importances_,
            "value": X_aligned.iloc[0].values
        }).sort_values("importance", ascending=False)
        
        explanation = {
            "predicted_stage": stage_pred,
            "stage_probabilities": {
                self.label_encoder.classes_[i]: prob
                for i, prob in enumerate(stage_prob)
            },
            "top_features": feature_importance.head(5).to_dict("records"),
            "feature_values": feature_values,
        }
        
        if self.pd_regressor:
            pd_pred = self.pd_regressor.predict(X_scaled)[0]
            explanation["predicted_pd"] = float(np.clip(pd_pred, 0, 1))
        
        return explanation


class EnhancedCreditRiskClassifier:
    """Enhanced ML pipeline integrating advanced models with backward compatibility.
    
    This class provides:
    - Integration with OptimizedMLPipeline for advanced models (XGBoost, LightGBM, CatBoost)
    - Advanced feature engineering (50+ features)
    - Optuna hyperparameter optimization
    - SHAP explainability
    - Backward compatibility with existing CreditRiskClassifier
    - Automated model selection
    """
    
    def __init__(self, 
                 use_advanced_features: bool = True,
                 use_optimization: bool = True,
                 model_selection_strategy: str = "auto",
                 fallback_to_simple: bool = True):
        """Initialize the enhanced credit risk classifier.
        
        Args:
            use_advanced_features: Whether to use advanced feature engineering
            use_optimization: Whether to use hyperparameter optimization
            model_selection_strategy: "auto", "advanced_only", "simple_only"
            fallback_to_simple: Fallback to simple model if advanced fails
        """
        self.use_advanced_features = use_advanced_features
        self.use_optimization = use_optimization
        self.model_selection_strategy = model_selection_strategy
        self.fallback_to_simple = fallback_to_simple
        
        # Initialize both pipelines
        self.advanced_pipeline = None
        self.simple_classifier = None
        
        # Model selection results
        self.selected_model_type = None
        self.training_metrics = {}
        self.is_fitted = False
        
        # Compatibility layer
        self.feature_columns = None
        
    def _initialize_pipelines(self):
        """Initialize the ML pipelines."""
        try:
            from src.enhanced_ml_models import OptimizedMLPipeline
            self.advanced_pipeline = OptimizedMLPipeline(
                optimization_method="optuna",
                cv_folds=5,
                random_state=42
            )
        except ImportError as e:
            logger.warning(f"Cannot import OptimizedMLPipeline: {e}")
            if not self.fallback_to_simple:
                raise
        
        # Always initialize simple classifier for fallback
        self.simple_classifier = CreditRiskClassifier(model_type="random_forest")
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare features with backward compatibility.
        
        Args:
            df: Raw DataFrame with loan data
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        if self.model_selection_strategy == "simple_only" or not self.use_advanced_features:
            # Use simple feature preparation
            return self.simple_classifier.prepare_features(df)
        
        # For advanced features, we'll prepare a compatibility layer
        # First get simple features for fallback
        X_simple, simple_feature_names = self.simple_classifier.prepare_features(df)
        
        # Store for compatibility
        self.feature_columns = simple_feature_names
        
        if self.model_selection_strategy == "advanced_only" or self.use_advanced_features:
            return X_simple, simple_feature_names
        
        return X_simple, simple_feature_names
    
    def train_models(self,
                    X: pd.DataFrame,
                    y_stage: pd.Series,
                    y_pd: pd.Series,
                    test_size: float = 0.2) -> Dict[str, Any]:
        """Train both simple and advanced models, then select the best.
        
        Args:
            X: Feature matrix
            y_stage: Target labels (provision stages)
            y_pd: Target values (PD rates)
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with training results and metrics
        """
        self._initialize_pipelines()
        results = {'simple': {}, 'advanced': {}, 'selected': {}}
        
        # Always train simple model for compatibility
        try:
            logger.info("Training simple models...")
            
            # Train stage classifier
            stage_metrics = self.simple_classifier.train_stage_classifier(X, y_stage, test_size)
            
            # Train PD model  
            pd_metrics = self.simple_classifier.train_pd_model(X, y_pd, test_size)
            
            results['simple'] = {
                'stage_metrics': stage_metrics,
                'pd_metrics': pd_metrics,
                'model_type': 'simple',
                'success': True
            }
            logger.info(f"Simple models trained. Stage accuracy: {stage_metrics['accuracy']:.4f}")
            
        except Exception as e:
            logger.error(f"Simple model training failed: {e}")
            results['simple'] = {'success': False, 'error': str(e)}
        
        # Train advanced models if enabled
        if (self.model_selection_strategy in ["auto", "advanced_only"] and 
            self.advanced_pipeline is not None):
            
            try:
                logger.info("Training advanced models...")
                
                # Prepare data for advanced pipeline
                # For stage classification
                X_train, X_test, y_train, y_test = self.advanced_pipeline.prepare_data(
                    df=pd.concat([X, y_stage], axis=1).rename(columns={y_stage.name: 'provision_stage'}),
                    target_column='provision_stage',
                    test_size=test_size
                )
                
                # Train all advanced models
                advanced_results = self.advanced_pipeline.train_all_models(
                    X_train, y_train, X_test, y_test,
                    optimize_hyperparams=self.use_optimization
                )
                
                # Calculate overall performance metrics
                best_model_auc = 0
                if self.advanced_pipeline.best_model_name:
                    best_result = advanced_results.get(self.advanced_pipeline.best_model_name, {})
                    best_model_auc = best_result.get('evaluation', {}).get('test_auc', 0)
                
                results['advanced'] = {
                    'model_results': advanced_results,
                    'best_model': self.advanced_pipeline.best_model_name,
                    'best_auc': best_model_auc,
                    'model_type': 'advanced',
                    'success': True
                }
                
                logger.info(f"Advanced models trained. Best model: {self.advanced_pipeline.best_model_name}, AUC: {best_model_auc:.4f}")
                
            except Exception as e:
                logger.error(f"Advanced model training failed: {e}")
                results['advanced'] = {'success': False, 'error': str(e)}
        
        # Select best model
        selected_result = self._select_best_pipeline(results)
        results['selected'] = selected_result
        
        self.training_metrics = results
        self.is_fitted = True
        
        return results
    
    def _select_best_pipeline(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Select the best performing pipeline."""
        
        if self.model_selection_strategy == "simple_only":
            if results['simple']['success']:
                self.selected_model_type = 'simple'
                return {
                    'selected_type': 'simple',
                    'reason': 'simple_only strategy',
                    'metrics': results['simple']
                }
        
        elif self.model_selection_strategy == "advanced_only":
            if results['advanced']['success']:
                self.selected_model_type = 'advanced'
                return {
                    'selected_type': 'advanced', 
                    'reason': 'advanced_only strategy',
                    'metrics': results['advanced']
                }
            elif self.fallback_to_simple and results['simple']['success']:
                self.selected_model_type = 'simple'
                return {
                    'selected_type': 'simple',
                    'reason': 'advanced failed, fallback to simple',
                    'metrics': results['simple']
                }
        
        else:  # auto selection
            # Compare performance if both succeeded
            if results['simple']['success'] and results['advanced']['success']:
                simple_acc = results['simple']['stage_metrics'].get('accuracy', 0)
                advanced_auc = results['advanced'].get('best_auc', 0)
                
                # Use AUC > 0.8 as threshold for advanced model selection
                # Otherwise prefer simple model for interpretability
                if advanced_auc > 0.8 and advanced_auc > simple_acc:
                    self.selected_model_type = 'advanced'
                    return {
                        'selected_type': 'advanced',
                        'reason': f'better performance (AUC: {advanced_auc:.4f} vs Acc: {simple_acc:.4f})',
                        'metrics': results['advanced']
                    }
                else:
                    self.selected_model_type = 'simple'
                    return {
                        'selected_type': 'simple', 
                        'reason': f'comparable performance, prefer interpretability',
                        'metrics': results['simple']
                    }
            
            elif results['advanced']['success']:
                self.selected_model_type = 'advanced'
                return {
                    'selected_type': 'advanced',
                    'reason': 'advanced succeeded, simple failed',
                    'metrics': results['advanced'] 
                }
            
            elif results['simple']['success']:
                self.selected_model_type = 'simple'
                return {
                    'selected_type': 'simple',
                    'reason': 'simple succeeded, advanced failed',
                    'metrics': results['simple']
                }
        
        # Both failed
        raise ValueError("Both simple and advanced model training failed")
    
    def predict_stage(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict provision stage using the selected model.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            Tuple of (predicted_stages, probabilities)
        """
        if not self.is_fitted:
            raise ValueError("Models not trained. Call train_models first.")
        
        if self.selected_model_type == 'simple':
            return self.simple_classifier.predict_stage(X)
        
        elif self.selected_model_type == 'advanced':
            # Convert advanced model predictions to stage format
            prediction_result = self.advanced_pipeline.predict_batch(X, return_probabilities=True)
            predictions = prediction_result['predictions']
            probabilities = prediction_result.get('probabilities', np.array([]))
            
            # Map numeric predictions back to stage labels
            stage_mapping = {0: 'Stage 1', 1: 'Stage 2', 2: 'Stage 3'}
            stage_predictions = np.array([stage_mapping.get(p, 'Stage 1') for p in predictions])
            
            return stage_predictions, probabilities
        
        else:
            raise ValueError(f"Unknown selected model type: {self.selected_model_type}")
    
    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        """Predict Probability of Default using the selected model.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            Array of PD predictions
        """
        if not self.is_fitted:
            raise ValueError("Models not trained. Call train_models first.")
        
        if self.selected_model_type == 'simple':
            return self.simple_classifier.predict_pd(X)
        
        elif self.selected_model_type == 'advanced':
            # For now, use simple model for PD prediction even with advanced stage model
            # This maintains consistency until we implement PD regression in advanced pipeline
            return self.simple_classifier.predict_pd(X)
        
        else:
            raise ValueError(f"Unknown selected model type: {self.selected_model_type}")
    
    def explain_prediction(self, X: pd.DataFrame, index: int = 0) -> Dict[str, Any]:
        """Explain model prediction with SHAP if available.
        
        Args:
            X: Feature matrix
            index: Index of the loan to explain
            
        Returns:
            Dictionary with explanation details
        """
        if not self.is_fitted:
            raise ValueError("Models not trained. Call explain_prediction first.")
        
        explanation = {}
        
        if self.selected_model_type == 'simple':
            explanation = self.simple_classifier.explain_prediction(X, index)
            explanation['model_type'] = 'simple'
        
        elif self.selected_model_type == 'advanced':
            try:
                # Try SHAP explanation first
                shap_results = self.advanced_pipeline.generate_shap_explanations(
                    X.iloc[[index]], max_samples=1
                )
                
                if 'error' not in shap_results:
                    explanation = {
                        'model_type': 'advanced',
                        'shap_explanation': shap_results,
                        'feature_importance': shap_results.get('feature_importance_shap', {}),
                        'model_used': shap_results.get('model_name')
                    }
                else:
                    # Fallback to simple explanation
                    explanation = self.simple_classifier.explain_prediction(X, index)
                    explanation['model_type'] = 'simple_fallback'
                    explanation['shap_error'] = shap_results['error']
            
            except Exception as e:
                # Fallback to simple explanation
                explanation = self.simple_classifier.explain_prediction(X, index)
                explanation['model_type'] = 'simple_fallback'
                explanation['shap_error'] = str(e)
        
        return explanation
    
    def save_models(self, path: str = "/opt/airflow/models/"):
        """Save all trained models to disk.
        
        Args:
            path: Directory path to save models
        """
        import os
        from datetime import datetime
        
        os.makedirs(path, exist_ok=True)
        
        # Save model selection metadata
        metadata = {
            'selected_model_type': self.selected_model_type,
            'use_advanced_features': self.use_advanced_features,
            'use_optimization': self.use_optimization,
            'model_selection_strategy': self.model_selection_strategy,
            'training_timestamp': datetime.now().isoformat(),
            'training_metrics': self.training_metrics,
            'is_fitted': self.is_fitted
        }
        
        import json
        with open(f"{path}/enhanced_model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Save simple classifier (always available)
        if self.simple_classifier and self.simple_classifier.is_fitted:
            self.simple_classifier.save_models(f"{path}/simple/")
        
        # Save advanced pipeline if available
        if self.advanced_pipeline and self.selected_model_type == 'advanced':
            try:
                self.advanced_pipeline.save_models(f"{path}/advanced/")
            except Exception as e:
                logger.error(f"Error saving advanced models: {e}")
        
        logger.info(f"Enhanced models saved to {path}")
    
    def load_models(self, path: str = "/opt/airflow/models/"):
        """Load trained models from disk.
        
        Args:
            path: Directory path containing saved models
        """
        import os
        import json
        
        # Load metadata
        metadata_file = f"{path}/enhanced_model_metadata.json"
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            
            self.selected_model_type = metadata.get('selected_model_type')
            self.use_advanced_features = metadata.get('use_advanced_features', True)
            self.use_optimization = metadata.get('use_optimization', True)
            self.model_selection_strategy = metadata.get('model_selection_strategy', 'auto')
            self.training_metrics = metadata.get('training_metrics', {})
            self.is_fitted = metadata.get('is_fitted', False)
        
        self._initialize_pipelines()
        
        # Load simple classifier
        simple_path = f"{path}/simple/"
        if os.path.exists(simple_path):
            try:
                self.simple_classifier.load_models(simple_path)
            except Exception as e:
                logger.error(f"Error loading simple models: {e}")
        
        # Load advanced pipeline if needed
        advanced_path = f"{path}/advanced/"
        if os.path.exists(advanced_path) and self.advanced_pipeline:
            try:
                self.advanced_pipeline.load_models(advanced_path)
            except Exception as e:
                logger.error(f"Error loading advanced models: {e}")
        
        logger.info(f"Enhanced models loaded from {path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model setup."""
        info = {
            'selected_model_type': self.selected_model_type,
            'use_advanced_features': self.use_advanced_features,
            'use_optimization': self.use_optimization,
            'model_selection_strategy': self.model_selection_strategy,
            'is_fitted': self.is_fitted
        }
        
        if self.is_fitted and self.training_metrics:
            if self.selected_model_type == 'simple':
                simple_metrics = self.training_metrics.get('simple', {})
                info['stage_accuracy'] = simple_metrics.get('stage_metrics', {}).get('accuracy', 0)
                info['pd_mae'] = simple_metrics.get('pd_metrics', {}).get('mae', 0)
            
            elif self.selected_model_type == 'advanced':
                advanced_metrics = self.training_metrics.get('advanced', {})
                info['best_model'] = advanced_metrics.get('best_model')
                info['best_auc'] = advanced_metrics.get('best_auc', 0)
        
        return info


# Create convenience function for backward compatibility
def create_ifrs9_ml_classifier(use_enhanced: bool = True, **kwargs) -> Union[CreditRiskClassifier, EnhancedCreditRiskClassifier]:
    """Factory function to create the appropriate ML classifier.
    
    Args:
        use_enhanced: Whether to use the enhanced classifier
        **kwargs: Additional arguments for the classifier
        
    Returns:
        Appropriate classifier instance
    """
    if use_enhanced:
        return EnhancedCreditRiskClassifier(**kwargs)
    else:
        return CreditRiskClassifier(**kwargs)

def get_ml_predictions_for_rules_engine(loans_df: pd.DataFrame, 
                                      model_path: str = "/opt/airflow/models/",
                                      fallback_to_simple: bool = True) -> Dict[str, Any]:
    """Get ML predictions for the rules engine integration.
    
    This function serves as the main integration point between the ML models
    and the IFRS9 rules engine. It loads the appropriate trained models and
    provides predictions in a standardized format.
    
    Args:
        loans_df: DataFrame with loan data for prediction
        model_path: Path to the saved models
        fallback_to_simple: Whether to fallback to simple models if enhanced models fail
        
    Returns:
        Dictionary containing:
        - stage_predictions: Array of predicted stages
        - stage_probabilities: Probability matrix for stage predictions  
        - pd_predictions: Array of PD predictions
        - model_info: Information about the model used
        - prediction_metadata: Metadata about the predictions
    """
    import os
    import json
    from datetime import datetime
    
    logger.info(f"Getting ML predictions for {len(loans_df)} loans")
    
    # Initialize result structure
    result = {
        'stage_predictions': None,
        'stage_probabilities': None,
        'pd_predictions': None,
        'model_info': {},
        'prediction_metadata': {
            'prediction_timestamp': datetime.now().isoformat(),
            'num_loans': len(loans_df),
            'model_path': model_path
        }
    }
    
    try:
        # Check if enhanced models exist
        enhanced_metadata_path = f"{model_path}/enhanced_model_metadata.json"
        use_enhanced = os.path.exists(enhanced_metadata_path)
        
        if use_enhanced:
            logger.info("Loading enhanced ML classifier")
            try:
                classifier = EnhancedCreditRiskClassifier()
                classifier.load_models(model_path)
                
                if not classifier.is_fitted:
                    raise ValueError("Enhanced classifier not properly loaded")
                
                # Prepare features
                X, feature_names = classifier.prepare_features(loans_df)
                
                # Get predictions
                stage_pred, stage_prob = classifier.predict_stage(X)
                pd_pred = classifier.predict_pd(X)
                
                # Store results
                result['stage_predictions'] = stage_pred
                result['stage_probabilities'] = stage_prob
                result['pd_predictions'] = pd_pred
                result['model_info'] = classifier.get_model_info()
                result['prediction_metadata']['model_type'] = 'enhanced'
                result['prediction_metadata']['feature_count'] = len(feature_names)
                
                logger.info(f"Enhanced predictions completed using {result['model_info']['selected_model_type']} model")
                
            except Exception as e:
                logger.error(f"Enhanced model prediction failed: {e}")
                if not fallback_to_simple:
                    raise
                use_enhanced = False
        
        if not use_enhanced:
            logger.info("Loading simple ML classifier")
            try:
                classifier = CreditRiskClassifier()
                classifier.load_models(model_path + "/simple/" if os.path.exists(model_path + "/simple/") else model_path)
                
                if not classifier.is_fitted:
                    raise ValueError("Simple classifier not properly loaded")
                
                # Prepare features
                X, feature_names = classifier.prepare_features(loans_df)
                
                # Get predictions
                stage_pred, stage_prob = classifier.predict_stage(X)
                pd_pred = classifier.predict_pd(X)
                
                # Store results
                result['stage_predictions'] = stage_pred
                result['stage_probabilities'] = stage_prob
                result['pd_predictions'] = pd_pred
                result['model_info'] = {
                    'model_type': 'simple',
                    'is_fitted': classifier.is_fitted,
                    'feature_count': len(feature_names)
                }
                result['prediction_metadata']['model_type'] = 'simple'
                result['prediction_metadata']['feature_count'] = len(feature_names)
                
                logger.info("Simple predictions completed")
                
            except Exception as e:
                logger.error(f"Simple model prediction failed: {e}")
                raise
    
    except Exception as e:
        logger.error(f"ML prediction failed: {str(e)}", exc_info=True)
        result['error'] = str(e)
        result['prediction_metadata']['status'] = 'failed'
        raise
    
    # Add prediction statistics
    if result['stage_predictions'] is not None:
        stage_counts = pd.Series(result['stage_predictions']).value_counts().to_dict()
        result['prediction_metadata']['stage_distribution'] = stage_counts
        result['prediction_metadata']['avg_pd'] = float(np.mean(result['pd_predictions']))
        result['prediction_metadata']['status'] = 'success'
    
    return result


def explain_ml_prediction_for_rules_engine(loans_df: pd.DataFrame, 
                                         loan_index: int,
                                         model_path: str = "/opt/airflow/models/") -> Dict[str, Any]:
    """Get detailed explanation for a specific loan prediction.
    
    This function provides model explanations for individual loan predictions,
    useful for regulatory compliance and business understanding.
    
    Args:
        loans_df: DataFrame with loan data
        loan_index: Index of the specific loan to explain
        model_path: Path to the saved models
        
    Returns:
        Dictionary containing detailed explanation of the prediction
    """
    logger.info(f"Explaining prediction for loan at index {loan_index}")
    
    try:
        # Check if enhanced models exist
        enhanced_metadata_path = f"{model_path}/enhanced_model_metadata.json"
        use_enhanced = os.path.exists(enhanced_metadata_path)
        
        if use_enhanced:
            classifier = EnhancedCreditRiskClassifier()
            classifier.load_models(model_path)
        else:
            classifier = CreditRiskClassifier()
            classifier.load_models(model_path + "/simple/" if os.path.exists(model_path + "/simple/") else model_path)
        
        # Prepare features
        X, feature_names = classifier.prepare_features(loans_df)
        
        # Get explanation
        explanation = classifier.explain_prediction(X, loan_index)
        
        # Add loan context
        loan_data = loans_df.iloc[loan_index].to_dict()
        explanation['loan_data'] = loan_data
        explanation['explanation_timestamp'] = datetime.now().isoformat()
        
        return explanation
    
    except Exception as e:
        logger.error(f"Prediction explanation failed: {str(e)}", exc_info=True)
        return {
            'error': str(e),
            'loan_index': loan_index,
            'explanation_timestamp': datetime.now().isoformat()
        }


def validate_ml_model_health(model_path: str = "/opt/airflow/models/") -> Dict[str, Any]:
    """Validate that ML models are properly loaded and functioning.
    
    This function performs health checks on the ML models to ensure
    they are ready for production use.
    
    Args:
        model_path: Path to the saved models
        
    Returns:
        Dictionary containing health check results
    """
    logger.info("Validating ML model health")
    
    health_report = {
        'timestamp': datetime.now().isoformat(),
        'model_path': model_path,
        'enhanced_available': False,
        'simple_available': False,
        'errors': [],
        'warnings': [],
        'status': 'unknown'
    }
    
    try:
        # Check enhanced models
        enhanced_metadata_path = f"{model_path}/enhanced_model_metadata.json"
        if os.path.exists(enhanced_metadata_path):
            try:
                classifier = EnhancedCreditRiskClassifier()
                classifier.load_models(model_path)
                
                if classifier.is_fitted:
                    health_report['enhanced_available'] = True
                    health_report['enhanced_info'] = classifier.get_model_info()
                else:
                    health_report['warnings'].append("Enhanced classifier loaded but not fitted")
                    
            except Exception as e:
                health_report['errors'].append(f"Enhanced model error: {str(e)}")
        
        # Check simple models
        simple_path = f"{model_path}/simple/" if os.path.exists(f"{model_path}/simple/") else model_path
        try:
            classifier = CreditRiskClassifier()
            classifier.load_models(simple_path)
            
            if classifier.is_fitted:
                health_report['simple_available'] = True
                health_report['simple_info'] = {
                    'model_type': 'simple',
                    'is_fitted': classifier.is_fitted,
                    'feature_columns_count': len(classifier.feature_columns) if classifier.feature_columns else 0
                }
            else:
                health_report['warnings'].append("Simple classifier loaded but not fitted")
                
        except Exception as e:
            health_report['errors'].append(f"Simple model error: {str(e)}")
        
        # Determine overall status
        if health_report['enhanced_available'] or health_report['simple_available']:
            if len(health_report['errors']) == 0:
                health_report['status'] = 'healthy'
            else:
                health_report['status'] = 'degraded'
        else:
            health_report['status'] = 'failed'
            
        logger.info(f"Model health check completed: {health_report['status']}")
        
    except Exception as e:
        health_report['errors'].append(f"Health check failed: {str(e)}")
        health_report['status'] = 'failed'
        logger.error(f"Model health check failed: {str(e)}", exc_info=True)
    
    return health_report


# Maintain backward compatibility with the original classifier
classifier = CreditRiskClassifier()


if __name__ == "__main__":
    # Example usage
    classifier = CreditRiskClassifier(model_type="random_forest")
    
    # Train on sample data
    # loans_df = pd.read_csv("data/raw/loan_portfolio.csv")
    # X, feature_names = classifier.prepare_features(loans_df)
    # metrics = classifier.train_stage_classifier(X, loans_df["provision_stage"])
    # print(metrics["classification_report"])
    # classifier.save_models()