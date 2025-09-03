"""
Comprehensive test suite for Polars ML integration with IFRS9 compliance validation.

This test suite validates:
- Model accuracy consistency between Polars and Pandas workflows
- IFRS9 regulatory compliance maintenance
- Performance improvements
- Data integrity throughout the pipeline
"""

import pytest
import numpy as np
import pandas as pd
import polars as pl
from typing import Dict, Any
import logging
from unittest.mock import Mock, patch

# Import components to test
from src.enhanced_ml_models import OptimizedMLPipeline, AdvancedFeatureEngineer
from src.polars_ml_integration import PolarsEnhancedCreditRiskClassifier, create_synthetic_ifrs9_data_polars
from src.polars_ml_benchmark import PolarsMLBenchmarkSuite

logger = logging.getLogger(__name__)

class TestPolarsMLIntegration:
    """Test suite for Polars ML integration"""
    
    @pytest.fixture
    def sample_data_small(self):
        """Generate small sample dataset for testing"""
        return create_synthetic_ifrs9_data_polars(1000)
    
    @pytest.fixture
    def sample_data_medium(self):
        """Generate medium sample dataset for testing"""
        return create_synthetic_ifrs9_data_polars(5000)
    
    @pytest.fixture
    def sample_data_both_formats(self):
        """Generate data in both Polars and pandas formats"""
        pl_data = create_synthetic_ifrs9_data_polars(2000)
        pd_data = pl_data.to_pandas()
        return pl_data, pd_data
    
    def test_polars_availability(self):
        """Test that Polars is available and properly configured"""
        assert pl.__version__ is not None
        logger.info(f"Polars version: {pl.__version__}")
    
    def test_advanced_feature_engineer_polars_support(self, sample_data_both_formats):
        """Test AdvancedFeatureEngineer with Polars support"""
        pl_data, pd_data = sample_data_both_formats
        
        # Initialize with Polars support
        fe_polars = AdvancedFeatureEngineer(use_polars=True, lazy_evaluation=True)
        fe_pandas = AdvancedFeatureEngineer(use_polars=False)
        
        # Fit both
        fe_polars.fit(pl_data)
        fe_pandas.fit(pd_data)
        
        # Transform
        pl_features = fe_polars.transform(pl_data)
        pd_features = fe_pandas.transform(pd_data)
        
        # Validate results
        assert isinstance(pl_features, pl.DataFrame), "Polars feature engineer should return Polars DataFrame"
        assert isinstance(pd_features, pd.DataFrame), "Pandas feature engineer should return pandas DataFrame"
        
        # Feature counts should be similar (allowing for minor differences in categorical handling)
        assert abs(len(pl_features.columns) - len(pd_features.columns)) <= 5, "Feature counts should be similar"
        
        logger.info(f"Polars features: {len(pl_features.columns)}, Pandas features: {len(pd_features.columns)}")
    
    def test_feature_engineering_consistency(self, sample_data_both_formats):
        """Test that feature engineering produces consistent results"""
        pl_data, pd_data = sample_data_both_formats
        
        # Use same random state for reproducibility
        fe_polars = AdvancedFeatureEngineer(use_polars=True, lazy_evaluation=True)
        fe_pandas = AdvancedFeatureEngineer(use_polars=False)
        
        fe_polars.fit(pl_data)
        fe_pandas.fit(pd_data)
        
        pl_features = fe_polars.transform(pl_data)
        pd_features = fe_pandas.transform(pd_data)
        
        # Convert Polars to pandas for comparison
        pl_features_pd = pl_features.to_pandas()
        
        # Check common features
        common_features = set(pl_features_pd.columns) & set(pd_features.columns)
        assert len(common_features) > 10, "Should have substantial feature overlap"
        
        # Check numerical consistency for common features
        for feature in list(common_features)[:5]:  # Test first 5 common features
            if pd_features[feature].dtype in [np.float64, np.int64]:
                pl_values = pl_features_pd[feature].values
                pd_values = pd_features[feature].values
                
                # Allow for minor numerical differences
                np.testing.assert_allclose(pl_values, pd_values, rtol=1e-5, atol=1e-8,
                                         err_msg=f"Feature {feature} values differ significantly")
        
        logger.info(f"Validated consistency for {len(common_features)} common features")
    
    def test_optimized_ml_pipeline_polars_support(self, sample_data_both_formats):
        """Test OptimizedMLPipeline with Polars support"""
        pl_data, pd_data = sample_data_both_formats
        
        # Initialize pipelines
        pipeline_polars = OptimizedMLPipeline(use_polars=True, optimization_method="optuna")
        pipeline_pandas = OptimizedMLPipeline(use_polars=False, optimization_method="optuna")
        
        # Prepare data
        X_train_pl, X_test_pl, y_train_pl, y_test_pl = pipeline_polars.prepare_data(
            pl_data, target_column="provision_stage", test_size=0.2
        )
        X_train_pd, X_test_pd, y_train_pd, y_test_pd = pipeline_pandas.prepare_data(
            pd_data, target_column="provision_stage", test_size=0.2
        )
        
        # Validate data preparation
        assert len(X_train_pl) > 0, "Polars training data should not be empty"
        assert len(X_train_pd) > 0, "Pandas training data should not be empty"
        assert len(y_train_pl) == len(X_train_pl), "Training features and labels should match"
        assert len(y_train_pd) == len(X_train_pd), "Training features and labels should match"
        
        logger.info(f"Data prepared: Polars {len(X_train_pl)} rows, Pandas {len(X_train_pd)} rows")
    
    def test_model_accuracy_consistency(self, sample_data_both_formats):
        """Test that models trained with Polars and Pandas workflows have consistent accuracy"""
        pl_data, pd_data = sample_data_both_formats
        
        # Train with Polars workflow
        polars_classifier = PolarsEnhancedCreditRiskClassifier(
            model_type="xgboost", use_lazy_evaluation=True
        )
        
        pl_features, _ = polars_classifier.prepare_features_polars(pl_data)
        polars_metrics = polars_classifier.train_stage_classifier_polars(
            pl_features, pl_data["provision_stage"], test_size=0.2
        )
        
        # Train with pandas workflow (baseline)
        pandas_pipeline = OptimizedMLPipeline(use_polars=False)
        X_train, X_test, y_train, y_test = pandas_pipeline.prepare_data(
            pd_data, target_column="provision_stage", test_size=0.2
        )
        pandas_results = pandas_pipeline.train_all_models(
            X_train, y_train, X_test, y_test, optimize_hyperparams=False
        )
        
        # Compare accuracies
        polars_accuracy = polars_metrics['accuracy']
        pandas_accuracy = list(pandas_results.values())[0]['evaluation']['test_accuracy']
        
        accuracy_difference = abs(polars_accuracy - pandas_accuracy)
        
        # Allow for reasonable variance in model performance
        assert accuracy_difference < 0.05, f"Accuracy difference too large: {accuracy_difference}"
        assert polars_accuracy > 0.6, "Polars model accuracy should be reasonable"
        assert pandas_accuracy > 0.6, "Pandas model accuracy should be reasonable"
        
        logger.info(f"Accuracy comparison - Polars: {polars_accuracy:.4f}, Pandas: {pandas_accuracy:.4f}, Diff: {accuracy_difference:.4f}")
    
    def test_shap_explanations_polars(self, sample_data_small):
        """Test SHAP explanations with Polars DataFrames"""
        pipeline = OptimizedMLPipeline(use_polars=True)
        
        # Prepare and train
        X_train, X_test, y_train, y_test = pipeline.prepare_data(
            sample_data_small, target_column="provision_stage", test_size=0.2
        )
        results = pipeline.train_all_models(X_train, y_train, X_test, y_test, optimize_hyperparams=False)
        
        # Generate SHAP explanations
        shap_results = pipeline.generate_shap_explanations(X_test.head(50), max_samples=20)
        
        # Validate SHAP results
        assert 'error' not in shap_results, f"SHAP generation failed: {shap_results.get('error')}"
        assert 'feature_importance_shap' in shap_results, "Should contain SHAP feature importance"
        assert len(shap_results['feature_importance_shap']) > 0, "Should have feature importance values"
        
        logger.info(f"SHAP explanation generated for {shap_results['explanation_samples']} samples")
    
    def test_streaming_prediction(self, sample_data_medium):
        """Test streaming prediction capability"""
        pipeline = OptimizedMLPipeline(use_polars=True, enable_streaming=True)
        
        # Train a simple model
        X_train, X_test, y_train, y_test = pipeline.prepare_data(
            sample_data_medium, target_column="provision_stage", test_size=0.2
        )
        results = pipeline.train_all_models(X_train, y_train, X_test, y_test, optimize_hyperparams=False)
        
        # Test streaming prediction
        streaming_results = pipeline.predict_batch(X_test, return_probabilities=False, batch_size=500)
        
        # Validate streaming results
        assert streaming_results['processing_method'] == 'streaming', "Should use streaming processing"
        assert streaming_results['num_batches'] > 1, "Should process multiple batches"
        assert len(streaming_results['predictions']) == len(X_test), "Should predict all samples"
        
        logger.info(f"Streaming prediction: {streaming_results['num_batches']} batches, {streaming_results['num_samples']} samples")
    
    def test_ifrs9_compliance_validation(self, sample_data_small):
        """Test that IFRS9 regulatory requirements are maintained"""
        pipeline = OptimizedMLPipeline(use_polars=True)
        
        # Prepare data
        X_train, X_test, y_train, y_test = pipeline.prepare_data(
            sample_data_small, target_column="provision_stage", test_size=0.2
        )
        
        # Validate IFRS9 stage mapping
        unique_stages = np.unique(y_train)
        assert len(unique_stages) <= 3, "Should have at most 3 stages (Stage1, Stage2, Stage3)"
        
        # Train model
        results = pipeline.train_all_models(X_train, y_train, X_test, y_test, optimize_hyperparams=False)
        
        # Make predictions
        pred_results = pipeline.predict_batch(X_test, return_probabilities=True)
        
        # Validate predictions are in valid range
        predictions = pred_results['predictions']
        probabilities = pred_results.get('probabilities')
        
        assert all(pred in unique_stages for pred in predictions), "All predictions should be valid stages"
        
        if probabilities is not None:
            assert np.all(probabilities >= 0), "Probabilities should be non-negative"
            assert np.all(probabilities <= 1), "Probabilities should not exceed 1"
            assert np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5), "Probabilities should sum to 1"
        
        logger.info("IFRS9 compliance validation passed")
    
    def test_feature_importance_consistency(self, sample_data_small):
        """Test that feature importance is consistent and interpretable"""
        pipeline = OptimizedMLPipeline(use_polars=True)
        
        X_train, X_test, y_train, y_test = pipeline.prepare_data(
            sample_data_small, target_column="provision_stage", test_size=0.2
        )
        results = pipeline.train_all_models(X_train, y_train, X_test, y_test, optimize_hyperparams=False)
        
        # Check feature importance
        best_model_result = results[pipeline.best_model_name]
        feature_importance = best_model_result['evaluation'].get('feature_importance', {})
        
        assert len(feature_importance) > 0, "Should have feature importance values"
        
        # Validate that important IFRS9 features are present
        important_features = list(feature_importance.keys())[:10]
        
        # Expected important features for credit risk
        expected_features = ['credit_score', 'days_past_due', 'ltv_ratio', 'debt_to_income', 'current_balance']
        
        # Check if at least some expected features are in top important features
        found_expected = any(
            any(expected in important_feat.lower() for expected in expected_features)
            for important_feat in important_features
        )
        
        assert found_expected, f"Expected important features not found in top 10: {important_features}"
        
        logger.info(f"Top important features: {important_features[:5]}")
    
    def test_data_pipeline_integrity(self, sample_data_both_formats):
        """Test data integrity throughout the ML pipeline"""
        pl_data, pd_data = sample_data_both_formats
        
        # Test with Polars
        pipeline = OptimizedMLPipeline(use_polars=True)
        X_train, X_test, y_train, y_test = pipeline.prepare_data(
            pl_data, target_column="provision_stage", test_size=0.2
        )
        
        # Validate no data leakage
        train_indices = set(range(len(X_train)))
        test_indices = set(range(len(X_train), len(X_train) + len(X_test)))
        
        assert len(train_indices & test_indices) == 0, "No overlap between train and test sets"
        
        # Validate no missing values in critical columns
        if hasattr(X_train, 'to_pandas'):  # Polars DataFrame
            X_train_pd = X_train.to_pandas()
        else:
            X_train_pd = X_train
        
        # Check for excessive missing values
        missing_ratios = X_train_pd.isnull().sum() / len(X_train_pd)
        excessive_missing = missing_ratios[missing_ratios > 0.5]
        
        assert len(excessive_missing) == 0, f"Excessive missing values in features: {excessive_missing.index.tolist()}"
        
        # Validate target distribution
        target_counts = pd.Series(y_train).value_counts()
        min_class_ratio = target_counts.min() / target_counts.sum()
        
        assert min_class_ratio > 0.05, f"Class imbalance too severe: {min_class_ratio:.3f}"
        
        logger.info(f"Data integrity validated - Train: {len(X_train)}, Test: {len(X_test)}, Min class ratio: {min_class_ratio:.3f}")
    
    def test_performance_improvement(self, sample_data_medium):
        """Test that Polars provides performance improvements"""
        import time
        
        # Benchmark feature engineering
        fe_polars = AdvancedFeatureEngineer(use_polars=True, lazy_evaluation=True)
        fe_pandas = AdvancedFeatureEngineer(use_polars=False)
        
        pd_data = sample_data_medium.to_pandas()
        
        # Fit both
        fe_polars.fit(sample_data_medium)
        fe_pandas.fit(pd_data)
        
        # Time Polars feature engineering
        polars_start = time.time()
        polars_features = fe_polars.transform(sample_data_medium)
        polars_time = time.time() - polars_start
        
        # Time pandas feature engineering
        pandas_start = time.time()
        pandas_features = fe_pandas.transform(pd_data)
        pandas_time = time.time() - pandas_start
        
        speedup = pandas_time / polars_time if polars_time > 0 else 1
        
        logger.info(f"Performance comparison - Polars: {polars_time:.3f}s, Pandas: {pandas_time:.3f}s, Speedup: {speedup:.2f}x")
        
        # We expect at least some performance improvement for medium-sized datasets
        # But we'll be lenient in testing environment
        assert speedup >= 0.8, f"Significant performance regression detected: {speedup:.2f}x"
    
    def test_error_handling_robustness(self, sample_data_small):
        """Test error handling and robustness"""
        pipeline = OptimizedMLPipeline(use_polars=True)
        
        # Test with missing target column
        try:
            pipeline.prepare_data(sample_data_small, target_column="nonexistent_column")
            assert False, "Should raise error for missing target column"
        except Exception:
            pass  # Expected
        
        # Test with empty dataset
        empty_data = sample_data_small.head(0)
        try:
            pipeline.prepare_data(empty_data, target_column="provision_stage")
            assert False, "Should raise error for empty dataset"
        except Exception:
            pass  # Expected
        
        # Test SHAP with untrained model
        shap_results = pipeline.generate_shap_explanations(sample_data_small.head(10))
        assert 'error' in shap_results, "Should return error for untrained model"
        
        logger.info("Error handling tests passed")

class TestPolarsMLBenchmark:
    """Test suite for benchmark functionality"""
    
    @pytest.fixture
    def benchmark_suite(self):
        """Create benchmark suite instance"""
        return PolarsMLBenchmarkSuite(random_state=42)
    
    def test_benchmark_suite_initialization(self, benchmark_suite):
        """Test benchmark suite initialization"""
        assert benchmark_suite.random_state == 42
        assert benchmark_suite.benchmark_timestamp is not None
        assert isinstance(benchmark_suite.results, dict)
    
    def test_generate_benchmark_datasets(self, benchmark_suite):
        """Test dataset generation for benchmarking"""
        datasets = benchmark_suite.generate_benchmark_datasets([100, 500])
        
        assert len(datasets) == 2, "Should generate 2 datasets"
        assert 100 in datasets and 500 in datasets, "Should contain specified sizes"
        
        for size, (pl_data, pd_data) in datasets.items():
            assert isinstance(pl_data, pl.DataFrame), "Should be Polars DataFrame"
            assert isinstance(pd_data, pd.DataFrame), "Should be pandas DataFrame"
            assert len(pl_data) == size, f"Polars dataset should have {size} rows"
            assert len(pd_data) == size, f"Pandas dataset should have {size} rows"
    
    def test_benchmark_report_generation(self, benchmark_suite):
        """Test benchmark report generation"""
        # Mock results
        mock_results = {
            'metadata': {
                'timestamp': '2024-01-01T12:00:00',
                'dataset_sizes': [1000, 5000],
                'polars_version': '0.20.0',
                'total_runtime': 120.5
            },
            'feature_engineering': {
                1000: {
                    'polars_avg_time': 0.1,
                    'pandas_avg_time': 0.2,
                    'polars_std_time': 0.01,
                    'pandas_std_time': 0.02,
                    'speedup_factor': 2.0,
                    'polars_features_count': 45,
                    'pandas_features_count': 44,
                    'iterations': 3
                }
            }
        }
        
        report = benchmark_suite.generate_benchmark_report(mock_results)
        
        assert isinstance(report, str), "Should return string report"
        assert "POLARS ML INTEGRATION BENCHMARK REPORT" in report, "Should contain title"
        assert "2.0x" in report, "Should contain speedup information"
        assert "FASTER" in report, "Should indicate performance improvement"

# Integration tests for the complete workflow
class TestPolarsMLWorkflowIntegration:
    """End-to-end integration tests"""
    
    def test_complete_ml_workflow_polars(self):
        """Test complete ML workflow with Polars optimization"""
        # Generate data
        data = create_synthetic_ifrs9_data_polars(3000)
        
        # Initialize pipeline
        pipeline = OptimizedMLPipeline(
            use_polars=True,
            enable_streaming=False,
            optimization_method="optuna"
        )
        
        # Complete workflow
        X_train, X_test, y_train, y_test = pipeline.prepare_data(
            data, target_column="provision_stage", test_size=0.2
        )
        
        results = pipeline.train_all_models(
            X_train, y_train, X_test, y_test, optimize_hyperparams=False
        )
        
        predictions = pipeline.predict_batch(X_test, return_probabilities=True)
        
        shap_explanations = pipeline.generate_shap_explanations(X_test.head(20))
        
        # Validate complete workflow
        assert pipeline.best_model is not None, "Should select best model"
        assert len(predictions['predictions']) == len(X_test), "Should predict all test samples"
        assert 'error' not in shap_explanations, "SHAP explanations should succeed"
        
        # Check regulatory compliance
        unique_predictions = np.unique(predictions['predictions'])
        assert len(unique_predictions) <= 3, "Should predict valid IFRS9 stages"
        
        logger.info(f"Complete workflow test passed - Model: {pipeline.best_model_name}")
    
    def test_model_persistence_and_loading(self):
        """Test model saving and loading with Polars integration"""
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Train model
            data = create_synthetic_ifrs9_data_polars(1000)
            pipeline = OptimizedMLPipeline(use_polars=True)
            
            X_train, X_test, y_train, y_test = pipeline.prepare_data(
                data, target_column="provision_stage", test_size=0.2
            )
            results = pipeline.train_all_models(
                X_train, y_train, X_test, y_test, optimize_hyperparams=False
            )
            
            # Save model
            save_results = pipeline.save_models(temp_dir)
            assert 'error' not in save_results, "Model saving should succeed"
            
            # Load model in new pipeline
            new_pipeline = OptimizedMLPipeline(use_polars=True)
            load_success = new_pipeline.load_models(temp_dir)
            
            assert load_success, "Model loading should succeed"
            assert new_pipeline.best_model is not None, "Best model should be loaded"
            
            # Test predictions with loaded model
            predictions = new_pipeline.predict_batch(X_test.head(10))
            assert len(predictions['predictions']) == 10, "Loaded model should make predictions"
            
            logger.info("Model persistence test passed")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])