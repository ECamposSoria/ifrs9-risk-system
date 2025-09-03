"""
Performance benchmarking suite for Polars ML integration.

This module provides comprehensive benchmarking tools to compare
Polars vs Pandas performance in ML workflows for IFRS9 credit risk modeling.
"""

import time
import logging
from typing import Dict, List, Any, Tuple, Union, Optional
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error

# Import our enhanced ML components
from enhanced_ml_models import OptimizedMLPipeline, AdvancedFeatureEngineer
from polars_ml_integration import PolarsEnhancedCreditRiskClassifier, create_synthetic_ifrs9_data_polars

logger = logging.getLogger(__name__)

class PolarsMLBenchmarkSuite:
    """Comprehensive benchmarking suite for Polars ML integration"""
    
    def __init__(self, random_state: int = 42):
        """Initialize benchmark suite"""
        self.random_state = random_state
        self.results = {}
        self.benchmark_timestamp = datetime.now().isoformat()
        
    def generate_benchmark_datasets(self, sizes: List[int] = [1000, 5000, 25000, 100000, 500000]) -> Dict[int, Tuple[pl.DataFrame, pd.DataFrame]]:
        """Generate datasets of various sizes for benchmarking"""
        
        datasets = {}
        
        for size in sizes:
            logger.info(f"Generating dataset with {size} rows")
            
            # Generate Polars dataset
            pl_data = create_synthetic_ifrs9_data_polars(size)
            
            # Convert to pandas
            pd_data = pl_data.to_pandas()
            
            datasets[size] = (pl_data, pd_data)
            
        logger.info(f"Generated {len(datasets)} datasets with sizes: {list(datasets.keys())}")
        return datasets
        
    def benchmark_feature_engineering(self, datasets: Dict[int, Tuple[pl.DataFrame, pd.DataFrame]], iterations: int = 3) -> Dict[str, Any]:
        """Benchmark feature engineering performance"""
        
        logger.info("Starting feature engineering benchmark")
        results = {}
        
        for size, (pl_data, pd_data) in datasets.items():
            logger.info(f"Benchmarking feature engineering for {size} rows")
            
            # Initialize feature engineers
            polars_fe = AdvancedFeatureEngineer(use_polars=True, lazy_evaluation=True)
            pandas_fe = AdvancedFeatureEngineer(use_polars=False)
            
            # Fit both
            polars_fe.fit(pl_data)
            pandas_fe.fit(pd_data)
            
            # Benchmark Polars feature engineering
            polars_times = []
            for i in range(iterations):
                start_time = time.time()
                polars_features = polars_fe.transform(pl_data)
                polars_times.append(time.time() - start_time)
                
            # Benchmark Pandas feature engineering
            pandas_times = []
            for i in range(iterations):
                start_time = time.time()
                pandas_features = pandas_fe.transform(pd_data)
                pandas_times.append(time.time() - start_time)
            
            # Calculate statistics
            polars_avg = np.mean(polars_times)
            pandas_avg = np.mean(pandas_times)
            speedup = pandas_avg / polars_avg if polars_avg > 0 else 0
            
            results[size] = {
                'polars_avg_time': polars_avg,
                'pandas_avg_time': pandas_avg,
                'polars_std_time': np.std(polars_times),
                'pandas_std_time': np.std(pandas_times),
                'speedup_factor': speedup,
                'polars_features_count': len(polars_features.columns),
                'pandas_features_count': len(pandas_features.columns),
                'iterations': iterations
            }
            
            logger.info(f"Size {size}: Polars {polars_avg:.3f}s, Pandas {pandas_avg:.3f}s, Speedup: {speedup:.2f}x")
        
        return results
    
    def benchmark_model_training(self, datasets: Dict[int, Tuple[pl.DataFrame, pd.DataFrame]], model_types: List[str] = ["xgboost", "lightgbm"]) -> Dict[str, Any]:
        """Benchmark model training performance"""
        
        logger.info("Starting model training benchmark")
        results = {}
        
        for model_type in model_types:
            logger.info(f"Benchmarking {model_type} training")
            results[model_type] = {}
            
            for size, (pl_data, pd_data) in datasets.items():
                if size > 100000:  # Skip very large datasets for training benchmark
                    continue
                    
                logger.info(f"Training {model_type} with {size} rows")
                
                # Polars-enhanced training
                polars_start = time.time()
                polars_classifier = PolarsEnhancedCreditRiskClassifier(
                    model_type=model_type,
                    use_lazy_evaluation=True,
                    polars_streaming=True
                )
                
                polars_features, _ = polars_classifier.prepare_features_polars(pl_data)
                polars_metrics = polars_classifier.train_stage_classifier_polars(
                    polars_features, pl_data["provision_stage"]
                )
                polars_time = time.time() - polars_start
                
                # Standard pandas training (using OptimizedMLPipeline)
                pandas_start = time.time()
                pandas_pipeline = OptimizedMLPipeline(
                    use_polars=False,
                    optimization_method="optuna"
                )
                
                try:
                    X_train, X_test, y_train, y_test = pandas_pipeline.prepare_data(
                        pd_data, target_column="provision_stage", test_size=0.2
                    )
                    pandas_results = pandas_pipeline.train_all_models(
                        X_train, y_train, X_test, y_test, optimize_hyperparams=False
                    )
                    pandas_time = time.time() - pandas_start
                    pandas_accuracy = pandas_results[list(pandas_results.keys())[0]]['evaluation']['test_accuracy']
                except Exception as e:
                    logger.error(f"Pandas training failed for {model_type}: {e}")
                    pandas_time = float('inf')
                    pandas_accuracy = 0
                
                speedup = pandas_time / polars_time if polars_time > 0 else 0
                
                results[model_type][size] = {
                    'polars_time': polars_time,
                    'pandas_time': pandas_time,
                    'speedup_factor': speedup,
                    'polars_accuracy': polars_metrics.get('accuracy', 0),
                    'pandas_accuracy': pandas_accuracy,
                    'accuracy_difference': abs(polars_metrics.get('accuracy', 0) - pandas_accuracy)
                }
                
                logger.info(f"{model_type} Size {size}: Polars {polars_time:.3f}s, Pandas {pandas_time:.3f}s, Speedup: {speedup:.2f}x")
        
        return results
    
    def benchmark_batch_prediction(self, datasets: Dict[int, Tuple[pl.DataFrame, pd.DataFrame]], trained_pipeline: OptimizedMLPipeline) -> Dict[str, Any]:
        """Benchmark batch prediction performance"""
        
        logger.info("Starting batch prediction benchmark")
        results = {}
        
        for size, (pl_data, pd_data) in datasets.items():
            logger.info(f"Benchmarking prediction for {size} rows")
            
            # Prepare features
            X_polars, _ = trained_pipeline.feature_engineer.transform(pl_data), None
            X_pandas = trained_pipeline.feature_engineer.transform(pd_data)
            
            # Benchmark Polars prediction
            polars_start = time.time()
            polars_pred_results = trained_pipeline.predict_batch(X_polars, return_probabilities=False)
            polars_time = time.time() - polars_start
            
            # Benchmark Pandas prediction
            pandas_start = time.time()
            pandas_pred_results = trained_pipeline.predict_batch(X_pandas, return_probabilities=False)
            pandas_time = time.time() - pandas_start
            
            # Benchmark streaming prediction for large datasets
            streaming_time = None
            if size >= 50000:
                streaming_start = time.time()
                streaming_results = trained_pipeline.predict_batch(
                    X_polars, return_probabilities=False, batch_size=10000
                )
                streaming_time = time.time() - streaming_start
            
            speedup = pandas_time / polars_time if polars_time > 0 else 0
            streaming_speedup = polars_time / streaming_time if streaming_time and streaming_time > 0 else None
            
            results[size] = {
                'polars_time': polars_time,
                'pandas_time': pandas_time,
                'streaming_time': streaming_time,
                'speedup_factor': speedup,
                'streaming_speedup': streaming_speedup,
                'predictions_match': np.array_equal(
                    polars_pred_results['predictions'], 
                    pandas_pred_results['predictions']
                ) if 'predictions' in polars_pred_results and 'predictions' in pandas_pred_results else False
            }
            
            logger.info(f"Size {size}: Polars {polars_time:.3f}s, Pandas {pandas_time:.3f}s, Speedup: {speedup:.2f}x")
            if streaming_time:
                logger.info(f"  Streaming: {streaming_time:.3f}s, Streaming speedup: {streaming_speedup:.2f}x")
    
        return results
    
    def benchmark_memory_usage(self, datasets: Dict[int, Tuple[pl.DataFrame, pd.DataFrame]]) -> Dict[str, Any]:
        """Benchmark memory usage comparison"""
        
        import psutil
        import os
        
        logger.info("Starting memory usage benchmark")
        results = {}
        
        process = psutil.Process(os.getpid())
        
        for size, (pl_data, pd_data) in datasets.items():
            logger.info(f"Benchmarking memory usage for {size} rows")
            
            # Measure baseline memory
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Measure Polars feature engineering memory
            polars_fe = AdvancedFeatureEngineer(use_polars=True, lazy_evaluation=True)
            polars_fe.fit(pl_data)
            
            polars_start_memory = process.memory_info().rss / 1024 / 1024
            polars_features = polars_fe.transform(pl_data)
            polars_peak_memory = process.memory_info().rss / 1024 / 1024
            del polars_features
            
            # Measure Pandas feature engineering memory
            pandas_fe = AdvancedFeatureEngineer(use_polars=False)
            pandas_fe.fit(pd_data)
            
            pandas_start_memory = process.memory_info().rss / 1024 / 1024
            pandas_features = pandas_fe.transform(pd_data)
            pandas_peak_memory = process.memory_info().rss / 1024 / 1024
            del pandas_features
            
            results[size] = {
                'baseline_memory_mb': baseline_memory,
                'polars_peak_memory_mb': polars_peak_memory,
                'pandas_peak_memory_mb': pandas_peak_memory,
                'polars_memory_increase': polars_peak_memory - baseline_memory,
                'pandas_memory_increase': pandas_peak_memory - baseline_memory,
                'memory_efficiency_ratio': (pandas_peak_memory - baseline_memory) / (polars_peak_memory - baseline_memory) if (polars_peak_memory - baseline_memory) > 0 else 1
            }
            
            logger.info(f"Size {size}: Polars {polars_peak_memory:.1f}MB, Pandas {pandas_peak_memory:.1f}MB")
        
        return results
    
    def run_comprehensive_benchmark(self, dataset_sizes: List[int] = [1000, 5000, 25000, 100000]) -> Dict[str, Any]:
        """Run comprehensive benchmark suite"""
        
        logger.info("Starting comprehensive Polars ML benchmark suite")
        start_time = time.time()
        
        # Generate datasets
        datasets = self.generate_benchmark_datasets(dataset_sizes)
        
        # Run all benchmarks
        benchmark_results = {
            'metadata': {
                'timestamp': self.benchmark_timestamp,
                'dataset_sizes': dataset_sizes,
                'polars_version': pl.__version__,
                'total_runtime': None
            },
            'feature_engineering': self.benchmark_feature_engineering(datasets),
            'model_training': self.benchmark_model_training(datasets),
            'memory_usage': self.benchmark_memory_usage(datasets)
        }
        
        # Train a model for prediction benchmarking
        small_dataset = datasets[min(dataset_sizes)]
        pipeline = OptimizedMLPipeline(use_polars=True, optimization_method="optuna")
        
        try:
            X_train, X_test, y_train, y_test = pipeline.prepare_data(
                small_dataset[0], target_column="provision_stage", test_size=0.2
            )
            pipeline.train_all_models(X_train, y_train, X_test, y_test, optimize_hyperparams=False)
            
            benchmark_results['batch_prediction'] = self.benchmark_batch_prediction(datasets, pipeline)
        except Exception as e:
            logger.error(f"Failed to train model for prediction benchmarking: {e}")
            benchmark_results['batch_prediction'] = {'error': str(e)}
        
        total_time = time.time() - start_time
        benchmark_results['metadata']['total_runtime'] = total_time
        
        logger.info(f"Comprehensive benchmark completed in {total_time:.2f} seconds")
        
        return benchmark_results
    
    def generate_benchmark_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive benchmark report"""
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("POLARS ML INTEGRATION BENCHMARK REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Timestamp: {results['metadata']['timestamp']}")
        report_lines.append(f"Polars Version: {results['metadata'].get('polars_version', 'Unknown')}")
        report_lines.append(f"Dataset Sizes: {results['metadata']['dataset_sizes']}")
        report_lines.append(f"Total Runtime: {results['metadata'].get('total_runtime', 0):.2f} seconds")
        report_lines.append("")
        
        # Feature Engineering Results
        if 'feature_engineering' in results:
            report_lines.append("FEATURE ENGINEERING PERFORMANCE")
            report_lines.append("-" * 50)
            for size, metrics in results['feature_engineering'].items():
                speedup = metrics['speedup_factor']
                report_lines.append(f"Dataset Size: {size:,} rows")
                report_lines.append(f"  Polars Time:  {metrics['polars_avg_time']:.3f}s (±{metrics['polars_std_time']:.3f})")
                report_lines.append(f"  Pandas Time:  {metrics['pandas_avg_time']:.3f}s (±{metrics['pandas_std_time']:.3f})")
                report_lines.append(f"  Speedup:      {speedup:.2f}x ({'FASTER' if speedup > 1 else 'SLOWER'})")
                report_lines.append(f"  Features:     {metrics['polars_features_count']} (Polars) vs {metrics['pandas_features_count']} (Pandas)")
                report_lines.append("")
        
        # Model Training Results
        if 'model_training' in results:
            report_lines.append("MODEL TRAINING PERFORMANCE")
            report_lines.append("-" * 50)
            for model_type, model_results in results['model_training'].items():
                report_lines.append(f"Model: {model_type.upper()}")
                for size, metrics in model_results.items():
                    speedup = metrics['speedup_factor']
                    report_lines.append(f"  Dataset Size: {size:,} rows")
                    report_lines.append(f"    Training Time: Polars {metrics['polars_time']:.3f}s vs Pandas {metrics['pandas_time']:.3f}s")
                    report_lines.append(f"    Speedup: {speedup:.2f}x")
                    report_lines.append(f"    Accuracy: Polars {metrics['polars_accuracy']:.4f} vs Pandas {metrics['pandas_accuracy']:.4f}")
                    report_lines.append(f"    Accuracy Diff: {metrics['accuracy_difference']:.4f}")
                report_lines.append("")
        
        # Memory Usage Results
        if 'memory_usage' in results:
            report_lines.append("MEMORY USAGE COMPARISON")
            report_lines.append("-" * 50)
            for size, metrics in results['memory_usage'].items():
                efficiency = metrics['memory_efficiency_ratio']
                report_lines.append(f"Dataset Size: {size:,} rows")
                report_lines.append(f"  Polars Peak:     {metrics['polars_peak_memory_mb']:.1f} MB")
                report_lines.append(f"  Pandas Peak:     {metrics['pandas_peak_memory_mb']:.1f} MB")
                report_lines.append(f"  Memory Efficiency: {efficiency:.2f}x ({'BETTER' if efficiency > 1 else 'WORSE'})")
                report_lines.append("")
        
        # Batch Prediction Results
        if 'batch_prediction' in results and 'error' not in results['batch_prediction']:
            report_lines.append("BATCH PREDICTION PERFORMANCE")
            report_lines.append("-" * 50)
            for size, metrics in results['batch_prediction'].items():
                speedup = metrics['speedup_factor']
                report_lines.append(f"Dataset Size: {size:,} rows")
                report_lines.append(f"  Polars Time:    {metrics['polars_time']:.3f}s")
                report_lines.append(f"  Pandas Time:    {metrics['pandas_time']:.3f}s")
                report_lines.append(f"  Speedup:        {speedup:.2f}x")
                if metrics.get('streaming_time'):
                    report_lines.append(f"  Streaming Time: {metrics['streaming_time']:.3f}s")
                    report_lines.append(f"  Streaming Speedup: {metrics.get('streaming_speedup', 0):.2f}x")
                report_lines.append(f"  Predictions Match: {metrics['predictions_match']}")
                report_lines.append("")
        
        # Summary
        report_lines.append("SUMMARY")
        report_lines.append("-" * 50)
        
        # Calculate overall performance gains
        if 'feature_engineering' in results:
            avg_fe_speedup = np.mean([m['speedup_factor'] for m in results['feature_engineering'].values()])
            report_lines.append(f"Average Feature Engineering Speedup: {avg_fe_speedup:.2f}x")
        
        if 'batch_prediction' in results and 'error' not in results['batch_prediction']:
            avg_pred_speedup = np.mean([m['speedup_factor'] for m in results['batch_prediction'].values()])
            report_lines.append(f"Average Prediction Speedup: {avg_pred_speedup:.2f}x")
        
        report_lines.append("")
        report_lines.append("RECOMMENDATIONS:")
        
        if 'feature_engineering' in results:
            max_fe_speedup = max([m['speedup_factor'] for m in results['feature_engineering'].values()])
            if max_fe_speedup > 2:
                report_lines.append("✓ Polars provides significant feature engineering performance improvements (>2x)")
            elif max_fe_speedup > 1.2:
                report_lines.append("✓ Polars provides moderate feature engineering improvements (>20%)")
            else:
                report_lines.append("- Limited feature engineering performance gains observed")
        
        if 'memory_usage' in results:
            avg_memory_efficiency = np.mean([m['memory_efficiency_ratio'] for m in results['memory_usage'].values()])
            if avg_memory_efficiency > 1.5:
                report_lines.append("✓ Polars demonstrates superior memory efficiency (>1.5x better)")
            elif avg_memory_efficiency > 1.1:
                report_lines.append("✓ Polars shows improved memory efficiency (>10% better)")
            else:
                report_lines.append("- Similar memory usage patterns between Polars and Pandas")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def save_benchmark_results(self, results: Dict[str, Any], output_dir: str = "benchmarks/") -> str:
        """Save benchmark results and report"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw results
        import json
        results_file = output_path / f"polars_ml_benchmark_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save report
        report = self.generate_benchmark_report(results)
        report_file = output_path / f"polars_ml_benchmark_report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Benchmark results saved to: {results_file}")
        logger.info(f"Benchmark report saved to: {report_file}")
        
        return str(report_file)


if __name__ == "__main__":
    # Run comprehensive benchmark
    benchmark_suite = PolarsMLBenchmarkSuite()
    
    # Use smaller dataset sizes for testing
    test_sizes = [1000, 5000, 25000]
    
    results = benchmark_suite.run_comprehensive_benchmark(test_sizes)
    
    # Print report
    report = benchmark_suite.generate_benchmark_report(results)
    print(report)
    
    # Save results
    report_file = benchmark_suite.save_benchmark_results(results)
    print(f"\nFull benchmark report saved to: {report_file}")