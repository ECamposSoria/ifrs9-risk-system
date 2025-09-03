#!/usr/bin/env python3
"""
IFRS9 Comprehensive Load Testing Framework
Production-scale load testing for 1M+ loan portfolios and ML pipelines
"""

import os
import sys
import time
import json
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import uuid
import random
import numpy as np
import polars as pl
import pandas as pd
from pathlib import Path
import logging
from contextlib import contextmanager
import psutil
import aiohttp
import asyncpg
from google.cloud import bigquery, storage
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class LoadTestConfig:
    """Configuration for load testing scenarios"""
    # Data volume settings
    total_loan_records: int = 1_000_000
    total_customer_records: int = 100_000
    payment_history_records: int = 10_000_000
    batch_size: int = 50_000
    
    # Performance targets
    max_pipeline_duration_minutes: int = 150
    target_throughput_records_per_second: int = 1000
    max_memory_usage_gb: int = 16
    max_cpu_usage_percent: int = 80
    
    # Concurrency settings
    max_concurrent_pipelines: int = 3
    max_concurrent_agents: int = 8
    thread_pool_size: int = 16
    process_pool_size: int = 4
    
    # Test scenarios
    scenarios: List[str] = None
    stress_test_multiplier: float = 2.0
    sustained_load_duration_hours: int = 4
    
    # External systems
    bigquery_project: str = "ifrs9-production"
    gcs_bucket: str = "ifrs9-data-bucket"
    postgres_connection: str = "postgresql://user:pass@localhost:5432/ifrs9"
    
    def __post_init__(self):
        if self.scenarios is None:
            self.scenarios = [
                'normal_load',
                'peak_load', 
                'stress_test',
                'endurance_test',
                'recovery_test'
            ]

@dataclass
class LoadTestResults:
    """Results from load testing"""
    scenario_name: str
    start_time: datetime
    end_time: datetime
    total_duration_seconds: float
    records_processed: int
    throughput_records_per_second: float
    peak_memory_usage_mb: float
    avg_cpu_usage_percent: float
    pipeline_success_rate: float
    sla_compliance_rate: float
    error_count: int
    errors: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]

class SyntheticDataGenerator:
    """Generate large-scale synthetic IFRS9 data for load testing"""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.random_seed = 42
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
    
    def generate_loan_portfolio(self, num_records: int) -> pl.DataFrame:
        """Generate synthetic loan portfolio data"""
        logger.info(f"Generating {num_records:,} loan portfolio records")
        
        # Generate loan IDs
        loan_ids = [f"LOAN_{i:010d}" for i in range(num_records)]
        
        # Generate customer IDs (multiple loans per customer)
        customer_ids = [
            f"CUST_{random.randint(1, self.config.total_customer_records):08d}" 
            for _ in range(num_records)
        ]
        
        # Generate realistic loan amounts with log-normal distribution
        loan_amounts = np.random.lognormal(mean=10.5, sigma=1.2, size=num_records)
        loan_amounts = np.clip(loan_amounts, 1000, 10_000_000).astype(int)
        
        # Generate interest rates
        base_rates = np.random.normal(5.5, 2.0, num_records)
        interest_rates = np.clip(base_rates, 0.1, 25.0)
        
        # Generate loan terms (months)
        loan_terms = np.random.choice([12, 24, 36, 48, 60, 84, 120, 240, 360], 
                                    size=num_records, 
                                    p=[0.05, 0.1, 0.15, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05])
        
        # Generate origination dates
        start_date = datetime(2015, 1, 1)
        end_date = datetime(2024, 12, 31)
        date_range = (end_date - start_date).days
        origination_dates = [
            start_date + timedelta(days=random.randint(0, date_range))
            for _ in range(num_records)
        ]
        
        # Generate loan products
        products = ['Personal Loan', 'Auto Loan', 'Mortgage', 'Credit Card', 
                   'Business Loan', 'Student Loan']
        loan_products = np.random.choice(products, size=num_records, 
                                       p=[0.25, 0.2, 0.2, 0.15, 0.1, 0.1])
        
        # Generate current balances
        outstanding_balances = loan_amounts * np.random.uniform(0.1, 1.0, num_records)
        outstanding_balances = outstanding_balances.astype(int)
        
        # Generate payment status
        payment_statuses = np.random.choice(
            ['Current', 'Late 1-30', 'Late 31-60', 'Late 61-90', 'Default'],
            size=num_records,
            p=[0.85, 0.08, 0.04, 0.02, 0.01]
        )
        
        # Generate IFRS9 staging
        # Most loans in Stage 1, some in Stage 2, few in Stage 3
        ifrs9_stages = np.random.choice([1, 2, 3], size=num_records, p=[0.85, 0.13, 0.02])
        
        # Generate credit scores
        credit_scores = np.random.normal(720, 80, num_records).astype(int)
        credit_scores = np.clip(credit_scores, 300, 850)
        
        # Create DataFrame
        loan_data = pl.DataFrame({
            'loan_id': loan_ids,
            'customer_id': customer_ids,
            'loan_amount': loan_amounts,
            'outstanding_balance': outstanding_balances,
            'interest_rate': interest_rates,
            'loan_term_months': loan_terms,
            'origination_date': origination_dates,
            'loan_product': loan_products,
            'payment_status': payment_statuses,
            'ifrs9_stage': ifrs9_stages,
            'credit_score': credit_scores,
            'generated_timestamp': [datetime.now()] * num_records
        })
        
        logger.info(f"Generated loan portfolio with {loan_data.height:,} records")
        return loan_data
    
    def generate_customer_data(self, num_records: int) -> pl.DataFrame:
        """Generate synthetic customer data"""
        logger.info(f"Generating {num_records:,} customer records")
        
        customer_ids = [f"CUST_{i:08d}" for i in range(1, num_records + 1)]
        
        # Generate customer demographics
        ages = np.random.normal(45, 15, num_records).astype(int)
        ages = np.clip(ages, 18, 80)
        
        annual_incomes = np.random.lognormal(mean=10.8, sigma=0.8, size=num_records)
        annual_incomes = np.clip(annual_incomes, 20000, 500000).astype(int)
        
        employment_statuses = np.random.choice(
            ['Employed', 'Self-Employed', 'Unemployed', 'Retired'],
            size=num_records,
            p=[0.65, 0.15, 0.05, 0.15]
        )
        
        # Generate risk indicators
        debt_to_income_ratios = np.random.uniform(0.05, 0.8, num_records)
        
        customer_data = pl.DataFrame({
            'customer_id': customer_ids,
            'age': ages,
            'annual_income': annual_incomes,
            'employment_status': employment_statuses,
            'debt_to_income_ratio': debt_to_income_ratios,
            'generated_timestamp': [datetime.now()] * num_records
        })
        
        logger.info(f"Generated customer data with {customer_data.height:,} records")
        return customer_data
    
    def generate_payment_history(self, num_records: int) -> pl.DataFrame:
        """Generate synthetic payment history data"""
        logger.info(f"Generating {num_records:,} payment history records")
        
        # Generate payment records for existing loans
        payment_ids = [f"PAY_{i:012d}" for i in range(num_records)]
        
        # Loan IDs (referencing existing loans)
        loan_ids = [
            f"LOAN_{random.randint(0, self.config.total_loan_records-1):010d}"
            for _ in range(num_records)
        ]
        
        # Generate payment dates
        start_date = datetime(2015, 1, 1)
        end_date = datetime.now()
        date_range = (end_date - start_date).days
        payment_dates = [
            start_date + timedelta(days=random.randint(0, date_range))
            for _ in range(num_records)
        ]
        
        # Generate payment amounts
        payment_amounts = np.random.lognormal(mean=6.0, sigma=1.0, size=num_records)
        payment_amounts = np.clip(payment_amounts, 50, 50000).astype(int)
        
        # Generate payment statuses
        payment_statuses = np.random.choice(
            ['On Time', 'Late', 'Missed', 'Partial'],
            size=num_records,
            p=[0.88, 0.08, 0.02, 0.02]
        )
        
        # Generate days past due
        days_past_due = np.where(
            payment_statuses == 'On Time', 0,
            np.random.choice([5, 15, 35, 65, 95], size=num_records, 
                           p=[0.4, 0.3, 0.2, 0.07, 0.03])
        )
        
        payment_data = pl.DataFrame({
            'payment_id': payment_ids,
            'loan_id': loan_ids,
            'payment_date': payment_dates,
            'payment_amount': payment_amounts,
            'payment_status': payment_statuses,
            'days_past_due': days_past_due,
            'generated_timestamp': [datetime.now()] * num_records
        })
        
        logger.info(f"Generated payment history with {payment_data.height:,} records")
        return payment_data

class LoadTestMetricsCollector:
    """Collect metrics during load testing"""
    
    def __init__(self):
        # Prometheus metrics
        self.load_test_duration = Histogram(
            'ifrs9_load_test_duration_seconds',
            'Duration of load test scenarios',
            ['scenario', 'status']
        )
        
        self.records_processed = Counter(
            'ifrs9_load_test_records_processed_total',
            'Total records processed during load testing',
            ['scenario', 'data_type']
        )
        
        self.throughput = Gauge(
            'ifrs9_load_test_throughput_records_per_second',
            'Load test throughput in records per second',
            ['scenario']
        )
        
        self.memory_usage = Gauge(
            'ifrs9_load_test_memory_usage_mb',
            'Memory usage during load testing',
            ['scenario']
        )
        
        self.cpu_usage = Gauge(
            'ifrs9_load_test_cpu_usage_percent',
            'CPU usage during load testing',
            ['scenario']
        )
        
        self.error_count = Counter(
            'ifrs9_load_test_errors_total',
            'Total errors during load testing',
            ['scenario', 'error_type']
        )
        
        self.sla_violations = Counter(
            'ifrs9_load_test_sla_violations_total',
            'SLA violations during load testing',
            ['scenario', 'sla_type']
        )

class IFRS9LoadTester:
    """Main load testing orchestrator"""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.data_generator = SyntheticDataGenerator(config)
        self.metrics_collector = LoadTestMetricsCollector()
        self.results: List[LoadTestResults] = []
        
        # Performance monitoring
        self.performance_data = []
        self.monitoring_active = False
        
        # Setup output directories
        self.output_dir = Path("load_test_results")
        self.output_dir.mkdir(exist_ok=True)
    
    @contextmanager
    def performance_monitor(self, scenario_name: str):
        """Context manager for performance monitoring"""
        self.monitoring_active = True
        monitor_thread = threading.Thread(
            target=self._monitor_resources,
            args=(scenario_name,),
            daemon=True
        )
        monitor_thread.start()
        
        try:
            yield
        finally:
            self.monitoring_active = False
    
    def _monitor_resources(self, scenario_name: str):
        """Monitor system resources during testing"""
        while self.monitoring_active:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.metrics_collector.cpu_usage.labels(scenario=scenario_name).set(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_mb = memory.used / 1024 / 1024
                self.metrics_collector.memory_usage.labels(scenario=scenario_name).set(memory_mb)
                
                # Store for analysis
                self.performance_data.append({
                    'timestamp': datetime.now(),
                    'scenario': scenario_name,
                    'cpu_percent': cpu_percent,
                    'memory_mb': memory_mb,
                    'memory_percent': memory.percent
                })
                
                time.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring resources: {e}")
                time.sleep(10)
    
    async def run_normal_load_test(self) -> LoadTestResults:
        """Run normal load test scenario"""
        scenario_name = "normal_load"
        logger.info(f"Starting {scenario_name} test scenario")
        
        start_time = datetime.now()
        errors = []
        records_processed = 0
        
        try:
            with self.performance_monitor(scenario_name):
                # Generate test data
                logger.info("Generating synthetic data...")
                
                # Generate data in batches to manage memory
                total_loans = self.config.total_loan_records
                batch_size = self.config.batch_size
                
                for batch_start in range(0, total_loans, batch_size):
                    batch_end = min(batch_start + batch_size, total_loans)
                    batch_size_actual = batch_end - batch_start
                    
                    logger.info(f"Processing batch {batch_start:,} to {batch_end:,}")
                    
                    # Generate loan data batch
                    loan_batch = self.data_generator.generate_loan_portfolio(batch_size_actual)
                    
                    # Simulate processing
                    await self._simulate_ifrs9_processing(loan_batch, scenario_name)
                    
                    records_processed += batch_size_actual
                    
                    # Update metrics
                    self.metrics_collector.records_processed.labels(
                        scenario=scenario_name, 
                        data_type='loan'
                    ).inc(batch_size_actual)
        
        except Exception as e:
            logger.error(f"Error in {scenario_name}: {e}")
            errors.append({
                'error_type': type(e).__name__,
                'error_message': str(e),
                'timestamp': datetime.now()
            })
            self.metrics_collector.error_count.labels(
                scenario=scenario_name,
                error_type=type(e).__name__
            ).inc()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Calculate metrics
        throughput = records_processed / duration if duration > 0 else 0
        self.metrics_collector.throughput.labels(scenario=scenario_name).set(throughput)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(scenario_name)
        
        # Record test duration
        status = 'success' if not errors else 'failure'
        self.metrics_collector.load_test_duration.labels(
            scenario=scenario_name,
            status=status
        ).observe(duration)
        
        # Check SLA compliance
        sla_compliance_rate = 100.0
        if duration > self.config.max_pipeline_duration_minutes * 60:
            sla_compliance_rate = 0.0
            self.metrics_collector.sla_violations.labels(
                scenario=scenario_name,
                sla_type='duration'
            ).inc()
        
        results = LoadTestResults(
            scenario_name=scenario_name,
            start_time=start_time,
            end_time=end_time,
            total_duration_seconds=duration,
            records_processed=records_processed,
            throughput_records_per_second=throughput,
            peak_memory_usage_mb=performance_metrics.get('peak_memory_mb', 0),
            avg_cpu_usage_percent=performance_metrics.get('avg_cpu_percent', 0),
            pipeline_success_rate=100.0 if not errors else 0.0,
            sla_compliance_rate=sla_compliance_rate,
            error_count=len(errors),
            errors=errors,
            performance_metrics=performance_metrics
        )
        
        self.results.append(results)
        logger.info(f"Completed {scenario_name} test: {throughput:.2f} records/sec")
        
        return results
    
    async def run_stress_test(self) -> LoadTestResults:
        """Run stress test with increased load"""
        scenario_name = "stress_test" 
        logger.info(f"Starting {scenario_name} test scenario")
        
        # Increase load by multiplier
        original_records = self.config.total_loan_records
        stress_records = int(original_records * self.config.stress_test_multiplier)
        
        # Temporarily modify config
        temp_config = LoadTestConfig(
            total_loan_records=stress_records,
            batch_size=self.config.batch_size * 2,  # Larger batches
            max_concurrent_pipelines=self.config.max_concurrent_pipelines * 2
        )
        
        temp_generator = SyntheticDataGenerator(temp_config)
        
        start_time = datetime.now()
        errors = []
        records_processed = 0
        
        try:
            with self.performance_monitor(scenario_name):
                # Run multiple concurrent processing streams
                tasks = []
                
                for stream_id in range(temp_config.max_concurrent_pipelines):
                    task = self._run_processing_stream(
                        temp_generator, 
                        stream_id, 
                        stress_records // temp_config.max_concurrent_pipelines,
                        scenario_name
                    )
                    tasks.append(task)
                
                # Wait for all streams to complete
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, Exception):
                        errors.append({
                            'error_type': type(result).__name__,
                            'error_message': str(result),
                            'timestamp': datetime.now()
                        })
                    else:
                        records_processed += result
        
        except Exception as e:
            logger.error(f"Error in {scenario_name}: {e}")
            errors.append({
                'error_type': type(e).__name__,
                'error_message': str(e),
                'timestamp': datetime.now()
            })
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Calculate results (similar to normal load test)
        throughput = records_processed / duration if duration > 0 else 0
        performance_metrics = self._calculate_performance_metrics(scenario_name)
        
        # More strict SLA for stress test
        sla_compliance_rate = 100.0
        if (duration > self.config.max_pipeline_duration_minutes * 60 * 1.5 or
            performance_metrics.get('peak_memory_mb', 0) > self.config.max_memory_usage_gb * 1024 * 1.2):
            sla_compliance_rate = 0.0
            self.metrics_collector.sla_violations.labels(
                scenario=scenario_name,
                sla_type='stress_limits'
            ).inc()
        
        results = LoadTestResults(
            scenario_name=scenario_name,
            start_time=start_time,
            end_time=end_time,
            total_duration_seconds=duration,
            records_processed=records_processed,
            throughput_records_per_second=throughput,
            peak_memory_usage_mb=performance_metrics.get('peak_memory_mb', 0),
            avg_cpu_usage_percent=performance_metrics.get('avg_cpu_percent', 0),
            pipeline_success_rate=100.0 if not errors else 0.0,
            sla_compliance_rate=sla_compliance_rate,
            error_count=len(errors),
            errors=errors,
            performance_metrics=performance_metrics
        )
        
        self.results.append(results)
        logger.info(f"Completed {scenario_name} test: {throughput:.2f} records/sec")
        
        return results
    
    async def _run_processing_stream(self, generator: SyntheticDataGenerator, 
                                   stream_id: int, records_count: int, scenario_name: str) -> int:
        """Run a single processing stream"""
        logger.info(f"Starting processing stream {stream_id} with {records_count:,} records")
        
        processed_count = 0
        batch_size = self.config.batch_size
        
        for batch_start in range(0, records_count, batch_size):
            batch_end = min(batch_start + batch_size, records_count)
            batch_size_actual = batch_end - batch_start
            
            # Generate data
            loan_batch = generator.generate_loan_portfolio(batch_size_actual)
            
            # Simulate processing
            await self._simulate_ifrs9_processing(loan_batch, f"{scenario_name}_stream_{stream_id}")
            
            processed_count += batch_size_actual
            
            # Small delay to prevent overwhelming the system
            await asyncio.sleep(0.1)
        
        logger.info(f"Completed processing stream {stream_id}: {processed_count:,} records")
        return processed_count
    
    async def _simulate_ifrs9_processing(self, data: pl.DataFrame, context: str):
        """Simulate IFRS9 processing pipeline"""
        try:
            # Simulate data validation
            await asyncio.sleep(0.01)  # Validation time
            
            # Simulate rules engine processing
            await asyncio.sleep(0.05)  # Rules processing time
            
            # Simulate ML model inference
            await asyncio.sleep(0.02)  # ML inference time
            
            # Simulate ECL calculation
            await asyncio.sleep(0.03)  # ECL calculation time
            
            # Simulate reporting
            await asyncio.sleep(0.01)  # Reporting time
            
            logger.debug(f"Processed {data.height:,} records in context: {context}")
            
        except Exception as e:
            logger.error(f"Error processing data in {context}: {e}")
            raise
    
    def _calculate_performance_metrics(self, scenario_name: str) -> Dict[str, Any]:
        """Calculate performance metrics from collected data"""
        scenario_data = [
            d for d in self.performance_data 
            if d['scenario'] == scenario_name
        ]
        
        if not scenario_data:
            return {}
        
        cpu_values = [d['cpu_percent'] for d in scenario_data]
        memory_values = [d['memory_mb'] for d in scenario_data]
        
        return {
            'avg_cpu_percent': sum(cpu_values) / len(cpu_values),
            'peak_cpu_percent': max(cpu_values),
            'avg_memory_mb': sum(memory_values) / len(memory_values),
            'peak_memory_mb': max(memory_values),
            'sample_count': len(scenario_data)
        }
    
    async def run_all_scenarios(self) -> List[LoadTestResults]:
        """Run all configured load test scenarios"""
        logger.info("Starting comprehensive load testing")
        
        all_results = []
        
        # Run each scenario
        if 'normal_load' in self.config.scenarios:
            result = await self.run_normal_load_test()
            all_results.append(result)
            
            # Brief pause between scenarios
            await asyncio.sleep(30)
        
        if 'stress_test' in self.config.scenarios:
            result = await self.run_stress_test()
            all_results.append(result)
        
        # Additional scenarios can be added here
        
        # Save results
        self._save_results(all_results)
        
        logger.info(f"Completed all load test scenarios. Total: {len(all_results)} scenarios")
        return all_results
    
    def _save_results(self, results: List[LoadTestResults]):
        """Save test results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results as JSON
        results_data = [asdict(result) for result in results]
        results_file = self.output_dir / f"load_test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        # Save summary as CSV
        summary_file = self.output_dir / f"load_test_summary_{timestamp}.csv"
        summary_df = pl.DataFrame([
            {
                'scenario': r.scenario_name,
                'duration_seconds': r.total_duration_seconds,
                'records_processed': r.records_processed,
                'throughput_rps': r.throughput_records_per_second,
                'peak_memory_mb': r.peak_memory_usage_mb,
                'avg_cpu_percent': r.avg_cpu_usage_percent,
                'success_rate': r.pipeline_success_rate,
                'sla_compliance': r.sla_compliance_rate,
                'error_count': r.error_count
            }
            for r in results
        ])
        
        summary_df.write_csv(summary_file)
        
        logger.info(f"Results saved to {results_file} and {summary_file}")

async def main():
    """Main load testing execution"""
    # Configuration
    config = LoadTestConfig(
        total_loan_records=1_000_000,  # 1M records for production scale
        total_customer_records=100_000,
        payment_history_records=5_000_000,
        batch_size=50_000,
        scenarios=['normal_load', 'stress_test']
    )
    
    # Initialize load tester
    load_tester = IFRS9LoadTester(config)
    
    # Run all scenarios
    results = await load_tester.run_all_scenarios()
    
    # Print summary
    logger.info("=== LOAD TEST SUMMARY ===")
    for result in results:
        logger.info(f"Scenario: {result.scenario_name}")
        logger.info(f"  Duration: {result.total_duration_seconds:.2f} seconds")
        logger.info(f"  Records: {result.records_processed:,}")
        logger.info(f"  Throughput: {result.throughput_records_per_second:.2f} records/sec")
        logger.info(f"  Peak Memory: {result.peak_memory_usage_mb:.2f} MB")
        logger.info(f"  SLA Compliance: {result.sla_compliance_rate:.2f}%")
        logger.info(f"  Errors: {result.error_count}")
        logger.info("")

if __name__ == "__main__":
    asyncio.run(main())