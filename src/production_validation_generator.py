#!/usr/bin/env python3
"""
IFRS9 Production Validation Dataset Generator
============================================

This module generates comprehensive validation datasets for production readiness testing
of the IFRS9 Risk Management System. It creates enterprise-grade synthetic data that
covers all testing scenarios required for Go-Live validation.

Features:
- Production-scale datasets (1M+ records)
- Edge case and stress test scenarios
- Multi-currency and multi-region support
- Regulatory compliance validation data
- ML model validation datasets
- Integration testing data
- Performance validation datasets
"""

import os
import sys
import json
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from faker import Faker
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats
import polars as pl

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from generate_data import DataGenerator


@dataclass
class ValidationScenario:
    """Configuration for a specific validation scenario."""
    name: str
    description: str
    dataset_size: int
    special_conditions: Dict[str, Any]
    expected_outcomes: Dict[str, Any]


class ProductionValidationGenerator:
    """
    Enterprise-grade validation dataset generator for IFRS9 production testing.
    
    This generator creates comprehensive datasets that simulate:
    - Normal production workloads
    - Edge cases and stress scenarios
    - Regulatory compliance situations
    - Multi-dimensional risk scenarios
    """
    
    def __init__(self, seed: int = 42, output_dir: str = "validation_datasets"):
        """Initialize the production validation generator."""
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self.fake = Faker(['en_US', 'es_ES', 'en_GB', 'fr_FR', 'de_DE'])
        Faker.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Initialize base data generator
        self.base_generator = DataGenerator(seed=seed)
        
        # Production configuration
        self.currencies = ["EUR", "USD", "GBP", "CAD", "AUD"]
        self.regions = {
            "EUR": ["Spain", "Germany", "France", "Italy", "Netherlands"],
            "USD": ["United_States", "Canada", "Mexico"],
            "GBP": ["United_Kingdom", "Ireland"],
            "CAD": ["Canada"],
            "AUD": ["Australia", "New_Zealand"]
        }
        
        # Economic scenarios for stress testing
        self.economic_scenarios = {
            "baseline": {"gdp_growth": 2.5, "unemployment": 6.0, "default_multiplier": 1.0},
            "mild_recession": {"gdp_growth": -0.5, "unemployment": 8.5, "default_multiplier": 1.5},
            "severe_recession": {"gdp_growth": -3.0, "unemployment": 12.0, "default_multiplier": 2.5},
            "financial_crisis": {"gdp_growth": -5.0, "unemployment": 15.0, "default_multiplier": 4.0},
            "recovery": {"gdp_growth": 4.0, "unemployment": 4.5, "default_multiplier": 0.7}
        }
        
        self.validation_scenarios = self._define_validation_scenarios()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        log_path = self.output_dir / "generation.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _define_validation_scenarios(self) -> List[ValidationScenario]:
        """Define comprehensive validation scenarios."""
        return [
            # 1. Production Scale Validation Datasets
            ValidationScenario(
                name="production_baseline",
                description="Standard production workload simulation",
                dataset_size=1000000,
                special_conditions={
                    "stage_distribution": {"stage_1": 0.85, "stage_2": 0.12, "stage_3": 0.03},
                    "product_mix": "balanced",
                    "temporal_patterns": True,
                    "seasonality": True
                },
                expected_outcomes={
                    "data_quality_score": 99.0,
                    "processing_time_limit": 150,  # minutes
                    "memory_usage_limit": "16GB"
                }
            ),
            
            ValidationScenario(
                name="edge_cases_stress",
                description="Edge cases and stress test scenarios",
                dataset_size=100000,
                special_conditions={
                    "high_default_rate": 0.15,
                    "extreme_ltv_ratios": True,
                    "borderline_dpd_cases": True,
                    "economic_scenario": "financial_crisis"
                },
                expected_outcomes={
                    "stage_3_percentage": ">10%",
                    "high_risk_flags": ">20%"
                }
            ),
            
            ValidationScenario(
                name="multi_currency_multi_region",
                description="Multi-currency and multi-region portfolio",
                dataset_size=500000,
                special_conditions={
                    "currencies": ["EUR", "USD", "GBP", "CAD", "AUD"],
                    "regional_diversity": True,
                    "fx_exposure": True,
                    "regulatory_variations": True
                },
                expected_outcomes={
                    "currency_coverage": 5,
                    "region_coverage": 15
                }
            ),
            
            ValidationScenario(
                name="corporate_retail_sme_mix",
                description="Mixed portfolio with Corporate, Retail, and SME segments",
                dataset_size=750000,
                special_conditions={
                    "segment_mix": {"corporate": 0.2, "retail": 0.6, "sme": 0.2},
                    "varied_risk_profiles": True,
                    "sector_concentration": True
                },
                expected_outcomes={
                    "segment_compliance": True,
                    "sector_diversification": ">10_sectors"
                }
            ),
            
            ValidationScenario(
                name="temporal_seasonality",
                description="Temporal patterns and seasonality simulation",
                dataset_size=300000,
                special_conditions={
                    "origination_seasonality": True,
                    "payment_patterns": "seasonal",
                    "economic_cycles": True,
                    "holiday_impacts": True
                },
                expected_outcomes={
                    "seasonal_patterns_detected": True,
                    "temporal_consistency": True
                }
            ),
            
            # 2. Regulatory Compliance Validation
            ValidationScenario(
                name="ifrs9_standard_validation",
                description="IFRS9 standard compliance validation",
                dataset_size=200000,
                special_conditions={
                    "strict_ifrs9_compliance": True,
                    "audit_trail_complete": True,
                    "documentation_standards": "full"
                },
                expected_outcomes={
                    "ifrs9_compliance_score": 100.0,
                    "audit_readiness": True
                }
            ),
            
            ValidationScenario(
                name="stage_classification_borderline",
                description="Borderline stage classification test cases",
                dataset_size=50000,
                special_conditions={
                    "dpd_29_30_cases": 5000,
                    "dpd_89_90_cases": 3000,
                    "significant_increase_edge_cases": 2000,
                    "forbearance_cases": 1000
                },
                expected_outcomes={
                    "classification_accuracy": ">98%",
                    "edge_case_handling": "correct"
                }
            ),
            
            ValidationScenario(
                name="forward_looking_scenarios",
                description="Forward-looking economic scenario testing",
                dataset_size=150000,
                special_conditions={
                    "multiple_scenarios": ["baseline", "adverse", "severely_adverse"],
                    "scenario_probability_weights": [0.6, 0.3, 0.1],
                    "macro_variable_correlation": True
                },
                expected_outcomes={
                    "scenario_consistency": True,
                    "ecl_sensitivity": "appropriate"
                }
            ),
            
            # 3. ML Model Validation Datasets
            ValidationScenario(
                name="ml_holdout_validation",
                description="Holdout datasets for ML model validation",
                dataset_size=100000,
                special_conditions={
                    "temporal_holdout": True,
                    "stratified_sampling": True,
                    "feature_completeness": 100,
                    "target_balance": "maintained"
                },
                expected_outcomes={
                    "model_performance": ">85%",
                    "feature_importance_stability": True
                }
            ),
            
            ValidationScenario(
                name="adversarial_robustness",
                description="Adversarial test cases for model robustness",
                dataset_size=25000,
                special_conditions={
                    "data_drift_simulation": True,
                    "outlier_injection": 0.05,
                    "feature_noise": 0.02,
                    "adversarial_examples": True
                },
                expected_outcomes={
                    "robustness_score": ">80%",
                    "drift_detection": True
                }
            ),
            
            ValidationScenario(
                name="explainability_validation",
                description="Model explainability validation using SHAP",
                dataset_size=10000,
                special_conditions={
                    "high_risk_cases": 2000,
                    "medium_risk_cases": 5000,
                    "low_risk_cases": 3000,
                    "shap_validation_ready": True
                },
                expected_outcomes={
                    "shap_consistency": True,
                    "feature_attribution_logical": True
                }
            ),
            
            # 4. Integration Validation Datasets
            ValidationScenario(
                name="bigquery_integration",
                description="BigQuery upload/download testing",
                dataset_size=500000,
                special_conditions={
                    "bigquery_schema_compliance": True,
                    "partition_strategy": "date",
                    "large_file_support": True,
                    "concurrent_access": True
                },
                expected_outcomes={
                    "upload_success_rate": 100.0,
                    "query_performance": "<30s"
                }
            ),
            
            ValidationScenario(
                name="gcs_integration",
                description="Google Cloud Storage integration validation",
                dataset_size=1000000,
                special_conditions={
                    "multiple_formats": ["parquet", "csv", "json"],
                    "compression_testing": True,
                    "versioning_support": True,
                    "backup_validation": True
                },
                expected_outcomes={
                    "storage_efficiency": ">80%",
                    "retrieval_speed": "<2min"
                }
            ),
            
            ValidationScenario(
                name="postgresql_stress",
                description="PostgreSQL connection stress testing",
                dataset_size=200000,
                special_conditions={
                    "concurrent_connections": 50,
                    "transaction_volume": "high",
                    "connection_pool_testing": True,
                    "failover_simulation": True
                },
                expected_outcomes={
                    "connection_success_rate": ">99%",
                    "transaction_integrity": 100.0
                }
            ),
            
            ValidationScenario(
                name="airflow_pipeline_e2e",
                description="Airflow pipeline end-to-end validation",
                dataset_size=300000,
                special_conditions={
                    "multi_dag_workflow": True,
                    "dependency_management": "complex",
                    "retry_logic_testing": True,
                    "monitoring_integration": True
                },
                expected_outcomes={
                    "pipeline_success_rate": ">99.5%",
                    "sla_compliance": True
                }
            ),
            
            # 5. Performance Validation Datasets
            ValidationScenario(
                name="sla_compliance_150min",
                description="150-minute SLA compliance testing",
                dataset_size=2000000,
                special_conditions={
                    "processing_complexity": "high",
                    "memory_optimization": True,
                    "parallel_processing": True,
                    "resource_monitoring": True
                },
                expected_outcomes={
                    "processing_time": "<150min",
                    "memory_usage": "<16GB",
                    "cpu_efficiency": ">85%"
                }
            ),
            
            ValidationScenario(
                name="concurrent_processing",
                description="Concurrent processing validation",
                dataset_size=500000,
                special_conditions={
                    "parallel_streams": 8,
                    "resource_contention": True,
                    "load_balancing": True,
                    "deadlock_prevention": True
                },
                expected_outcomes={
                    "throughput_improvement": ">300%",
                    "resource_utilization": "balanced"
                }
            ),
            
            ValidationScenario(
                name="polars_performance_benchmark",
                description="Polars performance benchmarking",
                dataset_size=1500000,
                special_conditions={
                    "complex_aggregations": True,
                    "join_operations": "multiple",
                    "window_functions": True,
                    "lazy_evaluation": True
                },
                expected_outcomes={
                    "polars_vs_pandas_speedup": ">5x",
                    "memory_efficiency": ">60%"
                }
            )
        ]
    
    def generate_all_validation_datasets(self) -> Dict[str, Any]:
        """
        Generate all validation datasets for production readiness testing.
        
        Returns:
            Dictionary containing generation results and metadata
        """
        self.logger.info("🚀 Starting Production Validation Dataset Generation")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Number of scenarios: {len(self.validation_scenarios)}")
        
        generation_results = {
            "generation_start": datetime.now().isoformat(),
            "scenarios": {},
            "summary": {
                "total_scenarios": len(self.validation_scenarios),
                "completed_scenarios": 0,
                "failed_scenarios": 0,
                "total_records_generated": 0,
                "total_disk_space_mb": 0
            },
            "quality_metrics": {},
            "performance_metrics": {}
        }
        
        # Create scenario-specific directories
        for scenario in self.validation_scenarios:
            scenario_dir = self.output_dir / scenario.name
            scenario_dir.mkdir(exist_ok=True)
        
        # Generate datasets for each scenario
        for scenario in self.validation_scenarios:
            self.logger.info(f"📊 Generating dataset: {scenario.name}")
            try:
                scenario_start = datetime.now()
                
                # Generate the scenario dataset
                dataset_result = self._generate_scenario_dataset(scenario)
                
                scenario_end = datetime.now()
                processing_time = (scenario_end - scenario_start).total_seconds()
                
                # Store results
                generation_results["scenarios"][scenario.name] = {
                    "status": "completed",
                    "processing_time_seconds": processing_time,
                    "records_generated": dataset_result["record_count"],
                    "files_generated": dataset_result["files"],
                    "quality_score": dataset_result["quality_score"],
                    "validation_passed": dataset_result["validation_passed"]
                }
                
                generation_results["summary"]["completed_scenarios"] += 1
                generation_results["summary"]["total_records_generated"] += dataset_result["record_count"]
                
                self.logger.info(f"✅ Completed {scenario.name} - {dataset_result['record_count']:,} records in {processing_time:.1f}s")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to generate {scenario.name}: {str(e)}")
                generation_results["scenarios"][scenario.name] = {
                    "status": "failed",
                    "error": str(e)
                }
                generation_results["summary"]["failed_scenarios"] += 1
        
        # Generate comprehensive quality report
        self._generate_quality_report(generation_results)
        
        # Generate master validation orchestrator configuration
        self._generate_master_orchestrator_config(generation_results)
        
        generation_results["generation_end"] = datetime.now().isoformat()
        
        # Save generation metadata
        with open(self.output_dir / "generation_results.json", "w") as f:
            json.dump(generation_results, f, indent=2, default=str)
        
        self.logger.info("🎉 Production Validation Dataset Generation Completed")
        self.logger.info(f"Total scenarios: {generation_results['summary']['completed_scenarios']}/{generation_results['summary']['total_scenarios']}")
        self.logger.info(f"Total records: {generation_results['summary']['total_records_generated']:,}")
        
        return generation_results
    
    def _generate_scenario_dataset(self, scenario: ValidationScenario) -> Dict[str, Any]:
        """Generate dataset for a specific validation scenario."""
        scenario_dir = self.output_dir / scenario.name
        
        # Initialize scenario-specific data generator
        scenario_generator = self._create_scenario_generator(scenario)
        
        # Generate core datasets
        datasets = {}
        
        # 1. Generate loan portfolio
        self.logger.info(f"  Generating loan portfolio ({scenario.dataset_size:,} loans)")
        loan_portfolio = self._generate_scenario_loan_portfolio(scenario, scenario_generator)
        datasets["loan_portfolio"] = loan_portfolio
        
        # 2. Generate payment history
        self.logger.info(f"  Generating payment history")
        payment_history = self._generate_scenario_payment_history(scenario, loan_portfolio)
        datasets["payment_history"] = payment_history
        
        # 3. Generate macroeconomic data
        self.logger.info(f"  Generating macroeconomic data")
        macro_data = self._generate_scenario_macro_data(scenario)
        datasets["macro_data"] = macro_data
        
        # 4. Generate stage transitions
        self.logger.info(f"  Generating stage transitions")
        stage_transitions = self._generate_scenario_stage_transitions(scenario, loan_portfolio)
        datasets["stage_transitions"] = stage_transitions
        
        # 5. Generate scenario-specific additional datasets
        additional_datasets = self._generate_additional_scenario_datasets(scenario, datasets)
        datasets.update(additional_datasets)
        
        # Save datasets in multiple formats
        files_generated = self._save_scenario_datasets(scenario_dir, datasets, scenario)
        
        # Validate generated data
        quality_score, validation_passed = self._validate_scenario_data(scenario, datasets)
        
        # Generate scenario documentation
        self._generate_scenario_documentation(scenario_dir, scenario, datasets)
        
        return {
            "record_count": sum(len(df) for df in datasets.values() if isinstance(df, pd.DataFrame)),
            "files": files_generated,
            "quality_score": quality_score,
            "validation_passed": validation_passed,
            "datasets": list(datasets.keys())
        }
    
    def _create_scenario_generator(self, scenario: ValidationScenario) -> DataGenerator:
        """Create a scenario-specific data generator."""
        # Create a new generator with scenario-specific seed
        scenario_seed = hash(scenario.name) % (2**32)
        return DataGenerator(seed=scenario_seed)
    
    def _generate_scenario_loan_portfolio(self, scenario: ValidationScenario, generator: DataGenerator) -> pd.DataFrame:
        """Generate loan portfolio for specific scenario."""
        conditions = scenario.special_conditions
        
        # Start with base generation
        loans = []
        batch_size = 10000
        
        for batch_start in range(0, scenario.dataset_size, batch_size):
            batch_end = min(batch_start + batch_size, scenario.dataset_size)
            
            batch_loans = []
            for loan_id in range(batch_start + 1, batch_end + 1):
                loan = generator._generate_single_loan(loan_id, "2020-01-01", "2024-12-31")
                
                # Apply scenario-specific modifications
                loan = self._apply_scenario_modifications(loan, scenario)
                
                batch_loans.append(loan)
            
            loans.extend(batch_loans)
            
            # Memory management for large datasets
            if len(loans) % 50000 == 0:
                self.logger.info(f"    Generated {len(loans):,} loans...")
        
        df = pd.DataFrame(loans)
        
        # Apply scenario-specific distribution adjustments
        df = self._adjust_scenario_distributions(df, scenario)
        
        return df
    
    def _apply_scenario_modifications(self, loan: Dict, scenario: ValidationScenario) -> Dict:
        """Apply scenario-specific modifications to individual loan."""
        conditions = scenario.special_conditions
        
        # Multi-currency support
        if "currencies" in conditions:
            loan["currency"] = random.choice(conditions["currencies"])
            if loan["currency"] != "EUR":
                # Apply currency-specific adjustments
                fx_rate = random.uniform(0.8, 1.3)
                loan["loan_amount"] *= fx_rate
                loan["current_balance"] *= fx_rate
        
        # High default rate scenarios
        if "high_default_rate" in conditions:
            if random.random() < conditions["high_default_rate"]:
                loan["days_past_due"] = random.randint(91, 365)
                loan["provision_stage"] = 3
                loan["credit_score"] = random.randint(300, 500)
        
        # Extreme LTV ratios for stress testing
        if "extreme_ltv_ratios" in conditions:
            if random.random() < 0.1:  # 10% of loans have extreme LTV
                loan["ltv_ratio"] = random.uniform(0.95, 1.2)
                if loan["collateral_value"] > 0:
                    loan["loan_amount"] = loan["collateral_value"] * loan["ltv_ratio"]
        
        # Borderline DPD cases
        if "borderline_dpd_cases" in conditions:
            if random.random() < 0.1:  # 10% borderline cases
                loan["days_past_due"] = random.choice([29, 30, 89, 90])
        
        # Economic scenario adjustments
        if "economic_scenario" in conditions:
            scenario_name = conditions["economic_scenario"]
            if scenario_name in self.economic_scenarios:
                multiplier = self.economic_scenarios[scenario_name]["default_multiplier"]
                loan["pd_12m"] = min(1.0, loan["pd_12m"] * multiplier)
                loan["pd_lifetime"] = min(1.0, loan["pd_lifetime"] * multiplier)
        
        return loan
    
    def _adjust_scenario_distributions(self, df: pd.DataFrame, scenario: ValidationScenario) -> pd.DataFrame:
        """Adjust dataset distributions based on scenario requirements."""
        conditions = scenario.special_conditions
        
        # Adjust stage distribution
        if "stage_distribution" in conditions:
            target_dist = conditions["stage_distribution"]
            
            # Randomly assign stages based on target distribution
            n_loans = len(df)
            stage_1_count = int(n_loans * target_dist.get("stage_1", 0.85))
            stage_2_count = int(n_loans * target_dist.get("stage_2", 0.12))
            stage_3_count = n_loans - stage_1_count - stage_2_count
            
            # Create stage assignments
            stages = ([1] * stage_1_count + [2] * stage_2_count + [3] * stage_3_count)
            random.shuffle(stages)
            
            df["provision_stage"] = stages[:n_loans]
            
            # Adjust DPD based on stage
            df.loc[df["provision_stage"] == 1, "days_past_due"] = 0
            df.loc[df["provision_stage"] == 2, "days_past_due"] = np.random.randint(1, 90, size=sum(df["provision_stage"] == 2))
            df.loc[df["provision_stage"] == 3, "days_past_due"] = np.random.randint(91, 365, size=sum(df["provision_stage"] == 3))
        
        # Segment mix adjustments
        if "segment_mix" in conditions:
            mix = conditions["segment_mix"]
            n_loans = len(df)
            
            segments = []
            for segment, proportion in mix.items():
                count = int(n_loans * proportion)
                segments.extend([segment] * count)
            
            # Fill remaining
            while len(segments) < n_loans:
                segments.append(random.choice(list(mix.keys())))
            
            random.shuffle(segments)
            df["customer_segment"] = segments[:n_loans]
        
        return df
    
    def _generate_scenario_payment_history(self, scenario: ValidationScenario, loan_portfolio: pd.DataFrame) -> pd.DataFrame:
        """Generate payment history for scenario."""
        # For large datasets, sample a subset for payment history
        sample_size = min(50000, len(loan_portfolio))
        sampled_loans = loan_portfolio.sample(n=sample_size)
        
        payment_history = []
        
        for _, loan in sampled_loans.iterrows():
            # Generate 12 months of payment history
            for month in range(12):
                payment = self._generate_scenario_payment_record(loan, month, scenario)
                payment_history.append(payment)
        
        return pd.DataFrame(payment_history)
    
    def _generate_scenario_payment_record(self, loan: pd.Series, month: int, scenario: ValidationScenario) -> Dict:
        """Generate scenario-specific payment record."""
        base_payment = {
            "loan_id": loan["loan_id"],
            "payment_date": (pd.to_datetime(loan["origination_date"]) + pd.DateOffset(months=month)).date(),
            "scheduled_payment": loan["monthly_payment"],
            "actual_payment": loan["monthly_payment"],
            "payment_status": "PAID",
            "days_late": 0
        }
        
        # Apply scenario-specific payment behavior
        conditions = scenario.special_conditions
        
        # Seasonal payment patterns
        if conditions.get("payment_patterns") == "seasonal":
            # Worse performance in winter months (Dec, Jan, Feb)
            if month in [0, 1, 11]:  # Jan, Feb, Dec
                miss_probability = 0.15
            else:
                miss_probability = 0.05
        else:
            miss_probability = 0.02
        
        # Adjust based on loan stage
        if loan["provision_stage"] == 2:
            miss_probability *= 3
        elif loan["provision_stage"] == 3:
            miss_probability *= 5
        
        if random.random() < miss_probability:
            base_payment["actual_payment"] = 0
            base_payment["payment_status"] = "MISSED"
            base_payment["days_late"] = random.randint(1, 30)
        
        return base_payment
    
    def _generate_scenario_macro_data(self, scenario: ValidationScenario) -> pd.DataFrame:
        """Generate macroeconomic data for scenario."""
        conditions = scenario.special_conditions
        
        # Generate 5 years of monthly data
        dates = pd.date_range(start="2020-01-01", end="2024-12-31", freq="M")
        macro_data = []
        
        # Get economic scenario parameters
        econ_scenario = conditions.get("economic_scenario", "baseline")
        if econ_scenario in self.economic_scenarios:
            base_params = self.economic_scenarios[econ_scenario]
        else:
            base_params = self.economic_scenarios["baseline"]
        
        for date in dates:
            record = {
                "date": date.date(),
                "gdp_growth": base_params["gdp_growth"] + random.gauss(0, 0.5),
                "unemployment_rate": base_params["unemployment"] + random.gauss(0, 0.3),
                "inflation_rate": random.uniform(0.5, 5.0),
                "interest_rate": random.uniform(0.0, 7.0),
                "housing_price_index": random.uniform(85, 130),
                "currency_exchange_rate": random.uniform(0.85, 1.20),
                "stock_market_index": random.uniform(8000, 15000),
                "credit_spread": random.uniform(0.5, 4.0),
                "economic_scenario": econ_scenario,
                "created_at": datetime.now()
            }
            macro_data.append(record)
        
        return pd.DataFrame(macro_data)
    
    def _generate_scenario_stage_transitions(self, scenario: ValidationScenario, loan_portfolio: pd.DataFrame) -> pd.DataFrame:
        """Generate stage transitions for scenario."""
        # Sample loans for transition history
        sample_size = min(10000, len(loan_portfolio))
        sampled_loans = loan_portfolio.sample(n=sample_size)
        
        transitions = []
        
        for _, loan in sampled_loans.iterrows():
            # Generate 12 months of transition history
            for month_offset in range(12):
                transition_date = datetime.now() - timedelta(days=30 * month_offset)
                
                # Simulate realistic stage transitions
                current_stage = loan["provision_stage"]
                
                if month_offset == 0:
                    # Current stage
                    from_stage = current_stage
                    to_stage = current_stage
                else:
                    # Historical stages (simplified backward simulation)
                    if current_stage == 3:
                        from_stage = random.choices([1, 2, 3], weights=[0.1, 0.4, 0.5])[0]
                    elif current_stage == 2:
                        from_stage = random.choices([1, 2], weights=[0.6, 0.4])[0]
                    else:
                        from_stage = 1
                    to_stage = current_stage if month_offset == 1 else from_stage
                
                transition = {
                    "loan_id": loan["loan_id"],
                    "transition_date": transition_date.date(),
                    "from_stage": from_stage,
                    "to_stage": to_stage,
                    "trigger_reason": self._get_scenario_transition_reason(from_stage, to_stage, scenario),
                    "created_at": datetime.now()
                }
                transitions.append(transition)
        
        return pd.DataFrame(transitions)
    
    def _get_scenario_transition_reason(self, from_stage: int, to_stage: int, scenario: ValidationScenario) -> str:
        """Get transition reason based on scenario."""
        if scenario.name == "forward_looking_scenarios":
            if from_stage < to_stage:
                return "Forward-looking deterioration"
            elif from_stage > to_stage:
                return "Forward-looking improvement"
        
        return self.base_generator._get_transition_reason(from_stage, to_stage)
    
    def _generate_additional_scenario_datasets(self, scenario: ValidationScenario, datasets: Dict) -> Dict:
        """Generate additional datasets specific to scenario requirements."""
        additional = {}
        
        # Explainability validation dataset
        if scenario.name == "explainability_validation":
            additional["shap_test_cases"] = self._generate_shap_test_cases(datasets["loan_portfolio"])
        
        # Performance benchmark dataset
        if scenario.name == "polars_performance_benchmark":
            additional["benchmark_operations"] = self._generate_benchmark_operations()
        
        # Integration test configurations
        if "integration" in scenario.name:
            additional["integration_config"] = self._generate_integration_config(scenario)
        
        return additional
    
    def _generate_shap_test_cases(self, loan_portfolio: pd.DataFrame) -> pd.DataFrame:
        """Generate specific test cases for SHAP explainability validation."""
        # Select diverse cases across risk spectrum
        high_risk = loan_portfolio[loan_portfolio["provision_stage"] == 3].sample(min(2000, sum(loan_portfolio["provision_stage"] == 3)))
        medium_risk = loan_portfolio[loan_portfolio["provision_stage"] == 2].sample(min(5000, sum(loan_portfolio["provision_stage"] == 2)))
        low_risk = loan_portfolio[loan_portfolio["provision_stage"] == 1].sample(min(3000, sum(loan_portfolio["provision_stage"] == 1)))
        
        shap_cases = pd.concat([high_risk, medium_risk, low_risk])
        
        # Add SHAP-specific metadata
        shap_cases["explainability_test_case"] = True
        shap_cases["expected_top_features"] = shap_cases.apply(
            lambda row: ["credit_score", "days_past_due", "ltv_ratio"] if row["provision_stage"] > 1 else ["credit_score", "loan_amount", "dti_ratio"],
            axis=1
        )
        
        return shap_cases
    
    def _generate_benchmark_operations(self) -> pd.DataFrame:
        """Generate benchmark operations for performance testing."""
        operations = [
            {"operation": "group_by_aggregation", "complexity": "high", "expected_speedup": 5.0},
            {"operation": "window_functions", "complexity": "medium", "expected_speedup": 3.0},
            {"operation": "join_operations", "complexity": "high", "expected_speedup": 8.0},
            {"operation": "filtering", "complexity": "low", "expected_speedup": 2.0},
            {"operation": "sorting", "complexity": "medium", "expected_speedup": 4.0},
        ]
        return pd.DataFrame(operations)
    
    def _generate_integration_config(self, scenario: ValidationScenario) -> Dict:
        """Generate integration-specific configuration."""
        config = {
            "scenario_name": scenario.name,
            "test_parameters": scenario.special_conditions,
            "expected_outcomes": scenario.expected_outcomes,
            "validation_rules": [],
            "monitoring_metrics": []
        }
        
        if "bigquery" in scenario.name:
            config["bigquery_config"] = {
                "project_id": "test-project",
                "dataset_id": "ifrs9_validation",
                "table_prefix": f"test_{scenario.name}",
                "partition_field": "origination_date",
                "clustering_fields": ["provision_stage", "region"]
            }
        
        if "gcs" in scenario.name:
            config["gcs_config"] = {
                "bucket_name": "ifrs9-validation-datasets",
                "object_prefix": f"scenarios/{scenario.name}/",
                "compression": "gzip",
                "versioning": True
            }
        
        return config
    
    def _save_scenario_datasets(self, scenario_dir: Path, datasets: Dict, scenario: ValidationScenario) -> List[str]:
        """Save scenario datasets in multiple formats."""
        files_generated = []
        
        # Determine formats based on scenario requirements
        if scenario.dataset_size > 500000:
            formats = ["parquet"]  # Only parquet for large datasets
        else:
            formats = ["parquet", "csv"]
        
        for dataset_name, df in datasets.items():
            if isinstance(df, pd.DataFrame):
                # Save in parquet (always)
                parquet_path = scenario_dir / f"{dataset_name}.parquet"
                df.to_parquet(parquet_path, index=False)
                files_generated.append(str(parquet_path))
                
                # Save in CSV for smaller datasets
                if "csv" in formats and len(df) < 100000:
                    csv_path = scenario_dir / f"{dataset_name}.csv"
                    df.to_csv(csv_path, index=False)
                    files_generated.append(str(csv_path))
            
            elif isinstance(df, dict):
                # Save configuration as JSON
                json_path = scenario_dir / f"{dataset_name}.json"
                with open(json_path, "w") as f:
                    json.dump(df, f, indent=2, default=str)
                files_generated.append(str(json_path))
        
        return files_generated
    
    def _validate_scenario_data(self, scenario: ValidationScenario, datasets: Dict) -> Tuple[float, bool]:
        """Validate generated scenario data quality."""
        quality_checks = []
        
        # Basic data quality checks
        for dataset_name, df in datasets.items():
            if isinstance(df, pd.DataFrame):
                # Completeness check
                completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                quality_checks.append(completeness)
                
                # Uniqueness check for loan IDs
                if "loan_id" in df.columns:
                    uniqueness = (df["loan_id"].nunique() / len(df)) * 100
                    quality_checks.append(uniqueness)
        
        # Scenario-specific validations
        if scenario.name == "production_baseline":
            # Check stage distribution
            loan_df = datasets.get("loan_portfolio")
            if loan_df is not None and "provision_stage" in loan_df.columns:
                stage_dist = loan_df["provision_stage"].value_counts(normalize=True)
                target_dist = scenario.special_conditions.get("stage_distribution", {})
                
                # Validate distribution is close to target
                if abs(stage_dist.get(1, 0) - target_dist.get("stage_1", 0.85)) < 0.05:
                    quality_checks.append(100.0)
                else:
                    quality_checks.append(60.0)
        
        # Calculate overall quality score
        quality_score = np.mean(quality_checks) if quality_checks else 0.0
        validation_passed = quality_score >= 90.0
        
        return quality_score, validation_passed
    
    def _generate_scenario_documentation(self, scenario_dir: Path, scenario: ValidationScenario, datasets: Dict):
        """Generate comprehensive documentation for the scenario."""
        doc_content = f"""# {scenario.name.replace('_', ' ').title()} Validation Dataset

## Description
{scenario.description}

## Dataset Specifications
- **Dataset Size**: {scenario.dataset_size:,} records
- **Special Conditions**: {json.dumps(scenario.special_conditions, indent=2)}
- **Expected Outcomes**: {json.dumps(scenario.expected_outcomes, indent=2)}

## Generated Datasets
"""
        
        for dataset_name, df in datasets.items():
            if isinstance(df, pd.DataFrame):
                doc_content += f"""
### {dataset_name.replace('_', ' ').title()}
- **Records**: {len(df):,}
- **Columns**: {len(df.columns)}
- **Memory Usage**: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB
- **Key Statistics**:
"""
                # Add key statistics for loan portfolio
                if dataset_name == "loan_portfolio":
                    if "provision_stage" in df.columns:
                        stage_dist = df["provision_stage"].value_counts()
                        doc_content += f"  - Stage Distribution: {dict(stage_dist)}\n"
                    
                    if "loan_amount" in df.columns:
                        doc_content += f"  - Avg Loan Amount: €{df['loan_amount'].mean():,.0f}\n"
                        doc_content += f"  - Total Portfolio Value: €{df['loan_amount'].sum():,.0f}\n"
        
        doc_content += f"""
## Validation Instructions

### Data Quality Validation
1. Check completeness: All required fields should have < 1% missing values
2. Check consistency: Business rules and IFRS9 compliance
3. Check distributions: Statistical properties match expectations

### Integration Testing
1. Load datasets into target systems
2. Verify schema compatibility
3. Test processing performance
4. Validate output consistency

### Performance Testing
1. Measure processing time against SLA requirements
2. Monitor memory and CPU usage
3. Test concurrent processing capabilities
4. Validate scalability patterns

## Generated Files
"""
        
        # List all generated files
        for file_path in scenario_dir.glob("*"):
            if file_path.is_file():
                file_size = file_path.stat().st_size / 1024 / 1024
                doc_content += f"- `{file_path.name}` ({file_size:.1f} MB)\n"
        
        # Save documentation
        with open(scenario_dir / "README.md", "w") as f:
            f.write(doc_content)
    
    def _generate_quality_report(self, generation_results: Dict):
        """Generate comprehensive data quality report."""
        report = {
            "report_generated": datetime.now().isoformat(),
            "overall_quality_score": 0.0,
            "scenario_quality_scores": {},
            "quality_metrics": {
                "completeness": {},
                "consistency": {},
                "distribution": {},
                "correlation": {},
                "ifrs9_compliance": {},
                "ml_readiness": {}
            },
            "recommendations": []
        }
        
        # Calculate quality scores per scenario
        quality_scores = []
        for scenario_name, results in generation_results["scenarios"].items():
            if results.get("status") == "completed":
                quality_score = results.get("quality_score", 0.0)
                report["scenario_quality_scores"][scenario_name] = quality_score
                quality_scores.append(quality_score)
        
        # Overall quality score
        report["overall_quality_score"] = np.mean(quality_scores) if quality_scores else 0.0
        
        # Add recommendations based on quality scores
        if report["overall_quality_score"] >= 95.0:
            report["recommendations"].append("✅ Excellent data quality - Ready for production validation")
        elif report["overall_quality_score"] >= 90.0:
            report["recommendations"].append("⚠️  Good data quality - Minor improvements recommended")
        else:
            report["recommendations"].append("❌ Data quality needs improvement before production use")
        
        # Save quality report
        with open(self.output_dir / "data_quality_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"📊 Data quality report generated - Overall score: {report['overall_quality_score']:.1f}%")
    
    def _generate_master_orchestrator_config(self, generation_results: Dict):
        """Generate master validation orchestrator configuration."""
        config = {
            "validation_orchestrator": {
                "version": "1.0.0",
                "generated": datetime.now().isoformat(),
                "total_scenarios": generation_results["summary"]["total_scenarios"],
                "total_records": generation_results["summary"]["total_records_generated"]
            },
            "scenarios": [],
            "execution_order": [],
            "dependencies": {},
            "sla_requirements": {
                "max_processing_time_minutes": 150,
                "max_memory_usage_gb": 16,
                "min_success_rate_percent": 99.0
            },
            "monitoring_config": {
                "metrics_collection": True,
                "performance_tracking": True,
                "quality_validation": True,
                "alert_thresholds": {
                    "processing_time_threshold_minutes": 120,
                    "memory_usage_threshold_gb": 12,
                    "error_rate_threshold_percent": 1.0
                }
            }
        }
        
        # Add scenario configurations
        for scenario_name, results in generation_results["scenarios"].items():
            if results.get("status") == "completed":
                scenario_config = {
                    "name": scenario_name,
                    "dataset_path": f"validation_datasets/{scenario_name}",
                    "record_count": results["records_generated"],
                    "quality_score": results["quality_score"],
                    "validation_passed": results["validation_passed"],
                    "files": results["files_generated"]
                }
                config["scenarios"].append(scenario_config)
                config["execution_order"].append(scenario_name)
        
        # Define scenario dependencies
        config["dependencies"] = {
            "production_baseline": [],
            "edge_cases_stress": ["production_baseline"],
            "multi_currency_multi_region": ["production_baseline"],
            "ml_holdout_validation": ["production_baseline"],
            "bigquery_integration": ["production_baseline", "multi_currency_multi_region"],
            "sla_compliance_150min": ["production_baseline", "edge_cases_stress"]
        }
        
        # Save orchestrator configuration
        with open(self.output_dir / "master_orchestrator_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        self.logger.info("🎛️  Master orchestrator configuration generated")


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate IFRS9 Production Validation Datasets')
    parser.add_argument('--output-dir', type=str, default='validation_datasets', 
                       help='Output directory for generated datasets')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed for reproducibility')
    parser.add_argument('--scenarios', nargs='+', 
                       help='Specific scenarios to generate (default: all)')
    parser.add_argument('--parallel', action='store_true', 
                       help='Enable parallel processing (experimental)')
    
    args = parser.parse_args()
    
    print("🏦 IFRS9 Production Validation Dataset Generator")
    print("=" * 80)
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.seed}")
    print("=" * 80)
    
    # Initialize generator
    generator = ProductionValidationGenerator(
        seed=args.seed,
        output_dir=args.output_dir
    )
    
    # Filter scenarios if specified
    if args.scenarios:
        generator.validation_scenarios = [
            scenario for scenario in generator.validation_scenarios 
            if scenario.name in args.scenarios
        ]
        print(f"Generating {len(generator.validation_scenarios)} specified scenarios")
    else:
        print(f"Generating all {len(generator.validation_scenarios)} scenarios")
    
    # Generate datasets
    try:
        results = generator.generate_all_validation_datasets()
        
        print("\n" + "=" * 80)
        print("🎉 GENERATION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Scenarios completed: {results['summary']['completed_scenarios']}")
        print(f"Scenarios failed: {results['summary']['failed_scenarios']}")
        print(f"Total records generated: {results['summary']['total_records_generated']:,}")
        print(f"Output location: {args.output_dir}")
        
        if results['summary']['failed_scenarios'] == 0:
            print("\n✅ All scenarios generated successfully - Ready for production validation!")
            return 0
        else:
            print(f"\n⚠️  {results['summary']['failed_scenarios']} scenarios failed - Check logs for details")
            return 1
            
    except Exception as e:
        print(f"\n💥 GENERATION FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())