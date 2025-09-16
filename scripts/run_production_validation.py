#!/usr/bin/env python3
"""
Quick Production Validation Script
==================================

This script provides a quick way to generate and validate IFRS9 production datasets
for immediate testing and validation. It creates a subset of the full validation
scenarios for rapid feedback and development.
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# Add src and validation to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'validation'))

from production_validation_generator import ProductionValidationGenerator, ValidationScenario
from production_readiness_validator import ProductionReadinessValidator


def setup_logging():
    """Setup logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def create_quick_validation_scenarios():
    """Create a subset of scenarios for quick validation."""
    return [
        ValidationScenario(
            name="quick_production_baseline",
            description="Quick production baseline test",
            dataset_size=10000,
            special_conditions={
                "stage_distribution": {"stage_1": 0.85, "stage_2": 0.12, "stage_3": 0.03},
                "product_mix": "balanced"
            },
            expected_outcomes={
                "data_quality_score": 95.0,
                "processing_time_limit": 5
            }
        ),
        
        ValidationScenario(
            name="quick_edge_cases",
            description="Quick edge cases validation",
            dataset_size=5000,
            special_conditions={
                "high_default_rate": 0.1,
                "borderline_dpd_cases": True
            },
            expected_outcomes={
                "stage_3_percentage": ">5%"
            }
        ),
        
        ValidationScenario(
            name="quick_multi_currency",
            description="Quick multi-currency test",
            dataset_size=7500,
            special_conditions={
                "currencies": ["EUR", "USD", "GBP"]
            },
            expected_outcomes={
                "currency_coverage": 3
            }
        ),
        
        ValidationScenario(
            name="quick_ml_validation",
            description="Quick ML model validation",
            dataset_size=5000,
            special_conditions={
                "feature_completeness": 100,
                "shap_validation_ready": True
            },
            expected_outcomes={
                "model_performance": ">80%"
            }
        )
    ]


def main():
    """Main execution function."""
    logger = setup_logging()
    
    print("🚀 IFRS9 Quick Production Validation")
    print("=" * 60)
    
    # Setup output directory
    output_dir = Path("quick_validation_results")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Phase 1: Generate quick validation datasets
        logger.info("📊 Phase 1: Generating quick validation datasets")
        
        generator = ProductionValidationGenerator(
            seed=42,
            output_dir=str(output_dir / "datasets")
        )
        
        # Replace scenarios with quick versions
        generator.validation_scenarios = create_quick_validation_scenarios()
        
        generation_results = generator.generate_all_validation_datasets()
        
        logger.info(f"Generated {generation_results['summary']['total_records_generated']:,} records")
        logger.info(f"Completed {generation_results['summary']['completed_scenarios']}/{generation_results['summary']['total_scenarios']} scenarios")
        
        # Phase 2: Validate generated datasets
        logger.info("🔍 Phase 2: Validating generated datasets")
        
        validator = ProductionReadinessValidator()
        validation_results = {}
        
        for scenario_name, scenario_result in generation_results["scenarios"].items():
            if scenario_result.get("status") == "completed":
                logger.info(f"Validating {scenario_name}")
                
                # Load dataset
                dataset_path = output_dir / "datasets" / scenario_name / "loan_portfolio.parquet"
                if dataset_path.exists():
                    loan_portfolio = pd.read_parquet(dataset_path)
                    
                    # Run validation
                    validation_result = validator.validate_comprehensive_dataset(
                        loan_portfolio=loan_portfolio
                    )
                    
                    validation_results[scenario_name] = validation_result
                    logger.info(f"✅ {scenario_name} validation score: {validator.overall_score:.1f}%")
        
        # Phase 3: Generate summary report
        logger.info("📊 Phase 3: Generating summary report")
        
        # Calculate overall metrics
        validation_scores = []
        for result in validation_results.values():
            if "validation_summary" in result:
                validation_scores.append(result["validation_summary"]["overall_score"])
        
        overall_validation_score = np.mean(validation_scores) if validation_scores else 0.0
        
        # Generate summary
        summary_report = {
            "quick_validation_summary": {
                "timestamp": datetime.now().isoformat(),
                "scenarios_processed": len(generation_results["scenarios"]),
                "scenarios_completed": generation_results["summary"]["completed_scenarios"],
                "total_records_generated": generation_results["summary"]["total_records_generated"],
                "overall_validation_score": overall_validation_score,
                "readiness_status": "READY" if overall_validation_score >= 90.0 else "NEEDS_IMPROVEMENT"
            },
            "scenario_results": {},
            "recommendations": []
        }
        
        # Add scenario details
        for scenario_name in generation_results["scenarios"].keys():
            generation_info = generation_results["scenarios"][scenario_name]
            validation_info = validation_results.get(scenario_name, {})
            
            summary_report["scenario_results"][scenario_name] = {
                "generation": {
                    "status": generation_info.get("status"),
                    "records": generation_info.get("records_generated", 0),
                    "quality_score": generation_info.get("quality_score", 0.0)
                },
                "validation": {
                    "overall_score": validation_info.get("validation_summary", {}).get("overall_score", 0.0),
                    "compliance_status": validation_info.get("validation_summary", {}).get("compliance_status", "UNKNOWN")
                }
            }
        
        # Add recommendations
        if overall_validation_score >= 95.0:
            summary_report["recommendations"] = [
                "✅ Excellent validation results - ready for full production validation",
                "🚀 Proceed with complete dataset generation and validation",
                "📊 Consider running performance benchmarks"
            ]
        elif overall_validation_score >= 85.0:
            summary_report["recommendations"] = [
                "✅ Good validation results - minor improvements recommended",
                "🔍 Review validation warnings and address them",
                "⚡ Proceed with caution to full production validation"
            ]
        else:
            summary_report["recommendations"] = [
                "❌ Validation issues detected - requires attention",
                "🔧 Review data generation logic and business rules",
                "📋 Address validation failures before proceeding"
            ]
        
        # Save summary report
        with open(output_dir / "quick_validation_summary.json", "w") as f:
            json.dump(summary_report, f, indent=2, default=str)
        
        # Print results
        print("\n" + "=" * 60)
        print("📊 QUICK VALIDATION RESULTS")
        print("=" * 60)
        print(f"Overall Validation Score: {overall_validation_score:.1f}%")
        print(f"Records Generated: {generation_results['summary']['total_records_generated']:,}")
        print(f"Scenarios Completed: {generation_results['summary']['completed_scenarios']}/{generation_results['summary']['total_scenarios']}")
        print(f"Status: {summary_report['quick_validation_summary']['readiness_status']}")
        
        print("\nScenario Breakdown:")
        for scenario_name, scenario_info in summary_report["scenario_results"].items():
            gen_score = scenario_info["generation"]["quality_score"]
            val_score = scenario_info["validation"]["overall_score"] 
            status_emoji = "✅" if val_score >= 90 else "⚠️" if val_score >= 80 else "❌"
            print(f"  {status_emoji} {scenario_name}: Gen={gen_score:.1f}%, Val={val_score:.1f}%")
        
        print("\nRecommendations:")
        for rec in summary_report["recommendations"]:
            print(f"  {rec}")
        
        print(f"\n📁 Results saved to: {output_dir}")
        print(f"📄 Summary: {output_dir / 'quick_validation_summary.json'}")
        
        if overall_validation_score >= 85.0:
            print("\n🎉 Quick validation successful! Ready for full production validation.")
            return 0
        else:
            print("\n⚠️  Issues detected. Review and fix before proceeding.")
            return 1
            
    except Exception as e:
        logger.error(f"💥 Quick validation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())