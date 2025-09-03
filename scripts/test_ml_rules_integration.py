#!/usr/bin/env python3
"""
Test script for IFRS9 Rules Engine with ML Integration

This script validates the enhanced IFRS9 Rules Engine functionality including:
- Configuration loading and validation
- ML model integration 
- Enhanced staging rules with SICR detection
- ML-enhanced PD, LGD, EAD calculations
- Comprehensive ECL calculations with discounting
- Audit trail and compliance reporting
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List

# Add src to path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from pyspark.sql import SparkSession
    from rules_engine import IFRS9RulesEngine
    print("✅ Successfully imported enhanced IFRS9RulesEngine")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def create_test_data(num_loans: int = 1000) -> pd.DataFrame:
    """Create comprehensive test loan portfolio data."""
    np.random.seed(42)
    
    # Generate loan IDs
    loan_ids = [f"LOAN_{i:06d}" for i in range(1, num_loans + 1)]
    
    # Generate loan types with realistic distribution
    loan_types = np.random.choice(
        ['MORTGAGE', 'AUTO', 'PERSONAL', 'CREDIT_CARD'], 
        size=num_loans,
        p=[0.4, 0.3, 0.2, 0.1]  # Realistic distribution
    )
    
    # Generate credit scores with different distributions by loan type
    credit_scores = []
    for loan_type in loan_types:
        if loan_type == 'MORTGAGE':
            score = np.random.normal(720, 50)  # Higher scores for mortgages
        elif loan_type == 'AUTO':
            score = np.random.normal(680, 60)
        elif loan_type == 'PERSONAL':
            score = np.random.normal(650, 70)
        else:  # CREDIT_CARD
            score = np.random.normal(620, 80)
        
        credit_scores.append(max(300, min(850, int(score))))
    
    # Generate current balances based on loan type
    current_balances = []
    for loan_type in loan_types:
        if loan_type == 'MORTGAGE':
            balance = np.random.lognormal(12, 0.5)  # $150K average
        elif loan_type == 'AUTO':
            balance = np.random.lognormal(10, 0.4)  # $25K average
        elif loan_type == 'PERSONAL':
            balance = np.random.lognormal(9, 0.6)   # $10K average
        else:  # CREDIT_CARD
            balance = np.random.lognormal(8, 0.8)   # $3K average
        
        current_balances.append(round(balance, 2))
    
    # Generate days past due with realistic distribution
    # Most loans current, some delinquent
    dpd_weights = [0.7, 0.15, 0.08, 0.04, 0.02, 0.01]  # 70% current, declining delinquency
    dpd_ranges = [0, 30, 60, 90, 120, 180]
    
    days_past_due = []
    for _ in range(num_loans):
        range_idx = np.random.choice(len(dpd_ranges), p=dpd_weights)
        if range_idx == 0:
            dpd = 0  # Current
        else:
            # Random within range
            min_dpd = dpd_ranges[range_idx - 1] if range_idx > 0 else 0
            max_dpd = dpd_ranges[range_idx]
            dpd = np.random.randint(min_dpd, max_dpd + 1)
        days_past_due.append(dpd)
    
    # Generate origination and maturity dates
    base_date = datetime.now()
    origination_dates = []
    maturity_dates = []
    
    for i, loan_type in enumerate(loan_types):
        # Origination: 1-5 years ago
        orig_date = base_date - timedelta(days=np.random.randint(365, 1825))
        origination_dates.append(orig_date.strftime('%Y-%m-%d'))
        
        # Maturity based on loan type
        if loan_type == 'MORTGAGE':
            term_years = np.random.choice([15, 20, 25, 30], p=[0.1, 0.2, 0.3, 0.4])
        elif loan_type == 'AUTO':
            term_years = np.random.choice([3, 4, 5, 6, 7], p=[0.1, 0.2, 0.4, 0.2, 0.1])
        elif loan_type == 'PERSONAL':
            term_years = np.random.choice([2, 3, 4, 5], p=[0.3, 0.4, 0.2, 0.1])
        else:  # CREDIT_CARD (revolving, use arbitrary maturity)
            term_years = 10
            
        maturity_date = orig_date + timedelta(days=term_years * 365)
        maturity_dates.append(maturity_date.strftime('%Y-%m-%d'))
    
    # Generate collateral values (for secured loans)
    collateral_values = []
    for i, loan_type in enumerate(loan_types):
        if loan_type in ['MORTGAGE', 'AUTO']:
            # Collateral worth 80-120% of loan amount
            collateral_ratio = np.random.uniform(0.8, 1.2)
            collateral_values.append(round(current_balances[i] * collateral_ratio, 2))
        else:
            collateral_values.append(0.0)  # Unsecured
    
    # Generate LTV and DTI ratios
    ltv_ratios = []
    dti_ratios = []
    
    for i, loan_type in enumerate(loan_types):
        if loan_type == 'MORTGAGE':
            ltv = np.random.beta(3, 2) * 0.95  # Concentrated around 0.75-0.85
            dti = np.random.beta(2, 3) * 0.5   # Most below 0.4
        elif loan_type == 'AUTO':
            ltv = np.random.beta(2, 1.5) * 0.9  # Higher LTV for autos
            dti = np.random.beta(2, 3) * 0.4
        else:  # Unsecured
            ltv = 0.0  # No collateral
            dti = np.random.beta(1.5, 2) * 0.6
            
        ltv_ratios.append(round(ltv, 3))
        dti_ratios.append(round(dti, 3))
    
    # Generate credit score changes (for SICR detection)
    credit_score_changes = np.random.normal(0, 25, num_loans).astype(int)
    
    # Create DataFrame
    test_data = pd.DataFrame({
        'loan_id': loan_ids,
        'loan_type': loan_types,
        'current_balance': current_balances,
        'credit_score': credit_scores,
        'days_past_due': days_past_due,
        'origination_date': origination_dates,
        'maturity_date': maturity_dates,
        'collateral_value': collateral_values,
        'ltv_ratio': ltv_ratios,
        'dti_ratio': dti_ratios,
        'credit_score_change': credit_score_changes
    })
    
    print(f"✅ Generated {num_loans} test loans with realistic distributions")
    print(f"   - Loan types: {dict(pd.Series(loan_types).value_counts())}")
    print(f"   - DPD distribution: Current: {sum(1 for d in days_past_due if d == 0)} loans")
    print(f"   - Average credit score: {np.mean(credit_scores):.0f}")
    print(f"   - Total portfolio value: ${sum(current_balances):,.2f}")
    
    return test_data

def test_configuration_loading():
    """Test YAML configuration loading and validation."""
    print("\n🔧 Testing Configuration Loading...")
    
    try:
        # Test with default configuration
        engine = IFRS9RulesEngine(config_path="config/ifrs9_rules.yaml")
        
        # Validate key configuration sections are loaded
        required_sections = ['staging_rules', 'risk_parameters', 'ecl_calculation']
        for section in required_sections:
            assert section in engine.config, f"Missing config section: {section}"
        
        # Validate ML integration settings
        ml_config = engine.config['staging_rules']['ml_integration']
        assert 'enabled' in ml_config, "Missing ML integration enabled flag"
        assert 'weight_ml_predictions' in ml_config, "Missing ML prediction weight"
        
        print("✅ Configuration loaded successfully")
        print(f"   - ML Integration Enabled: {engine.ml_enabled}")
        print(f"   - Stage 1 DPD Threshold: {engine.config['staging_rules']['stage_1_dpd_threshold']}")
        print(f"   - ML Prediction Weight: {ml_config['weight_ml_predictions']}")
        
        engine.stop()
        return True
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False

def test_ml_integration():
    """Test ML model integration capabilities."""
    print("\n🤖 Testing ML Integration...")
    
    try:
        engine = IFRS9RulesEngine(config_path="config/ifrs9_rules.yaml")
        
        if not engine.ml_enabled:
            print("⚠️  ML integration disabled (expected if models not available)")
            print("   - Testing will proceed with rule-based fallback")
        else:
            print("✅ ML integration initialized successfully")
            print(f"   - ML functions loaded: {hasattr(engine, 'get_ml_predictions')}")
            
        engine.stop()
        return True
        
    except Exception as e:
        print(f"❌ ML integration test failed: {e}")
        return False

def test_enhanced_staging_rules(engine, test_data):
    """Test enhanced staging rules with SICR detection."""
    print("\n📊 Testing Enhanced Staging Rules...")
    
    try:
        # Convert to Spark DataFrame
        spark_df = engine.spark.createDataFrame(test_data)
        
        # Apply enhanced staging rules
        staged_df = engine._apply_enhanced_staging_rules(spark_df)
        
        # Collect results for analysis
        results = staged_df.select(
            "loan_id", "loan_type", "days_past_due", "credit_score", 
            "rule_based_stage", "calculated_stage", "stage_confidence"
        ).toPandas()
        
        # Analyze staging distribution
        stage_dist = results['calculated_stage'].value_counts()
        print(f"✅ Staging rules applied successfully")
        print(f"   - Stage distribution: {dict(stage_dist)}")
        
        # Validate staging logic
        stage_3_loans = results[results['calculated_stage'] == 'STAGE_3']
        stage_2_loans = results[results['calculated_stage'] == 'STAGE_2']
        stage_1_loans = results[results['calculated_stage'] == 'STAGE_1']
        
        print(f"   - Stage 3 avg DPD: {stage_3_loans['days_past_due'].mean():.1f}")
        print(f"   - Stage 2 avg DPD: {stage_2_loans['days_past_due'].mean():.1f}")
        print(f"   - Stage 1 avg DPD: {stage_1_loans['days_past_due'].mean():.1f}")
        
        # Check confidence scores
        avg_confidence = results['stage_confidence'].mean()
        print(f"   - Average confidence: {avg_confidence:.3f}")
        
        return staged_df, True
        
    except Exception as e:
        print(f"❌ Enhanced staging rules test failed: {e}")
        return None, False

def test_risk_parameters(engine, staged_df):
    """Test enhanced risk parameter calculations."""
    print("\n📈 Testing Enhanced Risk Parameters...")
    
    try:
        # Calculate enhanced risk parameters
        risk_df = engine._calculate_enhanced_risk_parameters(staged_df)
        
        # Collect sample for analysis
        sample = risk_df.select(
            "loan_id", "loan_type", "calculated_stage",
            "calculated_pd", "calculated_lgd", "calculated_ead"
        ).limit(100).toPandas()
        
        print("✅ Risk parameters calculated successfully")
        print(f"   - PD range: {sample['calculated_pd'].min():.4f} - {sample['calculated_pd'].max():.4f}")
        print(f"   - LGD range: {sample['calculated_lgd'].min():.3f} - {sample['calculated_lgd'].max():.3f}")
        print(f"   - Average PD by stage:")
        
        for stage in ['STAGE_1', 'STAGE_2', 'STAGE_3']:
            stage_data = sample[sample['calculated_stage'] == stage]
            if len(stage_data) > 0:
                avg_pd = stage_data['calculated_pd'].mean()
                print(f"     - {stage}: {avg_pd:.4f}")
        
        return risk_df, True
        
    except Exception as e:
        print(f"❌ Risk parameters test failed: {e}")
        return None, False

def test_ecl_calculations(engine, risk_df):
    """Test enhanced ECL calculations with discounting."""
    print("\n💰 Testing Enhanced ECL Calculations...")
    
    try:
        # Calculate enhanced ECL
        ecl_df = engine._calculate_enhanced_ecl(risk_df)
        
        # Collect sample for analysis
        sample = ecl_df.select(
            "loan_id", "calculated_stage", "current_balance",
            "calculated_ecl", "provision_rate", "discount_factor",
            "forward_looking_adjustment"
        ).limit(100).toPandas()
        
        print("✅ ECL calculations completed successfully")
        print(f"   - ECL range: ${sample['calculated_ecl'].min():.2f} - ${sample['calculated_ecl'].max():.2f}")
        print(f"   - Provision rate range: {sample['provision_rate'].min():.4f} - {sample['provision_rate'].max():.4f}")
        print(f"   - Average discount factor: {sample['discount_factor'].mean():.4f}")
        
        # ECL by stage analysis
        print(f"   - Average provision rates by stage:")
        for stage in ['STAGE_1', 'STAGE_2', 'STAGE_3']:
            stage_data = sample[sample['calculated_stage'] == stage]
            if len(stage_data) > 0:
                avg_provision = stage_data['provision_rate'].mean()
                print(f"     - {stage}: {avg_provision:.4f}")
        
        return ecl_df, True
        
    except Exception as e:
        print(f"❌ ECL calculations test failed: {e}")
        return None, False

def test_comprehensive_processing(engine, test_data):
    """Test end-to-end portfolio processing."""
    print("\n🔄 Testing Comprehensive Portfolio Processing...")
    
    try:
        # Convert to Spark DataFrame
        spark_df = engine.spark.createDataFrame(test_data)
        
        # Process entire portfolio
        processed_df = engine.process_portfolio(spark_df)
        
        # Generate comprehensive summary report
        summary = engine.generate_comprehensive_summary_report(processed_df)
        
        print("✅ Comprehensive processing completed successfully")
        print(f"   - Total loans processed: {summary['portfolio_metrics']['total_loans']}")
        print(f"   - Total exposure: ${summary['portfolio_metrics']['total_exposure']:,.2f}")
        print(f"   - Total ECL: ${summary['portfolio_metrics']['total_ecl']:,.2f}")
        print(f"   - Coverage ratio: {summary['portfolio_metrics']['coverage_ratio']:.4f}")
        
        # Stage distribution
        print(f"   - Stage distribution:")
        for stage, metrics in summary['stage_distribution'].items():
            print(f"     - {stage}: {metrics['count']} loans (${metrics['exposure']:,.0f})")
        
        # Validation results
        validation_results = summary['validation_results']
        failed_checks = [v for v in validation_results if not v['passed']]
        
        print(f"   - Validation checks: {len(validation_results) - len(failed_checks)}/{len(validation_results)} passed")
        if failed_checks:
            print("   - Failed checks:")
            for check in failed_checks:
                print(f"     - {check['check']}: {check['message']} ({check['severity']})")
        
        # Audit trail
        print(f"   - Audit events: {len(engine.audit_trail)}")
        
        return processed_df, summary, True
        
    except Exception as e:
        print(f"❌ Comprehensive processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, False

def test_export_functionality(engine, processed_df):
    """Test result export functionality."""
    print("\n📤 Testing Export Functionality...")
    
    try:
        # Create test output directory
        test_output_path = "/tmp/ifrs9_test_output"
        os.makedirs(test_output_path, exist_ok=True)
        
        # Export results
        export_paths = engine.export_results(processed_df, test_output_path)
        
        print("✅ Export functionality completed successfully")
        print(f"   - Export paths:")
        for format_type, path in export_paths.items():
            file_exists = os.path.exists(path)
            print(f"     - {format_type}: {path} {'✅' if file_exists else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Export functionality test failed: {e}")
        return False

def main():
    """Main test execution function."""
    print("🚀 IFRS9 Rules Engine ML Integration Test Suite")
    print("=" * 60)
    
    # Test results tracking
    test_results = {}
    
    # Test 1: Configuration Loading
    test_results['config_loading'] = test_configuration_loading()
    
    # Test 2: ML Integration
    test_results['ml_integration'] = test_ml_integration()
    
    if not test_results['config_loading']:
        print("\n❌ Cannot proceed without configuration loading")
        return
    
    # Initialize engine for main tests
    try:
        engine = IFRS9RulesEngine(config_path="config/ifrs9_rules.yaml")
        print(f"\n✅ Initialized IFRS9RulesEngine")
        print(f"   - Spark session: {engine.spark.sparkContext.appName}")
        print(f"   - ML integration: {'Enabled' if engine.ml_enabled else 'Disabled'}")
    except Exception as e:
        print(f"\n❌ Failed to initialize IFRS9RulesEngine: {e}")
        return
    
    try:
        # Generate test data
        print("\n📊 Generating Test Data...")
        test_data = create_test_data(num_loans=500)  # Smaller dataset for testing
        
        # Test 3: Enhanced Staging Rules
        staged_df, test_results['staging'] = test_enhanced_staging_rules(engine, test_data)
        
        if staged_df is not None and test_results['staging']:
            # Test 4: Risk Parameters
            risk_df, test_results['risk_params'] = test_risk_parameters(engine, staged_df)
            
            if risk_df is not None and test_results['risk_params']:
                # Test 5: ECL Calculations
                ecl_df, test_results['ecl'] = test_ecl_calculations(engine, risk_df)
        
        # Test 6: Comprehensive Processing
        processed_df, summary, test_results['comprehensive'] = test_comprehensive_processing(engine, test_data)
        
        if processed_df is not None and test_results['comprehensive']:
            # Test 7: Export Functionality
            test_results['export'] = test_export_functionality(engine, processed_df)
        
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        engine.stop()
        print("\n🔧 Spark session stopped")
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! ML-enhanced IFRS9 Rules Engine is ready.")
    else:
        print("⚠️  Some tests failed. Review the output above for details.")
        
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)