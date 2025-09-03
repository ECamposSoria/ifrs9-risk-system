#!/usr/bin/env python3
"""Test script for Enhanced ML Models Integration.

This script validates the integration between enhanced ML models
and the existing IFRS9 system components.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def create_test_data(n_loans: int = 100) -> pd.DataFrame:
    """Create synthetic test data for ML model testing."""
    np.random.seed(42)
    
    # Create loan data similar to the existing system
    loan_types = ['mortgage', 'auto', 'personal', 'credit_card']
    employment_statuses = ['employed', 'self_employed', 'unemployed', 'retired']
    provision_stages = ['Stage 1', 'Stage 2', 'Stage 3']
    
    data = {
        'loan_id': [f'LOAN_{i:06d}' for i in range(n_loans)],
        'loan_amount': np.random.uniform(10000, 500000, n_loans),
        'interest_rate': np.random.uniform(0.03, 0.25, n_loans),
        'term_months': np.random.choice([12, 24, 36, 48, 60, 84, 120], n_loans),
        'current_balance': np.random.uniform(0, 500000, n_loans),
        'credit_score': np.random.uniform(300, 850, n_loans),
        'days_past_due': np.random.exponential(10, n_loans),
        'customer_income': np.random.uniform(30000, 200000, n_loans),
        'ltv_ratio': np.random.uniform(0.4, 1.2, n_loans),
        'monthly_payment': np.random.uniform(200, 5000, n_loans),
        'loan_type': np.random.choice(loan_types, n_loans),
        'employment_status': np.random.choice(employment_statuses, n_loans),
        'calculated_stage': np.random.choice(provision_stages, n_loans),
        'calculated_pd': np.random.uniform(0.001, 0.5, n_loans),
        'origination_date': pd.date_range('2020-01-01', periods=n_loans, freq='D')
    }
    
    return pd.DataFrame(data)


def test_simple_classifier():
    """Test the simple CreditRiskClassifier functionality."""
    print("=" * 60)
    print("TESTING SIMPLE CREDIT RISK CLASSIFIER")
    print("=" * 60)
    
    try:
        from ml_model import CreditRiskClassifier
        
        # Create test data
        test_data = create_test_data(200)
        print(f"Created test data with {len(test_data)} loans")
        
        # Initialize classifier
        classifier = CreditRiskClassifier(model_type="random_forest")
        print("✓ Simple classifier initialized")
        
        # Prepare features
        X, feature_names = classifier.prepare_features(test_data)
        print(f"✓ Features prepared: {len(feature_names)} features")
        print(f"  First 10 features: {feature_names[:10]}")
        
        # Train stage classifier
        stage_metrics = classifier.train_stage_classifier(
            X, test_data['calculated_stage'], test_size=0.2
        )
        print(f"✓ Stage classifier trained - Accuracy: {stage_metrics['accuracy']:.4f}")
        
        # Train PD model
        pd_metrics = classifier.train_pd_model(
            X, test_data['calculated_pd'], test_size=0.2
        )
        print(f"✓ PD model trained - MAE: {pd_metrics['mae']:.4f}")
        
        # Test predictions
        stage_pred, stage_prob = classifier.predict_stage(X.iloc[:10])
        pd_pred = classifier.predict_pd(X.iloc[:10])
        print(f"✓ Predictions generated for 10 loans")
        print(f"  Stage predictions: {stage_pred}")
        print(f"  PD predictions: {pd_pred[:3]}")
        
        # Test explanation
        explanation = classifier.explain_prediction(X, index=0)
        print(f"✓ Prediction explanation generated")
        print(f"  Predicted stage: {explanation['predicted_stage']}")
        
        # Save models
        classifier.save_models("./test_models/simple/")
        print("✓ Models saved successfully")
        
        # Load models
        new_classifier = CreditRiskClassifier()
        new_classifier.load_models("./test_models/simple/")
        print("✓ Models loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Simple classifier test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_enhanced_classifier():
    """Test the enhanced CreditRiskClassifier functionality."""
    print("=" * 60)
    print("TESTING ENHANCED CREDIT RISK CLASSIFIER")
    print("=" * 60)
    
    try:
        from ml_model import EnhancedCreditRiskClassifier
        
        # Create test data
        test_data = create_test_data(150)
        print(f"Created test data with {len(test_data)} loans")
        
        # Initialize enhanced classifier
        classifier = EnhancedCreditRiskClassifier(
            use_advanced_features=True,
            use_optimization=False,  # Skip optimization for faster testing
            model_selection_strategy="auto",
            fallback_to_simple=True
        )
        print("✓ Enhanced classifier initialized")
        
        # Prepare features
        X, feature_names = classifier.prepare_features(test_data)
        print(f"✓ Features prepared: {len(feature_names)} features")
        
        # Train models
        training_results = classifier.train_models(
            X=X,
            y_stage=test_data['calculated_stage'],
            y_pd=test_data['calculated_pd'],
            test_size=0.2
        )
        
        selected_info = training_results['selected']
        model_info = classifier.get_model_info()
        
        print(f"✓ Enhanced training completed")
        print(f"  Selected model: {model_info['selected_model_type']}")
        print(f"  Selection reason: {selected_info['reason']}")
        
        # Test predictions
        stage_pred, stage_prob = classifier.predict_stage(X.iloc[:5])
        pd_pred = classifier.predict_pd(X.iloc[:5])
        print(f"✓ Enhanced predictions generated for 5 loans")
        print(f"  Stage predictions: {stage_pred}")
        
        # Test explanation
        explanation = classifier.explain_prediction(X, index=0)
        print(f"✓ Enhanced prediction explanation generated")
        print(f"  Model type used: {explanation.get('model_type', 'unknown')}")
        
        # Save models
        classifier.save_models("./test_models/enhanced/")
        print("✓ Enhanced models saved successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Enhanced classifier test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_rules_engine_integration():
    """Test the rules engine integration functions."""
    print("=" * 60)
    print("TESTING RULES ENGINE INTEGRATION")
    print("=" * 60)
    
    try:
        from ml_model import (
            get_ml_predictions_for_rules_engine,
            explain_ml_prediction_for_rules_engine,
            validate_ml_model_health
        )
        
        # Create test data
        test_data = create_test_data(50)
        print(f"Created test data with {len(test_data)} loans")
        
        # Test model health validation
        health_report = validate_ml_model_health("./test_models/simple/")
        print(f"✓ Model health check completed")
        print(f"  Status: {health_report['status']}")
        print(f"  Simple available: {health_report['simple_available']}")
        
        if health_report['simple_available']:
            # Test predictions for rules engine
            predictions = get_ml_predictions_for_rules_engine(
                test_data, 
                model_path="./test_models/simple/"
            )
            
            print(f"✓ Rules engine predictions generated")
            print(f"  Model type: {predictions['prediction_metadata']['model_type']}")
            print(f"  Feature count: {predictions['prediction_metadata']['feature_count']}")
            print(f"  Stage distribution: {predictions['prediction_metadata']['stage_distribution']}")
            print(f"  Average PD: {predictions['prediction_metadata']['avg_pd']:.4f}")
            
            # Test explanation for specific loan
            explanation = explain_ml_prediction_for_rules_engine(
                test_data, 
                loan_index=0,
                model_path="./test_models/simple/"
            )
            
            print(f"✓ Individual loan explanation generated")
            print(f"  Loan ID: {explanation['loan_data']['loan_id']}")
            print(f"  Predicted stage: {explanation.get('predicted_stage', 'N/A')}")
        else:
            print("⚠ No models available for integration testing")
        
        return True
        
    except Exception as e:
        print(f"✗ Rules engine integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_factory_function():
    """Test the create_ifrs9_ml_classifier factory function."""
    print("=" * 60)
    print("TESTING FACTORY FUNCTION")
    print("=" * 60)
    
    try:
        from ml_model import create_ifrs9_ml_classifier, CreditRiskClassifier, EnhancedCreditRiskClassifier
        
        # Test simple classifier creation
        simple_classifier = create_ifrs9_ml_classifier(use_enhanced=False)
        print(f"✓ Simple classifier created: {type(simple_classifier).__name__}")
        assert isinstance(simple_classifier, CreditRiskClassifier)
        
        # Test enhanced classifier creation
        enhanced_classifier = create_ifrs9_ml_classifier(use_enhanced=True)
        print(f"✓ Enhanced classifier created: {type(enhanced_classifier).__name__}")
        assert isinstance(enhanced_classifier, EnhancedCreditRiskClassifier)
        
        # Test with parameters
        configured_classifier = create_ifrs9_ml_classifier(
            use_enhanced=True,
            model_selection_strategy="simple_only",
            use_optimization=False
        )
        print(f"✓ Configured classifier created with custom parameters")
        assert configured_classifier.model_selection_strategy == "simple_only"
        assert configured_classifier.use_optimization == False
        
        return True
        
    except Exception as e:
        print(f"✗ Factory function test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("IFRS9 Enhanced ML Models Integration Test Suite")
    print("=" * 60)
    
    # Create test directory
    os.makedirs("./test_models/simple", exist_ok=True)
    os.makedirs("./test_models/enhanced", exist_ok=True)
    
    test_results = []
    
    # Run tests
    test_results.append(("Factory Function", test_factory_function()))
    test_results.append(("Simple Classifier", test_simple_classifier()))
    test_results.append(("Enhanced Classifier", test_enhanced_classifier()))
    test_results.append(("Rules Engine Integration", test_rules_engine_integration()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "PASSED" if result else "FAILED"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed successfully!")
        return 0
    else:
        print("⚠ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)