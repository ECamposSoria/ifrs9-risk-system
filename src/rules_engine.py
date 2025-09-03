"""IFRS9 Rules Engine using PySpark.

This module implements the core IFRS9 rules for credit risk classification,
expected credit loss calculation, and provision staging.
"""

from typing import Dict, List, Optional, Tuple

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)


class IFRS9RulesEngine:
    """Enhanced PySpark-based IFRS9 rules processing engine with ML integration.
    
    This class implements:
    - ML-enhanced stage classification (Stage 1, 2, 3)
    - ML-predicted Probability of Default (PD) calculation
    - Advanced Loss Given Default (LGD) calculation
    - Exposure at Default (EAD) calculation
    - Enhanced Expected Credit Loss (ECL) calculation with discounting
    - Configuration-driven parameters
    - Comprehensive audit trail
    """
    
    def __init__(self, config_path: str = "config/ifrs9_rules.yaml", spark: Optional[SparkSession] = None):
        """Initialize the enhanced IFRS9 Rules Engine.
        
        Args:
            config_path: Path to YAML configuration file
            spark: SparkSession instance. If None, creates a new session.
        """
        import yaml
        import os
        from datetime import datetime
        
        # Initialize Spark session
        if spark is None:
            self.spark = SparkSession.builder \
                .appName("IFRS9RulesEngine") \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .config("spark.sql.adaptive.skewJoin.enabled", "true") \
                .getOrCreate()
        else:
            self.spark = spark
            
        # Load configuration
        self.config_path = config_path
        self.config = self._load_configuration(config_path)
        
        # Initialize ML integration
        self._init_ml_integration()
        
        # Initialize audit trail
        self.audit_trail = []
        self.processing_timestamp = datetime.now().isoformat()
        
    def _load_configuration(self, config_path: str) -> Dict:
        """Load IFRS9 configuration from YAML file."""
        import yaml
        import os
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Validate required sections
        required_sections = ['staging_rules', 'risk_parameters', 'ecl_calculation']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
                
        return config
        
    def _init_ml_integration(self):
        """Initialize ML integration components."""
        self.ml_enabled = self.config['staging_rules']['ml_integration']['enabled']
        
        if self.ml_enabled:
            try:
                # Import ML functions - these were created by the ifrs9-ml-models agent
                from src.ml_model import (
                    get_ml_predictions_for_rules_engine,
                    explain_ml_prediction_for_rules_engine,
                    validate_ml_model_health
                )
                
                self.get_ml_predictions = get_ml_predictions_for_rules_engine
                self.explain_ml_prediction = explain_ml_prediction_for_rules_engine
                self.validate_ml_health = validate_ml_model_health
                
                # Validate ML model health
                model_path = self.config.get('model_settings', {}).get('model_path', '/opt/airflow/models/')
                health_report = self.validate_ml_health(model_path)
                
                if health_report.get('status') != 'healthy':
                    print(f"Warning: ML model health check failed: {health_report}")
                    if not self.config['staging_rules']['ml_integration']['fallback_to_rules']:
                        raise RuntimeError("ML models unhealthy and fallback disabled")
                        
            except ImportError as e:
                print(f"ML integration disabled due to import error: {e}")
                self.ml_enabled = False
            except Exception as e:
                print(f"ML integration initialization failed: {e}")
                if not self.config['staging_rules']['ml_integration']['fallback_to_rules']:
                    raise
                self.ml_enabled = False

    def process_portfolio(self, loan_df: DataFrame) -> DataFrame:
        """Process loan portfolio through enhanced IFRS9 rules with ML integration.
        
        Args:
            loan_df: DataFrame with loan portfolio data
            
        Returns:
            DataFrame with enhanced IFRS9 calculations
        """
        self._log_audit_event("process_portfolio_start", {"loan_count": loan_df.count()})
        
        # Validate input data
        validated_df = self._validate_input_data(loan_df)
        
        # Apply enhanced staging rules with ML integration
        staged_df = self._apply_enhanced_staging_rules(validated_df)
        
        # Calculate enhanced risk parameters with ML predictions
        risk_df = self._calculate_enhanced_risk_parameters(staged_df)
        
        # Calculate enhanced ECL with discounting and forward-looking adjustments
        ecl_df = self._calculate_enhanced_ecl(risk_df)
        
        # Add comprehensive aggregations and metrics
        final_df = self._add_comprehensive_aggregations(ecl_df)
        
        # Add audit trail information
        final_df = self._add_audit_information(final_df)
        
        self._log_audit_event("process_portfolio_complete", {"processed_loan_count": final_df.count()})
        
        return final_df
    
    def _validate_input_data(self, df: DataFrame) -> DataFrame:
        """Validate input data according to IFRS9 requirements."""
        self._log_audit_event("input_validation_start", {})
        
        # Check required columns
        required_cols = ['loan_id', 'current_balance', 'days_past_due', 'credit_score', 
                        'loan_type', 'origination_date', 'maturity_date']
        
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Data quality checks
        null_counts = {}
        for col in required_cols:
            null_count = df.filter(F.col(col).isNull()).count()
            null_counts[col] = null_count
            
        self._log_audit_event("input_validation_complete", {"null_counts": null_counts})
        
        return df
    
    def _apply_enhanced_staging_rules(self, df: DataFrame) -> DataFrame:
        """Apply enhanced IFRS9 staging rules with ML integration and SICR detection.
        
        Stage 1: Performing loans (low credit risk)
        Stage 2: Underperforming loans (significant increase in credit risk) 
        Stage 3: Credit-impaired loans (default)
        """
        self._log_audit_event("staging_rules_start", {"ml_enabled": self.ml_enabled})
        
        # Get configuration parameters
        staging_config = self.config['staging_rules']
        sicr_config = staging_config['sicr_detection']
        ml_config = staging_config['ml_integration']
        
        # Apply rule-based staging first
        df = self._apply_rule_based_staging(df, staging_config)
        
        # Apply SICR detection
        if sicr_config['enabled']:
            df = self._detect_significant_increase_credit_risk(df, sicr_config)
        
        # Integrate ML predictions if enabled
        if self.ml_enabled:
            df = self._integrate_ml_staging_predictions(df, ml_config)
        else:
            # Use rule-based staging only
            df = df.withColumn("calculated_stage", F.col("rule_based_stage"))
            df = df.withColumn("stage_confidence", F.lit(0.8))  # Default confidence
            
        self._log_audit_event("staging_rules_complete", {
            "stage_distribution": self._get_stage_distribution(df)
        })
        
        return df
    
    def _apply_rule_based_staging(self, df: DataFrame, staging_config: Dict) -> DataFrame:
        """Apply traditional rule-based staging logic."""
        stage_1_threshold = staging_config['stage_1_dpd_threshold']
        stage_2_threshold = staging_config['stage_2_dpd_threshold']
        
        return df.withColumn(
            "rule_based_stage",
            F.when(F.col("days_past_due") >= stage_2_threshold, "STAGE_3")
            .when(F.col("days_past_due") >= stage_1_threshold, "STAGE_2")
            .when(
                (F.col("credit_score") < 500) | 
                (F.col("ltv_ratio") > staging_config['sicr_detection']['ltv_ratio_threshold']),
                "STAGE_2"
            )
            .otherwise("STAGE_1")
        )
    
    def _detect_significant_increase_credit_risk(self, df: DataFrame, sicr_config: Dict) -> DataFrame:
        """Detect Significant Increase in Credit Risk (SICR) indicators."""
        
        # Add SICR flags
        df = df.withColumn("sicr_credit_score_decline", 
                          F.when(F.col("credit_score_change") < -sicr_config['credit_score_decline_threshold'], True)
                          .otherwise(False))
        
        df = df.withColumn("sicr_high_ltv", 
                          F.when(F.col("ltv_ratio") > sicr_config['ltv_ratio_threshold'], True)
                          .otherwise(False))
                          
        df = df.withColumn("sicr_high_dti",
                          F.when(F.col("dti_ratio") > sicr_config['dti_ratio_threshold'], True)
                          .otherwise(False))
        
        # Upgrade to Stage 2 if SICR detected and currently Stage 1
        df = df.withColumn(
            "rule_based_stage",
            F.when(
                (F.col("rule_based_stage") == "STAGE_1") & 
                (F.col("sicr_credit_score_decline") | F.col("sicr_high_ltv") | F.col("sicr_high_dti")),
                "STAGE_2"
            ).otherwise(F.col("rule_based_stage"))
        )
        
        return df
    
    def _integrate_ml_staging_predictions(self, df: DataFrame, ml_config: Dict) -> DataFrame:
        """Integrate ML staging predictions with rule-based decisions."""
        try:
            # Convert Spark DataFrame to pandas for ML prediction
            pandas_df = df.toPandas()
            
            # Get ML predictions
            model_path = self.config.get('model_settings', {}).get('model_path', '/opt/airflow/models/')
            ml_results = self.get_ml_predictions(pandas_df, model_path)
            
            if ml_results.get('error'):
                raise RuntimeError(f"ML prediction error: {ml_results['error']}")
            
            # Add ML predictions to the DataFrame
            pandas_df['ml_stage_prediction'] = ml_results['stage_predictions']
            pandas_df['ml_stage_confidence'] = ml_results['stage_probabilities'].max(axis=1)
            pandas_df['ml_pd_prediction'] = ml_results['pd_predictions']
            
            # Convert back to Spark DataFrame
            df_with_ml = self.spark.createDataFrame(pandas_df)
            
            # Hybrid decision logic: combine ML and rule-based predictions
            confidence_threshold = ml_config['confidence_threshold']
            ml_weight = ml_config['weight_ml_predictions']
            rule_weight = ml_config['weight_rule_based']
            
            # Use ML prediction if confidence is high, otherwise blend with rules
            df_with_ml = df_with_ml.withColumn(
                "calculated_stage",
                F.when(
                    F.col("ml_stage_confidence") >= confidence_threshold,
                    F.col("ml_stage_prediction")
                ).when(
                    # Agreement between ML and rules
                    F.col("ml_stage_prediction") == F.col("rule_based_stage"),
                    F.col("ml_stage_prediction")
                ).otherwise(
                    # Disagreement: take the more conservative (higher risk) stage
                    F.when(
                        (F.col("ml_stage_prediction") == "STAGE_3") | (F.col("rule_based_stage") == "STAGE_3"),
                        "STAGE_3"
                    ).when(
                        (F.col("ml_stage_prediction") == "STAGE_2") | (F.col("rule_based_stage") == "STAGE_2"),
                        "STAGE_2"
                    ).otherwise("STAGE_1")
                )
            )
            
            # Set confidence based on agreement
            df_with_ml = df_with_ml.withColumn(
                "stage_confidence",
                F.when(
                    F.col("ml_stage_prediction") == F.col("rule_based_stage"),
                    F.greatest(F.col("ml_stage_confidence"), F.lit(0.8))
                ).otherwise(
                    F.least(F.col("ml_stage_confidence"), F.lit(0.6))
                )
            )
            
            self._log_audit_event("ml_integration_success", {
                "model_info": ml_results.get('model_info', {}),
                "prediction_metadata": ml_results.get('prediction_metadata', {})
            })
            
            return df_with_ml
            
        except Exception as e:
            self._log_audit_event("ml_integration_failure", {"error": str(e)})
            
            if ml_config['fallback_to_rules']:
                # Fallback to rule-based staging
                df = df.withColumn("calculated_stage", F.col("rule_based_stage"))
                df = df.withColumn("stage_confidence", F.lit(0.7))  # Lower confidence for fallback
                df = df.withColumn("ml_stage_prediction", F.lit(None))
                df = df.withColumn("ml_stage_confidence", F.lit(None))
                df = df.withColumn("ml_pd_prediction", F.lit(None))
                return df
            else:
                raise
    
    def _calculate_enhanced_risk_parameters(self, df: DataFrame) -> DataFrame:
        """Calculate enhanced IFRS9 risk parameters with ML integration."""
        self._log_audit_event("risk_parameters_start", {})
        
        risk_config = self.config['risk_parameters']
        
        # Enhanced PD calculation with ML predictions
        df = self._calculate_enhanced_pd(df, risk_config['pd_calculation'])
        
        # Enhanced LGD calculation
        df = self._calculate_enhanced_lgd(df, risk_config['lgd_calculation'])
        
        # Enhanced EAD calculation
        df = self._calculate_enhanced_ead(df, risk_config['ead_calculation'])
        
        self._log_audit_event("risk_parameters_complete", {})
        return df
    
    def _calculate_enhanced_pd(self, df: DataFrame, pd_config: Dict) -> DataFrame:
        """Calculate enhanced Probability of Default using ML predictions and rules."""
        
        if pd_config['use_ml_predictions'] and self.ml_enabled:
            # Use ML PD predictions if available and valid
            df = df.withColumn(
                "calculated_pd",
                F.when(
                    F.col("ml_pd_prediction").isNotNull() & 
                    (F.col("ml_pd_prediction") >= 0) & 
                    (F.col("ml_pd_prediction") <= 1),
                    # Apply stage-specific bounds to ML predictions
                    F.when(
                        F.col("calculated_stage") == "STAGE_3",
                        F.lit(pd_config['stage3_pd'])
                    ).when(
                        F.col("calculated_stage") == "STAGE_2",
                        F.greatest(
                            F.lit(pd_config['minimum_pd_stage2']),
                            F.least(F.lit(pd_config['maximum_pd_stage2']), F.col("ml_pd_prediction"))
                        )
                    ).otherwise(
                        F.greatest(
                            F.lit(pd_config['minimum_pd_stage1']),
                            F.least(F.lit(pd_config['maximum_pd_stage1']), F.col("ml_pd_prediction"))
                        )
                    )
                ).otherwise(
                    # Fallback to rule-based PD calculation
                    self._calculate_rule_based_pd(df, pd_config)
                )
            )
        else:
            # Use rule-based PD calculation
            df = df.withColumn("calculated_pd", self._calculate_rule_based_pd(df, pd_config))
        
        return df
        
    def _calculate_rule_based_pd(self, df: DataFrame, pd_config: Dict) -> F.Column:
        """Calculate rule-based PD using traditional credit scoring methods."""
        return F.when(F.col("calculated_stage") == "STAGE_3", F.lit(pd_config['stage3_pd'])) \
                .when(F.col("calculated_stage") == "STAGE_2",
                      F.least(F.lit(pd_config['maximum_pd_stage2']), 
                             F.greatest(F.lit(pd_config['minimum_pd_stage2']),
                                       (900 - F.col("credit_score")) / 900))) \
                .otherwise(
                    F.least(F.lit(pd_config['maximum_pd_stage1']),
                           F.greatest(F.lit(pd_config['minimum_pd_stage1']),
                                     (850 - F.col("credit_score")) / 1700)))
    
    def _calculate_enhanced_lgd(self, df: DataFrame, lgd_config: Dict) -> DataFrame:
        """Calculate enhanced Loss Given Default with loan-type specific parameters."""
        
        # Get collateral recovery rate
        df = df.withColumn(
            "recovery_rate",
            F.when(F.col("collateral_value") > 0,
                  F.least(F.lit(0.95),
                         F.col("collateral_value") / F.col("current_balance")))
            .otherwise(0.0)
        )
        
        # Enhanced LGD calculation based on loan type
        secured_config = lgd_config['secured_loans']
        unsecured_config = lgd_config['unsecured_loans']
        
        df = df.withColumn(
            "calculated_lgd",
            F.when(F.col("loan_type") == "MORTGAGE",
                  F.greatest(F.lit(secured_config['mortgage']['minimum_lgd']),
                            1 - F.col("recovery_rate") * secured_config['mortgage']['recovery_rate_factor']))
            .when(F.col("loan_type") == "AUTO",
                  F.greatest(F.lit(secured_config['auto']['minimum_lgd']),
                            1 - F.col("recovery_rate") * secured_config['auto']['recovery_rate_factor']))
            .when(F.col("loan_type") == "CREDIT_CARD",
                  F.greatest(F.lit(unsecured_config['credit_card']['minimum_lgd']),
                            1 - F.col("recovery_rate") * unsecured_config['credit_card']['recovery_rate_factor']))
            .when(F.col("loan_type") == "PERSONAL",
                  F.greatest(F.lit(unsecured_config['personal']['minimum_lgd']),
                            1 - F.col("recovery_rate") * unsecured_config['personal']['recovery_rate_factor']))
            .otherwise(F.greatest(F.lit(unsecured_config['other_unsecured']['minimum_lgd']),
                                 1 - F.col("recovery_rate") * unsecured_config['other_unsecured']['recovery_rate_factor']))
        )
        
        return df
    
    def _calculate_enhanced_ead(self, df: DataFrame, ead_config: Dict) -> DataFrame:
        """Calculate enhanced Exposure at Default with Credit Conversion Factors."""
        
        df = df.withColumn(
            "calculated_ead",
            F.when(F.col("loan_type") == "CREDIT_CARD",
                  F.col("current_balance") * ead_config['credit_card_ccf'])
            .when(F.col("loan_type") == "LINE_OF_CREDIT", 
                  F.col("current_balance") * ead_config['line_of_credit_ccf'])
            .otherwise(F.col("current_balance") * ead_config['term_loan_ccf'])
        )
        
        return df
    
    def _calculate_enhanced_ecl(self, df: DataFrame) -> DataFrame:
        """Calculate enhanced Expected Credit Loss with discounting and forward-looking adjustments."""
        self._log_audit_event("ecl_calculation_start", {})
        
        ecl_config = self.config['ecl_calculation']
        
        # Calculate base ECL: PD × LGD × EAD
        df = df.withColumn(
            "base_ecl_12_month",
            F.col("calculated_pd") * F.col("calculated_lgd") * F.col("calculated_ead")
        )
        
        # Calculate remaining term for lifetime ECL
        df = df.withColumn(
            "remaining_term_years",
            F.greatest(
                F.lit(1.0/12),  # Minimum 1 month
                F.least(
                    F.lit(ecl_config['time_horizons']['stage2_lifetime_cap_years']),
                    F.datediff(F.col("maturity_date"), F.current_date()) / 365.25
                )
            )
        )
        
        # Apply discounting if enabled
        if ecl_config['discounting']['enabled']:
            df = self._apply_discounting(df, ecl_config['discounting'])
        else:
            df = df.withColumn("discount_factor", F.lit(1.0))
        
        # Apply forward-looking adjustments
        if ecl_config['forward_looking']['enabled']:
            df = self._apply_forward_looking_adjustments(df, ecl_config['forward_looking'])
        else:
            df = df.withColumn("forward_looking_adjustment", F.lit(1.0))
        
        # Calculate lifetime ECL for Stage 2 and 3
        df = df.withColumn(
            "ecl_lifetime",
            F.col("base_ecl_12_month") * 
            F.col("remaining_term_years") * 
            F.col("discount_factor") * 
            F.col("forward_looking_adjustment")
        )
        
        # Calculate discounted 12-month ECL for Stage 1
        df = df.withColumn(
            "ecl_12_month",
            F.col("base_ecl_12_month") * 
            F.col("discount_factor") * 
            F.col("forward_looking_adjustment")
        )
        
        # Final ECL based on stage
        df = df.withColumn(
            "calculated_ecl",
            F.when(F.col("calculated_stage") == "STAGE_1", F.col("ecl_12_month"))
            .otherwise(F.col("ecl_lifetime"))
        )
        
        self._log_audit_event("ecl_calculation_complete", {})
        return df
    
    def _apply_discounting(self, df: DataFrame, discount_config: Dict) -> DataFrame:
        """Apply discounting to ECL calculations."""
        risk_free_rate = discount_config['risk_free_rate']
        cycle_adjustment = discount_config['economic_cycle_adjustment']
        total_rate = risk_free_rate + cycle_adjustment
        
        df = df.withColumn(
            "discount_factor",
            F.pow(F.lit(1 + total_rate), -F.col("remaining_term_years"))
        )
        
        return df
    
    def _apply_forward_looking_adjustments(self, df: DataFrame, fl_config: Dict) -> DataFrame:
        """Apply forward-looking macroeconomic adjustments."""
        scenarios = fl_config['economic_scenarios']
        macro_factors = fl_config['macroeconomic_factors']
        
        # Simplified forward-looking adjustment based on loan characteristics
        # In production, this would use actual macroeconomic forecasts
        df = df.withColumn(
            "forward_looking_adjustment",
            F.lit(1.0) + 
            F.when(F.col("loan_type").isin(["MORTGAGE", "AUTO"]), 
                   F.lit(macro_factors['gdp_growth_impact'])) +
            F.when(F.col("calculated_stage").isin(["STAGE_2", "STAGE_3"]),
                   F.lit(macro_factors['unemployment_impact'])) +
            F.lit(macro_factors['interest_rate_impact'])
        )
        
        return df
    
    def _add_comprehensive_aggregations(self, df: DataFrame) -> DataFrame:
        """Add comprehensive aggregations and metrics for monitoring."""
        
        # Add provision rate
        df = df.withColumn(
            "provision_rate",
            F.col("calculated_ecl") / F.col("current_balance")
        )
        
        # Enhanced risk rating with ML confidence consideration
        df = df.withColumn(
            "risk_rating",
            F.when(F.col("calculated_pd") < 0.005, "EXCELLENT")
            .when(F.col("calculated_pd") < 0.01, "GOOD") 
            .when(F.col("calculated_pd") < 0.05, "FAIR")
            .when(F.col("calculated_pd") < 0.20, "POOR")
            .otherwise("DEFAULT")
        )
        
        # Enhanced monitoring flags
        df = df.withColumn(
            "watch_list_flag",
            F.when(
                (F.col("calculated_stage") == "STAGE_2") |
                (F.col("days_past_due") > 15) |
                (F.col("credit_score") < 550) |
                (F.col("provision_rate") > 0.1) |
                (F.col("stage_confidence") < 0.5),
                True
            ).otherwise(False)
        )
        
        # Add model performance indicators
        df = df.withColumn(
            "model_reliability",
            F.when(F.col("stage_confidence") >= 0.8, "HIGH")
            .when(F.col("stage_confidence") >= 0.6, "MEDIUM")
            .otherwise("LOW")
        )
        
        return df
    
    def _add_audit_information(self, df: DataFrame) -> DataFrame:
        """Add audit trail information to the DataFrame."""
        from datetime import datetime
        
        df = df.withColumn("processing_timestamp", F.lit(self.processing_timestamp))
        df = df.withColumn("config_version", F.lit("1.0"))
        df = df.withColumn("ml_enabled", F.lit(self.ml_enabled))
        
        return df
    
    def _get_stage_distribution(self, df: DataFrame) -> Dict:
        """Get stage distribution for audit logging."""
        stage_counts = df.groupBy("calculated_stage").count().collect()
        return {row["calculated_stage"]: row["count"] for row in stage_counts}
    
    def _log_audit_event(self, event_type: str, metadata: Dict):
        """Log audit event with metadata."""
        from datetime import datetime
        
        audit_event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "metadata": metadata
        }
        self.audit_trail.append(audit_event)
        
        # Also log to stdout for Airflow/monitoring systems
        print(f"IFRS9_AUDIT: {event_type} - {metadata}")

    def generate_comprehensive_summary_report(self, df: DataFrame) -> Dict:
        """Generate comprehensive summary statistics and compliance report."""
        self._log_audit_event("summary_report_start", {})
        
        # Cache for performance
        df.cache()
        
        try:
            # Portfolio summary
            total_loans = df.count()
            total_exposure = df.agg(F.sum("current_balance")).collect()[0][0] or 0
            total_ecl = df.agg(F.sum("calculated_ecl")).collect()[0][0] or 0
            
            # Stage distribution with enhanced metrics
            stage_metrics = df.groupBy("calculated_stage").agg(
                F.count("loan_id").alias("count"),
                F.sum("current_balance").alias("exposure"),
                F.sum("calculated_ecl").alias("ecl"),
                F.avg("calculated_pd").alias("avg_pd"),
                F.avg("calculated_lgd").alias("avg_lgd"),
                F.avg("stage_confidence").alias("avg_confidence")
            ).collect()
            
            # Risk rating distribution
            risk_metrics = df.groupBy("risk_rating").agg(
                F.count("loan_id").alias("count"),
                F.avg("calculated_pd").alias("avg_pd"),
                F.sum("calculated_ecl").alias("ecl")
            ).collect()
            
            # ML performance metrics if ML is enabled
            ml_metrics = {}
            if self.ml_enabled:
                ml_metrics = {
                    "ml_predictions_used": df.filter(F.col("ml_stage_prediction").isNotNull()).count(),
                    "high_confidence_predictions": df.filter(F.col("stage_confidence") >= 0.8).count(),
                    "ml_rule_agreement": df.filter(F.col("ml_stage_prediction") == F.col("rule_based_stage")).count(),
                    "avg_ml_confidence": df.agg(F.avg("stage_confidence")).collect()[0][0] or 0
                }
            
            # Validation results
            validation_results = self.enhanced_validate_calculations(df)
            
            summary = {
                "report_metadata": {
                    "generation_timestamp": self.processing_timestamp,
                    "config_version": "1.0",
                    "ml_integration_enabled": self.ml_enabled,
                    "total_audit_events": len(self.audit_trail)
                },
                "portfolio_metrics": {
                    "total_loans": total_loans,
                    "total_exposure": float(total_exposure),
                    "total_ecl": float(total_ecl),
                    "coverage_ratio": float(total_ecl / total_exposure) if total_exposure > 0 else 0,
                    "avg_provision_rate": float(total_ecl / total_exposure) if total_exposure > 0 else 0
                },
                "stage_distribution": {
                    row["calculated_stage"]: {
                        "count": row["count"],
                        "exposure": float(row["exposure"]) if row["exposure"] else 0,
                        "ecl": float(row["ecl"]) if row["ecl"] else 0,
                        "avg_pd": float(row["avg_pd"]) if row["avg_pd"] else 0,
                        "avg_lgd": float(row["avg_lgd"]) if row["avg_lgd"] else 0,
                        "avg_confidence": float(row["avg_confidence"]) if row["avg_confidence"] else 0
                    }
                    for row in stage_metrics
                },
                "risk_distribution": {
                    row["risk_rating"]: {
                        "count": row["count"],
                        "avg_pd": float(row["avg_pd"]) if row["avg_pd"] else 0,
                        "ecl": float(row["ecl"]) if row["ecl"] else 0,
                    }
                    for row in risk_metrics
                },
                "ml_performance_metrics": ml_metrics,
                "validation_results": validation_results,
                "audit_trail": self.audit_trail
            }
            
            self._log_audit_event("summary_report_complete", {
                "total_loans": total_loans,
                "total_ecl": float(total_ecl)
            })
            
            return summary
            
        finally:
            # Unpersist cache
            df.unpersist()
    
    def enhanced_validate_calculations(self, df: DataFrame) -> List[Dict]:
        """Enhanced validation of IFRS9 calculations for regulatory compliance."""
        validations = []
        validation_config = self.config['validation']
        
        # Check for null values in critical columns
        critical_cols = ["calculated_pd", "calculated_lgd", "calculated_ead", "calculated_ecl", "calculated_stage"]
        for col in critical_cols:
            null_count = df.filter(F.col(col).isNull()).count()
            validations.append({
                "check": f"null_check_{col}",
                "passed": null_count == 0,
                "message": f"{null_count} null values in {col}",
                "severity": "CRITICAL" if null_count > 0 else "INFO"
            })
        
        # Enhanced PD range validation
        invalid_pd = df.filter((F.col("calculated_pd") < 0) | (F.col("calculated_pd") > 1)).count()
        validations.append({
            "check": "pd_range_validation",
            "passed": invalid_pd == 0,
            "message": f"{invalid_pd} loans with PD outside [0, 1]",
            "severity": "CRITICAL" if invalid_pd > 0 else "INFO"
        })
        
        # LGD range validation
        invalid_lgd = df.filter((F.col("calculated_lgd") < 0) | (F.col("calculated_lgd") > 1)).count()
        validations.append({
            "check": "lgd_range_validation", 
            "passed": invalid_lgd == 0,
            "message": f"{invalid_lgd} loans with LGD outside [0, 1]",
            "severity": "CRITICAL" if invalid_lgd > 0 else "INFO"
        })
        
        # ECL consistency validation
        ecl_inconsistent = df.filter(F.col("calculated_ecl") > F.col("calculated_ead")).count()
        validations.append({
            "check": "ecl_consistency_validation",
            "passed": ecl_inconsistent == 0,
            "message": f"{ecl_inconsistent} loans with ECL > EAD",
            "severity": "HIGH" if ecl_inconsistent > 0 else "INFO"
        })
        
        # Portfolio-level ECL validation
        total_exposure = df.agg(F.sum("current_balance")).collect()[0][0] or 0
        total_ecl = df.agg(F.sum("calculated_ecl")).collect()[0][0] or 0
        portfolio_ecl_rate = (total_ecl / total_exposure) if total_exposure > 0 else 0
        max_allowed_ecl_rate = validation_config['max_ecl_portfolio_percentage']
        
        validations.append({
            "check": "portfolio_ecl_rate_validation",
            "passed": portfolio_ecl_rate <= max_allowed_ecl_rate,
            "message": f"Portfolio ECL rate {portfolio_ecl_rate:.2%} vs max allowed {max_allowed_ecl_rate:.2%}",
            "severity": "HIGH" if portfolio_ecl_rate > max_allowed_ecl_rate else "INFO"
        })
        
        # Stage distribution validation
        stage_dist = self._get_stage_distribution(df)
        total_loans = sum(stage_dist.values())
        stage3_rate = stage_dist.get("STAGE_3", 0) / total_loans if total_loans > 0 else 0
        
        validations.append({
            "check": "stage3_distribution_check",
            "passed": stage3_rate < 0.1,  # Less than 10% in Stage 3
            "message": f"Stage 3 rate: {stage3_rate:.2%}",
            "severity": "MEDIUM" if stage3_rate >= 0.1 else "INFO"
        })
        
        # ML confidence validation if ML is enabled
        if self.ml_enabled:
            low_confidence_count = df.filter(F.col("stage_confidence") < 0.5).count()
            validations.append({
                "check": "ml_confidence_validation",
                "passed": low_confidence_count < total_loans * 0.1,  # Less than 10% low confidence
                "message": f"{low_confidence_count} loans with low ML confidence",
                "severity": "MEDIUM" if low_confidence_count >= total_loans * 0.1 else "INFO"
            })
        
        return validations

    def export_results(self, df: DataFrame, output_path: str = None) -> Dict[str, str]:
        """Export results to configured output formats with audit trail."""
        if output_path is None:
            output_path = self.config['output_settings']['results_path']
        
        output_formats = self.config['output_settings']['output_formats']
        compression = self.config['output_settings']['compression']
        
        export_paths = {}
        
        # Create timestamped output directory
        import os
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{output_path}/ifrs9_results_{timestamp}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Export in configured formats
        if 'parquet' in output_formats:
            parquet_path = f"{output_dir}/ifrs9_calculations.parquet"
            df.write.mode("overwrite").option("compression", compression).parquet(parquet_path)
            export_paths['parquet'] = parquet_path
        
        if 'json' in output_formats:
            # Export summary report as JSON
            summary_report = self.generate_comprehensive_summary_report(df)
            json_path = f"{output_dir}/ifrs9_summary_report.json"
            
            import json
            with open(json_path, 'w') as f:
                json.dump(summary_report, f, indent=2, default=str)
            export_paths['json'] = json_path
        
        # Export audit trail
        if self.config['validation']['audit_trail']['enabled']:
            audit_path = f"{output_dir}/audit_trail.json"
            import json
            with open(audit_path, 'w') as f:
                json.dump(self.audit_trail, f, indent=2, default=str)
            export_paths['audit'] = audit_path
        
        self._log_audit_event("export_complete", {"export_paths": export_paths})
        
        return export_paths

    def stop(self):
        """Stop the Spark session and cleanup resources."""
        self._log_audit_event("engine_shutdown", {})
        if self.spark:
            self.spark.stop()


if __name__ == "__main__":
    # Example usage
    engine = IFRS9RulesEngine()
    
    # Load sample data
    # loan_df = engine.spark.read.csv("data/raw/loan_portfolio.csv", header=True, inferSchema=True)
    # processed_df = engine.process_portfolio(loan_df)
    # summary = engine.generate_summary_report(processed_df)
    # print(summary)
    
    engine.stop()