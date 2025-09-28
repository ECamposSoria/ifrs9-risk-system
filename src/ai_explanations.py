"""
Explanation utilities for the IFRS9 Risk System.

This module now operates entirely offline using rule-based logic and local
statistical models. All external LLM integrations have been removed to ensure
the project runs without generative AI dependencies.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

# Optional BigQuery support (non-LLM)
try:
    from google.cloud import bigquery
except Exception:  # pragma: no cover - optional dependency
    bigquery = None

AI_EXPLANATIONS_ENABLED = os.getenv('AI_EXPLANATIONS_ENABLED', 'false').lower() == 'true'

# ML and explanation imports
import shap
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import IsolationForest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IFRS9ExplanationEngine:
    """Main class for IFRS9 explanations with Vertex AI and local fallback support"""

    def __init__(self,
                 project_id: Optional[str] = None,
                 location: str = "us-central1",
                 pd_endpoint_id: Optional[str] = None,
                 anomaly_endpoint_id: Optional[str] = None,
                 credentials_path: Optional[str] = None):
        """
        Initialize IFRS9 Explanation Engine with local/cloud modes

        Args:
            project_id: GCP project ID (required only if VERTEX_AI_ENABLED=true)
            location: Vertex AI location
            pd_endpoint_id: Probability of Default model endpoint ID
            anomaly_endpoint_id: Anomaly detection model endpoint ID
            credentials_path: Path to service account credentials
        """
        self.vertex_ai_enabled = False
        self.explanations_enabled = AI_EXPLANATIONS_ENABLED
        self.project_id = project_id
        self.location = location
        self.pd_endpoint_id = pd_endpoint_id
        self.anomaly_endpoint_id = anomaly_endpoint_id

        # Initialize local explanation helpers
        self._init_local_explainer()

        # Optional BigQuery client for data retrieval (non-LLM)
        if bigquery is not None:
            try:
                self.bq_client = bigquery.Client(project=self.project_id) if self.project_id else bigquery.Client()
            except Exception:
                self.bq_client = None
        else:
            self.bq_client = None

    def _init_local_explainer(self):
        """Initialize local rule-based explanation system"""
        self.explanation_templates = {
            "STAGE_1": "Loan classified as Stage 1 (Performing) due to: {reasons}",
            "STAGE_2": "Loan moved to Stage 2 (Underperforming) due to: {reasons}",
            "STAGE_3": "Loan classified as Stage 3 (Non-performing) due to: {reasons}"
        }

        self.risk_factors = {
            "high_dpd": "Days past due exceeds threshold",
            "credit_score_decline": "Significant credit score deterioration",
            "payment_history": "Poor payment history pattern",
            "economic_stress": "Economic stress indicators present",
            "collateral_decline": "Collateral value depreciation"
        }

    def explain_loan_decision(self, loan_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate explanation for loan staging decision

        Args:
            loan_data: Dictionary containing loan information

        Returns:
            Dict containing explanation details
        """
        return self._generate_rule_based_explanation(loan_data)

    def _generate_rule_based_explanation(self, loan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate rule-based explanation using local logic"""
        stage = loan_data.get("calculated_stage", "STAGE_1")
        reasons = []

        # Simple rule-based logic
        if loan_data.get("days_past_due", 0) > 30:
            reasons.append(self.risk_factors["high_dpd"])

        if loan_data.get("credit_score", 750) < 600:
            reasons.append(self.risk_factors["credit_score_decline"])

        # Default reasons if none identified
        if not reasons:
            if stage == "STAGE_1":
                reasons = ["No significant credit risk indicators detected"]
            else:
                reasons = ["Credit risk assessment triggered staging change"]

        explanation = self.explanation_templates[stage].format(
            reasons=", ".join(reasons)
        )

        return {
            "explanation_type": "rule_based",
            "explanation": explanation,
            "confidence": 0.75,
            "risk_factors": reasons,
            "stage": stage
        }

    def _build_explanation_prompt(self, loan_data: Dict[str, Any]) -> str:
        """Build prompt for AI explanation generation"""
        return f"""
        Explain why this loan is classified as {loan_data.get('calculated_stage', 'STAGE_1')}:

        Loan Details:
        - Credit Score: {loan_data.get('credit_score', 'N/A')}
        - Days Past Due: {loan_data.get('days_past_due', 0)}
        - Loan Amount: {loan_data.get('loan_amount', 'N/A')}
        - Current Balance: {loan_data.get('current_balance', 'N/A')}

        Please provide a clear, professional explanation suitable for regulatory reporting.
        """

    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the explanation system configuration"""
        return {
            "explanations_enabled": self.explanations_enabled,
            "explanation_mode": "rule_based",
            "project_id": getattr(self, 'project_id', None)
        }

class IFRS9ExplainerCore(VertexAIExplanationEngine):
    """Core explanation functionality for IFRS9 decisions"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # IFRS9 business rules thresholds
        self.stage_thresholds = {
            'dpd_stage_3': 90,
            'pd_stage_2': 0.05,
            'significant_increase_factors': {
                'credit_score_drop': 100,
                'dti_increase': 0.2,
                'employment_loss': True
            }
        }
    
    def get_loan_data(self, loan_id: str, table_id: str) -> pd.DataFrame:
        """Retrieve comprehensive loan data from BigQuery"""
        try:
            query = f"""
            SELECT 
                loan_id,
                customer_id,
                loan_amount,
                current_balance,
                interest_rate,
                credit_score,
                dti_ratio,
                employment_length,
                days_past_due,
                provision_stage,
                pd_12m,
                pd_lifetime,
                lgd,
                ead,
                ecl,
                region,
                producto_tipo,
                sector,
                historial_de_pagos,
                risk_rating,
                origination_date,
                created_at
            FROM `{table_id}`
            WHERE loan_id = '{loan_id}'
            """
            
            df = self.bq_client.query(query).to_dataframe()
            if df.empty:
                logger.warning(f"No data found for loan {loan_id}")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving loan data for {loan_id}: {str(e)}")
            return pd.DataFrame()
    
    def predict_pd_with_endpoint(self, loan_data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Get PD prediction (rule-based fallback)."""
        return self._fallback_pd_calculation(loan_data)

    def _prepare_prediction_instance(self, loan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare loan data for ML model prediction (kept for compatibility)."""
        features = {
            'loan_amount': float(loan_data.get('loan_amount', 0)),
            'current_balance': float(loan_data.get('current_balance', 0)),
            'interest_rate': float(loan_data.get('interest_rate', 0)),
            'credit_score': int(loan_data.get('credit_score', 600)),
            'dti_ratio': float(loan_data.get('dti_ratio', 0)),
            'employment_length': int(loan_data.get('employment_length', 0)),
            'days_past_due': int(loan_data.get('days_past_due', 0)),
            'term_months': int(loan_data.get('term_months', 60)),
            'ltv_ratio': float(loan_data.get('ltv_ratio', 0.8))
        }
        features.update(self._encode_product_type(loan_data.get('producto_tipo', '')))
        features.update(self._encode_region(loan_data.get('region', '')))
        return features
    
    def _encode_product_type(self, product_type: str) -> Dict[str, float]:
        """One-hot encode product type"""
        products = [
            'Hipoteca Residencial', 'Hipoteca Comercial', 'Préstamo Personal',
            'Préstamo Empresarial', 'Línea de Crédito', 'Tarjeta de Crédito',
            'Préstamo Vehículo', 'Préstamo Estudiantil', 'Microcrédito'
        ]
        
        encoding = {}
        for product in products:
            key = f"product_{product.lower().replace(' ', '_')}"
            encoding[key] = 1.0 if product == product_type else 0.0
        
        return encoding
    
    def _encode_region(self, region: str) -> Dict[str, float]:
        """One-hot encode region"""
        regions = [
            'Madrid', 'Cataluña', 'Valencia', 'Andalucía', 'País Vasco',
            'Galicia', 'Castilla y León', 'Murcia', 'Aragón', 'Canarias'
        ]
        
        encoding = {}
        for reg in regions:
            key = f"region_{reg.lower().replace(' ', '_')}"
            encoding[key] = 1.0 if reg == region else 0.0
        
        return encoding
    
    def _fallback_pd_calculation(self, loan_data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Fallback PD calculation when endpoint is unavailable"""
        credit_score = loan_data.get('credit_score', 600)
        days_past_due = loan_data.get('days_past_due', 0)
        dti_ratio = loan_data.get('dti_ratio', 0.3)
        
        # Simple logistic-like calculation
        base_pd = 0.02
        
        # Credit score adjustment
        if credit_score >= 800:
            score_adjustment = 0.5
        elif credit_score >= 700:
            score_adjustment = 1.0
        elif credit_score >= 600:
            score_adjustment = 1.5
        else:
            score_adjustment = 3.0
        
        # DPD adjustment
        dpd_adjustment = 1.0 + (days_past_due / 30) * 0.5
        
        # DTI adjustment
        dti_adjustment = 1.0 + max(0, (dti_ratio - 0.4) * 2)
        
        pd_value = min(1.0, base_pd * score_adjustment * dpd_adjustment * dti_adjustment)
        
        metadata = {
            'model_type': 'fallback_calculation',
            'base_pd': base_pd,
            'adjustments': {
                'credit_score': score_adjustment,
                'days_past_due': dpd_adjustment,
                'dti_ratio': dti_adjustment
            }
        }
        
        return pd_value, metadata
    
    def determine_ifrs9_stage(self, loan_data: Dict[str, Any], pd_value: float) -> Tuple[int, Dict[str, Any]]:
        """Apply IFRS9 rules to determine stage"""
        days_past_due = loan_data.get('days_past_due', 0)
        original_credit_score = loan_data.get('original_credit_score')
        current_credit_score = loan_data.get('credit_score', 600)
        
        # Stage 3: Credit-impaired (default)
        if days_past_due >= self.stage_thresholds['dpd_stage_3']:
            return 3, {
                'primary_reason': 'days_past_due',
                'dpd_value': days_past_due,
                'threshold_exceeded': self.stage_thresholds['dpd_stage_3']
            }
        
        # Check for significant increase in credit risk
        significant_increase_indicators = []
        
        # High PD trigger
        if pd_value > self.stage_thresholds['pd_stage_2']:
            significant_increase_indicators.append('high_pd')
        
        # Credit score deterioration
        if (original_credit_score and 
            (original_credit_score - current_credit_score) >= 
            self.stage_thresholds['significant_increase_factors']['credit_score_drop']):
            significant_increase_indicators.append('credit_score_deterioration')
        
        # DTI increase
        original_dti = loan_data.get('original_dti_ratio', 0.3)
        current_dti = loan_data.get('dti_ratio', 0.3)
        if (current_dti - original_dti) >= self.stage_thresholds['significant_increase_factors']['dti_increase']:
            significant_increase_indicators.append('dti_increase')
        
        # Employment length reduction
        if loan_data.get('employment_length', 12) < 6:
            significant_increase_indicators.append('short_employment')
        
        # Stage 2: Significant increase in credit risk
        if len(significant_increase_indicators) >= 2 or days_past_due > 30:
            return 2, {
                'primary_reason': 'significant_increase',
                'indicators': significant_increase_indicators,
                'pd_value': pd_value,
                'days_past_due': days_past_due
            }
        
        # Stage 1: Normal credit risk
        return 1, {
            'primary_reason': 'normal_credit_risk',
            'pd_value': pd_value,
            'indicators_count': len(significant_increase_indicators)
        }

class NaturalLanguageExplainer(IFRS9ExplainerCore):
    """Natural language explanation generation for IFRS9 decisions"""
    
    def generate_ifrs9_explanation(self, 
                                  loan_id: str, 
                                  table_id: str,
                                  use_gemini: bool = False) -> Dict[str, Any]:
        """Generate comprehensive IFRS9 explanation for a loan"""
        
        # Get loan data
        loan_df = self.get_loan_data(loan_id, table_id)
        if loan_df.empty:
            return {"error": f"Loan {loan_id} not found"}
        
        loan_data = loan_df.iloc[0].to_dict()
        
        # Get PD prediction
        pd_value, pd_metadata = self.predict_pd_with_endpoint(loan_data)
        
        # Determine IFRS9 stage
        stage, stage_reasoning = self.determine_ifrs9_stage(loan_data, pd_value)
        
        # Generate natural language explanation (rule-based only)
        if use_gemini:
            logger.warning("Generative explanations are disabled; using rule-based output instead.")

        explanation = self._generate_rule_based_explanation(
            loan_id, loan_data, stage, pd_value, stage_reasoning
        )
        
        return {
            'loan_id': loan_id,
            'ifrs9_stage': stage,
            'probability_of_default': pd_value,
            'natural_language_explanation': explanation,
            'technical_reasoning': stage_reasoning,
            'prediction_metadata': pd_metadata,
            'explanation_timestamp': datetime.now().isoformat()
        }
    
    
    def _generate_rule_based_explanation(self,
                                       loan_id: str,
                                       loan_data: Dict[str, Any],
                                       stage: int,
                                       pd_value: float,
                                       stage_reasoning: Dict[str, Any]) -> str:
        """Generate rule-based explanation as fallback"""
        
        explanation_parts = []
        
        # Stage-specific explanations
        if stage == 3:
            explanation_parts.append(
                f"El préstamo {loan_id} está clasificado como Etapa 3 (Deterioro Crediticio) "
                f"debido a que presenta {loan_data.get('days_past_due', 0)} días de mora, "
                f"superando el umbral de 90 días establecido por IFRS9."
            )
        elif stage == 2:
            explanation_parts.append(
                f"El préstamo {loan_id} está clasificado como Etapa 2 (Incremento Significativo del Riesgo) "
                f"debido a indicadores de deterioro crediticio."
            )
            
            indicators = stage_reasoning.get('indicators', [])
            if 'high_pd' in indicators:
                explanation_parts.append(
                    f"La probabilidad de incumplimiento ({pd_value:.2%}) supera el umbral del 5%."
                )
            if 'credit_score_deterioration' in indicators:
                explanation_parts.append("Se observa un deterioro significativo en la puntuación crediticia.")
        else:
            explanation_parts.append(
                f"El préstamo {loan_id} está clasificado como Etapa 1 (Riesgo Crediticio Normal) "
                f"ya que mantiene indicadores de riesgo dentro de parámetros normales."
            )
        
        # Add additional context
        credit_score = loan_data.get('credit_score')
        if credit_score:
            if credit_score >= 700:
                explanation_parts.append("La puntuación crediticia es buena.")
            elif credit_score >= 600:
                explanation_parts.append("La puntuación crediticia es aceptable.")
            else:
                explanation_parts.append("La puntuación crediticia requiere monitoreo.")
        
        return ' '.join(explanation_parts)

class OutlierDetectionExplainer(IFRS9ExplainerCore):
    """Outlier detection and explanation for unusual loans"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.isolation_forest = None
        self.outlier_threshold = -0.1
    
    def detect_outliers_batch(self, 
                            table_id: str,
                            n_samples: int = 10000) -> pd.DataFrame:
        """Detect outliers in loan portfolio using Isolation Forest"""
        
        try:
            # Get sample of loan data for outlier detection
            query = f"""
            SELECT 
                loan_id,
                loan_amount,
                current_balance,
                interest_rate,
                credit_score,
                dti_ratio,
                days_past_due,
                pd_12m,
                lgd,
                ecl
            FROM `{table_id}`
            WHERE current_balance > 0
            LIMIT {n_samples}
            """
            
            df = self.bq_client.query(query).to_dataframe()
            
            if df.empty:
                logger.warning("No data found for outlier detection")
                return pd.DataFrame()
            
            # Prepare features for outlier detection
            features = df[['loan_amount', 'current_balance', 'interest_rate', 
                          'credit_score', 'dti_ratio', 'days_past_due', 
                          'pd_12m', 'lgd', 'ecl']].fillna(0)
            
            # Normalize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Train Isolation Forest
            self.isolation_forest = IsolationForest(
                contamination=0.05,  # 5% outliers
                random_state=42,
                n_estimators=100
            )
            
            # Detect outliers
            outlier_scores = self.isolation_forest.fit_predict(features_scaled)
            anomaly_scores = self.isolation_forest.score_samples(features_scaled)
            
            # Add results to dataframe
            df['is_outlier'] = outlier_scores == -1
            df['anomaly_score'] = anomaly_scores
            
            # Filter to outliers only
            outliers_df = df[df['is_outlier']].copy()
            
            logger.info(f"Detected {len(outliers_df)} outliers out of {len(df)} samples")
            
            return outliers_df
            
        except Exception as e:
            logger.error(f"Error in outlier detection: {str(e)}")
            return pd.DataFrame()
    
    def explain_outlier(self, 
                       loan_id: str, 
                       table_id: str,
                       anomaly_score: float) -> Dict[str, Any]:
        """Generate explanation for why a loan is an outlier"""
        
        loan_df = self.get_loan_data(loan_id, table_id)
        if loan_df.empty:
            return {"error": f"Loan {loan_id} not found"}
        
        loan_data = loan_df.iloc[0].to_dict()
        
        # Analyze unusual characteristics
        unusual_factors = self._analyze_unusual_factors(loan_data)
        
        # Generate explanation
        explanation = self._generate_outlier_explanation(
            loan_id, loan_data, anomaly_score, unusual_factors
        )
        
        return {
            'loan_id': loan_id,
            'anomaly_score': anomaly_score,
            'outlier_explanation': explanation,
            'unusual_factors': unusual_factors,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _analyze_unusual_factors(self, loan_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify unusual characteristics of the loan"""
        
        unusual_factors = []
        
        # Check loan amount
        loan_amount = loan_data.get('loan_amount', 0)
        if loan_amount > 1000000:
            unusual_factors.append({
                'factor': 'loan_amount',
                'value': loan_amount,
                'description': 'Importe del préstamo excepcionalmente alto',
                'severity': 'high'
            })
        elif loan_amount < 1000:
            unusual_factors.append({
                'factor': 'loan_amount',
                'value': loan_amount,
                'description': 'Importe del préstamo excepcionalmente bajo',
                'severity': 'medium'
            })
        
        # Check interest rate
        interest_rate = loan_data.get('interest_rate', 0)
        if interest_rate > 20:
            unusual_factors.append({
                'factor': 'interest_rate',
                'value': interest_rate,
                'description': 'Tipo de interés excepcionalmente alto',
                'severity': 'high'
            })
        elif interest_rate < 1:
            unusual_factors.append({
                'factor': 'interest_rate',
                'value': interest_rate,
                'description': 'Tipo de interés excepcionalmente bajo',
                'severity': 'medium'
            })
        
        # Check credit score
        credit_score = loan_data.get('credit_score', 600)
        if credit_score < 400:
            unusual_factors.append({
                'factor': 'credit_score',
                'value': credit_score,
                'description': 'Puntuación crediticia extremadamente baja',
                'severity': 'high'
            })
        elif credit_score > 900:
            unusual_factors.append({
                'factor': 'credit_score',
                'value': credit_score,
                'description': 'Puntuación crediticia extremadamente alta',
                'severity': 'low'
            })
        
        # Check DTI ratio
        dti_ratio = loan_data.get('dti_ratio', 0.3)
        if dti_ratio > 0.8:
            unusual_factors.append({
                'factor': 'dti_ratio',
                'value': dti_ratio,
                'description': 'Ratio de endeudamiento sobre ingresos muy alto',
                'severity': 'high'
            })
        
        # Check ECL rate
        current_balance = loan_data.get('current_balance', 1)
        ecl = loan_data.get('ecl', 0)
        if current_balance > 0:
            ecl_rate = ecl / current_balance
            if ecl_rate > 0.1:
                unusual_factors.append({
                    'factor': 'ecl_rate',
                    'value': ecl_rate,
                    'description': 'Tasa de pérdida crediticia esperada muy alta',
                    'severity': 'high'
                })
        
        return unusual_factors
    
    def _generate_outlier_explanation(self,
                                    loan_id: str,
                                    loan_data: Dict[str, Any],
                                    anomaly_score: float,
                                    unusual_factors: List[Dict[str, Any]]) -> str:
        """Generate natural language explanation for outlier"""
        
        explanation_parts = []
        
        explanation_parts.append(
            f"El préstamo {loan_id} ha sido identificado como atípico "
            f"con una puntuación de anomalía de {anomaly_score:.3f}."
        )
        
        if unusual_factors:
            high_severity_factors = [f for f in unusual_factors if f['severity'] == 'high']
            
            if high_severity_factors:
                explanation_parts.append("Los factores más destacados son:")
                for factor in high_severity_factors:
                    explanation_parts.append(f"- {factor['description']}")
            
            medium_severity_factors = [f for f in unusual_factors if f['severity'] == 'medium']
            if medium_severity_factors:
                explanation_parts.append("Otros factores a considerar:")
                for factor in medium_severity_factors:
                    explanation_parts.append(f"- {factor['description']}")
        else:
            explanation_parts.append(
                "La anomalía se debe a una combinación de características "
                "que no se observan frecuentemente en el portafolio."
            )
        
        explanation_parts.append(
            "Se recomienda una revisión manual detallada para determinar "
            "acciones apropiadas de gestión de riesgo."
        )
        
        return ' '.join(explanation_parts)

class SHAPExplainer(IFRS9ExplainerCore):
    """SHAP-based model explanations for ML transparency"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shap_explainer = None
        self.background_data = None
    
    def initialize_shap_explainer(self, 
                                table_id: str,
                                n_background_samples: int = 100):
        """Initialize SHAP explainer with background data"""
        
        try:
            # Get background data for SHAP
            query = f"""
            SELECT 
                loan_amount,
                current_balance,
                interest_rate,
                credit_score,
                dti_ratio,
                employment_length,
                days_past_due
            FROM `{table_id}`
            WHERE current_balance > 0
            LIMIT {n_background_samples}
            """
            
            background_df = self.bq_client.query(query).to_dataframe()
            
            if background_df.empty:
                raise ValueError("No background data available for SHAP")
            
            # Prepare background data
            self.background_data = background_df.fillna(0).values
            
            # Create prediction function for SHAP
            def prediction_function(data):
                results = []
                for row in data:
                    # Convert to dictionary format
                    loan_dict = {
                        'loan_amount': row[0],
                        'current_balance': row[1],
                        'interest_rate': row[2],
                        'credit_score': row[3],
                        'dti_ratio': row[4],
                        'employment_length': row[5],
                        'days_past_due': row[6]
                    }
                    
                    # Get PD prediction
                    pd_value, _ = self.predict_pd_with_endpoint(loan_dict)
                    results.append(pd_value)
                
                return np.array(results)
            
            # Initialize SHAP explainer
            self.shap_explainer = shap.KernelExplainer(
                prediction_function, 
                self.background_data,
                link="identity"
            )
            
            logger.info("SHAP explainer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing SHAP explainer: {str(e)}")
            raise
    
    def generate_shap_explanation(self, 
                                loan_id: str, 
                                table_id: str) -> Dict[str, Any]:
        """Generate SHAP explanations for a specific loan"""
        
        if self.shap_explainer is None:
            self.initialize_shap_explainer(table_id)
        
        # Get loan data
        loan_df = self.get_loan_data(loan_id, table_id)
        if loan_df.empty:
            return {"error": f"Loan {loan_id} not found"}
        
        loan_data = loan_df.iloc[0].to_dict()
        
        # Prepare instance for SHAP
        instance = np.array([[
            loan_data.get('loan_amount', 0),
            loan_data.get('current_balance', 0),
            loan_data.get('interest_rate', 0),
            loan_data.get('credit_score', 600),
            loan_data.get('dti_ratio', 0.3),
            loan_data.get('employment_length', 12),
            loan_data.get('days_past_due', 0)
        ]])
        
        try:
            # Generate SHAP values
            shap_values = self.shap_explainer.shap_values(instance, nsamples=100)
            
            # Feature names
            feature_names = [
                'loan_amount', 'current_balance', 'interest_rate', 
                'credit_score', 'dti_ratio', 'employment_length', 
                'days_past_due'
            ]
            
            # Create explanation dictionary
            shap_explanation = {}
            for i, feature in enumerate(feature_names):
                shap_explanation[feature] = {
                    'value': instance[0][i],
                    'shap_value': float(shap_values[0][i]),
                    'impact': 'positive' if shap_values[0][i] > 0 else 'negative',
                    'magnitude': abs(float(shap_values[0][i]))
                }
            
            # Rank features by importance
            feature_importance = sorted(
                shap_explanation.items(),
                key=lambda x: abs(x[1]['shap_value']),
                reverse=True
            )
            
            # Generate textual explanation
            top_features = feature_importance[:3]
            text_explanation = self._generate_shap_text_explanation(
                loan_id, top_features, shap_values[0]
            )
            
            return {
                'loan_id': loan_id,
                'shap_values': shap_explanation,
                'feature_importance_ranking': [f[0] for f in feature_importance],
                'text_explanation': text_explanation,
                'base_value': float(self.shap_explainer.expected_value),
                'prediction': float(sum(shap_values[0]) + self.shap_explainer.expected_value),
                'explanation_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanation: {str(e)}")
            return {"error": f"Failed to generate SHAP explanation: {str(e)}"}
    
    def _generate_shap_text_explanation(self,
                                      loan_id: str,
                                      top_features: List[Tuple[str, Dict[str, Any]]],
                                      shap_values: np.ndarray) -> str:
        """Generate textual explanation from SHAP values"""
        
        explanation_parts = []
        
        explanation_parts.append(
            f"Para el préstamo {loan_id}, los factores que más influyen "
            f"en la predicción del modelo son:"
        )
        
        for i, (feature, details) in enumerate(top_features, 1):
            impact_text = "aumenta" if details['impact'] == 'positive' else "reduce"
            
            feature_descriptions = {
                'loan_amount': 'el importe del préstamo',
                'current_balance': 'el saldo actual',
                'interest_rate': 'el tipo de interés',
                'credit_score': 'la puntuación crediticia',
                'dti_ratio': 'el ratio de endeudamiento',
                'employment_length': 'la antigüedad laboral',
                'days_past_due': 'los días de mora'
            }
            
            feature_desc = feature_descriptions.get(feature, feature)
            
            explanation_parts.append(
                f"{i}. {feature_desc.capitalize()} ({details['value']}) "
                f"{impact_text} el riesgo de impago significativamente."
            )
        
        total_impact = sum(shap_values)
        if total_impact > 0:
            explanation_parts.append(
                "En conjunto, estos factores indican un riesgo de impago "
                "superior a la media del portafolio."
            )
        else:
            explanation_parts.append(
                "En conjunto, estos factores indican un riesgo de impago "
                "inferior a la media del portafolio."
            )
        
        return ' '.join(explanation_parts)

class BatchExplanationProcessor(IFRS9ExplainerCore):
    """Batch processing for explanations across loan portfolios"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nl_explainer = NaturalLanguageExplainer(*args, **kwargs)
        self.outlier_explainer = OutlierDetectionExplainer(*args, **kwargs)
    
    def process_batch_explanations(self,
                                 input_table_id: str,
                                 output_table_id: str,
                                 batch_size: int = 100,
                                 include_outliers: bool = True) -> Dict[str, Any]:
        """Process explanations for entire loan portfolio"""
        
        try:
            # Get total count
            count_query = f"SELECT COUNT(*) as total FROM `{input_table_id}`"
            total_loans = self.bq_client.query(count_query).to_dataframe()['total'].iloc[0]
            
            logger.info(f"Processing explanations for {total_loans} loans")
            
            # Process in batches
            processed_count = 0
            results = []
            
            for offset in range(0, total_loans, batch_size):
                batch_query = f"""
                SELECT loan_id 
                FROM `{input_table_id}` 
                LIMIT {batch_size} OFFSET {offset}
                """
                
                batch_df = self.bq_client.query(batch_query).to_dataframe()
                
                for _, row in batch_df.iterrows():
                    loan_id = row['loan_id']
                    
                    try:
                        # Generate IFRS9 explanation
                        explanation = self.nl_explainer.generate_ifrs9_explanation(
                            loan_id, input_table_id, use_gemini=False
                        )
                        
                        results.append({
                            'loan_id': loan_id,
                            'ifrs9_stage': explanation.get('ifrs9_stage'),
                            'probability_of_default': explanation.get('probability_of_default'),
                            'explanation': explanation.get('natural_language_explanation'),
                            'processing_timestamp': datetime.now().isoformat()
                        })
                        
                        processed_count += 1
                        
                        if processed_count % 50 == 0:
                            logger.info(f"Processed {processed_count}/{total_loans} loans")
                            
                    except Exception as e:
                        logger.error(f"Error processing loan {loan_id}: {str(e)}")
                        continue
                
                # Save batch results to BigQuery
                if results:
                    self._save_batch_results(results, output_table_id)
                    results = []  # Clear for next batch
            
            # Process outliers if requested
            outlier_results = {}
            if include_outliers:
                outlier_results = self._process_outlier_batch(
                    input_table_id, f"{output_table_id}_outliers"
                )
            
            return {
                'total_loans_processed': processed_count,
                'output_table': output_table_id,
                'outlier_results': outlier_results,
                'processing_completed': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            raise
    
    def _save_batch_results(self, results: List[Dict[str, Any]], output_table_id: str):
        """Save batch results to BigQuery"""
        
        try:
            results_df = pd.DataFrame(results)
            
            job_config = bigquery.LoadJobConfig(
                schema=[
                    bigquery.SchemaField("loan_id", "STRING"),
                    bigquery.SchemaField("ifrs9_stage", "INTEGER"),
                    bigquery.SchemaField("probability_of_default", "FLOAT64"),
                    bigquery.SchemaField("explanation", "STRING"),
                    bigquery.SchemaField("processing_timestamp", "TIMESTAMP"),
                ],
                write_disposition="WRITE_APPEND"
            )
            
            job = self.bq_client.load_table_from_dataframe(
                results_df, output_table_id, job_config=job_config
            )
            
            job.result()  # Wait for completion
            
        except Exception as e:
            logger.error(f"Error saving batch results: {str(e)}")
            raise
    
    def _process_outlier_batch(self, 
                             input_table_id: str, 
                             output_table_id: str) -> Dict[str, Any]:
        """Process outlier detection for the entire portfolio"""
        
        try:
            # Detect outliers
            outliers_df = self.outlier_explainer.detect_outliers_batch(input_table_id)
            
            if outliers_df.empty:
                return {"outliers_detected": 0}
            
            # Generate explanations for outliers
            outlier_explanations = []
            
            for _, outlier in outliers_df.iterrows():
                loan_id = outlier['loan_id']
                anomaly_score = outlier['anomaly_score']
                
                explanation = self.outlier_explainer.explain_outlier(
                    loan_id, input_table_id, anomaly_score
                )
                
                if 'error' not in explanation:
                    outlier_explanations.append({
                        'loan_id': loan_id,
                        'anomaly_score': anomaly_score,
                        'outlier_explanation': explanation.get('outlier_explanation'),
                        'unusual_factors': json.dumps(explanation.get('unusual_factors', [])),
                        'detection_timestamp': datetime.now().isoformat()
                    })
            
            # Save outlier explanations
            if outlier_explanations:
                explanations_df = pd.DataFrame(outlier_explanations)
                
                job_config = bigquery.LoadJobConfig(
                    schema=[
                        bigquery.SchemaField("loan_id", "STRING"),
                        bigquery.SchemaField("anomaly_score", "FLOAT64"),
                        bigquery.SchemaField("outlier_explanation", "STRING"),
                        bigquery.SchemaField("unusual_factors", "STRING"),
                        bigquery.SchemaField("detection_timestamp", "TIMESTAMP"),
                    ],
                    write_disposition="WRITE_TRUNCATE"
                )
                
                job = self.bq_client.load_table_from_dataframe(
                    explanations_df, output_table_id, job_config=job_config
                )
                
                job.result()
            
            return {
                "outliers_detected": len(outlier_explanations),
                "output_table": output_table_id
            }
            
        except Exception as e:
            logger.error(f"Error in outlier batch processing: {str(e)}")
            return {"error": str(e)}

# Utility functions
def initialize_explanation_system(project_id: str, 
                                location: str = "us-central1",
                                credentials_path: Optional[str] = None) -> Dict[str, Any]:
    """Initialize the complete explanation system"""
    
    config = {
        'project_id': project_id,
        'location': location,
        'credentials_path': credentials_path
    }
    
    # Initialize all explainer components
    components = {
        'natural_language': NaturalLanguageExplainer(**config),
        'outlier_detection': OutlierDetectionExplainer(**config),
        'shap_explainer': SHAPExplainer(**config),
        'batch_processor': BatchExplanationProcessor(**config)
    }
    
    logger.info("IFRS9 AI Explanation System initialized successfully")
    
    return components

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='IFRS9 AI Explanations Module')
    parser.add_argument('--project-id', required=True, help='GCP Project ID')
    parser.add_argument('--table-id', required=True, help='BigQuery table with loan data')
    parser.add_argument('--loan-id', help='Specific loan to explain')
    parser.add_argument('--batch-process', action='store_true', help='Process all loans')
    parser.add_argument('--output-table', help='Output table for batch processing')
    
    args = parser.parse_args()
    
    # Initialize system
    system_components = initialize_explanation_system(args.project_id)
    
    if args.loan_id:
        # Single loan explanation
        explainer = system_components['natural_language']
        result = explainer.generate_ifrs9_explanation(args.loan_id, args.table_id)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    elif args.batch_process and args.output_table:
        # Batch processing
        processor = system_components['batch_processor']
        result = processor.process_batch_explanations(
            args.table_id, args.output_table
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    else:
        print("Please specify either --loan-id for single explanation or --batch-process with --output-table")
