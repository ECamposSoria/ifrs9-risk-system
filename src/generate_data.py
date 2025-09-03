"""Synthetic data generator for IFRS9 risk analysis.

This module generates realistic synthetic loan portfolio data for testing
and development of the IFRS9 credit provisions system.
"""

import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker
import pyarrow as pa
import pyarrow.parquet as pq


class DataGenerator:
    """Generate synthetic loan portfolio data for IFRS9 analysis.
    
    This class creates realistic synthetic data including:
    - Loan accounts with various characteristics
    - Payment history and delinquency patterns
    - Customer demographics
    - Collateral information
    - Macroeconomic indicators
    """
    
    def __init__(self, seed: Optional[int] = 42):
        """Initialize the data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.fake = Faker(['es_ES', 'en_US'])  # Spanish and English locales
        Faker.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Regional configurations for Spain/LATAM
        self.regions = [
            "Madrid", "Cataluña", "Valencia", "Andalucía", "País Vasco",
            "Galicia", "Castilla y León", "Castilla-La Mancha", "Murcia",
            "Aragón", "Canarias", "Extremadura", "Asturias", "Baleares"
        ]
        
        self.product_types = [
            "Hipoteca Residencial", "Hipoteca Comercial", "Préstamo Personal",
            "Préstamo Empresarial", "Línea de Crédito", "Tarjeta de Crédito",
            "Préstamo Vehículo", "Préstamo Estudiantil", "Microcrédito",
            "Factoring", "Confirming", "Leasing Inmobiliario", "Renting"
        ]
        
        self.payment_history_patterns = [
            "Excelente", "Muy Bueno", "Bueno", "Regular", "Malo", "Muy Malo"
        ]
        
        # Economic cycle parameters
        self.economic_cycles = {
            "expansion": {"gdp_growth": (2.0, 4.5), "unemployment": (5.0, 8.0)},
            "peak": {"gdp_growth": (3.5, 5.0), "unemployment": (4.0, 6.5)},
            "contraction": {"gdp_growth": (-2.0, 1.5), "unemployment": (8.0, 12.0)},
            "trough": {"gdp_growth": (-3.0, 0.5), "unemployment": (10.0, 15.0)}
        }
        
    def generate_loan_portfolio(
        self,
        n_loans: int = 10000,
        start_date: str = "2020-01-01",
        end_date: str = "2024-01-01"
    ) -> pd.DataFrame:
        """Generate synthetic loan portfolio data.
        
        Args:
            n_loans: Number of loans to generate
            start_date: Portfolio start date
            end_date: Portfolio end date
            
        Returns:
            DataFrame with loan portfolio data
        """
        loans = []
        
        for loan_id in range(1, n_loans + 1):
            loan = self._generate_single_loan(loan_id, start_date, end_date)
            loans.append(loan)
            
        return pd.DataFrame(loans)
    
    def _generate_single_loan(
        self,
        loan_id: int,
        start_date: str,
        end_date: str
    ) -> Dict:
        """Generate a single loan record.
        
        Args:
            loan_id: Unique loan identifier
            start_date: Loan origination date range start
            end_date: Loan origination date range end
            
        Returns:
            Dictionary with loan attributes
        """
        # Convert string dates to datetime objects for faker
        from datetime import datetime
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        # Select region and product type
        region = random.choice(self.regions)
        producto_tipo = random.choice(self.product_types)
        
        # Generate realistic credit score distribution
        credit_score = self._generate_realistic_credit_score()
        
        # Generate loan characteristics
        loan_data = {
            "loan_id": f"L{loan_id:06d}",
            "customer_id": f"C{random.randint(1, 5000):06d}",
            "loan_amount": self._generate_loan_amount_by_type(producto_tipo),
            "interest_rate": self._generate_interest_rate(producto_tipo, credit_score),
            "term_months": self._generate_term_months(producto_tipo),
            "loan_type": producto_tipo,  # Use Spanish product types
            "origination_date": self.fake.date_between(start_date=start_dt, end_date=end_dt),
            "maturity_date": None,  # Will be calculated
            "payment_frequency": random.choice(["MONTHLY", "QUARTERLY", "SEMI_ANNUAL"]),
            "collateral_value": None,  # Will be set based on loan type
            "ltv_ratio": None,  # Loan-to-value ratio
            "customer_income": self._generate_income_by_region(region),
            "credit_score": credit_score,
            "employment_status": random.choice(["EMPLOYED", "SELF_EMPLOYED", "RETIRED", "UNEMPLOYED"]),
            "dti_ratio": None,  # Debt-to-income ratio (will be calculated)
            "employment_length": random.randint(0, 300),  # Months of employment
            "loan_purpose": self._get_loan_purpose(producto_tipo),
            "days_past_due": 0,
            "current_balance": None,  # Will be calculated
            "monthly_payment": None,  # Will be calculated
            "provision_stage": "STAGE_1",  # Initial stage
            "pd_12m": None,  # 12-month PD
            "pd_lifetime": None,  # Lifetime PD
            "lgd": None,  # Loss given default
            "ead": None,  # Exposure at default
            "ecl": None,  # Expected credit loss
            # New required fields
            "historial_de_pagos": random.choice(self.payment_history_patterns),
            "region": region,
            "producto_tipo": producto_tipo,
            # Additional fields for enhanced analysis
            "currency": "EUR",
            "sector": self._get_sector_by_product(producto_tipo),
            "risk_rating": None,  # Will be calculated
            "created_at": datetime.now(),
        }
        
        # Calculate derived fields
        loan_data = self._calculate_loan_metrics(loan_data)
        loan_data = self._assign_risk_parameters(loan_data)
        
        return loan_data
    
    def _generate_realistic_credit_score(self) -> int:
        """Generate realistic credit score distribution."""
        # Spanish credit scoring typically ranges 300-999
        score_ranges = [
            (300, 500, 0.05),  # Very poor
            (500, 600, 0.10),  # Poor
            (600, 700, 0.25),  # Fair
            (700, 800, 0.35),  # Good
            (800, 900, 0.20),  # Very good
            (900, 999, 0.05),  # Excellent
        ]
        
        range_choice = random.choices(
            score_ranges, 
            weights=[weight for _, _, weight in score_ranges]
        )[0]
        
        return random.randint(range_choice[0], range_choice[1])
    
    def _generate_loan_amount_by_type(self, product_type: str) -> float:
        """Generate loan amount based on product type."""
        amount_ranges = {
            "Hipoteca Residencial": (50000, 800000),
            "Hipoteca Comercial": (100000, 2000000),
            "Préstamo Personal": (1000, 50000),
            "Préstamo Empresarial": (10000, 500000),
            "Línea de Crédito": (5000, 100000),
            "Tarjeta de Crédito": (500, 15000),
            "Préstamo Vehículo": (5000, 80000),
            "Préstamo Estudiantil": (3000, 60000),
            "Microcrédito": (500, 25000),
            "Factoring": (10000, 200000),
            "Confirming": (5000, 150000),
            "Leasing Inmobiliario": (50000, 1000000),
            "Renting": (15000, 100000),
        }
        
        min_amount, max_amount = amount_ranges.get(product_type, (5000, 100000))
        return round(random.uniform(min_amount, max_amount), 2)
    
    def _generate_interest_rate(self, product_type: str, credit_score: int) -> float:
        """Generate interest rate based on product type and credit score."""
        base_rates = {
            "Hipoteca Residencial": (1.5, 4.5),
            "Hipoteca Comercial": (2.0, 6.0),
            "Préstamo Personal": (4.0, 12.0),
            "Préstamo Empresarial": (3.0, 8.0),
            "Línea de Crédito": (5.0, 15.0),
            "Tarjeta de Crédito": (12.0, 25.0),
            "Préstamo Vehículo": (3.0, 9.0),
            "Préstamo Estudiantil": (2.0, 7.0),
            "Microcrédito": (8.0, 20.0),
            "Factoring": (4.0, 10.0),
            "Confirming": (3.0, 8.0),
            "Leasing Inmobiliario": (2.5, 6.5),
            "Renting": (4.0, 10.0),
        }
        
        min_rate, max_rate = base_rates.get(product_type, (3.0, 10.0))
        base_rate = random.uniform(min_rate, max_rate)
        
        # Adjust based on credit score
        if credit_score >= 800:
            rate_adjustment = -0.5
        elif credit_score >= 700:
            rate_adjustment = 0.0
        elif credit_score >= 600:
            rate_adjustment = 1.0
        else:
            rate_adjustment = 2.5
        
        return round(max(0.1, base_rate + rate_adjustment), 2)
    
    def _generate_term_months(self, product_type: str) -> int:
        """Generate term months based on product type."""
        term_options = {
            "Hipoteca Residencial": [180, 240, 300, 360, 420, 480],
            "Hipoteca Comercial": [60, 120, 180, 240, 300],
            "Préstamo Personal": [12, 24, 36, 48, 60],
            "Préstamo Empresarial": [12, 24, 36, 60, 84, 120],
            "Línea de Crédito": [12, 24, 36],
            "Tarjeta de Crédito": [1, 3, 6, 12],  # Revolving but set term for calculation
            "Préstamo Vehículo": [36, 48, 60, 72, 84, 96],
            "Préstamo Estudiantil": [60, 84, 120, 180],
            "Microcrédito": [6, 12, 18, 24, 36],
            "Factoring": [1, 3, 6, 12],
            "Confirming": [1, 3, 6],
            "Leasing Inmobiliario": [84, 120, 180, 240],
            "Renting": [12, 24, 36, 48],
        }
        
        return random.choice(term_options.get(product_type, [12, 24, 36, 48, 60]))
    
    def _generate_income_by_region(self, region: str) -> float:
        """Generate income based on regional economic data."""
        # Regional income multipliers (based on Spanish regional GDP per capita)
        regional_multipliers = {
            "Madrid": 1.3,
            "País Vasco": 1.2,
            "Cataluña": 1.1,
            "Baleares": 1.0,
            "Valencia": 0.9,
            "Aragón": 0.95,
            "Galicia": 0.85,
            "Castilla y León": 0.8,
            "Andalucía": 0.75,
            "Murcia": 0.8,
            "Canarias": 0.85,
            "Castilla-La Mancha": 0.75,
            "Extremadura": 0.7,
            "Asturias": 0.85,
        }
        
        base_income = random.uniform(18000, 120000)  # Base income range in EUR
        multiplier = regional_multipliers.get(region, 1.0)
        
        return round(base_income * multiplier, 2)
    
    def _get_loan_purpose(self, product_type: str) -> str:
        """Get loan purpose based on product type."""
        purpose_mapping = {
            "Hipoteca Residencial": "Compra vivienda habitual",
            "Hipoteca Comercial": "Inversión inmobiliaria",
            "Préstamo Personal": random.choice(["Consumo", "Mejoras hogar", "Vacaciones", "Consolidación deuda"]),
            "Préstamo Empresarial": "Inversión empresarial",
            "Línea de Crédito": "Liquidez empresarial",
            "Tarjeta de Crédito": "Consumo",
            "Préstamo Vehículo": "Compra vehículo",
            "Préstamo Estudiantil": "Educación",
            "Microcrédito": "Emprendimiento",
            "Factoring": "Factoring facturas",
            "Confirming": "Pago proveedores",
            "Leasing Inmobiliario": "Inversión inmobiliaria",
            "Renting": "Uso vehículo",
        }
        
        return purpose_mapping.get(product_type, "Otros")
    
    def _get_sector_by_product(self, product_type: str) -> str:
        """Get economic sector based on product type."""
        sector_mapping = {
            "Hipoteca Residencial": "Particulares",
            "Hipoteca Comercial": "Inmobiliario",
            "Préstamo Personal": "Particulares",
            "Préstamo Empresarial": random.choice(["Industria", "Servicios", "Comercio", "Construcción"]),
            "Línea de Crédito": random.choice(["Servicios", "Comercio", "Industria"]),
            "Tarjeta de Crédito": "Particulares",
            "Préstamo Vehículo": "Particulares",
            "Préstamo Estudiantil": "Particulares",
            "Microcrédito": "PYMES",
            "Factoring": "Servicios",
            "Confirming": "Industria",
            "Leasing Inmobiliario": "Inmobiliario",
            "Renting": "Servicios",
        }
        
        return sector_mapping.get(product_type, "Otros")
    
    def _calculate_loan_metrics(self, loan: Dict) -> Dict:
        """Calculate derived loan metrics.
        
        Args:
            loan: Loan data dictionary
            
        Returns:
            Updated loan dictionary with calculated metrics
        """
        # Calculate maturity date
        loan["maturity_date"] = loan["origination_date"] + timedelta(days=loan["term_months"] * 30)
        
        # Calculate monthly payment (simplified)
        principal = loan["loan_amount"]
        rate = loan["interest_rate"] / 100 / 12
        n_payments = loan["term_months"]
        
        if rate > 0:
            loan["monthly_payment"] = round(
                principal * (rate * (1 + rate) ** n_payments) / ((1 + rate) ** n_payments - 1), 2
            )
        else:
            loan["monthly_payment"] = round(principal / n_payments, 2)
        
        # Set collateral value based on loan type
        if loan["loan_type"] == "MORTGAGE":
            loan["collateral_value"] = round(loan["loan_amount"] * random.uniform(1.1, 1.5), 2)
        elif loan["loan_type"] == "AUTO":
            loan["collateral_value"] = round(loan["loan_amount"] * random.uniform(1.0, 1.3), 2)
        elif loan["loan_type"] == "BUSINESS":
            loan["collateral_value"] = round(loan["loan_amount"] * random.uniform(0.5, 1.2), 2)
        else:
            loan["collateral_value"] = 0
        
        # Calculate LTV ratio
        if loan["collateral_value"] > 0:
            loan["ltv_ratio"] = round(loan["loan_amount"] / loan["collateral_value"], 2)
        else:
            loan["ltv_ratio"] = None
        
        # Calculate current balance (simplified - random reduction)
        months_elapsed = random.randint(0, min(loan["term_months"], 60))
        payment_ratio = 1 - (months_elapsed / loan["term_months"])
        loan["current_balance"] = round(loan["loan_amount"] * payment_ratio, 2)
        
        # Calculate DTI ratio
        annual_payment = loan["monthly_payment"] * 12
        loan["dti_ratio"] = round(annual_payment / loan["customer_income"], 2)
        
        return loan
    
    def _assign_risk_parameters(self, loan: Dict) -> Dict:
        """Assign IFRS9 risk parameters based on loan characteristics.
        
        Args:
            loan: Loan data dictionary
            
        Returns:
            Updated loan dictionary with risk parameters
        """
        # Enhanced DPD assignment based on multiple factors
        credit_score = loan["credit_score"]
        dti_ratio = loan["dti_ratio"]
        payment_history = loan["historial_de_pagos"]
        
        # Base DPD probability influenced by payment history
        history_dpd_weights = {
            "Excelente": [0.95, 0.04, 0.01, 0.0],
            "Muy Bueno": [0.85, 0.10, 0.04, 0.01],
            "Bueno": [0.75, 0.15, 0.08, 0.02],
            "Regular": [0.60, 0.25, 0.12, 0.03],
            "Malo": [0.40, 0.30, 0.20, 0.10],
            "Muy Malo": [0.20, 0.25, 0.30, 0.25]
        }
        
        dpd_options = [0, random.randint(1, 30), random.randint(31, 90), random.randint(91, 180)]
        base_weights = history_dpd_weights.get(payment_history, [0.8, 0.1, 0.07, 0.03])
        
        # Adjust weights based on credit score and DTI
        if credit_score < 500 or dti_ratio > 0.5:
            base_weights = [w * 0.6 if i == 0 else w * 1.2 for i, w in enumerate(base_weights)]
        elif credit_score >= 800 and dti_ratio < 0.3:
            base_weights = [w * 1.1 if i == 0 else w * 0.8 for i, w in enumerate(base_weights)]
        
        # Normalize weights
        total_weight = sum(base_weights)
        normalized_weights = [w / total_weight for w in base_weights]
        
        loan["days_past_due"] = random.choices(dpd_options, weights=normalized_weights)[0]
        
        # Assign provision stage based on days past due and significant increase criteria
        significant_increase = self._check_significant_increase(loan)
        
        if loan["days_past_due"] == 0 and not significant_increase:
            loan["provision_stage"] = 1
        elif loan["days_past_due"] <= 30 or significant_increase:
            loan["provision_stage"] = 2
        elif loan["days_past_due"] <= 90:
            loan["provision_stage"] = 2
        else:
            loan["provision_stage"] = 3
        
        # Calculate 12-month and lifetime PD
        pd_12m, pd_lifetime = self._calculate_pd_rates(loan)
        loan["pd_12m"] = pd_12m
        loan["pd_lifetime"] = pd_lifetime
        
        # Assign LGD based on collateral type and recovery prospects
        loan["lgd"] = self._calculate_lgd(loan)
        
        # Calculate EAD (current balance + potential future exposure)
        loan["ead"] = self._calculate_ead(loan)
        
        # Calculate ECL based on stage
        loan["ecl"] = self._calculate_ecl(loan)
        
        # Assign risk rating
        loan["risk_rating"] = self._assign_risk_rating(loan)
        
        return loan
    
    def _check_significant_increase(self, loan: Dict) -> bool:
        """Check for significant increase in credit risk."""
        credit_score = loan["credit_score"]
        dti_ratio = loan["dti_ratio"]
        employment_length = loan["employment_length"]
        
        # IFRS9 significant increase triggers
        triggers = [
            credit_score < 600,  # Low credit score
            dti_ratio > 0.45,    # High debt-to-income ratio
            employment_length < 12,  # Short employment history
            loan["historial_de_pagos"] in ["Malo", "Muy Malo"]
        ]
        
        return sum(triggers) >= 2  # At least 2 triggers
    
    def _calculate_pd_rates(self, loan: Dict) -> Tuple[float, float]:
        """Calculate 12-month and lifetime PD rates."""
        stage = loan["provision_stage"]
        credit_score = loan["credit_score"]
        product_type = loan["producto_tipo"]
        
        # Base PD rates by stage and product type
        if stage == 1:
            if "Hipoteca" in product_type:
                pd_12m = random.uniform(0.001, 0.02)
            elif product_type in ["Préstamo Personal", "Tarjeta de Crédito"]:
                pd_12m = random.uniform(0.01, 0.05)
            else:
                pd_12m = random.uniform(0.005, 0.03)
        elif stage == 2:
            if "Hipoteca" in product_type:
                pd_12m = random.uniform(0.05, 0.15)
            else:
                pd_12m = random.uniform(0.1, 0.3)
        else:  # Stage 3
            pd_12m = 1.0
        
        # Adjust for credit score
        if credit_score >= 800:
            pd_12m *= 0.5
        elif credit_score < 500:
            pd_12m *= 2.0
        
        # Lifetime PD is typically higher
        if stage == 3:
            pd_lifetime = 1.0
        else:
            pd_lifetime = min(1.0, pd_12m * random.uniform(2.5, 4.0))
        
        return round(pd_12m, 4), round(pd_lifetime, 4)
    
    def _calculate_lgd(self, loan: Dict) -> float:
        """Calculate Loss Given Default based on collateral and product type."""
        product_type = loan["producto_tipo"]
        collateral_value = loan["collateral_value"]
        loan_amount = loan["loan_amount"]
        
        # Base LGD by product type
        if "Hipoteca" in product_type and collateral_value > 0:
            base_lgd = random.uniform(0.15, 0.35)  # Secured by real estate
        elif product_type in ["Préstamo Vehículo", "Leasing Inmobiliario"]:
            base_lgd = random.uniform(0.25, 0.45)  # Asset-backed
        elif product_type in ["Factoring", "Confirming"]:
            base_lgd = random.uniform(0.10, 0.25)  # Trade finance
        else:
            base_lgd = random.uniform(0.45, 0.85)  # Unsecured
        
        # Adjust for LTV ratio if applicable
        if collateral_value > 0:
            ltv = loan_amount / collateral_value
            if ltv > 0.9:
                base_lgd *= 1.2
            elif ltv < 0.6:
                base_lgd *= 0.8
        
        return round(min(0.95, max(0.05, base_lgd)), 4)
    
    def _calculate_ead(self, loan: Dict) -> float:
        """Calculate Exposure at Default."""
        current_balance = loan["current_balance"]
        product_type = loan["producto_tipo"]
        
        # For revolving products, add credit conversion factor
        if product_type in ["Línea de Crédito", "Tarjeta de Crédito"]:
            unused_limit = loan["loan_amount"] - current_balance
            ccf = 0.75  # Credit conversion factor
            ead = current_balance + (unused_limit * ccf)
        else:
            ead = current_balance
        
        return round(ead, 2)
    
    def _calculate_ecl(self, loan: Dict) -> float:
        """Calculate Expected Credit Loss."""
        stage = loan["provision_stage"]
        ead = loan["ead"]
        lgd = loan["lgd"]
        
        if stage == 1:
            pd = loan["pd_12m"]
        else:
            pd = loan["pd_lifetime"]
        
        ecl = ead * pd * lgd
        return round(ecl, 2)
    
    def _assign_risk_rating(self, loan: Dict) -> str:
        """Assign internal risk rating."""
        credit_score = loan["credit_score"]
        stage = loan["provision_stage"]
        pd_12m = loan["pd_12m"]
        
        if stage == 3:
            return "D"  # Default
        elif pd_12m > 0.2:
            return "CCC"
        elif pd_12m > 0.1:
            return "CC"
        elif pd_12m > 0.05:
            return "C"
        elif credit_score >= 800:
            return "AAA"
        elif credit_score >= 750:
            return "AA"
        elif credit_score >= 700:
            return "A"
        elif credit_score >= 650:
            return "BBB"
        elif credit_score >= 600:
            return "BB"
        else:
            return "B"
    
    def generate_payment_history(
        self,
        loan_portfolio: pd.DataFrame,
        n_months: int = 12
    ) -> pd.DataFrame:
        """Generate payment history for loans.
        
        Args:
            loan_portfolio: DataFrame with loan data
            n_months: Number of months of history to generate
            
        Returns:
            DataFrame with payment history
        """
        payment_history = []
        
        for _, loan in loan_portfolio.iterrows():
            for month in range(n_months):
                payment = self._generate_payment_record(loan, month)
                payment_history.append(payment)
        
        return pd.DataFrame(payment_history)
    
    def _generate_payment_record(self, loan: pd.Series, month: int) -> Dict:
        """Generate a single payment record.
        
        Args:
            loan: Loan data
            month: Month number
            
        Returns:
            Dictionary with payment record
        """
        payment_date = pd.to_datetime(loan["origination_date"]) + pd.DateOffset(months=month)
        
        # Determine if payment was made based on DPD
        if loan["days_past_due"] > 0:
            payment_made = random.random() < 0.7  # 70% chance of payment if already delinquent
        else:
            payment_made = random.random() < 0.98  # 98% chance of payment if current
        
        payment_record = {
            "loan_id": loan["loan_id"],
            "payment_date": payment_date.date(),
            "scheduled_payment": loan["monthly_payment"],
            "actual_payment": loan["monthly_payment"] if payment_made else 0,
            "payment_status": "PAID" if payment_made else "MISSED",
            "days_late": 0 if payment_made else random.randint(1, 30),
        }
        
        return payment_record
    
    def generate_macroeconomic_data(
        self,
        start_date: str = "2020-01-01",
        end_date: str = "2024-01-01",
        include_cycles: bool = True
    ) -> pd.DataFrame:
        """Generate synthetic macroeconomic indicators with economic cycles.
        
        Args:
            start_date: Start date for data generation
            end_date: End date for data generation
            include_cycles: Whether to include economic cycle patterns
            
        Returns:
            DataFrame with macroeconomic indicators
        """
        dates = pd.date_range(start=start_date, end=end_date, freq="M")
        
        macro_data = []
        current_cycle = "expansion"  # Starting cycle phase
        months_in_cycle = 0
        
        for i, date in enumerate(dates):
            # Economic cycle progression (simplified)
            if include_cycles:
                if current_cycle == "expansion" and months_in_cycle > 24:
                    current_cycle = "peak"
                    months_in_cycle = 0
                elif current_cycle == "peak" and months_in_cycle > 6:
                    current_cycle = "contraction"
                    months_in_cycle = 0
                elif current_cycle == "contraction" and months_in_cycle > 18:
                    current_cycle = "trough"
                    months_in_cycle = 0
                elif current_cycle == "trough" and months_in_cycle > 6:
                    current_cycle = "expansion"
                    months_in_cycle = 0
                
                # Get cycle parameters
                cycle_params = self.economic_cycles[current_cycle]
                gdp_range = cycle_params["gdp_growth"]
                unemployment_range = cycle_params["unemployment"]
            else:
                gdp_range = (-2.0, 5.0)
                unemployment_range = (3.0, 10.0)
            
            # Generate with some trend and volatility
            base_gdp = random.uniform(*gdp_range)
            base_unemployment = random.uniform(*unemployment_range)
            
            # Add some persistence (trend following)
            if i > 0:
                prev_gdp = macro_data[-1]["gdp_growth"]
                prev_unemployment = macro_data[-1]["unemployment_rate"]
                base_gdp = 0.7 * prev_gdp + 0.3 * base_gdp
                base_unemployment = 0.8 * prev_unemployment + 0.2 * base_unemployment
            
            record = {
                "date": date.date(),
                "gdp_growth": round(base_gdp, 2),
                "unemployment_rate": round(base_unemployment, 2),
                "inflation_rate": round(random.uniform(0.5, 5.0), 2),
                "interest_rate": round(random.uniform(0.0, 7.0), 2),
                "housing_price_index": round(random.uniform(85, 130), 2),
                "currency_exchange_rate": round(random.uniform(0.85, 1.20), 4),  # EUR/USD
                "stock_market_index": round(random.uniform(8000, 15000), 2),
                "credit_spread": round(random.uniform(0.5, 4.0), 2),  # basis points over risk-free
                "economic_cycle": current_cycle,
                "created_at": datetime.now(),
            }
            macro_data.append(record)
            months_in_cycle += 1
        
        return pd.DataFrame(macro_data)
    
    def save_data(self, 
                  output_dir: str = "data/raw",
                  n_loans: int = 50000,
                  formats: List[str] = None,
                  batch_size: int = 10000):
        """Generate and save all synthetic data with multiple format support.
        
        Args:
            output_dir: Directory to save generated data
            n_loans: Number of loans to generate
            formats: List of formats to save ('csv', 'parquet', 'json')
            batch_size: Batch size for large dataset processing
        """
        if formats is None:
            formats = ['csv', 'parquet']
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Generating {n_loans:,} loan portfolio records...")
        
        # Generate loans in batches for memory efficiency
        all_loans = []
        for batch_start in range(0, n_loans, batch_size):
            batch_end = min(batch_start + batch_size, n_loans)
            print(f"  Processing batch {batch_start+1:,} to {batch_end:,}")
            
            batch_loans = []
            for loan_id in range(batch_start + 1, batch_end + 1):
                loan = self._generate_single_loan(loan_id, "2020-01-01", "2024-01-01")
                batch_loans.append(loan)
            
            all_loans.extend(batch_loans)
        
        loans_df = pd.DataFrame(all_loans)
        
        print("Generating payment history...")
        payments = self.generate_payment_history(loans_df, n_months=12)
        
        print("Generating macroeconomic data...")
        macro = self.generate_macroeconomic_data(include_cycles=True)
        
        # Generate additional IFRS9 specific datasets
        print("Generating IFRS9 stage transitions...")
        stage_transitions = self.generate_stage_transitions(loans_df)
        
        datasets = {
            'loan_portfolio': loans_df,
            'payment_history': payments,
            'macroeconomic_data': macro,
            'stage_transitions': stage_transitions
        }
        
        # Save in requested formats
        for dataset_name, df in datasets.items():
            print(f"Saving {dataset_name} ({len(df):,} records)...")
            
            if 'csv' in formats:
                df.to_csv(f"{output_dir}/{dataset_name}.csv", index=False)
            
            if 'parquet' in formats:
                # Convert datetime columns for Parquet compatibility
                df_parquet = df.copy()
                for col in df_parquet.columns:
                    if df_parquet[col].dtype == 'object':
                        # Check if it's a date column
                        if 'date' in col.lower() and col != 'created_at':
                            try:
                                df_parquet[col] = pd.to_datetime(df_parquet[col])
                            except:
                                pass
                
                df_parquet.to_parquet(f"{output_dir}/{dataset_name}.parquet", index=False)
            
            if 'json' in formats:
                # Convert datetime objects to strings for JSON serialization
                df_json = df.copy()
                for col in df_json.columns:
                    if df_json[col].dtype == 'object':
                        df_json[col] = df_json[col].astype(str)
                df_json.to_json(f"{output_dir}/{dataset_name}.json", orient='records', indent=2)
        
        print(f"\nData generation completed!")
        print(f"Output directory: {output_dir}")
        print(f"Formats generated: {', '.join(formats)}")
        print(f"\nDataset summary:")
        for name, df in datasets.items():
            print(f"  - {name}: {len(df):,} records")
        
        # Generate data quality report
        self.generate_data_quality_report(datasets, output_dir)
        
        return datasets
    
    def generate_stage_transitions(self, loans_df: pd.DataFrame) -> pd.DataFrame:
        """Generate IFRS9 stage transition data for historical analysis."""
        transitions = []
        
        # Simulate stage transitions over time
        for _, loan in loans_df.sample(min(1000, len(loans_df))).iterrows():
            loan_id = loan['loan_id']
            
            # Create monthly transitions for the past 12 months
            for month_offset in range(12):
                transition_date = datetime.now() - timedelta(days=30 * month_offset)
                
                # Simulate stage transitions based on credit quality
                if loan['provision_stage'] == 3:
                    # Stage 3 loans may have been Stage 2 or 1 previously
                    prev_stage = random.choices([1, 2, 3], weights=[0.1, 0.4, 0.5])[0]
                elif loan['provision_stage'] == 2:
                    prev_stage = random.choices([1, 2], weights=[0.3, 0.7])[0]
                else:
                    prev_stage = 1
                
                transition = {
                    'loan_id': loan_id,
                    'transition_date': transition_date.date(),
                    'from_stage': prev_stage,
                    'to_stage': loan['provision_stage'],
                    'trigger_reason': self._get_transition_reason(prev_stage, loan['provision_stage']),
                    'created_at': datetime.now()
                }
                transitions.append(transition)
        
        return pd.DataFrame(transitions)
    
    def _get_transition_reason(self, from_stage: int, to_stage: int) -> str:
        """Get reason for stage transition."""
        if from_stage < to_stage:
            reasons = [
                "Significant increase in credit risk",
                "Days past due threshold exceeded",
                "Deterioration in credit score",
                "Economic downturn impact",
                "Industry sector stress"
            ]
        elif from_stage > to_stage:
            reasons = [
                "Improved payment behavior",
                "Credit score improvement",
                "Economic recovery",
                "Restructuring agreement",
                "Additional collateral provided"
            ]
        else:
            reasons = ["No stage change", "Regular review"]
        
        return random.choice(reasons)
    
    def generate_data_quality_report(self, datasets: Dict[str, pd.DataFrame], output_dir: str):
        """Generate data quality report for the synthetic datasets."""
        report = {
            'generation_date': datetime.now().isoformat(),
            'datasets': {},
            'summary_statistics': {}
        }
        
        for name, df in datasets.items():
            # Basic statistics
            report['datasets'][name] = {
                'record_count': len(df),
                'column_count': len(df.columns),
                'missing_values': df.isnull().sum().to_dict(),
                'data_types': df.dtypes.astype(str).to_dict()
            }
            
            # Specific validation for loan portfolio
            if name == 'loan_portfolio':
                stage_distribution = df['provision_stage'].value_counts().to_dict()
                region_distribution = df['region'].value_counts().to_dict()
                product_distribution = df['producto_tipo'].value_counts().to_dict()
                
                report['summary_statistics'][name] = {
                    'stage_distribution': stage_distribution,
                    'region_distribution': region_distribution,
                    'product_distribution': product_distribution,
                    'avg_loan_amount': float(df['loan_amount'].mean()),
                    'avg_credit_score': float(df['credit_score'].mean()),
                    'total_portfolio_value': float(df['current_balance'].sum()),
                    'total_ecl': float(df['ecl'].sum())
                }
        
        # Save report
        import json
        with open(f"{output_dir}/data_quality_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Data quality report saved to {output_dir}/data_quality_report.json")


if __name__ == "__main__":
    # Generate enhanced synthetic data
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Generate IFRS9 synthetic data')
    parser.add_argument('--loans', type=int, default=50000, help='Number of loans to generate')
    parser.add_argument('--output', type=str, default='data/raw', help='Output directory')
    parser.add_argument('--formats', nargs='+', choices=['csv', 'parquet', 'json'], 
                       default=['csv', 'parquet'], help='Output formats')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch-size', type=int, default=10000, help='Batch size for processing')
    
    args = parser.parse_args()
    
    print("🏦 IFRS9 Enhanced Data Generator")
    print("=" * 50)
    print(f"Loans to generate: {args.loans:,}")
    print(f"Output directory: {args.output}")
    print(f"Formats: {', '.join(args.formats)}")
    print(f"Random seed: {args.seed}")
    print("=" * 50)
    
    # Generate synthetic data
    generator = DataGenerator(seed=args.seed)
    datasets = generator.save_data(
        output_dir=args.output,
        n_loans=args.loans,
        formats=args.formats,
        batch_size=args.batch_size
    )
    
    print("\n✅ Data generation completed successfully!")
    print(f"Generated datasets available in: {args.output}")
    
    # Show sample records
    if 'loan_portfolio' in datasets:
        sample_loans = datasets['loan_portfolio'].head(3)
        print("\n📊 Sample loan records:")
        for col in ['loan_id', 'loan_amount', 'credit_score', 'provision_stage', 'region', 'producto_tipo']:
            print(f"  {col}: {sample_loans[col].tolist()}")
    
    # Integration with GCP (if configured)
    try:
        from .gcp_integrations import get_gcp_config, initialize_ifrs9_gcp
        
        config = get_gcp_config()
        if config.get('project_id'):
            print("\n☁️  GCP Integration detected...")
            gcp = initialize_ifrs9_gcp(config['project_id'], config.get('credentials_path'))
            
            print("Setting up IFRS9 GCP infrastructure...")
            if gcp.setup_ifrs9_infrastructure():
                print("✅ GCP infrastructure ready")
                
                print("Uploading data to GCP...")
                if gcp.upload_ifrs9_data(datasets):
                    print("✅ Data uploaded to GCP successfully")
                else:
                    print("❌ Failed to upload data to GCP")
            else:
                print("❌ Failed to setup GCP infrastructure")
        else:
            print("\n💡 To enable GCP integration, set GCP_PROJECT_ID environment variable")
    
    except ImportError:
        print("\n💡 GCP integration module not available")