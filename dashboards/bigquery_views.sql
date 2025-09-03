-- IFRS9 Risk System - BigQuery Views for Looker Studio Integration
-- These views provide optimized data structures for dashboard visualization

-- =============================================================================
-- 1. LOAN PORTFOLIO OVERVIEW VIEW
-- =============================================================================
CREATE OR REPLACE VIEW `${project_id}.${dataset_id}.vw_loan_portfolio_overview` AS
SELECT 
    loan_id,
    customer_id,
    loan_amount,
    current_balance,
    interest_rate,
    term_months,
    loan_type AS producto_tipo,
    origination_date,
    maturity_date,
    credit_score,
    dti_ratio,
    employment_length,
    region,
    sector,
    currency,
    
    -- IFRS9 Risk Metrics
    provision_stage,
    pd_12m,
    pd_lifetime,
    lgd,
    ead,
    ecl,
    days_past_due,
    risk_rating,
    historial_de_pagos,
    
    -- Calculated fields for dashboard
    CASE 
        WHEN provision_stage = 1 THEN 'Stage 1 - Normal'
        WHEN provision_stage = 2 THEN 'Stage 2 - Significant Increase'
        WHEN provision_stage = 3 THEN 'Stage 3 - Credit Impaired'
        ELSE 'Unknown'
    END AS stage_description,
    
    CASE 
        WHEN days_past_due = 0 THEN 'Current'
        WHEN days_past_due BETWEEN 1 AND 30 THEN '1-30 DPD'
        WHEN days_past_due BETWEEN 31 AND 90 THEN '31-90 DPD'
        WHEN days_past_due > 90 THEN '90+ DPD'
        ELSE 'Unknown'
    END AS dpd_bucket,
    
    CASE 
        WHEN credit_score >= 800 THEN 'Excellent (800+)'
        WHEN credit_score >= 700 THEN 'Good (700-799)'
        WHEN credit_score >= 600 THEN 'Fair (600-699)'
        WHEN credit_score >= 500 THEN 'Poor (500-599)'
        ELSE 'Very Poor (<500)'
    END AS credit_score_band,
    
    -- Portfolio metrics
    loan_amount - current_balance AS principal_paid,
    (loan_amount - current_balance) / loan_amount AS paydown_ratio,
    ecl / current_balance AS ecl_rate,
    
    -- Date calculations
    DATE_DIFF(CURRENT_DATE(), origination_date, MONTH) AS months_on_books,
    DATE_DIFF(maturity_date, CURRENT_DATE(), MONTH) AS months_to_maturity,
    
    created_at
FROM `${project_id}.${dataset_id}.loan_data`
WHERE current_balance > 0;

-- =============================================================================
-- 2. IFRS9 STAGE DISTRIBUTION VIEW
-- =============================================================================
CREATE OR REPLACE VIEW `${project_id}.${dataset_id}.vw_stage_distribution` AS
SELECT 
    calculation_date,
    provision_stage,
    region,
    producto_tipo AS product_type,
    sector,
    
    -- Aggregated metrics
    COUNT(*) AS loan_count,
    SUM(current_balance) AS total_exposure,
    SUM(ecl) AS total_ecl,
    AVG(pd_12m) AS avg_pd_12m,
    AVG(pd_lifetime) AS avg_pd_lifetime,
    AVG(lgd) AS avg_lgd,
    AVG(credit_score) AS avg_credit_score,
    
    -- ECL Coverage ratios
    SUM(ecl) / NULLIF(SUM(current_balance), 0) AS ecl_coverage_ratio,
    
    -- Portfolio quality indicators
    SUM(CASE WHEN days_past_due > 0 THEN 1 ELSE 0 END) AS delinquent_count,
    SUM(CASE WHEN days_past_due > 90 THEN 1 ELSE 0 END) AS npl_count,
    
    -- Risk-weighted metrics
    SUM(current_balance * pd_12m) / NULLIF(SUM(current_balance), 0) AS portfolio_weighted_pd,
    
    CURRENT_TIMESTAMP() AS last_updated
FROM `${project_id}.${dataset_id}.loan_data` 
WHERE current_balance > 0
GROUP BY calculation_date, provision_stage, region, producto_tipo, sector;

-- =============================================================================
-- 3. CREDIT QUALITY TRENDS VIEW
-- =============================================================================
CREATE OR REPLACE VIEW `${project_id}.${dataset_id}.vw_credit_quality_trends` AS
WITH monthly_snapshots AS (
    SELECT 
        DATE_TRUNC(origination_date, MONTH) AS reporting_month,
        provision_stage,
        COUNT(*) AS new_loans_count,
        SUM(loan_amount) AS new_loans_volume,
        AVG(credit_score) AS avg_credit_score,
        AVG(pd_12m) AS avg_pd,
        SUM(ecl) AS total_ecl
    FROM `${project_id}.${dataset_id}.loan_data`
    GROUP BY reporting_month, provision_stage
),
trend_calculations AS (
    SELECT 
        reporting_month,
        provision_stage,
        new_loans_count,
        new_loans_volume,
        avg_credit_score,
        avg_pd,
        total_ecl,
        
        -- Month-over-month changes
        LAG(new_loans_count) OVER (
            PARTITION BY provision_stage 
            ORDER BY reporting_month
        ) AS prev_month_count,
        
        LAG(avg_credit_score) OVER (
            PARTITION BY provision_stage 
            ORDER BY reporting_month
        ) AS prev_month_score,
        
        LAG(total_ecl) OVER (
            PARTITION BY provision_stage 
            ORDER BY reporting_month
        ) AS prev_month_ecl
    FROM monthly_snapshots
)
SELECT 
    reporting_month,
    provision_stage,
    new_loans_count,
    new_loans_volume,
    avg_credit_score,
    avg_pd,
    total_ecl,
    
    -- Growth rates
    SAFE_DIVIDE(
        (new_loans_count - prev_month_count), 
        prev_month_count
    ) * 100 AS loan_count_growth_pct,
    
    SAFE_DIVIDE(
        (avg_credit_score - prev_month_score), 
        prev_month_score
    ) * 100 AS credit_score_change_pct,
    
    SAFE_DIVIDE(
        (total_ecl - prev_month_ecl), 
        prev_month_ecl
    ) * 100 AS ecl_growth_pct,
    
    CURRENT_TIMESTAMP() AS last_updated
FROM trend_calculations
ORDER BY reporting_month DESC, provision_stage;

-- =============================================================================
-- 4. REGIONAL RISK ANALYSIS VIEW
-- =============================================================================
CREATE OR REPLACE VIEW `${project_id}.${dataset_id}.vw_regional_risk_analysis` AS
SELECT 
    region,
    
    -- Portfolio composition
    COUNT(*) AS total_loans,
    SUM(current_balance) AS total_exposure,
    AVG(loan_amount) AS avg_loan_amount,
    AVG(current_balance) AS avg_current_balance,
    
    -- Credit quality metrics
    AVG(credit_score) AS avg_credit_score,
    AVG(pd_12m) AS avg_pd_12m,
    AVG(lgd) AS avg_lgd,
    SUM(ecl) AS total_ecl,
    SUM(ecl) / NULLIF(SUM(current_balance), 0) AS ecl_coverage_ratio,
    
    -- Stage distribution
    COUNTIF(provision_stage = 1) AS stage1_count,
    COUNTIF(provision_stage = 2) AS stage2_count,
    COUNTIF(provision_stage = 3) AS stage3_count,
    
    SAFE_DIVIDE(COUNTIF(provision_stage = 1), COUNT(*)) * 100 AS stage1_pct,
    SAFE_DIVIDE(COUNTIF(provision_stage = 2), COUNT(*)) * 100 AS stage2_pct,
    SAFE_DIVIDE(COUNTIF(provision_stage = 3), COUNT(*)) * 100 AS stage3_pct,
    
    -- Delinquency metrics
    COUNTIF(days_past_due > 0) AS delinquent_loans,
    COUNTIF(days_past_due > 90) AS npl_loans,
    SAFE_DIVIDE(COUNTIF(days_past_due > 90), COUNT(*)) * 100 AS npl_ratio,
    
    -- Risk concentration
    MAX(current_balance) AS largest_exposure,
    STDDEV(current_balance) AS exposure_volatility,
    
    -- Economic indicators (join with macro data)
    m.gdp_growth,
    m.unemployment_rate,
    m.economic_cycle,
    
    CURRENT_TIMESTAMP() AS last_updated
FROM `${project_id}.${dataset_id}.loan_data` l
LEFT JOIN (
    SELECT DISTINCT
        gdp_growth,
        unemployment_rate,
        economic_cycle,
        ROW_NUMBER() OVER (ORDER BY date DESC) as rn
    FROM `${project_id}.${dataset_id}.macro_economic_data`
) m ON m.rn = 1
WHERE l.current_balance > 0
GROUP BY 
    region, 
    m.gdp_growth, 
    m.unemployment_rate, 
    m.economic_cycle
ORDER BY total_exposure DESC;

-- =============================================================================
-- 5. PRODUCT TYPE PERFORMANCE VIEW
-- =============================================================================
CREATE OR REPLACE VIEW `${project_id}.${dataset_id}.vw_product_performance` AS
SELECT 
    producto_tipo AS product_type,
    sector,
    
    -- Volume metrics
    COUNT(*) AS loan_count,
    SUM(loan_amount) AS total_originated,
    SUM(current_balance) AS current_exposure,
    AVG(loan_amount) AS avg_loan_size,
    
    -- Profitability proxies
    AVG(interest_rate) AS avg_interest_rate,
    SUM(loan_amount - current_balance) AS total_principal_collected,
    
    -- Risk metrics
    AVG(credit_score) AS avg_credit_score,
    AVG(pd_12m) AS avg_pd_12m,
    AVG(lgd) AS avg_lgd,
    SUM(ecl) AS total_ecl,
    SUM(ecl) / NULLIF(SUM(current_balance), 0) AS ecl_rate,
    
    -- Performance indicators
    SAFE_DIVIDE(COUNTIF(provision_stage = 3), COUNT(*)) * 100 AS default_rate,
    SAFE_DIVIDE(COUNTIF(days_past_due > 30), COUNT(*)) * 100 AS delinquency_rate,
    AVG(days_past_due) AS avg_days_past_due,
    
    -- Portfolio quality distribution
    SAFE_DIVIDE(COUNTIF(credit_score >= 700), COUNT(*)) * 100 AS prime_percentage,
    SAFE_DIVIDE(COUNTIF(credit_score < 600), COUNT(*)) * 100 AS subprime_percentage,
    
    -- Maturity profile
    AVG(term_months) AS avg_term_months,
    AVG(DATE_DIFF(maturity_date, CURRENT_DATE(), MONTH)) AS avg_months_to_maturity,
    
    CURRENT_TIMESTAMP() AS last_updated
FROM `${project_id}.${dataset_id}.loan_data`
WHERE current_balance > 0
GROUP BY producto_tipo, sector
ORDER BY current_exposure DESC;

-- =============================================================================
-- 6. DATA QUALITY MONITORING VIEW
-- =============================================================================
CREATE OR REPLACE VIEW `${project_id}.${dataset_id}.vw_data_quality_metrics` AS
WITH quality_checks AS (
    SELECT 
        'loan_data' AS table_name,
        COUNT(*) AS total_records,
        
        -- Completeness checks
        COUNTIF(loan_id IS NULL) AS missing_loan_id,
        COUNTIF(customer_id IS NULL) AS missing_customer_id,
        COUNTIF(loan_amount IS NULL OR loan_amount <= 0) AS invalid_loan_amount,
        COUNTIF(current_balance IS NULL OR current_balance < 0) AS invalid_balance,
        COUNTIF(credit_score IS NULL OR credit_score < 300 OR credit_score > 850) AS invalid_credit_score,
        COUNTIF(provision_stage IS NULL OR provision_stage NOT IN (1, 2, 3)) AS invalid_stage,
        COUNTIF(pd_12m IS NULL OR pd_12m < 0 OR pd_12m > 1) AS invalid_pd,
        COUNTIF(lgd IS NULL OR lgd < 0 OR lgd > 1) AS invalid_lgd,
        COUNTIF(ecl IS NULL OR ecl < 0) AS invalid_ecl,
        
        -- Business rule validations
        COUNTIF(current_balance > loan_amount) AS balance_exceeds_original,
        COUNTIF(provision_stage = 3 AND days_past_due = 0) AS stage3_no_dpd,
        COUNTIF(provision_stage = 1 AND days_past_due > 90) AS stage1_high_dpd,
        COUNTIF(maturity_date <= origination_date) AS invalid_dates,
        
        -- Data freshness
        MAX(created_at) AS latest_update,
        DATE_DIFF(CURRENT_TIMESTAMP(), MAX(created_at), DAY) AS days_since_update,
        
        CURRENT_TIMESTAMP() AS check_timestamp
    FROM `${project_id}.${dataset_id}.loan_data`
),
quality_summary AS (
    SELECT 
        table_name,
        total_records,
        
        -- Calculate quality scores
        SAFE_DIVIDE(
            (total_records - missing_loan_id - missing_customer_id - invalid_loan_amount - 
             invalid_balance - invalid_credit_score - invalid_stage - invalid_pd - 
             invalid_lgd - invalid_ecl), 
            total_records
        ) * 100 AS completeness_score,
        
        SAFE_DIVIDE(
            (total_records - balance_exceeds_original - stage3_no_dpd - 
             stage1_high_dpd - invalid_dates), 
            total_records
        ) * 100 AS consistency_score,
        
        CASE 
            WHEN days_since_update <= 1 THEN 100
            WHEN days_since_update <= 7 THEN 80
            WHEN days_since_update <= 30 THEN 60
            ELSE 0
        END AS freshness_score,
        
        latest_update,
        days_since_update,
        check_timestamp
    FROM quality_checks
)
SELECT 
    *,
    (completeness_score + consistency_score + freshness_score) / 3 AS overall_quality_score,
    
    CASE 
        WHEN (completeness_score + consistency_score + freshness_score) / 3 >= 95 THEN 'Excellent'
        WHEN (completeness_score + consistency_score + freshness_score) / 3 >= 85 THEN 'Good'
        WHEN (completeness_score + consistency_score + freshness_score) / 3 >= 70 THEN 'Fair'
        ELSE 'Poor'
    END AS quality_rating
FROM quality_summary;

-- =============================================================================
-- 7. EXECUTIVE SUMMARY VIEW
-- =============================================================================
CREATE OR REPLACE VIEW `${project_id}.${dataset_id}.vw_executive_summary` AS
WITH portfolio_summary AS (
    SELECT 
        COUNT(*) AS total_loans,
        SUM(current_balance) AS total_exposure,
        SUM(ecl) AS total_provisions,
        AVG(credit_score) AS avg_credit_score,
        
        -- Stage distribution
        COUNTIF(provision_stage = 1) AS stage1_loans,
        COUNTIF(provision_stage = 2) AS stage2_loans,
        COUNTIF(provision_stage = 3) AS stage3_loans,
        
        SUM(CASE WHEN provision_stage = 1 THEN current_balance ELSE 0 END) AS stage1_exposure,
        SUM(CASE WHEN provision_stage = 2 THEN current_balance ELSE 0 END) AS stage2_exposure,
        SUM(CASE WHEN provision_stage = 3 THEN current_balance ELSE 0 END) AS stage3_exposure,
        
        -- Key ratios
        SAFE_DIVIDE(COUNTIF(days_past_due > 90), COUNT(*)) * 100 AS npl_ratio,
        SAFE_DIVIDE(SUM(ecl), SUM(current_balance)) * 100 AS provision_coverage_ratio
    FROM `${project_id}.${dataset_id}.loan_data`
    WHERE current_balance > 0
)
SELECT 
    -- Portfolio size
    total_loans,
    total_exposure,
    total_provisions,
    avg_credit_score,
    
    -- Stage distribution (count)
    stage1_loans,
    stage2_loans,
    stage3_loans,
    SAFE_DIVIDE(stage1_loans, total_loans) * 100 AS stage1_pct,
    SAFE_DIVIDE(stage2_loans, total_loans) * 100 AS stage2_pct,
    SAFE_DIVIDE(stage3_loans, total_loans) * 100 AS stage3_pct,
    
    -- Stage distribution (exposure)
    stage1_exposure,
    stage2_exposure,
    stage3_exposure,
    SAFE_DIVIDE(stage1_exposure, total_exposure) * 100 AS stage1_exposure_pct,
    SAFE_DIVIDE(stage2_exposure, total_exposure) * 100 AS stage2_exposure_pct,
    SAFE_DIVIDE(stage3_exposure, total_exposure) * 100 AS stage3_exposure_pct,
    
    -- Key risk indicators
    npl_ratio,
    provision_coverage_ratio,
    
    -- Risk assessment
    CASE 
        WHEN npl_ratio <= 2.0 AND provision_coverage_ratio >= 1.0 THEN 'Low Risk'
        WHEN npl_ratio <= 5.0 AND provision_coverage_ratio >= 0.5 THEN 'Moderate Risk'
        ELSE 'High Risk'
    END AS portfolio_risk_assessment,
    
    CURRENT_TIMESTAMP() AS report_date
FROM portfolio_summary;

-- =============================================================================
-- 8. IFRS9 REGULATORY REPORTING VIEW
-- =============================================================================
CREATE OR REPLACE VIEW `${project_id}.${dataset_id}.vw_ifrs9_regulatory_report` AS
SELECT 
    DATE_TRUNC(CURRENT_DATE(), QUARTER) AS reporting_quarter,
    
    -- Stage 1 metrics
    COUNT(CASE WHEN provision_stage = 1 THEN 1 END) AS stage1_count,
    SUM(CASE WHEN provision_stage = 1 THEN current_balance ELSE 0 END) AS stage1_gross_carrying_amount,
    SUM(CASE WHEN provision_stage = 1 THEN ecl ELSE 0 END) AS stage1_ecl_provisions,
    AVG(CASE WHEN provision_stage = 1 THEN pd_12m END) AS stage1_avg_pd,
    
    -- Stage 2 metrics
    COUNT(CASE WHEN provision_stage = 2 THEN 1 END) AS stage2_count,
    SUM(CASE WHEN provision_stage = 2 THEN current_balance ELSE 0 END) AS stage2_gross_carrying_amount,
    SUM(CASE WHEN provision_stage = 2 THEN ecl ELSE 0 END) AS stage2_ecl_provisions,
    AVG(CASE WHEN provision_stage = 2 THEN pd_lifetime END) AS stage2_avg_pd,
    
    -- Stage 3 metrics
    COUNT(CASE WHEN provision_stage = 3 THEN 1 END) AS stage3_count,
    SUM(CASE WHEN provision_stage = 3 THEN current_balance ELSE 0 END) AS stage3_gross_carrying_amount,
    SUM(CASE WHEN provision_stage = 3 THEN ecl ELSE 0 END) AS stage3_ecl_provisions,
    
    -- Total metrics
    COUNT(*) AS total_count,
    SUM(current_balance) AS total_gross_carrying_amount,
    SUM(ecl) AS total_ecl_provisions,
    
    -- Coverage ratios
    SAFE_DIVIDE(
        SUM(CASE WHEN provision_stage = 1 THEN ecl ELSE 0 END),
        SUM(CASE WHEN provision_stage = 1 THEN current_balance ELSE 0 END)
    ) * 100 AS stage1_coverage_ratio,
    
    SAFE_DIVIDE(
        SUM(CASE WHEN provision_stage = 2 THEN ecl ELSE 0 END),
        SUM(CASE WHEN provision_stage = 2 THEN current_balance ELSE 0 END)
    ) * 100 AS stage2_coverage_ratio,
    
    SAFE_DIVIDE(
        SUM(CASE WHEN provision_stage = 3 THEN ecl ELSE 0 END),
        SUM(CASE WHEN provision_stage = 3 THEN current_balance ELSE 0 END)
    ) * 100 AS stage3_coverage_ratio,
    
    SAFE_DIVIDE(SUM(ecl), SUM(current_balance)) * 100 AS overall_coverage_ratio,
    
    CURRENT_TIMESTAMP() AS report_generation_time
FROM `${project_id}.${dataset_id}.loan_data`
WHERE current_balance > 0;

-- =============================================================================
-- MATERIALIZED VIEWS FOR PERFORMANCE (Optional - for large datasets)
-- =============================================================================

-- Uncomment these for production environments with large datasets:

/*
CREATE MATERIALIZED VIEW `${project_id}.${dataset_id}.mv_daily_portfolio_summary`
PARTITION BY DATE(report_date)
CLUSTER BY provision_stage, region
AS
SELECT 
    CURRENT_DATE() AS report_date,
    provision_stage,
    region,
    producto_tipo,
    COUNT(*) AS loan_count,
    SUM(current_balance) AS total_exposure,
    SUM(ecl) AS total_ecl,
    AVG(credit_score) AS avg_credit_score,
    AVG(pd_12m) AS avg_pd,
    CURRENT_TIMESTAMP() AS created_at
FROM `${project_id}.${dataset_id}.loan_data`
WHERE current_balance > 0
GROUP BY provision_stage, region, producto_tipo;

CREATE MATERIALIZED VIEW `${project_id}.${dataset_id}.mv_risk_metrics_snapshot`
PARTITION BY DATE(snapshot_date)
AS
SELECT 
    CURRENT_DATE() AS snapshot_date,
    loan_id,
    provision_stage,
    pd_12m,
    pd_lifetime,
    lgd,
    ead,
    ecl,
    days_past_due,
    current_balance,
    CURRENT_TIMESTAMP() AS created_at
FROM `${project_id}.${dataset_id}.loan_data`
WHERE current_balance > 0;
*/