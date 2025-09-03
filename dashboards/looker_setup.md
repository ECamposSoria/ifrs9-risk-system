# IFRS9 Risk System - Looker Studio Setup Guide

## Overview
This guide provides step-by-step instructions for setting up Looker Studio dashboards connected to your BigQuery IFRS9 data warehouse. The dashboards provide comprehensive credit risk monitoring, regulatory reporting, and business intelligence capabilities.

## Prerequisites

### 1. Google Cloud Platform Setup
- GCP Project with billing enabled
- BigQuery API enabled
- Cloud IAM permissions configured
- Service account with appropriate permissions

### 2. Data Infrastructure
- IFRS9 data loaded into BigQuery using the `gcp_integrations.py` module
- BigQuery views created using `bigquery_views.sql`
- Data validation completed

### 3. Access Requirements
- Looker Studio access (free Google account)
- BigQuery data viewer/editor permissions
- Dashboard sharing permissions (if required)

---

## Step 1: BigQuery Setup

### 1.1 Create Views in BigQuery

Execute the SQL scripts to create optimized views for dashboard consumption:

```bash
# Replace placeholders with your actual values
export PROJECT_ID="your-gcp-project-id"
export DATASET_ID="ifrs9_risk_system"

# Execute the BigQuery views script
bq query --use_legacy_sql=false --project_id=$PROJECT_ID < dashboards/bigquery_views.sql
```

### 1.2 Verify View Creation

```sql
-- Check that all views were created successfully
SELECT 
    table_name,
    table_type,
    creation_time
FROM `${PROJECT_ID}.${DATASET_ID}.INFORMATION_SCHEMA.TABLES`
WHERE table_name LIKE 'vw_%'
ORDER BY creation_time DESC;
```

Expected views:
- `vw_loan_portfolio_overview`
- `vw_stage_distribution`
- `vw_credit_quality_trends`
- `vw_regional_risk_analysis`
- `vw_product_performance`
- `vw_data_quality_metrics`
- `vw_executive_summary`
- `vw_ifrs9_regulatory_report`

---

## Step 2: Looker Studio Data Source Setup

### 2.1 Create BigQuery Connection

1. **Open Looker Studio**: Navigate to [https://lookerstudio.google.com](https://lookerstudio.google.com)

2. **Create Data Source**:
   - Click "Create" → "Data Source"
   - Select "BigQuery" connector
   - Choose your Google account with BigQuery access

3. **Configure Connection**:
   - **Project ID**: Enter your GCP project ID
   - **Dataset**: Select `ifrs9_risk_system` (or your dataset name)
   - **Table**: Select your first view (e.g., `vw_executive_summary`)

4. **Test Connection**:
   - Click "Connect" to test the connection
   - Verify that data loads correctly

### 2.2 Create Data Sources for Each View

Repeat the data source creation process for each view:

```
Data Source Names:
- IFRS9_Executive_Summary (vw_executive_summary)
- IFRS9_Portfolio_Overview (vw_loan_portfolio_overview)
- IFRS9_Stage_Distribution (vw_stage_distribution)
- IFRS9_Credit_Trends (vw_credit_quality_trends)
- IFRS9_Regional_Analysis (vw_regional_risk_analysis)
- IFRS9_Product_Performance (vw_product_performance)
- IFRS9_Regulatory_Report (vw_ifrs9_regulatory_report)
- IFRS9_Data_Quality (vw_data_quality_metrics)
```

---

## Step 3: Dashboard Creation

### 3.1 Executive Summary Dashboard

**Create New Report**:
1. Click "Create" → "Report"
2. Select "IFRS9_Executive_Summary" data source
3. Configure layout with 4 columns, multiple rows

**Key Components**:

```
Scorecards (Top Row):
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│ Total Portfolio │ Total Provisions│    NPL Ratio    │ Coverage Ratio  │
│    €X.XX M      │    €X.XX M      │     X.XX%       │     X.XX%       │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘

Charts (Second Row):
┌─────────────────────────────────────┬───────────────────────────────────────┐
│        Stage Distribution           │         Risk Assessment Gauge        │
│         (Pie Chart)                 │           (Gauge Chart)               │
└─────────────────────────────────────┴───────────────────────────────────────┘
```

**Configuration Details**:

1. **Total Portfolio Scorecard**:
   - Metric: `total_exposure`
   - Format: Currency (EUR)
   - Comparison: None

2. **NPL Ratio Scorecard**:
   - Metric: `npl_ratio`
   - Format: Percentage
   - Conditional formatting:
     - Green: < 2%
     - Yellow: 2-5%
     - Red: > 5%

3. **Stage Distribution Pie Chart**:
   - Dimension: Stage percentages
   - Metrics: `stage1_pct`, `stage2_pct`, `stage3_pct`
   - Colors: Green, Orange, Red

### 3.2 Credit Quality Analysis Dashboard

**Layout Design**:
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Credit Score Distribution                            │
│                            (Bar Chart)                                     │
├─────────────────────────────────┬───────────────────────────────────────────┤
│        ECL by Score Band        │         DPD Distribution                  │
│        (Column Chart)           │         (Histogram)                       │
├─────────────────────────────────┴───────────────────────────────────────────┤
│                      PD vs LGD Scatter Plot by Stage                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Regional Risk Analysis Dashboard

**Map Integration**:
1. **Geographic Chart**:
   - Chart type: Geo map
   - Geographic dimension: `region`
   - Metric: `total_exposure`
   - Map type: Spain (custom boundaries if available)

2. **Regional Comparison Table**:
   - Columns: Region, Exposure, NPL Ratio, ECL Coverage
   - Sorting: By exposure (descending)
   - Conditional formatting for risk metrics

### 3.4 Product Performance Dashboard

**Treemap Configuration**:
1. **Product Composition Treemap**:
   - Dimension: `product_type`
   - Size metric: `current_exposure`
   - Color metric: `ecl_rate`
   - Color scale: Red-Yellow-Green (reversed)

---

## Step 4: Advanced Features Setup

### 4.1 Interactive Filters

**Global Filters** (apply to all charts):
1. **Date Range Filter**:
   - Type: Date range
   - Default: Last 30 days
   - Control type: Date picker

2. **Region Filter**:
   - Type: Multi-select dropdown
   - Source: `region` dimension
   - Default: All regions

3. **Product Type Filter**:
   - Type: Multi-select dropdown
   - Source: `producto_tipo` dimension
   - Default: All products

4. **Stage Filter**:
   - Type: Multi-select dropdown
   - Options: Stage 1, Stage 2, Stage 3
   - Default: All stages

### 4.2 Calculated Fields

Create these calculated fields in your data sources:

```sql
-- Risk Level Indicator
CASE
  WHEN npl_ratio <= 2.0 THEN "Low Risk"
  WHEN npl_ratio <= 5.0 THEN "Moderate Risk"
  ELSE "High Risk"
END

-- ECL Coverage Status
CASE
  WHEN ecl_coverage_ratio >= 2.0 THEN "Well Covered"
  WHEN ecl_coverage_ratio >= 1.0 THEN "Adequately Covered"
  ELSE "Under Covered"
END

-- Credit Quality Score
CASE
  WHEN credit_score >= 800 THEN 5
  WHEN credit_score >= 700 THEN 4
  WHEN credit_score >= 600 THEN 3
  WHEN credit_score >= 500 THEN 2
  ELSE 1
END
```

### 4.3 Conditional Formatting Rules

**NPL Ratio Formatting**:
- Background color: Red if > 5%, Yellow if 2-5%, Green if < 2%
- Text color: White for better contrast

**ECL Coverage Formatting**:
- Background color: Green if >= 2%, Yellow if 1-2%, Red if < 1%

**Stage Distribution Formatting**:
- Stage 3 percentage: Red background if > 10%
- Stage 2 percentage: Orange background if > 30%

---

## Step 5: Sharing and Permissions

### 5.1 Dashboard Sharing

1. **Public Access** (if allowed):
   - Click "Share" button
   - Enable "Anyone with link can view"
   - Copy shareable link

2. **Specific Users**:
   - Add email addresses
   - Set permissions: View, Edit, or Manage
   - Send invitation emails

3. **Domain Sharing** (G Suite/Workspace):
   - Enable domain sharing
   - Set default permissions for domain users

### 5.2 Embedded Dashboards

For embedding in internal systems:

```html
<iframe 
  width="100%" 
  height="600"
  src="https://lookerstudio.google.com/embed/reporting/YOUR-REPORT-ID/page/YOUR-PAGE-ID"
  frameborder="0" 
  style="border:0" 
  allowfullscreen>
</iframe>
```

---

## Step 6: Automation and Scheduling

### 6.1 Automated Refresh

1. **Data Source Refresh**:
   - BigQuery data refreshes automatically
   - Views refresh when underlying tables update
   - No additional configuration needed

2. **Dashboard Cache**:
   - Looker Studio caches results for performance
   - Force refresh: Ctrl+Shift+R (Chrome)

### 6.2 Scheduled Reports

1. **Email Delivery**:
   - Click "Share" → "Schedule delivery"
   - Select recipients and frequency
   - Choose format: PDF or link

2. **Export Scheduling**:
   - Set up automated exports to Google Drive
   - Configure CSV/Excel exports for further analysis

---

## Step 7: Performance Optimization

### 7.1 Query Optimization

**Best Practices**:
- Use aggregated views instead of raw tables
- Implement partitioning on date fields
- Create clustered tables for large datasets
- Use LIMIT clauses in development

**Materialized Views** (for large datasets):
```sql
CREATE MATERIALIZED VIEW `project.dataset.mv_daily_summary`
PARTITION BY DATE(report_date)
AS
SELECT 
  DATE(CURRENT_TIMESTAMP()) as report_date,
  provision_stage,
  region,
  COUNT(*) as loan_count,
  SUM(current_balance) as total_exposure
FROM `project.dataset.loan_data`
GROUP BY provision_stage, region;
```

### 7.2 Dashboard Performance

**Optimization Tips**:
- Limit date ranges in filters
- Use summary tables for overview charts
- Implement drill-down functionality
- Cache frequently accessed data

---

## Step 8: Monitoring and Maintenance

### 8.1 Usage Monitoring

**Track Key Metrics**:
- Dashboard page views
- User engagement time
- Filter usage patterns
- Export frequency

**Access Analytics**:
- Google Analytics integration
- Looker Studio usage reports
- BigQuery query logs

### 8.2 Data Quality Alerts

**Set up notifications for**:
- Data freshness issues
- Unexpected value changes
- Schema modifications
- Query failures

**Implementation**:
```sql
-- Example alerting query
SELECT 
  'DATA_QUALITY_ALERT' as alert_type,
  table_name,
  quality_score,
  CURRENT_TIMESTAMP() as alert_time
FROM `project.dataset.vw_data_quality_metrics`
WHERE quality_score < 85;
```

---

## Troubleshooting Guide

### Common Issues and Solutions

**1. Connection Errors**
- Verify BigQuery permissions
- Check project ID and dataset names
- Ensure billing is enabled

**2. Data Not Loading**
- Verify view queries execute successfully
- Check data types compatibility
- Confirm row-level security settings

**3. Performance Issues**
- Review query complexity
- Check for large data scans
- Implement appropriate partitioning

**4. Visualization Problems**
- Verify metric aggregation settings
- Check field data types
- Review filter interactions

---

## Advanced Configuration

### 8.1 Custom Themes

Create consistent branding:

```json
{
  "theme": {
    "colors": {
      "primary": "#1976D2",
      "secondary": "#388E3C", 
      "accent": "#F57C00",
      "warning": "#D32F2F"
    },
    "fonts": {
      "primary": "Roboto",
      "headers": "Roboto Medium"
    }
  }
}
```

### 8.2 API Integration

For programmatic access:

```python
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Authenticate
credentials = service_account.Credentials.from_service_account_file(
    'path/to/credentials.json',
    scopes=['https://www.googleapis.com/auth/analytics.readonly']
)

# Build service
service = build('analyticsreporting', 'v4', credentials=credentials)

# Create custom reports
report_request = {
    'reportRequests': [{
        'viewId': 'VIEW_ID',
        'dateRanges': [{'startDate': '30daysAgo', 'endDate': 'today'}],
        'metrics': [{'expression': 'ga:sessions'}],
        'dimensions': [{'name': 'ga:country'}]
    }]
}
```

---

## Security Considerations

### 9.1 Data Access Control

**Row-Level Security**:
- Implement in BigQuery views
- Control data visibility by user
- Use parameterized filters

**Column-Level Security**:
- Restrict sensitive PII fields
- Implement field-level permissions
- Use data masking where appropriate

### 9.2 Audit Trail

**Track Access**:
- Enable BigQuery audit logs
- Monitor Looker Studio usage
- Log data exports and shares

---

## Support and Resources

### Documentation Links
- [Looker Studio Help Center](https://support.google.com/looker-studio)
- [BigQuery Documentation](https://cloud.google.com/bigquery/docs)
- [IFRS9 Implementation Guide](https://www.ifrs.org/issued-standards/list-of-standards/ifrs-9-financial-instruments/)

### Support Contacts
- Technical Issues: `data-team@yourbank.com`
- Business Questions: `risk-management@yourbank.com`
- Dashboard Requests: `analytics@yourbank.com`

### Training Resources
- Looker Studio Training: Internal LMS
- BigQuery Best Practices: Cloud Training
- IFRS9 Business Rules: Risk Training Portal

---

This comprehensive setup guide should enable your team to create professional-grade IFRS9 risk management dashboards that provide actionable insights for credit risk management and regulatory compliance.