# IFRS9 Risk System - User Guide

## Introduction

The IFRS9 Risk System provides comprehensive credit risk management and regulatory reporting capabilities. This guide helps business users, risk managers, and analysts effectively use the system's features for IFRS9 compliance and credit risk analysis.

## Getting Started

### Accessing the System

**Dashboard URL**: `https://ifrs9-dashboards.yourbank.com`
**API Documentation**: `https://ifrs9-api.yourbank.com/docs`

### User Roles and Permissions

| Role | Access Level | Capabilities |
|------|-------------|-------------|
| **Executive** | View-only | Executive summary, regulatory reports |
| **Risk Manager** | Full access | All dashboards, alert configuration |
| **Analyst** | Analysis tools | Data exploration, custom reports |
| **Data Team** | Technical access | Data quality monitoring, system health |

### Initial Setup

1. **Contact IT**: Request access to the IFRS9 Risk System
2. **Receive Credentials**: Get login credentials and MFA setup
3. **Access Training**: Complete mandatory IFRS9 system training
4. **Dashboard Setup**: Configure personal dashboard preferences

## Dashboard Navigation

### Executive Summary Dashboard

**Purpose**: High-level portfolio overview for senior management

**Key Metrics**:
- **Total Portfolio Value**: Current outstanding loan balance
- **Total Provisions**: IFRS9 Expected Credit Loss provisions
- **NPL Ratio**: Non-performing loans as percentage of total portfolio
- **Coverage Ratio**: Provisions divided by NPL exposure

**Visual Components**:

```
┌─────────────────────────────────────────────────────────────┐
│ IFRS9 Executive Summary Dashboard                           │
├─────────────────┬─────────────────┬─────────────────────────┤
│ Portfolio €2.1B │ Provisions €45M │ NPL Ratio 2.8%         │
├─────────────────┼─────────────────┼─────────────────────────┤
│              Stage Distribution    │ Risk Assessment Gauge   │
│              Stage 1: 85.2%        │      Moderate Risk      │
│              Stage 2: 12.0%        │        ████████         │
│              Stage 3: 2.8%         │         68/100          │
└─────────────────┴─────────────────┴─────────────────────────┘
```

**How to Use**:
1. **Quick Health Check**: Review the four scorecard metrics
2. **Trend Analysis**: Click on scorecards to view historical trends  
3. **Drill Down**: Click on stage distribution to see detailed breakdown
4. **Export**: Use the export button for board presentations

### Credit Quality Analysis Dashboard

**Purpose**: Detailed credit risk metrics and loan portfolio analysis

**Key Features**:
- **Credit Score Distribution**: Histogram of borrower credit scores
- **ECL by Score Band**: Average Expected Credit Loss by credit rating
- **Stage Heat Map**: Geographic and product-based risk distribution
- **PD vs LGD Scatter**: Relationship between probability of default and loss given default

**Navigation Tips**:
```
📊 Credit Quality Analysis
├── 📈 Credit Score Distribution
│   ├── Filter by: Region, Product Type
│   └── Drill down: Individual score bands
├── 💰 ECL Analysis
│   ├── View by: Score band, Product, Region
│   └── Export: CSV, Excel formats
└── 🗺️ Risk Heat Map
    ├── Hover: Detailed metrics tooltip
    └── Click: Focus on specific region/product
```

### Regional Risk Analysis Dashboard

**Purpose**: Geographic risk distribution and regional performance

**Map Visualization**:
- **Portfolio Exposure**: Bubble size represents loan volume
- **Risk Level**: Color intensity shows NPL ratio
- **Regional Comparison**: Side-by-side metrics table

**Regional Metrics Table**:
| Region | Exposure | Avg Score | NPL Ratio | ECL Coverage | Stage 3% |
|--------|----------|-----------|-----------|--------------|----------|
| Madrid | €650M    | 720       | 2.1%      | 1.8%         | 2.1%     |
| Catalonia | €480M | 695      | 3.2%      | 2.1%         | 3.2%     |
| Valencia | €320M   | 705       | 2.8%      | 1.9%         | 2.8%     |

### Product Performance Dashboard

**Purpose**: Product-specific risk analysis and profitability assessment

**Treemap Visualization**:
- **Size**: Current exposure amount
- **Color**: ECL rate (red = high risk, green = low risk)
- **Segments**: Product categories

**Product Categories**:
- **Mortgages**: Residential and commercial real estate loans
- **Personal Loans**: Unsecured consumer lending  
- **Credit Cards**: Revolving credit facilities
- **Business Loans**: SME and corporate lending
- **Auto Loans**: Vehicle financing

## Working with Data

### Filtering Data

**Date Range Filter**:
1. Click the date picker in the top toolbar
2. Select "Custom Range" for specific periods
3. Use predefined ranges: Last 30 days, Last quarter, YTD

**Multi-Select Filters**:
```
Region Filter:
☑️ Madrid
☑️ Catalonia  
☐ Valencia
☐ Andalusia
☐ Basque Country
```

**Advanced Filtering**:
- **Loan Amount**: Use slider for exposure ranges
- **Credit Score**: Set minimum/maximum thresholds
- **Stage Filter**: Include/exclude specific IFRS9 stages

### Interpreting IFRS9 Stages

**Stage 1 - Performing (Normal)**:
- **Criteria**: No significant increase in credit risk since origination
- **ECL Calculation**: 12-month expected credit losses
- **Typical Range**: 80-90% of portfolio
- **Provision Rate**: 0.1-0.5% typically

**Stage 2 - Underperforming (SICR)**:
- **Criteria**: Significant increase in credit risk detected
- **ECL Calculation**: Lifetime expected credit losses
- **Typical Range**: 8-15% of portfolio  
- **Provision Rate**: 2-8% typically

**Stage 3 - Non-Performing (Impaired)**:
- **Criteria**: Credit-impaired, objective evidence of default
- **ECL Calculation**: Lifetime expected credit losses
- **Typical Range**: 2-5% of portfolio
- **Provision Rate**: 20-60% typically

### Understanding Key Metrics

**NPL Ratio (Non-Performing Loans)**:
```
NPL Ratio = Stage 3 Loans / Total Portfolio
Target: < 3% (varies by market)
Alert Threshold: > 5%
```

**Coverage Ratio**:
```
Coverage Ratio = Total Provisions / NPL Exposure  
Target: > 50% (regulatory minimum varies)
Best Practice: > 70%
```

**ECL Rate**:
```
ECL Rate = Total ECL Provisions / Total Exposure
Industry Benchmark: 0.5-2.0% depending on portfolio
```

## Common Workflows

### Monthly Risk Review Process

**Week 1 - Data Validation**:
1. Access **Data Quality Dashboard**
2. Verify data completeness > 95%
3. Review and resolve any data quality alerts
4. Confirm all data sources are up-to-date

**Week 2 - Portfolio Analysis**:
1. Open **Executive Summary Dashboard**  
2. Compare current vs. previous month metrics
3. Identify significant changes or trends
4. Document findings in monthly report template

**Week 3 - Deep Dive Analysis**:
1. Use **Credit Quality Dashboard** for detailed analysis
2. Examine any deteriorating credit score trends
3. Review **Regional Analysis** for geographic concentrations
4. Analyze **Product Performance** for line-of-business insights

**Week 4 - Reporting and Actions**:
1. Generate executive summary report
2. Prepare board presentation materials
3. Configure alerts for next month monitoring
4. Document action items and follow-up

### Alert Management

**Setting Up Alerts**:
1. Navigate to **Settings > Alerts**
2. Choose alert type: Email, SMS, or Dashboard notification
3. Set thresholds:
   ```
   NPL Ratio Alert: > 4.0%
   Data Quality Alert: < 90%  
   Large Exposure Alert: > €10M single exposure
   ```

**Alert Response Process**:
1. **Immediate**: Verify alert is not false positive
2. **Within 1 Hour**: Investigate root cause
3. **Within 4 Hours**: Notify relevant stakeholders  
4. **Within 24 Hours**: Document findings and actions

### Custom Report Generation

**Using Looker Studio**:
1. Click **"Explore"** on any dashboard
2. Modify filters and date ranges
3. Add or remove visualization components
4. Save as custom dashboard: **File > Save As**

**Exporting Data**:
```
Available Formats:
├── 📊 PDF - Executive presentations
├── 📈 Excel - Detailed analysis
├── 📄 CSV - Raw data export
└── 🖼️ PNG - Charts and visualizations
```

**Scheduling Automated Reports**:
1. Open desired dashboard
2. Click **Share > Schedule Delivery**
3. Configure:
   - Recipients: email addresses
   - Frequency: daily, weekly, monthly
   - Format: PDF or Excel
   - Time: preferred delivery time

## Advanced Features

### Scenario Analysis

**Access**: Navigate to **Advanced Analytics > Scenario Modeling**

**Economic Scenario Types**:
- **Base Case**: Most likely economic scenario
- **Adverse**: Moderate economic downturn
- **Severely Adverse**: Significant recession scenario

**Running Scenarios**:
1. Select scenario parameters:
   ```
   GDP Growth: -2.5% (Adverse scenario)
   Unemployment: +3.0%
   Interest Rates: +1.5%
   House Prices: -15%
   ```
2. Click **"Run Analysis"**
3. Review impact on ECL provisions
4. Export results for stress testing reports

### Model Performance Monitoring

**Access**: **Analytics > Model Performance**

**Key Model Metrics**:
- **Accuracy**: Current vs. historical performance
- **Calibration**: Predicted vs. actual default rates  
- **Discrimination**: ROC curves and AUC scores
- **Stability**: Population stability index (PSI)

**Model Alert Thresholds**:
```
⚠️  Warning: Accuracy drop > 5%
🚨 Critical: Accuracy drop > 10%
📈 Info: Monthly performance report available
```

### Data Lineage and Audit

**Access**: **System > Data Lineage**

**Tracking Features**:
- **Source Systems**: Origin of each data element
- **Transformation Steps**: Processing pipeline details
- **Quality Checks**: Validation rules applied
- **User Access**: Who accessed what data when

**Audit Trail Example**:
```
2024-01-15 09:30:15 | user@bank.com | Accessed Portfolio Dashboard
2024-01-15 09:31:22 | user@bank.com | Applied Region Filter: Madrid
2024-01-15 09:32:45 | user@bank.com | Exported NPL Report (Excel)
2024-01-15 09:35:12 | SYSTEM | Automated data refresh completed
```

## Mobile Access

### Mobile Dashboard Features

**Optimized Views**:
- Responsive design for tablets and smartphones
- Touch-friendly navigation
- Essential metrics prominently displayed
- Simplified chart interactions

**Mobile-Specific Workflows**:
1. **Quick Health Check**: Swipe between key metric cards
2. **Alert Monitoring**: Push notifications for critical alerts  
3. **Approval Workflows**: Review and approve risk decisions
4. **Emergency Access**: Critical metrics available offline

## Integration with External Systems

### Excel Add-in

**Installation**: Download from internal software portal

**Features**:
- Direct data connection to IFRS9 system
- Real-time metric refresh
- Template-based reporting
- Automated chart generation

**Usage**:
```
Excel Ribbon > IFRS9 Add-in > Connect
1. Authenticate with system credentials
2. Select data source (Portfolio, Provisions, etc.)
3. Choose refresh frequency
4. Insert data into worksheet
```

### API Access for Analysts

**API Documentation**: `https://ifrs9-api.yourbank.com/docs`

**Common API Endpoints**:
```python
# Python example for analysts
import requests

# Get portfolio summary
response = requests.get(
    "https://ifrs9-api.yourbank.com/api/v1/portfolio/summary",
    headers={"Authorization": "Bearer YOUR_TOKEN"}
)

portfolio_data = response.json()
```

## Best Practices

### Dashboard Usage

**Performance Optimization**:
- Use date filters to limit data scope
- Apply regional/product filters when possible  
- Schedule large exports during off-peak hours
- Clear browser cache if dashboards load slowly

**Data Interpretation**:
- Always check data freshness timestamp
- Understand methodology behind calculated metrics
- Compare trends rather than point-in-time values
- Validate unusual values before taking action

### Regulatory Compliance

**Documentation Requirements**:
- Screenshot key dashboard views for audit trail
- Export supporting data for regulatory submissions  
- Maintain change log for model parameter updates
- Document business decisions based on system insights

**Quality Assurance**:
- Cross-validate critical metrics with source systems
- Review data quality scores before analysis
- Report discrepancies immediately to data team
- Participate in monthly data governance reviews

## Troubleshooting

### Common Issues

**Dashboard Not Loading**:
1. Clear browser cache and cookies
2. Try incognito/private browsing mode
3. Check internet connectivity
4. Contact IT if issue persists

**Data Appears Outdated**:
1. Check data freshness indicator on dashboard
2. Verify refresh schedule with data team
3. Confirm source system availability
4. Review any scheduled maintenance windows

**Export Failures**:
1. Reduce date range or apply more filters
2. Try different export format (PDF vs Excel)
3. Check available disk space
4. Contact support for large exports

**Performance Issues**:
1. Close unnecessary browser tabs
2. Apply more restrictive filters
3. Use dashboard during off-peak hours
4. Request dashboard optimization from IT team

### Getting Help

**Support Channels**:
- **Helpdesk**: extensions 5555 (urgent issues)
- **Email**: ifrs9-support@yourbank.com  
- **Training**: Monthly user training sessions
- **Documentation**: Internal wiki and video tutorials

**Escalation Process**:
1. **Level 1**: Local IT support for technical issues
2. **Level 2**: IFRS9 system administrators  
3. **Level 3**: Vendor support for complex issues
4. **Business**: Risk management team for functional questions

This user guide ensures all stakeholders can effectively leverage the IFRS9 Risk System for regulatory compliance and business insight generation.