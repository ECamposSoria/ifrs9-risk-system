# IFRS9 Risk System - Developer Guide

## Overview

The IFRS9 Risk System is a comprehensive credit risk management platform designed for local development with Visual Studio and optional GCP cloud integration. This guide covers the technical setup, development workflow, and system architecture.

## Local Development Setup

### Prerequisites

- **Docker Desktop** (with Docker Compose v2+)
- **Visual Studio** or **Visual Studio Code**
- **Python 3.9+** (for local testing)
- **Git** for version control
- **Windows Subsystem for Linux (WSL2)** (Windows users)

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ifrs9-risk-system
   ```

2. **Setup local environment**:
   ```bash
   # Copy local development template
   cp .env.local .env

   # Create necessary directories
   make setup
   ```

3. **Start services**:
   ```bash
   # Initialize Airflow (first time only)
   make init-airflow

   # Start all services
   make up
   ```

4. **Access services**:
   - **Airflow WebUI**: http://localhost:8080 (airflow/airflow)
   - **Jupyter Notebooks**: http://localhost:8888 (token: ifrs9)
   - **Spark Master UI**: http://localhost:8090
   - **PostgreSQL**: localhost:5432 (airflow/airflow)
   - **PgAdmin**: http://localhost:5050 (optional)

## Visual Studio Integration

### Python Environment Setup

1. **Create Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # OR
   venv\Scripts\activate     # Windows
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

3. **Configure Python Path**:
   ```json
   // .vscode/settings.json
   {
     "python.defaultInterpreterPath": "./venv/bin/python",
     "python.terminal.activateEnvironment": true,
     "python.linting.enabled": true,
     "python.linting.pylintEnabled": true,
     "python.formatting.provider": "black",
     "python.sortImports.args": ["--profile", "black"]
   }
   ```

### Development Tasks

**Common VS Code Tasks** (`.vscode/tasks.json`):
```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Run Tests",
      "type": "shell",
      "command": "pytest",
      "args": ["tests/", "-v", "--tb=short"],
      "group": "test"
    },
    {
      "label": "Start Docker Services",
      "type": "shell",
      "command": "make",
      "args": ["up"],
      "group": "build"
    },
    {
      "label": "Stop Docker Services",
      "type": "shell",
      "command": "make",
      "args": ["down"],
      "group": "build"
    }
  ]
}
```

### Debugging Configuration

**Launch Configuration** (`.vscode/launch.json`):
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Debug IFRS9 Rules Engine",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/src/rules_engine.py",
      "env": {
        "PYTHONPATH": "${workspaceFolder}/src",
        "DEBUG": "true"
      },
      "console": "integratedTerminal"
    },
    {
      "name": "Debug Tests",
      "type": "python",
      "request": "launch",
      "module": "pytest",
      "args": ["${file}", "-v"],
      "console": "integratedTerminal"
    }
  ]
}
```

## Architecture Overview

### Core Services

```
┌─────────────────────────────────────────────────────────────┐
│                    IFRS9 Risk System                       │
├─────────────────┬─────────────────┬─────────────────────────┤
│   PostgreSQL    │   Spark Cluster │      Airflow DAGs       │
│  (Data Storage) │ (Processing)    │   (Orchestration)       │
├─────────────────┼─────────────────┼─────────────────────────┤
│   Jupyter Lab   │   Local Python  │    Optional: GCP        │
│  (Notebooks)    │   (Development) │   (Cloud Integration)   │
└─────────────────┴─────────────────┴─────────────────────────┘
```

### Directory Structure

```
ifrs9-risk-system/
├── src/                          # Core Python modules
│   ├── rules_engine.py           # IFRS9 calculation engine
│   ├── ml_model.py               # Machine learning models
│   ├── validation.py             # Data validation
│   ├── gcp_integrations.py       # Optional GCP services
│   ├── ai_explanations.py        # Optional Vertex AI
│   └── security/                 # Security middleware
├── tests/                        # Test suite
│   ├── test_rules.py             # Rules engine tests
│   ├── conftest.py               # Test fixtures
│   └── integration/              # Integration tests
├── dags/                         # Airflow DAGs
│   └── ifrs9_pipeline.py         # Main orchestration DAG
├── notebooks/                    # Jupyter notebooks
├── data/                         # Data directories
│   ├── raw/                      # Input data
│   ├── processed/                # Processed data
│   └── models/                   # ML model artifacts
├── docker/                       # Docker configurations
├── config/                       # Configuration files
├── docs/                         # Documentation
└── scripts/                      # Utility scripts
```

## Development Workflow

### Local Development Mode

The system supports **offline-first development** with optional cloud integration:

**Local Mode Features**:
- PostgreSQL instead of BigQuery
- Local ML models instead of Vertex AI
- File-based storage instead of Cloud Storage
- Simplified authentication (disabled by default)

**Environment Configuration**:
```bash
# .env (copied from .env.local)
GCP_ENABLED=false
VERTEX_AI_ENABLED=false
AI_EXPLANATIONS_ENABLED=false
API_AUTH_DISABLED=true
```

### Testing Strategy

**Test Categories**:
1. **Unit Tests**: Individual component testing
2. **Integration Tests**: Service interaction testing
3. **End-to-End Tests**: Full pipeline testing

**Running Tests**:
```bash
# Run all tests
make test

# Run specific test categories
pytest tests/ -m "not slow"              # Skip slow tests
pytest tests/ -m "spark"                 # Only Spark tests
pytest tests/ -m "validation"            # Only validation tests
pytest tests/ -k "test_staging"          # Specific test pattern
```

**Test Configuration**:
```ini
# pytest.ini
[tool:pytest]
markers =
    spark: Spark-related tests
    validation: Data validation tests
    ml: Machine learning tests
    slow: Tests that take >30 seconds
```

### Code Quality

**Linting and Formatting**:
```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/ --profile black

# Lint code
pylint src/

# Type checking
mypy src/
```

**Pre-commit Hooks** (`.pre-commit-config.yaml`):
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.12.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
```

## IFRS9 Components

### Rules Engine

**Core Functionality**:
```python
# src/rules_engine.py
from src.rules_engine import IFRS9RulesEngine

engine = IFRS9RulesEngine(spark=spark_session)

# Process portfolio data
result_df = engine.process_portfolio(loan_data)

# Generate summary report
summary = engine.generate_summary_report(result_df)
```

**Key Methods**:
- `_apply_enhanced_staging_rules()`: IFRS9 stage classification
- `_calculate_enhanced_risk_parameters()`: PD, LGD, EAD calculation
- `process_portfolio()`: End-to-end processing
- `validate_calculations()`: Data quality validation

### Machine Learning Models

**Model Types**:
- **Random Forest**: Stage classification
- **Gradient Boosting**: PD prediction
- **XGBoost**: Enhanced risk modeling

**Usage Example**:
```python
from src.ml_model import CreditRiskClassifier

classifier = CreditRiskClassifier(model_type="random_forest")
classifier.train_stage_classifier(X_train, y_train)
predictions = classifier.predict_stage(X_test)
```

### Data Validation

**Validation Framework**:
```python
from src.validation import DataValidator

validator = DataValidator()
passed, errors = validator.validate_loan_portfolio(df)

# Business rule validation
results = validator.validate_business_rules(df)
```

## GCP Integration (Optional)

### Cloud Services Configuration

**Enabling GCP Integration**:
```bash
# .env
GCP_ENABLED=true
GCP_PROJECT_ID=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

**Supported Services**:
- **BigQuery**: Data warehousing
- **Cloud Storage**: File storage
- **Vertex AI**: ML model training
- **Cloud Functions**: Serverless execution

**Usage Pattern**:
```python
from src.gcp_integrations import GCPIntegration

if os.getenv('GCP_ENABLED', 'false').lower() == 'true':
    gcp = GCPIntegration(project_id=PROJECT_ID)
    gcp.upload_to_bigquery(df, table_id)
else:
    # Use local PostgreSQL fallback
    df.to_sql('loan_data', local_db_connection)
```

## Performance Optimization

### Spark Configuration

**Local Development Settings**:
```bash
# .env
SPARK_DRIVER_MEMORY=2g
SPARK_EXECUTOR_MEMORY=2g
SPARK_EXECUTOR_CORES=2
```

**Production Tuning**:
```python
# Optimize DataFrame operations
df.cache()  # Cache frequently accessed data
df.repartition(4)  # Optimize partitioning
df.persist(StorageLevel.MEMORY_AND_DISK)
```

### Data Processing

**Best Practices**:
- Use Parquet format for data storage
- Implement incremental processing
- Leverage Spark's lazy evaluation
- Monitor memory usage and GC

**Example Optimization**:
```python
# Efficient data processing
def process_loans_optimized(spark, input_path):
    return (
        spark.read.parquet(input_path)
        .filter(col("loan_amount") > 0)
        .repartition(4, "loan_id")
        .cache()
    )
```

## Monitoring and Debugging

### Logging Configuration

**Structured Logging**:
```python
import logging
import json

# Configure JSON logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Log with context
logger.info("Processing portfolio", extra={
    "portfolio_size": len(df),
    "processing_date": datetime.now().isoformat()
})
```

### Health Checks

**Service Health Monitoring**:
```bash
# Check service status
make status

# View service logs
docker-compose logs airflow-webserver
docker-compose logs spark-master

# Monitor resource usage
docker stats
```

### Debugging Tips

**Common Issues**:
1. **Memory Issues**: Increase Spark driver memory
2. **Connection Errors**: Check service dependencies
3. **Data Quality**: Review validation reports
4. **Performance**: Analyze Spark UI at localhost:8090

## Deployment

### Local Testing

**Full Pipeline Test**:
```bash
# Run complete validation
make test-full

# Validate Docker compose
make validate-compose

# Check data quality
python scripts/run_production_validation.py
```

### Production Considerations

**Checklist**:
- [ ] All tests passing
- [ ] GCP credentials configured
- [ ] Security settings enabled
- [ ] Monitoring configured
- [ ] Backup procedures tested

## Troubleshooting

### Common Issues

**Docker Services Won't Start**:
```bash
# Check system resources
docker system df
docker system prune  # Clean up if needed

# Reset environment
make clean
make setup
```

**Python Module Import Errors**:
```bash
# Check PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Verify dependencies
pip install -r requirements.txt
```

**Spark Job Failures**:
```bash
# Check Spark logs
docker-compose logs spark-master spark-worker

# Monitor Spark UI
open http://localhost:8090
```

### Getting Help

**Resources**:
- Check existing issues in the repository
- Review system logs: `make logs`
- Consult Airflow UI for DAG execution details
- Monitor Spark UI for job execution

This developer guide provides comprehensive technical documentation for working with the IFRS9 Risk System in a local Visual Studio environment.