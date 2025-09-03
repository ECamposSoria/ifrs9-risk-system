# IFRS9 Risk System - Code Style and Conventions

## Python Code Style
- **Formatter**: Black with 100 character line length
- **Linter**: Flake8 with max-line-length=100, ignoring E203,W503
- **Type Checking**: MyPy with --ignore-missing-imports
- **Type Hints**: Encouraged throughout codebase
- **Docstrings**: Comprehensive docstrings required

## File Organization
```
ifrs9-risk-system/
├── src/                   # Main source code
│   ├── rules_engine.py    # IFRS9 PySpark engine  
│   ├── enhanced_ml_models.py # Advanced ML models
│   ├── ai_explanations.py # AI explanation engine
│   ├── generate_data.py   # Synthetic data generator
│   └── validation.py      # Data validation
├── tests/                 # Unit tests
├── dags/                  # Airflow DAGs
├── notebooks/             # Jupyter analysis notebooks
├── docker/                # Docker configurations
└── docs/                  # Documentation
```

## Naming Conventions
- **Classes**: PascalCase (e.g., IFRS9RulesEngine, OptimizedMLPipeline)
- **Functions**: snake_case (e.g., create_ifrs9_ml_pipeline)
- **Variables**: snake_case
- **Constants**: UPPER_SNAKE_CASE
- **Files**: snake_case.py

## Testing Standards
- **Framework**: pytest
- **Coverage**: pytest-cov with HTML reporting
- **Test Files**: test_*.py pattern in tests/ directory
- **Coverage Target**: Aim for high coverage on business logic

## Docker Conventions
- **Image Names**: ifrs9-specific prefixes to avoid conflicts
- **Compose File**: docker-compose.ifrs9.yml (renamed from default)
- **Dockerfiles**: Named with ifrs9 prefix (e.g., Dockerfile.ifrs9-spark)

## Documentation Standards
- **README**: Comprehensive with quickstart guide
- **Architecture**: Detailed technical documentation in docs/
- **API**: Auto-generated OpenAPI specifications
- **Comments**: Focus on why, not what

## Dependency Management
- **Versions**: Carefully selected to avoid PySpark/Airflow conflicts
- **PyArrow**: >=6.0.0,<7.0.0 (compatible with older dependencies)
- **PySpark**: >=3.4.0,<3.5.0
- **Google Cloud**: Versions under 3.0 to prevent BigQuery conflicts
- **Airflow**: Compatible with 2.6-2.8 range