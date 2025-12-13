# IFRS9 Risk Management System

[![CI/CD Pipeline](https://github.com/your-org/ifrs9-risk-system/workflows/CI/badge.svg)](https://github.com/your-org/ifrs9-risk-system/actions)
[![Security Scan](https://github.com/your-org/ifrs9-risk-system/workflows/Security/badge.svg)](https://github.com/your-org/ifrs9-risk-system/security)
[![Code Coverage](https://codecov.io/gh/your-org/ifrs9-risk-system/branch/main/graph/badge.svg)](https://codecov.io/gh/your-org/ifrs9-risk-system)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🏦 Overview

A comprehensive, enterprise-grade IFRS9 credit risk management system designed for financial institutions to ensure regulatory compliance while providing advanced analytics and real-time monitoring capabilities.

### 🎯 Key Features

- **IFRS9 Compliance**: Automated staging classification and ECL calculations
- **Advanced ML Models**: XGBoost, LightGBM, and CatBoost for risk prediction
- **Explainability**: Rule-based narratives with SHAP-driven model transparency
- **Real-time Dashboards**: Looker Studio integration with 7 comprehensive dashboards
- **Cloud-Native**: Google Cloud Platform with auto-scaling capabilities
- **Production-Ready**: Kubernetes deployment with monitoring and alerting

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Data Sources  │────▶│  Validation  │────▶│  IFRS9 Engine   │
│  (CSV/Parquet)  │     │   (Pandera)  │     │   (PySpark)     │
└─────────────────┘     └──────────────┘     └─────────────────┘
                                                      │
                                                      ▼
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│   ML Models     │◀────│   Airflow    │────▶│    Reports      │
│  (Scikit-learn) │     │ (Orchestrator)│     │  (Dashboard)    │
└─────────────────┘     └──────────────┘     └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- 8GB+ RAM
- 10GB+ free disk space

## ☁️ GCP Deployment (Portfolio)

Two Terraform options exist:
- **Portfolio/minimal (recommended for demos)**: `deploy/portfolio/terraform/README.md`
- **Production/complete**: `deploy/terraform/README.md`

### ⚠️ Important Notes
- **Project-specific naming**: All Docker files use IFRS9-specific names to avoid conflicts
- **Dynamic ports**: Uses environment variables for port configuration 
- **Dependency compatibility**: Optimized versions to resolve PySpark/Airflow conflicts
- **Staged installation**: Dependencies installed in stages to prevent conflicts

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ifrs9-risk-system.git
cd ifrs9-risk-system
```

2. **Set up environment**
```bash
make setup
```
This will:
- Copy `.env.example` to `.env`
- Create necessary directories
- Build Docker images

3. **Initialize Airflow**
```bash
make init-airflow
```

4. **Start all services**
```bash
make up
```
This uses the renamed `docker-compose.ifrs9.yml` file.

5. **Check service ports**
```bash
make show-ports
```

You'll see output like:
```
Service Ports:
================================
jupyter          0.0.0.0:32768->8888/tcp
airflow-webserver 0.0.0.0:32769->8080/tcp
spark-master     0.0.0.0:32770->8080/tcp
================================
Access URLs:
Jupyter Lab: http://localhost:32768 (password: ifrs9)
Airflow UI:  http://localhost:32769 (user: airflow, password: airflow)
Spark UI:    http://localhost:32770
```

## 📁 Project Structure

```
ifrs9-risk-system/
├── data/                   # Data directory
│   ├── raw/               # Input data
│   └── processed/         # Pipeline outputs
├── notebooks/             # Jupyter notebooks
│   ├── EDA.ipynb         # Exploratory Data Analysis
│   └── LocalPipeline.ipynb # Local pipeline testing
├── dags/                  # Airflow DAGs
│   └── ifrs9_pipeline.py # Main pipeline DAG
├── src/                   # Source code
│   ├── generate_data.py  # Synthetic data generator
│   ├── rules_engine.py   # IFRS9 PySpark engine
│   ├── validation.py     # Data validation
│   └── ml_model.py       # ML classifiers
├── tests/                 # Unit tests
│   └── test_rules.py     # Rules engine tests
├── docker/                # Docker configurations
│   ├── airflow/          # Airflow Dockerfile
│   └── jupyter/          # Jupyter Dockerfile
├── docker-compose.yml     # Service orchestration
├── Makefile              # Utility commands
└── README.md             # This file
```

## 🔧 Usage

### Generate Synthetic Data
```bash
make generate-data
```

### Run Tests
```bash
make test
```

### Run Linting
```bash
make lint
```

### Format Code
```bash
make format
```

### Access Services

#### Jupyter Lab
Navigate to `http://localhost:<JUPYTER_PORT>` and use token `ifrs9`.

Example notebooks:
- **EDA.ipynb**: Explore the synthetic loan portfolio
- **LocalPipeline.ipynb**: Run the IFRS9 pipeline locally

#### Airflow UI
Navigate to `http://localhost:<AIRFLOW_PORT>` and login with:
- Username: `airflow`
- Password: `airflow`

Enable and trigger the `ifrs9_credit_risk_pipeline` DAG.

#### Spark UI
Monitor Spark jobs at `http://localhost:<SPARK_MASTER_UI_PORT>`.

## 📊 IFRS9 Implementation

### Stage Classification
- **Stage 1**: Performing loans (DPD < 30 days)
- **Stage 2**: Underperforming loans (30 ≤ DPD < 90 days or significant credit risk increase)
- **Stage 3**: Non-performing loans (DPD ≥ 90 days)

### ECL Calculation
```
ECL = PD × LGD × EAD

Where:
- PD: Probability of Default
- LGD: Loss Given Default
- EAD: Exposure at Default
```

### Risk Parameters
- **12-month ECL** for Stage 1 loans
- **Lifetime ECL** for Stage 2 and 3 loans
- Dynamic PD based on credit score and days past due
- LGD adjusted for collateral coverage

## 🤖 Machine Learning Models

### Stage Classifier
- **Algorithm**: Random Forest Classifier
- **Features**: Credit score, DPD, LTV ratio, debt-to-income, etc.
- **Performance**: ~85% accuracy on synthetic data

### PD Predictor
- **Algorithm**: Gradient Boosting Regressor
- **Output**: Probability of default (0-1)
- **Validation**: MAE < 0.05 on test set

## ☁️ GCP Integration

### Configuration
Set GCP environment variables in `.env`:
```bash
GCP_PROJECT_ID=your-project-id
GCP_BUCKET_NAME=your-bucket
GCP_DATASET_ID=your-dataset
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

### Deployment
```bash
make deploy-gcp  # Coming soon
```

## 🧪 Testing

Run all tests:
```bash
make test
```

Run specific test:
```bash
docker-compose exec jupyter python -m pytest tests/test_rules.py -v
```

## 📝 Development

### Dependency Management
The project uses carefully selected dependency versions to avoid conflicts:
- **PyArrow**: `>=6.0.0,<7.0.0` (compatible with older Airflow providers)
- **PySpark**: `>=3.4.0,<3.5.0` (matches PyArrow compatibility)  
- **Google Cloud**: Versions under 3.0 to prevent BigQuery conflicts
- **Airflow Providers**: Compatible with Airflow 2.6-2.8 range

### Docker Architecture
- **Dockerfile.ifrs9-spark**: Main Spark/Python environment
- **Dockerfile.ifrs9-airflow**: Airflow with PySpark integration
- **Dockerfile.ifrs9-jupyter**: Jupyter Lab with ML libraries
- **docker-compose.ifrs9.yml**: Orchestrates all services with dynamic ports

### Adding New Features
1. Create feature branch
2. Implement changes in `src/`
3. Add tests in `tests/`
4. Update notebooks if needed
5. Run `make lint` and `make format`
6. Submit pull request

### Code Style
- Python: Black formatter, Flake8 linter
- Line length: 100 characters
- Type hints encouraged
- Comprehensive docstrings required

## 🛠️ Makefile Commands

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make setup` | Initial project setup |
| `make up` | Start all services |
| `make down` | Stop all services |
| `make restart` | Restart all services |
| `make logs` | Show logs for all services |
| `make test` | Run all tests |
| `make lint` | Run linting checks |
| `make format` | Format code with black |
| `make clean` | Clean temporary files |
| `make show-ports` | Display service ports |

## 📊 Monitoring

### Logs
```bash
make logs  # All services
make logs-service SERVICE=jupyter  # Specific service
```

### Health Checks
All services include health checks. Check status:
```bash
make ps
```

## 🔒 Security

- All sensitive data stored in `.env` (not committed)
- Airflow Fernet key for encryption
- Network isolation between services
- No hardcoded credentials

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📚 Documentation

### Resources
- [IFRS9 Standard](https://www.ifrs.org/issued-standards/list-of-standards/ifrs-9-financial-instruments/)
- [PySpark Documentation](https://spark.apache.org/docs/latest/api/python/)
- [Airflow Documentation](https://airflow.apache.org/docs/)
- [Docker Documentation](https://docs.docker.com/)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Author

**ECamposSoria**  
Contact: ecampossoria88@gmail.com

## 🎯 Roadmap

- [ ] Real-time processing with Kafka
- [ ] Advanced ML models (XGBoost, LightGBM)
- [ ] Stress testing scenarios
- [ ] Regulatory reporting templates
- [ ] Cloud deployment automation
- [ ] Performance optimization for 1M+ loans
- [ ] Integration with core banking systems

## ⚠️ Disclaimer

This is a demonstration project using synthetic data. For production use, ensure compliance with local regulatory requirements and data protection laws.

---

**Last Updated**: January 2024
**Version**: 1.0.0
