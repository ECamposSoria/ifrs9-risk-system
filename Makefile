.PHONY: help setup up down restart logs clean test lint format build push deploy init-airflow show-ports monitoring-up monitoring-down loadtest-up loadtest-down analyze-codebase agents-readiness

# Variables
DOCKER_COMPOSE = docker-compose -f docker-compose.ifrs9.yml
PYTHON = python3
PROJECT_NAME = ifrs9-risk-system
ENV_FILE = .env

# Colors for output
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[1;33m
NC = \033[0m # No Color

help: ## Show this help message
	@echo "$(GREEN)IFRS9 Risk System - Makefile Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

setup: ## Initial project setup
	@echo "$(GREEN)Setting up IFRS9 Risk System...$(NC)"
	@cp -n .env.example $(ENV_FILE) 2>/dev/null || echo "$(YELLOW).env file already exists$(NC)"
	@echo "$(GREEN)Creating necessary directories...$(NC)"
	@mkdir -p data/raw data/processed logs notebooks/outputs models
	@echo "$(GREEN)Setting correct permissions...$(NC)"
	@echo "AIRFLOW_UID=$$(id -u)" >> $(ENV_FILE)
	@echo "$(GREEN)Building Docker images...$(NC)"
	@$(DOCKER_COMPOSE) build
	@echo "$(GREEN)Setup complete!$(NC)"

init-airflow: ## Initialize Airflow database and create admin user
	@echo "$(GREEN)Initializing Airflow...$(NC)"
	@$(DOCKER_COMPOSE) up airflow-init
	@echo "$(GREEN)Airflow initialized!$(NC)"

up: ## Start all services
	@echo "$(GREEN)Starting all services...$(NC)"
	@$(DOCKER_COMPOSE) up -d
	@$(MAKE) show-ports
	@echo "$(GREEN)All services started!$(NC)"

down: ## Stop all services
	@echo "$(YELLOW)Stopping all services...$(NC)"
	@$(DOCKER_COMPOSE) down
	@echo "$(GREEN)All services stopped!$(NC)"

restart: ## Restart all services
	@echo "$(YELLOW)Restarting all services...$(NC)"
	@$(MAKE) down
	@$(MAKE) up

logs: ## Show logs for all services
	@$(DOCKER_COMPOSE) logs -f

logs-service: ## Show logs for specific service (use SERVICE=<name>)
	@$(DOCKER_COMPOSE) logs -f $(SERVICE)

show-ports: ## Display the ports being used by each service
	@echo "$(GREEN)Service Ports:$(NC)"
	@echo "================================"
	@docker ps --format "table {{.Names}}\t{{.Ports}}" | grep -E "$(PROJECT_NAME)|PORTS" || echo "No services running"
	@echo "================================"
	@echo "$(YELLOW)Access URLs:$(NC)"
	@echo "Jupyter Lab: http://localhost:$$(docker port jupyter 8888 2>/dev/null | cut -d: -f2 || echo '8888')"
	@echo "Airflow UI:  http://localhost:$$(docker port airflow-webserver 8080 2>/dev/null | cut -d: -f2 || echo '8080')"
	@echo "Spark UI:    http://localhost:$$(docker port spark-master 8080 2>/dev/null | cut -d: -f2 || echo '8090')"
	@echo "================================"

monitoring-up: ## Start Prometheus/Grafana/Alertmanager monitoring stack
	@echo "$(GREEN)Starting monitoring stack...$(NC)"
	docker-compose -f deploy/monitoring/docker-compose.monitoring.yml up -d
	@echo "Prometheus: http://localhost:9090 | Grafana: http://localhost:3000 | Alertmanager: http://localhost:9093"

monitoring-down: ## Stop monitoring stack
	@echo "$(YELLOW)Stopping monitoring stack...$(NC)"
	docker-compose -f deploy/monitoring/docker-compose.monitoring.yml down

loadtest-up: ## Start Locust UI against IFRS9 API
	@echo "$(GREEN)Starting Locust load testing...$(NC)"
	LOCUST_HOST=$${LOCUST_HOST:-http://localhost:8080} docker-compose -f deploy/loadtest/docker-compose.loadtest.yml up -d
	@echo "Locust UI: http://localhost:8089"

loadtest-down: ## Stop Locust load testing
	@echo "$(YELLOW)Stopping Locust...$(NC)"
	docker-compose -f deploy/loadtest/docker-compose.loadtest.yml down

analyze-codebase: ## Run offline codebase analysis (optional Gemini enrichment with flags)
	@echo "$(GREEN)Analyzing codebase...$(NC)"
	$(PYTHON) src/analysis/gemini_codebase_analyzer.py --root . --out reports/codebase_analysis_report.json
	@echo "$(GREEN)Report: reports/codebase_analysis_report.json$(NC)"

agents-readiness: ## Probe all IFRS9 agent health endpoints and summarize
	@echo "$(GREEN)Checking agents readiness...$(NC)"
	$(PYTHON) scripts/agents_readiness.py

ps: ## Show running containers
	@$(DOCKER_COMPOSE) ps

exec-jupyter: ## Open Jupyter terminal
	@$(DOCKER_COMPOSE) exec jupyter /bin/bash

exec-airflow: ## Open Airflow terminal
	@$(DOCKER_COMPOSE) exec airflow-webserver /bin/bash

exec-spark: ## Open Spark master terminal
	@$(DOCKER_COMPOSE) exec spark-master /bin/bash

test: ## Run all tests
	@echo "$(GREEN)Running tests...$(NC)"
	@$(DOCKER_COMPOSE) exec spark-master python -c "import pyspark; print(f'PySpark version: {pyspark.__version__}')" || (echo "$(RED)PySpark not available, please rebuild container$(NC)" && exit 1)
	@$(DOCKER_COMPOSE) exec spark-master bash -c "cd /app && python -m pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html"

test-watch: ## Run tests in watch mode
	@$(DOCKER_COMPOSE) exec spark-master bash -c "cd /app && python -m pytest tests/ -v --watch"

test-basic: ## Run basic tests without coverage
	@echo "$(GREEN)Running basic tests...$(NC)"
	@$(DOCKER_COMPOSE) exec spark-master bash -c "cd /app && python -m pytest tests/test_basic.py -v"

test-rules: ## Run rules engine tests specifically
	@echo "$(GREEN)Running rules engine tests...$(NC)"
	@$(DOCKER_COMPOSE) exec spark-master bash -c "cd /app && python -m pytest tests/test_rules.py -v"

test-parallel: ## Run tests in parallel for faster execution
	@echo "$(GREEN)Running tests in parallel...$(NC)"
	@$(DOCKER_COMPOSE) exec spark-master bash -c "cd /app && python -m pytest tests/ -v --cov=src --cov-report=term-missing -n auto"

test-polars: ## Run Polars-specific integration tests
	@echo "$(GREEN)Running Polars integration tests...$(NC)"
	@$(DOCKER_COMPOSE) exec spark-master bash -c "cd /app && python -m pytest tests/test_polars_integration.py -v"

test-polars-performance: ## Run Polars performance benchmarks
	@echo "$(GREEN)Running Polars performance benchmarks...$(NC)"
	@$(DOCKER_COMPOSE) exec spark-master bash -c "cd /app && python -m pytest tests/test_polars_integration.py::TestPolarsPerformance -v -s"

test-docker-polars: ## Run Docker-optimized Polars tests and validations
	@echo "$(GREEN)Running Docker-optimized Polars test suite...$(NC)"
	@bash scripts/run_docker_polars_tests.sh
	@echo "$(GREEN)Docker-optimized Polars tests finished$(NC)"

lint: ## Run linting checks
	@echo "$(GREEN)Running linting checks...$(NC)"
	@$(DOCKER_COMPOSE) exec spark-master bash -c "cd /app && flake8 src/ tests/ --max-line-length=100 --ignore=E203,W503"
	@$(DOCKER_COMPOSE) exec spark-master bash -c "cd /app && mypy src/ --ignore-missing-imports"

format: ## Format code with black
	@echo "$(GREEN)Formatting code...$(NC)"
	@$(DOCKER_COMPOSE) exec spark-master bash -c "cd /app && black src/ tests/ --line-length=100"

generate-data: ## Generate synthetic IFRS9 data
	@echo "$(GREEN)Generating synthetic data...$(NC)"
	@$(DOCKER_COMPOSE) exec jupyter python src/generate_data.py

run-pipeline: ## Run the IFRS9 pipeline locally
	@echo "$(GREEN)Running IFRS9 pipeline...$(NC)"
	@$(DOCKER_COMPOSE) exec jupyter python src/main.py

clean: ## Clean temporary files and caches
	@echo "$(YELLOW)Cleaning temporary files...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name ".DS_Store" -delete 2>/dev/null || true
	@rm -rf .pytest_cache .mypy_cache .coverage htmlcov 2>/dev/null || true
	@echo "$(GREEN)Cleanup complete!$(NC)"

clean-data: ## Clean all data files (careful!)
	@echo "$(RED)WARNING: This will delete all data files!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo ""; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf data/raw/* data/processed/* 2>/dev/null || true; \
		echo "$(GREEN)Data cleaned!$(NC)"; \
	else \
		echo "$(YELLOW)Cancelled$(NC)"; \
	fi

reset: ## Reset everything (stop services and clean)
	@echo "$(RED)Resetting project...$(NC)"
	@$(MAKE) down
	@$(MAKE) clean
	@$(MAKE) clean-data
	@echo "$(GREEN)Reset complete!$(NC)"

build: ## Build Docker images
	@echo "$(GREEN)Building Docker images...$(NC)"
	@$(DOCKER_COMPOSE) build
	@echo "$(GREEN)Build complete!$(NC)"

pull: ## Pull latest base images
	@echo "$(GREEN)Pulling latest images...$(NC)"
	@$(DOCKER_COMPOSE) pull
	@echo "$(GREEN)Pull complete!$(NC)"

validate: ## Validate configuration files and test environment
	@echo "$(GREEN)Validating configuration...$(NC)"
	@$(DOCKER_COMPOSE) config > /dev/null && echo "$(GREEN)✓ docker-compose.yml valid$(NC)" || echo "$(RED)✗ docker-compose.yml invalid$(NC)"
	@$(PYTHON) -m py_compile src/*.py 2>/dev/null && echo "$(GREEN)✓ Python files valid$(NC)" || echo "$(YELLOW)⚠ Some Python files have syntax errors$(NC)"
	@echo "$(GREEN)Validation complete!$(NC)"

validate-test-env: ## Validate testing environment setup
	@echo "$(GREEN)Validating test environment...$(NC)"
	@echo "Checking Python version..."
	@$(DOCKER_COMPOSE) exec spark-master python --version
	@echo "Checking PySpark availability..."
	@$(DOCKER_COMPOSE) exec spark-master python -c "import pyspark; print(f'PySpark version: {pyspark.__version__}')"
	@echo "Checking Polars availability..."
	@$(DOCKER_COMPOSE) exec spark-master python -c "import polars as pl; print(f'Polars version: {pl.__version__}')"
	@echo "Checking pytest availability..."
	@$(DOCKER_COMPOSE) exec spark-master python -c "import pytest; print(f'Pytest version: {pytest.__version__}')"
	@echo "Checking coverage availability..."
	@$(DOCKER_COMPOSE) exec spark-master python -c "import coverage; print(f'Coverage version: {coverage.__version__}')"
	@echo "Checking test files..."
	@$(DOCKER_COMPOSE) exec spark-master find /app/tests/ -name "*.py" -type f
	@echo "$(GREEN)✓ Test environment validation complete$(NC)"

validate-polars: ## Validate Polars installation and performance
	@echo "$(GREEN)Validating Polars installation and integration...$(NC)"
	@echo "Testing Polars basic operations..."
	@$(DOCKER_COMPOSE) exec spark-master python -c "import polars as pl; df = pl.DataFrame({'x': [1,2,3]}); print(f'✅ Polars basic ops: {df.shape}')"
	@echo "Testing Polars-Pandas interoperability..."
	@$(DOCKER_COMPOSE) exec spark-master python -c "import polars as pl; import pandas as pd; df = pl.DataFrame({'x': [1,2,3]}); pdf = df.to_pandas(); print(f'✅ Polars->Pandas: {type(pdf)}')"
	@echo "Testing ML library compatibility..."
	@$(DOCKER_COMPOSE) exec spark-master python -c "import polars as pl; try: import xgboost; print('✅ XGBoost available'); except: print('⚠️ XGBoost not available')"
	@$(DOCKER_COMPOSE) exec spark-master python -c "import polars as pl; try: import lightgbm; print('✅ LightGBM available'); except: print('⚠️ LightGBM not available')"
	@echo "$(GREEN)✓ Polars validation complete$(NC)"

# GCP Deployment commands (for future use)
deploy-gcp: ## Deploy to Google Cloud Platform
	@echo "$(YELLOW)GCP deployment not yet implemented$(NC)"

# Development helpers
dev: ## Start development environment
	@$(MAKE) setup
	@$(MAKE) init-airflow
	@$(MAKE) up
	@echo "$(GREEN)Development environment ready!$(NC)"

stop: down ## Alias for down

status: ps ## Alias for ps
