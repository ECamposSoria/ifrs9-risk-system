# IFRS9 Risk System - Essential Commands

## Development Setup
```bash
make setup              # Initial project setup (copy .env, create directories, build images)
make init-airflow       # Initialize Airflow database and create admin user
make up                 # Start all services
make show-ports         # Display the ports being used by each service
```

## Development Workflow
```bash
make dev                # Start complete development environment
make validate-test-env  # Validate testing environment setup
make ps                 # Show running containers status
```

## Testing Commands
```bash
make test               # Run all tests with coverage
make test-basic         # Run basic tests without coverage
make test-rules         # Run rules engine tests specifically
make test-parallel      # Run tests in parallel for faster execution
make test-watch         # Run tests in watch mode
```

## Code Quality
```bash
make lint               # Run linting checks (flake8, mypy)
make format             # Format code with black
make validate           # Validate configuration files and test environment
```

## Data Operations
```bash
make generate-data      # Generate synthetic IFRS9 data
make run-pipeline       # Run the IFRS9 pipeline locally
```

## Service Management
```bash
make up                 # Start all services
make down               # Stop all services
make restart            # Restart all services
make logs               # Show logs for all services
make logs-service SERVICE=jupyter  # Show logs for specific service
```

## Container Access
```bash
make exec-jupyter       # Open Jupyter terminal
make exec-airflow       # Open Airflow terminal
make exec-spark         # Open Spark master terminal
```

## Cleanup
```bash
make clean              # Clean temporary files and caches
make clean-data         # Clean all data files (warning: destructive!)
make reset              # Reset everything (stop services and clean)
```

## Service URLs (after running `make show-ports`)
- **Jupyter Lab**: http://localhost:[dynamic_port] (password: ifrs9)
- **Airflow UI**: http://localhost:[dynamic_port] (user: airflow, password: airflow) 
- **Spark UI**: http://localhost:[dynamic_port]

## Important Notes
- All Docker files use IFRS9-specific names to avoid conflicts
- Uses dynamic ports via environment variables
- Dependencies optimized to resolve PySpark/Airflow conflicts
- System runs on Linux environment