#!/bin/bash

# IFRS9 Docker Environment Validation Runner
# Post-build validation execution script for comprehensive environment testing
# Created: 2025-01-14

set -e  # Exit on any error

# Configuration
VALIDATION_DIR="/home/eze/projects/ifrs9-risk-system/validation"
REPORTS_DIR="/home/eze/projects/ifrs9-risk-system/reports/validation"
LOG_FILE="${REPORTS_DIR}/validation_execution.log"
COMPOSE_FILE="docker-compose.ifrs9.yml"
MAX_WAIT_TIME=600  # 10 minutes maximum wait for services

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level="$1"
    shift
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $*" | tee -a "$LOG_FILE"
}

# Print banner
print_banner() {
    echo -e "${BLUE}"
    echo "=================================================================="
    echo "    IFRS9 Docker Environment Validation Suite"
    echo "    Comprehensive post-build validation framework"
    echo "=================================================================="
    echo -e "${NC}"
}

# Check if Docker Compose is available
check_docker_compose() {
    log "INFO" "Checking Docker Compose availability..."
    if ! command -v docker-compose &> /dev/null && ! command -v docker &> /dev/null; then
        log "ERROR" "Docker Compose not found. Please install Docker and Docker Compose."
        exit 1
    fi
    
    # Use docker compose (newer) or docker-compose (legacy)
    if docker compose version &> /dev/null; then
        DOCKER_COMPOSE_CMD="docker compose"
    else
        DOCKER_COMPOSE_CMD="docker-compose"
    fi
    
    log "INFO" "Using Docker Compose command: $DOCKER_COMPOSE_CMD"
}

# Wait for Docker services to be healthy
wait_for_services() {
    log "INFO" "Waiting for Docker services to be healthy..."
    local wait_time=0
    local services=("postgres" "spark-master" "spark-worker" "jupyter" "airflow-webserver" "airflow-scheduler")
    
    while [ $wait_time -lt $MAX_WAIT_TIME ]; do
        local all_healthy=true
        
        for service in "${services[@]}"; do
            local health_status=$($DOCKER_COMPOSE_CMD -f $COMPOSE_FILE ps --format json | jq -r ".[] | select(.Service==\"$service\") | .Health // \"unknown\"")
            
            if [[ "$health_status" != "healthy" && "$health_status" != "unknown" ]]; then
                all_healthy=false
                break
            fi
        done
        
        if $all_healthy; then
            log "INFO" "All services are healthy!"
            return 0
        fi
        
        log "INFO" "Waiting for services to become healthy... (${wait_time}s elapsed)"
        sleep 10
        wait_time=$((wait_time + 10))
    done
    
    log "WARN" "Some services may not be fully healthy after ${MAX_WAIT_TIME}s. Proceeding with validation..."
    return 0
}

# Check Python environment
check_python_environment() {
    log "INFO" "Checking Python environment..."
    
    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        log "ERROR" "Python3 not found. Please install Python 3.8+"
        exit 1
    fi
    
    # Check Python version
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    log "INFO" "Python version: $python_version"
    
    # Check required packages
    local required_packages=("docker" "pandas" "requests" "jinja2")
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import $package" &> /dev/null; then
            log "WARN" "Package '$package' not found. Installing..."
            pip3 install "$package" || log "ERROR" "Failed to install $package"
        fi
    done
}

# Create reports directory
setup_reports_directory() {
    log "INFO" "Setting up reports directory..."
    mkdir -p "$REPORTS_DIR"
    
    # Create timestamp-based subdirectory
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    CURRENT_REPORT_DIR="${REPORTS_DIR}/${TIMESTAMP}"
    mkdir -p "$CURRENT_REPORT_DIR"
    
    log "INFO" "Reports will be saved to: $CURRENT_REPORT_DIR"
}

# Run validation suite
run_validation_suite() {
    log "INFO" "Starting IFRS9 Docker validation suite..."
    
    # Change to validation directory
    cd "$VALIDATION_DIR"
    
    # Run master validation orchestrator
    log "INFO" "Executing master validation orchestrator..."
    
    python3 -c "
import sys
import os
sys.path.append('$VALIDATION_DIR')
sys.path.append('/home/eze/projects/ifrs9-risk-system/src')

from master_validation_orchestrator import MasterValidationOrchestrator
import json

# Initialize orchestrator
orchestrator = MasterValidationOrchestrator()

# Run all validations
print('Starting comprehensive validation...')
results = orchestrator.run_all_validations()

# Save results
results_file = '$CURRENT_REPORT_DIR/validation_results.json'
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2, default=str)

# Generate text report
text_report = orchestrator.generate_text_report(results)
with open('$CURRENT_REPORT_DIR/validation_report.txt', 'w') as f:
    f.write(text_report)

# Generate HTML report
html_report = orchestrator.generate_html_report(results)
with open('$CURRENT_REPORT_DIR/validation_report.html', 'w') as f:
    f.write(html_report)

print(f'Validation complete! Reports saved to $CURRENT_REPORT_DIR')
print(f'Overall Status: {\"PASSED\" if results[\"overall_status\"] == \"success\" else \"FAILED\"}')

# Print summary
print('\n=== VALIDATION SUMMARY ===')
for agent_name, agent_result in results.get('agent_results', {}).items():
    status = agent_result.get('status', 'unknown').upper()
    print(f'{agent_name}: {status}')

sys.exit(0 if results['overall_status'] == 'success' else 1)
"
    
    local validation_exit_code=$?
    return $validation_exit_code
}

# Display results summary
display_results_summary() {
    local exit_code=$1
    
    echo -e "\n${BLUE}=================================================================="
    echo "    VALIDATION RESULTS SUMMARY"
    echo -e "==================================================================${NC}"
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✅ VALIDATION PASSED${NC}"
        echo -e "${GREEN}All Docker environment validations completed successfully!${NC}"
    else
        echo -e "${RED}❌ VALIDATION FAILED${NC}"
        echo -e "${RED}Some validations failed. Please review the detailed reports.${NC}"
    fi
    
    echo ""
    echo "📁 Reports Location: $CURRENT_REPORT_DIR"
    echo "📄 Detailed Report: $CURRENT_REPORT_DIR/validation_report.txt"
    echo "🌐 HTML Report: $CURRENT_REPORT_DIR/validation_report.html"
    echo "📊 JSON Results: $CURRENT_REPORT_DIR/validation_results.json"
    echo ""
    
    if [ -f "$CURRENT_REPORT_DIR/validation_report.html" ]; then
        echo -e "${YELLOW}💡 Open the HTML report in your browser for interactive results:${NC}"
        echo "   file://$CURRENT_REPORT_DIR/validation_report.html"
    fi
}

# Cleanup function
cleanup() {
    log "INFO" "Cleaning up temporary files..."
    # Add any cleanup logic here if needed
}

# Main execution function
main() {
    # Set up signal handlers for cleanup
    trap cleanup EXIT
    
    print_banner
    
    # Create log file directory
    mkdir -p "$(dirname "$LOG_FILE")"
    
    log "INFO" "Starting IFRS9 Docker validation process..."
    
    # Pre-validation checks
    check_docker_compose
    check_python_environment
    setup_reports_directory
    
    # Wait for Docker services
    wait_for_services
    
    # Run validation suite
    if run_validation_suite; then
        display_results_summary 0
        log "INFO" "Validation completed successfully!"
        exit 0
    else
        display_results_summary 1
        log "ERROR" "Validation failed!"
        exit 1
    fi
}

# Script usage information
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "IFRS9 Docker Environment Validation Runner"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -q, --quiet    Suppress verbose output"
    echo "  -t, --timeout  Set service wait timeout in seconds (default: 600)"
    echo ""
    echo "Examples:"
    echo "  $0                    # Run full validation suite"
    echo "  $0 --timeout 300      # Run with 5-minute timeout"
    echo "  $0 --quiet           # Run with minimal output"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -q|--quiet)
            exec >/dev/null 2>&1
            shift
            ;;
        -t|--timeout)
            if [[ -n $2 && $2 =~ ^[0-9]+$ ]]; then
                MAX_WAIT_TIME=$2
                shift 2
            else
                echo "Error: --timeout requires a numeric value"
                exit 1
            fi
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Execute main function
main "$@"