#!/bin/bash
# Fix Test Environment Script for IFRS9 Risk System
# This script resolves PySpark testing environment issues

set -e  # Exit on error

echo "================================================"
echo "IFRS9 Risk System - Test Environment Fix Script"
echo "================================================"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Step 1: Rebuilding Jupyter container with dependency fixes...${NC}"
docker-compose -f docker-compose.ifrs9.yml build jupyter

echo -e "${YELLOW}Step 2: Restarting Jupyter container...${NC}"
docker-compose -f docker-compose.ifrs9.yml restart jupyter

echo -e "${YELLOW}Step 3: Waiting for container to be healthy...${NC}"
for i in {1..30}; do
    if docker exec jupyter python -c "import pyspark; print('OK')" 2>/dev/null; then
        echo -e "${GREEN}✓ Jupyter container is ready${NC}"
        break
    fi
    echo "Waiting... ($i/30)"
    sleep 2
done

echo -e "${YELLOW}Step 4: Validating PySpark installation...${NC}"
docker exec jupyter python -c "
import pyspark
print(f'PySpark version: {pyspark.__version__}')
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('TestValidation').getOrCreate()
print('✓ SparkSession created successfully')
spark.stop()
"

echo -e "${YELLOW}Step 5: Running quick test validation...${NC}"
docker exec jupyter python -c "
import sys
sys.path.insert(0, '/home/jovyan/src')
from rules_engine import IFRS9RulesEngine
print('✓ IFRS9RulesEngine imported successfully')
"

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}Test environment fix completed successfully!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "You can now run tests using:"
echo "  make test                # Run all tests"
echo "  make test-rules          # Run rules engine tests only"
echo "  make validate-test-env   # Validate test environment"
echo ""
echo "Note: Some test failures may still occur due to business logic issues,"
echo "but PySpark should now be working correctly."