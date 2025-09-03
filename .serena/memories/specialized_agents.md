# IFRS9 Risk System - Specialized Agents

## Available Specialized Agents in .claude/agents/

The project includes 8 specialized agents for different aspects of the IFRS9 system:

### Core Processing Agents
1. **ifrs9-rules-engine.md** - IFRS9 credit risk calculations, staging rules, ECL computations, compliance reports
2. **ifrs9-data-generator.md** - Generate synthetic financial data compliant with IFRS9 standards
3. **ifrs9-ml-models.md** - Train, evaluate, and maintain ML models for credit risk assessment
4. **ifrs9-validator.md** - Data quality validation and regulatory compliance checking

### Integration & Orchestration Agents  
5. **ifrs9-integrator.md** - Manage external system connections, BigQuery uploads, GCS operations
6. **ifrs9-orchestrator.md** - Coordinate entire IFRS9 pipeline workflow, monitor Airflow DAGs
7. **ifrs9-reporter.md** - Transform technical data into business-ready reports and visualizations

### Support Agents
8. **ifrs9-debugger.md** - Complex technical failures, performance bottlenecks, system-wide troubleshooting

## Agent Coordination Strategy

### Typical Workflow
1. **Data Generator** → creates synthetic data
2. **Validator** → validates data quality and compliance  
3. **Rules Engine** → processes IFRS9 calculations
4. **ML Models** → provides predictions and staging
5. **Integrator** → manages data flow to external systems
6. **Reporter** → generates dashboards and reports
7. **Orchestrator** → coordinates the entire pipeline
8. **Debugger** → troubleshoots any issues

### Agent Specializations
- **Rules Engine**: Core IFRS9 business logic and regulatory calculations
- **ML Models**: Random Forest staging, Gradient Boosting PD prediction
- **Data Generator**: Realistic loan portfolios, customer data, economic scenarios
- **Validator**: Schema validation, business rules, anomaly detection
- **Integrator**: BigQuery, GCS, PostgreSQL, external APIs  
- **Reporter**: Executive dashboards, regulatory reports, visualizations
- **Orchestrator**: Airflow DAG management, SLA monitoring, error recovery
- **Debugger**: PySpark optimization, memory issues, performance tuning

## Coordination Points
- All agents share common data formats and schemas
- Orchestrator manages dependencies and execution order
- Validator ensures data quality gates between stages
- Debugger provides cross-cutting troubleshooting support
- Reporter consumes outputs from all processing agents