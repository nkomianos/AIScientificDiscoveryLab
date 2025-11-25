# Kosmos Production Implementation Plan

This document outlines the remaining work to make Kosmos production-ready.

## Current Status

### Completed (Phase 1-3)
- **339 tests passing** (unit, integration, E2E)
- **10/10 paper requirements verified**
- **5/6 gaps fully implemented**
- Documentation updated and organized
- Smoke tests and E2E verification scripts

### Gap Summary
| Gap | Status | Component |
|-----|--------|-----------|
| Gap 0 | Complete | Context compression (20:1 ratio) |
| Gap 1 | Complete | State manager with JSON artifacts |
| Gap 2 | Complete | Orchestration (plan, review, delegation) |
| Gap 3 | Complete | Skill loader (566 skills) |
| Gap 4 | Partial | Python-first (mock execution) |
| Gap 5 | Complete | ScholarEval validation |

---

## Phase 4: Production Readiness

### 4.1 Gap 4 Completion: Sandboxed Code Execution

The execution environment is the remaining gap. Current implementation uses mock executors.

#### Requirements
1. **Docker-based Jupyter Kernel**
   - Isolated container per research cycle
   - Pre-built images with common scientific packages
   - Resource limits (CPU, memory, runtime)

2. **Package Management**
   - Automatic dependency detection from generated code
   - Pip/conda installation within sandbox
   - Version pinning for reproducibility

3. **Security**
   - Network isolation (no external access by default)
   - Filesystem sandboxing
   - Process timeout enforcement

#### Implementation Steps

```
Phase 4.1a: Docker Infrastructure (Week 1)
├── Create base Dockerfile with scientific stack
├── Implement container lifecycle management
├── Add resource limit enforcement
└── Create container pool for performance

Phase 4.1b: Jupyter Integration (Week 2)
├── Set up Jupyter kernel gateway
├── Implement code execution API
├── Add output capture and streaming
└── Handle execution errors gracefully

Phase 4.1c: Package Management (Week 3)
├── Build dependency resolver
├── Create package whitelist
├── Implement dynamic installation
└── Add version pinning

Phase 4.1d: Security Hardening (Week 4)
├── Network policy enforcement
├── Filesystem restrictions
├── Audit logging
└── Security testing
```

#### Docker Configuration
```dockerfile
# Dockerfile.kosmos-executor
FROM jupyter/scipy-notebook:latest

# Scientific packages
RUN pip install \
    pandas numpy scipy matplotlib seaborn \
    scikit-learn statsmodels \
    biopython scanpy anndata \
    rdkit-pypi chembl_webresource_client

# Security
USER jovyan
WORKDIR /home/jovyan/work
```

#### Integration Point
```python
# kosmos/execution/sandbox.py
class SandboxedExecutor:
    async def execute_code(
        self,
        code: str,
        timeout: int = 300,
        memory_limit: str = "2g"
    ) -> ExecutionResult:
        container = await self.get_container()
        try:
            result = await container.execute(code, timeout=timeout)
            return ExecutionResult(
                output=result.stdout,
                errors=result.stderr,
                execution_time=result.elapsed
            )
        finally:
            await container.cleanup()
```

### 4.2 CI/CD Pipeline Configuration

#### GitHub Actions Workflow
```yaml
# .github/workflows/ci.yml
name: Kosmos CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest pytest-asyncio pytest-cov

      - name: Run smoke tests
        run: python scripts/smoke_test.py

      - name: Run unit tests
        run: |
          pytest tests/unit/compression/ \
                 tests/unit/orchestration/ \
                 tests/unit/validation/ \
                 tests/unit/workflow/ \
                 tests/unit/agents/test_skill_loader.py \
                 tests/unit/world_model/test_artifacts.py \
                 --cov=kosmos -v

      - name: Run integration tests
        run: |
          pytest tests/integration/ tests/e2e/ \
                 --cov=kosmos --cov-append -v

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

#### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.3.0
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort

  - repo: local
    hooks:
      - id: smoke-test
        name: Smoke Test
        entry: python scripts/smoke_test.py
        language: python
        pass_filenames: false
        always_run: true
```

### 4.3 Monitoring Setup

#### Metrics Collection
```python
# kosmos/monitoring/metrics.py
from dataclasses import dataclass
from datetime import datetime

@dataclass
class WorkflowMetrics:
    workflow_id: str
    start_time: datetime
    end_time: datetime
    cycles_completed: int
    total_tasks: int
    tasks_completed: int
    findings_generated: int
    findings_validated: int
    validation_rate: float
    total_execution_time: float
    average_cycle_time: float

class MetricsCollector:
    def __init__(self, backend: str = "prometheus"):
        self.backend = backend
        self.metrics = []

    async def record_workflow(self, metrics: WorkflowMetrics):
        # Send to monitoring backend
        pass
```

#### Prometheus Metrics
```python
# kosmos/monitoring/prometheus.py
from prometheus_client import Counter, Histogram, Gauge

# Counters
cycles_total = Counter('kosmos_cycles_total', 'Total cycles executed')
tasks_total = Counter('kosmos_tasks_total', 'Total tasks executed')
findings_total = Counter('kosmos_findings_total', 'Total findings generated')

# Histograms
cycle_duration = Histogram('kosmos_cycle_duration_seconds', 'Cycle duration')
task_duration = Histogram('kosmos_task_duration_seconds', 'Task duration')

# Gauges
active_workflows = Gauge('kosmos_active_workflows', 'Currently running workflows')
validation_rate = Gauge('kosmos_validation_rate', 'Current validation rate')
```

### 4.4 Deployment Checklist

#### Pre-deployment
- [ ] All 339 tests passing
- [ ] E2E verification (scripts/verify_e2e.py) passing
- [ ] Smoke tests (scripts/smoke_test.py) passing
- [ ] Security scan completed
- [ ] Dependencies pinned in requirements.txt
- [ ] Docker images built and tested

#### Infrastructure
- [ ] Docker/Kubernetes cluster configured
- [ ] Container registry set up
- [ ] Secrets management (API keys)
- [ ] Network policies applied
- [ ] Resource quotas defined

#### Monitoring
- [ ] Prometheus metrics endpoint exposed
- [ ] Grafana dashboard configured
- [ ] Alert rules defined
- [ ] Log aggregation set up

#### Documentation
- [ ] API documentation generated
- [ ] Deployment runbook created
- [ ] Troubleshooting guide written
- [ ] Changelog updated

---

## Appendix: Quick Commands

### Run Tests
```bash
# Smoke tests (< 30s)
python scripts/smoke_test.py

# Unit tests
pytest tests/unit/compression/ tests/unit/orchestration/ \
       tests/unit/validation/ tests/unit/workflow/ -v

# E2E verification
python scripts/verify_e2e.py --cycles 5 --tasks 10

# Full test suite
pytest tests/ --ignore=tests/unit/domains/ --ignore=tests/unit/literature/ -v
```

### Demo Workflow
```python
import asyncio
from kosmos.workflow.research_loop import ResearchWorkflow

async def run():
    workflow = ResearchWorkflow(
        research_objective="Your research question here",
        artifacts_dir="./artifacts"
    )
    result = await workflow.run(num_cycles=5, tasks_per_cycle=10)
    report = await workflow.generate_report()
    print(report)

asyncio.run(run())
```

---

## Timeline Estimate

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Test Suite Overhaul | Complete |
| Phase 2 | Documentation Cleanup | Complete |
| Phase 3 | E2E Verification | Complete |
| Phase 4.1 | Gap 4 Docker Sandbox | Pending |
| Phase 4.2 | CI/CD Pipeline | Pending |
| Phase 4.3 | Monitoring Setup | Pending |
| Phase 4.4 | Production Deploy | Pending |

---

*Generated: 2025-11-25*
*Version: 0.2.0-alpha*
