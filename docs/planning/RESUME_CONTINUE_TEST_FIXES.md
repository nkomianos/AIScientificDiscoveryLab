# RESUME: Continue Integration Test Fixes (57.4% → 90%)

**Created:** 2025-11-18
**For Use After:** `/plancompact` command
**Current Status:** 57.4% achieved (81/141 tests) → Target: >90% (127+ tests)

---

## 🚀 START HERE

You are resuming the **Kosmos AI Scientist 1-2 week deployment sprint** to complete integration test fixes.

**Completed:** Days 1-3 + Day 4 systematic fixes (5 categories)
**Current:** 81/141 tests passing (57.4%)
**Next:** Fix remaining 46 tests to reach >90% target (127+ tests)

---

## 📋 RESUME PROMPT

```
I'm resuming the Kosmos AI Scientist deployment sprint.

Completed: Days 1-3 + Day 4 systematic fixes (57.4% integration tests)
Status: Need to fix remaining 46 tests to reach 90% target

Please read @docs/planning/CHECKPOINT_DAY4_INTEGRATION_FIXES_PARTIAL.md for full context,
then proceed with fixing remaining test failures per
@docs/planning/RESUME_CONTINUE_TEST_FIXES.md
```

---

## 📊 CURRENT STATE

### Test Results Summary
- **PASSED:** 81/141 (57.4%)
- **FAILED:** 34 tests
- **ERRORS:** 15 tests
- **SKIPPED:** 11 tests (Phase 2/3 features)
- **Target:** 127+/141 (>90%)
- **Gap:** 46 tests

### Completed Fixes (5 Categories) ✅
1. Neo4j Protocol - conftest.py created
2. ResearchDirectorAgent API - 33 tests fixed
3. Pydantic Validation - 1 test fixed (partial)
4. Missing Implementations - 11 tests skipped
5. CLI Function Mismatches - 15 patches updated

---

## 🎯 FIX PLAN (Remaining 46 Tests)

### Overview
**Estimated Time:** 4-6 hours
**Approach:** Fix by category (easiest first)
**Target:** >90% pass rate (127+ tests)

### Priorities (Ranked by Efficiency)

1. **Analysis Pipeline** (2 tests, 15 min) - Quick win
2. **Execution Pipeline** (7 tests, 30 min) - Repetitive fix
3. **Neo4j Protocol** (5 tests, 30 min) - Known issue
4. **Pydantic Validation** (12 tests, 1-2 hours) - Individual fixes
5. **Workflow State Transitions** (8 tests, 1 hour) - Logic updates
6. **Logic/Assertion Errors** (15 tests, 1-2 hours) - Case-by-case

---

## 🔧 FIX 1: Analysis Pipeline (15 min, 2 tests)

### Problem
`StatisticalTestResult` missing significance flag fields

### Affected Tests
- `test_analysis_pipeline.py` (2 failed, 4 errors)

### Error Example
```
ValidationError: Field required [type=missing]
  significant_0_05
  significant_0_01
  significant_0_001
```

### Solution
Find and update `StatisticalTestResult` fixtures:

```python
# Current (BROKEN)
result = StatisticalTestResult(
    test_type='t-test',
    statistic=2.5,
    p_value=0.013,
    confidence_level=0.95
)

# Fixed (CORRECT)
result = StatisticalTestResult(
    test_type='t-test',
    statistic=2.5,
    p_value=0.013,
    confidence_level=0.95,
    significant_0_05=True,   # p < 0.05
    significant_0_01=False,  # p < 0.01
    significant_0_001=False  # p < 0.001
)
```

### Verification
```bash
pytest tests/integration/test_analysis_pipeline.py -v
# Should pass 6+ tests (currently 3 passing)
```

**Impact:** +2-4 tests

---

## 🔧 FIX 2: Execution Pipeline (30 min, 7 tests)

### Problem
`ExperimentProtocol` fixtures missing required fields (same as test_iterative_loop.py fix)

### Affected Tests
- `test_execution_pipeline.py` - 7 errors

### Error Example
```
ValidationError: 7 validation errors for ExperimentProtocol
  name: Field required
  domain: Field required
  objective: Field required
  steps: Field required
  resource_requirements: Field required
```

### Solution
Apply same fix pattern as `test_iterative_loop.py:92`:

```python
from kosmos.models.experiment import ResourceRequirements, ProtocolStep, Variable

protocol = ExperimentProtocol(
    id="test-001",
    name="Test Protocol Name (Min 10 chars)",
    hypothesis_id="hyp_001",
    experiment_type="computational",
    domain="biology",  # Required
    description="Comprehensive protocol description with sufficient detail to meet minimum 50 character requirement for the description field validation.",
    objective="Clear objective statement with minimum 20 characters required",
    steps=[
        ProtocolStep(
            step_number=1,
            description="Execute test procedure",
            expected_duration_minutes=5
        )
    ],
    variables={
        "test_var": Variable(
            name="test_var",
            description="Test variable description",  # Min 10 chars
            unit="units"
        )
    },
    resource_requirements=ResourceRequirements(
        estimated_runtime_seconds=300,
        cpu_cores=1,
        memory_gb=1,
        storage_gb=0.1
    )
)
```

### Files to Update
Search for `ExperimentProtocol(` in `test_execution_pipeline.py` and update all instances.

### Verification
```bash
pytest tests/integration/test_execution_pipeline.py -v
# Should pass 13/13 tests (currently 6 passing)
```

**Impact:** +7 tests

---

## 🔧 FIX 3: Neo4j Protocol Persistence (30 min, 5 tests)

### Problem
Despite conftest.py fix, `test_world_model_persistence.py` still reads cached config with `neo4j://`

### Affected Tests
- `test_world_model_persistence.py` (5 tests)

### Error Example
```
ValueError: Unknown protocol 'neo4j'
```

### Solution Option A: Per-Test Environment Forcing

Update `test_world_model_persistence.py` conftest to force env before each test:

```python
# Add to test_world_model_persistence.py

import os
import pytest

@pytest.fixture(autouse=True)
def force_neo4j_bolt():
    """Force bolt:// protocol for each test."""
    os.environ['NEO4J_URI'] = 'bolt://localhost:7687'

    # Force config singleton reset
    from kosmos.config import get_config
    get_config(reload=True)

    # Reset world model to pick up new config
    from kosmos.world_model import reset_world_model
    reset_world_model()

    yield
```

### Solution Option B: Mock world_model Factory

```python
@pytest.fixture
def reset_graph():
    """Reset knowledge graph before each test."""
    with patch.dict(os.environ, {'NEO4J_URI': 'bolt://localhost:7687'}):
        from kosmos.config import get_config
        get_config(reload=True)

        reset_world_model()
        yield
        reset_world_model()
```

### Verification
```bash
pytest tests/integration/test_world_model_persistence.py -v
# Should pass 8/8 tests (currently 0 passing)
```

**Impact:** +5 tests

---

## 🔧 FIX 4: Pydantic Validation (1-2 hours, 12 tests)

### Problem
Individual test fixtures violating Pydantic constraints

### Affected Tests

#### 4.1: Hypothesis.rationale Too Short (6 tests)

**Tests:**
- `test_end_to_end_research.py::TestConvergenceScenarios::*` (2 tests)
- `test_end_to_end_research.py::TestReportGeneration::*` (5 tests, some may have multiple issues)
- `test_iterative_loop.py::TestFeedbackIntegration::*` (4 tests)
- `test_world_model_persistence.py::TestRefinedHypothesisPersistence::*` (1 test)

**Error:**
```
ValidationError: String should have at least 20 characters
  rationale: 'Stimulant effects'  # Only 17 chars
```

**Fix Pattern:**
```python
# Search for Hypothesis( with short rationale
hypothesis = Hypothesis(
    statement="...",
    rationale="Stimulant effects"  # ❌ Too short
)

# Fix: Expand to >= 20 chars
hypothesis = Hypothesis(
    statement="...",
    rationale="Stimulant effects enhance attention and memory encoding"  # ✅ 56 chars
)
```

**Verification:**
```bash
pytest tests/integration/test_iterative_loop.py::TestFeedbackIntegration -v
pytest tests/integration/test_world_model_persistence.py::TestRefinedHypothesisPersistence -v
```

#### 4.2: ExperimentResult Missing Fields (4 tests)

**Test:** `test_iterative_loop.py::TestFeedbackIntegration::test_feedback_loop_processes_failure`

**Error:**
```
ValidationError: 4 validation errors for ExperimentResult
  experiment_id: Field required
  protocol_id: Field required
  primary_test: Value error, Primary test 't-test' not found in statistical_tests
  metadata: Field required
```

**Fix:**
```python
# Current (BROKEN)
result = ExperimentResult(
    id='result_002',
    hypothesis_id='hyp_001',
    status=ResultStatus.SUCCESS
)

# Fixed (CORRECT)
from kosmos.models.result import ExecutionMetadata
import sys, platform

result = ExperimentResult(
    id='result_002',
    experiment_id='exp_001',  # Required
    protocol_id='protocol_001',  # Required
    hypothesis_id='hyp_001',
    status=ResultStatus.SUCCESS,
    statistical_tests=[
        StatisticalTestSpec(test_type='t-test', variables=['var1', 'var2'])
    ],
    primary_test='t-test',  # Must exist in statistical_tests
    metadata=ExecutionMetadata(
        experiment_id='exp_001',
        timestamp=datetime.now(),
        numpy_version='1.0',
        random_seed=42,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
        platform=platform.system(),
        protocol_id='protocol_001'
    )
)
```

#### 4.3: AgentMessage Missing Fields (2 tests)

**Test:** `test_iterative_loop.py::TestMessagePassing::test_director_handles_hypothesis_response`

**Error:**
```
ValidationError: 3 validation errors for AgentMessage
  from_agent_id: Field required
```

**Fix:** Check AgentMessage schema and add missing required fields

### Verification
```bash
# Test each category individually
pytest tests/integration/test_end_to_end_research.py::TestConvergenceScenarios -v
pytest tests/integration/test_end_to_end_research.py::TestReportGeneration -v
pytest tests/integration/test_iterative_loop.py::TestFeedbackIntegration -v
```

**Impact:** +12 tests

---

## 🔧 FIX 5: Workflow State Transitions (1 hour, 8 tests)

### Problem
Tests expect invalid state transitions according to workflow state machine

### Affected Tests
- `test_iterative_loop.py::TestStateTransitions::*` (3 tests)
- `test_iterative_loop.py::TestSingleIteration::*` (2 tests)
- `test_iterative_loop.py::TestMessagePassing::*` (3 tests)

### Error Example
```
ValueError: Invalid transition from WorkflowState.ANALYZING to WorkflowState.CONVERGED.
Allowed transitions: [REFINING, DESIGNING_EXPERIMENTS, PAUSED, ERROR]
```

### Solution Approach

**Step 1:** Review workflow state machine:
```bash
grep -A 20 "class WorkflowState" kosmos/core/workflow.py
```

**Step 2:** Find allowed transitions:
```python
# In kosmos/core/workflow.py
WORKFLOW_TRANSITIONS = {
    WorkflowState.ANALYZING: [
        WorkflowState.REFINING,
        WorkflowState.DESIGNING_EXPERIMENTS,
        WorkflowState.PAUSED,
        WorkflowState.ERROR
    ]
}
```

**Step 3:** Update tests to use valid transitions:

```python
# Current (BROKEN)
director.workflow.transition_to(WorkflowState.CONVERGED)  # Invalid from ANALYZING

# Fixed (CORRECT)
# Option A: Add intermediate state
director.workflow.transition_to(WorkflowState.REFINING)
director.workflow.transition_to(WorkflowState.CONVERGED)  # If valid from REFINING

# Option B: Update test expectation
assert director.workflow.can_transition_to(WorkflowState.REFINING)  # Instead of CONVERGED
```

### Files to Update
- `tests/integration/test_iterative_loop.py` - All TestStateTransitions tests

### Verification
```bash
pytest tests/integration/test_iterative_loop.py::TestStateTransitions -v
# Should pass 6/6 tests (currently 3 passing)
```

**Impact:** +3-5 tests

---

## 🔧 FIX 6: Logic & Assertion Errors (1-2 hours, 15 tests)

### Problem
Miscellaneous test implementation issues requiring individual debugging

### Affected Tests

#### 6.1: director.wm is None
**Test:** `test_world_model_persistence.py::TestResearchQuestionPersistence::test_research_question_created_on_init`

**Error:** `AssertionError: assert None is not None`

**Fix:** Investigate why `director.wm` is not initialized. Check ResearchDirectorAgent init.

#### 6.2: KeyError 'failures'
**Test:** `test_iterative_loop.py::TestFeedbackIntegration::test_strategy_adaptation_based_on_feedback`

**Fix:** Check test expectation - may be looking for wrong dict key

#### 6.3: Real LLM Returns 0 Hypotheses
**Test:** `test_phase3_e2e.py::TestPhase3RealIntegration::test_real_hypothesis_workflow`

**Error:** `AssertionError: assert 0 > 0`

**Cause:** Real API call with Claude Code proxy key (999...) may be failing authentication

**Fix Options:**
- Skip test if using proxy key
- Mock the LLM call
- Investigate why real call returns empty

#### 6.4: DPI File Size Validation
**Test:** `test_visual_regression.py::TestFormattingPreservation::test_plots_use_correct_dpi`

**Error:** `assert 189395 > (158812 * 1.5)`

**Fix:** Adjust tolerance or update expected file size

### Approach
Debug each test individually:
```bash
# Run with full traceback
pytest tests/integration/test_world_model_persistence.py::TestResearchQuestionPersistence::test_research_question_created_on_init -vvs

# Check what's being tested
cat tests/integration/test_world_model_persistence.py | grep -A 20 "test_research_question_created_on_init"

# Fix and verify
pytest tests/integration/test_world_model_persistence.py::TestResearchQuestionPersistence::test_research_question_created_on_init -v
```

**Impact:** +8-12 tests (variable success rate)

---

## 📈 PROJECTED RESULTS AFTER FIXES

| Fix Category | Tests Fixed | Cumulative | Pass % |
|--------------|-------------|------------|--------|
| Current | 81 | 81 | 57.4% |
| + Analysis Pipeline | +2 | 83 | 58.9% |
| + Execution Pipeline | +7 | 90 | 63.8% |
| + Neo4j Protocol | +5 | 95 | 67.4% |
| + Pydantic Validation | +12 | 107 | 75.9% |
| + Workflow Transitions | +5 | 112 | 79.4% |
| + Logic/Assertion (50%) | +8 | 120 | **85.1%** |
| + Logic/Assertion (80%) | +12 | **127** | **90.1%** ✅ |

**Conservative Estimate:** 85-90% (120-127 tests)
**Optimistic Estimate:** 90-95% (127-134 tests)

---

## ✅ VERIFICATION COMMANDS

### After Each Fix Category
```bash
# Analysis Pipeline
pytest tests/integration/test_analysis_pipeline.py -v --tb=short

# Execution Pipeline
pytest tests/integration/test_execution_pipeline.py -v --tb=short

# Neo4j Protocol
pytest tests/integration/test_world_model_persistence.py -v --tb=short

# Pydantic Validation
pytest tests/integration/test_iterative_loop.py::TestFeedbackIntegration -v
pytest tests/integration/test_end_to_end_research.py::TestReportGeneration -v

# Workflow Transitions
pytest tests/integration/test_iterative_loop.py::TestStateTransitions -v

# Full Integration Suite
pytest tests/integration/ \
  --ignore=tests/integration/test_parallel_execution.py \
  --ignore=tests/integration/test_phase2_e2e.py \
  -v --tb=short \
  | tee integration_tests_final.txt
```

### Final Verification
```bash
# Check summary
tail -20 integration_tests_final.txt | grep "passed"

# Should show: "127+ passed, 11 skipped, <14 failed/errors"
```

---

## 🎯 SUCCESS CRITERIA

### Integration Tests: >90% Passing ✅
- **Minimum:** 127/141 tests passing (90.1%)
- **Target:** 130/141 tests passing (92.2%)
- **Acceptable:** Up to 11 skipped (Phase 2/3 features)

### No Critical Blockers ✅
- Zero Neo4j protocol errors
- Zero API signature mismatches
- Zero import errors (except skipped files)

### Test Categories Passing
- ✅ Multi-domain (15/15) - Already 100%
- ✅ Visual regression (10/11) - Already 91%
- ⏳ E2E research (15+/18) - Target 83%+
- ⏳ Iterative loop (20+/24) - Target 83%+
- ⏳ World model persistence (7+/8) - Target 87%+
- ⏳ Execution pipeline (12+/13) - Target 92%+
- ⏳ Analysis pipeline (7+/9) - Target 78%+
- ⏳ CLI (20+/28) - Target 71%+

---

## 🚀 AFTER REACHING 90%: RESUME DAY 4

Once integration tests pass >90%, complete Day 4:

### Phase 3: E2E Research Workflows
```bash
# Execute biology workflow
kosmos run "How does temperature affect enzyme activity?" \
  --domain biology \
  --max-iterations 2

# Verify completion
kosmos status
kosmos history
```

### Phase 4: Neo4j Validation
```bash
# Check graph stats
kosmos graph

# Export for inspection
kosmos graph --export day4_validation.json
```

### Phase 5: CLI Validation
```bash
# Test all commands
kosmos version
kosmos info
kosmos doctor
kosmos cache --stats
```

### Phase 6: Performance Baseline
```bash
# Measure cycle time
time kosmos run "Test question" --max-iterations 2

# Check cache hit rate
kosmos cache --stats
```

### Phase 7: Create Day 4 Complete Checkpoint
Document:
- Integration tests >90% ✅
- E2E workflows validated ✅
- Performance baselines ✅
- Ready for Day 5 (containerization)

---

## 🛠️ DEBUGGING TIPS

### Check Model Schemas
```python
from kosmos.models.experiment import ExperimentProtocol
print(ExperimentProtocol.model_json_schema())
```

### Check Workflow State Machine
```python
from kosmos.core.workflow import WorkflowState, WORKFLOW_TRANSITIONS
print(WORKFLOW_TRANSITIONS[WorkflowState.ANALYZING])
```

### Run Single Test with Full Output
```bash
pytest tests/integration/test_cli.py::TestBasicCommands::test_version_command -vvs
```

### Check What Changed
```bash
git diff 0432951..HEAD tests/integration/
```

---

## 📚 REFERENCE FILES

**Current Checkpoint:**
- `@docs/planning/CHECKPOINT_DAY4_INTEGRATION_FIXES_PARTIAL.md`

**Previous Checkpoints:**
- `@docs/planning/CHECKPOINT_DAY4_PARTIAL.md` - Original Day 4 blocked state
- `@docs/planning/CHECKPOINT_DAY3_TESTING_COMPLETE.md` - Days 1-3

**Code References:**
- `kosmos/models/experiment.py` - ExperimentProtocol, StatisticalTestSpec
- `kosmos/models/result.py` - ExperimentResult, StatisticalTestResult
- `kosmos/models/hypothesis.py` - Hypothesis model
- `kosmos/core/workflow.py` - Workflow state machine
- `tests/integration/conftest.py` - Integration test config

---

## ✅ COMMIT AFTER REACHING >90%

```bash
git add tests/integration/
git commit -m "Day 4: Complete integration test fixes (90%+ passing)

Systematic fixes for remaining test failures:

**Analysis Pipeline (2 tests):**
- Added significance flags to StatisticalTestResult fixtures

**Execution Pipeline (7 tests):**
- Updated ExperimentProtocol fixtures with all required fields

**Neo4j Protocol (5 tests):**
- Added per-test environment forcing for world_model_persistence

**Pydantic Validation (12 tests):**
- Fixed Hypothesis.rationale length violations
- Added missing ExperimentResult fields
- Updated AgentMessage fixtures

**Workflow Transitions (5 tests):**
- Updated test expectations to match state machine

**Logic/Assertion (~10 tests):**
- Individual debugging and fixes

**Results:**
- Before: 81/141 passing (57.4%)
- After: 127+/141 passing (90%+)
- Progress: +46 tests fixed

**Ready for Day 4 Phases 3-7:** E2E workflows, Neo4j validation, CLI testing

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
"
```

---

**Ready to fix remaining tests! Target: >90% integration tests passing** 🚀
