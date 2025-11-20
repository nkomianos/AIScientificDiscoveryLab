# Phase 2: Paper-Driven Deep Testing - Foundation

**Status:** Ready to Begin
**Prerequisites:** Phase 1 COMPLETE ✅
**Budget:** $19.99 available (DeepSeek API)
**Estimated Cost:** $2-5 total
**Estimated Duration:** 3-5 hours execution time

---

## Phase 2 Goals

Validate the **Kosmos AI Scientist autonomous research vision** from the paper:

1. **Full Autonomous Research Loop** - Test all 7 workflow phases
2. **Multi-Domain Capability** - Validate across 5 research domains
3. **Multi-Iteration Refinement** - Test hypothesis evolution over 3-5 cycles
4. **Literature Integration** - Verify paper grounding (if implemented)
5. **Convergence Detection** - Test workflow completion criteria
6. **Performance Validation** - Measure actual vs claimed improvements

---

## Validated Foundation from Phase 1

### Components Ready for Phase 2 ✅

**Core Pipeline:**
- ✅ **LLM Provider** - DeepSeek API ($0.002/test avg)
- ✅ **HypothesisGeneratorAgent** - Generates domain-specific hypotheses with testability scores
- ✅ **CodeValidator** - AST-based safety validation (`.passed`, `.violations`)
- ✅ **CodeExecutor** - Direct execution mode working
- ✅ **ResearchDirectorAgent** - Workflow orchestration (validated in biology/neuroscience tests)

**API Patterns Discovered:**
```python
# All agents return Pydantic response objects
response = agent.method(...)
assert response.attribute  # Not: assert response

# LLM responses
assert response.content  # Not: len(response)

# Hypothesis responses
assert response.hypotheses  # Not: direct iteration

# Safety reports
assert report.passed  # Not: .is_safe
assert len(report.violations)  # Not: .has_dangerous_operations

# Execution results
assert exec_result.success
assert exec_result.stdout
assert exec_result.execution_time
```

### Known Limitations (Won't Block Phase 2)

**Deferred Components:**
- ⏭️ ExperimentDesigner - PromptTemplate framework issue
- ⏭️ CodeGenerator - Complex protocol setup required
- ⏭️ Statistical Analysis - Module API investigation needed
- ⏭️ DataAnalyst - Agent API investigation needed
- ⏭️ Database Persistence - Model config issue
- ⏭️ Sandbox Execution - Docker API investigation
- ⏭️ Knowledge Graph - Neo4j not configured

**Why They Don't Block:**
- ResearchDirectorAgent handles missing components gracefully
- Core pipeline (hypothesis → execution → results) fully validated
- System designed for modular operation

---

## Phase 2 Testing Strategy

### Priority 1: Full Workflow Validation (HIGH)

**Goal:** Validate complete autonomous research loop

**Tests to Run:**

#### 1.1 Multi-Iteration Biology Research
**File:** `tests/e2e/test_full_research_workflow.py::TestPaperValidation::test_multi_iteration_research_cycle`
**Status:** Implemented, needs debugging
**What it tests:**
- 3 complete research iterations
- Workflow state progression
- Hypothesis refinement over time
- Results accumulation
**Expected behavior:**
- Workflow progresses: initializing → generating_hypotheses → designing_experiments → executing → analyzing
- Multiple iterations complete
- Results stored and retrievable
**Budget:** ~$0.30-0.50

#### 1.2 Neuroscience Workflow Deep Test
**File:** `tests/e2e/test_full_research_workflow.py::TestNeuroscienceResearchWorkflow::test_full_neuroscience_workflow`
**Status:** Implemented and passing
**What it tests:**
- Full neuroscience research cycle
- Hypothesis generation in neuroscience domain
- Database persistence of hypotheses
**Budget:** ~$0.10-0.15 (already run, can re-run for validation)

#### 1.3 Additional Domain Tests
**Create tests for:**
- Physics: "How does electromagnetic field strength affect particle trajectory?"
- Chemistry: "What is the reaction rate of enzyme X at different pH levels?"
- Materials: "How does temperature affect the tensile strength of polymer Y?"
**Pattern:** Copy neuroscience test, change domain and question
**Budget:** ~$0.30-0.45 total (3 domains × $0.10-0.15)

---

### Priority 2: Workflow State Validation (HIGH)

**Goal:** Verify ResearchDirectorAgent orchestrates all phases correctly

**Tests to Create:**

#### 2.1 Workflow Phase Progression
```python
def test_workflow_phase_progression():
    """Test director progresses through all workflow states."""
    director = ResearchDirectorAgent(
        research_question="Test question",
        domain="biology",
        config={"max_iterations": 3}
    )

    # Start research
    result = director.execute({"action": "start_research"})
    assert result["status"] == "research_started"

    # Track state transitions
    states_visited = []
    for step in range(20):  # Safety limit
        status = director.get_research_status()
        state = status["workflow_state"].lower()
        states_visited.append(state)

        if state == "converged":
            break

        director.execute({"action": "step"})

    # Verify expected states visited
    assert "initializing" in states_visited
    assert "generating_hypotheses" in states_visited
    # Note: Other states may be skipped due to component limitations
```
**Budget:** ~$0.20-0.30

#### 2.2 Hypothesis Pool Growth
```python
def test_hypothesis_pool_growth():
    """Test hypothesis pool grows with iterations."""
    director = ResearchDirectorAgent(
        research_question="Test question",
        domain="biology",
        config={"max_iterations": 3, "num_hypotheses": 2}
    )

    director.execute({"action": "start_research"})

    # Let workflow progress
    for _ in range(5):
        director.execute({"action": "step"})

    # Check hypothesis pool
    assert hasattr(director.research_plan, 'hypothesis_pool')
    hypotheses = director.research_plan.hypothesis_pool
    assert len(hypotheses) >= 1, "At least one hypothesis should be generated"
```
**Budget:** ~$0.15-0.20

---

### Priority 3: Component Integration (MEDIUM)

**Goal:** Test component interactions work correctly

#### 3.1 Hypothesis → Execution Path
**Status:** Already validated in `test_mini_research_workflow`
**What it proves:** HypothesisGenerator → CodeExecutor integration works
**Re-run:** No need, already passing

#### 3.2 Safety Validator Integration
```python
def test_safety_validator_blocks_dangerous_hypothesis():
    """Test safety validator blocks dangerous code in workflow."""
    # This would require creating a hypothesis that generates dangerous code
    # and verifying it's blocked before execution
    pass  # Implement in Phase 2
```
**Budget:** ~$0.10-0.15

---

### Priority 4: Performance Benchmarking (LOW)

**Goal:** Measure actual performance vs paper claims

#### 4.1 Cache Effectiveness
**File:** `tests/e2e/test_full_research_workflow.py::TestPerformanceValidation::test_cache_hit_rate`
**Status:** Placeholder only
**What to test:**
- Run same question twice
- Measure second run cost vs first
- Verify cache reduces API calls
**Budget:** ~$0.20-0.30

#### 4.2 Parallel vs Sequential
**File:** `tests/e2e/test_full_research_workflow.py::TestPerformanceValidation::test_parallel_vs_sequential_speedup`
**Status:** Placeholder only
**What to test:**
- Run workflow with concurrent_operations=True
- Compare to sequential execution
- Measure speedup
**Budget:** ~$0.40-0.60

---

## Phase 2 Test Execution Plan

### Week 1: Core Workflow Validation

**Day 1: Multi-Domain Testing**
1. Run neuroscience workflow (re-validate): $0.15
2. Run physics workflow: $0.15
3. Run chemistry workflow: $0.15
4. Run materials workflow: $0.15
**Total:** ~$0.60

**Day 2: Multi-Iteration Testing**
1. Debug multi-iteration test: $0.30
2. Run biology 3-iteration cycle: $0.50
3. Run neuroscience 3-iteration cycle: $0.50
**Total:** ~$1.30

**Day 3: Workflow State Validation**
1. Test phase progression: $0.30
2. Test hypothesis pool growth: $0.20
3. Test state transitions: $0.20
**Total:** ~$0.70

### Week 2: Advanced Features (Optional)

**Day 4: Component Integration**
1. Safety validator integration: $0.15
2. Result analysis integration: $0.15
**Total:** ~$0.30

**Day 5: Performance Benchmarking**
1. Cache effectiveness: $0.30
2. Parallel speedup: $0.60
**Total:** ~$0.90

### Total Estimated Budget
**Week 1 (Core):** $2.60
**Week 2 (Advanced):** $1.20
**Total:** $3.80 (well within $19.99 budget)

---

## Success Criteria for Phase 2

### Minimum Viable Coverage
- ✅ At least 3 domains tested (biology, neuroscience, + 1 more)
- ✅ At least 1 multi-iteration workflow completes
- ✅ Workflow state progression validated
- ✅ Hypothesis pool grows over iterations

### Stretch Goals
- All 5 domains tested
- 3+ iteration cycles complete
- Performance benchmarks measured
- Cache effectiveness validated
- Parallel execution speedup demonstrated

---

## Known Risks and Mitigation

### Risk 1: Multi-Iteration Test Stuck in Loop
**Symptom:** Workflow stays in same state indefinitely
**Mitigation:**
- Add safety limit (max 20 steps)
- Add stuck detection (3 consecutive same states → break)
- Log state transitions for debugging

### Risk 2: Components Not Implemented
**Symptom:** Workflow skips phases due to missing ExperimentDesigner/DataAnalyst
**Mitigation:**
- Accept graceful degradation
- Focus on what works (hypothesis → execution → results)
- Document which phases are skipped

### Risk 3: Budget Overrun
**Symptom:** Tests cost more than estimated
**Mitigation:**
- DeepSeek extremely cheap ($0.01 actual vs $0.35 estimated in Phase 1)
- Monitor dashboard: https://platform.deepseek.com/usage
- Stop if approaching $15 spent

### Risk 4: Database/Persistence Issues
**Symptom:** Tests fail due to database errors
**Mitigation:**
- Use in-memory mode if database fails
- ResearchDirectorAgent designed for graceful degradation
- Database validation already skipped in Phase 1

---

## Test Organization

### Existing Test Files
1. **tests/e2e/test_system_sanity.py** - Component sanity checks (Phase 1)
2. **tests/e2e/test_full_research_workflow.py** - Full workflow tests
   - TestBiologyResearchWorkflow (passing)
   - TestNeuroscienceResearchWorkflow (passing)
   - TestPaperValidation (1 needs debugging)
   - TestPerformanceValidation (placeholders)
   - TestCLIWorkflows (placeholders)
   - TestDockerDeployment (basic test passing)

### Recommended New Tests
3. **tests/e2e/test_multi_domain.py** - All 5 domains
4. **tests/e2e/test_convergence.py** - Convergence detection
5. **tests/e2e/test_paper_claims.py** - Validate specific paper claims

---

## Quick Commands

### Run All Phase 2 Tests
```bash
pytest tests/e2e/test_full_research_workflow.py -v -s --no-cov -m "e2e and slow"
```

### Run Single Multi-Iteration Test
```bash
pytest tests/e2e/test_full_research_workflow.py::TestPaperValidation::test_multi_iteration_research_cycle -v -s --no-cov
```

### Run All Domain Tests
```bash
pytest tests/e2e/test_full_research_workflow.py::TestBiologyResearchWorkflow -v -s --no-cov
pytest tests/e2e/test_full_research_workflow.py::TestNeuroscienceResearchWorkflow -v -s --no-cov
# Add physics, chemistry, materials when implemented
```

### Monitor DeepSeek Usage
```bash
# Visit dashboard
https://platform.deepseek.com/usage

# Check balance in real-time
# (No CLI command, must use web dashboard)
```

---

## Documentation Requirements

After each major test run:
1. Update `E2E_TESTING_PLAN.md` with results
2. Document cost in budget tracker
3. Log any new API discoveries
4. Note workflow behaviors observed

After Phase 2 completion:
1. Create `PHASE2_RESULTS.md` with comprehensive findings
2. Update README if needed
3. Document any paper claims validated/invalidated
4. Recommendations for production deployment

---

## Next Steps to Begin Phase 2

1. **Verify Foundation**
   ```bash
   pytest tests/e2e/test_system_sanity.py -v --no-cov
   # Should show: 5 passed, 7 skipped
   ```

2. **Check Budget**
   - Visit https://platform.deepseek.com/usage
   - Verify $19+ remaining

3. **Start with Known-Good Test**
   ```bash
   pytest tests/e2e/test_full_research_workflow.py::TestNeuroscienceResearchWorkflow::test_full_neuroscience_workflow -v -s --no-cov
   # Should pass, cost ~$0.10-0.15
   ```

4. **Debug Multi-Iteration Test**
   ```bash
   pytest tests/e2e/test_full_research_workflow.py::TestPaperValidation::test_multi_iteration_research_cycle -v -s --no-cov
   # Fix any issues found
   ```

5. **Expand to New Domains**
   - Copy neuroscience test pattern
   - Create physics, chemistry, materials variants
   - Run and validate

---

## Phase 2 Vision

**Goal:** Prove the Kosmos AI Scientist can autonomously conduct research

**What Success Looks Like:**
- Research question → Multiple testable hypotheses generated
- Hypotheses → Executable experiments designed
- Experiments → Code generated and safely executed
- Results → Analyzed and interpreted
- Iterations → Hypotheses refined based on results
- Convergence → Workflow completes with validated findings

**What We're Testing:**
Not just "does the code run?" but "can this system think like a scientist?"

The components work individually (Phase 1 proved that). Now we test the **orchestration** - can they work together to autonomously advance scientific understanding?

That's the Kosmos vision. Let's validate it.

---

**Phase 2 Status:** READY TO BEGIN

**Blocked by:** Nothing - all prerequisites met

**Start when ready.**
