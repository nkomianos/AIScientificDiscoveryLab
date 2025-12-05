# Kosmos Implementation Plan: Resolving Documented Gaps

**Date**: December 5, 2025
**Source**: `120525_implementation_gaps_v2.md`
**Scope**: 6 implementation tasks across 4 phases
**Estimated Effort**: 19-27 hours

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Phase A: Critical Safety Fixes](#phase-a-critical-safety-fixes)
3. [Phase B: Data Integrity](#phase-b-data-integrity)
4. [Phase C: Performance Improvements](#phase-c-performance-improvements)
5. [Phase D: Future-Proofing](#phase-d-future-proofing)
6. [Testing Strategy](#testing-strategy)
7. [Rollback Procedures](#rollback-procedures)
8. [Checkpoints](#checkpoints)

---

## Executive Summary

### Gap Categories

| Category | Count | Priority |
|----------|-------|----------|
| Critical Safety | 2 | IMMEDIATE |
| Data Integrity | 2 | HIGH |
| Performance | 1 | MEDIUM |
| Future-Proofing | 1 | LOW |

### Phase Overview

| Phase | Tasks | Effort | Risk | Dependencies |
|-------|-------|--------|------|--------------|
| A: Safety | A1, A2 | 6h | Low-Medium | None |
| B: Data | B1, B2 | 5h | Low | Phase A |
| C: Performance | C1 | 4h | Medium | Phase A |
| D: Future | D2 | 4h | Low | Phase B |

### Dependency Graph

```
┌─────────────────────────────────────────────────────────────┐
│                    PHASE A: SAFETY                          │
│  ┌──────────────────┐    ┌──────────────────┐              │
│  │ A1: Budget       │    │ A2: Error        │              │
│  │ Enforcement      │    │ Recovery         │              │
│  └────────┬─────────┘    └────────┬─────────┘              │
└───────────┼───────────────────────┼─────────────────────────┘
            │                       │
            ▼                       ▼
┌───────────────────────────────────────────────────────────┐
│                    PHASE B: DATA                          │
│  ┌──────────────────┐    ┌──────────────────┐            │
│  │ B1: Annotation   │◄───│ B2: Data         │            │
│  │ Storage          │    │ Loading          │            │
│  └────────┬─────────┘    └──────────────────┘            │
└───────────┼───────────────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────────────────────────┐
│                    PHASE C: PERFORMANCE                   │
│  ┌──────────────────┐                                     │
│  │ C1: Async        │                                     │
│  │ Providers        │                                     │
│  └────────┬─────────┘                                     │
└───────────┼───────────────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────────────────────────┐
│                    PHASE D: FUTURE                        │
│  ┌──────────────────┐                                     │
│  │ D2: Neo4j        │                                     │
│  │ Integration      │                                     │
│  └──────────────────┘                                     │
└───────────────────────────────────────────────────────────┘
```

---

## Phase A: Critical Safety Fixes

### A1. Budget Enforcement in Research Loop

**Gap Reference**: `120525_implementation_gaps_v2.md` Section 5

**Problem**: System tracks budget via `MetricsCollector.check_budget()` but never halts execution when budget is exceeded. Research continues until max iterations regardless of cost.

**Impact**: Potential runaway API costs in production environments.

#### Files to Modify

| File | Changes | Lines |
|------|---------|-------|
| `kosmos/core/metrics.py` | Add `BudgetExceededError` class | +15 lines |
| `kosmos/core/metrics.py` | Add `enforce_budget()` method | +12 lines |
| `kosmos/agents/research_director.py` | Add budget check in `decide_next_action()` | +15 lines |

#### Implementation Details

**Step 1: Add Exception Class**

Location: `kosmos/core/metrics.py` (after line 59)

```python
class BudgetExceededError(Exception):
    """
    Raised when budget limit is exceeded.

    Attributes:
        current_cost: Actual cost incurred
        limit: Configured budget limit
        usage_percent: Percentage of budget used
    """

    def __init__(
        self,
        current_cost: float,
        limit: float,
        usage_percent: float = None,
        message: str = None
    ):
        self.current_cost = current_cost
        self.limit = limit
        self.usage_percent = usage_percent or (current_cost / limit * 100 if limit else 0)
        super().__init__(
            message or f"Budget exceeded: ${current_cost:.2f} spent (limit: ${limit:.2f}, {self.usage_percent:.1f}%)"
        )
```

**Step 2: Add Enforcement Method**

Location: `kosmos/core/metrics.py` (in `MetricsCollector` class, after `check_budget()` method)

```python
def enforce_budget(self) -> None:
    """
    Check budget and raise exception if exceeded.

    This method should be called before each expensive operation
    to prevent runaway costs.

    Raises:
        BudgetExceededError: If current spending exceeds budget limit
    """
    if not self.budget_enabled:
        return  # No enforcement if budget not enabled

    status = self.check_budget()

    if status.get('budget_exceeded'):
        raise BudgetExceededError(
            current_cost=status.get('current_cost_usd', 0),
            limit=self.budget_limit_usd,
            usage_percent=status.get('usage_percent', 100)
        )
```

**Step 3: Add Budget Check in Research Director**

Location: `kosmos/agents/research_director.py`, in `decide_next_action()` method (at start)

```python
def decide_next_action(self) -> NextAction:
    """Decide the next action for the research workflow."""

    # CHECKPOINT A1: Budget enforcement
    # Check budget before any decision that could trigger API calls
    try:
        from kosmos.core.metrics import get_metrics, BudgetExceededError
        metrics = get_metrics()
        if metrics.budget_enabled:
            metrics.enforce_budget()
    except BudgetExceededError as e:
        logger.error(f"[BUDGET] Research halted: {e}")

        # Transition to CONVERGED state gracefully
        with self._workflow_context():
            self.workflow.transition_to(
                WorkflowState.CONVERGED,
                action="Budget limit reached - research halted",
                metadata={"reason": "budget_exceeded", "cost": e.current_cost, "limit": e.limit}
            )

        # Return CONVERGE to signal workflow completion
        return NextAction.CONVERGE
    except ImportError:
        # Metrics module not available - continue without enforcement
        logger.debug("Metrics module not available for budget check")

    # ... existing decision logic continues ...
```

#### Verification Checklist

- [ ] `BudgetExceededError` is importable from `kosmos.core.metrics`
- [ ] `enforce_budget()` raises when `budget_exceeded: True`
- [ ] `enforce_budget()` does nothing when budget not enabled
- [ ] Research director catches exception and transitions to CONVERGED
- [ ] Logs show clear budget exceeded message with amounts

#### Test Cases

```python
# tests/unit/core/test_budget_enforcement.py

def test_budget_exceeded_error_attributes():
    """Error contains cost, limit, and percentage."""
    error = BudgetExceededError(current_cost=10.50, limit=10.00)
    assert error.current_cost == 10.50
    assert error.limit == 10.00
    assert error.usage_percent == 105.0

def test_enforce_budget_raises_when_exceeded():
    """enforce_budget raises when budget exceeded."""
    metrics = MetricsCollector()
    metrics.budget_enabled = True
    metrics.budget_limit_usd = 10.0

    # Simulate spending over budget
    metrics.record_api_call(model="claude", input_tokens=1000000, output_tokens=500000, duration_seconds=1.0)

    with pytest.raises(BudgetExceededError):
        metrics.enforce_budget()

def test_enforce_budget_silent_when_disabled():
    """enforce_budget does nothing when budget disabled."""
    metrics = MetricsCollector()
    metrics.budget_enabled = False
    metrics.enforce_budget()  # Should not raise

def test_research_director_halts_on_budget():
    """Research director transitions to CONVERGED on budget exceeded."""
    # Mock metrics to return budget_exceeded
    with patch('kosmos.core.metrics.get_metrics') as mock:
        mock.return_value.budget_enabled = True
        mock.return_value.enforce_budget.side_effect = BudgetExceededError(10.5, 10.0)

        director = ResearchDirectorAgent(research_question="test", domain="test")
        action = director.decide_next_action()

        assert action == NextAction.CONVERGE
        assert director.workflow.current_state == WorkflowState.CONVERGED
```

---

### A2. Error Recovery Strategy in Research Director

**Gap Reference**: `120525_implementation_gaps_v2.md` Section 3 (TODO at line 475)

**Problem**: Error handlers log errors and increment counter but take no recovery action. Workflow continues in degraded state.

**Impact**: Silent failures, no retry logic, no circuit breaking.

#### Files to Modify

| File | Changes | Lines |
|------|---------|-------|
| `kosmos/agents/research_director.py` | Add error recovery constants | +5 lines |
| `kosmos/agents/research_director.py` | Add `_handle_error_with_recovery()` | +50 lines |
| `kosmos/agents/research_director.py` | Add `_reset_error_streak()` | +8 lines |
| `kosmos/agents/research_director.py` | Update 6 error handlers | +30 lines |

#### Implementation Details

**Step 1: Add Constants and Instance Variables**

Location: `kosmos/agents/research_director.py` (at class level, near line 100)

```python
# Error recovery configuration
MAX_CONSECUTIVE_ERRORS = 3  # Halt after this many failures in a row
ERROR_BACKOFF_SECONDS = [2, 4, 8]  # Exponential backoff delays
ERROR_RECOVERY_LOG_PREFIX = "[ERROR-RECOVERY]"
```

Location: `kosmos/agents/research_director.py` (in `__init__`, after line 120)

```python
# Error recovery tracking
self._consecutive_errors: int = 0
self._error_history: List[Dict[str, Any]] = []
self._last_error_time: Optional[datetime] = None
```

**Step 2: Add Error Recovery Method**

Location: `kosmos/agents/research_director.py` (after `__init__`, ~line 200)

```python
def _handle_error_with_recovery(
    self,
    error_source: str,
    error_message: str,
    recoverable: bool = True,
    error_details: Optional[Dict[str, Any]] = None
) -> Optional[NextAction]:
    """
    Handle an error with recovery strategy.

    Implements:
    - Consecutive error counting
    - Exponential backoff
    - Circuit breaker (halt after MAX_CONSECUTIVE_ERRORS)
    - Error history tracking

    Args:
        error_source: Name of the agent/component that failed
        error_message: Human-readable error description
        recoverable: Whether this error type can be retried
        error_details: Additional error context

    Returns:
        NextAction if recovery possible, None if should abort current handler
    """
    import time

    # Update error tracking
    self.errors_encountered += 1
    self._consecutive_errors += 1
    self._last_error_time = datetime.utcnow()

    # Record in error history
    error_record = {
        'source': error_source,
        'message': error_message,
        'timestamp': self._last_error_time.isoformat(),
        'consecutive_count': self._consecutive_errors,
        'recoverable': recoverable,
        'details': error_details or {}
    }
    self._error_history.append(error_record)

    # Log the error
    logger.error(
        f"{ERROR_RECOVERY_LOG_PREFIX} {error_source}: {error_message} "
        f"(attempt {self._consecutive_errors}/{MAX_CONSECUTIVE_ERRORS})"
    )

    # Check if we've hit the circuit breaker threshold
    if self._consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
        logger.error(
            f"{ERROR_RECOVERY_LOG_PREFIX} Max consecutive errors reached. "
            f"Transitioning to ERROR state."
        )

        with self._workflow_context():
            self.workflow.transition_to(
                WorkflowState.ERROR,
                action=f"Max errors exceeded: {error_message}",
                metadata={'error_history': self._error_history[-MAX_CONSECUTIVE_ERRORS:]}
            )

        return NextAction.ERROR_RECOVERY

    # For recoverable errors, apply backoff and retry
    if recoverable:
        backoff_index = min(self._consecutive_errors - 1, len(ERROR_BACKOFF_SECONDS) - 1)
        backoff_seconds = ERROR_BACKOFF_SECONDS[backoff_index]

        logger.info(
            f"{ERROR_RECOVERY_LOG_PREFIX} Waiting {backoff_seconds}s before retry "
            f"(attempt {self._consecutive_errors + 1})"
        )

        time.sleep(backoff_seconds)

        # Re-evaluate what action to take
        return self.decide_next_action()

    # Non-recoverable error - just return None to exit handler
    logger.warning(
        f"{ERROR_RECOVERY_LOG_PREFIX} Non-recoverable error from {error_source}. "
        f"Skipping to next action."
    )
    return None


def _reset_error_streak(self) -> None:
    """
    Reset consecutive error counter after successful operation.

    Call this at the end of each successful handler to reset the
    circuit breaker counter.
    """
    if self._consecutive_errors > 0:
        logger.debug(
            f"{ERROR_RECOVERY_LOG_PREFIX} Error streak reset "
            f"(was {self._consecutive_errors} consecutive errors)"
        )
    self._consecutive_errors = 0
```

**Step 3: Update Error Handlers**

Location: `kosmos/agents/research_director.py`, update each handler

Example for `_handle_hypothesis_generator_response` (lines 462-501):

```python
def _handle_hypothesis_generator_response(self, message: AgentMessage):
    """Handle response from HypothesisGeneratorAgent."""
    content = message.content

    if message.type == MessageType.ERROR:
        # CHECKPOINT A2: Error recovery with backoff
        recovery_action = self._handle_error_with_recovery(
            error_source="HypothesisGeneratorAgent",
            error_message=content.get('error', 'Unknown error'),
            recoverable=True,
            error_details={'hypothesis_count_before': len(self.research_plan.hypothesis_pool)}
        )
        if recovery_action:
            self._execute_next_action(recovery_action)
        return

    # Success path - reset error streak
    self._reset_error_streak()

    # ... rest of existing handler code ...
```

Apply same pattern to:
- `_handle_experiment_designer_response` (line 503)
- `_handle_executor_response` (line 541)
- `_handle_data_analyst_response` (line 598)
- `_handle_hypothesis_refiner_response` (line 658)
- `_handle_convergence_detector_response` (if exists)

#### Verification Checklist

- [ ] `_consecutive_errors` increments on each error
- [ ] `_consecutive_errors` resets on success
- [ ] Backoff timing matches `ERROR_BACKOFF_SECONDS`
- [ ] Workflow transitions to ERROR after 3 consecutive failures
- [ ] Error history contains all required fields
- [ ] Non-recoverable errors are handled gracefully

#### Test Cases

```python
# tests/unit/agents/test_error_recovery.py

def test_consecutive_error_counting():
    """Consecutive errors increment counter."""
    director = ResearchDirectorAgent(research_question="test", domain="test")

    # Simulate 2 errors
    director._handle_error_with_recovery("TestAgent", "Error 1", recoverable=False)
    director._handle_error_with_recovery("TestAgent", "Error 2", recoverable=False)

    assert director._consecutive_errors == 2
    assert len(director._error_history) == 2

def test_error_streak_reset():
    """Success resets error counter."""
    director = ResearchDirectorAgent(research_question="test", domain="test")

    director._consecutive_errors = 2
    director._reset_error_streak()

    assert director._consecutive_errors == 0

def test_max_errors_triggers_error_state():
    """3 consecutive errors trigger ERROR state."""
    director = ResearchDirectorAgent(research_question="test", domain="test")

    for i in range(3):
        director._handle_error_with_recovery("TestAgent", f"Error {i}", recoverable=False)

    assert director.workflow.current_state == WorkflowState.ERROR

def test_backoff_timing():
    """Backoff follows exponential pattern."""
    # This test would need to mock time.sleep
    # Verify sleep called with [2, 4, 8] sequence
    pass
```

---

## Phase B: Data Integrity

### B1. Annotation Storage (Phase 2 Feature)

**Gap Reference**: `120525_implementation_gaps_v2.md` Section 2 (lines 879, 893)

**Problem**: `add_annotation()` only logs, `get_annotations()` returns empty list. Annotations don't persist.

**Impact**: Curation workflow unusable; user annotations lost.

#### Files to Modify

| File | Changes | Lines |
|------|---------|-------|
| `kosmos/world_model/simple.py` | Implement `add_annotation()` | +25 lines |
| `kosmos/world_model/simple.py` | Implement `get_annotations()` | +30 lines |
| `kosmos/world_model/simple.py` | Update `_node_to_entity()` | +15 lines |

#### Implementation Details

**Step 1: Implement add_annotation()**

Location: `kosmos/world_model/simple.py` (replace lines 868-881)

```python
def add_annotation(self, entity_id: str, annotation: Annotation) -> None:
    """
    Add annotation to entity.

    Stores annotations as a JSON array in the Neo4j node's 'annotations' property.
    Each annotation is serialized as a JSON string within the array.

    Args:
        entity_id: ID of entity to annotate (entity_id or paper_id)
        annotation: Annotation object to add

    Note:
        - Creates annotations array if it doesn't exist
        - Appends to existing annotations array
        - Timestamps are ISO 8601 formatted
    """
    if not self.connected:
        logger.warning(
            f"Not connected to Neo4j, annotation not persisted for {entity_id}"
        )
        return

    # Serialize annotation to JSON-compatible dict
    ann_dict = {
        'text': annotation.text,
        'created_by': annotation.created_by,
        'created_at': (annotation.created_at or datetime.utcnow()).isoformat(),
        'annotation_id': str(uuid.uuid4())  # Unique ID for each annotation
    }

    # Cypher query: append to annotations array, create if null
    query = """
    MATCH (n)
    WHERE n.entity_id = $entity_id OR n.paper_id = $entity_id
    SET n.annotations = CASE
        WHEN n.annotations IS NULL THEN [$annotation]
        ELSE n.annotations + $annotation
    END,
    n.updated_at = $updated_at
    RETURN count(n) as updated, n.entity_id as eid
    """

    try:
        result = self.graph.run(
            query,
            entity_id=entity_id,
            annotation=json.dumps(ann_dict),
            updated_at=datetime.utcnow().isoformat()
        ).data()

        if result and result[0]['updated'] > 0:
            logger.info(
                f"Annotation added to {entity_id} by {annotation.created_by}: "
                f"{annotation.text[:50]}{'...' if len(annotation.text) > 50 else ''}"
            )
        else:
            logger.warning(f"Entity not found for annotation: {entity_id}")

    except Exception as e:
        logger.error(f"Failed to add annotation to {entity_id}: {e}")
        raise
```

**Step 2: Implement get_annotations()**

Location: `kosmos/world_model/simple.py` (replace lines 883-894)

```python
def get_annotations(self, entity_id: str) -> List[Annotation]:
    """
    Get all annotations for an entity.

    Retrieves and deserializes the annotations array from the Neo4j node.

    Args:
        entity_id: ID of entity to query (entity_id or paper_id)

    Returns:
        List of Annotation objects, empty list if none or on error

    Note:
        - Gracefully handles missing annotations property
        - Skips malformed annotation entries with warning
        - Returns annotations in order they were added
    """
    if not self.connected:
        logger.debug(f"Not connected to Neo4j, returning empty annotations for {entity_id}")
        return []

    query = """
    MATCH (n)
    WHERE n.entity_id = $entity_id OR n.paper_id = $entity_id
    RETURN n.annotations as annotations
    """

    try:
        result = self.graph.run(query, entity_id=entity_id).data()

        if not result:
            logger.debug(f"Entity not found: {entity_id}")
            return []

        annotations_raw = result[0].get('annotations')
        if not annotations_raw:
            return []

        annotations = []
        for i, ann_json in enumerate(annotations_raw):
            try:
                # Parse JSON string to dict
                if isinstance(ann_json, str):
                    ann_dict = json.loads(ann_json)
                else:
                    ann_dict = ann_json

                # Reconstruct Annotation object
                created_at = None
                if ann_dict.get('created_at'):
                    try:
                        created_at = datetime.fromisoformat(ann_dict['created_at'])
                    except ValueError:
                        logger.debug(f"Invalid created_at format: {ann_dict['created_at']}")

                annotations.append(Annotation(
                    text=ann_dict['text'],
                    created_by=ann_dict['created_by'],
                    created_at=created_at
                ))

            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(
                    f"Failed to parse annotation {i} for {entity_id}: {e}"
                )
                continue

        return annotations

    except Exception as e:
        logger.error(f"Failed to get annotations for {entity_id}: {e}")
        return []
```

**Step 3: Update _node_to_entity()**

Location: `kosmos/world_model/simple.py` (in `_node_to_entity()` method, around line 353)

```python
# Load annotations from node property
annotations = []
annotations_raw = node.get('annotations', [])
if annotations_raw:
    for ann_json in annotations_raw:
        try:
            if isinstance(ann_json, str):
                ann_dict = json.loads(ann_json)
            else:
                ann_dict = ann_json

            created_at = None
            if ann_dict.get('created_at'):
                try:
                    created_at = datetime.fromisoformat(ann_dict['created_at'])
                except ValueError:
                    pass

            annotations.append(Annotation(
                text=ann_dict['text'],
                created_by=ann_dict['created_by'],
                created_at=created_at
            ))
        except Exception as e:
            logger.debug(f"Failed to parse annotation in node: {e}")

return Entity(
    # ... other fields ...
    annotations=annotations,  # Was: annotations=[]
)
```

#### Verification Checklist

- [ ] `add_annotation()` creates `annotations` array if null
- [ ] `add_annotation()` appends to existing array
- [ ] `get_annotations()` returns Annotation objects
- [ ] Round-trip works (add then get returns same data)
- [ ] `_node_to_entity()` loads annotations
- [ ] Export/import preserves annotations

---

### B2. Load Actual Data for Hypothesis/Result Evaluation

**Gap Reference**: `120525_implementation_gaps_v2.md` Section 3 (lines 1021, 1105)

**Problem**: Concurrent evaluation methods use placeholder prompts instead of loading actual hypothesis/result data from database.

**Impact**: LLM evaluations lack context, reducing accuracy.

#### Files to Modify

| File | Changes | Lines |
|------|---------|-------|
| `kosmos/agents/research_director.py` | Add `_build_hypothesis_evaluation_prompt()` | +35 lines |
| `kosmos/agents/research_director.py` | Add `_build_result_analysis_prompt()` | +35 lines |
| `kosmos/agents/research_director.py` | Update `evaluate_hypotheses_concurrently()` | +5 lines |
| `kosmos/agents/research_director.py` | Update `analyze_results_concurrently()` | +5 lines |

#### Implementation Details

**Step 1: Add Prompt Building Methods**

Location: `kosmos/agents/research_director.py` (new helper methods, ~line 1000)

```python
def _build_hypothesis_evaluation_prompt(self, hyp_id: str) -> str:
    """
    Build evaluation prompt with actual hypothesis data from database.

    Args:
        hyp_id: Hypothesis ID to load

    Returns:
        Formatted prompt string with hypothesis details, or fallback if unavailable
    """
    try:
        from kosmos.db.operations import get_hypothesis, get_session

        with get_session() as session:
            hypothesis = get_hypothesis(session, hyp_id, with_experiments=True)

            if hypothesis:
                # Build rich prompt with actual data
                related_papers = hypothesis.related_papers or []
                related_str = ', '.join(related_papers[:5]) if related_papers else 'None identified'

                testability = hypothesis.testability_score or 0.0
                novelty = hypothesis.novelty_score or 0.0

                return f"""Evaluate this hypothesis for testability and scientific merit:

## Hypothesis Details
- **ID**: {hyp_id}
- **Statement**: {hypothesis.statement}
- **Rationale**: {hypothesis.rationale or 'Not provided'}
- **Current Scores**: Testability={testability:.2f}, Novelty={novelty:.2f}

## Research Context
- **Research Question**: {self.research_question}
- **Domain**: {self.domain or 'General'}
- **Related Papers**: {related_str}

## Evaluation Criteria
Please provide:
1. Testability assessment (1-10): Can this hypothesis be empirically tested?
2. Scientific merit (1-10): Is this hypothesis novel and significant?
3. Suggested experimental approach: How would you test this?
4. Potential confounds: What factors could affect the results?
5. Confidence in evaluation (0-1): How certain are you of this assessment?
"""
    except ImportError:
        logger.debug("Database module not available for hypothesis loading")
    except Exception as e:
        logger.warning(f"Failed to load hypothesis {hyp_id}: {e}")

    # Fallback to basic prompt
    return f"""Evaluate this hypothesis for testability and scientific merit:

Hypothesis ID: {hyp_id}
Research Question: {self.research_question}
Domain: {self.domain or "General"}

Provide: testability (1-10), scientific merit (1-10), suggested approach.
"""


def _build_result_analysis_prompt(self, result_id: str) -> str:
    """
    Build analysis prompt with actual result data from database.

    Args:
        result_id: Result ID to load

    Returns:
        Formatted prompt string with result details, or fallback if unavailable
    """
    try:
        from kosmos.db.operations import get_result, get_session

        with get_session() as session:
            result = get_result(session, result_id)

            if result:
                # Get related experiment and hypothesis
                experiment = result.experiment
                hypothesis = experiment.hypothesis if experiment else None

                # Format data for prompt
                result_data = result.data or {}
                stats = result.statistical_tests or {}

                return f"""Analyze this experiment result:

## Result Details
- **Result ID**: {result_id}
- **Experiment**: {experiment.description if experiment else 'Unknown'}
- **Hypothesis Tested**: {hypothesis.statement if hypothesis else 'Unknown'}

## Research Context
- **Research Question**: {self.research_question}
- **Domain**: {self.domain or 'General'}

## Result Data
```json
{json.dumps(result_data, indent=2, default=str)}
```

## Statistical Tests
{json.dumps(stats, indent=2) if stats else 'No statistical tests performed'}

## Previous Interpretation
{result.interpretation or 'None available'}

## Analysis Required
Please provide:
1. Summary of key findings
2. Statistical significance assessment
3. Support/refute decision for hypothesis (with reasoning)
4. Confidence level (0-1)
5. Suggested follow-up experiments
"""
    except ImportError:
        logger.debug("Database module not available for result loading")
    except Exception as e:
        logger.warning(f"Failed to load result {result_id}: {e}")

    # Fallback to basic prompt
    return f"""Analyze this experiment result:

Result ID: {result_id}
Research Question: {self.research_question}

Provide: key findings, support/refute decision, confidence (0-1).
"""
```

**Step 2: Update Concurrent Methods**

Location: `kosmos/agents/research_director.py`, in `evaluate_hypotheses_concurrently()` (~line 1021)

```python
# Replace placeholder prompt with:
prompt = self._build_hypothesis_evaluation_prompt(hyp_id)
```

Location: `kosmos/agents/research_director.py`, in `analyze_results_concurrently()` (~line 1105)

```python
# Replace placeholder prompt with:
prompt = self._build_result_analysis_prompt(result_id)
```

#### Verification Checklist

- [ ] Prompt includes hypothesis statement and rationale
- [ ] Prompt includes related papers
- [ ] Prompt includes current scores
- [ ] Fallback works when database unavailable
- [ ] Result prompt includes experiment data
- [ ] Result prompt includes statistical tests

---

## Phase C: Performance Improvements

### C1. True Async LLM Providers

**Gap Reference**: `120525_implementation_gaps_v2.md` Section 3 (lines 304, 362)

**Problem**: `generate_async()` methods in both OpenAI and Anthropic providers just call the sync `generate()` method. No true async execution.

**Impact**: Concurrent hypothesis evaluation doesn't benefit from async; blocks event loop.

#### Files to Modify

| File | Changes | Lines |
|------|---------|-------|
| `kosmos/core/providers/openai.py` | Add `_async_client` property | +15 lines |
| `kosmos/core/providers/openai.py` | Implement true `generate_async()` | +45 lines |
| `kosmos/core/providers/anthropic.py` | Add `_async_client` property | +15 lines |
| `kosmos/core/providers/anthropic.py` | Implement true `generate_async()` | +45 lines |

#### Implementation Details

**OpenAI Provider**

Location: `kosmos/core/providers/openai.py`

```python
from openai import OpenAI, AsyncOpenAI

class OpenAIProvider(LLMProvider):
    def __init__(self, ...):
        # ... existing init ...
        self.client = OpenAI(api_key=api_key, base_url=base_url, organization=organization)

        # Lazy-initialized async client
        self._async_client: Optional[AsyncOpenAI] = None

    @property
    def async_client(self) -> AsyncOpenAI:
        """Lazy-initialize async client with same config as sync client."""
        if self._async_client is None:
            self._async_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                organization=self.organization,
                timeout=self.timeout if hasattr(self, 'timeout') else 60.0
            )
        return self._async_client

    async def generate_async(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None
    ) -> LLMResponse:
        """
        Generate text asynchronously using true async OpenAI client.

        CHECKPOINT C1: True async implementation
        """
        import time as time_module
        start_time = time_module.time()

        # Build messages
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            # Use async client for true async execution
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop_sequences
            )

            # Parse response (same as sync)
            content = response.choices[0].message.content or ""
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0

            duration = time_module.time() - start_time

            # Log if enabled
            self._log_call_if_enabled(input_tokens, output_tokens, duration)

            return LLMResponse(
                content=content,
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration_seconds=duration
            )

        except Exception as e:
            logger.error(f"Async OpenAI API error: {e}")
            raise ProviderAPIError(
                str(e),
                provider="openai",
                recoverable=self._is_recoverable_error(e) if hasattr(self, '_is_recoverable_error') else True
            )
```

**Anthropic Provider**

Location: `kosmos/core/providers/anthropic.py`

```python
from anthropic import Anthropic, AsyncAnthropic

class AnthropicProvider(LLMProvider):
    def __init__(self, ...):
        # ... existing init ...
        self.client = Anthropic(api_key=api_key, base_url=base_url)

        # Lazy-initialized async client
        self._async_client: Optional[AsyncAnthropic] = None

    @property
    def async_client(self) -> AsyncAnthropic:
        """Lazy-initialize async client with same config as sync client."""
        if self._async_client is None:
            self._async_client = AsyncAnthropic(
                api_key=self.api_key,
                base_url=self.base_url if hasattr(self, 'base_url') else None,
                timeout=self.timeout if hasattr(self, 'timeout') else 60.0
            )
        return self._async_client

    async def generate_async(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None
    ) -> LLMResponse:
        """
        Generate text asynchronously using true async Anthropic client.

        CHECKPOINT C1: True async implementation
        """
        import time as time_module
        start_time = time_module.time()

        try:
            # Use async client for true async execution
            response = await self.async_client.messages.create(
                model=self._resolve_model() if hasattr(self, '_resolve_model') else self.model,
                max_tokens=max_tokens,
                system=system or "",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                stop_sequences=stop_sequences or []
            )

            # Parse response (same as sync)
            content = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    content += block.text

            input_tokens = response.usage.input_tokens if response.usage else 0
            output_tokens = response.usage.output_tokens if response.usage else 0

            duration = time_module.time() - start_time

            # Log if enabled
            self._log_call_if_enabled(input_tokens, output_tokens, duration)

            return LLMResponse(
                content=content,
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration_seconds=duration
            )

        except Exception as e:
            logger.error(f"Async Anthropic API error: {e}")
            raise ProviderAPIError(
                str(e),
                provider="anthropic",
                recoverable=self._is_recoverable_error(e) if hasattr(self, '_is_recoverable_error') else True
            )
```

#### Verification Checklist

- [ ] `async_client` property lazily initializes
- [ ] `generate_async()` uses `await` with async client
- [ ] Response parsing matches sync version
- [ ] Logging and metrics still recorded
- [ ] Errors properly wrapped in `ProviderAPIError`

---

## Phase D: Future-Proofing

### D2. Neo4j Research Loop Integration

**Gap Reference**: `120525_implementation_gaps_v2.md` Section 1

**Problem**: Neo4j is fully implemented (1,025 lines) but not tested end-to-end and not verified as integrated into the research loop.

**Impact**: Knowledge graph functionality may not work in production.

#### Tasks

1. **Start Neo4j Instance**
   ```bash
   docker run -d \
     --name kosmos-neo4j \
     -p 7474:7474 -p 7687:7687 \
     -e NEO4J_AUTH=neo4j/kosmos-password \
     neo4j:5.15
   ```

2. **Set Environment Variables**
   ```bash
   export NEO4J_URI=bolt://localhost:7687
   export NEO4J_USER=neo4j
   export NEO4J_PASSWORD=kosmos-password
   ```

3. **Enable Skipped E2E Tests**
   - Remove `@pytest.mark.skip` from `tests/e2e/test_system_sanity.py:447`
   - Update `tests/conftest.py` to check for Neo4j connection

4. **Run E2E Tests**
   ```bash
   pytest tests/e2e/ -v -k neo4j
   ```

5. **Verify Integration Points**
   - `_persist_hypothesis_to_graph()` creates nodes
   - `_persist_protocol_to_graph()` creates relationships
   - `_persist_result_to_graph()` links results
   - `_add_support_relationship()` adds SUPPORTS/REFUTES

---

## Testing Strategy

### Test File Structure

```
tests/
├── unit/
│   ├── core/
│   │   ├── test_budget_enforcement.py      # A1
│   │   └── providers/
│   │       └── test_async_providers.py     # C1
│   ├── agents/
│   │   ├── test_error_recovery.py          # A2
│   │   └── test_prompt_building.py         # B2
│   └── world_model/
│       └── test_annotations.py             # B1
├── integration/
│   ├── test_budget_integration.py          # A1
│   ├── test_error_recovery_integration.py  # A2
│   └── test_annotation_persistence.py      # B1
└── e2e/
    └── test_neo4j_integration.py           # D2
```

### Test Coverage Requirements

| Task | Unit | Integration | E2E |
|------|------|-------------|-----|
| A1: Budget | 4 tests | 2 tests | 1 test |
| A2: Error Recovery | 5 tests | 2 tests | - |
| B1: Annotations | 4 tests | 2 tests | 1 test |
| B2: Data Loading | 3 tests | 1 test | - |
| C1: Async | 4 tests | 1 test | - |
| D2: Neo4j | - | - | 3 tests |

---

## Rollback Procedures

### A1: Budget Enforcement

**Rollback**: Remove `enforce_budget()` call from `decide_next_action()`. Budget tracking remains, just no enforcement.

### A2: Error Recovery

**Rollback**: Replace `_handle_error_with_recovery()` calls with original simple error handling:
```python
if message.type == MessageType.ERROR:
    logger.error(f"... failed: {content.get('error')}")
    self.errors_encountered += 1
    return
```

### B1: Annotations

**Rollback**: Revert `add_annotation()` and `get_annotations()` to stub implementations. Data in Neo4j remains but is unused.

### B2: Data Loading

**Rollback**: Replace `_build_*_prompt()` calls with original placeholder prompts.

### C1: Async

**Rollback**: In `generate_async()`, change back to:
```python
return self.generate(prompt, system, max_tokens, temperature, stop_sequences)
```

---

## Checkpoints

Use these checkpoints to verify progress at each stage.

### Checkpoint 1: After A1 (Budget Enforcement)

```bash
# Verify budget exception exists
python -c "from kosmos.core.metrics import BudgetExceededError; print('OK')"

# Run unit tests
pytest tests/unit/core/test_budget_enforcement.py -v

# Verify in logs (set low budget, run research)
# Should see: "[BUDGET] Research halted: Budget exceeded"
```

### Checkpoint 2: After A2 (Error Recovery)

```bash
# Run unit tests
pytest tests/unit/agents/test_error_recovery.py -v

# Verify backoff in logs (inject failure)
# Should see: "[ERROR-RECOVERY] Waiting Xs before retry"

# Verify circuit breaker
# Should see: "[ERROR-RECOVERY] Max consecutive errors reached"
```

### Checkpoint 3: After B1 (Annotations)

```bash
# Run unit tests
pytest tests/unit/world_model/test_annotations.py -v

# Verify round-trip (with Neo4j running)
python -c "
from kosmos.world_model.factory import get_world_model
from kosmos.world_model.models import Entity, Annotation

wm = get_world_model()
entity = Entity(type='Paper', properties={'title': 'Test'})
eid = wm.add_entity(entity)
wm.add_annotation(eid, Annotation(text='Test annotation', created_by='test'))
anns = wm.get_annotations(eid)
print(f'Annotations: {len(anns)}')  # Should be 1
"
```

### Checkpoint 4: After B2 (Data Loading)

```bash
# Run unit tests
pytest tests/unit/agents/test_prompt_building.py -v

# Verify prompt content (with database)
# Prompts should contain actual hypothesis.statement, not just ID
```

### Checkpoint 5: After C1 (Async)

```bash
# Run unit tests
pytest tests/unit/core/providers/test_async_providers.py -v

# Verify async execution
python -c "
import asyncio
from kosmos.core.providers.openai import OpenAIProvider

async def test():
    provider = OpenAIProvider(api_key='test')
    # Should use AsyncOpenAI, not fall back to sync
    print(f'Async client type: {type(provider.async_client).__name__}')

asyncio.run(test())
"
```

### Checkpoint 6: After D2 (Neo4j)

```bash
# Run E2E tests
pytest tests/e2e/test_system_sanity.py -v -k neo4j

# Verify graph population
cypher-shell -u neo4j -p kosmos-password \
  "MATCH (n) RETURN labels(n), count(*)"
```

---

## Summary

| Phase | Task | Status | Checkpoint |
|-------|------|--------|------------|
| A | A1: Budget Enforcement | Pending | CP1 |
| A | A2: Error Recovery | Pending | CP2 |
| B | B1: Annotation Storage | Pending | CP3 |
| B | B2: Data Loading | Pending | CP4 |
| C | C1: Async Providers | Pending | CP5 |
| D | D2: Neo4j Integration | Pending | CP6 |

**Total Estimated Effort**: 19-27 hours

---

## Appendix: File Change Summary

| File | Task | Changes |
|------|------|---------|
| `kosmos/core/metrics.py` | A1 | +27 lines (exception + method) |
| `kosmos/agents/research_director.py` | A1, A2, B2 | +120 lines |
| `kosmos/world_model/simple.py` | B1 | +70 lines |
| `kosmos/core/providers/openai.py` | C1 | +60 lines |
| `kosmos/core/providers/anthropic.py` | C1 | +60 lines |
| `tests/e2e/test_system_sanity.py` | D2 | Remove skip marker |

**Total New Lines**: ~340 lines (excluding tests)
