# Resume Prompt - Post Compaction

## Context

You are resuming work on the Kosmos project after a context compaction. The previous sessions implemented **13 paper implementation gaps** (3 BLOCKER + 5 Critical + 5 High).

## What Was Done

### All Fixed Issues

| Issue | Description | Implementation |
|-------|-------------|----------------|
| #66 | CLI Deadlock | Full async refactor of message passing |
| #67 | SkillLoader | Domain-to-bundle mapping fixed |
| #68 | Pydantic V2 | Model config migration complete |
| #54 | Self-Correcting Code Execution | Enhanced RetryStrategy with 11 error handlers + LLM repair |
| #55 | World Model Update Categories | UpdateType enum (CONFIRMATION/CONFLICT/PRUNING) + conflict detection |
| #56 | 12-Hour Runtime Constraint | `max_runtime_hours` config + runtime tracking in ResearchDirector |
| #57 | Parallel Task Execution | Changed `max_concurrent_experiments` default from 4 to 10 |
| #58 | Agent Rollout Tracking | New RolloutTracker class + integration in ResearchDirector |
| #59 | h5ad/Parquet Data Formats | `DataLoader.load_h5ad()` and `load_parquet()` methods |
| #69 | R Language Execution | `RExecutor` class + Docker image with TwoSampleMR |
| #60 | Figure Generation | `FigureManager` class + code template integration |
| #61 | Jupyter Notebook Generation | `NotebookGenerator` class + nbformat integration |
| #70 | Null Model Statistical Validation | `NullModelValidator` class + ScholarEval integration |

### Key Files Created/Modified (Recent)

| File | Changes |
|------|---------|
| `kosmos/validation/null_model.py` | **NEW** - NullModelValidator, NullModelResult classes (430+ lines) |
| `kosmos/validation/scholar_eval.py` | Integrated null model validation into evaluate_finding() |
| `kosmos/world_model/artifacts.py` | Added null_model_result field to Finding |
| `tests/unit/validation/test_null_model.py` | **NEW** - 45 unit tests |
| `tests/integration/validation/test_null_validation.py` | **NEW** - 19 integration tests |

## Remaining Work (4 gaps)

### Implementation Order

| Phase | Order | Issue | Description | Status |
|-------|-------|-------|-------------|--------|
| 3 | 5 | #70 | Null Model Statistical Validation | ✅ Complete |
| 3 | 6 | #63 | Failure Mode Detection | **Next** |
| 4 | 7 | #62 | Code Line Provenance | Pending |
| 5 | 8 | #64 | Multi-Run Convergence | Pending |
| 5 | 9 | #65 | Paper Accuracy Validation | Pending |

### Testing Requirements

- All tests must pass (no skipped tests except environment-dependent)
- Mock tests must be accompanied by real-world tests
- Do not proceed until current task is fully working

## Key Documentation

- `docs/CHECKPOINT.md` - Full session summary
- `docs/PAPER_IMPLEMENTATION_GAPS.md` - 17 tracked gaps (13 complete)
- `/home/jim/.claude/plans/peppy-floating-feather.md` - Full implementation plan
- GitHub Issues #54-#70 - Detailed tracking

## Quick Verification Commands

```bash
# Verify null model validation
python -c "
from kosmos.validation import NullModelValidator, NullModelResult, ScholarEvalValidator

# Test NullModelValidator
validator = NullModelValidator(n_permutations=100, random_seed=42)
finding = {
    'statistics': {
        'test_type': 't_test',
        'statistic': 3.5,
        'p_value': 0.001,
        'degrees_of_freedom': 50
    }
}
result = validator.validate_finding(finding)
print(f'Null model passes: {result.passes_null_test}')
print(f'Persists in noise: {result.persists_in_noise}')
print(f'P-value: {result.permutation_p_value:.4f}')

# Test ScholarEval integration
scholar = ScholarEvalValidator()
score = scholar.evaluate_finding(finding)
print(f'ScholarEval includes null_model_result: {score.null_model_result is not None}')
print(f'Statistical validity: {score.statistical_validity}')
print('All imports successful')
"

# Run tests
python -m pytest tests/unit/validation/test_null_model.py -v --tb=short
python -m pytest tests/integration/validation/test_null_validation.py -v --tb=short
```

## Resume Command

Start by reading the checkpoint:
```
Read docs/CHECKPOINT.md and docs/PAPER_IMPLEMENTATION_GAPS.md, then continue with the next item: #63 - Failure Mode Detection
```

## Progress Summary

**13/17 gaps fixed (76% complete)**

| Priority | Status |
|----------|--------|
| BLOCKER | 3/3 complete ✅ |
| Critical | 5/5 complete ✅ |
| High | 5/5 complete ✅ |
| Medium | 0/2 remaining |
| Low | 0/2 remaining |

## Next Step

Continue with **#63 - Failure Mode Detection**:
- Create `FailureDetector` class for identifying AI failure modes
- Implement over-interpretation detection (confidence score for claims vs evidence)
- Validate that claimed metrics exist in data (invented metrics detection)
- Add rabbit hole detection (relevance to original research question)
- Integrate with validation framework
