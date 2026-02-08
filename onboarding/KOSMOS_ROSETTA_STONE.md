# KOSMOS Rosetta Stone

> **Purpose:** Single-source onboarding for AI coding assistants working in KOSMOS
> **Target audience:** Fresh Claude instances
> **Token budget:** ~12K tokens
> **Last verified:** 2026-01-26
> **Supersedes:** KOSMOS_ONBOARD.md, KOSMOS_ONBOARD_v2.md

---

## 1. Quick Start (30 seconds)

**What is KOSMOS?** An autonomous AI research platform that uses Claude-powered agents to conduct end-to-end scientific research cycles: hypothesis generation, experiment design, sandboxed execution, result analysis, and iterative refinement.

**Key command:**
```bash
kosmos run "What factors affect galaxy formation?" --domain astrophysics --iterations 5
```

**Critical first read:** `kosmos/config.py` (lines 1-100) — understand config singleton, CLI mode trick, caching

---

## 2. Architecture (2 minutes)

### 4-Layer Structure

```
┌─────────────────────────────────────────────────────────────┐
│ LEAF (Entry Points)                                          │
│   cli/, api/, tests/, examples/                              │
├─────────────────────────────────────────────────────────────┤
│ ORCHESTRATION (Workflow Coordination)                        │
│   workflow.py, research_loop.py, cache_manager.py            │
├─────────────────────────────────────────────────────────────┤
│ CORE (Business Logic & Agents)                               │
│   research_director.py, hypothesis_generator.py,             │
│   experiment_designer.py, executor.py, data_analyst.py       │
├─────────────────────────────────────────────────────────────┤
│ FOUNDATION (Shared Infrastructure)                           │
│   logging.py, config.py, hypothesis.py, experiment.py,       │
│   llm.py, base_client.py                                     │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Research Question (CLI/API)
    │
    ▼
ResearchDirectorAgent                    ─── agents/research_director.py
    │
    ├──▶ get_config()                   ─── config.py (singleton)
    ├──▶ ResearchWorkflow               ─── core/workflow.py (state machine)
    │
    ▼
State: GENERATING_HYPOTHESES
    ├──▶ HypothesisGeneratorAgent       ─── agents/hypothesis_generator.py
    │       └──▶ Literature Search      ─── literature/unified_search.py
    │       └──▶ NoveltyChecker         ─── hypothesis/novelty_checker.py
    ▼
State: DESIGNING_EXPERIMENTS
    ├──▶ ExperimentDesignerAgent        ─── agents/experiment_designer.py
    ▼
State: EXECUTING
    ├──▶ Executor                       ─── execution/executor.py
    │       └──▶ Sandbox (Docker)       ─── execution/sandbox.py
    ▼
State: ANALYZING
    ├──▶ DataAnalystAgent               ─── agents/data_analyst.py
    ▼
State: REFINING or CONVERGED
    └──▶ Loop or Finish
```

### Circular Dependency (Intentional)

`kosmos.core.llm` ↔ `kosmos.core.providers.anthropic` — This is the provider factory pattern. `llm.py` provides the abstract interface and `get_client()`, `anthropic.py` implements it. Do NOT "fix" this.

---

## 3. Critical Files Guide

### Files to Read First

| Priority | File | Lines | What You Learn |
|----------|------|-------|----------------|
| 1 | `kosmos/config.py` | 1-150 | Config singleton, CLI mode trick, caching |
| 2 | `kosmos/core/workflow.py` | 1-200 | WorkflowState enum, state machine |
| 3 | `kosmos/models/hypothesis.py` | 1-150 | Core data model, validators |
| 4 | `kosmos/core/llm.py` | 1-200 | LLM abstraction, provider selection |
| 5 | `kosmos/agents/base.py` | all | BaseAgent interface |
| 6 | `kosmos/agents/research_director.py` | 1-100 | Master orchestrator (skeleton only!) |

### Files to NEVER Read Fully (Context Hazards)

| File | Size | Why Skip |
|------|------|----------|
| `agents/research_director.py` | ~21K tokens | Use skeleton (first 100 lines) only |
| `document.py` (scientific-skills) | ~12K tokens | External skill, not core |
| `workflow/ensemble.py` | ~10K tokens | Read only if needed |
| `tests/requirements/test_req_*.py` | ~10K+ each | Skip unless debugging |

### File Size Reference (Core Package)

As of 2026-01-26:
- **Core kosmos/ files:** ~189 Python files
- **Full repository:** ~1500+ Python files (includes skills, reference, tests)
- **Test files:** 205

---

## 4. Key Patterns

### Singleton Pattern

```python
# Configuration - MUST load first
from kosmos.config import get_config, reset_config
config = get_config()  # Returns cached instance

# LLM Client
from kosmos.core.llm import get_client
client = get_client()  # Auto-selects provider based on config

# Database Session
from kosmos.db import get_session, init_from_config
init_from_config()  # Idempotent - safe to call multiple times
```

**Testing:** Always call `reset_config()`, `reset_world_model()` in test teardown.

### Agent Pattern

```python
class MyAgent(BaseAgent):
    def __init__(self, agent_id, config):
        super().__init__(agent_id, agent_type="my_agent", config=config)

    async def execute(self, message: AgentMessage) -> AgentMessage:
        # 1. Validate input
        # 2. Perform domain logic
        # 3. Call LLM if needed (via self.llm_client)
        # 4. Persist results
        # 5. Return message with results
```

### State Machine (WorkflowState)

```python
class WorkflowState(str, Enum):
    INITIALIZING = "initializing"
    GENERATING_HYPOTHESES = "generating_hypotheses"
    DESIGNING_EXPERIMENTS = "designing_experiments"
    EXECUTING = "executing"
    ANALYZING = "analyzing"
    REFINING = "refining"
    CONVERGED = "converged"
    PAUSED = "paused"      # Additional state
    ERROR = "error"        # Additional state

# Transitions are strictly validated via ALLOWED_TRANSITIONS dict (line 175)
```

### Error Recovery Constants

```python
# research_director.py:45-46
MAX_CONSECUTIVE_ERRORS = 3
ERROR_BACKOFF_SECONDS = [2, 4, 8]  # Exponential backoff
```

After 3 consecutive errors, execution halts.

---

## 5. Gotchas & Edge Cases

### CLI Mode Trick (Zero API Cost Development)

```python
# config.py:79-82
@property
def is_cli_mode(self) -> bool:
    """API key of all 9s enables CLI mode."""
    return self.api_key.replace('9', '') == ''
```

Set `ANTHROPIC_API_KEY=999999999999999999999999999999999999999999` to route through Claude Code CLI instead of API.

### Config Load Order

Config MUST be loaded before other singletons. Many depend on `get_config()`:
```python
# CORRECT
config = get_config()
world_model = get_world_model()

# WRONG - may fail
world_model = get_world_model()  # Config not loaded!
```

### Prompt Caching (90% Cost Reduction)

```python
# config.py:60-64
class ClaudeConfig(BaseSettings):
    enable_cache: bool = True  # DEFAULT: enabled
```

Claude API prompt caching is ON by default. System prompts are cached, reducing token costs dramatically for repetitive operations.

### Hypothesis Validation Rules

```python
# models/hypothesis.py - @field_validator('statement')
# Hypothesis MUST be:
# - A predictive statement (not a question)
# - Cannot end with '?'
# - Should contain predictive words: 'will', 'increases', 'causes', 'affects'
```

### Database Init is Idempotent

```python
# Safe to call multiple times
init_from_config()  # Catches "already initialized" RuntimeError
```

### Async Locks (Not Threading)

Issue #66 converted `threading.RLock` to `asyncio.Lock` in ResearchDirector for async safety:
```python
# research_director.py:179-181
self._research_plan_lock = asyncio.Lock()
```

---

## 6. Common Tasks

### Add New Agent

1. Start: `kosmos/agents/base.py` — understand `BaseAgent` interface
2. Example: `kosmos/agents/hypothesis_generator.py` — working implementation
3. Register: Add to `kosmos/agents/registry.py` using `register_agent()`
4. Test: Create tests in `tests/unit/agents/test_your_agent.py`

### Add Literature Source

1. Interface: `kosmos/literature/base_client.py` — implement `BaseLiteratureClient`
2. Methods: `search()`, `get_paper_by_id()`, `download_pdf()`
3. Register: Add to `kosmos/literature/unified_search.py`

### Modify Workflow

1. States: `kosmos/core/workflow.py` — `WorkflowState` enum
2. Transitions: Update `ALLOWED_TRANSITIONS` dict
3. Test: Verify state graph in `tests/unit/core/test_workflow.py`

### Debug Experiments

1. Entry: `kosmos/execution/executor.py`
2. Sandbox: `kosmos/execution/sandbox.py` — Docker container execution
3. Logs: Check `artifacts/{experiment_id}/` for output
4. Code gen: `kosmos/execution/code_generator.py`
5. Data provider: `kosmos/execution/data_provider.py` (Issue #51 fix)

### Run Tests

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# Specific marker
pytest tests/ -v -m unit

# With coverage
pytest tests/unit/ -v --cov=kosmos
```

---

## 7. Environment Setup

### Required Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `ANTHROPIC_API_KEY` | **Yes** | Claude API (or "999..." for CLI mode) |

### Optional Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `CLAUDE_MODEL` | claude-sonnet-4-5 | Model version |
| `CLAUDE_MAX_TOKENS` | 4096 | Token limit |
| `CLAUDE_TEMPERATURE` | 0.7 | Sampling temp |
| `CLAUDE_TIMEOUT` | 120 | Request timeout (seconds) |
| `DATABASE_URL` | sqlite:///kosmos.db | Database connection |
| `NEO4J_PASSWORD` | — | Knowledge graph (optional) |
| `LITELLM_API_KEY` | — | LiteLLM provider |
| `SEMANTIC_SCHOLAR_API_KEY` | — | Higher rate limit (5000 vs 100 req/5min) |
| `KOSMOS_LOG_LEVEL` | INFO | Logging verbosity |
| `KOSMOS_LOG_FORMAT` | json | json or text |

### Provider Configuration

```bash
# Anthropic (default)
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your-key

# OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=your-key

# LiteLLM (100+ providers)
LLM_PROVIDER=litellm
LITELLM_API_KEY=your-key
LITELLM_API_BASE=your-base-url
```

### Configuration Files

| File | Purpose |
|------|---------|
| `.env` | Local overrides (not committed) |
| `.env.example` | Template with all variables |
| `pyproject.toml` | Dependencies, ruff config |
| `alembic.ini` | Database migrations |

---

## 8. Quick Reference

### Key Imports

```python
# Configuration
from kosmos.config import get_config, reset_config

# Logging
from kosmos.core.logging import get_logger, setup_logging

# LLM
from kosmos.core.llm import get_client

# Data Models
from kosmos.models.hypothesis import Hypothesis, HypothesisStatus
from kosmos.models.experiment import ExperimentProtocol, ExperimentType
from kosmos.models.result import ExperimentResult

# Agents
from kosmos.agents.research_director import ResearchDirectorAgent
from kosmos.agents.hypothesis_generator import HypothesisGeneratorAgent
from kosmos.agents.experiment_designer import ExperimentDesignerAgent
from kosmos.agents.data_analyst import DataAnalystAgent

# Workflow
from kosmos.core.workflow import ResearchWorkflow, WorkflowState, ResearchPlan

# Database
from kosmos.db import get_session, init_from_config

# Knowledge
from kosmos.knowledge.graph import get_knowledge_graph
from kosmos.knowledge.vector_db import get_vector_db
```

### CLI Commands

```bash
# Run research
kosmos run "research question" --domain biology --iterations 5

# Interactive mode
kosmos interactive

# Check status
kosmos status

# View knowledge graph
kosmos graph show
```

### Linter Rules (ruff)

```toml
[tool.ruff]
line-length = 100
select = ["E", "W", "F", "I", "B"]
ignore = ["E501", "B008", "C901"]
target-version = "py311"
```

**Translation:** Max 100 chars/line (but E501 ignored), imports sorted (I), no bare excepts (B), complexity warnings disabled (C901).

---

## 9. Debugging Guide

| Symptom | Check First | Common Cause |
|---------|-------------|--------------|
| LLM call hangs | `ANTHROPIC_API_KEY` | Invalid key, rate limited, network timeout |
| Experiment fails silently | `docker ps` | Container not running, Docker daemon down |
| Import error on startup | `pip install -e .` | Package not installed in dev mode |
| Config not found | Working directory | Must run from project root with `.env` |
| Database not initialized | `init_from_config()` | Missing DB init in entry point |
| Circular import error | Import order | Config/logging must import first |
| Hypothesis validation fails | Statement ends with `?` | Must be statement, not question |
| Rate limit exceeded | API key | Missing key or quota exceeded |

---

## 10. Verified Components (2026-01-26)

All claims in this document were verified against source code:

| Component | File:Line | Verified |
|-----------|-----------|----------|
| MAX_CONSECUTIVE_ERRORS=3 | research_director.py:45 | ✅ |
| ERROR_BACKOFF_SECONDS=[2,4,8] | research_director.py:46 | ✅ |
| asyncio.Lock usage | research_director.py:179-181 | ✅ |
| WorkflowState (9 states) | workflow.py:~165 | ✅ |
| ALLOWED_TRANSITIONS | workflow.py:175 | ✅ |
| is_cli_mode property | config.py:79-82 | ✅ |
| enable_cache=True default | config.py:60-64 | ✅ |
| get_config() singleton | config.py:1074 | ✅ |
| session.commit() | experiment_designer.py:783 | ✅ |
| session.commit() | hypothesis_generator.py:469 | ✅ |
| cursor.execute() | experiment_cache.py:247 | ✅ |
| ConvergenceDetector direct call | research_director.py:33,156 | ✅ |
| LITELLM_API_KEY | .env.example:89 | ✅ |
| data_provider.py (Issue #51) | execution/data_provider.py | ✅ |

---

*This Rosetta Stone consolidates and verifies content from KOSMOS_ONBOARD.md, KOSMOS_ONBOARD_v2.md, and kosmos_xray_output_v31.md. For raw X-Ray scan data, see kosmos_xray_output_v31.md.*
