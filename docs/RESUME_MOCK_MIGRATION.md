# Resume Mock to Real Test Migration

## Context
Converting mock-based tests to real LLM API calls for production readiness.

## Completed
- **Phase 1: Core LLM tests** - 43 tests pass with real APIs

## Current Status
Ready to continue. Most tests only need Anthropic/DeepSeek keys (which are configured).

## Deferred (Waiting for Semantic Scholar API Key)
- `tests/unit/agents/test_hypothesis_generator.py` - Uses literature search

## Resume Task: Phase 2 - Knowledge Layer

### Files to Convert (No Semantic Scholar needed)
1. `tests/unit/knowledge/test_embeddings.py` - SentenceTransformer
2. `tests/unit/knowledge/test_concept_extractor.py` - Claude API
3. `tests/unit/knowledge/test_vector_db.py` - ChromaDB
4. `tests/unit/knowledge/test_graph.py` - Neo4j

### Then Phase 3 - Agents (1 deferred)
1. `tests/unit/agents/test_research_director.py` - Claude API
2. `tests/unit/agents/test_literature_analyzer.py` - Claude API + Neo4j
3. `tests/unit/agents/test_data_analyst.py` - Claude API
4. ~~`tests/unit/agents/test_hypothesis_generator.py`~~ - DEFERRED (needs Semantic Scholar)

### Pattern
```python
import os, pytest, uuid

pytestmark = [
    pytest.mark.requires_claude,
    pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="Requires API key")
]

def unique_prompt(base: str) -> str:
    return f"{base} [test-id: {uuid.uuid4().hex[:8]}]"
```

### Fixtures Available
- `real_anthropic_client`, `deepseek_client` - LLM clients
- `real_vector_db` - Ephemeral ChromaDB
- `real_embedder` - SentenceTransformer
- `real_knowledge_graph` - Neo4j

### Verify
```bash
pytest tests/unit/knowledge/ -v --no-cov
```
