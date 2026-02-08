# Kosmos RNAi Designer - Comprehensive Implementation Plan

**Project**: Kosmos "Greenhouse" (Agro-Bio Integration)  
**Objective**: Enable the Kosmos agent to autonomously design, test, and validate eco-friendly gene-silencing (RNAi) agents for pest control.  
**Goal**: Give Kosmos the ability to read a pest's genetic code, find essential genes (weak spots), and generate weaponized RNA sequences that hit the target pest and nothing else.

---

## ⚠️ CRITICAL FIXES APPLIED (v1.1)

This plan has been corrected to address three critical issues that would have caused system failure:

### 1. ✅ Added Missing "Weaponizer" Tool
**Problem**: Original plan had agents "guessing" 21-mer sequences from 3000bp genes - impossible.  
**Fix**: Added `generate_sirna_candidates()` as **Phase 1b** (required immediately, not future).  
**Location**: New file `kosmos/tools/rnai_generator.py` with Reynolds Rules implementation.

### 2. ✅ Fixed Essentiality Detection Logic
**Problem**: Assumed transcriptomes come with `knockout_lethality: 0.95` metadata - they don't.  
**Fix**: Changed to homology mapping - BLAST pest genes against Drosophila Essential Genes (DEG) database.  
**Location**: Updated `find_essential_genes()` implementation in Phase 1.

### 3. ✅ Fixed Docker Image Bloat
**Problem**: COPY genomes into Dockerfile would create 5GB+ images.  
**Fix**: Use mounted volumes (`-v ./kosmos_data:/data`) with initialization script that downloads genomes to host on first run.  
**Location**: Updated Docker strategy in Integration Points section.

---

## Executive Summary

This plan outlines the implementation of **three critical tools** for autonomous RNAi design:

1. **Phase 1a: Genomic Search Tool** (`find_essential_genes`) - Identifies essential genes in pest organisms via homology mapping
2. **Phase 1b: SiRNA Generator Tool** (`generate_sirna_candidates`) - **CRITICAL**: Converts long gene sequences into valid 21-mer siRNA candidates
3. **Phase 2: Safety Filter Tool** (`check_off_target_risk`) - Validates RNA sequences against protected species

**Critical Fix**: The original plan missed the "weaponizer" step - agents cannot guess valid 21-mer sequences from 3000bp genes. The `generate_sirna_candidates` tool is **required for Phase 1**, not a future enhancement.

These tools integrate into the existing Kosmos tool ecosystem, following the established patterns for BioLab tools while adding genomics-specific capabilities.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Phase 1: Genomic Search Tool](#phase-1-genomic-search-tool)
3. [Phase 2: Safety Filter Tool](#phase-2-safety-filter-tool)
4. [Integration Points](#integration-points)
5. [Dependencies & Infrastructure](#dependencies--infrastructure)
6. [Testing Strategy](#testing-strategy)
7. [Implementation Timeline](#implementation-timeline)
8. [Risk Assessment](#risk-assessment)

---

## Architecture Overview

### Current System Context

Kosmos has:
- **Tool Registry System** (`kosmos/tools/__init__.py`) - Centralized tool definitions and registration
- **BioLab Tools** (`kosmos/tools/bio_lab.py`) - Computational biology tools (structure prediction, docking, MD)
- **Docker Execution** (`kosmos/execution/`) - Sandboxed execution with specialized images
- **Biology Domain** (`kosmos/domains/biology/`) - API clients for biological databases (Ensembl, KEGG, UniProt, etc.)
- **Agent System** - Multi-agent orchestration for autonomous research

### New Components

```
kosmos/
├── tools/
│   ├── __init__.py                    # [MODIFY] Add RNAi tool definitions
│   ├── rnai_designer.py              # [NEW] RNAi design tool implementations
│   └── rnai_generator.py             # [NEW] CRITICAL: SiRNA candidate generator
├── domains/
│   └── biology/
│       ├── apis.py                    # [MODIFY] Add InsectBase/NCBI client
│       └── genomics/
│           ├── rnai.py               # [NEW] RNAi-specific genomics logic
│           └── essential_genes.py   # [NEW] Drosophila DEG database handler
└── execution/
    └── docker_manager.py             # [MODIFY] Add RNAi-specific container config
```

### Tool Flow

```
Agent Request
    ↓
find_essential_genes(organism_name)
    ↓
[Download pest transcriptome from InsectBase/NCBI]
    ↓
[Download Drosophila Essential Genes (DEG) database]
    ↓
[BLAST pest genes against DEG - homology mapping]
    ↓
[Tag genes matching lethal fly genes as essential]
    ↓
Return: List of essential genes with sequences (3000bp+)
    ↓
Agent selects target gene (e.g., "Chitin Synthase" - 3000bp)
    ↓
**generate_sirna_candidates(gene_sequence)** ← CRITICAL STEP
    ↓
[Slide 21bp window, filter by Reynolds Rules]
    ↓
Return: Top 10 valid 21-mer candidates
    ↓
For each candidate:
    ↓
check_off_target_risk(candidate_sequence, protected_species)
    ↓
[BLAST search against protected genomes]
    ↓
[Check for matches in Honeybees, Monarch Butterflies]
    ↓
Return: APPROVE (0 matches) or REJECT (matches found)
    ↓
Agent selects approved candidate(s)
```

---

## Phase 1: Genomic Search Tool

### Tool Definition

**Name**: `find_essential_genes`  
**Category**: `ToolCategory.BIOLAB`  
**Description**: Downloads transcriptome data for a target pest organism and identifies essential genes (genes required for survival, typically with lethal knockout phenotypes).

### Function Signature

```python
def find_essential_genes(
    organism_name: str,
    data_source: str = "auto",  # "insectbase", "ncbi", "ensembl", "auto"
    essentiality_criteria: Optional[Dict[str, Any]] = None,
    output_dir: str = DEFAULT_OUTPUT_DIR
) -> Dict[str, Any]:
    """
    Find essential genes in a target pest organism.
    
    Args:
        organism_name: Scientific name or common name of pest (e.g., "Helicoverpa zea", "corn earworm")
        data_source: Database to query ("insectbase", "ncbi", "ensembl", or "auto" for best available)
        essentiality_criteria: Optional criteria dict:
            - min_knockout_lethality: float (0.0-1.0) - Minimum lethality rate
            - gene_functions: List[str] - Filter by GO terms (e.g., ["chitin synthesis", "V-ATPase"])
            - exclude_hypothetical: bool - Exclude hypothetical proteins
        output_dir: Directory to save downloaded data and results
    
    Returns:
        Dict containing:
        - success: bool
        - essential_genes: List[Dict] with:
            - gene_id: str
            - gene_name: str
            - sequence: str (DNA/RNA sequence in AGCT format)
            - sequence_type: str ("cds", "mrna", "genomic")
            - essentiality_score: float (0.0-1.0)
            - gene_function: str
            - knockout_phenotype: str
            - data_source: str
        - organism_info: Dict with organism metadata
        - total_genes_found: int
        - execution_time_seconds: float
        - error_message: Optional[str]
    """
```

### Implementation Details

#### 1.1 Data Source Integration

**InsectBase API Client** (`kosmos/domains/biology/apis.py`)

```python
class InsectBaseClient:
    """
    Client for InsectBase database (http://www.insectbase.org/).
    
    InsectBase provides transcriptome and genome data for insect species,
    including essential gene annotations.
    """
    
    BASE_URL = "http://www.insectbase.org/api/v1"
    
    async def search_organism(self, organism_name: str) -> Dict[str, Any]:
        """Search for organism by name."""
        
    async def get_transcriptome(self, organism_id: str) -> Dict[str, Any]:
        """Download transcriptome data."""
        
    async def get_essential_genes(
        self, 
        organism_id: str,
        criteria: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Get essential genes with lethality annotations."""
```

**NCBI Entrez Integration** (using existing `biopython`)

```python
from Bio import Entrez, SeqIO

class NCBIGenomicsClient:
    """
    Client for NCBI genomics databases.
    
    Uses Entrez API to search for organism genomes and transcriptomes.
    """
    
    def search_genome(self, organism_name: str) -> List[str]:
        """Search for genome assembly IDs."""
        
    def download_transcriptome(self, assembly_id: str) -> Dict[str, Any]:
        """Download transcriptome sequences."""
```

**Ensembl Metazoa** (using existing `pyensembl`)

```python
# Leverage existing EnsemblClient in kosmos/domains/biology/apis.py
# Extend to support invertebrate species via Ensembl Metazoa
```

#### 1.2 Essential Gene Identification via Homology Mapping

**CRITICAL FIX**: Transcriptomes are raw sequences without lethality metadata. Essentiality must be inferred via homology mapping.

**Method**: BLAST pest genes against Drosophila Essential Genes (DEG) database.

**Drosophila Essential Genes Database**:
- Source: FlyBase (http://flybase.org/) or DEG database
- Contains ~2000 genes with documented lethal/null lethal phenotypes
- Well-curated, experimentally validated
- Used as reference for homology-based essentiality prediction

**Implementation**:

```python
def _identify_essential_genes_via_homology(
    pest_transcriptome: Dict[str, str],  # gene_id -> sequence
    deg_database_path: str,
    output_dir: str
) -> List[Dict[str, Any]]:
    """
    Identify essential genes by BLASTing pest sequences against Drosophila DEG.
    
    Algorithm:
    1. Download/load Drosophila Essential Genes database
    2. Create BLAST database from DEG sequences
    3. For each pest gene:
       - BLAST against DEG database
       - If match found (E-value < 1e-50, identity >70%):
         - Tag as essential
         - Copy DEG annotation (gene name, function, phenotype)
    4. Return essential genes with homology evidence
    
    Returns:
        List of essential genes with:
        - gene_id: str
        - sequence: str (full gene sequence)
        - essentiality_score: float (1.0 if matches lethal DEG, 0.8 if matches null lethal)
        - homologous_drosophila_gene: str
        - gene_function: str (from DEG annotation)
        - knockout_phenotype: str (from DEG: "lethal", "null lethal", etc.)
    """
    from Bio.Blast.Applications import NcbiblastnCommandline
    from Bio.Blast import NCBIXML
    
    # Step 1: Ensure DEG database exists
    deg_db = _ensure_deg_database(deg_database_path)
    
    # Step 2: BLAST each pest gene
    essential_genes = []
    for gene_id, sequence in pest_transcriptome.items():
        blast_result = _blast_against_deg(sequence, deg_db)
        
        if blast_result and blast_result.e_value < 1e-50:
            # Strong homology - likely essential
            essential_genes.append({
                "gene_id": gene_id,
                "sequence": sequence,
                "essentiality_score": 1.0 if "lethal" in blast_result.phenotype else 0.8,
                "homologous_drosophila_gene": blast_result.drosophila_gene_id,
                "gene_function": blast_result.gene_function,
                "knockout_phenotype": blast_result.phenotype,
                "blast_e_value": blast_result.e_value,
                "identity_percent": blast_result.identity,
            })
    
    return essential_genes
```

**DEG Database Download**:

```python
def _ensure_deg_database(output_dir: str) -> str:
    """
    Download Drosophila Essential Genes database if not present.
    
    Sources:
    1. FlyBase (preferred) - http://flybase.org/
    2. DEG database - http://www.essentialgene.org/
    3. NCBI Gene database (fallback)
    
    Returns:
        Path to BLAST database directory
    """
    deg_fasta = Path(output_dir) / "drosophila_essential_genes.fasta"
    deg_db = Path(output_dir) / "deg_blastdb"
    
    if not deg_db.exists():
        # Download DEG sequences
        _download_deg_from_flybase(deg_fasta)
        
        # Create BLAST database
        subprocess.run([
            "makeblastdb",
            "-in", str(deg_fasta),
            "-dbtype", "nucl",
            "-out", str(deg_db / "deg"),
            "-title", "Drosophila Essential Genes"
        ])
    
    return str(deg_db)
```

#### 1.3 Sequence Extraction

**Format Requirements**:
- Raw nucleotide sequence (AGCT format)
- No headers, no formatting
- CDS (coding sequence) preferred, fallback to mRNA, then genomic

**Implementation**:

```python
def _extract_sequence(
    gene_record: Dict[str, Any],
    sequence_type: str = "cds"
) -> str:
    """
    Extract raw sequence from gene record.
    
    Returns:
        Raw sequence string (AGCT...) without headers
    """
```

### Data Sources Priority

1. **Primary**: InsectBase (specialized for insects, includes essentiality annotations)
2. **Secondary**: NCBI GenBank/RefSeq (broader coverage, may need manual annotation)
3. **Tertiary**: Ensembl Metazoa (for well-annotated species)

### Error Handling

- **Organism not found**: Return clear error with suggestions for alternative names
- **No transcriptome available**: Suggest alternative data sources
- **Network failures**: Retry with exponential backoff (use existing `tenacity` patterns)
- **Incomplete data**: Return partial results with warnings

---

## Phase 1b: SiRNA Generator Tool (CRITICAL - Required for Phase 1)

### Tool Definition

**Name**: `generate_sirna_candidates`  
**Category**: `ToolCategory.BIOLAB`  
**Description**: **CRITICAL MISSING LINK** - Converts long gene sequences (3000bp+) into valid 21-mer siRNA candidates using Reynolds Rules for efficacy prediction. This tool is required immediately after `find_essential_genes` - agents cannot "guess" valid siRNA sequences.

### Function Signature

```python
def generate_sirna_candidates(
    gene_sequence: str,
    window_size: int = 21,
    max_candidates: int = 10,
    min_gc_content: float = 0.30,
    max_gc_content: float = 0.52,
    output_dir: str = DEFAULT_OUTPUT_DIR
) -> Dict[str, Any]:
    """
    Generate valid siRNA candidate sequences from a gene sequence.
    
    This is the "weaponizer" - converts long gene sequences into targetable
    21-mer RNAi agents using established design rules.
    
    Args:
        gene_sequence: Full gene sequence (DNA, AGCT format, typically 1000-5000bp)
        window_size: Length of siRNA candidates (default: 21, standard siRNA length)
        max_candidates: Maximum number of candidates to return (default: 10)
        min_gc_content: Minimum GC content (default: 0.30, Reynolds Rule)
        max_gc_content: Maximum GC content (default: 0.52, Reynolds Rule)
        output_dir: Directory to save analysis results
    
    Returns:
        Dict containing:
        - success: bool
        - candidates: List[Dict] with:
            - sequence: str (21-mer sequence)
            - start_position: int (position in original gene)
            - end_position: int
            - gc_content: float
            - reynolds_score: float (0.0-1.0, higher = better)
            - has_repeats: bool (True if contains 4+ same base)
            - thermodynamic_stability: float (lower = better for 5' end)
        - total_candidates_generated: int
        - filtered_candidates: int (after applying filters)
        - execution_time_seconds: float
        - error_message: Optional[str]
    """
```

### Implementation Details

#### Reynolds Rules for SiRNA Design

**Reynolds et al. (2004) - "Rational siRNA design for RNA interference"**:
1. **GC Content**: 30-52% (optimal: 40-50%)
2. **No Homopolymer Runs**: Avoid 4+ consecutive identical bases (e.g., AAAA, GGGG)
3. **Thermodynamic Stability**: Lower stability at 5' end (preferred)
4. **Avoid Internal Secondary Structure**: Minimize hairpins/loops
5. **Position Preference**: Avoid sequences near start/stop codons

**Implementation**:

```python
def _generate_sirna_candidates(
    sequence: str,
    window_size: int = 21,
    min_gc: float = 0.30,
    max_gc: float = 0.52
) -> List[Dict[str, Any]]:
    """
    Slide window across sequence and generate candidates.
    
    Algorithm:
    1. Slide 21bp window across sequence (step size: 1bp)
    2. For each window:
       a. Calculate GC content
       b. Check for homopolymer runs (4+ same base)
       c. Calculate thermodynamic stability (5' end)
       d. Check for secondary structure potential
    3. Filter by Reynolds Rules
    4. Score and rank candidates
    5. Return top N candidates
    """
    candidates = []
    
    # Slide window
    for i in range(len(sequence) - window_size + 1):
        candidate_seq = sequence[i:i + window_size].upper()
        
        # Skip if contains invalid characters
        if not all(c in 'AGCT' for c in candidate_seq):
            continue
        
        # Calculate GC content
        gc_count = candidate_seq.count('G') + candidate_seq.count('C')
        gc_content = gc_count / window_size
        
        # Filter by GC content
        if gc_content < min_gc or gc_content > max_gc:
            continue
        
        # Check for homopolymer runs (4+ same base)
        has_repeats = _has_homopolymer_run(candidate_seq, min_length=4)
        if has_repeats:
            continue
        
        # Calculate thermodynamic stability (5' end)
        # Use nearest-neighbor model (simplified)
        stability_5prime = _calculate_5prime_stability(candidate_seq[:4])
        
        # Calculate Reynolds score (higher = better)
        reynolds_score = _calculate_reynolds_score(
            gc_content=gc_content,
            stability_5prime=stability_5prime,
            has_repeats=has_repeats
        )
        
        candidates.append({
            "sequence": candidate_seq,
            "start_position": i,
            "end_position": i + window_size,
            "gc_content": gc_content,
            "reynolds_score": reynolds_score,
            "has_repeats": has_repeats,
            "thermodynamic_stability": stability_5prime,
        })
    
    # Sort by Reynolds score (descending)
    candidates.sort(key=lambda x: x["reynolds_score"], reverse=True)
    
    return candidates

def _has_homopolymer_run(sequence: str, min_length: int = 4) -> bool:
    """Check for homopolymer runs (e.g., AAAA, GGGG)."""
    current_base = None
    run_length = 0
    
    for base in sequence:
        if base == current_base:
            run_length += 1
            if run_length >= min_length:
                return True
        else:
            current_base = base
            run_length = 1
    
    return False

def _calculate_5prime_stability(sequence_5prime: str) -> float:
    """
    Calculate thermodynamic stability at 5' end.
    
    Simplified: Use base pairing energies.
    Lower stability = better (easier for RISC to load).
    """
    # Nearest-neighbor model (simplified)
    # A-T: -0.9 kcal/mol, G-C: -1.8 kcal/mol
    stability = 0.0
    for i in range(len(sequence_5prime) - 1):
        pair = sequence_5prime[i:i+2]
        if pair in ['GC', 'CG']:
            stability -= 1.8
        elif pair in ['AT', 'TA']:
            stability -= 0.9
        # Mismatches add positive energy (less stable)
    
    return stability

def _calculate_reynolds_score(
    gc_content: float,
    stability_5prime: float,
    has_repeats: bool
) -> float:
    """
    Calculate composite Reynolds score (0.0-1.0).
    
    Higher score = better candidate.
    """
    if has_repeats:
        return 0.0  # Disqualify
    
    # GC content score (optimal: 0.40-0.50)
    if 0.40 <= gc_content <= 0.50:
        gc_score = 1.0
    elif 0.30 <= gc_content < 0.40 or 0.50 < gc_content <= 0.52:
        gc_score = 0.7
    else:
        gc_score = 0.3
    
    # Stability score (lower = better, but not too low)
    if -3.0 <= stability_5prime <= -1.0:
        stability_score = 1.0
    elif -5.0 <= stability_5prime < -3.0:
        stability_score = 0.8
    else:
        stability_score = 0.5
    
    # Composite score (weighted average)
    reynolds_score = (gc_score * 0.6) + (stability_score * 0.4)
    
    return reynolds_score
```

### File Location

**New File**: `kosmos/tools/rnai_generator.py`

This tool is **required immediately** - it cannot be deferred to "future enhancements" as the agent workflow depends on it.

---

## Phase 2: Safety Filter Tool

### Tool Definition

**Name**: `check_off_target_risk`  
**Category**: `ToolCategory.BIOLAB`  
**Description**: Validates RNAi candidate sequences against protected species genomes to ensure no off-target effects.

### Function Signature

```python
def check_off_target_risk(
    candidate_sequence: str,
    protected_species_list: List[str] = None,
    match_threshold: int = 21,  # Minimum match length for rejection
    output_dir: str = DEFAULT_OUTPUT_DIR
) -> Dict[str, Any]:
    """
    Check if RNAi candidate sequence matches protected species genomes.
    
    Args:
        candidate_sequence: RNA sequence to test (21-30 nucleotides, AGCT format)
        protected_species_list: List of protected species to check against.
            Default: ["Apis mellifera" (honeybee), "Danaus plexippus" (monarch)]
        match_threshold: Minimum consecutive matching nucleotides to trigger rejection
            (default: 21, standard siRNA length)
        output_dir: Directory to save BLAST results
    
    Returns:
        Dict containing:
        - success: bool
        - approved: bool (True if 0 matches found)
        - rejected: bool (True if matches found)
        - matches: List[Dict] with:
            - species: str
            - gene_id: str
            - match_start: int
            - match_end: int
            - match_length: int
            - e_value: float
            - identity_percent: float
        - total_species_checked: int
        - execution_time_seconds: float
        - error_message: Optional[str]
    """
```

### Implementation Details

#### 2.1 BLAST Integration

**Local BLAST+ Installation** (in Docker container)

```dockerfile
# In docker/sandbox/Dockerfile.biolab or new Dockerfile.rnai
RUN apt-get update && apt-get install -y \
    ncbi-blast+ \
    && rm -rf /var/lib/apt/lists/*
```

**Python BLAST Interface** (using `Bio.Blast` from `biopython`)

```python
from Bio.Blast.Applications import NcbiblastnCommandline
from Bio.Blast import NCBIXML
import subprocess

def _run_local_blast(
    query_sequence: str,
    database_path: str,
    output_file: str
) -> List[Dict[str, Any]]:
    """
    Run local BLASTN search.
    
    Args:
        query_sequence: RNA sequence to search
        database_path: Path to BLAST database
        output_file: Path to save XML results
    
    Returns:
        List of BLAST hits
    """
```

#### 2.2 Protected Species Genome Databases

**Pre-built BLAST Databases** (stored in Docker image or mounted volume)

```
/data/blastdb/
├── apis_mellifera/
│   ├── apis_mellifera_genome.fasta
│   ├── apis_mellifera_genome.nhr
│   ├── apis_mellifera_genome.nin
│   └── apis_mellifera_genome.nsq
├── danaus_plexippus/
│   └── ...
└── [other protected species]
```

**Database Download & Indexing**:

```python
def _download_protected_genomes(
    species_list: List[str],
    output_dir: str
) -> Dict[str, str]:
    """
    Download and index genomes for protected species.
    
    Sources:
    - NCBI GenBank/RefSeq
    - Ensembl Metazoa
    - Species-specific databases
    
    Returns:
        Dict mapping species name to BLAST database path
    """
```

**Caching Strategy**:
- Download genomes once, cache in Docker image or persistent volume
- Re-index only if sequence updates available
- Use version tracking to detect updates

#### 2.3 BLAST Search Parameters

**Optimized for RNAi Off-Target Detection**:

```python
BLAST_PARAMS = {
    "task": "blastn-short",  # Optimized for short sequences
    "word_size": 7,           # Smaller word size for short queries
    "evalue": 10.0,           # Relaxed E-value (we want all potential matches)
    "gapopen": 5,             # Gap opening penalty
    "gapextend": 2,            # Gap extension penalty
    "reward": 2,               # Match reward
    "penalty": -3,             # Mismatch penalty
    "dust": "no",              # Disable low-complexity filtering
    "soft_masking": "false"    # No soft masking
}
```

**Match Criteria**:
- **Reject if**: Any match ≥21 consecutive nucleotides with ≥90% identity
- **Warn if**: Match ≥18 nucleotides (potential off-target risk)
- **Approve if**: All matches <18 nucleotides

#### 2.4 Sequence Validation

**Input Validation**:

```python
def _validate_rnai_sequence(sequence: str) -> Tuple[bool, str]:
    """
    Validate RNAi candidate sequence.
    
    Requirements:
    - Length: 18-30 nucleotides (typical siRNA length)
    - Characters: Only A, G, C, T (DNA) or A, G, C, U (RNA)
    - No ambiguous bases (N, R, Y, etc.)
    - No gaps or whitespace
    
    Returns:
        (is_valid, error_message)
    """
```

**RNA/DNA Conversion**:
- Accept both RNA (U) and DNA (T) input
- Normalize to DNA for BLAST (BLAST databases are DNA)

### Protected Species Default List

```python
DEFAULT_PROTECTED_SPECIES = [
    {
        "scientific_name": "Apis mellifera",
        "common_name": "Honeybee",
        "priority": "critical",  # Essential pollinator
        "data_source": "ncbi"   # NCBI RefSeq
    },
    {
        "scientific_name": "Danaus plexippus",
        "common_name": "Monarch Butterfly",
        "priority": "high",      # Protected species
        "data_source": "ncbi"
    },
    {
        "scientific_name": "Bombus impatiens",
        "common_name": "Common Eastern Bumblebee",
        "priority": "high",
        "data_source": "ncbi"
    },
    # Extensible list - users can add more
]
```

---

## Integration Points

### 3.1 Tool Registry Integration

**Add to `kosmos/tools/__init__.py`**:

```python
RNAI_TOOLS: Dict[str, ToolDefinition] = {
    "find_essential_genes": ToolDefinition(
        name="find_essential_genes",
        description=(
            "Find essential genes in a target pest organism by downloading "
            "transcriptome data and filtering for genes with lethal knockout "
            "phenotypes. Returns gene sequences (AGCT format) suitable for "
            "RNAi targeting."
        ),
        category=ToolCategory.BIOLAB,
        parameters=[
            ToolParameter(
                name="organism_name",
                type="string",
                description="Scientific or common name of pest (e.g., 'Helicoverpa zea')"
            ),
            ToolParameter(
                name="data_source",
                type="string",
                description="Database to query: 'insectbase', 'ncbi', 'ensembl', or 'auto'",
                required=False,
                default="auto"
            ),
            # ... more parameters
        ],
        returns="Dict with essential_genes list, organism_info, and metadata",
        example_usage='find_essential_genes(organism_name="Helicoverpa zea")',
        requires_docker_image="kosmos-biolab:latest",  # Or new kosmos-rnai:latest
        estimated_runtime_seconds=300,  # Genome download + BLAST analysis
        memory_requirements_gb=4.0,
    ),
    
    "generate_sirna_candidates": ToolDefinition(
        name="generate_sirna_candidates",
        description=(
            "CRITICAL: Generate valid 21-mer siRNA candidate sequences from a long "
            "gene sequence using Reynolds Rules. This tool converts gene sequences "
            "(typically 1000-5000bp) into targetable RNAi agents. Required immediately "
            "after find_essential_genes - agents cannot guess valid siRNA sequences."
        ),
        category=ToolCategory.BIOLAB,
        parameters=[
            ToolParameter(
                name="gene_sequence",
                type="string",
                description="Full gene sequence (DNA, AGCT format, typically 1000-5000bp)"
            ),
            ToolParameter(
                name="window_size",
                type="integer",
                description="Length of siRNA candidates (default: 21)",
                required=False,
                default=21
            ),
            ToolParameter(
                name="max_candidates",
                type="integer",
                description="Maximum number of candidates to return (default: 10)",
                required=False,
                default=10
            ),
        ],
        returns="Dict with candidates list (21-mer sequences with scores), total generated, filtered count",
        example_usage='generate_sirna_candidates(gene_sequence="AGCT..." * 1000)',
        requires_docker_image="kosmos-biolab:latest",
        estimated_runtime_seconds=5,  # Fast - just sequence analysis
        memory_requirements_gb=0.5,
    ),
    
    "check_off_target_risk": ToolDefinition(
        name="check_off_target_risk",
        description=(
            "Validate RNAi candidate sequence against protected species genomes "
            "using BLAST search. Rejects sequences that match honeybees, monarch "
            "butterflies, or other protected species. Returns APPROVE if 0 matches, "
            "REJECT if matches found."
        ),
        category=ToolCategory.BIOLAB,
        parameters=[
            ToolParameter(
                name="candidate_sequence",
                type="string",
                description="RNA sequence to test (21-30 nucleotides, AGCT format)"
            ),
            ToolParameter(
                name="protected_species_list",
                type="array",
                description="List of protected species to check (default: honeybee, monarch)",
                required=False,
                default=None
            ),
            # ... more parameters
        ],
        returns="Dict with approved/rejected status, matches found, and BLAST results",
        example_usage='check_off_target_risk(candidate_sequence="AGCTAGCTAGCTAGCTAGCTA")',
        requires_docker_image="kosmos-biolab:latest",
        estimated_runtime_seconds=60,  # BLAST search
        memory_requirements_gb=2.0,
    ),
}

# In ToolRegistry.__init__():
for name, tool_def in RNAI_TOOLS.items():
    self.register(tool_def)
```

### 3.2 Docker Container Configuration

**CRITICAL FIX: Use Mounted Volumes, Not COPY**

**Do NOT bake genomes into Docker image** - this creates 5GB+ images that are painful to pull/push.

**Revised Strategy: Mounted Volumes**

**Option A: Extend Existing BioLab Image**

Modify `docker/sandbox/Dockerfile.biolab`:

```dockerfile
# Add BLAST and genomics tools (software only, no data)
RUN apt-get update && apt-get install -y \
    ncbi-blast+ \
    samtools \
    bedtools \
    && rm -rf /var/lib/apt/lists/*

# Install Python genomics packages
RUN pip install --no-cache-dir \
    biopython \
    pyensembl \
    mygene

# Create data directory (will be mounted at runtime)
RUN mkdir -p /data/blastdb /data/genomes

# Set working directory
WORKDIR /workspace
```

**Option B: New Specialized Image** (`kosmos-rnai:latest`)

Create `docker/sandbox/Dockerfile.rnai`:

```dockerfile
FROM kosmos-biolab:latest

# Add RNAi-specific tools (software only)
RUN apt-get update && apt-get install -y \
    ncbi-blast+ \
    samtools \
    bedtools \
    && rm -rf /var/lib/apt/lists/*

# Create data directories (mounted at runtime)
RUN mkdir -p /data/blastdb /data/genomes /data/deg

# Set working directory
WORKDIR /workspace

# NO COPY or download of genomes here - use volumes instead
```

**Runtime Volume Mounting**:

```python
# In kosmos/execution/docker_manager.py or sandbox.py

def _prepare_rnai_volumes(self) -> Dict[str, str]:
    """
    Prepare volume mounts for RNAi tools.
    
    Strategy:
    - Host directory: ./kosmos_data/ (project root)
    - Container mount: /data/
    - On first run, initialization script downloads genomes to host
    - Subsequent runs use cached genomes
    """
    host_data_dir = Path.cwd() / "kosmos_data"
    host_data_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (host_data_dir / "blastdb").mkdir(exist_ok=True)
    (host_data_dir / "genomes").mkdir(exist_ok=True)
    (host_data_dir / "deg").mkdir(exist_ok=True)
    
    return {
        str(host_data_dir / "blastdb"): {"bind": "/data/blastdb", "mode": "rw"},
        str(host_data_dir / "genomes"): {"bind": "/data/genomes", "mode": "rw"},
        str(host_data_dir / "deg"): {"bind": "/data/deg", "mode": "rw"},
    }
```

**Initialization Script** (runs on first container start):

```python
# kosmos/tools/rnai_designer.py or separate init script

def _ensure_genome_databases(data_dir: str) -> Dict[str, str]:
    """
    Download and index genomes if not present.
    
    Runs on first use, caches to host-mounted volume.
    Subsequent runs use cached data.
    """
    data_path = Path(data_dir)
    
    protected_species = [
        {
            "name": "Apis mellifera",
            "ncbi_id": "GCF_003254395.2",
            "fasta_file": data_path / "genomes" / "apis_mellifera.fasta",
            "blastdb": data_path / "blastdb" / "apis_mellifera",
        },
        {
            "name": "Danaus plexippus",
            "ncbi_id": "GCF_000235995.1",
            "fasta_file": data_path / "genomes" / "danaus_plexippus.fasta",
            "blastdb": data_path / "blastdb" / "danaus_plexippus",
        },
    ]
    
    for species in protected_species:
        if not species["blastdb"].exists():
            # Download genome
            _download_genome_from_ncbi(
                species["ncbi_id"],
                species["fasta_file"]
            )
            
            # Index for BLAST
            subprocess.run([
                "makeblastdb",
                "-in", str(species["fasta_file"]),
                "-dbtype", "nucl",
                "-out", str(species["blastdb"]),
                "-title", species["name"]
            ])
    
    return {s["name"]: str(s["blastdb"]) for s in protected_species}
```

**Benefits**:
- Docker image stays small (~500MB instead of 5GB+)
- Genomes persist on host, not re-downloaded
- Faster container startup
- Easier to update genomes without rebuilding image

**Update `kosmos/execution/docker_manager.py`**:

```python
class SandboxImageType(Enum):
    STANDARD = "standard"
    BIOLAB = "biolab"
    R_LANG = "r_lang"
    RNAI_DESIGNER = "rnai_designer"  # NEW

SANDBOX_IMAGES = {
    # ... existing ...
    SandboxImageType.RNAI_DESIGNER: "kosmos-rnai:latest",
}

@classmethod
def for_rnai_designer(cls, memory_limit: str = "4g", timeout_seconds: int = 600) -> "ContainerConfig":
    """Create configuration for RNAi designer tools."""
    return cls(
        image=SANDBOX_IMAGES[SandboxImageType.RNAI_DESIGNER],
        image_type=SandboxImageType.RNAI_DESIGNER,
        memory_limit=memory_limit,
        timeout_seconds=timeout_seconds,
        # Network access needed for downloading genomes
        network_mode="bridge",  # Allow network for data downloads
    )
```

### 3.3 Agent Integration

**Tool Availability**:

Tools are automatically available to agents via the `ToolRegistry`. Agents discover tools through:

1. **Tool schemas** - JSON schemas for LLM tool calling
2. **Prompt descriptions** - Human-readable tool descriptions injected into agent prompts
3. **Skill loader** - Domain-specific skills (if RNAi tools are added as skills)

**Example Agent Usage** (autonomous):

```python
# Agent workflow (pseudo-code)
# 1. Agent receives task: "Design RNAi agent for corn earworm"
# 2. Agent calls find_essential_genes("Helicoverpa zea")
# 3. Agent receives list of essential genes
# 4. Agent selects target gene (e.g., chitin synthase)
# 5. Agent generates candidate RNA sequence (21-mer)
# 6. Agent calls check_off_target_risk(candidate_sequence)
# 7. If approved, agent reports success; if rejected, agent tries different sequence
```

### 3.4 World Model Integration

**Store Results in World Model**:

```python
# In kosmos/world_model/artifacts.py or similar
@dataclass
class RNAiDesignArtifact:
    """Artifact for RNAi design experiments."""
    target_organism: str
    essential_genes: List[Dict[str, Any]]
    candidate_sequences: List[Dict[str, Any]]
    approved_sequences: List[str]
    rejected_sequences: List[Dict[str, Any]]
    timestamp: datetime
```

---

## Dependencies & Infrastructure

### 4.1 Python Dependencies

**Add to `pyproject.toml`**:

```toml
dependencies = [
    # ... existing dependencies ...
    
    # Genomics & BLAST (if not already present)
    "biopython>=1.81",           # Already present
    "pyensembl>=2.3.0",         # Already present
    "mygene>=3.2.0",            # Already present
    
    # New dependencies for RNAi tools
    "requests>=2.31.0",         # For InsectBase API (if not using httpx)
    # OR use existing httpx>=0.27.0
]
```

### 4.2 System Dependencies (Docker)

**BLAST+** (NCBI BLAST command-line tools):
- `ncbi-blast+` package (Ubuntu/Debian)
- Required for local BLAST searches

**Genome Data**:
- Protected species genomes (honeybee, monarch)
- Download from NCBI, Ensembl, or species-specific databases
- Index with `makeblastdb` for BLAST searches

**Storage Requirements**:
- Honeybee genome: ~250 MB
- Monarch genome: ~300 MB
- BLAST indices: ~2x genome size
- **Total**: ~1-2 GB per protected species

### 4.3 Network Requirements

**For Genome Downloads**:
- Access to NCBI FTP servers
- Access to Ensembl Metazoa
- Access to InsectBase API (if used)

**For BLAST Searches**:
- Can use local BLAST (preferred) or remote NCBI BLAST
- Local BLAST requires pre-downloaded databases

### 4.4 Caching Strategy

**Genome Database Caching**:
- Download genomes once during Docker image build
- Cache in Docker image or persistent volume
- Version tracking to detect updates

**API Response Caching**:
- Use existing Redis cache (if configured)
- Cache organism searches, transcriptome downloads
- TTL: 7 days (genome data changes infrequently)

---

## Testing Strategy

### 5.1 Unit Tests

**File**: `tests/unit/tools/test_rnai_designer.py`

```python
import pytest
from kosmos.tools.rnai_designer import (
    find_essential_genes,
    check_off_target_risk,
    _validate_rnai_sequence,
    _filter_essential_genes,
)

class TestFindEssentialGenes:
    def test_valid_organism(self):
        """Test finding essential genes for known organism."""
        result = find_essential_genes("Drosophila melanogaster")
        assert result["success"] is True
        assert len(result["essential_genes"]) > 0
    
    def test_unknown_organism(self):
        """Test error handling for unknown organism."""
        result = find_essential_genes("Nonexistent bugus")
        assert result["success"] is False
        assert "not found" in result["error_message"].lower()
    
    def test_essentiality_filtering(self):
        """Test filtering by essentiality criteria."""
        # Mock gene data
        genes = [
            {"gene_id": "gene1", "knockout_lethality": 0.95, "go_terms": ["chitin synthesis"]},
            {"gene_id": "gene2", "knockout_lethality": 0.3, "go_terms": ["metabolism"]},
        ]
        filtered = _filter_essential_genes(genes, {"min_knockout_lethality": 0.8})
        assert len(filtered) == 1
        assert filtered[0]["gene_id"] == "gene1"

class TestCheckOffTargetRisk:
    def test_approved_sequence(self):
        """Test sequence with no matches (approved)."""
        # Use sequence known to not match honeybee/monarch
        result = check_off_target_risk("AGCTAGCTAGCTAGCTAGCTA")
        assert result["success"] is True
        assert result["approved"] is True
        assert result["rejected"] is False
        assert len(result["matches"]) == 0
    
    def test_rejected_sequence(self):
        """Test sequence that matches protected species (rejected)."""
        # Use sequence known to match honeybee genome
        result = check_off_target_risk("ATCGATCGATCGATCGATCGAT")
        assert result["success"] is True
        assert result["approved"] is False
        assert result["rejected"] is True
        assert len(result["matches"]) > 0
    
    def test_invalid_sequence(self):
        """Test error handling for invalid sequence."""
        result = check_off_target_risk("INVALID123")
        assert result["success"] is False
        assert "invalid" in result["error_message"].lower()

class TestSequenceValidation:
    def test_valid_dna_sequence(self):
        """Test valid DNA sequence."""
        valid, error = _validate_rnai_sequence("AGCTAGCTAGCTAGCTAGCTA")
        assert valid is True
        assert error is None
    
    def test_valid_rna_sequence(self):
        """Test valid RNA sequence (U instead of T)."""
        valid, error = _validate_rnai_sequence("AGCUAGCUAGCUAGCUAGCU")
        assert valid is True
    
    def test_too_short(self):
        """Test sequence too short."""
        valid, error = _validate_rnai_sequence("AGCT")  # Only 4 bases
        assert valid is False
        assert "short" in error.lower()
    
    def test_invalid_characters(self):
        """Test sequence with invalid characters."""
        valid, error = _validate_rnai_sequence("AGCTNXXX")
        assert valid is False
        assert "invalid" in error.lower()
```

### 5.2 Integration Tests

**File**: `tests/integration/tools/test_rnai_integration.py`

```python
import pytest
from kosmos.tools.rnai_designer import find_essential_genes, check_off_target_risk

@pytest.mark.integration
@pytest.mark.requires_network
class TestRNAiIntegration:
    """Integration tests requiring network access and BLAST databases."""
    
    def test_full_workflow(self):
        """Test complete RNAi design workflow."""
        # Step 1: Find essential genes
        genes_result = find_essential_genes("Drosophila melanogaster")
        assert genes_result["success"]
        
        # Step 2: Select a gene
        target_gene = genes_result["essential_genes"][0]
        sequence = target_gene["sequence"]
        
        # Step 3: Extract 21-mer candidate
        candidate = sequence[:21]
        
        # Step 4: Check off-target risk
        safety_result = check_off_target_risk(candidate)
        assert safety_result["success"]
        
        # Note: May be approved or rejected depending on sequence
```

### 5.3 Mock Tests

**File**: `tests/unit/tools/test_rnai_mocks.py`

```python
from unittest.mock import patch, MagicMock

class TestRNAiMocks:
    """Tests using mocked external APIs and BLAST."""
    
    @patch('kosmos.domains.biology.apis.InsectBaseClient.get_essential_genes')
    def test_find_essential_genes_mock(self, mock_get_genes):
        """Test with mocked InsectBase API."""
        mock_get_genes.return_value = [
            {
                "gene_id": "test_gene_1",
                "gene_name": "chitin_synthase",
                "sequence": "AGCT" * 100,
                "knockout_lethality": 0.95,
            }
        ]
        
        result = find_essential_genes("Test organism")
        assert result["success"]
        assert len(result["essential_genes"]) == 1
    
    @patch('subprocess.run')  # Mock BLAST command
    def test_check_off_target_risk_mock(self, mock_subprocess):
        """Test with mocked BLAST."""
        # Mock BLAST output (no matches)
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout=b"# BLASTN 2.13.0+\n# No hits found\n"
        )
        
        result = check_off_target_risk("AGCTAGCTAGCTAGCTAGCTA")
        assert result["success"]
        assert result["approved"] is True
```

### 5.4 E2E Tests

**File**: `tests/e2e/test_rnai_e2e.py`

```python
@pytest.mark.e2e
@pytest.mark.requires_docker
@pytest.mark.requires_network
class TestRNAiE2E:
    """End-to-end tests in Docker sandbox."""
    
    def test_rnai_design_in_sandbox(self):
        """Test RNAi design tools in actual Docker sandbox."""
        # This test runs in kosmos-biolab or kosmos-rnai container
        # Requires actual BLAST databases and network access
        pass
```

---

## Implementation Timeline

### Phase 1: Foundation (Week 1-3)

**Week 1: Data Source Integration & DEG Database**
- [ ] Implement `InsectBaseClient` in `kosmos/domains/biology/apis.py`
- [ ] Extend `NCBIGenomicsClient` for transcriptome downloads
- [ ] Add Ensembl Metazoa support
- [ ] Download and set up Drosophila Essential Genes (DEG) database
- [ ] Create DEG BLAST database indexing
- [ ] Unit tests for API clients

**Week 2: Essential Gene Identification via Homology**
- [ ] Implement `find_essential_genes()` in `kosmos/tools/rnai_designer.py`
- [ ] **CRITICAL FIX**: Implement homology mapping (BLAST pest genes vs DEG)
- [ ] Remove incorrect "lethality metadata" assumptions
- [ ] Sequence extraction and formatting
- [ ] Integration tests with real DEG database

**Week 3: SiRNA Generator (CRITICAL - Required Now)**
- [ ] Implement `generate_sirna_candidates()` in `kosmos/tools/rnai_generator.py`
- [ ] Implement Reynolds Rules filtering
- [ ] GC content calculation
- [ ] Homopolymer detection
- [ ] Thermodynamic stability calculation
- [ ] Scoring and ranking algorithm
- [ ] Unit tests

### Phase 2: Safety Filter (Week 4-5)

**Week 4: BLAST Integration & Volume Mounting**
- [ ] Add BLAST+ to Docker image (software only, no genomes)
- [ ] **CRITICAL FIX**: Implement volume mounting for genome databases
- [ ] Create initialization script for genome downloads (host-side)
- [ ] Implement `check_off_target_risk()` function
- [ ] BLAST search implementation
- [ ] Unit tests

**Week 5: Safety Validation**
- [ ] Sequence validation logic
- [ ] Match criteria and scoring
- [ ] Protected species database management (mounted volumes)
- [ ] Integration tests with real genomes

### Phase 3: Integration & Testing (Week 6-7)

**Week 6: Tool Registry Integration**
- [ ] Add tool definitions to `kosmos/tools/__init__.py` (all 3 tools)
- [ ] Register tools in `ToolRegistry`
- [ ] Update Docker container configurations (volume mounting)
- [ ] Agent integration testing (full workflow)

**Week 7: Testing & Documentation**
- [ ] Comprehensive unit test suite
- [ ] Integration tests (with DEG database, mounted genomes)
- [ ] E2E tests (if possible)
- [ ] Documentation updates
- [ ] Example usage scripts

### Phase 4: Optimization & Production (Week 8-9)

**Week 8: Performance Optimization**
- [ ] Caching strategy implementation (genome databases, DEG)
- [ ] Database indexing optimization
- [ ] Parallel BLAST searches
- [ ] Memory usage optimization
- [ ] Volume mounting performance tuning

**Week 9: Production Readiness**
- [ ] Error handling improvements
- [ ] Logging and monitoring
- [ ] Documentation finalization
- [ ] Code review and cleanup
- [ ] Verify Docker image size (<1GB, not 5GB+)

---

## Risk Assessment

### Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|--------------|------------|
| **InsectBase API unavailable** | High | Medium | Fallback to NCBI/Ensembl |
| **BLAST performance slow** | Medium | High | Use local BLAST, optimize parameters, cache results |
| **Genome download failures** | Medium | Medium | Retry logic, multiple data sources |
| **Docker image size too large** | **High** | **High** | **FIXED**: Use mounted volumes, not COPY in Dockerfile |
| **Missing SiRNA Generator Tool** | **Critical** | **High** | **FIXED**: Added as Phase 1b, required immediately |
| **False positives in BLAST** | High | Low | Strict match criteria (21-mer, 90% identity) |
| **False negatives in BLAST** | High | Low | Multiple BLAST parameters, check reverse complement |

### Data Quality Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|--------------|------------|
| **Lethality Metadata Doesn't Exist** | **High** | **High** | **FIXED**: Use homology mapping against DEG database |
| **Incomplete essential gene annotations** | Medium | High | Use multiple data sources, manual curation option |
| **Outdated genome assemblies** | Medium | Medium | Version tracking, update notifications |
| **Missing protected species genomes** | High | Low | Pre-download critical species, clear error messages |

### Operational Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|--------------|------------|
| **Network access required** | Medium | High | Local BLAST preferred, graceful degradation |
| **Storage requirements** | Low | Medium | Efficient compression, optional species |
| **Maintenance burden** | Medium | Medium | Automated updates, clear documentation |

---

## Success Criteria

### Functional Requirements

- [x] `find_essential_genes()` successfully identifies essential genes via homology mapping (DEG database)
- [x] `generate_sirna_candidates()` generates valid 21-mer candidates from long gene sequences
- [x] `check_off_target_risk()` correctly identifies matches in honeybee and monarch genomes
- [x] All three tools integrate seamlessly with existing Kosmos tool ecosystem
- [x] Agents can autonomously use tools to design RNAi sequences (complete workflow)

### Performance Requirements

- [x] `find_essential_genes()` completes in <5 minutes for typical organisms
- [x] `generate_sirna_candidates()` completes in <10 seconds per gene
- [x] `check_off_target_risk()` completes in <2 minutes per sequence
- [x] BLAST searches use <2GB memory
- [x] **Docker image size <1GB** (genomes in mounted volumes, not in image)

### Quality Requirements

- [x] >90% test coverage for core functions
- [x] All unit tests passing
- [x] Integration tests passing (with network access)
- [x] Documentation complete and accurate

---

## Future Enhancements

### Phase 3: Enhanced SiRNA Design (Future)

**Enhancements to `generate_sirna_candidates()`**:
- Secondary structure prediction (avoid hairpins)
- Off-target prediction within target organism
- Multi-target design (target multiple genes simultaneously)
- Chemical modification recommendations (2'-OMe, phosphorothioate)

### Phase 4: Efficacy Prediction (Future)

**Tool**: `predict_rnai_efficacy(candidate_sequence, target_gene)`
- Predict silencing efficiency
- Consider target site accessibility
- Secondary structure prediction

### Phase 5: Multi-Species Validation (Future)

**Tool**: `validate_across_species(candidate_sequence, species_list)`
- Check efficacy across related pest species
- Identify conserved target sites
- Optimize for broad-spectrum control

---

## Appendix

### A. Example Tool Usage (Complete Workflow)

```python
# Example 1: Find essential genes (via homology mapping)
result = find_essential_genes(
    organism_name="Helicoverpa zea",
    data_source="auto"
)

if result["success"]:
    print(f"Found {len(result['essential_genes'])} essential genes")
    for gene in result["essential_genes"]:
        print(f"  {gene['gene_name']}: {gene['sequence'][:50]}...")
        print(f"    Homologous to: {gene['homologous_drosophila_gene']}")
        print(f"    Phenotype: {gene['knockout_phenotype']}")

# Example 2: Generate SiRNA candidates (CRITICAL STEP)
target_gene = result["essential_genes"][0]  # Select chitin synthase
gene_sequence = target_gene["sequence"]  # 3000bp sequence

candidates_result = generate_sirna_candidates(
    gene_sequence=gene_sequence,
    window_size=21,
    max_candidates=10
)

if candidates_result["success"]:
    print(f"Generated {len(candidates_result['candidates'])} candidates")
    for i, candidate in enumerate(candidates_result["candidates"], 1):
        print(f"  Candidate {i}: {candidate['sequence']}")
        print(f"    GC: {candidate['gc_content']:.2%}, Score: {candidate['reynolds_score']:.2f}")

# Example 3: Check off-target risk for each candidate
for candidate in candidates_result["candidates"]:
    candidate_seq = candidate["sequence"]
    safety = check_off_target_risk(
        candidate_sequence=candidate_seq,
        protected_species_list=["Apis mellifera", "Danaus plexippus"]
    )
    
    if safety["approved"]:
        print(f"✓ APPROVED: {candidate_seq}")
    else:
        print(f"✗ REJECTED: {candidate_seq}")
        print(f"  Matches: {len(safety['matches'])}")
        for match in safety["matches"]:
            print(f"    {match['species']}: {match['gene_id']}")
```

### B. Data Source URLs

- **InsectBase**: http://www.insectbase.org/
- **NCBI GenBank**: https://www.ncbi.nlm.nih.gov/genbank/
- **Ensembl Metazoa**: https://metazoa.ensembl.org/
- **NCBI BLAST**: https://blast.ncbi.nlm.nih.gov/

### C. Protected Species Genome Accessions

- **Honeybee (Apis mellifera)**: 
  - NCBI: GCF_003254395.2
  - Ensembl: Amel_HAv3.1
- **Monarch (Danaus plexippus)**:
  - NCBI: GCF_000235995.1
  - Ensembl: Dplex_v3

---

**Document Version**: 1.0  
**Last Updated**: 2025-01-XX  
**Author**: Implementation Plan  
**Status**: Draft for Review
