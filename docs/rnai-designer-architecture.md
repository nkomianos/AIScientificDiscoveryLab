# Kosmos RNAi Designer - Architecture Diagram

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Kosmos Agent System                              │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      Tool Registry                               │   │
│  │  - find_essential_genes                                          │   │
│  │  - generate_sirna_candidates                                     │   │
│  │  - check_off_target_risk                                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    RNAi Designer Tools Layer                             │
│                                                                           │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐     │
│  │ find_essential_  │  │ generate_sirna_  │  │ check_off_target │     │
│  │     genes        │  │   candidates     │  │      _risk       │     │
│  │                  │  │                  │  │                  │     │
│  │ • Validate input │  │ • Validate seq   │  │ • Validate seq   │     │
│  │ • Call genomics  │  │ • Slide window   │  │ • Load genomes   │     │
│  │   modules        │  │ • Apply Reynolds │  │ • Run BLAST      │     │
│  │ • Return results │  │   Rules          │  │ • Parse matches  │     │
│  └──────────────────┘  │ • Score & rank   │  └──────────────────┘     │
│                        └──────────────────┘                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Genomics Domain Layer                                 │
│                                                                           │
│  ┌──────────────────────────────┐  ┌──────────────────────────────┐   │
│  │   essential_genes.py         │  │        rnai.py               │   │
│  │                              │  │                              │   │
│  │ • download_transcriptome     │  │ • ensure_protected_genomes   │   │
│  │ • ensure_deg_database        │  │ • run_blast_search           │   │
│  │ • identify_via_homology      │  │ • parse_blast_results        │   │
│  └──────────────────────────────┘  └──────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      API Clients Layer                                   │
│                                                                           │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐     │
│  │ InsectBaseClient │  │ NCBIGenomics     │  │ EnsemblClient    │     │
│  │                  │  │    Client        │  │                  │     │
│  │ • search_organism│  │ • search_genome  │  │ • get_gene       │     │
│  │ • get_transcr... │  │ • download_...   │  │ • get_vep_...    │     │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    External Data Sources                                 │
│                                                                           │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐     │
│  │   InsectBase     │  │   NCBI GenBank   │  │ Ensembl Metazoa  │     │
│  │ www.insectbase   │  │   RefSeq         │  │                  │     │
│  │     .org         │  │                  │  │                  │     │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘     │
│                                                                           │
│  ┌──────────────────┐  ┌──────────────────────────────────────────┐   │
│  │   FlyBase DEG    │  │   Protected Species Genomes             │   │
│  │   Database       │  │   • Apis mellifera (Honeybee)           │   │
│  │                  │  │   • Danaus plexippus (Monarch)          │   │
│  └──────────────────┘  │   • Bombus impatiens (Bumblebee)        │   │
│                        └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Docker Container Layer                                │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              kosmos-biolab:latest                                │   │
│  │                                                                   │   │
│  │  • Python 3.11+                                                  │   │
│  │  • Biopython (sequence parsing)                                  │   │
│  │  • BLAST+ (ncbi-blast+)                                          │   │
│  │  • SAMtools (genome processing)                                  │   │
│  │  • BEDtools (genomic intervals)                                  │   │
│  │                                                                   │   │
│  │  Volume Mounts:                                                  │   │
│  │  • /data/blastdb  ← Host: ./kosmos_data/blastdb                 │   │
│  │  • /data/genomes  ← Host: ./kosmos_data/genomes                 │   │
│  │  • /data/deg      ← Host: ./kosmos_data/deg                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Data Flow Diagram

```
┌──────────────────┐
│  Agent Request   │
│ "Design RNAi for │
│  corn earworm"   │
└────────┬─────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 1: find_essential_genes("Helicoverpa zea")             │
├─────────────────────────────────────────────────────────────┤
│ 1. Download transcriptome from NCBI/InsectBase             │
│    → 5000 genes, 3000bp each                                │
│                                                              │
│ 2. Download DEG database (Drosophila essential genes)       │
│    → 2000 essential genes with lethality annotations        │
│                                                              │
│ 3. BLAST pest genes against DEG                             │
│    → Find homologs (E-value < 1e-50, identity > 70%)        │
│                                                              │
│ 4. Tag matching genes as essential                          │
│    → Transfer annotations from Drosophila                   │
│                                                              │
│ Output: 500 essential genes with sequences                  │
└────────┬────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ Agent selects target: "Chitin Synthase" (3000bp)            │
└────────┬────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: generate_sirna_candidates(gene_sequence)            │
├─────────────────────────────────────────────────────────────┤
│ 1. Slide 21bp window across 3000bp sequence                 │
│    → 2980 possible positions                                │
│                                                              │
│ 2. For each window:                                         │
│    a. Calculate GC content (30-52%)                         │
│    b. Check for homopolymer runs (4+ same base)             │
│    c. Calculate thermodynamic stability (5' end)            │
│    d. Calculate Reynolds score                              │
│                                                              │
│ 3. Filter by Reynolds Rules                                 │
│    → 150 candidates pass filters                            │
│                                                              │
│ 4. Sort by Reynolds score (descending)                      │
│    → Return top 10 candidates                               │
│                                                              │
│ Output: 10 siRNA candidates (21-mers) with scores           │
└────────┬────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ Agent tests each candidate for safety                       │
└────────┬────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: check_off_target_risk(candidate_sequence)           │
├─────────────────────────────────────────────────────────────┤
│ For each protected species:                                 │
│                                                              │
│ 1. Load protected genome BLAST database                     │
│    • Apis mellifera (Honeybee) - 250MB                      │
│    • Danaus plexippus (Monarch) - 300MB                     │
│                                                              │
│ 2. Run BLASTN search                                        │
│    • Task: blastn-short (optimized for 21-mers)             │
│    • Word size: 7                                           │
│    • E-value: 10.0 (relaxed - find all matches)             │
│                                                              │
│ 3. Parse results                                            │
│    • Filter: Match length ≥ 21bp, identity ≥ 90%            │
│    • Count matches                                          │
│                                                              │
│ 4. Determine approval                                       │
│    • APPROVE if 0 matches                                   │
│    • REJECT if any matches found                            │
│                                                              │
│ Output: APPROVE/REJECT with match details                   │
└────────┬────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ Agent receives results:                                     │
│ • 10 candidates tested                                      │
│ • 7 approved (safe)                                         │
│ • 3 rejected (off-target matches)                           │
│                                                              │
│ Agent selects best approved candidate:                      │
│ "AGCTAGCTAGCTAGCTAGCTA"                                     │
│ • Reynolds score: 0.92                                      │
│ • GC content: 48%                                           │
│ • No off-target matches                                     │
└─────────────────────────────────────────────────────────────┘
```

## Reynolds Rules Pipeline

```
Gene Sequence (3000bp)
         │
         ▼
┌─────────────────────────────────────┐
│   Slide 21bp Window                 │
│   Position 0: AGCTAGCTAGCTAGCTAGCTA │
│   Position 1: GCTAGCTAGCTAGCTAGCTAG │
│   Position 2: CTAGCTAGCTAGCTAGCTAGC │
│   ...                                │
│   Position 2979: ...                 │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   Filter 1: GC Content              │
│   • Calculate: (G+C) / 21            │
│   • Require: 30-52%                  │
│   • Optimal: 40-50%                  │
│   ✗ Reject: <30% or >52%             │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   Filter 2: Homopolymer Runs        │
│   • Detect: 4+ consecutive bases     │
│   • Examples: AAAA, GGGG, TTTT       │
│   ✗ Reject: Any homopolymer runs     │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   Calculate: 5' Stability           │
│   • First 4 bases: AGCT              │
│   • Use nearest-neighbor model       │
│   • GC pairs: -1.8 kcal/mol          │
│   • AT pairs: -0.9 kcal/mol          │
│   • Lower = better (easier RISC)     │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   Calculate: Reynolds Score         │
│   • GC score (60% weight)            │
│   • Stability score (40% weight)     │
│   • Range: 0.0-1.0                   │
│   • Higher = better                  │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   Sort & Rank                        │
│   1. AGCTAGCTAGCTAGCTAGCTA (0.92)   │
│   2. GCTAGCTAGCTAGCTAGCTAG (0.89)   │
│   3. CTAGCTAGCTAGCTAGCTAGC (0.87)   │
│   ...                                │
│   10. ...                  (0.75)   │
└─────────────────────────────────────┘
```

## File Organization

```
kosmos/
├── tools/
│   ├── __init__.py                 [MODIFIED] Tool registry
│   ├── rnai_designer.py           [NEW] Main RNAi tools
│   └── rnai_generator.py          [NEW] SiRNA generator
│
├── domains/biology/
│   ├── apis.py                    [MODIFIED] API clients
│   └── genomics/
│       ├── __init__.py            [NEW] Package init
│       ├── essential_genes.py     [NEW] DEG & homology
│       └── rnai.py                [NEW] BLAST & safety
│
├── tests/
│   ├── unit/tools/
│   │   └── test_rnai_designer.py [NEW] Unit tests
│   └── integration/tools/
│       └── test_rnai_integration.py [NEW] Integration tests
│
├── docs/
│   ├── rnai-designer-implementation.md [NEW] Full docs
│   ├── rnai-designer-quick-start.md    [NEW] Quick start
│   └── rnai-designer-architecture.md   [NEW] This file
│
├── examples/
│   └── rnai_designer_example.py   [NEW] Working example
│
└── docker/sandbox/
    └── Dockerfile.biolab          [MODIFIED] Added BLAST+
```

## Component Interactions

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent Workflow                            │
└────────┬────────────────────────────────────────────────────┘
         │
         ├─────────────────────────────────────────────────────┐
         │                                                      │
         ▼                                                      ▼
┌──────────────────┐                              ┌──────────────────┐
│ find_essential_  │                              │ generate_sirna_  │
│     genes        │                              │   candidates     │
└────────┬─────────┘                              └────────┬─────────┘
         │                                                  │
         ├──────────────────┬──────────────────┐           │
         ▼                  ▼                  ▼           ▼
┌────────────────┐ ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
│ download_      │ │ ensure_deg_    │ │ identify_via_  │ │ Reynolds Rules │
│ transcriptome  │ │ database       │ │ homology       │ │ Pipeline       │
└────────┬───────┘ └────────┬───────┘ └────────┬───────┘ └────────┬───────┘
         │                  │                  │                  │
         ▼                  ▼                  ▼                  │
┌────────────────┐ ┌────────────────┐ ┌────────────────┐         │
│ InsectBase     │ │ FlyBase        │ │ BLAST          │         │
│ NCBI           │ │ DEG Database   │ │ Search         │         │
└────────────────┘ └────────────────┘ └────────────────┘         │
                                                                  │
         ┌────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────┐
│ check_off_target │
│      _risk       │
└────────┬─────────┘
         │
         ├──────────────────┬──────────────────┐
         ▼                  ▼                  ▼
┌────────────────┐ ┌────────────────┐ ┌────────────────┐
│ ensure_        │ │ run_blast_     │ │ parse_blast_   │
│ protected_     │ │ search         │ │ results        │
│ genomes        │ │                │ │                │
└────────┬───────┘ └────────┬───────┘ └────────┬───────┘
         │                  │                  │
         ▼                  ▼                  ▼
┌────────────────┐ ┌────────────────┐ ┌────────────────┐
│ Honeybee       │ │ BLAST+         │ │ Match          │
│ Monarch        │ │ blastn-short   │ │ Filtering      │
│ Bumblebee      │ │                │ │                │
└────────────────┘ └────────────────┘ └────────────────┘
```

---

**Last Updated**: February 8, 2026  
**Status**: ✅ Complete
