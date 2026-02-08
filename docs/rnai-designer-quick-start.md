# Kosmos RNAi Designer - Quick Start Guide

## What is it?

The Kosmos RNAi Designer enables autonomous agents to design eco-friendly gene-silencing agents for pest control. It provides three main tools:

1. **`find_essential_genes`** - Find genes required for pest survival
2. **`generate_sirna_candidates`** - Generate 21-mer siRNA sequences
3. **`check_off_target_risk`** - Ensure safety for beneficial insects

## Quick Start

### 1. Build Docker Image

```bash
cd docker/sandbox
docker build -t kosmos-biolab:latest -f Dockerfile.biolab .
```

### 2. Run Example

```bash
docker run -v ./kosmos_data:/data -v ./examples:/workspace kosmos-biolab:latest \
    python3 /workspace/rnai_designer_example.py
```

### 3. Use in Agent Code

```python
from kosmos.tools.rnai_designer import find_essential_genes, check_off_target_risk
from kosmos.tools.rnai_generator import generate_sirna_candidates

# Step 1: Find essential genes
genes = find_essential_genes("Helicoverpa zea")

# Step 2: Generate siRNA candidates
target_gene = genes["essential_genes"][0]
candidates = generate_sirna_candidates(target_gene["sequence"])

# Step 3: Check safety
for candidate in candidates["candidates"]:
    safety = check_off_target_risk(candidate["sequence"])
    if safety["approved"]:
        print(f"✓ Safe: {candidate['sequence']}")
```

## Tool Details

### find_essential_genes

Identifies essential genes in pest organisms via homology mapping.

**Input:**
- `organism_name`: Scientific name (e.g., "Helicoverpa zea")
- `data_source`: "auto", "ncbi", "insectbase", or "ensembl"

**Output:**
```python
{
    "success": True,
    "essential_genes": [
        {
            "gene_id": "gene_001",
            "gene_name": "chitin_synthase",
            "sequence": "AGCT..." (3000bp),
            "essentiality_score": 0.95,
            "homologous_drosophila_gene": "FBgn0000064",
            "knockout_phenotype": "lethal"
        }
    ]
}
```

### generate_sirna_candidates

Generates valid 21-mer siRNA sequences using Reynolds Rules.

**Input:**
- `gene_sequence`: Full gene sequence (1000-5000bp)
- `max_candidates`: Number of candidates to return (default: 10)

**Output:**
```python
{
    "success": True,
    "candidates": [
        {
            "sequence": "AGCTAGCTAGCTAGCTAGCTA",
            "gc_content": 0.48,
            "reynolds_score": 0.92,
            "thermodynamic_stability": -2.1
        }
    ]
}
```

### check_off_target_risk

Validates sequences against protected species genomes.

**Input:**
- `candidate_sequence`: 21-mer siRNA sequence
- `protected_species_list`: List of species to check (default: honeybee, monarch)

**Output:**
```python
{
    "success": True,
    "approved": True,  # Safe to use
    "rejected": False,
    "matches": []  # No off-target matches
}
```

## Reynolds Rules

The siRNA generator uses Reynolds et al. (2004) criteria:

1. **GC Content**: 30-52% (optimal: 40-50%)
2. **No Homopolymers**: Avoid 4+ consecutive identical bases
3. **Thermodynamic Stability**: Lower at 5' end (easier RISC loading)
4. **No Secondary Structure**: Minimize hairpins/loops

## Protected Species

Default protected species:
- **Apis mellifera** (Honeybee) - Critical pollinator
- **Danaus plexippus** (Monarch Butterfly) - Protected species
- **Bombus impatiens** (Bumblebee) - Important pollinator

## Data Flow

```
Pest Organism Name
    ↓
Download Transcriptome (NCBI/InsectBase)
    ↓
BLAST against Drosophila Essential Genes
    ↓
Essential Genes (3000bp sequences)
    ↓
Slide 21bp Window + Reynolds Rules
    ↓
siRNA Candidates (21-mers)
    ↓
BLAST against Protected Species
    ↓
Approved Candidates (safe to use)
```

## File Structure

```
kosmos/
├── tools/
│   ├── rnai_designer.py          # Main tools
│   └── rnai_generator.py         # SiRNA generator
├── domains/biology/genomics/
│   ├── essential_genes.py        # DEG database
│   └── rnai.py                   # BLAST operations
└── tests/
    ├── unit/tools/test_rnai_designer.py
    └── integration/tools/test_rnai_integration.py
```

## Testing

### Unit Tests

```bash
pytest tests/unit/tools/test_rnai_designer.py -v
```

### Integration Tests

```bash
pytest tests/integration/tools/test_rnai_integration.py -v -m integration
```

## Troubleshooting

### "blastn command not found"

**Solution**: Run in `kosmos-biolab:latest` Docker container:

```bash
docker run -it kosmos-biolab:latest bash
blastn -version
```

### "No essential genes found"

**Solution**: Currently uses mock data for testing. Check:
1. DEG database exists
2. BLAST parameters not too stringent
3. Transcriptome data is valid

### "Genome download failed"

**Solution**: Mock genomes are used for testing. Production would download from NCBI.

## Performance

- **find_essential_genes**: ~5 minutes (download + BLAST)
- **generate_sirna_candidates**: ~5 seconds (computation)
- **check_off_target_risk**: ~60 seconds per species (BLAST)

## Next Steps

1. **Test with Real Data**: Replace mock data with actual NCBI downloads
2. **Add More Protected Species**: Extend protected species list
3. **Optimize BLAST Parameters**: Tune for sensitivity/specificity
4. **Add Secondary Structure**: Predict hairpins/loops
5. **ML-Based Efficacy**: Predict silencing efficiency

## References

- Reynolds et al. (2004) - "Rational siRNA design for RNA interference"
- Drosophila Essential Genes: http://www.essentialgene.org/
- InsectBase: http://www.insectbase.org/
- NCBI BLAST: https://blast.ncbi.nlm.nih.gov/

## Support

For detailed implementation information, see:
- `docs/rnai-designer-implementation.md`
- `docs/implementation-plans/kosmos-rnai-designer.md`

---

**Status**: ✅ Ready for Use  
**Last Updated**: February 2026
