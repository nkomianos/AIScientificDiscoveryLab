#!/usr/bin/env python3
"""
Kosmos RNAi Designer - Example Usage

This script demonstrates how to use the RNAi Designer tools to:
1. Find essential genes in a pest organism
2. Generate siRNA candidates from target genes
3. Validate candidates against protected species

Run this script inside the kosmos-biolab Docker container:
    docker run -v ./kosmos_data:/data -v ./examples:/workspace kosmos-biolab:latest \
        python3 /workspace/rnai_designer_example.py
"""

import sys
from pathlib import Path

# Add kosmos to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kosmos.tools.rnai_designer import (
    find_essential_genes,
    check_off_target_risk,
)
from kosmos.tools.rnai_generator import generate_sirna_candidates


def main():
    """Run complete RNAi design workflow."""
    
    print("=" * 80)
    print("Kosmos RNAi Designer - Example Workflow")
    print("=" * 80)
    print()
    
    # Configuration
    target_organism = "Helicoverpa zea"  # Corn earworm
    protected_species = ["Apis mellifera", "Danaus plexippus"]
    output_dir = "/workspace/output"
    
    # =========================================================================
    # Step 1: Find Essential Genes
    # =========================================================================
    
    print(f"Step 1: Finding essential genes in {target_organism}...")
    print("-" * 80)
    
    genes_result = find_essential_genes(
        organism_name=target_organism,
        data_source="auto",
        output_dir=output_dir
    )
    
    if not genes_result["success"]:
        print(f"❌ Error: {genes_result['error_message']}")
        return 1
    
    essential_genes = genes_result["essential_genes"]
    print(f"✓ Found {len(essential_genes)} essential genes")
    print()
    
    # Display top 5 genes
    print("Top essential genes:")
    for i, gene in enumerate(essential_genes[:5], 1):
        print(f"  {i}. {gene['gene_name']}")
        print(f"     Essentiality: {gene['essentiality_score']:.2f}")
        print(f"     Function: {gene['gene_function']}")
        print(f"     Sequence length: {len(gene['sequence'])} bp")
        print()
    
    # =========================================================================
    # Step 2: Select Target Gene and Generate siRNA Candidates
    # =========================================================================
    
    print("Step 2: Generating siRNA candidates...")
    print("-" * 80)
    
    # Select first essential gene as target
    target_gene = essential_genes[0]
    print(f"Target gene: {target_gene['gene_name']}")
    print(f"Sequence length: {len(target_gene['sequence'])} bp")
    print()
    
    candidates_result = generate_sirna_candidates(
        gene_sequence=target_gene["sequence"],
        window_size=21,
        max_candidates=10,
        min_gc_content=0.30,
        max_gc_content=0.52,
        output_dir=output_dir
    )
    
    if not candidates_result["success"]:
        print(f"❌ Error: {candidates_result['error_message']}")
        return 1
    
    candidates = candidates_result["candidates"]
    print(f"✓ Generated {candidates_result['total_candidates_generated']} candidates")
    print(f"  Returning top {len(candidates)} candidates")
    print()
    
    # Display top 5 candidates
    print("Top siRNA candidates:")
    for i, candidate in enumerate(candidates[:5], 1):
        print(f"  {i}. {candidate['sequence']}")
        print(f"     Position: {candidate['start_position']}-{candidate['end_position']}")
        print(f"     GC content: {candidate['gc_content']:.2%}")
        print(f"     Reynolds score: {candidate['reynolds_score']:.3f}")
        print(f"     Stability (5'): {candidate['thermodynamic_stability']:.2f} kcal/mol")
        print()
    
    # =========================================================================
    # Step 3: Check Off-Target Risk
    # =========================================================================
    
    print("Step 3: Checking off-target risk...")
    print("-" * 80)
    print(f"Protected species: {', '.join(protected_species)}")
    print()
    
    approved_candidates = []
    rejected_candidates = []
    
    for i, candidate in enumerate(candidates[:5], 1):
        print(f"Checking candidate {i}: {candidate['sequence'][:10]}...")
        
        safety_result = check_off_target_risk(
            candidate_sequence=candidate["sequence"],
            protected_species_list=protected_species,
            match_threshold=21,
            output_dir=output_dir
        )
        
        if not safety_result["success"]:
            print(f"  ⚠️  Error: {safety_result['error_message']}")
            continue
        
        if safety_result["approved"]:
            approved_candidates.append(candidate)
            print(f"  ✓ APPROVED - No off-target matches")
        else:
            rejected_candidates.append(candidate)
            matches = safety_result["matches"]
            print(f"  ✗ REJECTED - {len(matches)} off-target matches found:")
            for match in matches[:3]:  # Show first 3 matches
                print(f"     - {match['species']}: {match['gene_id']}")
                print(f"       Identity: {match['identity_percent']:.1f}%")
        
        print()
    
    # =========================================================================
    # Summary
    # =========================================================================
    
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Target organism: {target_organism}")
    print(f"Essential genes found: {len(essential_genes)}")
    print(f"siRNA candidates generated: {candidates_result['total_candidates_generated']}")
    print(f"Candidates tested: {len(candidates[:5])}")
    print(f"Approved candidates: {len(approved_candidates)}")
    print(f"Rejected candidates: {len(rejected_candidates)}")
    print()
    
    if approved_candidates:
        print("✓ SUCCESS: Safe RNAi candidates identified!")
        print()
        print("Approved sequences:")
        for i, candidate in enumerate(approved_candidates, 1):
            print(f"  {i}. {candidate['sequence']}")
            print(f"     Reynolds score: {candidate['reynolds_score']:.3f}")
        print()
        print("These sequences can be synthesized and tested for pest control.")
    else:
        print("⚠️  WARNING: No approved candidates found.")
        print("Consider:")
        print("  - Generating more candidates (increase max_candidates)")
        print("  - Trying a different target gene")
        print("  - Adjusting Reynolds Rules parameters")
    
    print()
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
