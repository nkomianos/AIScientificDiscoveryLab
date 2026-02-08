"""
Integration tests for RNAi Designer tools.

These tests require:
- BLAST+ installed
- Network access (for genome downloads)
- Sufficient disk space for genome databases
"""

import pytest
from pathlib import Path

from kosmos.tools.rnai_designer import (
    find_essential_genes,
    check_off_target_risk,
)
from kosmos.tools.rnai_generator import generate_sirna_candidates
from kosmos.domains.biology.genomics.essential_genes import (
    ensure_deg_database,
    download_transcriptome,
)
from kosmos.domains.biology.genomics.rnai import (
    ensure_protected_genomes,
    run_blast_search,
)


@pytest.mark.integration
@pytest.mark.requires_network
@pytest.mark.requires_blast
class TestRNAiIntegration:
    """Integration tests for RNAi tools."""
    
    def test_deg_database_creation(self, tmp_path):
        """Test DEG database download and BLAST DB creation."""
        deg_db_path = ensure_deg_database(str(tmp_path))
        
        assert Path(deg_db_path).exists()
        assert (Path(deg_db_path) / "deg.nhr").exists()
    
    def test_protected_genome_creation(self, tmp_path):
        """Test protected species genome database creation."""
        genome_paths = ensure_protected_genomes(
            species_list=["Apis mellifera"],
            output_dir=str(tmp_path)
        )
        
        assert "Apis mellifera" in genome_paths
        assert Path(genome_paths["Apis mellifera"] + ".nhr").exists()
    
    def test_generate_sirna_candidates_real(self):
        """Test siRNA generation with real gene sequence."""
        # Example gene sequence (partial chitin synthase)
        gene_sequence = (
            "ATGGCGAAGCTGCTGAAGCTGATCGCCGAGCGCATCAAGCAGATCCTGGACGAGGCC"
            "ATCGAGGAGATCCTGCGCCAGCTGATCGCCGAGCGCATCAAGCAGATCCTGGACGAG"
            "GCCATCGAGGAGATCCTGCGCCAGCTGATCGCCGAGCGCATCAAGCAGATCCTGGAC"
        ) * 10  # 1700bp
        
        result = generate_sirna_candidates(
            gene_sequence=gene_sequence,
            max_candidates=10
        )
        
        assert result["success"]
        assert len(result["candidates"]) > 0
        
        # Verify candidates meet Reynolds criteria
        for candidate in result["candidates"]:
            assert len(candidate["sequence"]) == 21
            assert 0.30 <= candidate["gc_content"] <= 0.52
            assert candidate["has_repeats"] is False
            assert candidate["reynolds_score"] > 0.0
    
    def test_mock_transcriptome_download(self, tmp_path):
        """Test transcriptome download (with mock data)."""
        result = download_transcriptome(
            organism_name="Test organism",
            data_source="auto",
            output_dir=str(tmp_path)
        )
        
        assert result["success"]
        assert len(result["transcriptome"]) > 0
        assert "organism_info" in result
    
    @pytest.mark.slow
    def test_full_workflow_end_to_end(self, tmp_path):
        """Test complete RNAi design workflow end-to-end."""
        # Step 1: Find essential genes
        print("\n1. Finding essential genes...")
        genes_result = find_essential_genes(
            organism_name="Test pest",
            data_source="auto",
            output_dir=str(tmp_path)
        )
        
        assert genes_result["success"], f"Failed: {genes_result.get('error_message')}"
        assert len(genes_result["essential_genes"]) > 0
        
        print(f"   Found {len(genes_result['essential_genes'])} essential genes")
        
        # Step 2: Generate siRNA candidates
        print("\n2. Generating siRNA candidates...")
        target_gene = genes_result["essential_genes"][0]
        print(f"   Target gene: {target_gene['gene_id']}")
        
        candidates_result = generate_sirna_candidates(
            gene_sequence=target_gene["sequence"],
            max_candidates=10
        )
        
        assert candidates_result["success"]
        assert len(candidates_result["candidates"]) > 0
        
        print(f"   Generated {len(candidates_result['candidates'])} candidates")
        
        # Step 3: Check off-target risk for each candidate
        print("\n3. Checking off-target risk...")
        approved_count = 0
        rejected_count = 0
        
        for i, candidate in enumerate(candidates_result["candidates"][:3], 1):
            safety_result = check_off_target_risk(
                candidate_sequence=candidate["sequence"],
                output_dir=str(tmp_path)
            )
            
            assert safety_result["success"]
            
            if safety_result["approved"]:
                approved_count += 1
                print(f"   Candidate {i}: ✓ APPROVED")
            else:
                rejected_count += 1
                print(f"   Candidate {i}: ✗ REJECTED ({len(safety_result['matches'])} matches)")
        
        print(f"\n   Summary: {approved_count} approved, {rejected_count} rejected")
        
        # At least one candidate should be checked
        assert approved_count + rejected_count > 0


@pytest.mark.integration
@pytest.mark.requires_blast
class TestBLASTOperations:
    """Test BLAST search operations."""
    
    def test_blast_search_no_match(self, tmp_path):
        """Test BLAST search with sequence unlikely to match."""
        # First ensure we have a protected genome database
        genome_paths = ensure_protected_genomes(
            species_list=["Apis mellifera"],
            output_dir=str(tmp_path)
        )
        
        # Create a random sequence unlikely to match
        query_sequence = "ACGTACGTACGTACGTACGTA"
        
        result = run_blast_search(
            query_sequence=query_sequence,
            database_path=genome_paths["Apis mellifera"],
            output_dir=str(tmp_path),
            match_threshold=21
        )
        
        assert result["success"]
        # May or may not have matches (random sequence)
        assert isinstance(result["matches"], list)
