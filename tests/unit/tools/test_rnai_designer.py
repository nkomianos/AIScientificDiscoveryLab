"""
Unit tests for RNAi Designer tools.

Tests the three main RNAi tools:
1. find_essential_genes
2. generate_sirna_candidates
3. check_off_target_risk
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Tools to test
from kosmos.tools.rnai_designer import (
    find_essential_genes,
    check_off_target_risk,
    _validate_organism_name,
    _validate_rnai_sequence,
)
from kosmos.tools.rnai_generator import (
    generate_sirna_candidates,
    _has_homopolymer_run,
    _calculate_5prime_stability,
    _calculate_reynolds_score,
)


# =============================================================================
# Test Validation Functions
# =============================================================================

class TestValidation:
    """Test input validation functions."""
    
    def test_validate_organism_name_valid(self):
        """Test valid organism name."""
        valid, result = _validate_organism_name("Helicoverpa zea")
        assert valid is True
        assert result == "Helicoverpa zea"
    
    def test_validate_organism_name_empty(self):
        """Test empty organism name."""
        valid, error = _validate_organism_name("")
        assert valid is False
        assert "Empty" in error
    
    def test_validate_organism_name_too_short(self):
        """Test organism name too short."""
        valid, error = _validate_organism_name("AB")
        assert valid is False
        assert "too short" in error
    
    def test_validate_rnai_sequence_valid_dna(self):
        """Test valid DNA sequence."""
        valid, result = _validate_rnai_sequence("AGCTAGCTAGCTAGCTAGCTA")
        assert valid is True
        assert result == "AGCTAGCTAGCTAGCTAGCTA"
    
    def test_validate_rnai_sequence_valid_rna(self):
        """Test valid RNA sequence (U instead of T)."""
        valid, result = _validate_rnai_sequence("AGCUAGCUAGCUAGCUAGCUA")
        assert valid is True
        assert result == "AGCTAGCTAGCTAGCTAGCTA"  # Converted to DNA
    
    def test_validate_rnai_sequence_too_short(self):
        """Test sequence too short."""
        valid, error = _validate_rnai_sequence("AGCT")
        assert valid is False
        assert "short" in error.lower()
    
    def test_validate_rnai_sequence_too_long(self):
        """Test sequence too long."""
        valid, error = _validate_rnai_sequence("A" * 31)
        assert valid is False
        assert "long" in error.lower()
    
    def test_validate_rnai_sequence_invalid_chars(self):
        """Test sequence with invalid characters."""
        valid, error = _validate_rnai_sequence("AGCTNXXX")
        assert valid is False
        assert "invalid" in error.lower()


# =============================================================================
# Test SiRNA Generator
# =============================================================================

class TestSiRNAGenerator:
    """Test siRNA candidate generation."""
    
    def test_has_homopolymer_run_true(self):
        """Test detection of homopolymer runs."""
        assert _has_homopolymer_run("AGCTAAAAGCT", min_length=4) is True
        assert _has_homopolymer_run("AGCTGGGGAGCT", min_length=4) is True
    
    def test_has_homopolymer_run_false(self):
        """Test no homopolymer runs."""
        assert _has_homopolymer_run("AGCTAGCTAGCT", min_length=4) is False
        assert _has_homopolymer_run("AGCAAAGCT", min_length=4) is False  # Only 3 As
    
    def test_calculate_5prime_stability(self):
        """Test 5' end stability calculation."""
        # GC-rich (more stable = more negative)
        stability_gc = _calculate_5prime_stability("GCGC")
        # AT-rich (less stable = less negative)
        stability_at = _calculate_5prime_stability("ATAT")
        
        assert stability_gc < stability_at  # GC more stable (more negative)
    
    def test_calculate_reynolds_score_optimal(self):
        """Test Reynolds score for optimal parameters."""
        score = _calculate_reynolds_score(
            gc_content=0.45,  # Optimal
            stability_5prime=-2.0,  # Optimal
            has_repeats=False
        )
        assert score > 0.8  # Should be high
    
    def test_calculate_reynolds_score_with_repeats(self):
        """Test Reynolds score disqualifies sequences with repeats."""
        score = _calculate_reynolds_score(
            gc_content=0.45,
            stability_5prime=-2.0,
            has_repeats=True
        )
        assert score == 0.0  # Disqualified
    
    def test_generate_sirna_candidates_basic(self):
        """Test basic siRNA candidate generation."""
        # Create a gene sequence (100bp)
        gene_sequence = "AGCTAGCTAGCTAGCTAGCT" * 5
        
        result = generate_sirna_candidates(
            gene_sequence=gene_sequence,
            window_size=21,
            max_candidates=5
        )
        
        assert result["success"] is True
        assert len(result["candidates"]) > 0
        assert result["candidates"][0]["sequence"]
        assert len(result["candidates"][0]["sequence"]) == 21
    
    def test_generate_sirna_candidates_empty_sequence(self):
        """Test with empty sequence."""
        result = generate_sirna_candidates(gene_sequence="")
        
        assert result["success"] is False
        assert "Empty" in result["error_message"]
    
    def test_generate_sirna_candidates_sequence_too_short(self):
        """Test with sequence too short for window size."""
        result = generate_sirna_candidates(
            gene_sequence="AGCTAGCT",  # Only 8bp
            window_size=21
        )
        
        assert result["success"] is False
        assert "too short" in result["error_message"]
    
    def test_generate_sirna_candidates_invalid_characters(self):
        """Test with invalid nucleotides."""
        result = generate_sirna_candidates(
            gene_sequence="AGCTXYZAGCT" * 10
        )
        
        assert result["success"] is False
        assert "Invalid" in result["error_message"]
    
    def test_generate_sirna_candidates_filters_by_gc(self):
        """Test GC content filtering."""
        # All GC (100% GC content - should be filtered out)
        gc_rich = "GCGCGCGCGCGCGCGCGCGCGC" * 5
        
        result = generate_sirna_candidates(
            gene_sequence=gc_rich,
            min_gc_content=0.30,
            max_gc_content=0.52
        )
        
        assert result["success"] is True
        # Should have very few or no candidates due to high GC
        assert result["total_candidates_generated"] < 20


# =============================================================================
# Test Essential Gene Finder
# =============================================================================

class TestEssentialGeneFinder:
    """Test essential gene identification."""
    
    @patch('kosmos.tools.rnai_designer.download_transcriptome')
    @patch('kosmos.tools.rnai_designer.ensure_deg_database')
    @patch('kosmos.tools.rnai_designer.identify_essential_genes_via_homology')
    def test_find_essential_genes_success(
        self,
        mock_identify,
        mock_ensure_deg,
        mock_download
    ):
        """Test successful essential gene finding."""
        # Mock transcriptome download
        mock_download.return_value = {
            "success": True,
            "transcriptome": {
                "gene_001": "AGCTAGCT" * 100,
                "gene_002": "GCTAGCTA" * 100,
            },
            "organism_info": {
                "name": "Helicoverpa zea",
                "source": "mock"
            }
        }
        
        # Mock DEG database
        mock_ensure_deg.return_value = "/path/to/deg"
        
        # Mock essential gene identification
        mock_identify.return_value = [
            {
                "gene_id": "gene_001",
                "gene_name": "chitin_synthase",
                "sequence": "AGCTAGCT" * 100,
                "essentiality_score": 0.95,
            }
        ]
        
        result = find_essential_genes(
            organism_name="Helicoverpa zea",
            output_dir="/tmp/test"
        )
        
        assert result["success"] is True
        assert len(result["essential_genes"]) == 1
        assert result["essential_genes"][0]["gene_name"] == "chitin_synthase"
    
    def test_find_essential_genes_invalid_organism(self):
        """Test with invalid organism name."""
        result = find_essential_genes(organism_name="")
        
        assert result["success"] is False
        assert "Empty" in result["error_message"]


# =============================================================================
# Test Off-Target Risk Checker
# =============================================================================

class TestOffTargetRiskChecker:
    """Test off-target risk checking."""
    
    @patch('kosmos.tools.rnai_designer.ensure_protected_genomes')
    @patch('kosmos.tools.rnai_designer.run_blast_search')
    def test_check_off_target_risk_approved(
        self,
        mock_blast,
        mock_genomes
    ):
        """Test sequence with no matches (approved)."""
        # Mock genome databases
        mock_genomes.return_value = {
            "Apis mellifera": "/path/to/apis_blastdb",
            "Danaus plexippus": "/path/to/danaus_blastdb"
        }
        
        # Mock BLAST search with no matches
        mock_blast.return_value = {
            "success": True,
            "matches": []
        }
        
        result = check_off_target_risk(
            candidate_sequence="AGCTAGCTAGCTAGCTAGCTA"
        )
        
        assert result["success"] is True
        assert result["approved"] is True
        assert result["rejected"] is False
        assert len(result["matches"]) == 0
    
    @patch('kosmos.tools.rnai_designer.ensure_protected_genomes')
    @patch('kosmos.tools.rnai_designer.run_blast_search')
    def test_check_off_target_risk_rejected(
        self,
        mock_blast,
        mock_genomes
    ):
        """Test sequence with matches (rejected)."""
        # Mock genome databases
        mock_genomes.return_value = {
            "Apis mellifera": "/path/to/apis_blastdb"
        }
        
        # Mock BLAST search with matches
        mock_blast.return_value = {
            "success": True,
            "matches": [
                {
                    "gene_id": "gene_12345",
                    "match_length": 21,
                    "e_value": 1e-10,
                    "identity_percent": 95.0
                }
            ]
        }
        
        result = check_off_target_risk(
            candidate_sequence="AGCTAGCTAGCTAGCTAGCTA",
            protected_species_list=["Apis mellifera"]
        )
        
        assert result["success"] is True
        assert result["approved"] is False
        assert result["rejected"] is True
        assert len(result["matches"]) > 0
    
    def test_check_off_target_risk_invalid_sequence(self):
        """Test with invalid sequence."""
        result = check_off_target_risk(candidate_sequence="INVALID123")
        
        assert result["success"] is False
        assert "invalid" in result["error_message"].lower()


# =============================================================================
# Integration Test (requires Docker/BLAST)
# =============================================================================

@pytest.mark.integration
@pytest.mark.requires_docker
class TestRNAiIntegration:
    """Integration tests requiring Docker and BLAST."""
    
    def test_full_workflow(self, tmp_path):
        """Test complete RNAi design workflow."""
        # This test would run in actual Docker container
        # For now, it's a placeholder
        pytest.skip("Requires Docker environment with BLAST+")
        
        # Step 1: Find essential genes
        genes_result = find_essential_genes(
            organism_name="Drosophila melanogaster",
            output_dir=str(tmp_path)
        )
        
        assert genes_result["success"]
        assert len(genes_result["essential_genes"]) > 0
        
        # Step 2: Generate siRNA candidates
        target_gene = genes_result["essential_genes"][0]
        candidates_result = generate_sirna_candidates(
            gene_sequence=target_gene["sequence"],
            max_candidates=5
        )
        
        assert candidates_result["success"]
        assert len(candidates_result["candidates"]) > 0
        
        # Step 3: Check off-target risk
        candidate = candidates_result["candidates"][0]
        safety_result = check_off_target_risk(
            candidate_sequence=candidate["sequence"]
        )
        
        assert safety_result["success"]
