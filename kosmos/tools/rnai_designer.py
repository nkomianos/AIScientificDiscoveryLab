"""
RNAi Designer Tools - Gene Silencing for Pest Control.

This module provides tools for designing RNAi-based pest control agents:
1. Essential Gene Finder - Identifies essential genes in pest organisms
2. SiRNA Generator - Generates valid 21-mer siRNA candidates from gene sequences
3. Safety Filter - Validates sequences against protected species

These tools enable autonomous design of eco-friendly gene-silencing agents.
"""

import os
import hashlib
import logging
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Default output directory
DEFAULT_OUTPUT_DIR = "/workspace/output"


# =============================================================================
# Result Types
# =============================================================================

@dataclass
class RNAiResult:
    """Base result class for all RNAi operations."""
    success: bool
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    execution_time_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "error_message": self.error_message,
            "error_type": self.error_type,
            "execution_time_seconds": self.execution_time_seconds,
        }


@dataclass
class EssentialGenesResult(RNAiResult):
    """Result from essential gene finding."""
    essential_genes: List[Dict[str, Any]] = None
    organism_info: Dict[str, Any] = None
    total_genes_found: int = 0
    
    def __post_init__(self):
        if self.essential_genes is None:
            self.essential_genes = []
        if self.organism_info is None:
            self.organism_info = {}
    
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "essential_genes": self.essential_genes,
            "organism_info": self.organism_info,
            "total_genes_found": self.total_genes_found,
        })
        return d


@dataclass
class SiRNAGeneratorResult(RNAiResult):
    """Result from siRNA candidate generation."""
    candidates: List[Dict[str, Any]] = None
    total_candidates_generated: int = 0
    filtered_candidates: int = 0
    
    def __post_init__(self):
        if self.candidates is None:
            self.candidates = []
    
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "candidates": self.candidates,
            "total_candidates_generated": self.total_candidates_generated,
            "filtered_candidates": self.filtered_candidates,
        })
        return d


@dataclass
class SafetyCheckResult(RNAiResult):
    """Result from off-target risk checking."""
    approved: bool = False
    rejected: bool = False
    matches: List[Dict[str, Any]] = None
    total_species_checked: int = 0
    
    def __post_init__(self):
        if self.matches is None:
            self.matches = []
    
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "approved": self.approved,
            "rejected": self.rejected,
            "matches": self.matches,
            "total_species_checked": self.total_species_checked,
        })
        return d


# =============================================================================
# Utility Functions
# =============================================================================

def _validate_organism_name(organism_name: str) -> Tuple[bool, str]:
    """Validate organism name."""
    if not organism_name or not organism_name.strip():
        return False, "Empty organism name provided"
    
    organism_name = organism_name.strip()
    
    if len(organism_name) < 3:
        return False, f"Organism name too short: '{organism_name}'"
    
    return True, organism_name


def _validate_rnai_sequence(sequence: str) -> Tuple[bool, str]:
    """
    Validate RNAi candidate sequence.
    
    Requirements:
    - Length: 18-30 nucleotides (typical siRNA length)
    - Characters: Only A, G, C, T (DNA) or A, G, C, U (RNA)
    - No ambiguous bases (N, R, Y, etc.)
    - No gaps or whitespace
    
    Returns:
        (is_valid, error_message_or_cleaned_sequence)
    """
    if not sequence or not sequence.strip():
        return False, "Empty sequence provided"
    
    sequence = sequence.strip().upper()
    
    # Check length
    if len(sequence) < 18:
        return False, f"Sequence too short ({len(sequence)} nucleotides). Minimum is 18."
    
    if len(sequence) > 30:
        return False, f"Sequence too long ({len(sequence)} nucleotides). Maximum is 30."
    
    # Check for valid nucleotides (DNA or RNA)
    valid_dna = set("AGCT")
    valid_rna = set("AGCU")
    sequence_set = set(sequence)
    
    is_dna = sequence_set.issubset(valid_dna)
    is_rna = sequence_set.issubset(valid_rna)
    
    if not (is_dna or is_rna):
        invalid_chars = sequence_set - valid_dna - valid_rna
        return False, f"Invalid nucleotide characters: {invalid_chars}"
    
    # Normalize to DNA (BLAST databases use DNA)
    if is_rna:
        sequence = sequence.replace('U', 'T')
    
    return True, sequence


def _ensure_output_dir(output_dir: str) -> Path:
    """Ensure output directory exists."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _generate_output_name(content: str, suffix: str) -> str:
    """Generate deterministic filename from content hash."""
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
    return f"{content_hash}{suffix}"


# =============================================================================
# Essential Gene Finder
# =============================================================================

def find_essential_genes(
    organism_name: str,
    data_source: str = "auto",
    essentiality_criteria: Optional[Dict[str, Any]] = None,
    output_dir: str = DEFAULT_OUTPUT_DIR
) -> Dict[str, Any]:
    """
    Find essential genes in a target pest organism.
    
    This function downloads transcriptome data and identifies essential genes
    via homology mapping against Drosophila Essential Genes (DEG) database.
    
    Args:
        organism_name: Scientific name or common name of pest (e.g., "Helicoverpa zea")
        data_source: Database to query ("insectbase", "ncbi", "ensembl", or "auto")
        essentiality_criteria: Optional criteria dict:
            - min_essentiality_score: float (0.0-1.0) - Minimum essentiality score
            - gene_functions: List[str] - Filter by gene functions
            - exclude_hypothetical: bool - Exclude hypothetical proteins
        output_dir: Directory to save downloaded data and results
    
    Returns:
        Dict containing:
        - success: bool
        - essential_genes: List[Dict] with gene information
        - organism_info: Dict with organism metadata
        - total_genes_found: int
        - execution_time_seconds: float
        - error_message: Optional[str]
    """
    start_time = time.time()
    
    # Validate input
    valid, organism_or_error = _validate_organism_name(organism_name)
    if not valid:
        return EssentialGenesResult(
            success=False,
            error_message=organism_or_error,
            error_type="ValidationError",
            execution_time_seconds=time.time() - start_time,
        ).to_dict()
    
    organism_name = organism_or_error
    
    try:
        # Ensure output directory exists
        out_path = _ensure_output_dir(output_dir)
        
        logger.info(f"Finding essential genes for {organism_name}...")
        
        # Import genomics module (created separately)
        from kosmos.domains.biology.genomics.essential_genes import (
            identify_essential_genes_via_homology,
            download_transcriptome,
            ensure_deg_database
        )
        
        # Step 1: Download transcriptome
        logger.info(f"Downloading transcriptome from {data_source}...")
        transcriptome_result = download_transcriptome(
            organism_name=organism_name,
            data_source=data_source,
            output_dir=str(out_path)
        )
        
        if not transcriptome_result["success"]:
            return EssentialGenesResult(
                success=False,
                error_message=transcriptome_result.get("error_message", "Failed to download transcriptome"),
                error_type="TranscriptomeDownloadError",
                execution_time_seconds=time.time() - start_time,
            ).to_dict()
        
        transcriptome = transcriptome_result["transcriptome"]
        organism_info = transcriptome_result["organism_info"]
        
        # Step 2: Ensure DEG database exists
        logger.info("Loading Drosophila Essential Genes database...")
        deg_db_path = ensure_deg_database(str(out_path))
        
        # Step 3: Identify essential genes via homology mapping
        logger.info("Identifying essential genes via BLAST homology mapping...")
        essential_genes = identify_essential_genes_via_homology(
            pest_transcriptome=transcriptome,
            deg_database_path=deg_db_path,
            output_dir=str(out_path),
            criteria=essentiality_criteria
        )
        
        logger.info(f"Found {len(essential_genes)} essential genes")
        
        return EssentialGenesResult(
            success=True,
            essential_genes=essential_genes,
            organism_info=organism_info,
            total_genes_found=len(essential_genes),
            execution_time_seconds=time.time() - start_time,
        ).to_dict()
        
    except ImportError as e:
        return EssentialGenesResult(
            success=False,
            error_message=f"Required module not found: {str(e)}. Ensure genomics modules are installed.",
            error_type="ImportError",
            execution_time_seconds=time.time() - start_time,
        ).to_dict()
        
    except Exception as e:
        logger.error(f"Essential gene finding failed: {e}")
        return EssentialGenesResult(
            success=False,
            error_message=str(e),
            error_type=type(e).__name__,
            execution_time_seconds=time.time() - start_time,
        ).to_dict()


# =============================================================================
# Off-Target Risk Checker
# =============================================================================

def check_off_target_risk(
    candidate_sequence: str,
    protected_species_list: Optional[List[str]] = None,
    match_threshold: int = 21,
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
        - matches: List[Dict] with match information
        - total_species_checked: int
        - execution_time_seconds: float
        - error_message: Optional[str]
    """
    start_time = time.time()
    
    # Validate input sequence
    valid, sequence_or_error = _validate_rnai_sequence(candidate_sequence)
    if not valid:
        return SafetyCheckResult(
            success=False,
            error_message=sequence_or_error,
            error_type="ValidationError",
            execution_time_seconds=time.time() - start_time,
        ).to_dict()
    
    candidate_sequence = sequence_or_error
    
    # Default protected species
    if protected_species_list is None:
        protected_species_list = [
            "Apis mellifera",  # Honeybee
            "Danaus plexippus",  # Monarch butterfly
        ]
    
    try:
        # Ensure output directory exists
        out_path = _ensure_output_dir(output_dir)
        
        logger.info(f"Checking off-target risk for sequence: {candidate_sequence[:20]}...")
        
        # Import BLAST module
        from kosmos.domains.biology.genomics.rnai import (
            run_blast_search,
            ensure_protected_genomes,
            parse_blast_results
        )
        
        # Step 1: Ensure protected genomes are downloaded and indexed
        logger.info(f"Loading {len(protected_species_list)} protected species genomes...")
        genome_db_paths = ensure_protected_genomes(
            species_list=protected_species_list,
            output_dir=str(out_path)
        )
        
        # Step 2: Run BLAST search against each protected species
        all_matches = []
        for species in protected_species_list:
            if species not in genome_db_paths:
                logger.warning(f"Genome database not available for {species}")
                continue
            
            logger.info(f"Searching {species} genome...")
            blast_result = run_blast_search(
                query_sequence=candidate_sequence,
                database_path=genome_db_paths[species],
                output_dir=str(out_path),
                match_threshold=match_threshold
            )
            
            if blast_result["matches"]:
                for match in blast_result["matches"]:
                    match["species"] = species
                    all_matches.append(match)
        
        # Step 3: Determine approval status
        approved = len(all_matches) == 0
        rejected = len(all_matches) > 0
        
        if approved:
            logger.info("✓ APPROVED: No off-target matches found")
        else:
            logger.warning(f"✗ REJECTED: Found {len(all_matches)} off-target matches")
        
        return SafetyCheckResult(
            success=True,
            approved=approved,
            rejected=rejected,
            matches=all_matches,
            total_species_checked=len(protected_species_list),
            execution_time_seconds=time.time() - start_time,
        ).to_dict()
        
    except ImportError as e:
        return SafetyCheckResult(
            success=False,
            error_message=f"Required module not found: {str(e)}. Ensure genomics modules are installed.",
            error_type="ImportError",
            execution_time_seconds=time.time() - start_time,
        ).to_dict()
        
    except Exception as e:
        logger.error(f"Off-target risk check failed: {e}")
        return SafetyCheckResult(
            success=False,
            error_message=str(e),
            error_type=type(e).__name__,
            execution_time_seconds=time.time() - start_time,
        ).to_dict()


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Main functions
    "find_essential_genes",
    "check_off_target_risk",
    # Result types
    "RNAiResult",
    "EssentialGenesResult",
    "SafetyCheckResult",
]
