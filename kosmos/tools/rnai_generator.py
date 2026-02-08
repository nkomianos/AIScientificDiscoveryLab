"""
SiRNA Generator - Converts gene sequences into valid siRNA candidates.

This module implements the "weaponizer" step that was missing from the original plan.
It converts long gene sequences (3000bp+) into valid 21-mer siRNA candidates using
Reynolds Rules for efficacy prediction.

CRITICAL: This tool is required immediately after find_essential_genes - agents
cannot "guess" valid siRNA sequences from long gene sequences.
"""

import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Default output directory
DEFAULT_OUTPUT_DIR = "/workspace/output"


# =============================================================================
# Result Types
# =============================================================================

@dataclass
class SiRNAGeneratorResult:
    """Result from siRNA candidate generation."""
    success: bool
    candidates: List[Dict[str, Any]] = None
    total_candidates_generated: int = 0
    filtered_candidates: int = 0
    execution_time_seconds: float = 0.0
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    
    def __post_init__(self):
        if self.candidates is None:
            self.candidates = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "candidates": self.candidates,
            "total_candidates_generated": self.total_candidates_generated,
            "filtered_candidates": self.filtered_candidates,
            "execution_time_seconds": self.execution_time_seconds,
            "error_message": self.error_message,
            "error_type": self.error_type,
        }


# =============================================================================
# Reynolds Rules Implementation
# =============================================================================

def _has_homopolymer_run(sequence: str, min_length: int = 4) -> bool:
    """
    Check for homopolymer runs (e.g., AAAA, GGGG).
    
    Args:
        sequence: DNA sequence
        min_length: Minimum run length to flag (default: 4)
    
    Returns:
        True if homopolymer run found, False otherwise
    """
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
    
    Uses simplified nearest-neighbor model.
    Lower stability = better (easier for RISC to load).
    
    Args:
        sequence_5prime: First 4 nucleotides of siRNA
    
    Returns:
        Stability score (kcal/mol, negative = more stable)
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
        elif pair in ['GT', 'TG']:
            stability -= 1.3
        elif pair in ['AC', 'CA']:
            stability -= 1.2
        elif pair in ['AG', 'GA']:
            stability -= 1.0
        elif pair in ['CT', 'TC']:
            stability -= 1.1
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
    
    Based on Reynolds et al. (2004) - "Rational siRNA design for RNA interference"
    
    Args:
        gc_content: GC content ratio (0.0-1.0)
        stability_5prime: Thermodynamic stability at 5' end (kcal/mol)
        has_repeats: Whether sequence contains homopolymer runs
    
    Returns:
        Reynolds score (0.0-1.0, higher is better)
    """
    if has_repeats:
        return 0.0  # Disqualify sequences with homopolymer runs
    
    # GC content score (optimal: 0.40-0.50)
    if 0.40 <= gc_content <= 0.50:
        gc_score = 1.0
    elif 0.35 <= gc_content < 0.40 or 0.50 < gc_content <= 0.52:
        gc_score = 0.8
    elif 0.30 <= gc_content < 0.35 or 0.52 < gc_content <= 0.55:
        gc_score = 0.5
    else:
        gc_score = 0.2
    
    # Stability score (lower = better, but not too low)
    # Optimal range: -3.0 to -1.0 kcal/mol
    if -3.0 <= stability_5prime <= -1.0:
        stability_score = 1.0
    elif -5.0 <= stability_5prime < -3.0 or -1.0 < stability_5prime <= 0.0:
        stability_score = 0.7
    else:
        stability_score = 0.4
    
    # Composite score (weighted average)
    # GC content is slightly more important (60%)
    reynolds_score = (gc_score * 0.6) + (stability_score * 0.4)
    
    return reynolds_score


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
       d. Calculate Reynolds score
    3. Filter by Reynolds Rules
    4. Sort by score
    
    Args:
        sequence: Gene sequence (DNA, AGCT format)
        window_size: Length of siRNA candidates (default: 21)
        min_gc: Minimum GC content (default: 0.30)
        max_gc: Maximum GC content (default: 0.52)
    
    Returns:
        List of candidate dictionaries with sequence and scores
    """
    candidates = []
    total_windows = 0
    
    # Normalize sequence
    sequence = sequence.upper().strip()
    
    # Slide window
    for i in range(len(sequence) - window_size + 1):
        total_windows += 1
        candidate_seq = sequence[i:i + window_size]
        
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
        
        # Calculate thermodynamic stability (5' end - first 4 bases)
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
    
    logger.info(f"Generated {len(candidates)} valid candidates from {total_windows} windows")
    
    return candidates


# =============================================================================
# Main Function
# =============================================================================

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
    21-mer RNAi agents using established design rules (Reynolds et al., 2004).
    
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
    
    Example:
        >>> result = generate_sirna_candidates("AGCTAGCT..." * 500)  # 4000bp gene
        >>> if result["success"]:
        ...     for candidate in result["candidates"][:3]:
        ...         print(f"{candidate['sequence']} (score: {candidate['reynolds_score']:.2f})")
    """
    start_time = time.time()
    
    # Validate input
    if not gene_sequence or not gene_sequence.strip():
        return SiRNAGeneratorResult(
            success=False,
            error_message="Empty gene sequence provided",
            error_type="ValidationError",
            execution_time_seconds=time.time() - start_time,
        ).to_dict()
    
    gene_sequence = gene_sequence.strip().upper()
    
    # Check sequence length
    if len(gene_sequence) < window_size:
        return SiRNAGeneratorResult(
            success=False,
            error_message=f"Gene sequence too short ({len(gene_sequence)} bp). Must be at least {window_size} bp.",
            error_type="ValidationError",
            execution_time_seconds=time.time() - start_time,
        ).to_dict()
    
    # Check for valid nucleotides
    valid_bases = set('AGCT')
    sequence_bases = set(gene_sequence)
    invalid_bases = sequence_bases - valid_bases
    
    if invalid_bases:
        return SiRNAGeneratorResult(
            success=False,
            error_message=f"Invalid nucleotide characters in sequence: {invalid_bases}",
            error_type="ValidationError",
            execution_time_seconds=time.time() - start_time,
        ).to_dict()
    
    # Validate parameters
    if window_size < 18 or window_size > 30:
        return SiRNAGeneratorResult(
            success=False,
            error_message=f"Invalid window size: {window_size}. Must be between 18 and 30.",
            error_type="ValidationError",
            execution_time_seconds=time.time() - start_time,
        ).to_dict()
    
    if not (0.0 <= min_gc_content <= 1.0) or not (0.0 <= max_gc_content <= 1.0):
        return SiRNAGeneratorResult(
            success=False,
            error_message=f"Invalid GC content range: {min_gc_content}-{max_gc_content}. Must be between 0.0 and 1.0.",
            error_type="ValidationError",
            execution_time_seconds=time.time() - start_time,
        ).to_dict()
    
    if min_gc_content >= max_gc_content:
        return SiRNAGeneratorResult(
            success=False,
            error_message=f"min_gc_content ({min_gc_content}) must be less than max_gc_content ({max_gc_content})",
            error_type="ValidationError",
            execution_time_seconds=time.time() - start_time,
        ).to_dict()
    
    try:
        logger.info(f"Generating siRNA candidates from {len(gene_sequence)}bp sequence...")
        
        # Generate all candidates
        all_candidates = _generate_sirna_candidates(
            sequence=gene_sequence,
            window_size=window_size,
            min_gc=min_gc_content,
            max_gc=max_gc_content
        )
        
        total_generated = len(all_candidates)
        
        # Return top N candidates
        top_candidates = all_candidates[:max_candidates]
        
        logger.info(f"Returning top {len(top_candidates)} candidates (from {total_generated} total)")
        
        return SiRNAGeneratorResult(
            success=True,
            candidates=top_candidates,
            total_candidates_generated=total_generated,
            filtered_candidates=len(top_candidates),
            execution_time_seconds=time.time() - start_time,
        ).to_dict()
        
    except Exception as e:
        logger.error(f"siRNA generation failed: {e}")
        return SiRNAGeneratorResult(
            success=False,
            error_message=str(e),
            error_type=type(e).__name__,
            execution_time_seconds=time.time() - start_time,
        ).to_dict()


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Main function
    "generate_sirna_candidates",
    # Result type
    "SiRNAGeneratorResult",
]
