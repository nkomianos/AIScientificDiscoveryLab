"""
RNAi Module - BLAST Search and Protected Species Management.

This module handles off-target risk checking by BLASTing RNAi candidates
against protected species genomes (honeybees, monarch butterflies, etc.).
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
import httpx
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

logger = logging.getLogger(__name__)


# =============================================================================
# Protected Species Configuration
# =============================================================================

DEFAULT_PROTECTED_SPECIES = [
    {
        "scientific_name": "Apis mellifera",
        "common_name": "Honeybee",
        "priority": "critical",  # Essential pollinator
        "ncbi_assembly": "GCF_003254395.2",
        "genome_size_mb": 250,
    },
    {
        "scientific_name": "Danaus plexippus",
        "common_name": "Monarch Butterfly",
        "priority": "high",  # Protected species
        "ncbi_assembly": "GCF_000235995.1",
        "genome_size_mb": 300,
    },
    {
        "scientific_name": "Bombus impatiens",
        "common_name": "Common Eastern Bumblebee",
        "priority": "high",
        "ncbi_assembly": "GCF_000188095.3",
        "genome_size_mb": 280,
    },
]


# =============================================================================
# Protected Genome Management
# =============================================================================

def ensure_protected_genomes(
    species_list: List[str],
    output_dir: str
) -> Dict[str, str]:
    """
    Download and index genomes for protected species.
    
    Strategy:
    - Check if genome already exists
    - If not, download from NCBI or use mock data
    - Create BLAST database
    - Return paths to BLAST databases
    
    Args:
        species_list: List of species names
        output_dir: Output directory for genomes
    
    Returns:
        Dict mapping species name to BLAST database path
    """
    output_path = Path(output_dir) / "protected_genomes"
    output_path.mkdir(parents=True, exist_ok=True)
    
    genome_db_paths = {}
    
    for species_name in species_list:
        logger.info(f"Ensuring genome for {species_name}...")
        
        # Get species config
        species_config = _get_species_config(species_name)
        if not species_config:
            logger.warning(f"Unknown species: {species_name}. Skipping.")
            continue
        
        # Check if BLAST database exists
        species_slug = species_name.replace(" ", "_").lower()
        species_dir = output_path / species_slug
        blast_db_dir = species_dir / "blastdb"
        fasta_file = species_dir / f"{species_slug}.fasta"
        
        if blast_db_dir.exists() and (blast_db_dir / f"{species_slug}.nhr").exists():
            logger.info(f"BLAST database already exists for {species_name}")
            genome_db_paths[species_name] = str(blast_db_dir / species_slug)
            continue
        
        # Download genome
        species_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            if not fasta_file.exists():
                _download_genome(species_config, fasta_file)
        except Exception as e:
            logger.warning(f"Failed to download genome for {species_name}: {e}")
            logger.info("Creating mock genome for testing...")
            _create_mock_genome(species_name, fasta_file)
        
        # Create BLAST database
        blast_db_dir.mkdir(parents=True, exist_ok=True)
        _create_blast_database(fasta_file, blast_db_dir / species_slug)
        
        genome_db_paths[species_name] = str(blast_db_dir / species_slug)
    
    logger.info(f"Prepared {len(genome_db_paths)} protected species genomes")
    return genome_db_paths


def _get_species_config(species_name: str) -> Optional[Dict[str, Any]]:
    """Get configuration for a species."""
    for config in DEFAULT_PROTECTED_SPECIES:
        if config["scientific_name"] == species_name:
            return config
    return None


def _download_genome(species_config: Dict[str, Any], output_file: Path):
    """
    Download genome from NCBI.
    
    In production, this would:
    1. Query NCBI Assembly database
    2. Download genomic FASTA
    3. Decompress and save
    
    For now, raises NotImplementedError to trigger mock data generation.
    """
    raise NotImplementedError("Genome download not yet implemented. Using mock data.")


def _create_mock_genome(species_name: str, output_file: Path):
    """
    Create mock genome for testing.
    
    Generates synthetic genome sequences that can be used for testing
    BLAST searches without downloading large genome files.
    """
    logger.info(f"Creating mock genome for {species_name}...")
    
    # Generate 10 random "chromosomes" with different sequences
    import random
    random.seed(hash(species_name))  # Deterministic for each species
    
    bases = ['A', 'G', 'C', 'T']
    records = []
    
    for chr_num in range(1, 11):
        # Generate 10kb random sequence
        sequence = ''.join(random.choices(bases, k=10000))
        
        record = SeqRecord(
            Seq(sequence),
            id=f"chr{chr_num}",
            description=f"{species_name} chromosome {chr_num} (mock)"
        )
        records.append(record)
    
    # Write FASTA
    with open(output_file, 'w') as f:
        SeqIO.write(records, f, "fasta")
    
    logger.info(f"Created mock genome with {len(records)} chromosomes at {output_file}")


def _create_blast_database(fasta_file: Path, db_prefix: Path):
    """Create BLAST database from FASTA file."""
    try:
        logger.info(f"Creating BLAST database: {db_prefix}...")
        
        cmd = [
            "makeblastdb",
            "-in", str(fasta_file),
            "-dbtype", "nucl",
            "-out", str(db_prefix),
            "-title", db_prefix.name,
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"BLAST database created: {db_prefix}")
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"makeblastdb failed: {e.stderr.decode()}")
    except FileNotFoundError:
        raise RuntimeError("makeblastdb command not found. Install ncbi-blast+ package.")


# =============================================================================
# BLAST Search Operations
# =============================================================================

def run_blast_search(
    query_sequence: str,
    database_path: str,
    output_dir: str,
    match_threshold: int = 21
) -> Dict[str, Any]:
    """
    Run BLAST search for off-target matches.
    
    Args:
        query_sequence: RNAi candidate sequence (DNA format)
        database_path: Path to BLAST database
        output_dir: Output directory for results
        match_threshold: Minimum match length to report
    
    Returns:
        Dict with:
        - success: bool
        - matches: List[Dict] with match information
        - error_message: Optional str
    """
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create temporary query file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as query_file:
            query_file.write(f">query\n{query_sequence}\n")
            query_path = query_file.name
        
        # Run BLAST
        blast_output = output_path / "blast_output.xml"
        
        _run_blastn(
            query_file=query_path,
            database_path=database_path,
            output_file=str(blast_output),
            word_size=7,
            e_value=10.0
        )
        
        # Parse results
        matches = parse_blast_results(
            blast_output=str(blast_output),
            match_threshold=match_threshold
        )
        
        # Clean up
        Path(query_path).unlink()
        
        return {
            "success": True,
            "matches": matches,
        }
        
    except Exception as e:
        logger.error(f"BLAST search failed: {e}")
        return {
            "success": False,
            "matches": [],
            "error_message": str(e),
        }


def _run_blastn(
    query_file: str,
    database_path: str,
    output_file: str,
    word_size: int = 7,
    e_value: float = 10.0
):
    """
    Run BLASTN with parameters optimized for short sequences.
    
    Uses blastn-short task for improved sensitivity with short queries.
    """
    try:
        cmd = [
            "blastn",
            "-task", "blastn-short",  # Optimized for short sequences
            "-query", query_file,
            "-db", database_path,
            "-out", output_file,
            "-outfmt", "5",  # XML output
            "-word_size", str(word_size),
            "-evalue", str(e_value),
            "-dust", "no",  # Disable low-complexity filtering
            "-soft_masking", "false",
            "-num_threads", "4",
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True)
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"blastn failed: {e.stderr.decode()}")
    except FileNotFoundError:
        raise RuntimeError("blastn command not found. Install ncbi-blast+ package.")


def parse_blast_results(
    blast_output: str,
    match_threshold: int = 21
) -> List[Dict[str, Any]]:
    """
    Parse BLAST XML output and extract matches.
    
    Args:
        blast_output: Path to BLAST XML output file
        match_threshold: Minimum consecutive match length
    
    Returns:
        List of match dictionaries
    """
    from Bio.Blast import NCBIXML
    
    matches = []
    
    try:
        with open(blast_output, 'r') as f:
            blast_records = NCBIXML.parse(f)
            
            for record in blast_records:
                for alignment in record.alignments:
                    for hsp in alignment.hsps:
                        # Check if match length meets threshold
                        if hsp.align_length < match_threshold:
                            continue
                        
                        # Calculate identity percentage
                        identity = (hsp.identities / hsp.align_length) * 100
                        
                        # Only report high-identity matches (≥90%)
                        if identity < 90.0:
                            continue
                        
                        match = {
                            "gene_id": alignment.hit_def.split()[0],
                            "match_start": hsp.sbjct_start,
                            "match_end": hsp.sbjct_end,
                            "match_length": hsp.align_length,
                            "e_value": float(hsp.expect),
                            "identity_percent": float(identity),
                            "query_coverage": (hsp.align_length / record.query_length) * 100,
                        }
                        
                        matches.append(match)
        
        logger.info(f"Found {len(matches)} off-target matches")
        return matches
        
    except Exception as e:
        logger.error(f"Failed to parse BLAST results: {e}")
        return []


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "ensure_protected_genomes",
    "run_blast_search",
    "parse_blast_results",
    "DEFAULT_PROTECTED_SPECIES",
]
