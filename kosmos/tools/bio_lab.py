"""
BioLab Tools - Computational Biology Instruments for Drug Discovery.

This module provides high-level functions for the drug discovery pipeline:
1. Structure Prediction - Predict 3D protein structures from sequences (ESMFold)
2. Molecular Docking - Dock small molecules to protein targets (AutoDock Vina)
3. Molecular Dynamics - Simulate protein/complex stability (OpenMM)

These functions are designed to be called by LLM agents and handle all file I/O
automatically, returning structured results with clear success/failure status.

Usage:
    These tools run inside the kosmos-biolab Docker container which has all
    required dependencies pre-installed. Do not import this module directly
    in the host environment - it is meant to be executed in the sandbox.
    
Example (within Docker sandbox):
    from kosmos.tools.bio_lab import predict_structure, dock_molecule, run_simulation
    
    # Predict structure
    result = predict_structure("MKFLILLFNILCLFPVLAADNHGVGPQGAS...")
    
    # Dock a molecule
    docking = dock_molecule(result["pdb_path"], "CC(=O)Oc1ccccc1C(=O)O")
    
    # Simulate stability
    simulation = run_simulation(result["pdb_path"], duration_ns=1.0)
"""

import os
import hashlib
import logging
import time
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Default output directory (mounted in Docker container)
DEFAULT_OUTPUT_DIR = "/workspace/output"


# =============================================================================
# Result Types
# =============================================================================

@dataclass
class BioLabResult:
    """Base result class for all BioLab operations."""
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
class StructurePredictionResult(BioLabResult):
    """Result from structure prediction."""
    pdb_path: Optional[str] = None
    sequence_length: int = 0
    mean_plddt: float = 0.0  # Confidence score (0-100)
    
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "pdb_path": self.pdb_path,
            "sequence_length": self.sequence_length,
            "mean_plddt": self.mean_plddt,
        })
        return d


@dataclass
class DockingResult(BioLabResult):
    """Result from molecular docking."""
    best_affinity: float = 0.0  # kcal/mol (negative = better)
    docked_pose_path: Optional[str] = None
    all_poses: List[Dict[str, Any]] = None
    ligand_pdbqt_path: Optional[str] = None
    receptor_pdbqt_path: Optional[str] = None
    
    def __post_init__(self):
        if self.all_poses is None:
            self.all_poses = []
    
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "best_affinity": self.best_affinity,
            "docked_pose_path": self.docked_pose_path,
            "all_poses": self.all_poses,
            "ligand_pdbqt_path": self.ligand_pdbqt_path,
            "receptor_pdbqt_path": self.receptor_pdbqt_path,
        })
        return d


@dataclass
class SimulationResult(BioLabResult):
    """Result from molecular dynamics simulation."""
    stability_assessment: str = ""  # "stable", "unstable", "inconclusive"
    average_rmsd: float = 0.0  # Angstroms
    max_rmsd: float = 0.0
    final_rmsd: float = 0.0
    trajectory_path: Optional[str] = None
    final_structure_path: Optional[str] = None
    energy_data: Optional[Dict[str, float]] = None
    simulation_time_ns: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "stability_assessment": self.stability_assessment,
            "average_rmsd": self.average_rmsd,
            "max_rmsd": self.max_rmsd,
            "final_rmsd": self.final_rmsd,
            "trajectory_path": self.trajectory_path,
            "final_structure_path": self.final_structure_path,
            "energy_data": self.energy_data,
            "simulation_time_ns": self.simulation_time_ns,
        })
        return d


# =============================================================================
# Utility Functions
# =============================================================================

def _validate_sequence(sequence: str) -> Tuple[bool, str]:
    """Validate amino acid sequence."""
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    sequence = sequence.upper().strip()
    
    if not sequence:
        return False, "Empty sequence provided"
    
    if len(sequence) < 10:
        return False, f"Sequence too short ({len(sequence)} residues). Minimum is 10."
    
    if len(sequence) > 2000:
        return False, f"Sequence too long ({len(sequence)} residues). Maximum is 2000 for ESMFold."
    
    invalid_chars = set(sequence) - valid_aa
    if invalid_chars:
        return False, f"Invalid amino acid characters: {invalid_chars}"
    
    return True, sequence


def _validate_smiles(smiles: str) -> Tuple[bool, str]:
    """Validate SMILES string using RDKit."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, f"Invalid SMILES string: '{smiles}'"
        return True, smiles
    except ImportError:
        # RDKit not available - basic validation only
        if not smiles or len(smiles) < 1:
            return False, "Empty SMILES string"
        return True, smiles
    except Exception as e:
        return False, f"SMILES validation error: {str(e)}"


def _validate_pdb_file(pdb_path: str) -> Tuple[bool, str]:
    """Validate PDB file exists and has content."""
    path = Path(pdb_path)
    
    if not path.exists():
        return False, f"PDB file not found: {pdb_path}"
    
    if not path.is_file():
        return False, f"Path is not a file: {pdb_path}"
    
    if path.stat().st_size == 0:
        return False, f"PDB file is empty: {pdb_path}"
    
    # Check for valid PDB content
    with open(path, 'r') as f:
        content = f.read(1000)
        if 'ATOM' not in content and 'HETATM' not in content:
            return False, f"File does not appear to be a valid PDB: {pdb_path}"
    
    return True, str(path.absolute())


def _generate_output_name(content: str, suffix: str) -> str:
    """Generate deterministic filename from content hash."""
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
    return f"{content_hash}{suffix}"


def _ensure_output_dir(output_dir: str) -> Path:
    """Ensure output directory exists."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


# =============================================================================
# Structure Prediction (ESMFold)
# =============================================================================

def predict_structure(
    sequence: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    output_name: Optional[str] = None,
    use_api: bool = False
) -> Dict[str, Any]:
    """
    Predict 3D protein structure from amino acid sequence using ESMFold.
    
    This function uses Facebook's ESMFold model via HuggingFace Transformers
    to predict the 3D structure of a protein from its primary sequence.
    
    Args:
        sequence: Amino acid sequence in single-letter code (10-2000 residues)
        output_dir: Directory to save output PDB file
        output_name: Optional custom name for output file (without extension)
        use_api: If True, use ESM Atlas API instead of local model
    
    Returns:
        Dict containing:
        - success: bool
        - pdb_path: Path to generated PDB file
        - sequence_length: Number of residues
        - mean_plddt: Average confidence score (0-100, higher is better)
        - error_message: Error description if failed
    
    Example:
        >>> result = predict_structure("MKFLILLFNILCLFPVLAADNHGVGPQGAS")
        >>> if result["success"]:
        ...     print(f"Structure saved to: {result['pdb_path']}")
        ...     print(f"Confidence: {result['mean_plddt']:.1f}")
    """
    start_time = time.time()
    
    # Validate input
    valid, sequence_or_error = _validate_sequence(sequence)
    if not valid:
        return StructurePredictionResult(
            success=False,
            error_message=sequence_or_error,
            error_type="ValidationError",
            execution_time_seconds=time.time() - start_time,
        ).to_dict()
    
    sequence = sequence_or_error
    
    try:
        # Ensure output directory exists
        out_path = _ensure_output_dir(output_dir)
        
        # Generate output filename
        if output_name:
            pdb_filename = f"{output_name}.pdb"
        else:
            pdb_filename = _generate_output_name(sequence, ".pdb")
        
        pdb_path = out_path / pdb_filename
        
        if use_api:
            # Use ESM Atlas API (simpler, no GPU needed)
            pdb_content, plddt = _predict_structure_api(sequence)
        else:
            # Use local ESMFold model via transformers
            pdb_content, plddt = _predict_structure_local(sequence)
        
        # Save PDB file
        with open(pdb_path, 'w') as f:
            f.write(pdb_content)
        
        logger.info(f"Structure prediction complete: {pdb_path}")
        
        return StructurePredictionResult(
            success=True,
            pdb_path=str(pdb_path),
            sequence_length=len(sequence),
            mean_plddt=plddt,
            execution_time_seconds=time.time() - start_time,
        ).to_dict()
        
    except ImportError as e:
        return StructurePredictionResult(
            success=False,
            error_message=f"Required package not installed: {str(e)}. Run in kosmos-biolab container.",
            error_type="ImportError",
            execution_time_seconds=time.time() - start_time,
        ).to_dict()
        
    except Exception as e:
        logger.error(f"Structure prediction failed: {e}")
        return StructurePredictionResult(
            success=False,
            error_message=str(e),
            error_type=type(e).__name__,
            execution_time_seconds=time.time() - start_time,
        ).to_dict()


def _predict_structure_local(sequence: str) -> Tuple[str, float]:
    """
    Predict structure using local ESMFold model.
    
    Returns:
        Tuple of (pdb_content, mean_plddt)
    """
    import torch
    from transformers import EsmForProteinFolding, AutoTokenizer
    
    logger.info(f"Loading ESMFold model for sequence of length {len(sequence)}...")
    
    # Load model (will download on first use)
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
    
    # Use CPU or GPU based on availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    # For long sequences, use chunking
    if len(sequence) > 400:
        model.set_chunk_size(128)
    
    # Tokenize and predict
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    inputs = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract PDB content
    pdb_content = model.output_to_pdb(outputs)[0]
    
    # Calculate mean pLDDT (confidence score)
    plddt = outputs.plddt[0].mean().item()
    
    logger.info(f"Prediction complete. Mean pLDDT: {plddt:.2f}")
    
    return pdb_content, plddt


def _predict_structure_api(sequence: str) -> Tuple[str, float]:
    """
    Predict structure using ESM Atlas API.
    
    This is a fallback for when local inference is not available.
    Requires network access.
    
    Returns:
        Tuple of (pdb_content, mean_plddt)
    """
    import httpx
    
    logger.info(f"Calling ESM Atlas API for sequence of length {len(sequence)}...")
    
    url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
    
    response = httpx.post(
        url,
        content=sequence,
        headers={"Content-Type": "text/plain"},
        timeout=300.0  # 5 minute timeout for long sequences
    )
    
    response.raise_for_status()
    pdb_content = response.text
    
    # API doesn't return pLDDT directly, extract from PDB B-factor column
    plddt_values = []
    for line in pdb_content.split('\n'):
        if line.startswith('ATOM'):
            try:
                bfactor = float(line[60:66].strip())
                plddt_values.append(bfactor)
            except (ValueError, IndexError):
                pass
    
    mean_plddt = sum(plddt_values) / len(plddt_values) if plddt_values else 0.0
    
    return pdb_content, mean_plddt


# =============================================================================
# Molecular Docking (AutoDock Vina)
# =============================================================================

def dock_molecule(
    protein_pdb_path: str,
    ligand_smiles: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    center: Optional[List[float]] = None,
    box_size: Optional[List[float]] = None,
    exhaustiveness: int = 8,
    n_poses: int = 9
) -> Dict[str, Any]:
    """
    Dock a small molecule to a protein target using AutoDock Vina.
    
    Performs molecular docking to predict how a ligand binds to a protein.
    Returns binding affinity (kcal/mol) and docked poses.
    
    Args:
        protein_pdb_path: Path to the protein structure PDB file
        ligand_smiles: SMILES string of the ligand molecule
        output_dir: Directory to save output files
        center: Docking box center [x, y, z] (default: protein center of mass)
        box_size: Docking box size [x, y, z] in Angstroms (default: [30, 30, 30])
        exhaustiveness: Search exhaustiveness (higher = more thorough but slower)
        n_poses: Number of binding poses to generate
    
    Returns:
        Dict containing:
        - success: bool
        - best_affinity: Best binding affinity in kcal/mol (more negative = stronger)
        - docked_pose_path: Path to best docked pose PDB
        - all_poses: List of all poses with affinities
        - error_message: Error description if failed
    
    Example:
        >>> result = dock_molecule(
        ...     protein_pdb_path="/workspace/output/protein.pdb",
        ...     ligand_smiles="CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
        ... )
        >>> if result["success"]:
        ...     print(f"Best affinity: {result['best_affinity']:.1f} kcal/mol")
    """
    start_time = time.time()
    
    # Validate inputs
    valid, pdb_path_or_error = _validate_pdb_file(protein_pdb_path)
    if not valid:
        return DockingResult(
            success=False,
            error_message=pdb_path_or_error,
            error_type="ValidationError",
            execution_time_seconds=time.time() - start_time,
        ).to_dict()
    
    protein_pdb_path = pdb_path_or_error
    
    valid, smiles_or_error = _validate_smiles(ligand_smiles)
    if not valid:
        return DockingResult(
            success=False,
            error_message=smiles_or_error,
            error_type="ValidationError",
            execution_time_seconds=time.time() - start_time,
        ).to_dict()
    
    ligand_smiles = smiles_or_error
    
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        from vina import Vina
        
        out_path = _ensure_output_dir(output_dir)
        
        # Generate unique prefix for output files
        prefix = _generate_output_name(f"{protein_pdb_path}_{ligand_smiles}", "")
        
        # Step 1: Prepare ligand - SMILES to 3D to PDBQT
        logger.info("Preparing ligand from SMILES...")
        ligand_pdbqt_path = _prepare_ligand(ligand_smiles, out_path, prefix)
        
        # Step 2: Prepare receptor - PDB to PDBQT
        logger.info("Preparing receptor...")
        receptor_pdbqt_path = _prepare_receptor(protein_pdb_path, out_path, prefix)
        
        # Step 3: Calculate docking box if not provided
        if center is None:
            center = _calculate_protein_center(protein_pdb_path)
        
        if box_size is None:
            box_size = [30.0, 30.0, 30.0]  # Default blind docking box
        
        # Step 4: Run Vina docking
        logger.info(f"Running docking (exhaustiveness={exhaustiveness})...")
        
        v = Vina(sf_name='vina')
        v.set_receptor(receptor_pdbqt_path)
        v.set_ligand_from_file(ligand_pdbqt_path)
        v.compute_vina_maps(center=center, box_size=box_size)
        
        v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
        
        # Get results
        energies = v.energies()
        poses = []
        for i, energy_row in enumerate(energies):
            poses.append({
                "pose_index": i + 1,
                "affinity": float(energy_row[0]),  # kcal/mol
                "rmsd_lb": float(energy_row[1]) if len(energy_row) > 1 else 0.0,
                "rmsd_ub": float(energy_row[2]) if len(energy_row) > 2 else 0.0,
            })
        
        # Save best pose
        best_pose_pdbqt = out_path / f"{prefix}_docked.pdbqt"
        v.write_poses(str(best_pose_pdbqt), n_poses=1, overwrite=True)
        
        # Convert best pose to PDB for easier visualization
        best_pose_pdb = out_path / f"{prefix}_docked.pdb"
        _convert_pdbqt_to_pdb(str(best_pose_pdbqt), str(best_pose_pdb))
        
        best_affinity = poses[0]["affinity"] if poses else 0.0
        
        logger.info(f"Docking complete. Best affinity: {best_affinity:.2f} kcal/mol")
        
        return DockingResult(
            success=True,
            best_affinity=best_affinity,
            docked_pose_path=str(best_pose_pdb),
            all_poses=poses,
            ligand_pdbqt_path=ligand_pdbqt_path,
            receptor_pdbqt_path=receptor_pdbqt_path,
            execution_time_seconds=time.time() - start_time,
        ).to_dict()
        
    except ImportError as e:
        return DockingResult(
            success=False,
            error_message=f"Required package not installed: {str(e)}. Run in kosmos-biolab container.",
            error_type="ImportError",
            execution_time_seconds=time.time() - start_time,
        ).to_dict()
        
    except Exception as e:
        logger.error(f"Docking failed: {e}")
        return DockingResult(
            success=False,
            error_message=str(e),
            error_type=type(e).__name__,
            execution_time_seconds=time.time() - start_time,
        ).to_dict()


def _prepare_ligand(smiles: str, output_dir: Path, prefix: str) -> str:
    """Convert SMILES to 3D PDBQT file for docking."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    
    # Convert SMILES to RDKit mol
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES: {smiles}")
    
    # Add hydrogens
    mol = Chem.AddHs(mol)
    
    # Generate 3D coordinates
    result = AllChem.EmbedMolecule(mol, randomSeed=42)
    if result == -1:
        raise ValueError("Failed to generate 3D coordinates for molecule")
    
    # Optimize geometry
    AllChem.MMFFOptimizeMolecule(mol)
    
    # Prepare for Vina using Meeko
    preparator = MoleculePreparation()
    mol_setup = preparator.prepare(mol)[0]
    
    # Write PDBQT
    pdbqt_path = output_dir / f"{prefix}_ligand.pdbqt"
    pdbqt_string = PDBQTWriterLegacy.write_string(mol_setup)[0]
    
    with open(pdbqt_path, 'w') as f:
        f.write(pdbqt_string)
    
    return str(pdbqt_path)


def _prepare_receptor(pdb_path: str, output_dir: Path, prefix: str) -> str:
    """Convert receptor PDB to PDBQT format."""
    # Use Open Babel for PDB to PDBQT conversion
    pdbqt_path = output_dir / f"{prefix}_receptor.pdbqt"
    
    try:
        # Try using obabel command line
        result = subprocess.run(
            ["obabel", pdb_path, "-O", str(pdbqt_path), "-xr"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"obabel failed: {result.stderr}")
            
    except FileNotFoundError:
        # Fall back to simple conversion (less accurate but works)
        logger.warning("obabel not found, using simple PDB->PDBQT conversion")
        _simple_pdb_to_pdbqt(pdb_path, str(pdbqt_path))
    
    return str(pdbqt_path)


def _simple_pdb_to_pdbqt(pdb_path: str, pdbqt_path: str):
    """Simple PDB to PDBQT conversion (adds default charges)."""
    with open(pdb_path, 'r') as f:
        lines = f.readlines()
    
    output_lines = []
    for line in lines:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            # Add Gasteiger charge (placeholder) and atom type
            atom_name = line[12:16].strip()
            element = atom_name[0] if atom_name else 'C'
            
            # Create PDBQT line with charge and atom type
            pdbqt_line = line[:66].ljust(66) + f"  0.000 {element:>2}\n"
            output_lines.append(pdbqt_line)
        elif line.startswith('END'):
            output_lines.append(line)
    
    with open(pdbqt_path, 'w') as f:
        f.writelines(output_lines)


def _calculate_protein_center(pdb_path: str) -> List[float]:
    """Calculate center of mass of protein from PDB file."""
    coords = []
    
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
                except ValueError:
                    continue
    
    if not coords:
        return [0.0, 0.0, 0.0]
    
    center = [
        sum(c[0] for c in coords) / len(coords),
        sum(c[1] for c in coords) / len(coords),
        sum(c[2] for c in coords) / len(coords),
    ]
    
    return center


def _convert_pdbqt_to_pdb(pdbqt_path: str, pdb_path: str):
    """Convert PDBQT back to PDB format."""
    with open(pdbqt_path, 'r') as f:
        lines = f.readlines()
    
    output_lines = []
    for line in lines:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            # Remove PDBQT-specific columns (charge and atom type)
            pdb_line = line[:66] + "\n"
            output_lines.append(pdb_line)
        elif line.startswith('END') or line.startswith('MODEL') or line.startswith('ENDMDL'):
            output_lines.append(line)
    
    with open(pdb_path, 'w') as f:
        f.writelines(output_lines)


# =============================================================================
# Molecular Dynamics Simulation (OpenMM)
# =============================================================================

def run_simulation(
    pdb_path: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    duration_ns: float = 1.0,
    temperature_k: float = 300.0,
    solvent: str = "implicit",
    minimize_only: bool = False,
    save_trajectory: bool = True,
    report_interval_ps: float = 10.0
) -> Dict[str, Any]:
    """
    Run molecular dynamics simulation using OpenMM.
    
    Simulates the dynamics of a protein or protein-ligand complex to assess
    structural stability. Returns RMSD analysis indicating if the structure
    remains stable over time.
    
    Args:
        pdb_path: Path to the structure PDB file
        output_dir: Directory to save output files
        duration_ns: Simulation duration in nanoseconds (default: 1.0)
        temperature_k: Simulation temperature in Kelvin (default: 300)
        solvent: Solvent model - "implicit" (faster) or "explicit" (more accurate)
        minimize_only: If True, only run energy minimization (faster)
        save_trajectory: If True, save trajectory file
        report_interval_ps: Interval for saving trajectory frames (picoseconds)
    
    Returns:
        Dict containing:
        - success: bool
        - stability_assessment: "stable", "unstable", or "inconclusive"
        - average_rmsd: Average RMSD in Angstroms
        - trajectory_path: Path to trajectory file (if saved)
        - final_structure_path: Path to minimized/equilibrated structure
        - energy_data: Dictionary of energy terms
        - error_message: Error description if failed
    
    Example:
        >>> result = run_simulation(
        ...     pdb_path="/workspace/output/protein.pdb",
        ...     duration_ns=1.0
        ... )
        >>> if result["success"]:
        ...     print(f"Stability: {result['stability_assessment']}")
        ...     print(f"RMSD: {result['average_rmsd']:.2f} Å")
    """
    start_time = time.time()
    
    # Validate input
    valid, pdb_path_or_error = _validate_pdb_file(pdb_path)
    if not valid:
        return SimulationResult(
            success=False,
            error_message=pdb_path_or_error,
            error_type="ValidationError",
            execution_time_seconds=time.time() - start_time,
        ).to_dict()
    
    pdb_path = pdb_path_or_error
    
    if duration_ns <= 0 or duration_ns > 100:
        return SimulationResult(
            success=False,
            error_message=f"Invalid duration: {duration_ns} ns. Must be between 0 and 100 ns.",
            error_type="ValidationError",
            execution_time_seconds=time.time() - start_time,
        ).to_dict()
    
    try:
        from openmm import app, unit, Platform, LangevinMiddleIntegrator
        from openmm.app import PDBFile, Simulation, Modeller
        from openmm.app import ForceField
        from pdbfixer import PDBFixer
        import numpy as np
        
        out_path = _ensure_output_dir(output_dir)
        prefix = _generate_output_name(pdb_path, "")
        
        # Step 1: Fix PDB structure (missing atoms, residues)
        logger.info("Fixing PDB structure...")
        fixed_pdb_path = out_path / f"{prefix}_fixed.pdb"
        _fix_pdb_structure(pdb_path, str(fixed_pdb_path))
        
        # Step 2: Load fixed structure
        pdb = PDBFile(str(fixed_pdb_path))
        modeller = Modeller(pdb.topology, pdb.positions)
        
        # Step 3: Set up force field and system
        logger.info(f"Setting up simulation with {solvent} solvent...")
        
        if solvent == "implicit":
            # Implicit solvent (OBC2) - faster
            forcefield = ForceField('amber14-all.xml', 'implicit/obc2.xml')
            system = forcefield.createSystem(
                modeller.topology,
                nonbondedMethod=app.NoCutoff,
                constraints=app.HBonds
            )
        else:
            # Explicit solvent - more accurate but slower
            forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
            modeller.addSolvent(forcefield, padding=1.0*unit.nanometers)
            system = forcefield.createSystem(
                modeller.topology,
                nonbondedMethod=app.PME,
                nonbondedCutoff=1.0*unit.nanometers,
                constraints=app.HBonds
            )
        
        # Step 4: Set up integrator
        integrator = LangevinMiddleIntegrator(
            temperature_k*unit.kelvin,
            1.0/unit.picoseconds,
            2.0*unit.femtoseconds
        )
        
        # Use fastest available platform
        try:
            platform = Platform.getPlatformByName('CUDA')
        except Exception:
            try:
                platform = Platform.getPlatformByName('OpenCL')
            except Exception:
                platform = Platform.getPlatformByName('CPU')
        
        logger.info(f"Using platform: {platform.getName()}")
        
        # Step 5: Create simulation
        simulation = Simulation(modeller.topology, system, integrator, platform)
        simulation.context.setPositions(modeller.positions)
        
        # Step 6: Energy minimization
        logger.info("Running energy minimization...")
        initial_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
        simulation.minimizeEnergy(maxIterations=1000)
        minimized_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
        
        logger.info(f"Energy: {initial_energy} -> {minimized_energy}")
        
        # Save minimized structure
        minimized_pdb_path = out_path / f"{prefix}_minimized.pdb"
        positions = simulation.context.getState(getPositions=True).getPositions()
        with open(minimized_pdb_path, 'w') as f:
            PDBFile.writeFile(simulation.topology, positions, f)
        
        if minimize_only:
            return SimulationResult(
                success=True,
                stability_assessment="minimized_only",
                average_rmsd=0.0,
                final_structure_path=str(minimized_pdb_path),
                energy_data={
                    "initial_kj_mol": initial_energy.value_in_unit(unit.kilojoules_per_mole),
                    "minimized_kj_mol": minimized_energy.value_in_unit(unit.kilojoules_per_mole),
                },
                execution_time_seconds=time.time() - start_time,
            ).to_dict()
        
        # Step 7: Equilibration
        logger.info("Running equilibration...")
        simulation.context.setVelocitiesToTemperature(temperature_k*unit.kelvin)
        simulation.step(5000)  # 10 ps equilibration
        
        # Step 8: Production run
        n_steps = int(duration_ns * 500000)  # 2 fs timestep -> 500000 steps/ns
        report_steps = int(report_interval_ps * 500)  # Convert ps to steps
        
        logger.info(f"Running production simulation: {duration_ns} ns ({n_steps} steps)...")
        
        # Track RMSD
        reference_positions = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
        rmsd_values = []
        
        # Set up trajectory output
        trajectory_path = None
        if save_trajectory:
            trajectory_path = out_path / f"{prefix}_trajectory.pdb"
            simulation.reporters.append(
                app.PDBReporter(str(trajectory_path), report_steps)
            )
        
        # Run simulation in chunks and calculate RMSD
        chunk_size = min(report_steps, n_steps)
        for i in range(0, n_steps, chunk_size):
            steps_to_run = min(chunk_size, n_steps - i)
            simulation.step(steps_to_run)
            
            # Calculate RMSD
            current_positions = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
            rmsd = _calculate_rmsd(reference_positions, current_positions)
            rmsd_values.append(rmsd)
        
        # Step 9: Save final structure
        final_pdb_path = out_path / f"{prefix}_final.pdb"
        positions = simulation.context.getState(getPositions=True).getPositions()
        with open(final_pdb_path, 'w') as f:
            PDBFile.writeFile(simulation.topology, positions, f)
        
        # Step 10: Analyze results
        avg_rmsd = float(np.mean(rmsd_values))
        max_rmsd = float(np.max(rmsd_values))
        final_rmsd = float(rmsd_values[-1]) if rmsd_values else 0.0
        
        # Assess stability based on RMSD
        if avg_rmsd < 2.0 and max_rmsd < 3.0:
            stability = "stable"
        elif avg_rmsd < 4.0 and max_rmsd < 6.0:
            stability = "moderately_stable"
        elif avg_rmsd > 6.0 or max_rmsd > 10.0:
            stability = "unstable"
        else:
            stability = "inconclusive"
        
        final_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
        
        logger.info(f"Simulation complete. Stability: {stability}, Avg RMSD: {avg_rmsd:.2f} Å")
        
        return SimulationResult(
            success=True,
            stability_assessment=stability,
            average_rmsd=avg_rmsd,
            max_rmsd=max_rmsd,
            final_rmsd=final_rmsd,
            trajectory_path=str(trajectory_path) if trajectory_path else None,
            final_structure_path=str(final_pdb_path),
            energy_data={
                "initial_kj_mol": initial_energy.value_in_unit(unit.kilojoules_per_mole),
                "minimized_kj_mol": minimized_energy.value_in_unit(unit.kilojoules_per_mole),
                "final_kj_mol": final_energy.value_in_unit(unit.kilojoules_per_mole),
            },
            simulation_time_ns=duration_ns,
            execution_time_seconds=time.time() - start_time,
        ).to_dict()
        
    except ImportError as e:
        return SimulationResult(
            success=False,
            error_message=f"Required package not installed: {str(e)}. Run in kosmos-biolab container.",
            error_type="ImportError",
            execution_time_seconds=time.time() - start_time,
        ).to_dict()
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        return SimulationResult(
            success=False,
            error_message=str(e),
            error_type=type(e).__name__,
            execution_time_seconds=time.time() - start_time,
        ).to_dict()


def _fix_pdb_structure(input_path: str, output_path: str):
    """Fix PDB structure using PDBFixer."""
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile
    
    fixer = PDBFixer(filename=input_path)
    
    # Find and add missing residues
    fixer.findMissingResidues()
    
    # Find and add missing atoms
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    
    # Add missing hydrogens
    fixer.addMissingHydrogens(7.0)  # pH 7.0
    
    # Write fixed structure
    with open(output_path, 'w') as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)


def _calculate_rmsd(reference: 'np.ndarray', current: 'np.ndarray') -> float:
    """Calculate RMSD between two sets of positions."""
    import numpy as np
    
    # Convert to numpy arrays in Angstroms
    ref = np.array([[p.value_in_unit(unit.angstroms) for p in pos] 
                    for pos in reference]) if hasattr(reference[0][0], 'value_in_unit') else np.array(reference) * 10
    cur = np.array([[p.value_in_unit(unit.angstroms) for p in pos] 
                    for pos in current]) if hasattr(current[0][0], 'value_in_unit') else np.array(current) * 10
    
    # Calculate RMSD
    diff = ref - cur
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    
    return float(rmsd)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Main functions
    "predict_structure",
    "dock_molecule",
    "run_simulation",
    # Result types
    "BioLabResult",
    "StructurePredictionResult",
    "DockingResult",
    "SimulationResult",
]
