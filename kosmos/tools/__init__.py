"""
Kosmos Tools Module - Experimental Instruments for AI Scientist.

This module provides high-level tool functions that LLM agents can invoke
to perform computational experiments. Tools are designed to:

1. Have simple, well-documented interfaces
2. Handle file I/O automatically (no path management by LLM)
3. Return structured results with success/failure status
4. Provide descriptive error messages for recovery

Available Tool Categories:
- BioLab: Computational biology (structure prediction, docking, MD)
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ToolCategory(str, Enum):
    """Categories of available tools."""
    BIOLAB = "biolab"
    DATA_ANALYSIS = "data_analysis"
    LITERATURE = "literature"
    VISUALIZATION = "visualization"


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


@dataclass
class ToolDefinition:
    """
    Complete definition of a tool for LLM consumption.
    
    This schema is used to generate tool descriptions for the LLM
    and validate tool invocations.
    """
    name: str
    description: str
    category: ToolCategory
    parameters: List[ToolParameter]
    returns: str
    example_usage: Optional[str] = None
    requires_docker_image: Optional[str] = None
    estimated_runtime_seconds: Optional[int] = None
    memory_requirements_gb: Optional[float] = None
    
    def to_schema(self) -> Dict[str, Any]:
        """Convert to JSON schema format for LLM tool calling."""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description,
            }
            if param.default is not None:
                properties[param.name]["default"] = param.default
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            }
        }
    
    def to_prompt_description(self) -> str:
        """Generate human-readable description for prompt injection."""
        params_str = ", ".join(
            f"{p.name}: {p.type}" + (f" = {p.default}" if p.default else "")
            for p in self.parameters
        )
        
        desc = f"**{self.name}**({params_str}) -> {self.returns}\n"
        desc += f"  {self.description}\n"
        
        if self.estimated_runtime_seconds:
            desc += f"  Runtime: ~{self.estimated_runtime_seconds}s\n"
        if self.memory_requirements_gb:
            desc += f"  Memory: ~{self.memory_requirements_gb}GB\n"
        
        return desc


@dataclass
class ToolResult:
    """
    Standardized result from tool execution.
    
    All tools return this structure to ensure consistent handling
    by the LLM agent.
    """
    success: bool
    result: Any = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    output_files: List[str] = field(default_factory=list)
    execution_time_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "result": self.result,
            "error_message": self.error_message,
            "error_type": self.error_type,
            "output_files": self.output_files,
            "execution_time_seconds": self.execution_time_seconds,
            "metadata": self.metadata,
        }
    
    def to_llm_response(self) -> str:
        """Format result as human-readable string for LLM consumption."""
        if self.success:
            result_str = f"SUCCESS: {self.result}"
            if self.output_files:
                result_str += f"\nOutput files: {', '.join(self.output_files)}"
            return result_str
        else:
            return f"ERROR ({self.error_type}): {self.error_message}"


# =============================================================================
# BioLab Tool Definitions
# =============================================================================

# =============================================================================
# RNAi Tool Definitions
# =============================================================================

RNAI_TOOLS: Dict[str, ToolDefinition] = {
    "find_essential_genes": ToolDefinition(
        name="find_essential_genes",
        description=(
            "Find essential genes in a target pest organism by downloading "
            "transcriptome data and identifying genes required for survival via "
            "homology mapping against Drosophila Essential Genes database. "
            "Returns gene sequences (AGCT format) suitable for RNAi targeting."
        ),
        category=ToolCategory.BIOLAB,
        parameters=[
            ToolParameter(
                name="organism_name",
                type="string",
                description="Scientific or common name of pest (e.g., 'Helicoverpa zea', 'corn earworm')"
            ),
            ToolParameter(
                name="data_source",
                type="string",
                description="Database to query: 'insectbase', 'ncbi', 'ensembl', or 'auto'",
                required=False,
                default="auto"
            ),
            ToolParameter(
                name="essentiality_criteria",
                type="object",
                description="Optional criteria: min_essentiality_score, gene_functions, exclude_hypothetical",
                required=False,
                default=None
            ),
            ToolParameter(
                name="output_dir",
                type="string",
                description="Directory to save downloaded data and results",
                required=False,
                default="/workspace/output"
            ),
        ],
        returns="Dict with essential_genes list, organism_info, and metadata",
        example_usage='find_essential_genes(organism_name="Helicoverpa zea")',
        requires_docker_image="kosmos-biolab:latest",
        estimated_runtime_seconds=300,
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
            ToolParameter(
                name="min_gc_content",
                type="number",
                description="Minimum GC content (default: 0.30)",
                required=False,
                default=0.30
            ),
            ToolParameter(
                name="max_gc_content",
                type="number",
                description="Maximum GC content (default: 0.52)",
                required=False,
                default=0.52
            ),
            ToolParameter(
                name="output_dir",
                type="string",
                description="Directory to save analysis results",
                required=False,
                default="/workspace/output"
            ),
        ],
        returns="Dict with candidates list (21-mer sequences with scores), total generated, filtered count",
        example_usage='generate_sirna_candidates(gene_sequence="AGCT..." * 1000)',
        requires_docker_image="kosmos-biolab:latest",
        estimated_runtime_seconds=5,
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
            ToolParameter(
                name="match_threshold",
                type="integer",
                description="Minimum consecutive matching nucleotides to trigger rejection (default: 21)",
                required=False,
                default=21
            ),
            ToolParameter(
                name="output_dir",
                type="string",
                description="Directory to save BLAST results",
                required=False,
                default="/workspace/output"
            ),
        ],
        returns="Dict with approved/rejected status, matches found, and BLAST results",
        example_usage='check_off_target_risk(candidate_sequence="AGCTAGCTAGCTAGCTAGCTA")',
        requires_docker_image="kosmos-biolab:latest",
        estimated_runtime_seconds=60,
        memory_requirements_gb=2.0,
    ),
}


# =============================================================================
# BioLab Tool Definitions
# =============================================================================

BIOLAB_TOOLS: Dict[str, ToolDefinition] = {
    "predict_structure": ToolDefinition(
        name="predict_structure",
        description=(
            "Predict the 3D structure of a protein from its amino acid sequence "
            "using ESMFold. Returns the path to a PDB file containing the predicted "
            "structure with confidence scores (pLDDT)."
        ),
        category=ToolCategory.BIOLAB,
        parameters=[
            ToolParameter(
                name="sequence",
                type="string",
                description="Amino acid sequence in single-letter code (e.g., 'MKFLILLFNILCLFPVLAADNHGVGPQGAS...')"
            ),
            ToolParameter(
                name="output_name",
                type="string",
                description="Optional name for output file (default: auto-generated from sequence hash)",
                required=False,
                default=None
            ),
        ],
        returns="Path to the generated PDB file with predicted 3D structure",
        example_usage='predict_structure(sequence="MKFLILLFNILCLFPVLAADNHGVGPQGAS")',
        requires_docker_image="kosmos-biolab:latest",
        estimated_runtime_seconds=60,
        memory_requirements_gb=8.0,
    ),
    
    "dock_molecule": ToolDefinition(
        name="dock_molecule",
        description=(
            "Dock a small molecule (ligand) to a protein target using AutoDock Vina. "
            "Performs blind docking to find the best binding pose and returns the "
            "binding affinity in kcal/mol (more negative = stronger binding)."
        ),
        category=ToolCategory.BIOLAB,
        parameters=[
            ToolParameter(
                name="protein_pdb_path",
                type="string",
                description="Path to the protein structure PDB file"
            ),
            ToolParameter(
                name="ligand_smiles",
                type="string",
                description="SMILES string of the ligand molecule (e.g., 'CC(=O)Oc1ccccc1C(=O)O' for aspirin)"
            ),
            ToolParameter(
                name="center",
                type="array",
                description="Optional docking box center [x, y, z]. If not provided, uses protein center.",
                required=False,
                default=None
            ),
            ToolParameter(
                name="box_size",
                type="array",
                description="Optional docking box size [x, y, z] in Angstroms. Default: [30, 30, 30]",
                required=False,
                default=[30, 30, 30]
            ),
        ],
        returns="Dict with best_affinity (kcal/mol), docked_pose_path, and all poses",
        example_usage='dock_molecule(protein_pdb_path="/workspace/output/protein.pdb", ligand_smiles="CC(=O)Oc1ccccc1C(=O)O")',
        requires_docker_image="kosmos-biolab:latest",
        estimated_runtime_seconds=120,
        memory_requirements_gb=2.0,
    ),
    
    "run_simulation": ToolDefinition(
        name="run_simulation",
        description=(
            "Run a molecular dynamics simulation using OpenMM to assess the stability "
            "of a protein or protein-ligand complex. Returns RMSD analysis to determine "
            "if the structure remains stable over time."
        ),
        category=ToolCategory.BIOLAB,
        parameters=[
            ToolParameter(
                name="pdb_path",
                type="string",
                description="Path to the structure PDB file (protein or protein-ligand complex)"
            ),
            ToolParameter(
                name="duration_ns",
                type="number",
                description="Simulation duration in nanoseconds (default: 1.0)",
                required=False,
                default=1.0
            ),
            ToolParameter(
                name="temperature_k",
                type="number",
                description="Simulation temperature in Kelvin (default: 300)",
                required=False,
                default=300
            ),
            ToolParameter(
                name="solvent",
                type="string",
                description="Solvent model: 'implicit' (faster) or 'explicit' (more accurate)",
                required=False,
                default="implicit"
            ),
        ],
        returns="Dict with stability assessment, average RMSD, trajectory path, and energy data",
        example_usage='run_simulation(pdb_path="/workspace/output/complex.pdb", duration_ns=1.0)',
        requires_docker_image="kosmos-biolab:latest",
        estimated_runtime_seconds=600,
        memory_requirements_gb=4.0,
    ),
}


# =============================================================================
# Tool Registry
# =============================================================================

class ToolRegistry:
    """
    Central registry for all available tools.
    
    Provides:
    - Tool discovery and listing
    - Schema generation for LLM tool calling
    - Prompt generation for capability descriptions
    """
    
    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._handlers: Dict[str, Callable] = {}
        
        # Register built-in tools
        for name, tool_def in BIOLAB_TOOLS.items():
            self.register(tool_def)
        
        # Register RNAi tools
        for name, tool_def in RNAI_TOOLS.items():
            self.register(tool_def)
    
    def register(self, tool: ToolDefinition, handler: Optional[Callable] = None):
        """Register a tool definition and optional handler."""
        self._tools[tool.name] = tool
        if handler:
            self._handlers[tool.name] = handler
        logger.debug(f"Registered tool: {tool.name}")
    
    def get(self, name: str) -> Optional[ToolDefinition]:
        """Get tool definition by name."""
        return self._tools.get(name)
    
    def list_tools(self, category: Optional[ToolCategory] = None) -> List[ToolDefinition]:
        """List all tools, optionally filtered by category."""
        tools = list(self._tools.values())
        if category:
            tools = [t for t in tools if t.category == category]
        return tools
    
    def get_schemas(self, category: Optional[ToolCategory] = None) -> List[Dict[str, Any]]:
        """Get JSON schemas for all tools (for LLM tool calling)."""
        return [t.to_schema() for t in self.list_tools(category)]
    
    def get_prompt_description(self, category: Optional[ToolCategory] = None) -> str:
        """Generate prompt text describing available tools."""
        tools = self.list_tools(category)
        if not tools:
            return "No tools available."
        
        lines = ["## Available Tools\n"]
        
        # Group by category
        by_category: Dict[ToolCategory, List[ToolDefinition]] = {}
        for tool in tools:
            if tool.category not in by_category:
                by_category[tool.category] = []
            by_category[tool.category].append(tool)
        
        for cat, cat_tools in by_category.items():
            lines.append(f"### {cat.value.replace('_', ' ').title()}\n")
            for tool in cat_tools:
                lines.append(tool.to_prompt_description())
                lines.append("")
        
        return "\n".join(lines)


# Global registry instance
_tool_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
    return _tool_registry


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Types
    "ToolCategory",
    "ToolParameter",
    "ToolDefinition",
    "ToolResult",
    # Registry
    "ToolRegistry",
    "get_tool_registry",
    # Tool definitions
    "BIOLAB_TOOLS",
    "RNAI_TOOLS",
]
