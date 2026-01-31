"""
Biology domain module - metabolomics, genomics, and multi-modal integration.

This module provides:
- API clients for biological databases (KEGG, UniProt, PDB, etc.)
- Analysis tools for metabolomics and genomics
- BioLab tools for computational drug discovery (structure prediction, docking, MD)
"""

from kosmos.domains.biology.apis import (
    KEGGClient,
    GWASCatalogClient,
    GTExClient,
    ENCODEClient,
    dbSNPClient,
    EnsemblClient,
    HMDBClient,
    MetaboLightsClient,
    UniProtClient,
    PDBClient,
)

# BioLab Tools - Computational Drug Discovery Pipeline
from kosmos.tools.bio_lab import (
    predict_structure,
    dock_molecule,
    run_simulation,
    StructurePredictionResult,
    DockingResult,
    SimulationResult,
)

from kosmos.domains.biology.metabolomics import (
    MetabolomicsAnalyzer,
    MetabolomicsResult,
    PathwayPattern,
    PathwayComparison,
    MetaboliteCategory,
    MetaboliteType,
)

from kosmos.domains.biology.genomics import (
    GenomicsAnalyzer,
    GenomicsResult,
    CompositeScore,
    MechanismRanking,
    EvidenceLevel,
    EffectDirection,
)

from kosmos.domains.biology.ontology import (
    BiologyOntology,
    BiologicalConcept,
    BiologicalRelation,
    BiologicalRelationType,
)

__all__ = [
    # API Clients
    'KEGGClient',
    'GWASCatalogClient',
    'GTExClient',
    'ENCODEClient',
    'dbSNPClient',
    'EnsemblClient',
    'HMDBClient',
    'MetaboLightsClient',
    'UniProtClient',
    'PDBClient',

    # Metabolomics
    'MetabolomicsAnalyzer',
    'MetabolomicsResult',
    'PathwayPattern',
    'PathwayComparison',
    'MetaboliteCategory',
    'MetaboliteType',

    # Genomics
    'GenomicsAnalyzer',
    'GenomicsResult',
    'CompositeScore',
    'MechanismRanking',
    'EvidenceLevel',
    'EffectDirection',

    # Ontology
    'BiologyOntology',
    'BiologicalConcept',
    'BiologicalRelation',
    'BiologicalRelationType',

    # BioLab Tools - Computational Drug Discovery
    'predict_structure',
    'dock_molecule',
    'run_simulation',
    'StructurePredictionResult',
    'DockingResult',
    'SimulationResult',
]
