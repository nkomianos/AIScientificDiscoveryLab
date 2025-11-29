"""
Domain classification and routing models.

Defines data structures for domain detection, classification, expertise assessment,
and cross-domain synthesis.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from kosmos.config import _DEFAULT_CLAUDE_SONNET_MODEL

class ScientificDomain(str, Enum):
    """Scientific research domains supported by Kosmos."""

    BIOLOGY = "biology"
    NEUROSCIENCE = "neuroscience"
    MATERIALS = "materials"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    ASTRONOMY = "astronomy"
    SOCIAL_SCIENCE = "social_science"
    GENERAL = "general"  # For domain-agnostic research


class DomainConfidence(str, Enum):
    """Confidence levels for domain classification."""

    VERY_HIGH = "very_high"  # > 0.9
    HIGH = "high"             # 0.7 - 0.9
    MEDIUM = "medium"         # 0.5 - 0.7
    LOW = "low"               # 0.3 - 0.5
    VERY_LOW = "very_low"     # < 0.3


class DomainClassification(BaseModel):
    """Result of classifying a research question or hypothesis to a domain."""

    # Primary classification
    primary_domain: ScientificDomain = Field(
        description="The primary scientific domain identified"
    )
    confidence: DomainConfidence = Field(
        description="Confidence level in the primary classification"
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0,
        description="Numeric confidence score (0-1)"
    )

    # Secondary domains (for multi-domain questions)
    secondary_domains: List[ScientificDomain] = Field(
        default_factory=list,
        description="Additional relevant domains (for cross-domain research)"
    )
    domain_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Confidence scores for all considered domains"
    )

    # Classification details
    key_terms: List[str] = Field(
        default_factory=list,
        description="Key terms that influenced classification"
    )
    classification_reasoning: Optional[str] = Field(
        default=None,
        description="Explanation of why this domain was chosen"
    )

    # Multi-domain detection
    is_multi_domain: bool = Field(
        default=False,
        description="Whether this question spans multiple domains"
    )
    cross_domain_rationale: Optional[str] = Field(
        default=None,
        description="Explanation if multi-domain research is detected"
    )

    # Metadata
    classified_at: datetime = Field(default_factory=datetime.now)
    classifier_model: str = Field(default=_DEFAULT_CLAUDE_SONNET_MODEL)

    def to_domain_list(self) -> List[ScientificDomain]:
        """Get all relevant domains (primary + secondary) as a list."""
        domains = [self.primary_domain]
        domains.extend(self.secondary_domains)
        return list(set(domains))  # Remove duplicates

    def requires_cross_domain_synthesis(self) -> bool:
        """Check if cross-domain synthesis is needed."""
        return self.is_multi_domain and len(self.secondary_domains) > 0


class DomainExpertise(BaseModel):
    """Assessment of expertise level for a specific domain."""

    domain: ScientificDomain
    expertise_level: str = Field(
        description="Expertise level: beginner, intermediate, advanced, expert"
    )
    expertise_score: float = Field(
        ge=0.0, le=1.0,
        description="Numeric expertise score (0-1)"
    )

    # Capabilities
    available_tools: List[str] = Field(
        default_factory=list,
        description="Domain-specific tools/APIs available"
    )
    available_templates: List[str] = Field(
        default_factory=list,
        description="Experiment templates available for this domain"
    )
    knowledge_base_coverage: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Coverage of domain knowledge base (0-1)"
    )

    # Limitations
    known_limitations: List[str] = Field(
        default_factory=list,
        description="Known limitations or gaps in this domain"
    )
    recommended_human_expertise: Optional[str] = Field(
        default=None,
        description="Recommended human expertise if complex questions arise"
    )

    # Evidence
    successful_experiments: int = Field(
        default=0, ge=0,
        description="Number of successful experiments in this domain"
    )
    total_experiments: int = Field(
        default=0, ge=0,
        description="Total experiments attempted in this domain"
    )

    def success_rate(self) -> float:
        """Calculate success rate in this domain."""
        if self.total_experiments == 0:
            return 0.0
        return self.successful_experiments / self.total_experiments

    def has_sufficient_tools(self, min_tools: int = 1) -> bool:
        """Check if sufficient tools are available."""
        return len(self.available_tools) >= min_tools

    def has_templates(self) -> bool:
        """Check if any templates are available."""
        return len(self.available_templates) > 0


class DomainRoute(BaseModel):
    """Routing decision for a research question to domain-specific agents and tools."""

    # Classification
    classification: DomainClassification

    # Selected route
    selected_domains: List[ScientificDomain] = Field(
        description="Domains selected for this research"
    )
    routing_strategy: str = Field(
        description="Routing strategy: single_domain, parallel_multi_domain, sequential_multi_domain"
    )

    # Agent selection
    assigned_agents: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Agents assigned per domain (domain -> agent_types)"
    )

    # Tool selection
    required_tools: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Required tools per domain (domain -> tool_names)"
    )
    recommended_templates: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Recommended experiment templates per domain"
    )

    # Cross-domain synthesis
    synthesis_required: bool = Field(
        default=False,
        description="Whether cross-domain synthesis is needed"
    )
    synthesis_strategy: Optional[str] = Field(
        default=None,
        description="Strategy for synthesizing cross-domain results"
    )

    # Routing metadata
    routed_at: datetime = Field(default_factory=datetime.now)
    routing_reasoning: Optional[str] = Field(
        default=None,
        description="Explanation of routing decision"
    )

    def get_all_tools(self) -> List[str]:
        """Get all required tools across all domains."""
        tools = []
        for tool_list in self.required_tools.values():
            tools.extend(tool_list)
        return list(set(tools))  # Remove duplicates

    def get_all_templates(self) -> List[str]:
        """Get all recommended templates across all domains."""
        templates = []
        for template_list in self.recommended_templates.values():
            templates.extend(template_list)
        return list(set(templates))

    def is_single_domain(self) -> bool:
        """Check if this is single-domain research."""
        return len(self.selected_domains) == 1

    def get_primary_domain(self) -> ScientificDomain:
        """Get the primary domain for routing."""
        return self.classification.primary_domain


class CrossDomainMapping(BaseModel):
    """Mapping between concepts across different scientific domains."""

    source_domain: ScientificDomain
    target_domain: ScientificDomain

    # Concept mapping
    source_concept: str = Field(description="Concept in source domain")
    target_concept: str = Field(description="Equivalent/related concept in target domain")

    mapping_type: str = Field(
        description="Type of mapping: equivalent, analogous, related, none"
    )
    similarity_score: float = Field(
        ge=0.0, le=1.0,
        description="Similarity score between concepts (0-1)"
    )

    # Mapping details
    mapping_rationale: Optional[str] = Field(
        default=None,
        description="Explanation of why these concepts are related"
    )
    example_usage: Optional[str] = Field(
        default=None,
        description="Example of how this mapping could be used"
    )

    # Bidirectional support
    is_bidirectional: bool = Field(
        default=False,
        description="Whether mapping works in both directions"
    )

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    evidence_papers: List[str] = Field(
        default_factory=list,
        description="Papers that support this cross-domain connection"
    )


class DomainOntology(BaseModel):
    """Ontology definition for a scientific domain."""

    domain: ScientificDomain

    # Core concepts
    concepts: Dict[str, Any] = Field(
        default_factory=dict,
        description="Core concepts in this domain with definitions"
    )
    relationships: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Relationships between concepts (subject, predicate, object)"
    )

    # Hierarchies
    concept_hierarchy: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Hierarchical organization of concepts (parent -> children)"
    )

    # Methodologies
    standard_methods: List[str] = Field(
        default_factory=list,
        description="Standard research methods in this domain"
    )
    data_types: List[str] = Field(
        default_factory=list,
        description="Common data types in this domain"
    )
    analysis_techniques: List[str] = Field(
        default_factory=list,
        description="Common analysis techniques"
    )

    # Validation rules
    validation_rules: List[str] = Field(
        default_factory=list,
        description="Domain-specific validation rules"
    )

    # External references
    external_databases: List[str] = Field(
        default_factory=list,
        description="External databases/APIs for this domain"
    )

    # Metadata
    version: str = Field(default="1.0")
    last_updated: datetime = Field(default_factory=datetime.now)
    sources: List[str] = Field(
        default_factory=list,
        description="Sources used to build this ontology"
    )

    def get_concept(self, concept_name: str) -> Optional[Any]:
        """Get concept definition by name."""
        return self.concepts.get(concept_name)

    def get_children(self, concept_name: str) -> List[str]:
        """Get child concepts of a given concept."""
        return self.concept_hierarchy.get(concept_name, [])

    def has_method(self, method_name: str) -> bool:
        """Check if a method is in the standard methods."""
        return method_name in self.standard_methods


class DomainCapability(BaseModel):
    """Capabilities available for a specific domain."""

    domain: ScientificDomain

    # API integrations
    api_clients: List[str] = Field(
        default_factory=list,
        description="Available API client names"
    )
    api_status: Dict[str, str] = Field(
        default_factory=dict,
        description="Status of each API (api_name -> status)"
    )

    # Analysis capabilities
    analysis_modules: List[str] = Field(
        default_factory=list,
        description="Available analysis module names"
    )

    # Templates
    experiment_templates: List[str] = Field(
        default_factory=list,
        description="Available experiment template names"
    )

    # Knowledge base
    has_ontology: bool = Field(default=False)
    ontology_coverage: float = Field(default=0.0, ge=0.0, le=1.0)

    # Computational resources
    requires_gpu: bool = Field(default=False)
    estimated_memory_gb: Optional[float] = Field(default=None)

    # Metadata
    last_checked: datetime = Field(default_factory=datetime.now)

    def is_fully_operational(self) -> bool:
        """Check if all capabilities are operational."""
        return (
            len(self.api_clients) > 0 and
            len(self.analysis_modules) > 0 and
            len(self.experiment_templates) > 0 and
            self.has_ontology
        )

    def get_operational_apis(self) -> List[str]:
        """Get list of operational APIs."""
        return [
            api for api, status in self.api_status.items()
            if status in ["operational", "available"]
        ]
