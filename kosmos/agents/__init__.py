"""
Agents module for Kosmos.

Provides specialized agents for different research tasks:
- DataAnalystAgent: Executes data analysis tasks
- LiteratureAnalyzer: Searches and synthesizes literature
- HypothesisGenerator: Generates testable hypotheses
- ExperimentDesigner: Designs experiments
- ResearchDirector: Strategic planning

Gap 3 Enhancement:
- SkillLoader: Loads domain-specific scientific skills for agent prompts
"""

from .skill_loader import SkillLoader

__all__ = [
    "SkillLoader",
]
