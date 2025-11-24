"""
Plan Creator Agent for Kosmos.

Generates strategic research plans with exploration/exploitation balance.

Key Innovation: Adaptive strategy based on research cycle progress.

Exploration/Exploitation Ratios by Cycle:
- Early (cycles 1-7): 70% exploration (find new directions)
- Middle (cycles 8-14): 50% balanced
- Late (cycles 15-20): 30% exploration, 70% exploitation (deepen findings)

Performance Target: Generate plans with ~80% approval rate
"""

import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """Container for a research task."""
    task_id: int
    task_type: str  # data_analysis, literature_review, hypothesis_generation
    description: str
    expected_output: str
    required_skills: List[str]
    exploration: bool
    target_hypotheses: Optional[List[str]] = None
    priority: int = 1

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'id': self.task_id,
            'type': self.task_type,
            'description': self.description,
            'expected_output': self.expected_output,
            'required_skills': self.required_skills,
            'exploration': self.exploration,
            'target_hypotheses': self.target_hypotheses or [],
            'priority': self.priority
        }


@dataclass
class ResearchPlan:
    """Container for a research plan (10 tasks + rationale)."""
    cycle: int
    tasks: List[Task]
    rationale: str
    exploration_ratio: float

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'cycle': self.cycle,
            'tasks': [t.to_dict() for t in self.tasks],
            'rationale': self.rationale,
            'exploration_ratio': self.exploration_ratio
        }


class PlanCreatorAgent:
    """
    Strategic research planning agent.

    Generates 10 tasks per cycle that advance the research objective
    while balancing exploration of new directions with exploitation
    of promising findings.

    Design Philosophy:
    - Early cycles: Broad exploration to map the problem space
    - Middle cycles: Balance between deepening and branching
    - Late cycles: Focus on validating and extending key discoveries
    """

    def __init__(
        self,
        anthropic_client=None,
        model: str = "claude-3-5-sonnet-20241022",
        default_num_tasks: int = 10
    ):
        """
        Initialize Plan Creator Agent.

        Args:
            anthropic_client: Anthropic client for LLM-based planning
            model: Model to use for plan generation
            default_num_tasks: Default number of tasks per cycle
        """
        self.client = anthropic_client
        self.model = model
        self.default_num_tasks = default_num_tasks

    def _get_exploration_ratio(self, cycle: int) -> float:
        """
        Determine exploration vs. exploitation ratio.

        Args:
            cycle: Current research cycle

        Returns:
            Exploration ratio (0.0-1.0)
        """
        if cycle <= 7:
            return 0.70  # Early: explore widely
        elif cycle <= 14:
            return 0.50  # Middle: balanced
        else:
            return 0.30  # Late: exploit findings

    async def create_plan(
        self,
        research_objective: str,
        context: Dict,
        num_tasks: Optional[int] = None
    ) -> ResearchPlan:
        """
        Generate strategic research plan.

        Args:
            research_objective: Overall research goal
            context: Context from State Manager (findings, hypotheses, etc.)
            num_tasks: Number of tasks to generate (default: self.default_num_tasks)

        Returns:
            ResearchPlan with tasks and rationale
        """
        if num_tasks is None:
            num_tasks = self.default_num_tasks

        cycle = context.get('cycle', 1)
        exploration_ratio = self._get_exploration_ratio(cycle)

        # If no LLM client, use mock planning
        if self.client is None:
            return self._create_mock_plan(
                cycle, research_objective, context, num_tasks, exploration_ratio
            )

        # Build prompt
        prompt = self._build_planning_prompt(
            research_objective, context, num_tasks, exploration_ratio
        )

        try:
            # Query LLM
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7  # Allow creativity
            )

            # Parse response
            plan_data = self._parse_plan_response(response.content[0].text)

            # Validate and create ResearchPlan
            tasks = []
            for i, task_data in enumerate(plan_data.get('tasks', [])[:num_tasks], 1):
                task = Task(
                    task_id=i,
                    task_type=task_data.get('type', 'data_analysis'),
                    description=task_data.get('description', ''),
                    expected_output=task_data.get('expected_output', ''),
                    required_skills=task_data.get('required_skills', []),
                    exploration=task_data.get('exploration', False),
                    target_hypotheses=task_data.get('target_hypotheses'),
                    priority=task_data.get('priority', 1)
                )
                tasks.append(task)

            # Ensure we have enough tasks
            while len(tasks) < num_tasks:
                tasks.append(self._create_generic_task(len(tasks) + 1))

            return ResearchPlan(
                cycle=cycle,
                tasks=tasks,
                rationale=plan_data.get('rationale', ''),
                exploration_ratio=exploration_ratio
            )

        except Exception as e:
            logger.error(f"Plan generation failed: {e}, using mock plan")
            return self._create_mock_plan(
                cycle, research_objective, context, num_tasks, exploration_ratio
            )

    def _build_planning_prompt(
        self,
        research_objective: str,
        context: Dict,
        num_tasks: int,
        exploration_ratio: float
    ) -> str:
        """Build prompt for strategic planning."""
        cycle = context.get('cycle', 1)
        findings_count = len(context.get('recent_findings', []))
        unsupported_hyps = len(context.get('unsupported_hypotheses', []))

        # Format recent findings
        findings_summary = ""
        for finding in context.get('recent_findings', [])[:5]:
            findings_summary += f"- {finding.get('summary', 'N/A')[:100]}\n"

        # Format unsupported hypotheses
        hypotheses_summary = ""
        for hyp in context.get('unsupported_hypotheses', [])[:3]:
            hypotheses_summary += f"- {hyp.get('statement', 'N/A')}\n"

        return f"""You are a strategic research planning agent for an autonomous AI scientist.

**Research Objective**: {research_objective}

**Current State**:
- Cycle: {cycle}/20
- Past Findings: {findings_count}
- Unsupported Hypotheses: {unsupported_hyps}

**Recent Findings**:
{findings_summary or "No recent findings"}

**Unsupported Hypotheses**:
{hypotheses_summary or "No unsupported hypotheses"}

**Strategic Guidance**:
- Exploration ratio: {exploration_ratio*100:.0f}% (new directions)
- Exploitation ratio: {(1-exploration_ratio)*100:.0f}% (deepen findings)

**Task Requirements**:
1. Generate exactly {num_tasks} specific, executable tasks
2. Mix task types:
   - data_analysis: Analyze datasets, compute statistics
   - literature_review: Search and synthesize papers
   - hypothesis_generation: Generate new testable hypotheses
3. Each task must advance the research objective
4. Avoid redundancy with past work
5. Balance exploration ({int(exploration_ratio*num_tasks)} tasks) vs exploitation ({int((1-exploration_ratio)*num_tasks)} tasks)

**Task Types**:
- exploration=true: New directions, different domains, novel approaches
- exploration=false: Deepen existing findings, validate discoveries, test hypotheses

**Output Format** (JSON):
{{
  "tasks": [
    {{
      "id": 1,
      "type": "data_analysis" | "literature_review" | "hypothesis_generation",
      "description": "Specific task description (what to do)",
      "expected_output": "What this should produce",
      "required_skills": ["library1", "library2"],
      "exploration": true | false,
      "target_hypotheses": ["hypothesis_id"] (if applicable),
      "priority": 1-5 (1=highest)
    }},
    ...
  ],
  "rationale": "Strategic reasoning for this plan (2-3 sentences)"
}}

Generate a research plan as JSON (no additional text)."""

    def _parse_plan_response(self, response_text: str) -> Dict:
        """Parse LLM response to extract plan."""
        try:
            # Extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                plan_data = json.loads(json_str)
                return plan_data

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse plan JSON: {e}")

        return {'tasks': [], 'rationale': 'Failed to parse plan'}

    def _create_mock_plan(
        self,
        cycle: int,
        research_objective: str,
        context: Dict,
        num_tasks: int,
        exploration_ratio: float
    ) -> ResearchPlan:
        """Create mock plan for testing (when no LLM available)."""
        tasks = []
        num_exploration = int(num_tasks * exploration_ratio)

        # Create exploration tasks
        for i in range(1, num_exploration + 1):
            tasks.append(Task(
                task_id=i,
                task_type='data_analysis',
                description=f"Exploratory analysis {i} for {research_objective}",
                expected_output=f"Statistical findings and visualizations",
                required_skills=['pandas', 'scipy'],
                exploration=True,
                priority=1
            ))

        # Create exploitation tasks
        for i in range(num_exploration + 1, num_tasks + 1):
            tasks.append(Task(
                task_id=i,
                task_type='data_analysis',
                description=f"Validation analysis {i} for existing findings",
                expected_output=f"Hypothesis test results",
                required_skills=['pandas', 'statsmodels'],
                exploration=False,
                priority=2
            ))

        return ResearchPlan(
            cycle=cycle,
            tasks=tasks,
            rationale=f"Mock plan for cycle {cycle} (no LLM client provided)",
            exploration_ratio=exploration_ratio
        )

    def _create_generic_task(self, task_id: int) -> Task:
        """Create a generic task to fill gaps."""
        return Task(
            task_id=task_id,
            task_type='data_analysis',
            description=f"Additional analysis task {task_id}",
            expected_output="Statistical findings",
            required_skills=['pandas', 'scipy'],
            exploration=True,
            priority=3
        )

    async def revise_plan(
        self,
        original_plan: ResearchPlan,
        review_feedback: Dict,
        context: Dict
    ) -> ResearchPlan:
        """
        Revise plan based on reviewer feedback.

        Args:
            original_plan: Original plan that was rejected
            review_feedback: Feedback from PlanReviewerAgent
            context: Current context

        Returns:
            Revised ResearchPlan
        """
        # Simple revision: regenerate with feedback in context
        feedback_text = review_feedback.get('feedback', '')
        required_changes = review_feedback.get('required_changes', [])

        # Add feedback to context
        context_with_feedback = context.copy()
        context_with_feedback['previous_plan_feedback'] = feedback_text
        context_with_feedback['required_changes'] = required_changes

        # Regenerate plan
        return await self.create_plan(
            research_objective=context.get('research_objective', ''),
            context=context_with_feedback,
            num_tasks=len(original_plan.tasks)
        )
