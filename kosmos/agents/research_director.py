"""
Research Director Agent - Master orchestrator for autonomous research (Phase 7).

This agent coordinates all other agents to execute the full research cycle:
Research Question → Hypotheses → Experiments → Results → Analysis → Refinement → Iteration

Uses message-based async coordination with all specialized agents.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import asyncio

from kosmos.agents.base import BaseAgent, AgentMessage, MessageType, AgentStatus
from kosmos.core.workflow import (
    ResearchWorkflow,
    ResearchPlan,
    WorkflowState,
    NextAction
)
from kosmos.core.llm import get_client
from kosmos.models.hypothesis import Hypothesis, HypothesisStatus

logger = logging.getLogger(__name__)


class ResearchDirectorAgent(BaseAgent):
    """
    Master orchestrator for autonomous research.

    Coordinates:
    - HypothesisGeneratorAgent: Generate and refine hypotheses
    - ExperimentDesignerAgent: Design experiment protocols
    - Executor: Run experiments
    - DataAnalystAgent: Interpret results
    - HypothesisRefiner: Refine hypotheses based on results
    - ConvergenceDetector: Detect when research is complete

    Uses message-based coordination for async agent communication.
    """

    def __init__(
        self,
        research_question: str,
        domain: Optional[str] = None,
        agent_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Research Director.

        Args:
            research_question: The research question to investigate
            domain: Optional domain (biology, physics, etc.)
            agent_id: Optional agent ID
            config: Optional configuration (max_iterations, stopping_criteria, etc.)
        """
        super().__init__(
            agent_id=agent_id,
            agent_type="ResearchDirector",
            config=config or {}
        )

        self.research_question = research_question
        self.domain = domain

        # Configuration
        self.max_iterations = self.config.get("max_iterations", 10)
        self.mandatory_stopping_criteria = self.config.get(
            "mandatory_stopping_criteria",
            ["iteration_limit", "no_testable_hypotheses"]
        )
        self.optional_stopping_criteria = self.config.get(
            "optional_stopping_criteria",
            ["novelty_decline", "diminishing_returns"]
        )

        # Initialize research plan and workflow
        self.research_plan = ResearchPlan(
            research_question=research_question,
            domain=domain,
            max_iterations=self.max_iterations
        )

        self.workflow = ResearchWorkflow(
            initial_state=WorkflowState.INITIALIZING,
            research_plan=self.research_plan
        )

        # Claude client for research planning and decision-making
        self.llm_client = get_client()

        # Agent registry (will be populated during coordination)
        self.agent_registry: Dict[str, str] = {}  # agent_type -> agent_id

        # Message correlation tracking
        self.pending_requests: Dict[str, Dict[str, Any]] = {}  # correlation_id -> request_info

        # Strategy effectiveness tracking
        self.strategy_stats: Dict[str, Dict[str, Any]] = {
            "hypothesis_generation": {"attempts": 0, "successes": 0, "cost": 0.0},
            "experiment_design": {"attempts": 0, "successes": 0, "cost": 0.0},
            "hypothesis_refinement": {"attempts": 0, "successes": 0, "cost": 0.0},
            "literature_review": {"attempts": 0, "successes": 0, "cost": 0.0}
        }

        # Research history
        self.iteration_history: List[Dict[str, Any]] = []

        logger.info(
            f"ResearchDirector initialized for question: '{research_question}' "
            f"(max_iterations={self.max_iterations})"
        )

    # ========================================================================
    # LIFECYCLE HOOKS
    # ========================================================================

    def _on_start(self):
        """Initialize director when started."""
        logger.info(f"ResearchDirector {self.agent_id} starting research cycle")
        self.workflow.transition_to(
            WorkflowState.GENERATING_HYPOTHESES,
            action="Start research cycle"
        )

    def _on_stop(self):
        """Cleanup when stopped."""
        logger.info(f"ResearchDirector {self.agent_id} stopped")

    # ========================================================================
    # MESSAGE HANDLING
    # ========================================================================

    def process_message(self, message: AgentMessage):
        """
        Process incoming message from other agents.

        Routes messages to appropriate handlers based on source agent.
        """
        # Extract sender agent type from message metadata or from_agent
        sender_type = message.metadata.get("agent_type", "unknown")

        logger.debug(f"Processing message from {sender_type} ({message.from_agent})")

        # Route to appropriate handler
        if sender_type == "HypothesisGeneratorAgent":
            self._handle_hypothesis_generator_response(message)
        elif sender_type == "ExperimentDesignerAgent":
            self._handle_experiment_designer_response(message)
        elif sender_type == "Executor":
            self._handle_executor_response(message)
        elif sender_type == "DataAnalystAgent":
            self._handle_data_analyst_response(message)
        elif sender_type == "HypothesisRefiner":
            self._handle_hypothesis_refiner_response(message)
        elif sender_type == "ConvergenceDetector":
            self._handle_convergence_detector_response(message)
        else:
            logger.warning(f"No handler for agent type: {sender_type}")

    def _handle_hypothesis_generator_response(self, message: AgentMessage):
        """
        Handle response from HypothesisGeneratorAgent.

        Expected content:
        - hypotheses: List of generated Hypothesis objects
        - count: Number of hypotheses generated
        """
        content = message.content

        if message.type == MessageType.ERROR:
            logger.error(f"Hypothesis generation failed: {content.get('error')}")
            self.errors_encountered += 1
            # TODO: Implement error recovery strategy
            return

        # Extract hypotheses
        hypothesis_ids = content.get("hypothesis_ids", [])
        count = content.get("count", 0)

        logger.info(f"Received {count} hypotheses from generator")

        # Update research plan
        for hyp_id in hypothesis_ids:
            self.research_plan.add_hypothesis(hyp_id)

        # Update strategy stats
        self.strategy_stats["hypothesis_generation"]["attempts"] += 1
        if count > 0:
            self.strategy_stats["hypothesis_generation"]["successes"] += 1

        # Decide next action
        next_action = self.decide_next_action()
        self._execute_next_action(next_action)

    def _handle_experiment_designer_response(self, message: AgentMessage):
        """
        Handle response from ExperimentDesignerAgent.

        Expected content:
        - protocol_id: ID of designed experiment protocol
        - hypothesis_id: ID of hypothesis being tested
        """
        content = message.content

        if message.type == MessageType.ERROR:
            logger.error(f"Experiment design failed: {content.get('error')}")
            self.errors_encountered += 1
            return

        protocol_id = content.get("protocol_id")
        hypothesis_id = content.get("hypothesis_id")

        logger.info(f"Received experiment design: {protocol_id} for hypothesis {hypothesis_id}")

        # Update research plan
        self.research_plan.add_experiment(protocol_id)

        # Update strategy stats
        self.strategy_stats["experiment_design"]["attempts"] += 1
        if protocol_id:
            self.strategy_stats["experiment_design"]["successes"] += 1

        # Decide next action
        next_action = self.decide_next_action()
        self._execute_next_action(next_action)

    def _handle_executor_response(self, message: AgentMessage):
        """
        Handle response from Executor.

        Expected content:
        - result_id: ID of experiment result
        - protocol_id: ID of protocol executed
        - status: SUCCESS/FAILURE/ERROR
        """
        content = message.content

        if message.type == MessageType.ERROR:
            logger.error(f"Experiment execution failed: {content.get('error')}")
            self.errors_encountered += 1
            return

        result_id = content.get("result_id")
        protocol_id = content.get("protocol_id")
        status = content.get("status")

        logger.info(f"Received experiment result: {result_id} (status: {status})")

        # Update research plan
        self.research_plan.add_result(result_id)
        self.research_plan.mark_experiment_complete(protocol_id)

        # Transition to analyzing state
        self.workflow.transition_to(
            WorkflowState.ANALYZING,
            action=f"Analyze result {result_id}"
        )

        # Send to DataAnalystAgent for interpretation
        next_action = NextAction.ANALYZE_RESULT
        self._execute_next_action(next_action)

    def _handle_data_analyst_response(self, message: AgentMessage):
        """
        Handle response from DataAnalystAgent.

        Expected content:
        - interpretation: ResultInterpretation object
        - result_id: ID of analyzed result
        - hypothesis_supported: bool
        """
        content = message.content

        if message.type == MessageType.ERROR:
            logger.error(f"Result analysis failed: {content.get('error')}")
            self.errors_encountered += 1
            return

        result_id = content.get("result_id")
        hypothesis_id = content.get("hypothesis_id")
        hypothesis_supported = content.get("hypothesis_supported")

        logger.info(
            f"Received result interpretation for {result_id}: "
            f"hypothesis {hypothesis_id} supported={hypothesis_supported}"
        )

        # Update hypothesis status in research plan
        if hypothesis_supported is True:
            self.research_plan.mark_supported(hypothesis_id)
        elif hypothesis_supported is False:
            self.research_plan.mark_rejected(hypothesis_id)
        else:
            # Inconclusive
            self.research_plan.mark_tested(hypothesis_id)

        # Transition to refining state
        self.workflow.transition_to(
            WorkflowState.REFINING,
            action=f"Refine based on result {result_id}"
        )

        # Decide next action (may refine hypothesis, generate new ones, or converge)
        next_action = self.decide_next_action()
        self._execute_next_action(next_action)

    def _handle_hypothesis_refiner_response(self, message: AgentMessage):
        """
        Handle response from HypothesisRefiner.

        Expected content:
        - refined_hypothesis_ids: List of refined/spawned hypothesis IDs
        - retired_hypothesis_ids: List of retired hypothesis IDs
        - action_taken: REFINED/RETIRED/SPAWNED
        """
        content = message.content

        if message.type == MessageType.ERROR:
            logger.error(f"Hypothesis refinement failed: {content.get('error')}")
            self.errors_encountered += 1
            return

        refined_ids = content.get("refined_hypothesis_ids", [])
        retired_ids = content.get("retired_hypothesis_ids", [])

        logger.info(f"Hypothesis refinement: {len(refined_ids)} refined, {len(retired_ids)} retired")

        # Add refined hypotheses to pool
        for hyp_id in refined_ids:
            self.research_plan.add_hypothesis(hyp_id)

        # Update strategy stats
        self.strategy_stats["hypothesis_refinement"]["attempts"] += 1
        if refined_ids:
            self.strategy_stats["hypothesis_refinement"]["successes"] += 1

        # Decide next action
        next_action = self.decide_next_action()
        self._execute_next_action(next_action)

    def _handle_convergence_detector_response(self, message: AgentMessage):
        """
        Handle response from ConvergenceDetector.

        Expected content:
        - should_converge: bool
        - reason: str (why convergence detected)
        - metrics: ConvergenceMetrics
        """
        content = message.content

        should_converge = content.get("should_converge", False)
        reason = content.get("reason", "")

        if should_converge:
            logger.info(f"Convergence detected: {reason}")

            # Update research plan
            self.research_plan.has_converged = True
            self.research_plan.convergence_reason = reason

            # Transition to converged state
            self.workflow.transition_to(
                WorkflowState.CONVERGED,
                action=f"Research converged: {reason}"
            )

            # Stop the director
            self.stop()
        else:
            logger.debug("Convergence check: not yet converged")

    # ========================================================================
    # MESSAGE SENDING (to other agents)
    # ========================================================================

    def _send_to_hypothesis_generator(
        self,
        action: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentMessage:
        """
        Send request to HypothesisGeneratorAgent.

        Args:
            action: Action to request (generate, refine)
            context: Additional context (research_question, literature, etc.)

        Returns:
            AgentMessage: Sent message
        """
        content = {
            "action": action,
            "research_question": self.research_question,
            "domain": self.domain,
            "context": context or {}
        }

        target_agent = self.agent_registry.get("HypothesisGeneratorAgent", "hypothesis_generator")

        message = self.send_message(
            to_agent=target_agent,
            content=content,
            message_type=MessageType.REQUEST
        )

        self.pending_requests[message.id] = {
            "agent": "HypothesisGeneratorAgent",
            "action": action,
            "timestamp": datetime.utcnow()
        }

        logger.debug(f"Sent {action} request to HypothesisGeneratorAgent")
        return message

    def _send_to_experiment_designer(
        self,
        hypothesis_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentMessage:
        """Send request to ExperimentDesignerAgent to design protocol."""
        content = {
            "action": "design_experiment",
            "hypothesis_id": hypothesis_id,
            "context": context or {}
        }

        target_agent = self.agent_registry.get("ExperimentDesignerAgent", "experiment_designer")

        message = self.send_message(
            to_agent=target_agent,
            content=content,
            message_type=MessageType.REQUEST
        )

        self.pending_requests[message.id] = {
            "agent": "ExperimentDesignerAgent",
            "hypothesis_id": hypothesis_id,
            "timestamp": datetime.utcnow()
        }

        logger.debug(f"Sent design request to ExperimentDesignerAgent for hypothesis {hypothesis_id}")
        return message

    def _send_to_executor(
        self,
        protocol_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentMessage:
        """Send request to Executor to run experiment."""
        content = {
            "action": "execute_experiment",
            "protocol_id": protocol_id,
            "context": context or {}
        }

        target_agent = self.agent_registry.get("Executor", "executor")

        message = self.send_message(
            to_agent=target_agent,
            content=content,
            message_type=MessageType.REQUEST
        )

        self.pending_requests[message.id] = {
            "agent": "Executor",
            "protocol_id": protocol_id,
            "timestamp": datetime.utcnow()
        }

        logger.debug(f"Sent execution request to Executor for protocol {protocol_id}")
        return message

    def _send_to_data_analyst(
        self,
        result_id: str,
        hypothesis_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentMessage:
        """Send request to DataAnalystAgent to interpret results."""
        content = {
            "action": "interpret_results",
            "result_id": result_id,
            "hypothesis_id": hypothesis_id,
            "context": context or {}
        }

        target_agent = self.agent_registry.get("DataAnalystAgent", "data_analyst")

        message = self.send_message(
            to_agent=target_agent,
            content=content,
            message_type=MessageType.REQUEST
        )

        self.pending_requests[message.id] = {
            "agent": "DataAnalystAgent",
            "result_id": result_id,
            "timestamp": datetime.utcnow()
        }

        logger.debug(f"Sent interpretation request to DataAnalystAgent for result {result_id}")
        return message

    def _send_to_hypothesis_refiner(
        self,
        hypothesis_id: str,
        result_id: Optional[str] = None,
        action: str = "evaluate",
        context: Optional[Dict[str, Any]] = None
    ) -> AgentMessage:
        """Send request to HypothesisRefiner."""
        content = {
            "action": action,
            "hypothesis_id": hypothesis_id,
            "result_id": result_id,
            "context": context or {}
        }

        target_agent = self.agent_registry.get("HypothesisRefiner", "hypothesis_refiner")

        message = self.send_message(
            to_agent=target_agent,
            content=content,
            message_type=MessageType.REQUEST
        )

        self.pending_requests[message.id] = {
            "agent": "HypothesisRefiner",
            "hypothesis_id": hypothesis_id,
            "timestamp": datetime.utcnow()
        }

        logger.debug(f"Sent {action} request to HypothesisRefiner for hypothesis {hypothesis_id}")
        return message

    def _send_to_convergence_detector(
        self,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentMessage:
        """Send request to ConvergenceDetector to check if research is complete."""
        content = {
            "action": "check_convergence",
            "research_plan": self.research_plan.dict(),
            "context": context or {}
        }

        target_agent = self.agent_registry.get("ConvergenceDetector", "convergence_detector")

        message = self.send_message(
            to_agent=target_agent,
            content=content,
            message_type=MessageType.REQUEST
        )

        self.pending_requests[message.id] = {
            "agent": "ConvergenceDetector",
            "timestamp": datetime.utcnow()
        }

        logger.debug("Sent convergence check request to ConvergenceDetector")
        return message

    # ========================================================================
    # RESEARCH PLANNING (using Claude)
    # ========================================================================

    def generate_research_plan(self) -> str:
        """
        Generate initial research plan using Claude.

        Returns:
            str: Research plan description
        """
        prompt = f"""You are a research director planning an autonomous scientific investigation.

Research Question: {self.research_question}
Domain: {self.domain or "General"}

Please generate a research plan that includes:

1. **Initial Hypothesis Directions** (3-5 high-level directions to explore)
2. **Experiment Strategy** (what types of experiments would be most informative)
3. **Success Criteria** (how will we know when we've answered the question)
4. **Resource Considerations** (estimated experiments needed, complexity)

Provide a structured, actionable plan in 2-3 paragraphs.
"""

        try:
            response = self.llm_client.generate(prompt, max_tokens=1000)

            # Store in research plan
            self.research_plan.initial_strategy = response

            logger.info("Generated initial research plan using Claude")
            return response

        except Exception as e:
            logger.error(f"Failed to generate research plan: {e}")
            return f"Error generating plan: {str(e)}"

    # ========================================================================
    # DECISION MAKING
    # ========================================================================

    def decide_next_action(self) -> NextAction:
        """
        Decide what to do next based on current workflow state and research plan.

        Decision tree:
        - If no hypotheses: GENERATE_HYPOTHESIS
        - If untested hypotheses: DESIGN_EXPERIMENT
        - If experiments in queue: EXECUTE_EXPERIMENT
        - If results need analysis: ANALYZE_RESULT
        - If hypotheses need refinement: REFINE_HYPOTHESIS
        - If convergence criteria met: CONVERGE
        - Otherwise: Check convergence, then GENERATE_HYPOTHESIS

        Returns:
            NextAction: Next action to take
        """
        current_state = self.workflow.current_state

        logger.debug(f"Deciding next action (state: {current_state})")

        # Check convergence first
        if self._should_check_convergence():
            return NextAction.CONVERGE

        # State-based decision making
        if current_state == WorkflowState.GENERATING_HYPOTHESES:
            # Generate hypotheses
            return NextAction.GENERATE_HYPOTHESIS

        elif current_state == WorkflowState.DESIGNING_EXPERIMENTS:
            # Design experiments for untested hypotheses
            untested = self.research_plan.get_untested_hypotheses()
            if untested:
                return NextAction.DESIGN_EXPERIMENT
            else:
                # All hypotheses tested, check convergence or generate more
                return NextAction.CONVERGE

        elif current_state == WorkflowState.EXECUTING:
            # Execute queued experiments
            if self.research_plan.experiment_queue:
                return NextAction.EXECUTE_EXPERIMENT
            else:
                return NextAction.ANALYZE_RESULT

        elif current_state == WorkflowState.ANALYZING:
            # Analyze recent results
            return NextAction.ANALYZE_RESULT

        elif current_state == WorkflowState.REFINING:
            # Refine hypotheses based on results
            if self.research_plan.tested_hypotheses:
                return NextAction.REFINE_HYPOTHESIS
            else:
                return NextAction.GENERATE_HYPOTHESIS

        elif current_state == WorkflowState.CONVERGED:
            return NextAction.CONVERGE

        elif current_state == WorkflowState.ERROR:
            return NextAction.ERROR_RECOVERY

        else:
            # Default: generate hypotheses
            return NextAction.GENERATE_HYPOTHESIS

    def _execute_next_action(self, action: NextAction):
        """
        Execute the decided next action.

        Args:
            action: Action to execute
        """
        logger.info(f"Executing next action: {action}")

        if action == NextAction.GENERATE_HYPOTHESIS:
            self._send_to_hypothesis_generator(action="generate")

        elif action == NextAction.DESIGN_EXPERIMENT:
            # Get first untested hypothesis
            untested = self.research_plan.get_untested_hypotheses()
            if untested:
                self._send_to_experiment_designer(hypothesis_id=untested[0])

        elif action == NextAction.EXECUTE_EXPERIMENT:
            # Get first queued experiment
            if self.research_plan.experiment_queue:
                protocol_id = self.research_plan.experiment_queue[0]
                self._send_to_executor(protocol_id=protocol_id)

        elif action == NextAction.ANALYZE_RESULT:
            # Get most recent result
            if self.research_plan.results:
                result_id = self.research_plan.results[-1]
                self._send_to_data_analyst(result_id=result_id)

        elif action == NextAction.REFINE_HYPOTHESIS:
            # Refine most recently tested hypothesis
            if self.research_plan.tested_hypotheses:
                hypothesis_id = self.research_plan.tested_hypotheses[-1]
                self._send_to_hypothesis_refiner(
                    hypothesis_id=hypothesis_id,
                    action="evaluate"
                )

        elif action == NextAction.CONVERGE:
            self._send_to_convergence_detector()

        elif action == NextAction.PAUSE:
            self.pause()

        else:
            logger.warning(f"Unknown action: {action}")

    def _should_check_convergence(self) -> bool:
        """
        Check if convergence should be evaluated.

        Returns:
            bool: True if convergence check is needed
        """
        # Check iteration limit (mandatory)
        if self.research_plan.iteration_count >= self.research_plan.max_iterations:
            logger.info("Iteration limit reached")
            return True

        # Check if no testable hypotheses (mandatory)
        if not self.research_plan.hypothesis_pool:
            logger.info("No hypotheses in pool")
            return True

        untested = self.research_plan.get_untested_hypotheses()
        if not untested and not self.research_plan.experiment_queue:
            logger.info("No untested hypotheses and no queued experiments")
            return True

        return False

    # ========================================================================
    # STRATEGY ADAPTATION
    # ========================================================================

    def select_next_strategy(self) -> str:
        """
        Select next strategy based on effectiveness tracking.

        Strategies with higher success rates are favored.

        Returns:
            str: Selected strategy name
        """
        # Calculate effectiveness scores
        scores = {}
        for strategy, stats in self.strategy_stats.items():
            attempts = stats["attempts"]
            if attempts == 0:
                # Favor unexplored strategies
                scores[strategy] = 1.0
            else:
                success_rate = stats["successes"] / attempts
                scores[strategy] = success_rate

        # Select strategy with highest score
        best_strategy = max(scores.items(), key=lambda x: x[1])[0]

        logger.debug(f"Selected strategy: {best_strategy} (scores: {scores})")
        return best_strategy

    def update_strategy_effectiveness(self, strategy: str, success: bool, cost: float = 0.0):
        """
        Update strategy effectiveness tracking.

        Args:
            strategy: Strategy name
            success: Whether strategy was successful
            cost: Cost incurred (API tokens, compute time, etc.)
        """
        if strategy in self.strategy_stats:
            self.strategy_stats[strategy]["attempts"] += 1
            if success:
                self.strategy_stats[strategy]["successes"] += 1
            self.strategy_stats[strategy]["cost"] += cost

            logger.debug(f"Updated strategy {strategy}: success={success}, cost={cost}")

    # ========================================================================
    # AGENT REGISTRY
    # ========================================================================

    def register_agent(self, agent_type: str, agent_id: str):
        """
        Register an agent for coordination.

        Args:
            agent_type: Type of agent (HypothesisGeneratorAgent, etc.)
            agent_id: Unique agent ID
        """
        self.agent_registry[agent_type] = agent_id
        logger.info(f"Registered {agent_type} with ID {agent_id}")

    def get_agent_id(self, agent_type: str) -> Optional[str]:
        """
        Get agent ID for a given type.

        Args:
            agent_type: Agent type

        Returns:
            Optional[str]: Agent ID if registered
        """
        return self.agent_registry.get(agent_type)

    # ========================================================================
    # EXECUTE (BaseAgent interface)
    # ========================================================================

    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute research task.

        Args:
            task: Task specification (usually {"action": "start_research"})

        Returns:
            dict: Task result
        """
        action = task.get("action", "start_research")

        if action == "start_research":
            # Generate initial research plan
            plan = self.generate_research_plan()

            # Start the workflow
            self.start()

            # Execute first action
            next_action = self.decide_next_action()
            self._execute_next_action(next_action)

            return {
                "status": "research_started",
                "research_plan": plan,
                "next_action": next_action.value
            }

        elif action == "step":
            # Execute one step of research
            next_action = self.decide_next_action()
            self._execute_next_action(next_action)

            return {
                "status": "step_executed",
                "next_action": next_action.value,
                "workflow_state": self.workflow.current_state.value
            }

        else:
            raise ValueError(f"Unknown action: {action}")

    # ========================================================================
    # STATUS & REPORTING
    # ========================================================================

    def get_research_status(self) -> Dict[str, Any]:
        """
        Get comprehensive research status.

        Returns:
            dict: Full research status including plan, workflow, statistics
        """
        return {
            "research_question": self.research_question,
            "domain": self.domain,
            "workflow_state": self.workflow.current_state.value,
            "iteration": self.research_plan.iteration_count,
            "max_iterations": self.research_plan.max_iterations,
            "has_converged": self.research_plan.has_converged,
            "convergence_reason": self.research_plan.convergence_reason,
            "hypothesis_pool_size": len(self.research_plan.hypothesis_pool),
            "hypotheses_tested": len(self.research_plan.tested_hypotheses),
            "hypotheses_supported": len(self.research_plan.supported_hypotheses),
            "hypotheses_rejected": len(self.research_plan.rejected_hypotheses),
            "experiments_completed": len(self.research_plan.completed_experiments),
            "results_count": len(self.research_plan.results),
            "strategy_stats": self.strategy_stats,
            "agent_status": self.get_status()
        }
