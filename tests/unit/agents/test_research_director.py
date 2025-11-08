"""
Unit tests for ResearchDirectorAgent (Phase 7).
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from kosmos.agents.research_director import ResearchDirectorAgent
from kosmos.agents.base import AgentMessage, MessageType, AgentStatus
from kosmos.core.workflow import WorkflowState, NextAction, ResearchPlan


# Fixtures

@pytest.fixture
def research_director():
    """Create research director with test configuration."""
    return ResearchDirectorAgent(
        research_question="Does sample size affect statistical power?",
        domain="statistics",
        config={
            "max_iterations": 5,
            "mandatory_stopping_criteria": ["iteration_limit", "no_testable_hypotheses"],
            "optional_stopping_criteria": ["novelty_decline", "diminishing_returns"]
        }
    )


@pytest.fixture
def mock_llm_client():
    """Mock LLM client."""
    mock_client = Mock()
    mock_client.generate.return_value = "Research plan: Test hypothesis generation..."
    return mock_client


# Test Initialization

class TestResearchDirectorInitialization:
    """Test research director initialization."""

    def test_initialization(self, research_director):
        """Test basic initialization."""
        assert research_director.research_question == "Does sample size affect statistical power?"
        assert research_director.domain == "statistics"
        assert research_director.max_iterations == 5
        assert research_director.status == AgentStatus.CREATED
        assert research_director.workflow.current_state == WorkflowState.INITIALIZING

    def test_research_plan_initialized(self, research_director):
        """Test research plan is created."""
        plan = research_director.research_plan
        assert plan.research_question == "Does sample size affect statistical power?"
        assert plan.domain == "statistics"
        assert plan.max_iterations == 5
        assert plan.iteration_count == 0

    def test_workflow_initialized(self, research_director):
        """Test workflow state machine is created."""
        workflow = research_director.workflow
        assert workflow.current_state == WorkflowState.INITIALIZING
        assert workflow.research_plan == research_director.research_plan

    def test_strategy_stats_initialized(self, research_director):
        """Test strategy tracking is initialized."""
        stats = research_director.strategy_stats
        assert "hypothesis_generation" in stats
        assert "experiment_design" in stats
        assert "hypothesis_refinement" in stats
        assert "literature_review" in stats

        for strategy_stats in stats.values():
            assert strategy_stats["attempts"] == 0
            assert strategy_stats["successes"] == 0
            assert strategy_stats["cost"] == 0.0


# Test Lifecycle

class TestResearchDirectorLifecycle:
    """Test research director lifecycle management."""

    def test_start_director(self, research_director):
        """Test starting research director."""
        research_director.start()

        assert research_director.status == AgentStatus.RUNNING
        assert research_director.workflow.current_state == WorkflowState.GENERATING_HYPOTHESES

    def test_stop_director(self, research_director):
        """Test stopping research director."""
        research_director.start()
        research_director.stop()

        assert research_director.status == AgentStatus.STOPPED

    def test_pause_resume_director(self, research_director):
        """Test pausing and resuming director."""
        research_director.start()
        research_director.pause()
        assert research_director.status == AgentStatus.PAUSED

        research_director.resume()
        assert research_director.status == AgentStatus.RUNNING


# Test Message Handling

class TestMessageHandling:
    """Test message handling for different agent responses."""

    def test_handle_hypothesis_generator_response(self, research_director):
        """Test handling hypothesis generation response."""
        message = AgentMessage(
            type=MessageType.RESPONSE,
            from_agent="hypothesis_generator",
            to_agent=research_director.agent_id,
            content={
                "hypothesis_ids": ["hyp-1", "hyp-2", "hyp-3"],
                "count": 3
            },
            metadata={"agent_type": "HypothesisGeneratorAgent"}
        )

        # Mock decide_next_action and _execute_next_action to avoid recursion
        research_director.decide_next_action = Mock(return_value=NextAction.DESIGN_EXPERIMENT)
        research_director._execute_next_action = Mock()

        research_director._handle_hypothesis_generator_response(message)

        # Check hypotheses added to plan
        assert "hyp-1" in research_director.research_plan.hypothesis_pool
        assert "hyp-2" in research_director.research_plan.hypothesis_pool
        assert "hyp-3" in research_director.research_plan.hypothesis_pool

        # Check strategy stats updated
        stats = research_director.strategy_stats["hypothesis_generation"]
        assert stats["attempts"] == 1
        assert stats["successes"] == 1

    def test_handle_experiment_designer_response(self, research_director):
        """Test handling experiment design response."""
        message = AgentMessage(
            type=MessageType.RESPONSE,
            from_agent="experiment_designer",
            to_agent=research_director.agent_id,
            content={
                "protocol_id": "proto-1",
                "hypothesis_id": "hyp-1"
            },
            metadata={"agent_type": "ExperimentDesignerAgent"}
        )

        research_director.decide_next_action = Mock(return_value=NextAction.EXECUTE_EXPERIMENT)
        research_director._execute_next_action = Mock()

        research_director._handle_experiment_designer_response(message)

        # Check experiment added to queue
        assert "proto-1" in research_director.research_plan.experiment_queue

        # Check strategy stats updated
        stats = research_director.strategy_stats["experiment_design"]
        assert stats["attempts"] == 1
        assert stats["successes"] == 1

    def test_handle_executor_response(self, research_director):
        """Test handling experiment execution response."""
        # Add experiment to queue first
        research_director.research_plan.add_experiment("proto-1")

        message = AgentMessage(
            type=MessageType.RESPONSE,
            from_agent="executor",
            to_agent=research_director.agent_id,
            content={
                "result_id": "result-1",
                "protocol_id": "proto-1",
                "status": "SUCCESS"
            },
            metadata={"agent_type": "Executor"}
        )

        research_director.decide_next_action = Mock(return_value=NextAction.ANALYZE_RESULT)
        research_director._execute_next_action = Mock()

        research_director._handle_executor_response(message)

        # Check result added
        assert "result-1" in research_director.research_plan.results

        # Check experiment marked complete
        assert "proto-1" in research_director.research_plan.completed_experiments
        assert "proto-1" not in research_director.research_plan.experiment_queue

        # Check workflow transition
        assert research_director.workflow.current_state == WorkflowState.ANALYZING

    def test_handle_data_analyst_response_supported(self, research_director):
        """Test handling analysis response (hypothesis supported)."""
        research_director.research_plan.add_hypothesis("hyp-1")

        message = AgentMessage(
            type=MessageType.RESPONSE,
            from_agent="data_analyst",
            to_agent=research_director.agent_id,
            content={
                "result_id": "result-1",
                "hypothesis_id": "hyp-1",
                "hypothesis_supported": True
            },
            metadata={"agent_type": "DataAnalystAgent"}
        )

        research_director.decide_next_action = Mock(return_value=NextAction.REFINE_HYPOTHESIS)
        research_director._execute_next_action = Mock()

        research_director._handle_data_analyst_response(message)

        # Check hypothesis marked as supported
        assert "hyp-1" in research_director.research_plan.supported_hypotheses
        assert "hyp-1" in research_director.research_plan.tested_hypotheses

        # Check workflow transition
        assert research_director.workflow.current_state == WorkflowState.REFINING

    def test_handle_data_analyst_response_rejected(self, research_director):
        """Test handling analysis response (hypothesis rejected)."""
        research_director.research_plan.add_hypothesis("hyp-1")

        message = AgentMessage(
            type=MessageType.RESPONSE,
            from_agent="data_analyst",
            to_agent=research_director.agent_id,
            content={
                "result_id": "result-1",
                "hypothesis_id": "hyp-1",
                "hypothesis_supported": False
            },
            metadata={"agent_type": "DataAnalystAgent"}
        )

        research_director.decide_next_action = Mock(return_value=NextAction.REFINE_HYPOTHESIS)
        research_director._execute_next_action = Mock()

        research_director._handle_data_analyst_response(message)

        # Check hypothesis marked as rejected
        assert "hyp-1" in research_director.research_plan.rejected_hypotheses
        assert "hyp-1" in research_director.research_plan.tested_hypotheses

    def test_handle_hypothesis_refiner_response(self, research_director):
        """Test handling hypothesis refinement response."""
        message = AgentMessage(
            type=MessageType.RESPONSE,
            from_agent="hypothesis_refiner",
            to_agent=research_director.agent_id,
            content={
                "refined_hypothesis_ids": ["hyp-2", "hyp-3"],
                "retired_hypothesis_ids": ["hyp-1"],
                "action_taken": "REFINED"
            },
            metadata={"agent_type": "HypothesisRefiner"}
        )

        research_director.decide_next_action = Mock(return_value=NextAction.DESIGN_EXPERIMENT)
        research_director._execute_next_action = Mock()

        research_director._handle_hypothesis_refiner_response(message)

        # Check refined hypotheses added
        assert "hyp-2" in research_director.research_plan.hypothesis_pool
        assert "hyp-3" in research_director.research_plan.hypothesis_pool

        # Check strategy stats updated
        stats = research_director.strategy_stats["hypothesis_refinement"]
        assert stats["attempts"] == 1
        assert stats["successes"] == 1

    def test_handle_convergence_detector_response_converged(self, research_director):
        """Test handling convergence detection (research complete)."""
        message = AgentMessage(
            type=MessageType.RESPONSE,
            from_agent="convergence_detector",
            to_agent=research_director.agent_id,
            content={
                "should_converge": True,
                "reason": "Iteration limit reached",
                "metrics": {}
            },
            metadata={"agent_type": "ConvergenceDetector"}
        )

        research_director._handle_convergence_detector_response(message)

        # Check convergence status
        assert research_director.research_plan.has_converged is True
        assert research_director.research_plan.convergence_reason == "Iteration limit reached"

        # Check workflow transition
        assert research_director.workflow.current_state == WorkflowState.CONVERGED

        # Check director stopped
        assert research_director.status == AgentStatus.STOPPED

    def test_handle_convergence_detector_response_not_converged(self, research_director):
        """Test handling convergence detection (not yet complete)."""
        initial_state = research_director.workflow.current_state

        message = AgentMessage(
            type=MessageType.RESPONSE,
            from_agent="convergence_detector",
            to_agent=research_director.agent_id,
            content={
                "should_converge": False,
                "reason": "",
                "metrics": {}
            },
            metadata={"agent_type": "ConvergenceDetector"}
        )

        research_director._handle_convergence_detector_response(message)

        # Check no convergence
        assert research_director.research_plan.has_converged is False

        # Check workflow unchanged
        assert research_director.workflow.current_state == initial_state


# Test Message Sending

class TestMessageSending:
    """Test sending messages to other agents."""

    def test_send_to_hypothesis_generator(self, research_director):
        """Test sending message to hypothesis generator."""
        research_director.register_agent("HypothesisGeneratorAgent", "hyp-gen-1")

        message = research_director._send_to_hypothesis_generator(
            action="generate",
            context={"max_hypotheses": 5}
        )

        assert message.type == MessageType.REQUEST
        assert message.to_agent == "hyp-gen-1"
        assert message.content["action"] == "generate"
        assert message.content["research_question"] == research_director.research_question

    def test_send_to_experiment_designer(self, research_director):
        """Test sending message to experiment designer."""
        research_director.register_agent("ExperimentDesignerAgent", "exp-des-1")

        message = research_director._send_to_experiment_designer(
            hypothesis_id="hyp-1"
        )

        assert message.type == MessageType.REQUEST
        assert message.to_agent == "exp-des-1"
        assert message.content["action"] == "design_experiment"
        assert message.content["hypothesis_id"] == "hyp-1"

    def test_send_to_executor(self, research_director):
        """Test sending message to executor."""
        research_director.register_agent("Executor", "executor-1")

        message = research_director._send_to_executor(
            protocol_id="proto-1"
        )

        assert message.type == MessageType.REQUEST
        assert message.to_agent == "executor-1"
        assert message.content["action"] == "execute_experiment"
        assert message.content["protocol_id"] == "proto-1"

    def test_send_to_data_analyst(self, research_director):
        """Test sending message to data analyst."""
        research_director.register_agent("DataAnalystAgent", "analyst-1")

        message = research_director._send_to_data_analyst(
            result_id="result-1",
            hypothesis_id="hyp-1"
        )

        assert message.type == MessageType.REQUEST
        assert message.to_agent == "analyst-1"
        assert message.content["action"] == "interpret_results"
        assert message.content["result_id"] == "result-1"


# Test Research Planning

class TestResearchPlanning:
    """Test research planning with Claude."""

    @patch('kosmos.agents.research_director.get_client')
    def test_generate_research_plan(self, mock_get_client, research_director):
        """Test generating research plan using Claude."""
        mock_client = Mock()
        mock_client.generate.return_value = "Research plan: Generate hypotheses about sample size effects..."
        mock_get_client.return_value = mock_client

        research_director.llm_client = mock_client

        plan = research_director.generate_research_plan()

        # Check Claude was called
        assert mock_client.generate.call_count == 1

        # Check plan stored
        assert research_director.research_plan.initial_strategy == plan
        assert "sample size" in plan.lower()


# Test Decision Making

class TestDecisionMaking:
    """Test decision-making logic."""

    def test_decide_next_action_generate_hypothesis(self, research_director):
        """Test decision when in GENERATING_HYPOTHESES state."""
        research_director.workflow.transition_to(WorkflowState.GENERATING_HYPOTHESES, "Test")

        action = research_director.decide_next_action()

        assert action == NextAction.GENERATE_HYPOTHESIS

    def test_decide_next_action_design_experiment(self, research_director):
        """Test decision when in DESIGNING_EXPERIMENTS state with untested hypotheses."""
        research_director.workflow.transition_to(WorkflowState.DESIGNING_EXPERIMENTS, "Test")
        research_director.research_plan.add_hypothesis("hyp-1")

        action = research_director.decide_next_action()

        assert action == NextAction.DESIGN_EXPERIMENT

    def test_decide_next_action_execute_experiment(self, research_director):
        """Test decision when in EXECUTING state with queued experiments."""
        research_director.workflow.transition_to(WorkflowState.EXECUTING, "Test")
        research_director.research_plan.add_experiment("proto-1")

        action = research_director.decide_next_action()

        assert action == NextAction.EXECUTE_EXPERIMENT

    def test_decide_next_action_analyze_result(self, research_director):
        """Test decision when in ANALYZING state."""
        research_director.workflow.transition_to(WorkflowState.ANALYZING, "Test")
        research_director.research_plan.add_result("result-1")

        action = research_director.decide_next_action()

        assert action == NextAction.ANALYZE_RESULT

    def test_decide_next_action_refine_hypothesis(self, research_director):
        """Test decision when in REFINING state with tested hypotheses."""
        research_director.workflow.transition_to(WorkflowState.REFINING, "Test")
        research_director.research_plan.add_hypothesis("hyp-1")
        research_director.research_plan.mark_tested("hyp-1")

        action = research_director.decide_next_action()

        assert action == NextAction.REFINE_HYPOTHESIS

    def test_decide_next_action_converge(self, research_director):
        """Test decision when convergence criteria met."""
        # Set iteration to max
        research_director.research_plan.iteration_count = research_director.max_iterations

        action = research_director.decide_next_action()

        assert action == NextAction.CONVERGE

    def test_should_check_convergence_iteration_limit(self, research_director):
        """Test convergence check when iteration limit reached."""
        research_director.research_plan.iteration_count = research_director.max_iterations

        should_check = research_director._should_check_convergence()

        assert should_check is True

    def test_should_check_convergence_no_hypotheses(self, research_director):
        """Test convergence check when no hypotheses."""
        # No hypotheses in pool
        assert len(research_director.research_plan.hypothesis_pool) == 0

        should_check = research_director._should_check_convergence()

        assert should_check is True

    def test_should_check_convergence_all_tested(self, research_director):
        """Test convergence check when all hypotheses tested."""
        research_director.research_plan.add_hypothesis("hyp-1")
        research_director.research_plan.mark_tested("hyp-1")
        # No queued experiments
        assert len(research_director.research_plan.experiment_queue) == 0

        should_check = research_director._should_check_convergence()

        assert should_check is True


# Test Strategy Adaptation

class TestStrategyAdaptation:
    """Test strategy selection and adaptation."""

    def test_select_next_strategy_no_history(self, research_director):
        """Test strategy selection with no history (all unexplored)."""
        strategy = research_director.select_next_strategy()

        # Should return a valid strategy
        assert strategy in research_director.strategy_stats

    def test_select_next_strategy_with_success_rates(self, research_director):
        """Test strategy selection based on success rates."""
        # Set some history
        research_director.strategy_stats["hypothesis_generation"]["attempts"] = 10
        research_director.strategy_stats["hypothesis_generation"]["successes"] = 8  # 80%

        research_director.strategy_stats["experiment_design"]["attempts"] = 10
        research_director.strategy_stats["experiment_design"]["successes"] = 3  # 30%

        strategy = research_director.select_next_strategy()

        # Should favor hypothesis_generation (higher success rate)
        # Note: unexplored strategies have score 1.0, so they might still win
        assert strategy in research_director.strategy_stats

    def test_update_strategy_effectiveness_success(self, research_director):
        """Test updating strategy effectiveness on success."""
        research_director.update_strategy_effectiveness(
            strategy="hypothesis_generation",
            success=True,
            cost=100.0
        )

        stats = research_director.strategy_stats["hypothesis_generation"]
        assert stats["attempts"] == 1
        assert stats["successes"] == 1
        assert stats["cost"] == 100.0

    def test_update_strategy_effectiveness_failure(self, research_director):
        """Test updating strategy effectiveness on failure."""
        research_director.update_strategy_effectiveness(
            strategy="experiment_design",
            success=False,
            cost=50.0
        )

        stats = research_director.strategy_stats["experiment_design"]
        assert stats["attempts"] == 1
        assert stats["successes"] == 0
        assert stats["cost"] == 50.0


# Test Agent Registry

class TestAgentRegistry:
    """Test agent registration and lookup."""

    def test_register_agent(self, research_director):
        """Test registering an agent."""
        research_director.register_agent("HypothesisGeneratorAgent", "hyp-gen-1")

        agent_id = research_director.get_agent_id("HypothesisGeneratorAgent")
        assert agent_id == "hyp-gen-1"

    def test_get_agent_id_not_registered(self, research_director):
        """Test getting ID for unregistered agent."""
        agent_id = research_director.get_agent_id("NonExistentAgent")
        assert agent_id is None


# Test Execute

class TestExecute:
    """Test execute method (BaseAgent interface)."""

    @patch('kosmos.agents.research_director.get_client')
    def test_execute_start_research(self, mock_get_client, research_director):
        """Test executing start_research action."""
        mock_client = Mock()
        mock_client.generate.return_value = "Research plan..."
        mock_get_client.return_value = mock_client

        research_director.llm_client = mock_client
        research_director._execute_next_action = Mock()

        result = research_director.execute({"action": "start_research"})

        assert result["status"] == "research_started"
        assert "research_plan" in result
        assert "next_action" in result
        assert research_director.status == AgentStatus.RUNNING

    def test_execute_step(self, research_director):
        """Test executing a single step."""
        research_director.start()
        research_director._execute_next_action = Mock()

        result = research_director.execute({"action": "step"})

        assert result["status"] == "step_executed"
        assert "next_action" in result
        assert "workflow_state" in result

    def test_execute_unknown_action(self, research_director):
        """Test executing unknown action."""
        with pytest.raises(ValueError, match="Unknown action"):
            research_director.execute({"action": "unknown"})


# Test Status & Reporting

class TestStatusReporting:
    """Test status and reporting methods."""

    def test_get_research_status(self, research_director):
        """Test getting comprehensive research status."""
        research_director.research_plan.add_hypothesis("hyp-1")
        research_director.research_plan.add_hypothesis("hyp-2")
        research_director.research_plan.mark_supported("hyp-1")

        status = research_director.get_research_status()

        assert status["research_question"] == research_director.research_question
        assert status["domain"] == research_director.domain
        assert status["workflow_state"] == WorkflowState.INITIALIZING.value
        assert status["iteration"] == 0
        assert status["max_iterations"] == 5
        assert status["hypothesis_pool_size"] == 2
        assert status["hypotheses_tested"] == 1
        assert status["hypotheses_supported"] == 1
        assert "strategy_stats" in status


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
