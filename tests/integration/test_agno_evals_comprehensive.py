"""Comprehensive integration tests using Agno evals instead of scenario framework."""

from __future__ import annotations

import pytest
from agno.agent.agent import Agent
from agno.eval.accuracy import AccuracyEval, AccuracyResult
from agno.eval.agent_as_judge import AgentAsJudgeEval
from agno.models.openai import OpenAIChat
from agno.run.agent import RunOutput

from .utils import AgentWrapper, create_agent_with_adapter, create_judge_agent

# Constants for score thresholds
ACCURACY_SCORE_THRESHOLD = 8.0
JUDGE_SCORE_THRESHOLD = 8
CONVERSATION_THRESHOLD = 1


@pytest.mark.integration
@pytest.mark.agno_eval
async def test_agno_evaluation_frameworks():
    """Test both AccuracyEval and AgentAsJudgeEval frameworks with a simple hello test case."""

    # Create agent using existing utility
    agent_wrapper: AgentWrapper = create_agent_with_adapter(
        name="HelloAgent",
        instructions="When someone says 'hello', you must respond with exactly 'hey' and nothing else.",
    )
    agent: Agent = agent_wrapper.agent

    # Test 1: Accuracy Evaluation
    accuracy_eval = AccuracyEval(
        name="Hello Response Accuracy",
        model=OpenAIChat(
            api_key=agent.model.api_key
            if agent.model and isinstance(agent.model, OpenAIChat)
            else None,
            base_url=agent.model.base_url
            if agent.model and isinstance(agent.model, OpenAIChat)
            else None,
            id="gpt-4o",
        ),
        agent=agent,
        input="hello",
        expected_output="hey",
        additional_guidelines="Response should be exactly 'hey' in lowercase, with no extra words or punctuation.",
        num_iterations=1,
    )

    # Run accuracy evaluation
    accuracy_result: AccuracyResult | None = accuracy_eval.run(print_results=False)

    # Check accuracy result
    assert accuracy_result is not None
    assert accuracy_result.avg_score >= ACCURACY_SCORE_THRESHOLD
    assert accuracy_result.mean_score >= ACCURACY_SCORE_THRESHOLD

    # Test 2: Agent-as-Judge Evaluation
    judge_agent = create_judge_agent(
        criteria=[
            "Response must be exactly 'hey' in lowercase",
            "No extra words or punctuation",
            "Must be in English",
        ],
        name="ResponseJudge",
    )

    # Get agent response
    response: RunOutput = agent.run("hello", session_id="test_session")  # type: ignore[reportUnknownMemberType]

    # Create agent-as-judge evaluation
    judge_eval = AgentAsJudgeEval(
        name="Hello Response Judge",
        criteria="Agent should respond with exactly 'hey' - lowercase, no extra words or punctuation",
        evaluator_agent=judge_agent,
        scoring_strategy="numeric",
        threshold=JUDGE_SCORE_THRESHOLD,
    )

    # Run judge evaluation
    judge_result = judge_eval.run(
        input="hello",
        output=str(response.content) if response.content is not None else "",
        print_results=False,
    )

    # Check judge result
    assert judge_result is not None
    assert len(judge_result.results) > 0
    score = judge_result.results[0].score
    assert score is not None
    assert score >= JUDGE_SCORE_THRESHOLD
    assert judge_result.results[0].passed is True


@pytest.mark.integration
@pytest.mark.agno_eval
async def test_multi_turn_conversation_and_memory():
    """Test multi-turn conversation with memory using Agno AgentAsJudgeEval."""

    # Create conversation agent
    agent_wrapper: AgentWrapper = create_agent_with_adapter(
        name="ConversationAgent",
        instructions="You are a helpful assistant. Keep responses brief and friendly.",
    )
    agent: Agent = agent_wrapper.agent

    # Test conversation with memory
    session_id = "conversation_test"

    # First interaction
    response1: RunOutput = await agent.arun("My name is Alice", session_id=session_id)  # type: ignore[reportUnknownMemberType]
    assert response1.content is not None

    # Second interaction - should remember name
    response2: RunOutput = await agent.arun("What's my name?", session_id=session_id)  # type: ignore[reportUnknownMemberType]

    # Create judge for memory evaluation
    judge_agent = create_judge_agent(
        instructions="""Evaluate if the agent correctly remembered and used the name 'Alice' in the response.

        Context: The user previously said "My name is Alice" in this conversation.
        The agent should remember this information and use the name Alice when asked "What's my name?".
        """,
        name="MemoryJudge",
    )

    # Create evaluation
    evaluation = AgentAsJudgeEval(
        name="Name Memory Test",
        criteria="Agent should remember the user's name is Alice and use it in the response",
        evaluator_agent=judge_agent,
        scoring_strategy="numeric",
        threshold=CONVERSATION_THRESHOLD,
    )

    # Run evaluation
    result = evaluation.run(
        input="What's my name?",
        output=str(response2.content) if response2.content is not None else "",
        print_results=False,
    )

    # Verify evaluation worked
    assert result is not None
    assert len(result.results) > 0
    assert result.results[0].score is not None
    assert result.results[0].score >= CONVERSATION_THRESHOLD
    assert result.results[0].passed is not None
