"""Simple scenario test - agent should say hey when told hello."""

from __future__ import annotations

import pytest
import scenario

# Import utils directly
from .utils import CustomAgentAdapter, create_agent_with_adapter, create_judge_agent


@pytest.mark.integration
@pytest.mark.agent_test
async def test_hello_scenario() -> None:
    """Test that agent responds with 'hey' when told 'hello' using scenario framework."""

    # Create agent using our custom adapter fixture
    agent_adapter: CustomAgentAdapter = create_agent_with_adapter(
        name="HelloAgent",
        instructions="When someone says 'hello', you must respond with exactly 'hey' and nothing else.",
    )

    # Run scenario with JudgeAgent using our custom adapter
    result = await scenario.run(
        name="hello-test",
        description="Simple hello test with judge validation",
        agents=[
            agent_adapter,
            scenario.UserSimulatorAgent(),
            create_judge_agent(
                [
                    "Agent should respond with exactly 'hey'",
                    "Response should be lowercase",
                    "No extra words or punctuation",
                ]
            ),
        ],
        script=[scenario.user("hello"), scenario.agent(), scenario.judge()],
    )

    # Check result
    assert result.success
