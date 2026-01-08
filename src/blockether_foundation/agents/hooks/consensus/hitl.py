"""Consensus HITL Toolkit - User feedback during consensus refinement.

This toolkit uses Agno's native Dynamic User Input pattern:
- Agent can call get_user_input whenever it needs more information
- Agent can call get_user_input multiple times (Agno handles the pause/continue loop)
- Questionnaire is provided in the agent's context (refinement prompt)
"""

from __future__ import annotations

from typing import Any

from agno.tools import Toolkit
from agno.tools.user_control_flow import UserControlFlowTools

from .prompts import build_hitl_toolkit_instructions


class ConsensusHITLToolkit(Toolkit):
    """Toolkit for human-in-the-loop feedback during consensus.

    Uses Agno's native Dynamic User Input pattern:
    1. Agent can call get_user_input whenever it needs more information
    2. Agent can call get_user_input multiple times (Agno handles pause/continue)
    3. The questionnaire is provided in the agent's context (refinement prompt)

    The toolkit provides static instructions - questionnaire-specific content
    is passed via the refinement prompt, not via toolkit instructions.
    """

    TOOLKIT_NAME = "consensus_human_in_the_loop_feedback"

    # Instructions built dynamically from prompts module
    INSTRUCTIONS = build_hitl_toolkit_instructions()

    def __init__(self) -> None:
        """Initialize the HITL toolkit with static instructions."""
        # Use Agno's native UserControlFlowTools for get_user_input
        self._user_control_flow = UserControlFlowTools(
            add_instructions=False,  # We'll add our own instructions
            enable_get_user_input=True,
        )

        all_tools: list[Any] = [
            # Include get_user_input from UserControlFlowTools
            *self._user_control_flow.tools,
        ]

        super().__init__(
            name=self.TOOLKIT_NAME,
            tools=all_tools,
            instructions=self.INSTRUCTIONS,
            add_instructions=False,
        )
