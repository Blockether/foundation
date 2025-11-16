from typing import Literal

from pydantic import BaseModel, Field


class CuratorOperation(BaseModel):
    """Single operation to perform on the playbook"""

    type: Literal["ADD"] = Field(description="Type of operation")
    section: str = Field(description="Section to add the ground truth to.")
    content: str = Field(description="Content of the new ground truth.")


class CuratorOutput(BaseModel):
    """Output model for ACE Curator"""

    reasoning: str = Field(description="Analysis of what to add to playbook.")
    operations: list[CuratorOperation] = Field(
        default_factory=list, description="List of operations to perform."
    )
