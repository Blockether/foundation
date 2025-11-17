from typing import Literal

from pydantic import BaseModel, Field


class CuratorOperation(BaseModel):
    """Single operation to perform on the playbook"""

    type: Literal["ADD", "UPDATE", "REMOVE"] = Field(description="Type of operation")
    section: str = Field(description="Section to add/update the ground truth in.")
    content: str = Field(description="Content of the new/updated ground truth.")
    ground_truth_id: str | None = Field(
        default=None,
        description="ID of the ground truth to update/remove. Required for UPDATE and REMOVE operations.",
    )
    new_title: str | None = Field(
        default=None,
        description="New title for the ground truth. Optional for UPDATE operations.",
    )


class CuratorOutput(BaseModel):
    """Output model for ACE Curator"""

    reasoning: str = Field(description="Analysis of what to add to playbook.")
    operations: list[CuratorOperation] | None = Field(
        default_factory=list,
        description=(
            "List of operations to perform on the playbook."
            "Leave `None` if no changes are needed."
        ),
    )
