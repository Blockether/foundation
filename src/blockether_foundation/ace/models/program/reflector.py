from typing import Literal

from pydantic import BaseModel, Field


class GroundTruthTag(BaseModel):
    """Tag for a grand truth indicating its usefulness"""

    id: str = Field(description="Grand truth ID")
    tag: Literal["helpful", "harmful", "neutral"] = Field(description="Feedback tag")


class ReflectorOutput(BaseModel):
    reasoning: str = Field(description="Analysis of the previous step outcome.")

    error_identification: str | None = Field(
        description=(
            "Identification of any errors or issues in the previous step. "
            "Leave `None` if no errors were found."
        )
    )

    root_cause_analysis: str | None = Field(
        description=(
            "Analysis of the root cause of any identified errors. "
            "Leave `None` if no errors were found."
        )
    )

    correct_approach: str | None = Field(
        description=(
            "Description of the correct approach or solution to address the identified errors. "
            "Leave `None` if no errors were found."
        )
    )

    key_insights: str | None = Field(
        description=(
            "Key insights or lessons learned from the reflection process. "
            "Leave `None` if there are no specific insights."
        )
    )

    ground_truths: list[GroundTruthTag] = Field(
        default_factory=lambda: [], description="Feedback on used ground truths."
    )
