from typing import Literal

from pydantic import BaseModel, Field


class TherapySessionParticipant(BaseModel):
    name: str = Field(description="Name of the participant")
    role: Literal["therapist", "patient"] = Field(
        description="Role of the participant in the therapy session"
    )


class TherapeuticSessionAnalysis(BaseModel):
    participants: TherapySessionParticipant = Field(
        ..., description="Details of the therapy session participant"
    )
