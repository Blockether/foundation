import json

from pydantic import BaseModel, Field


class ChainOfThoughts(BaseModel):
    reasoning: str = Field(
        description="Step-by-step reasoning process explaining how you reached your conclusion. Include relevant context, considered alternatives, and key decision factors. Format as markdown with bullet points or numbered lists in case of reasoning steps for clarity. Prefer concise and clear explanations."
    )

    confidence: float = Field(
        description="Confidence level (0.0 to 1.0) in the reasoning process and the trust score of values of the accompanying fields.",
        ge=0.0,
        le=1.0,
    )


class PydanticFilePersistable(BaseModel):
    @classmethod
    def from_file(cls, file_path: str) -> "PydanticFilePersistable":
        with open(file_path) as f:
            data = json.load(f)
        return cls(**data)

    def save_to_file(self, file_path: str) -> None:
        with open(file_path, "w") as f:
            json.dump(self.model_dump(), f, indent=4)
