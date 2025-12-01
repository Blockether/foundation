import json
import pickle

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


class BaseModelFilePersistable(BaseModel):
    @classmethod
    def from_json_file(cls, file_path: str) -> "BaseModelFilePersistable":
        with open(file_path) as f:
            data = json.load(f)
        return cls(**data)

    def to_json_file(self, file_path: str) -> None:
        with open(file_path, "w") as f:
            json.dump(self.model_dump(), f, indent=4)

    @classmethod
    def from_pickle_file(cls, file_path: str) -> "BaseModelFilePersistable":
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        return cls(**data)

    def to_pickle_file(self, file_path: str) -> None:
        with open(file_path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
