"""
Core model classes with serialization support.
"""

import base64
import json
import pickle

from pydantic import BaseModel, Field


class BaseModelSerializable(BaseModel):
    """
    Base model with comprehensive serialization support.
    Provides JSON, pickle, base64, and file persistence methods.
    """

    @classmethod
    def from_json_file(cls, file_path: str) -> "BaseModelSerializable":
        """Load from JSON file."""
        with open(file_path) as f:
            data = json.load(f)
        return cls(**data)

    def to_json_file(self, file_path: str) -> None:
        """Save to JSON file."""
        with open(file_path, "w") as f:
            json.dump(self.model_dump(), f, indent=4)

    @classmethod
    def from_pickle_file(cls, file_path: str) -> "BaseModelSerializable":
        """Load from pickle file."""
        with open(file_path, "rb") as f:
            obj = pickle.load(f)
        return obj

    def to_pickle_file(self, file_path: str) -> None:
        """Save to pickle file."""
        with open(file_path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def to_json_string(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.model_dump(mode="json"), indent=2)

    @classmethod
    def from_json_string(cls, json_str: str) -> "BaseModelSerializable":
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls(**data)

    def to_base64(self) -> str:
        """Serialize to base64-encoded pickle string."""
        pickled = pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
        return base64.b64encode(pickled).decode("utf-8")

    @classmethod
    def from_base64(cls, base64_str: str) -> "BaseModelSerializable":
        """Deserialize from base64-encoded pickle string."""
        pickled = base64.b64decode(base64_str.encode("utf-8"))
        return pickle.loads(pickled)


class ChainOfThoughts(BaseModelSerializable):
    reasoning: str = Field(
        description="Step-by-step reasoning process explaining how you reached your conclusion. Include relevant context, considered alternatives, and key decision factors. Format as markdown with bullet points or numbered lists in case of reasoning steps for clarity. Prefer concise and clear explanations."
    )

    confidence: float = Field(
        description="Confidence level (0.0 to 1.0) in the reasoning process and the trust score of values of the accompanying fields.",
        ge=0.0,
        le=1.0,
    )
