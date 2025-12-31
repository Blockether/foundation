"""
Core model classes with serialization support.
"""

from __future__ import annotations

import base64
import datetime
import json
import pickle
from enum import Enum

from pydantic import BaseModel, Field

from .utils import generate_secure_id

__all__ = ["BaseModelSerializable", "ChainOfThoughts", "Goal", "GoalPriority", "GoalStatus"]


class BaseModelSerializable(BaseModel):
    """
    Base model with comprehensive serialization support.
    Provides JSON, pickle, base64, and file persistence methods.
    """

    @classmethod
    def from_json_file(cls, file_path: str) -> BaseModelSerializable:
        """Load from JSON file."""
        with open(file_path) as f:
            data = json.load(f)
        return cls(**data)

    def to_json_file(self, file_path: str) -> None:
        """Save to JSON file."""
        with open(file_path, "w") as f:
            json.dump(self.model_dump(), f, indent=4)

    @classmethod
    def from_pickle_file(cls, file_path: str) -> BaseModelSerializable:
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
    def from_json_string(cls, json_str: str) -> BaseModelSerializable:
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls(**data)

    def to_base64(self) -> str:
        """Serialize to base64-encoded pickle string."""
        pickled = pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
        return base64.b64encode(pickled).decode("utf-8")

    @classmethod
    def from_base64(cls, base64_str: str) -> BaseModelSerializable:
        """Deserialize from base64-encoded pickle string."""
        pickled = base64.b64decode(base64_str.encode("utf-8"))
        return pickle.loads(pickled)


class ChainOfThoughts(BaseModelSerializable):
    reasoning: str | None = Field(
        default=None,
        description="Step-by-step reasoning process explaining how you reached your conclusion. Include relevant context, step-by-step analysis including the rationale for the importance and confidence scores. Prefer concise and clear explanations. Omit reasoning for simple or self-evident observations. Plain text only, no markdown.",
    )

    importance: float | None = Field(
        default=None,
        description="Importance score (0.0 to 1.0) indicating how critical this observation is to the overall task or decision-making process.",
        ge=0.0,
        le=1.0,
    )

    confidence: float = Field(
        default=0.5,
        description="Confidence level (0.0 to 1.0) in the reasoning process and/or the trust score of values of the accompanying fields.",
        ge=0.0,
        le=1.0,
    )


class GoalStatus(str, Enum):
    """Status states for goal lifecycle management."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    REFINED = "refined"


class GoalPriority(str, Enum):
    """Priority levels for goal ordering."""

    HIGHEST = "P0"
    HIGH = "P1"
    MEDIUM = "P2"
    LOW = "P3"


class Goal(ChainOfThoughts):
    """
    Structured goal representation for task management.
    Supports hierarchical decomposition with subgoals, priority levels, and lifecycle management.
    """

    id: str = Field(
        default_factory=lambda: generate_secure_id(6),
        description="Unique identifier for this goal. Should never be assigned by LLM.",
    )

    title: str = Field(
        description="Short descriptive title for the goal.",
    )

    description: str = Field(
        description="Detailed description of what needs to be accomplished.",
    )

    summary: str | None = Field(
        default=None,
        description="Concise summary of the goal (1-2 sentences). Used for quick overview.",
    )

    priority: GoalPriority = Field(
        default=GoalPriority.MEDIUM,
        description="Priority level (P0=highest, P1=high, P2=medium, P3=low).",
    )

    status: GoalStatus = Field(
        default=GoalStatus.PENDING,
        description="Current status: 'pending', 'in_progress', 'completed', 'skipped', 'refined'.",
    )

    subgoals: list[Goal] = Field(
        default_factory=list["Goal"],
        description="List of subgoals that break this goal into smaller, manageable tasks.",
    )

    parent_goal_id: str | None = Field(
        default=None,
        description="ID of the parent goal if this is a subgoal.",
    )

    created_at: str | None = Field(
        default=datetime.datetime.now(datetime.UTC).isoformat(),
        description="ISO timestamp of when this goal was created.",
    )

    updated_at: str | None = Field(
        default=datetime.datetime.now(datetime.UTC).isoformat(),
        description="ISO timestamp of when this goal was last updated.",
    )

    def is_leaf(self) -> bool:
        return len(self.subgoals) == 0

    def get_all_subgoal_ids(self) -> list[str]:
        ids: list[str] = [g.id for g in self.subgoals]
        for subgoal in self.subgoals:
            ids.extend(subgoal.get_all_subgoal_ids())
        return ids

    def find_subgoal_by_id(self, goal_id: str) -> Goal | None:
        for subgoal in self.subgoals:
            if subgoal.id == goal_id:
                return subgoal
            found = subgoal.find_subgoal_by_id(goal_id)
            if found:
                return found
        return None
