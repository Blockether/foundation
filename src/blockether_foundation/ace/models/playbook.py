from abc import abstractmethod
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from .base import ChainOfThoughts

# Metadata statistics for tracking entry quality and impact
# helpful: Content that provides value, solves problems, or improves understanding
# harmful: Content that causes confusion, misleads, or has negative impact
# neutral: Content that is factual but neither particularly helpful nor harmful
type EntryMetadataStatistic = Literal["helpful", "harmful", "neutral"]


class PatternSituation(BaseModel):
    before: str = Field(
        description="Description of the situation before applying the pattern in markdown format"
    )
    context: str = Field(
        description="Detailed description of the situation in markdown format (usage context)"
    )
    after: str = Field(
        description="Description of the situation after applying the pattern in markdown format"
    )
    rationale: str = Field(
        description="Rationale why the pattern was effective or ineffective in this situation in markdown format"
    )


class PlaybookEntryDelta(ChainOfThoughts):
    entry_id: int = Field(description="ID of the playbook entry to apply the delta to")
    change_type: Literal["add", "update", "remove"] = Field(
        description="Type of change to apply to the entry"
    )
    change_attributes: dict[str, str] = Field(description="Attributes to update for the entry")
    entry_type: Literal["domain_knowledge", "ground_truth"] = Field(
        description="Type of the playbook entry"
    )


class PlaybookHighLevelOverview(BaseModel):
    description: str = Field(description="Description of the context in markdown format")

    def entry_to_markdown(self) -> str:
        return f"""
            ## Playbook Overview

            ### How to use?

            Playbook in general is a persisted structure to dynamically adapt the capabilities of the agent based on the conversation with the user.

            **Playbook contains the following primitives**:
            - Sections - sections group related entries together,
            - Entries - entries are individual pieces of knowledge, guidelines, patterns or hypotheses and set of entries form the section.
            - Each entry is formatted in the following way:

            (<identifier: IDENTIFIER> | <metadata: METADATA[helpful: HELPFUL_COUNTER, harmful: HARMFUL_COUNTER, neutral: NEUTRAL_COUNTER]>) -- <ENTRY_CONTENT>

            Where:
                - `IDENTIFIER` is a unique identifier for the entry,
                - `METADATA` is a set of three tags where to each one a counter is associated,
                - `helpful` (positive) - positive impact/true steering - how useful the entry is in between agent calls,
                - `harmful` (negative) - negative impact/false steering - how misleading the entry is between the agent calls,
                - `neutral` (neutral) - trust score - how reliable the entry is in between agent calls,
                - `ENTRY_CONTENT` is the actual content of the entry in markdown format.

            ### What this specific playbook is about?
            {self.description}"""


class BaseSectionEntry(BaseModel):
    id: int = Field(
        description="Unique identifier for the entry. Must be unique within the entire system."
    )
    section: str = Field(
        description="Section name that categorizes this entry. Used for grouping related entries."
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when entry was created. Uses UTC timezone.",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when entry was last modified. Uses UTC timezone.",
    )
    metadata: dict[EntryMetadataStatistic, int] = Field(
        default_factory=dict,
        description="Dictionary tracking user feedback statistics. Keys are 'helpful', 'harmful', 'neutral' with integer counts representing user ratings.",
    )

    @abstractmethod
    def entry_to_markdown(self) -> str:
        pass

    def metadata_to_markdown(self) -> str:
        return "[{}]".format(
            ", ".join(
                f"{key}: {self.metadata[key]}"
                for key in ["helpful", "harmful", "neutral"]
                if key in self.metadata
            )
        )

    def to_markdown(self) -> str:
        return f"(<identifier: {self.id}> | <metadata: {self.metadata_to_markdown()}>) -- {self.entry_to_markdown()}"


class DomainKnowledge(BaseSectionEntry):
    content: str = Field(description="Domain knowledge content in pure markdown format.")
    domain: str = Field(description="Domain to which the knowledge applies. Always lowercased.")


class GroundTruthProof(BaseModel):
    title: str = Field(description="Title of the proof")
    description: str = Field(description="Description of the proof in markdown format")
    source: str = Field(description="Source of the proof in markdown format")
    confidence: float = Field(
        description="Confidence level of the proof (0.0 to 1.0)", ge=0.0, le=1.0
    )

    def proof_to_markdown(self) -> str:
        return f"""
        - {self.title} (source: {self.source}, confidence: {self.confidence})
          {self.description}"""


class GroundTruth(BaseSectionEntry):
    title: str = Field(description="Title of the ground truth entry")
    content: str = Field(description="Ground truth content in markdown format.")
    proofs: list[GroundTruthProof] = Field(
        description="List of proofs supporting the ground truth."
    )

    def entry_to_markdown(self) -> str:
        return f"""
        ## {self.title}

        {self.content}

        {"### Supporting Proofs" if self.proofs else "No supporting proofs available."}
        {"\n".join([proof.proof_to_markdown() for proof in self.proofs])}"""


# class Guideline(BaseEntry):
#     content: str = Field(description="Guideline content in markdown format. Concise and clear.")
#     why: str = Field(description="Reasoning why this guideline is in place in markdown format.")


# class VerificationToolFunction(BaseEntry):
#     name: str = Field(description="Name of the verification function")
#     when: str = Field(description="Description of when to use the verification function")
#     input_json_schema: json_schema.JsonSchemaValue = Field(
#         description="JSON schema defining the expected input structure of the verification function"
#     )
#     output_json_schema: json_schema.JsonSchemaValue = Field(
#         description="JSON schema defining the expected output structure of the verification function"
#     )
#     implementation: str = Field(
#         description="Code snippet in Python language implementing the verification function"
#     )


# class Pattern(BaseEntry):
#     section: str = "Patterns"
#     order: int = 3
#     pattern: str = Field(description="Description/content of the pattern in markdown format")
#     when_use: list[PatternSituation] = Field(
#         description="List of situations where the pattern was effective and should be used in the future"
#     )
#     when_not_use: list[PatternSituation] = Field(
#         description="List of situations where the pattern was ineffective and should not be used in the future"
#     )
#     name: str = Field(description="Name of the pattern")


# class Hypothesis(BaseBulletEntry):
#     section: str = "hypothesis"
#     order: int = 3
#     content: str = Field(description="Hypothesis content")
#     reasoning: str = Field(description="Reasoning behind the hypothesis")


type SectionEntry = GroundTruth
