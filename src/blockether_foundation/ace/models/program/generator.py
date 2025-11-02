from pydantic import BaseModel, Field
from ..base import ChainOfThoughts


class GeneratorSectionEntryUsefulnessAnalysis(ChainOfThoughts):
    impact_score: float = Field(
        description="Impact score (0.0 to 1.0) indicating how much this entry contributed to the generator's output.",
        ge=0.0,
        le=1.0,
    )

    ground_truth_id: str = Field(
        description="Identifier of the ground truth entry being evaluated for usefulness."
    )

    justification: str = Field(
        description="Justification for the usefulness score in markdown format."
    )


class GeneratorOutput[T: BaseModel | str](ChainOfThoughts):
    answer: T = Field(description="The generated answer or output produced by the generator model.")
    ground_truths_used: list[GeneratorSectionEntryUsefulnessAnalysis] = Field(
        description="List of ground truth entries used in the generation process. If the specific ground truth was not used omit it from the list."
    )
