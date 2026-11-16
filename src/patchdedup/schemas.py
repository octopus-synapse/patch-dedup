"""Versioned public contracts; thresholds are always explicitly supplied."""

import unicodedata
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

Decision = Literal["redundant", "not_redundant", "review"]
Relation = Literal["redundant", "distinct", "partial_overlap", "contradictory", "ambiguous"]
Split = Literal["train", "validation", "test"]
Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Score = Annotated[float, Field(ge=-1, le=1, allow_inf_nan=False)]


def normalize_text(value: str) -> str:
    """Normalize NFC and whitespace without changing material facts."""
    return " ".join(unicodedata.normalize("NFC", value).split())


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TextPair(Contract):
    text_a: str
    text_b: str
    language: Literal["pt-BR"] = "pt-BR"

    @field_validator("text_a", "text_b")
    @classmethod
    def validate_text(cls, value: str) -> str:
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise PydanticCustomError("INVALID_UTF8", "Invalid Unicode text") from exc
        value = normalize_text(value)
        if not value:
            raise PydanticCustomError("EMPTY_TEXT", "Text must not be empty")
        if len(value) > 1500:
            raise PydanticCustomError("INPUT_TOO_LONG", "Text exceeds 1500 characters")
        return value


class CompareInput(TextPair):
    schema_version: Literal["0.1.0"] = "0.1.0"


class Thresholds(Contract):
    low: Score
    high: Score
    positive_enabled: bool = True
    negative_enabled: bool = True

    @model_validator(mode="after")
    def validate_order(self) -> "Thresholds":
        if self.low >= self.high:
            raise ValueError("low must be less than high")
        return self


class Similarity(Contract):
    metric: Literal["cosine"] = "cosine"
    score: Score


class DecisionScore(Contract):
    metric: Literal["uncalibrated_sigmoid"] = "uncalibrated_sigmoid"
    score: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class CompareOutput(Contract):
    schema_version: Literal["0.1.0", "0.2.0"] = "0.2.0"
    model_version: Identifier
    model_revision: Identifier
    decision_policy_version: Identifier
    language: Literal["pt-BR"] = "pt-BR"
    similarity: Similarity
    decision: Decision
    decision_source: Literal["cosine", "cross_encoder"] = "cosine"
    decision_score: DecisionScore | None = None
    decision_model_revision: Identifier | None = None
    thresholds: Thresholds
    requires_review: bool
    warnings: list[str] = Field(
        default_factory=lambda: ["semantic_similarity_is_not_factual_equivalence"]
    )

    @model_validator(mode="after")
    def validate_decision(self) -> "CompareOutput":
        from patchdedup.decision import decide

        score = self.similarity.score
        if self.decision_source == "cross_encoder":
            if (
                self.schema_version != "0.2.0"
                or self.decision_score is None
                or not self.decision_model_revision
            ):
                raise ValueError("Cross encoder requires schema 0.2.0, score and revision")
            score = self.decision_score.score
        elif self.decision_score is not None or self.decision_model_revision is not None:
            raise ValueError("Cosine policy cannot include a separate decision model")
        if self.decision != decide(score, self.thresholds):
            raise ValueError("Decision does not match score and thresholds")
        if self.requires_review != (self.decision == "review"):
            raise ValueError("requires_review must match decision")
        return self


class PairRecord(TextPair):
    pair_id: Identifier
    scenario_id: Identifier
    text_a_id: Identifier
    text_b_id: Identifier
    relation: Relation
    binary_label: Annotated[int, Field(strict=True, ge=0, le=1)] | None
    difficulty: Identifier
    domain: Identifier
    source_type: Literal["synthetic", "authorized"]
    review_status: Literal["unreviewed", "reviewed", "adjudicated"]
    annotation_version: Identifier
    split: Split

    @model_validator(mode="after")
    def validate_annotation(self) -> "PairRecord":
        expected = None if self.relation == "ambiguous" else int(self.relation == "redundant")
        if self.binary_label != expected:
            raise ValueError("binary_label does not match relation")
        if self.text_a_id == self.text_b_id:
            raise ValueError("Text IDs within a pair must differ")
        return self
