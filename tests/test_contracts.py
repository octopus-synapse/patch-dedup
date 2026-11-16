import math

import pytest
from pydantic import ValidationError

from patchdedup.decision import decide
from patchdedup.schemas import CompareInput, CompareOutput, PairRecord, Thresholds


def test_normalization_preserves_material_facts():
    value = CompareInput(text_a="  Na\u0303o\t reduzi 30%.  ", text_b="Usei C++ e Redis.")
    assert value.text_a == "Não reduzi 30%."
    assert value.text_b == "Usei C++ e Redis."
    assert value.schema_version == "0.1.0"


@pytest.mark.parametrize("text", ["", " \t\n", "a" * 1501, "\ud800"])
def test_invalid_text(text):
    with pytest.raises(ValidationError):
        CompareInput(text_a=text, text_b="B")


def test_character_limit_applies_after_normalization():
    assert len(CompareInput(text_a=" " + "a" * 1500 + " ", text_b="B").text_a) == 1500


@pytest.mark.parametrize("overrides", [{"language": "en"}, {"schema_version": "2"}, {"extra": 1}])
def test_contract_rejects_unsupported_values(overrides):
    with pytest.raises(ValidationError):
        CompareInput(text_a="A", text_b="B", **overrides)


@pytest.mark.parametrize(
    "low,high", [(0.8, 0.8), (0.9, 0.2), (-1.1, 0), (0, 1.1), (math.nan, 1), (0, math.inf)]
)
def test_invalid_thresholds(low, high):
    with pytest.raises(ValidationError):
        Thresholds(low=low, high=high)


@pytest.mark.parametrize(
    "score,expected",
    [
        (-1, "not_redundant"),
        (0.19, "not_redundant"),
        (0.2, "review"),
        (0.79, "review"),
        (0.8, "redundant"),
        (1, "redundant"),
    ],
)
def test_decision_boundaries(score, expected):
    assert decide(score, Thresholds(low=0.2, high=0.8)) == expected


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf, -1.001, 1.001])
def test_invalid_scores(score):
    with pytest.raises(ValueError, match="INVALID_SCORE"):
        decide(score, Thresholds(low=0.2, high=0.8))


def test_thresholds_have_no_operational_defaults():
    with pytest.raises(ValidationError):
        Thresholds()


def record(**overrides):
    return (
        dict(
            pair_id="p1",
            scenario_id="s1",
            text_a_id="a",
            text_b_id="b",
            text_a="Criei filas.",
            text_b="Implementei filas.",
            relation="redundant",
            binary_label=1,
            difficulty="paraphrase",
            domain="software",
            language="pt-BR",
            source_type="synthetic",
            review_status="unreviewed",
            annotation_version="0.1.0",
            split="train",
        )
        | overrides
    )


@pytest.mark.parametrize(
    "relation,label",
    [
        ("redundant", 1),
        ("distinct", 0),
        ("partial_overlap", 0),
        ("contradictory", 0),
        ("ambiguous", None),
    ],
)
def test_label_mapping(relation, label):
    assert PairRecord(**record(relation=relation, binary_label=label)).binary_label == label


@pytest.mark.parametrize(
    "overrides",
    [
        dict(relation="redundant", binary_label=0),
        dict(relation="ambiguous", binary_label=0),
        dict(binary_label=True),
        dict(pair_id=" "),
        dict(text_b_id="a"),
        dict(split="dev"),
        dict(source_type="scraped"),
    ],
)
def test_invalid_pair_record(overrides):
    with pytest.raises(ValidationError):
        PairRecord(**record(**overrides))


def output(**overrides):
    return (
        dict(
            model_version="0.1.0",
            model_revision="a" * 40,
            decision_policy_version="0.1.0",
            similarity={"metric": "cosine", "score": 0.5},
            decision="review",
            thresholds={"low": 0.2, "high": 0.8},
            requires_review=True,
            warnings=["semantic_similarity_is_not_factual_equivalence"],
        )
        | overrides
    )


def test_output_roundtrip():
    value = CompareOutput(**output())
    assert CompareOutput.model_validate_json(value.model_dump_json()) == value


@pytest.mark.parametrize(
    "overrides",
    [
        dict(requires_review=False),
        dict(decision="redundant"),
        dict(model_revision=""),
        dict(similarity={"metric": "cosine", "score": math.nan}),
    ],
)
def test_output_consistency(overrides):
    with pytest.raises(ValidationError):
        CompareOutput(**output(**overrides))


def test_disabled_policy_directions_abstain():
    assert decide(1, Thresholds(low=-1, high=1, positive_enabled=False)) == "review"
    assert decide(-1, Thresholds(low=0, high=1, negative_enabled=False)) == "review"
