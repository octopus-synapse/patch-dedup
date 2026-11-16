import pytest

from patchdedup.assin import convert_assin, remove_overlap


def test_assin_only_paraphrase_is_positive_without_invented_direction():
    source = [
        dict(sentence_pair_id=i, premise=f"a{i}", hypothesis=f"b{i}", entailment_judgment=i)
        for i in range(3)
    ]
    rows = convert_assin(source, "train")
    assert [r["binary_label"] for r in rows] == [0, 0, 1]
    assert all(r["nli_supervision"] is False for r in rows)
    with pytest.raises(ValueError):
        convert_assin([dict(source[0], entailment_judgment=9)], "train")


def test_overlap_removes_reversed_and_shared_normalized_sentences():
    train = [dict(text_a=" UMA frase ", text_b="outro"), dict(text_a="novo", text_b="texto")]
    reserved = [dict(text_a="uma FRASE", text_b="reservado")]
    kept, removed = remove_overlap(train, reserved)
    assert kept == train[1:]
    assert removed == 1


def test_unknown_direction_produces_no_nli_gradient():
    torch = pytest.importorskip("torch")
    from patchdedup.nli import directional_supervision_loss

    ab = torch.tensor([[3.0, 0.0, 0.0]], requires_grad=True)
    ba = torch.tensor([[0.0, 3.0, 0.0]], requires_grad=True)
    loss = directional_supervision_loss(ab, ba, [dict(nli_supervision=False)], {}, 0.0)
    loss.backward()
    assert loss.item() == 0
    assert ab.grad.abs().sum() == 0 and ba.grad.abs().sum() == 0
