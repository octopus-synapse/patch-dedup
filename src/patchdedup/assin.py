"""ASSIN binary equivalence labels and exact text leakage prevention."""


def convert_assin(source, split):
    rows = []
    for record in source:
        label = record["entailment_judgment"]
        if label not in (0, 1, 2):
            raise ValueError("Unknown ASSIN label")
        rows.append(
            dict(
                pair_id=f"assin-ptbr-{split}-{record['sentence_pair_id']}",
                scenario_id=f"assin-ptbr-{split}-{record['sentence_pair_id']}",
                text_a=record["premise"],
                text_b=record["hypothesis"],
                binary_label=int(label == 2),
                relation="redundant" if label == 2 else "distinct",
                nli_supervision=False,
                source="nilc-nlp/assin",
                original_label=label,
            )
        )
    return rows


def remove_overlap(rows, reserved):
    """Exclude shared normalized sentences, including reversed pairs."""

    def normalized(text):
        return " ".join(text.casefold().split())

    seen = {normalized(r[k]) for r in reserved for k in ("text_a", "text_b")}
    kept = [r for r in rows if not any(normalized(r[k]) in seen for k in ("text_a", "text_b"))]
    return kept, len(rows) - len(kept)
