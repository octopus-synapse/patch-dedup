#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
model_path="$(realpath "${1:-$project_root/artifacts/model}")"
image_name="${PATCHDEDUP_IMAGE:-patchdedup:experimental}"
output="${PATCHDEDUP_SMOKE_OUTPUT:-$project_root/artifacts/docker-smoke}"
bundled="${PATCHDEDUP_BUNDLED:-0}"
if [[ ! -f "$model_path/manifest.json" ]]; then
  echo 'Bundle ausente. Baixe a revisão de configs/release.json ou informe um bundle local.' >&2
  exit 2
fi
mkdir -p "$output"
tmp_root="$(mktemp -d)"
trap 'rm -rf -- "$tmp_root"' EXIT
if [[ "$bundled" == 1 ]]; then
  uv run --locked --project "$project_root" python -m patchdedup.deployment \
    --source "$model_path" --output "$tmp_root/checkpoint"
  docker build --target bundled --build-context "checkpoint=$tmp_root/checkpoint" \
    -t "$image_name" "$project_root"
else
  docker build --target runtime -t "$image_name" "$project_root"
fi
test "$(docker image inspect --format '{{.Config.User}}' "$image_name")" = '65532:65532'
run_args=(--rm --network none --read-only --tmpfs /tmp:rw,nosuid,size=256m
  --cap-drop ALL --security-opt no-new-privileges --pids-limit 256 --memory 4g --cpus 2)
if [[ "$bundled" != 1 ]]; then
  run_args+=(--mount "type=bind,source=$model_path,target=/model,readonly")
fi
cat "$project_root/examples/pairs.jsonl" > "$tmp_root/input.jsonl"
echo '{"text_a":" ","text_b":"B"}' >> "$tmp_root/input.jsonl"
if docker run -i "${run_args[@]}" "$image_name" batch --model /model --input - \
    < "$tmp_root/input.jsonl" > "$output/predictions.jsonl"; then
  batch_status=0
else
  batch_status=$?
fi
if [[ "$batch_status" != 1 ]]; then
  echo "Batch deveria retornar 1 pelo registro inválido; retornou $batch_status." >&2
  exit 1
fi
docker run "${run_args[@]}" "$image_name" compare --model /model \
  --text-a 'Automatizei relatórios com Python.' \
  --text-b 'Usei Python para automatizar relatórios.' > "$output/compare.json"
docker image inspect --format '{{.Id}}' "$image_name" > "$output/image-id.txt"
uv run --locked --project "$project_root" python - "$output" "$model_path" "$bundled" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
from patchdedup.schemas import CompareOutput

output, model = map(Path, sys.argv[1:3])
rows = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
assert [row["line"] for row in rows] == [1, 2, 3, 4]
results = [CompareOutput.model_validate(row["result"]) for row in rows[:3]]
assert rows[3]["error"]["code"] == "EMPTY_TEXT"
assert results[0].decision == results[1].decision
assert abs(results[0].similarity.score - results[1].similarity.score) < 1e-6
assert abs(results[0].decision_score.score - results[1].decision_score.score) < 1e-6
assert all(result.similarity.metric == "cosine" for result in results)
assert all(result.decision_score.metric == "pair_score" for result in results)
comparison = CompareOutput.model_validate_json((output / "compare.json").read_text())
assert comparison.decision == results[0].decision
assert abs(comparison.decision_score.score - results[0].decision_score.score) < 1e-6
summary = dict(
    image_id=(output / "image-id.txt").read_text().strip(),
    bundled=sys.argv[3] == "1", network="none", read_only=True, non_root=True,
    cpu_limit=2, memory_limit="4g", model_metadata=json.loads((model / "patchdedup.json").read_text()),
    model_manifest_sha256=hashlib.sha256((model / "manifest.json").read_bytes()).hexdigest(),
    valid_predictions=len(results), expected_invalid_records=1,
    decisions=[result.decision for result in results],
    symmetry_verified=True, compare_matches_batch=True,
    examples="constructed smoke examples; not an ML benchmark",
)
(output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print("Docker validado: modelo real offline, não root, lote, simetria e erro por registro.")
print(f"Evidências: {output}")
PY
docker image inspect --format 'Image size: {{.Size}} bytes' "$image_name"
