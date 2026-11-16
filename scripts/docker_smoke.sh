#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
model_path="$(realpath "${1:-$project_root/artifacts/run-42/selected-model}")"
image_name="${PATCHDEDUP_IMAGE:-patchdedup:experimental}"
docker build -t "$image_name" "$project_root"
test "$(docker image inspect --format '{{.Config.User}}' "$image_name")" = '65532:65532'
docker run --rm --network none --read-only --tmpfs /tmp \
  --mount "type=bind,source=$model_path,target=/model,readonly" \
  "$image_name" compare --model /model \
  --text-a 'Automatizei relatórios com Python.' \
  --text-b 'Usei Python para automatizar relatórios.'
docker image inspect --format 'Image size: {{.Size}} bytes' "$image_name"
