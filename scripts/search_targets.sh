#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
uv_bin="${PATCHDEDUP_UV:-$HOME/.local/bin/uv}"
background=false
arguments=()
for arg in "$@"; do
  if [[ "$arg" == '--background' ]]; then background=true; else arguments+=("$arg"); fi
done
if $background; then
  mkdir -p artifacts/target-search
  nohup bash "$project_root/scripts/search_targets.sh" "${arguments[@]}" >> artifacts/target-search/runner.log 2>&1 < /dev/null &
  echo "Executor iniciado: PID $!"
  echo "Status: $project_root/artifacts/target-search/status.json"
  echo "Log: $project_root/artifacts/target-search/runner.log"
  exit 0
fi
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "$uv_bin" run --locked --extra ml python scripts/search_targets.py "${arguments[@]}"
