#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
export PATCHDEDUP_HF_CLI="${PATCHDEDUP_HF_CLI:-$HOME/.local/bin/hf}"
uv_bin="${PATCHDEDUP_UV:-$HOME/.local/bin/uv}"
background=false
arguments=()
for arg in "$@"; do
  if [[ "$arg" == '--background' ]]; then background=true; else arguments+=("$arg"); fi
done
if $background; then
  mkdir -p artifacts/v03/orchestration
  nohup bash "$project_root/scripts/finish_remaining.sh" "${arguments[@]}" >> artifacts/v03/orchestration/runner.log 2>&1 < /dev/null &
  runner_pid=$!
  echo "Executor iniciado: PID $runner_pid"
  echo "Log: $project_root/artifacts/v03/orchestration/runner.log"
  echo "Status: $project_root/artifacts/v03/orchestration/status.json"
  echo 'O computador precisa continuar ligado e sem suspender.'
  exit 0
fi
if [[ ! -x "$uv_bin" ]]; then
  echo 'uv não encontrado. Instale uv ou defina PATCHDEDUP_UV.' >&2
  exit 2
fi
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
exec "$uv_bin" run --locked --extra ml python scripts/finish_remaining.py "${arguments[@]}"
