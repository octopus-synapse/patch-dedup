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
  if systemctl --user is-active --quiet patchdedup-assin-refinement.service; then
    echo 'O refinamento ASSIN já está rodando; nenhum processo duplicado foi iniciado.'
    exit 0
  fi
  mkdir -p artifacts/assin-refinement
  systemd-run --user --collect --unit=patchdedup-assin-refinement \
    --property="WorkingDirectory=$project_root" \
    --property="StandardOutput=append:$project_root/artifacts/assin-refinement/runner.log" \
    --property=StandardError=inherit \
    /usr/bin/bash "$project_root/scripts/refine_assin.sh" "${arguments[@]}"
  echo "Status: $project_root/artifacts/assin-refinement/status.json"
  echo 'O computador precisa continuar ligado e sem suspender.'
  exit 0
fi
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "$uv_bin" run --locked --extra ml python scripts/refine_assin.py "${arguments[@]}"
