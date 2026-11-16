#!/usr/bin/env bash
# Manual documentation-only uploads: no checkpoints, datasets or local folders.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
hf auth whoami
hf upload enzoferracini/patchdedup-ptbr-minilm \
  "$project_root/hub/model/README.md" README.md --type model \
  --commit-message 'docs: describe experimental model preparation status'
hf upload enzoferracini/patchdedup-synthetic-ptbr \
  "$project_root/hub/dataset/README.md" README.md --type dataset \
  --commit-message 'docs: describe dataset preparation and provenance requirements'
