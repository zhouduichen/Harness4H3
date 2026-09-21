#!/usr/bin/env bash
set -euo pipefail

# This installer only copies one named custom node.  It never restarts
# ComfyUI, vLLM, Geneval, or the campaign service; the caller decides when a
# ComfyUI process owned by the campaign can be recycled.
SSH_TARGET="${H3_SSH_TARGET:-Jiayu-intern}"
REMOTE_COMFYUI_ROOT="${H3_REMOTE_COMFYUI_ROOT:-/home/intern/huangjiahao/ComfyUI}"
REMOTE_PYTHON="${H3_REMOTE_PYTHON:-/home/intern/miniconda3/envs/comfy/bin/python}"
LOCAL_SOURCE="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/comfyui_h3_optimizations.py"
REMOTE_DIR="${REMOTE_COMFYUI_ROOT%/}/custom_nodes"
REMOTE_FILE="${REMOTE_DIR}/harness4h3_h3_optimizations.py"
STAMP="$(date +%Y%m%d-%H%M%S)"
REMOTE_BACKUP="${REMOTE_FILE}.bak-${STAMP}"

if [[ ! -f "$LOCAL_SOURCE" ]]; then
    echo "missing local extension: $LOCAL_SOURCE" >&2
    exit 1
fi

ssh "$SSH_TARGET" "mkdir -p '$REMOTE_DIR'; if test -e '$REMOTE_FILE'; then cp -p '$REMOTE_FILE' '$REMOTE_BACKUP'; fi"
scp "$LOCAL_SOURCE" "$SSH_TARGET:$REMOTE_FILE"
ssh "$SSH_TARGET" "'$REMOTE_PYTHON' -m py_compile '$REMOTE_FILE'"

echo "installed $REMOTE_FILE"
if ssh "$SSH_TARGET" "test -f '$REMOTE_BACKUP'"; then
    echo "previous copy backed up at $REMOTE_BACKUP"
fi
echo "ComfyUI was not restarted; restart only an explicitly campaign-owned process after validation."
