#!/usr/bin/env bash
set -euo pipefail

# Secure local entry point for the remote OpenAI-compatible Controller.
# The vLLM server stays bound to remote localhost; this forwards it to the
# local machine without opening port 8000 on the remote network interface.
llm_ssh_host="${CONTROLLER_SSH_HOST:-Jiayu-intern}"
llm_local_port="${CONTROLLER_LOCAL_PORT:-18000}"
llm_remote_port="${CONTROLLER_REMOTE_PORT:-8000}"

exec ssh \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=60 \
  -o ServerAliveCountMax=3 \
  -N \
  -L "${llm_local_port}:127.0.0.1:${llm_remote_port}" \
  "${llm_ssh_host}"
