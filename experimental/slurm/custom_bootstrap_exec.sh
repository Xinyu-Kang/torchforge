#!/usr/bin/env bash
set -euo pipefail

# Wrapper for Monarch Slurm workers that runs the bootstrap Python inside Docker.
# Usage: set `provisioner.python_exe` to this script.
#
# Optional env:
#   FORGE_DOCKER_IMAGE           Docker image to run (default: rocm/pytorch-private:torchforge-deps-rocm7.1-20260205-v1)
#
#   FORGE_DOCKER_WORKDIR         Container working directory (default: /workspace/torchforge)
#   FORGE_DOCKER_MOUNT           Host path to mount at FORGE_DOCKER_WORKDIR (default: current working dir)
#   FORGE_DOCKER_PYTHON          Python binary inside container (default: /opt/venv/bin/python)
#   FORGE_DOCKER_SITE_PACKAGES   Site-packages path to prepend to PYTHONPATH
#                               (default: /opt/venv/lib/python3.12/site-packages)
#   FORGE_DOCKER_RUN_ARGS        Extra args for `docker run` (space-separated, no shell quoting)
#   FORGE_DOCKER_CLEANUP         If "1", stop any previous worker containers
#                               started by this script for the same user (default: 1)
#   FORGE_DOCKER_INSTALL         If "1", run pip install inside container (default: 0)

IMAGE_DEFAULT="rocm/pytorch-private:torchforge-deps-rocm7.1-20260205-v1"
FORGE_DOCKER_IMAGE="${FORGE_DOCKER_IMAGE:-$IMAGE_DEFAULT}"

WORKDIR="${FORGE_DOCKER_WORKDIR:-/workspace/torchforge}"
HOST_MOUNT="${FORGE_DOCKER_MOUNT:-$PWD}"
PYTHON_BIN="${FORGE_DOCKER_PYTHON:-/opt/venv/bin/python}"
SITE_PACKAGES="${FORGE_DOCKER_SITE_PACKAGES:-/opt/venv/lib/python3.12/site-packages}"

CLEANUP="${FORGE_DOCKER_CLEANUP:-1}"
INSTALL="${FORGE_DOCKER_INSTALL:-0}"

HF_USER="${USER:-$(id -un)}"
HF_CACHE="/home/${HF_USER}/.cache/huggingface"

log() {
  echo "[forge-docker] $*" >&2
}

EXTRA_RUN_ARGS=()
if [[ -n "${FORGE_DOCKER_RUN_ARGS:-}" ]]; then
  read -r -a EXTRA_RUN_ARGS <<< "${FORGE_DOCKER_RUN_ARGS}"
fi

HOSTNAME_FQDN="$(hostname -f 2>/dev/null || hostname)"

SOURCE_PATH="${WORKDIR}/src"
if [[ -n "${PYTHONPATH:-}" ]]; then
  PYTHONPATH="${SITE_PACKAGES}:${SOURCE_PATH}:${PYTHONPATH}"
else
  PYTHONPATH="${SITE_PACKAGES}:${SOURCE_PATH}"
fi
export PYTHONPATH

INSTALL_CMD=':'
if [[ "$INSTALL" == "1" ]]; then
  INSTALL_CMD='$FORGE_DOCKER_PYTHON -m pip install . --no-deps'
fi
RUN_CMD='exec "$FORGE_DOCKER_PYTHON" "$@"'

if [[ "$CLEANUP" == "1" ]]; then
  # Stop any running containers on this host to avoid port conflicts.
  DOCKER_PS_OUTPUT="$(docker ps --format '{{.ID}} {{.Image}} {{.Names}}' 2>&1 || true)"
  if [[ -z "$DOCKER_PS_OUTPUT" ]]; then
    log "No running containers to stop."
  elif [[ "$DOCKER_PS_OUTPUT" == *"Cannot connect to the Docker daemon"* ]]; then
    log "docker ps failed: $DOCKER_PS_OUTPUT"
  else
    log "Stopping existing containers:"
    log "$DOCKER_PS_OUTPUT"
    echo "$DOCKER_PS_OUTPUT" | awk '{print $1}' | xargs -r docker stop >&2 || true
  fi
fi

# Ensure HF cache path exists on host before mounting.
mkdir -p "$HF_CACHE"

exec docker run --rm \
  --hostname "$HOSTNAME_FQDN" \
  --device /dev/dri --device /dev/kfd \
  --network host --ipc host \
  --group-add video \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  --privileged \
  -v "$HOST_MOUNT:$WORKDIR" \
  -v "$HF_CACHE:$HF_CACHE" \
  -w "$WORKDIR" \
  -e FORGE_DOCKER_PYTHON="$PYTHON_BIN" \
  -e HF_HOME="$HF_CACHE" \
  -e PYTHONPATH \
  "${EXTRA_RUN_ARGS[@]}" \
  --entrypoint bash \
  "$FORGE_DOCKER_IMAGE" \
  -lc "$INSTALL_CMD && $RUN_CMD" -- "$@"
