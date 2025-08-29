#!/bin/bash
# Wrapper script to run commands on remote with SSH key from .env
set -euo pipefail

# Source .env to get SSH_KEY_PATH
source .env
SSH_KEY_PATH="${SSH_KEY_PATH/#\~/$HOME}"

# Get connection details
VAST_IP="${1:-}"
VAST_PORT="${2:-}"
shift 2

# Run command on remote
ssh -i "$SSH_KEY_PATH" -p "$VAST_PORT" "root@$VAST_IP" "$@"