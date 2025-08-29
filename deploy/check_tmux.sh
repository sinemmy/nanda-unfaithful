#!/bin/bash
# Check tmux session status on vast.ai instance
set -euo pipefail

# Source .env to get SSH_KEY_PATH
source .env
SSH_KEY_PATH="${SSH_KEY_PATH/#\~/$HOME}"

# Arguments
VAST_IP="${1:-}"
VAST_PORT="${2:-}"
SESSION_NAME="${3:-experiment}"

[[ -z "$VAST_IP" || -z "$VAST_PORT" ]] && {
    echo "Usage: $0 <ip> <port> [session_name]"
    exit 1
}

echo "Checking tmux session '$SESSION_NAME' on $VAST_IP:$VAST_PORT..."
echo

ssh -i "$SSH_KEY_PATH" -p "$VAST_PORT" "root@$VAST_IP" << EOF
# Check if tmux session exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "✅ Session '$SESSION_NAME' is running"
    echo
    echo "Recent output:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    # Capture last 20 lines from the session
    tmux capture-pane -t $SESSION_NAME -p | tail -20
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo
    # Check if process is still active
    if tmux list-panes -t $SESSION_NAME -F '#{pane_current_command}' | grep -q python; then
        echo "🔄 Python process is still running"
    else
        echo "✅ Python process has completed"
        echo "Run ./deploy/download_results.sh $VAST_IP $VAST_PORT to get results"
    fi
else
    echo "❌ No session named '$SESSION_NAME' found"
    echo "Available sessions:"
    tmux list-sessions 2>/dev/null || echo "No tmux sessions running"
fi
EOF