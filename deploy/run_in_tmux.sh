#!/bin/bash
# Run experiment in tmux session on vast.ai instance
set -euo pipefail

# Source .env to get SSH_KEY_PATH
source .env
SSH_KEY_PATH="${SSH_KEY_PATH/#\~/$HOME}"

# Arguments
VAST_IP="${1:-}"
VAST_PORT="${2:-}"
EXPERIMENT_TYPE="${3:-initial}"  # initial or followup
SESSION_NAME="experiment"

# Colors
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

usage() {
    echo "Usage: $0 <ip> <port> [experiment_type]"
    echo "  experiment_type: 'initial' (quick, 3 examples) or 'followup' (full, 10 examples)"
    exit 1
}

[[ -z "$VAST_IP" || -z "$VAST_PORT" ]] && usage

echo -e "${BLUE}Starting experiment in tmux session...${NC}"

# First, pull latest code
echo -e "${GREEN}Pulling latest code from GitHub...${NC}"
ssh -i "$SSH_KEY_PATH" -p "$VAST_PORT" "root@$VAST_IP" << 'EOF'
cd /workspace/nanda-unfaithful
git pull origin main
EOF

# Set experiment parameters based on type
if [[ "$EXPERIMENT_TYPE" == "followup" ]]; then
    EXPERIMENT_ARGS="--num-samples 5 --verbose"
    EXPERIMENT_DESC="Full run: 6 problems × 5 variations × 5 samples"
else
    EXPERIMENT_ARGS="--num-samples 2 --problem-range 0-1 --verbose"
    EXPERIMENT_DESC="Quick test: 2 problems × 5 variations × 2 samples"
fi

# Start tmux session and run experiment
echo -e "${GREEN}Starting tmux session: $SESSION_NAME${NC}"
ssh -i "$SSH_KEY_PATH" -p "$VAST_PORT" "root@$VAST_IP" << EOF
# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null || true

# Create new tmux session and run experiment
tmux new-session -d -s $SESSION_NAME -c /workspace/nanda-unfaithful

# Send commands to the session
tmux send-keys -t $SESSION_NAME "cd /workspace/nanda-unfaithful" C-m
tmux send-keys -t $SESSION_NAME "echo '=================================='" C-m
tmux send-keys -t $SESSION_NAME "echo 'Starting $EXPERIMENT_TYPE experiment'" C-m
tmux send-keys -t $SESSION_NAME "echo '$EXPERIMENT_DESC'" C-m
tmux send-keys -t $SESSION_NAME "echo '=================================='" C-m
tmux send-keys -t $SESSION_NAME "source .venv/bin/activate" C-m
tmux send-keys -t $SESSION_NAME "python run_comparison.py $EXPERIMENT_ARGS" C-m
EOF

echo
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}           EXPERIMENT STARTED IN TMUX${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo
echo -e "${GREEN}The experiment is now running in tmux session: $SESSION_NAME${NC}"
echo
echo -e "${YELLOW}To monitor progress:${NC}"
echo "  ssh -i ~/.ssh/vast_ai_key -p $VAST_PORT root@$VAST_IP"
echo "  tmux attach -t $SESSION_NAME"
echo
echo -e "${YELLOW}To detach from tmux (keep it running):${NC}"
echo "  Press: Ctrl+B, then D"
echo
echo -e "${YELLOW}To check if still running:${NC}"
echo "  ./deploy/check_tmux.sh $VAST_IP $VAST_PORT"
echo
echo -e "${YELLOW}To download results when done:${NC}"
echo "  ./deploy/download_results.sh $VAST_IP $VAST_PORT"
echo