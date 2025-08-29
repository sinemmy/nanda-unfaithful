#!/bin/bash
# Download results from vast.ai instance
set -euo pipefail

# Source .env to get SSH_KEY_PATH
source .env
SSH_KEY_PATH="${SSH_KEY_PATH/#\~/$HOME}"

# Arguments
VAST_IP="${1:-}"
VAST_PORT="${2:-}"
EXPERIMENT_TYPE="${3:-initial}"  # initial or followup

[[ -z "$VAST_IP" || -z "$VAST_PORT" ]] && {
    echo "Usage: $0 <ip> <port> [experiment_type]"
    exit 1
}

# Colors
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m'

echo -e "${GREEN}Downloading results from vast.ai instance...${NC}"

# Create local results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="./results_${EXPERIMENT_TYPE}_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

# Set remote directory based on experiment type
if [[ "$EXPERIMENT_TYPE" == "followup" ]]; then
    REMOTE_DIR="/workspace/nanda-unfaithful/outputs/full_analysis"
else
    REMOTE_DIR="/workspace/nanda-unfaithful/outputs/initial_test"
fi

# Download all outputs
echo "Downloading from $REMOTE_DIR to $RESULTS_DIR..."

# Use rsync for better handling of directories
rsync -avz -e "ssh -i $SSH_KEY_PATH -p $VAST_PORT" \
    "root@$VAST_IP:$REMOTE_DIR/" \
    "$RESULTS_DIR/" 2>/dev/null || {
    # Fallback to scp if rsync fails
    echo "Rsync failed, trying scp..."
    scp -i "$SSH_KEY_PATH" -P "$VAST_PORT" -r \
        "root@$VAST_IP:$REMOTE_DIR/*" \
        "$RESULTS_DIR/" 2>/dev/null || {
        echo -e "${YELLOW}Warning: Could not download all files${NC}"
    }
}

# Also try to download the main outputs directory if it exists
echo "Checking for additional output files..."
scp -i "$SSH_KEY_PATH" -P "$VAST_PORT" -r \
    "root@$VAST_IP:/workspace/nanda-unfaithful/outputs/unfaithful_cot/*" \
    "$RESULTS_DIR/" 2>/dev/null || true

echo
echo -e "${GREEN}✅ Results downloaded to: $RESULTS_DIR${NC}"
echo

# Show what was downloaded
echo "Downloaded files:"
ls -la "$RESULTS_DIR" 2>/dev/null || echo "No files found"

# Check for key result files
if [[ -f "$RESULTS_DIR/cot_examples.json" ]]; then
    echo
    echo -e "${GREEN}✅ Found main results file: cot_examples.json${NC}"
    echo "Quick summary:"
    python3 -c "
import json
with open('$RESULTS_DIR/cot_examples.json') as f:
    data = json.load(f)
    print(f'  Faithful examples: {len(data.get(\"faithful\", []))}')
    print(f'  Unfaithful examples: {len(data.get(\"unfaithful\", []))}')
    unfaithful = [e for e in data.get('unfaithful', []) if e.get('unfaithfulness', {}).get('is_unfaithful')]
    print(f'  Detected unfaithful: {len(unfaithful)}')
" 2>/dev/null || true
fi

echo
echo -e "${YELLOW}Don't forget to terminate the instance if you're done!${NC}"
echo "  vastai destroy instance [INSTANCE_ID]"