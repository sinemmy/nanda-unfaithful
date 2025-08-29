#!/bin/bash
# Script to find and create a vast.ai instance for unfaithfulness experiments
# Usage: ./deploy/create_vast_instance.sh

set -euo pipefail

# Colors
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

# Configuration
readonly MIN_DISK_SPACE=80  # Need ~30GB for model + workspace + outputs
readonly PREFERRED_GPUS=("RTX_4090" "RTX_3090" "A5000" "A4000")
readonly IMAGE="pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime"
readonly MAX_PRICE_PER_HOUR=1.0

# Functions
die() {
    echo -e "${RED}Error: $1${NC}" >&2
    exit 1
}

info() {
    echo -e "${GREEN}$1${NC}"
}

warn() {
    echo -e "${YELLOW}$1${NC}"
}

check_requirements() {
    # Check if vastai CLI is installed
    if ! command -v vastai &>/dev/null; then
        warn "vastai CLI not found. Installing..."
        pip install vastai || die "Failed to install vastai CLI"
    fi
    
    # Check for .env file
    if [[ ! -f ".env" ]]; then
        die ".env file not found. Copy .env.example to .env and add your VAST_AI_API_KEY"
    fi
    
    # Load and check API key
    source .env
    if [[ -z "${VAST_AI_API_KEY:-}" ]]; then
        die "VAST_AI_API_KEY not found in .env file"
    fi
    
    # Configure vastai with API key
    vastai set api-key "$VAST_AI_API_KEY" || die "Failed to set vast.ai API key"
    
    # Verify authentication
    vastai show user &>/dev/null || die "Failed to authenticate with vast.ai"
    info "✅ Authentication successful"
}

search_instances() {
    info "Searching for available GPU instances..."
    
    local search_results=""
    local gpu_list=""
    
    # Build GPU search query
    for gpu in "${PREFERRED_GPUS[@]}"; do
        if [[ -z "$gpu_list" ]]; then
            gpu_list="gpu_name=$gpu"
        else
            gpu_list="$gpu_list gpu_name=$gpu"
        fi
    done
    
    # Search for instances
    echo -e "\n${BLUE}Available instances (sorted by price):${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Build the query with GPU list
    local query="rentable=true disk_space>$MIN_DISK_SPACE dph<$MAX_PRICE_PER_HOUR reliability>0.99 gpu_name in [RTX_4090,RTX_3090,A5000,A4000]"
    
    # Use vastai search with specific criteria
    vastai search offers "$query" \
        --order "dph" \
        --limit 10 | head -20
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

select_instance() {
    echo
    echo -e "${YELLOW}Enter the ID of the instance you want to rent:${NC}"
    echo "(Look for the ID in the leftmost column above)"
    read -r instance_id
    
    if [[ -z "$instance_id" ]]; then
        die "No instance ID provided"
    fi
    
    echo "$instance_id"
}

create_instance() {
    local instance_id=$1
    
    info "Creating instance $instance_id..."
    
    # Create the instance with SSH access
    local output=$(vastai create instance "$instance_id" \
        --image "$IMAGE" \
        --disk "$MIN_DISK_SPACE" \
        --ssh --direct \
        2>&1)
    
    echo "$output"
    
    # Extract the new instance ID from output
    local new_instance=$(echo "$output" | grep -oE "new_id: [0-9]+" | awk '{print $2}')
    
    if [[ -z "$new_instance" ]]; then
        die "Failed to create instance. Output: $output"
    fi
    
    info "✅ Instance created with ID: $new_instance"
    echo "$new_instance"
}

wait_for_instance() {
    local instance_id=$1
    
    info "Waiting for instance to be ready (this may take 2-3 minutes)..."
    
    local max_attempts=60
    local attempt=0
    
    while [[ $attempt -lt $max_attempts ]]; do
        # Get instance status
        local status=$(vastai show instance "$instance_id" --raw 2>/dev/null | \
            python3 -c "import sys, json; data = json.load(sys.stdin); print(data.get('actual_status', 'unknown'))" 2>/dev/null || echo "unknown")
        
        if [[ "$status" == "running" ]]; then
            info "✅ Instance is running!"
            return 0
        fi
        
        echo -n "."
        sleep 5
        ((attempt++))
    done
    
    die "Instance failed to start after 5 minutes"
}

get_instance_info() {
    local instance_id=$1
    
    info "Getting connection details..."
    
    # Get instance details
    local instance_info=$(vastai show instance "$instance_id" --raw 2>/dev/null)
    
    if [[ -z "$instance_info" ]]; then
        die "Failed to get instance information"
    fi
    
    # Parse SSH connection info
    local ssh_host=$(echo "$instance_info" | python3 -c "import sys, json; data = json.load(sys.stdin); print(data.get('public_ipaddr', ''))" 2>/dev/null)
    local ssh_port=$(echo "$instance_info" | python3 -c "import sys, json; data = json.load(sys.stdin); print(data.get('ssh_port', ''))" 2>/dev/null)
    
    if [[ -z "$ssh_host" ]] || [[ -z "$ssh_port" ]]; then
        die "Failed to get SSH connection details"
    fi
    
    echo "$ssh_host $ssh_port"
}

save_instance_info() {
    local instance_id=$1
    local ssh_host=$2
    local ssh_port=$3
    
    # Save instance info for later use
    cat > .vast_instance << EOF
INSTANCE_ID=$instance_id
SSH_HOST=$ssh_host
SSH_PORT=$ssh_port
CREATED_AT=$(date)
EOF
    
    info "Instance info saved to .vast_instance"
}

print_next_steps() {
    local ssh_host=$1
    local ssh_port=$2
    
    echo
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}                    INSTANCE READY!${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo
    echo -e "${GREEN}Connection details:${NC}"
    echo "  Host: $ssh_host"
    echo "  Port: $ssh_port"
    echo
    echo -e "${GREEN}To run the experiment:${NC}"
    echo -e "  ${YELLOW}./deploy/deploy_run_terminate.sh initial $ssh_host $ssh_port${NC}"
    echo
    echo -e "${GREEN}To connect manually:${NC}"
    echo -e "  ${YELLOW}ssh -p $ssh_port root@$ssh_host${NC}"
    echo
    echo -e "${GREEN}To monitor costs:${NC}"
    echo -e "  ${YELLOW}vastai show instances${NC}"
    echo
    echo -e "${RED}⚠️  IMPORTANT: The instance is now running and charging!${NC}"
    echo -e "${RED}    Run the experiment or terminate when done.${NC}"
    echo
}

# Main execution
main() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}           VAST.AI INSTANCE CREATOR${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo
    
    check_requirements
    search_instances
    
    # Get user selection
    instance_id=$(select_instance)
    
    # Create the instance
    new_instance_id=$(create_instance "$instance_id" | tail -1)
    
    # Wait for it to be ready
    wait_for_instance "$new_instance_id"
    
    # Get connection info
    read ssh_host ssh_port <<< $(get_instance_info "$new_instance_id")
    
    # Save for later use
    save_instance_info "$new_instance_id" "$ssh_host" "$ssh_port"
    
    # Show next steps
    print_next_steps "$ssh_host" "$ssh_port"
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi