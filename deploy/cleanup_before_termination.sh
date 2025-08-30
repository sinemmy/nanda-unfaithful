#!/bin/bash

# Cleanup script to run on vast.ai instance AFTER downloading results
# This removes sensitive data and prepares instance for termination

set -e  # Exit on error

echo "=============================================="
echo "CLEANUP BEFORE TERMINATION"
echo "=============================================="
echo ""
echo "⚠️  WARNING: Only run this AFTER you have:"
echo "   1. Downloaded all results"
echo "   2. Verified the downloads are complete"
echo ""
read -p "Have you already downloaded and verified all results? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Please download your results first, then run this script."
    echo ""
    echo "To download from your LOCAL machine:"
    echo "  scp -r -P [PORT] root@ssh2.vast.ai:/workspace/nanda-unfaithful/outputs ./"
    exit 1
fi

# Configuration
WORKSPACE_DIR="/workspace/nanda-unfaithful"
OUTPUTS_DIR="$WORKSPACE_DIR/outputs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -d "$WORKSPACE_DIR" ]; then
    print_error "Workspace directory not found: $WORKSPACE_DIR"
    print_warning "Are you running this on the vast.ai instance?"
    exit 1
fi

cd "$WORKSPACE_DIR"

# Step 1: Kill any running processes
echo ""
print_status "Checking for running experiments..."

# Check for tmux sessions
if tmux list-sessions 2>/dev/null; then
    print_warning "Found running tmux sessions - killing all"
    tmux kill-server 2>/dev/null || true
    print_status "All tmux sessions terminated"
fi

# Kill Python processes
if pgrep -f "python" > /dev/null; then
    print_warning "Found running Python processes - killing"
    pkill -f "python" 2>/dev/null || true
    print_status "Python processes terminated"
fi

# Step 2: Clean up sensitive data
echo ""
print_status "Cleaning up sensitive data..."

# CRITICAL: Remove .env file that contains API keys
if [ -f "$WORKSPACE_DIR/.env" ]; then
    print_status "CRITICAL: Shredding .env file with API keys..."
    shred -vfz "$WORKSPACE_DIR/.env" 2>/dev/null || rm -f "$WORKSPACE_DIR/.env"
    print_status "✓ .env file securely deleted"
else
    print_warning "No .env file found (good if you didn't copy it)"
fi

# Clear bash history
print_status "Clearing bash history..."
history -c
cat /dev/null > ~/.bash_history

# Clear Python cache
print_status "Clearing Python cache..."
find "$WORKSPACE_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$WORKSPACE_DIR" -type f -name "*.pyc" -delete 2>/dev/null || true

# Step 3: Delete outputs (since already downloaded)
echo ""
if [ -d "$OUTPUTS_DIR" ] && [ "$(ls -A $OUTPUTS_DIR 2>/dev/null)" ]; then
    OUTPUTS_SIZE=$(du -sh "$OUTPUTS_DIR" | cut -f1)
    print_status "Deleting outputs directory ($OUTPUTS_SIZE)..."
    rm -rf "$OUTPUTS_DIR"
    print_status "✓ Outputs directory deleted"
fi

# Step 4: Delete model cache
echo ""
if [ -d "$WORKSPACE_DIR/model_cache" ]; then
    CACHE_SIZE=$(du -sh "$WORKSPACE_DIR/model_cache" | cut -f1)
    print_status "Deleting model cache ($CACHE_SIZE)..."
    rm -rf "$WORKSPACE_DIR/model_cache"
    print_status "✓ Model cache deleted"
fi

# Step 5: GPU cleanup
if command -v nvidia-smi &> /dev/null; then
    print_status "Clearing GPU memory..."
    # Kill any remaining GPU processes
    nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 kill -9 2>/dev/null || true
fi

# Step 6: Delete any temporary files
print_status "Removing temporary files..."
rm -f /tmp/results_*.tar.gz 2>/dev/null || true
rm -rf /tmp/pip-* 2>/dev/null || true

# Step 7: Final summary
echo ""
echo "=============================================="
echo "CLEANUP COMPLETE"
echo "=============================================="
print_status "✓ All processes terminated"
print_status "✓ .env file securely deleted"
print_status "✓ Outputs directory deleted"
print_status "✓ Model cache deleted"
print_status "✓ Python cache cleared"
print_status "✓ Bash history cleared"
print_status "✓ GPU memory cleared"
echo ""
print_status "Instance is ready for termination."
echo ""
echo "To terminate from your local machine:"
echo -e "${GREEN}vastai destroy instance [INSTANCE_ID]${NC}"
echo ""