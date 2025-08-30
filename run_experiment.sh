#!/bin/bash
# Run experiment with configuration from experiment_config.sh

# Source the configuration
source experiment_config.sh

# Check if we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Warning: No virtual environment detected"
    echo "Activating .venv..."
    source .venv/bin/activate
fi

# Check if we're in tmux
if [[ -z "$TMUX" ]]; then
    echo "⚠️  Warning: Not running in tmux!"
    echo "It's recommended to run in tmux to prevent disconnection:"
    echo "  tmux new -s experiment"
    echo ""
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting. Please run: tmux new -s experiment"
        exit 1
    fi
fi

# Display what will run
echo ""
echo "=========================================="
echo "STARTING EXPERIMENT"
echo "=========================================="
echo "Command: $RUN_COMMAND"
echo "=========================================="
echo ""

# Run the experiment
eval $RUN_COMMAND

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ EXPERIMENT COMPLETED SUCCESSFULLY"
    echo "=========================================="
    echo "Results saved in: outputs/"
    echo ""
    echo "To download results from local machine:"
    echo "  scp -r -P [PORT] root@ssh2.vast.ai:/workspace/nanda-unfaithful/outputs/* ./"
else
    echo ""
    echo "=========================================="
    echo "❌ EXPERIMENT FAILED"
    echo "=========================================="
    echo "Check error messages above"
fi