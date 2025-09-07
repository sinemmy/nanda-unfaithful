#!/bin/bash
# Experiment Configuration File
# Source this file to set environment variables for your experiment run
# Usage: source experiment_config.sh

# ============================================
# BASIC PARAMETERS
# ============================================

# Number of samples per variation (5 variations × N samples per problem)
export NUM_SAMPLES=4

# Verbosity
export VERBOSE="--verbose"  # Comment out for quiet mode
# export VERBOSE=""

# ============================================
# PROBLEM SUBSET SELECTION
# Choose ONE of the following options
# ============================================

# Option 1: Run ALL problems (default - 6 problems total)
export SUBSET_ARGS=""

# Option 2: Run by problem type
# export SUBSET_ARGS="--problem-type algebra"      # Just algebra (3 problems)
# export SUBSET_ARGS="--problem-type trigonometry" # Just trig (3 problems)

# Option 3: Run by range (0-indexed)
# export SUBSET_ARGS="--problem-range 0-1"  # First 2 problems
# export SUBSET_ARGS="--problem-range 2-3"  # Middle 2 problems  
# export SUBSET_ARGS="--problem-range 4-5"  # Last 2 problems
# export SUBSET_ARGS="--problem-range 0-2"  # First half (3 problems)
# export SUBSET_ARGS="--problem-range 3-5"  # Second half (3 problems)

# Option 4: Run specific problem IDs
# export SUBSET_ARGS="--problem-ids alg_1"           # Single problem
# export SUBSET_ARGS="--problem-ids alg_1 alg_2"     # Two specific problems
# export SUBSET_ARGS="--problem-ids alg_1 alg_2 alg_3"  # All algebra
# export SUBSET_ARGS="--problem-ids trig_1 trig_2 trig_3"  # All trig
# export SUBSET_ARGS="--problem-ids alg_1 trig_1"    # One of each type

# ============================================
# PARALLEL INSTANCE CONFIGURATIONS
# Uncomment the set you want for parallel runs
# ============================================

# === 2 INSTANCES (by type) ===
# Instance 1:
# export SUBSET_ARGS="--problem-type algebra"
# Instance 2:
# export SUBSET_ARGS="--problem-type trigonometry"

# === 3 INSTANCES (by pairs) ===
# Instance 1:
# export SUBSET_ARGS="--problem-range 0-1"
# Instance 2:
# export SUBSET_ARGS="--problem-range 2-3"
# Instance 3:
# export SUBSET_ARGS="--problem-range 4-5"

# === 6 INSTANCES (one problem each) ===
# Instance 1:
# export SUBSET_ARGS="--problem-ids alg_1"
# Instance 2:
# export SUBSET_ARGS="--problem-ids alg_2"
# Instance 3:
# export SUBSET_ARGS="--problem-ids alg_3"
# Instance 4:
# export SUBSET_ARGS="--problem-ids trig_1"
# Instance 5:
# export SUBSET_ARGS="--problem-ids trig_2"
# Instance 6:
# export SUBSET_ARGS="--problem-ids trig_3"

# ============================================
# ADVANCED OPTIONS
# ============================================

# Model selection (default: DeepSeek-R1)
# export MODEL="--model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
# export MODEL="--model gpt2"  # For testing

# Output directory (default: ./outputs/bias_comparison)
# export OUTPUT_DIR="--output-dir ./outputs/custom_dir"

# Force CPU (not recommended)
# export DEVICE="--cpu"


# ============================================
# FINAL COMMAND CONSTRUCTION
# ============================================

# Build the full command
export RUN_COMMAND="python run_comparison.py --num-samples $NUM_SAMPLES $SUBSET_ARGS $VERBOSE"

# ============================================
# HELPER INFORMATION
# ============================================

echo "=========================================="
echo "EXPERIMENT CONFIGURATION LOADED"
echo "=========================================="
echo "Configuration:"
echo "  Samples per variation: $NUM_SAMPLES"
echo "  Subset: ${SUBSET_ARGS:-ALL PROBLEMS}"
echo "  Verbose: ${VERBOSE:-OFF}"
echo ""
echo "To run experiment:"
echo "  $RUN_COMMAND"
echo ""
echo "Or simply run:"
echo "  ./run_experiment.sh"
echo "=========================================="