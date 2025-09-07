# Deploying Bias Comparison Experiments to vast.ai

## 🚀 Updated Workflow for Controlled Experiments

## Prerequisites
1. Install vast.ai CLI: `pip install vastai`
2. Configure your `.env` file:
   ```bash
   cp .env.example .env
   # Edit .env and add:
   # - GITHUB_REPO=https://github.com/YOUR_USERNAME/nanda-unfaithful.git
   # - VAST_AI_API_KEY=YOUR_KEY (get from https://vast.ai/account)
   ```
3. Set API key using environment variable:
   ```bash
   # Load from .env file
   source .env
   vastai set api-key $VAST_AI_API_KEY
   
   # Or directly from .env without sourcing
   vastai set api-key $(grep VAST_AI_API_KEY .env | cut -d '=' -f2)
   ```

## Finding and Renting Instances via CLI

### Search for Available GPUs
```bash
# Find RTX 4090s with enough disk space, sorted by price (top 5)
vastai search offers 'gpu_name=RTX_4090 disk_space>80 reliability>0.99' --order dph --limit 5

# Find RTX 3090s (cheaper alternative, top 5)
vastai search offers 'gpu_name=RTX_3090 disk_space>80 reliability>0.99' --order dph --limit 5

# Find any GPU with 48GB+ VRAM (top 5)
vastai search offers 'gpu_ram>48 disk_space>100 reliability>0.99' --order dph --limit 5

# Show more details (top 10 with raw format)
vastai search offers 'gpu_name=RTX_4090 disk_space>80' --order dph --raw --limit 10
```

### Understanding the Output
```
ID       GPU           $/hr   Disk   Status
123456   RTX_4090×1    0.42   120GB  available
234567   RTX_4090×1    0.48   100GB  available
```

### Rent an Instance
```bash
# Basic rental (replace 123456 with actual offer ID)
# Using Ubuntu base (we install PyTorch ourselves)
vastai create instance 123456 --image ubuntu:22.04 --disk 80

# With SSH access (recommended)
vastai create instance 16441207 --image ubuntu:22.04 --disk 80 --ssh

# Or with CUDA base (no PyTorch pre-installed)
vastai create instance 123456 --image nvidia/cuda:11.8.0-base-ubuntu22.04 --disk 80 --ssh

# Check status
vastai show instances

# Get connection details (wait ~60 seconds after creation)
vastai show instance [INSTANCE_ID]
# Look for ssh_host (e.g., ssh5.vast.ai) and ssh_port (e.g., 21786)
# Or use: vastai ssh-url [INSTANCE_ID] to get the full SSH command
```

## Single Instance Workflow

### Step 1: Create GPU Instance
Use the CLI commands above to search for and rent an instance. Note the instance ID and PORT from `vastai show instance [ID]`.

### Step 2: SSH and Setup
```bash
# Get the exact SSH command from vast.ai
vastai ssh-url [INSTANCE_ID]
# This will show something like: ssh -p 21786 root@ssh5.vast.ai
# Or use: vastai show instance [INSTANCE_ID] and look for the ssh_host and ssh_port

# Then SSH using the provided details
ssh -i ~/.ssh/vast_ai_key -p [PORT] root@[SSH_HOST].vast.ai

# Clone and setup (in home directory)
cd ~
# or 
cd /workspace
# might need to 
apt install python3.10-venv
# Install Python, git, and nano if needed
apt-get update && apt-get install -y python3 python3-pip python3-venv git nano
git clone https://github.com/sinemmy/nanda-unfaithful.git
cd nanda-unfaithful
python3 -m venv .venv
source .venv/bin/activate
./setup_dependencies.sh
```

### Step 3: Copy .env File
```bash
# From your LOCAL machine:
scp -i ~/.ssh/vast_ai_key -P [PORT] .env root@[SSH_HOST].vast.ai:~/nanda-unfaithful/
```

### Step 4: Run Experiment (in tmux!)
```bash
# On vast.ai instance
tmux new -s experiment

# Option A: Use configuration file
source experiment_config.sh  # Edit first to select subset
./run_experiment.sh

# Option B: Direct command
python run_comparison.py --num-samples 5 --verbose

# DETACH FROM TMUX (keeps experiment running):
# Press: Ctrl+B then D

# You can now safely close SSH:
exit
```

### Reconnecting to Running Experiment
```bash
# From your local machine, SSH back in:
ssh -i ~/.ssh/vast_ai_key -p [PORT] root@[SSH_HOST].vast.ai

# Reactivate virtual environment:
cd ~/nanda-unfaithful
source .venv/bin/activate

# List running tmux sessions:
tmux ls

# Reattach to your experiment:
tmux attach -t experiment
# Or if you named it something else:
tmux attach -t algebra  # or trig, etc.

# To check progress without attaching:
ls -la outputs/
```

### Step 5: Download Results & Cleanup
```bash
# From LOCAL machine - download results
scp -i ~/.ssh/vast_ai_key -r -P [PORT] root@[SSH_HOST].vast.ai:~/nanda-unfaithful/outputs/* ./outputs/

# On vast.ai instance - cleanup
./deploy/cleanup_before_termination.sh

# From LOCAL machine - terminate
vastai destroy instance [INSTANCE_ID]
```

## Parallel Instance Workflow (Multiple GPUs)

### Option 1: Split by Problem Type (2 instances)
```bash

# Instance 1 (Algebra)
ssh -i ~/.ssh/vast_ai_key -p [PORT1] root@[SSH_HOST1].vast.ai
# ... setup steps ...
# Edit config using nano (install with: apt-get install -y nano)
nano experiment_config.sh  # Uncomment: export SUBSET_ARGS="--problem-type algebra"
# OR use sed to edit without nano:
sed -i 's/^export SUBSET_ARGS=""$/# export SUBSET_ARGS=""\nexport SUBSET_ARGS="--problem-type algebra"/' experiment_config.sh
tmux new -s algebra
./run_experiment.sh

# Instance 2 (Trigonometry) - in new terminal
ssh -i ~/.ssh/vast_ai_key -p [PORT2] root@[SSH_HOST2].vast.ai
# ... setup steps ...
nano experiment_config.sh  # Uncomment: export SUBSET_ARGS="--problem-type trigonometry"
# OR use sed:
sed -i 's/^export SUBSET_ARGS=""$/# export SUBSET_ARGS=""\nexport SUBSET_ARGS="--problem-type trigonometry"/' experiment_config.sh
tmux new -s trig
./run_experiment.sh
```

### Option 2: Split by Range (3 instances)
```bash
# Instance 1: Problems 0-1
export SUBSET_ARGS="--problem-range 0-1"

# Instance 2: Problems 2-3  
export SUBSET_ARGS="--problem-range 2-3"

# Instance 3: Problems 4-5
export SUBSET_ARGS="--problem-range 4-5"
```

### Option 3: One Problem Per Instance (6 instances - fastest)
```bash
# Each instance runs one problem
export SUBSET_ARGS="--problem-ids alg_1"  # Instance 1
export SUBSET_ARGS="--problem-ids alg_2"  # Instance 2
# ... etc
```

### Download from All Instances
```bash
# Create combined results folder
mkdir -p outputs/combined_run

# Download from each instance
scp -i ~/.ssh/vast_ai_key -r -P [PORT1] root@[SSH_HOST1].vast.ai:~/nanda-unfaithful/outputs/* ./outputs/combined_run/
scp -i ~/.ssh/vast_ai_key -r -P [PORT2] root@[SSH_HOST2].vast.ai:~/nanda-unfaithful/outputs/* ./outputs/combined_run/
# ... repeat for all instances

# Output folders will be uniquely named:
# - 20241230_143022_algebra/
# - 20241230_143045_trigonometry/
```

## Experiment Configuration Options

Edit `experiment_config.sh` on each instance to select:

### Problem Subsets
- `--problem-type algebra` - Just algebra problems (3)
- `--problem-type trigonometry` - Just trig problems (3)
- `--problem-range 0-2` - First half (3 problems)
- `--problem-range 3-5` - Second half (3 problems)
- `--problem-ids alg_1 alg_2` - Specific problems

### Sample Settings
- `NUM_SAMPLES=5` - 5 samples per variation (default)
- `NUM_SAMPLES=10` - More samples for better statistics

## Available Experiment Commands

```bash
# Full run (all 6 problems)
python run_comparison.py --num-samples 5 --verbose

# Quick test (1 problem only)
python run_comparison.py --num-samples 1 --problem-range 0-0

# Run specific subsets
python run_comparison.py --problem-type algebra --num-samples 5
python run_comparison.py --problem-type trigonometry --num-samples 5
python run_comparison.py --problem-range 0-2 --num-samples 5
python run_comparison.py --problem-ids alg_1 trig_1 --num-samples 5

# Using configuration file
source experiment_config.sh
./run_experiment.sh
```

### Key Parameters
- `--num-samples`: Samples per variation (default: 5)
- `--problem-type`: algebra, trigonometry, or all
- `--problem-range`: e.g., "0-2" for first 3 problems
- `--problem-ids`: Specific problem IDs
- Fixed: temperature=0.6, top_p=0.95

### CLI Options
- `--num-samples N`: Samples per variation (default: 5)
- `--problem-type TYPE`: algebra, trigonometry, or all
- `--problem-range RANGE`: e.g., "0-2" for first 3 problems
- `--problem-ids ID1 ID2`: Specific problem IDs
- `--model NAME`: Model to use (default: deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)
- `--output-dir PATH`: Output directory (default: ./outputs/bias_comparison)
- `--cpu`: Force CPU usage
- `--verbose`: Enable verbose logging
- `--seed N`: Random seed (default: 42)

## Cost/Time Analysis

| Strategy | Instances | Time per Instance | Total Time | Total Cost |
|----------|-----------|-------------------|------------|------------|
| Serial (1 GPU) | 1 | ~60 min | 60 min | ~$0.50 |
| By Type (2 GPUs) | 2 | ~30 min | 30 min | ~$0.50 |
| By Range (3 GPUs) | 3 | ~20 min | 20 min | ~$0.50 |
| Per Problem (6 GPUs) | 6 | ~10 min | 10 min | ~$0.50 |

Cost stays similar because each instance runs for proportionally less time!

## Tips
- Model downloads ~30GB on first run (happens on EACH instance)
- Always use `tmux` to prevent disconnection issues
- Monitor GPU with `watch -n 1 nvidia-smi`
- Results saved with unique timestamps and subset names
- Cleanup script ensures .env file is deleted

## Troubleshooting
- **Permission denied**: Check SSH port number (not 22!)
- **Out of memory**: Reduce batch size or use bigger GPU
- **Model download fails**: Need 80GB+ disk space
- **Connection drops**: Always use `tmux new -s experiment`