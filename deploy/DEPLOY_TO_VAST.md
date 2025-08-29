# Deploying Unfaithful CoT Experiments to vast.ai

## 🚀 100% CLI-Based Workflow (No Web Console Needed!)

## Prerequisites
1. Install vast.ai CLI: `pip install vastai`
2. Configure your `.env` file:
   ```bash
   cp .env.example .env
   # Edit .env and add:
   # - GITHUB_REPO=https://github.com/YOUR_USERNAME/nanda-unfaithful.git
   # - VAST_AI_API_KEY=YOUR_KEY (get from https://vast.ai/account)
   ```

## Quick Start (Fully Automated)

### Step 1: Create & Rent GPU Instance (Interactive)
```bash
# Run the interactive instance creator
./deploy/create_vast_instance.sh

# This will:
# 1. Search for available RTX 4090/3090 instances with 80GB+ disk
# 2. Show you prices sorted by cost
# 3. Let you select an instance by ID
# 4. Create and start the instance
# 5. Wait for it to be ready
# 6. Display the exact command to run experiments
```

### Step 2: Run Experiments (Auto-Terminates!)
```bash
# The create_vast_instance.sh script will show you the exact command, like:
./deploy/deploy_run_terminate.sh initial 123.45.67.89 12345

# For full run instead of quick test:
./deploy/deploy_run_terminate.sh followup 123.45.67.89 12345
```

### Alternative: Manual Instance Creation
If you prefer to manually search and create instances:
```bash
# Find cheapest RTX 4090
vastai search offers 'gpu_name=RTX_4090 disk_space>80 cuda_vers>=12.0' --order dph

# Rent the instance (replace OFFER_ID with ID from search)
vastai create instance OFFER_ID --image pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime --disk 80 --ssh

# Wait for it to start and get connection info
sleep 60
vastai show instances  # Note: ssh root@IP -p PORT

# Run experiments
./deploy/deploy_run_terminate.sh initial IP PORT
```

That's it! The script handles everything:
- Deploys code from GitHub
- Installs dependencies
- Runs experiments
- Downloads results to `results_[type]_[timestamp]/`
- **Terminates instance automatically** (no overcharging!)

## Manual Deployment (If Needed)

### Option B: Direct Upload (No GitHub)
1. On your local machine, create archive:
```bash
cd /Users/oshun/Documents/GitHub/NandaStream
tar -czf nanda-unfaithful.tar.gz nanda-unfaithful/ \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='model_cache' \
    --exclude='outputs'
```

2. Upload to vast.ai instance:
```bash
scp -P 12345 nanda-unfaithful.tar.gz root@123.45.67.89:/workspace/
```

3. SSH in and extract:
```bash
ssh root@123.45.67.89 -p 12345
cd /workspace
tar -xzf nanda-unfaithful.tar.gz
cd nanda-unfaithful
# Now use deploy_run_terminate.sh from your local machine instead
```

### Option C: Direct Copy (Simple but Slower)
```bash
# Copy entire directory (excluding large files)
rsync -avz -e "ssh -p 12345" \
    --exclude='.venv' \
    --exclude='model_cache' \
    --exclude='__pycache__' \
    /Users/oshun/Documents/GitHub/NandaStream/nanda-unfaithful/ \
    root@123.45.67.89:/workspace/nanda-unfaithful/
```

## CLI Usage Examples (Direct from Command Line!)

Once deployed, you can run experiments with various options:

```bash
# Test mode with minimal settings
python main.py --test --max-examples 2

# Generate 10 unfaithful examples
python main.py --max-examples 10

# Focus on specific bias type
python main.py --bias-type suggested_answer --max-examples 5

# Higher temperature for more variation
python main.py --temperature 0.9 --top-p 0.98 --max-examples 8

# Run with thought anchors analysis
python main.py --run-anchors --max-examples 10

# Full pipeline with custom output
python main.py --max-examples 15 --output-dir ./results/experiment1

# Verbose mode for debugging
python main.py --verbose --max-examples 3
```

### Available CLI Options:
- `--max-examples N`: Number of examples to generate (default: 5)
- `--bias-type TYPE`: Bias type: suggested_answer, sycophancy, spurious_few_shot, all (default: all)
- `--temperature FLOAT`: Generation temperature (default: 0.8)
- `--top-p FLOAT`: Top-p sampling (default: 0.95)
- `--max-tokens N`: Max new tokens (default: 1024)
- `--model NAME`: Model to use (default: deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)
- `--cache-dir PATH`: Model cache directory (default: ./model_cache)
- `--output-dir PATH`: Output directory (default: ./outputs/unfaithful_cot)
- `--test`: Test mode with smaller model
- `--cpu`: Force CPU usage
- `--verbose`: Enable verbose logging
- `--run-anchors`: Run thought anchors analysis after generation
- `--seed N`: Random seed (default: 42)

## Background Auto-Terminator (Perfect for Overnight Runs)
```bash
# Monitors, downloads results, and terminates after 4 hours
python deploy/start_monitor_and_auto_terminate.py IP PORT 4 &
```

## Tips
- Model downloads ~30GB on first run (cached for future runs)
- Use `tmux` or `screen` to keep experiments running if SSH disconnects
- Monitor GPU with `watch -n 1 nvidia-smi`
- Results saved incrementally in `outputs/unfaithful_cot/`
- **Auto-termination prevents overcharging** - always use the scripts!

## Troubleshooting
- **Permission denied**: Check SSH port number (it's not 22!)
- **Out of memory**: Use A100 instead of 4090 or reduce `--max-tokens`
- **Model download fails**: Check disk space with `df -h`
- **Connection drops**: Use `tmux new -s experiment` before running
- **Auto-termination fails**: Check VAST_AI_API_KEY in .env