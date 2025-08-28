# Deploying Unfaithful CoT Experiments to vast.ai

## Prerequisites
1. Create a vast.ai account and add credits
2. Have your SSH key ready

## Step 1: Choose an Instance
1. Go to vast.ai console
2. Filter for:
   - GPU: RTX 4090 (24GB) or A100 (40GB)
   - Disk: 50GB+
   - OS: Ubuntu 22.04
   - CUDA: 12.0+
3. Select a spot/interruptible instance (~$0.50/hr for 4090)
4. Click "RENT" and wait for instance to start

## Step 2: Get Connection Info
Once running, you'll see connection details like:
```
ssh root@123.45.67.89 -p 12345
```

## Step 3: Deploy Your Code

### Recommended: Use Automated Scripts
The easiest way is to use our automated scripts from your local machine:

```bash
# Configure .env with your GitHub repo
echo "GITHUB_REPO=https://github.com/NandaStream/nanda-unfaithful.git" > .env

# Run complete workflow (deploys, runs, downloads, terminates)
./deploy/deploy_run_terminate.sh initial 123.45.67.89 12345
```

This handles everything automatically!

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

## Step 4: Upload .env File (IMPORTANT!)
The .env file is gitignored, so upload it separately:
```bash
scp -P 12345 .env root@123.45.67.89:/workspace/nanda-unfaithful/
```

## Step 5: Run Setup
```bash
cd /workspace/nanda-unfaithful
# Now use deploy_run_terminate.sh from your local machine instead
```

## Step 6: Test & Run Experiments
```bash
# Test the setup with small model
python main.py --test --cpu --max-examples 1

# Run unfaithful CoT generation (quick test)
python main.py --max-examples 3

# Run full analysis
python main.py --max-examples 10
python run_anchors_analysis.py --input-dir outputs/unfaithful_cot/[timestamp]

# Monitor GPU usage
watch -n 1 nvidia-smi
```

## Step 7: Download Results
After experiments complete:
```bash
# From your local machine
scp -P 12345 -r root@123.45.67.89:/workspace/nanda-unfaithful/outputs ./
```

## Tips
- Use `tmux` or `screen` to keep experiments running if SSH disconnects
- Monitor GPU with `watch -n 1 nvidia-smi`
- Check logs in `outputs/unfaithful_cot/` for progress
- Stop instance when done to avoid charges

## Troubleshooting
- **Permission denied**: Check SSH port number (it's not 22!)
- **Out of memory**: Use A100 instead of 4090
- **Model download fails**: Check disk space with `df -h`
- **Connection drops**: Use `tmux new -s experiment` before running