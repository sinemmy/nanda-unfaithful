# Parallel Instance Execution Guide

## Overview
Run subsets of the experiment in parallel across multiple vast.ai instances to reduce total time.

## Splitting Strategies

### Option 1: By Problem Type (2 instances)
```bash
# Instance 1: Algebra problems
python run_comparison.py --problem-type algebra --num-samples 5 --verbose

# Instance 2: Trigonometry problems  
python run_comparison.py --problem-type trigonometry --num-samples 5 --verbose
```

### Option 2: By Problem Range (3 instances)
```bash
# Instance 1: Problems 0-1 (first 2)
python run_comparison.py --problem-range 0-1 --num-samples 5 --verbose

# Instance 2: Problems 2-3 (middle 2)
python run_comparison.py --problem-range 2-3 --num-samples 5 --verbose

# Instance 3: Problems 4-5 (last 2)
python run_comparison.py --problem-range 4-5 --num-samples 5 --verbose
```

### Option 3: By Specific Problems (6 instances - maximum parallelization)
```bash
# Instance 1:
python run_comparison.py --problem-ids alg_1 --num-samples 5 --verbose

# Instance 2:
python run_comparison.py --problem-ids alg_2 --num-samples 5 --verbose

# Instance 3:
python run_comparison.py --problem-ids alg_3 --num-samples 5 --verbose

# Instance 4:
python run_comparison.py --problem-ids trig_1 --num-samples 5 --verbose

# Instance 5:
python run_comparison.py --problem-ids trig_2 --num-samples 5 --verbose

# Instance 6:
python run_comparison.py --problem-ids trig_3 --num-samples 5 --verbose
```

## Instance Management Workflow

### 1. Create Multiple Instances
```bash
# Run this multiple times (once per instance needed)
./deploy/create_vast_instance.sh
# Note down each instance ID and port
```

### 2. Deploy to Each Instance (in parallel terminals)
```bash
# Terminal 1
ssh -i ~/.ssh/vast_ai_key -p [PORT1] root@ssh2.vast.ai
cd /workspace && git clone https://github.com/sinemmy/nanda-unfaithful.git
cd nanda-unfaithful && python -m venv .venv && source .venv/bin/activate
./setup_dependencies.sh
# Copy .env from local: scp -P [PORT1] .env root@ssh2.vast.ai:/workspace/nanda-unfaithful/
tmux new -s exp1
python run_comparison.py --problem-type algebra --num-samples 5 --verbose

# Terminal 2 (same setup, different subset)
ssh -i ~/.ssh/vast_ai_key -p [PORT2] root@ssh2.vast.ai
# ... same setup steps ...
tmux new -s exp2
python run_comparison.py --problem-type trigonometry --num-samples 5 --verbose
```

### 3. Monitor Progress
```bash
# Check all instances
vastai show instances

# Check specific instance
vastai show instance [INSTANCE_ID]
```

### 4. Download Results
Each instance creates uniquely named output folders:
- `20241230_143022_algebra/` - From algebra instance
- `20241230_143045_trigonometry/` - From trigonometry instance
- `20241230_143102_range_0_to_1/` - From range instance

```bash
# Download from each instance
scp -r -P [PORT1] root@ssh2.vast.ai:/workspace/nanda-unfaithful/outputs/* ./outputs/
scp -r -P [PORT2] root@ssh2.vast.ai:/workspace/nanda-unfaithful/outputs/* ./outputs/
```

### 5. Cleanup and Terminate
```bash
# On each instance
./deploy/cleanup_before_termination.sh

# From local machine
vastai destroy instance [INSTANCE_ID_1]
vastai destroy instance [INSTANCE_ID_2]
```

## Cost/Time Tradeoffs

| Strategy | Instances | Time | Cost | Best For |
|----------|-----------|------|------|----------|
| Serial (1 instance) | 1 | ~60 min | $0.50 | Budget-conscious |
| By Type (2 instances) | 2 | ~30 min | $0.50 | Balanced |
| By Range (3 instances) | 3 | ~20 min | $0.50 | Faster results |
| Per Problem (6 instances) | 6 | ~10 min | $0.50 | Maximum speed |

Note: Cost stays ~$0.50 total because each instance runs for less time!

## Combining Results

After downloading all outputs, you can combine them:

```python
# combine_results.py (create this if needed)
import json
from pathlib import Path

all_results = []
for result_dir in Path("outputs").glob("*/raw_comparison_results.json"):
    with open(result_dir) as f:
        all_results.extend(json.load(f))

with open("outputs/combined_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
```

## Tips
- Model download happens on EACH instance (~30GB each)
- First instance takes longer due to download
- Use tmux on each instance to prevent disconnection
- Name your tmux sessions descriptively (exp_alg, exp_trig, etc.)