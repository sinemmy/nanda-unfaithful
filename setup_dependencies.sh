#!/bin/bash
# Setup script for installing dependencies on vast.ai instances

echo "Setting up dependencies for nanda-unfaithful project..."

# Update pip first
pip install --upgrade pip setuptools wheel

# Install numpy with pre-built wheels (avoid compilation)
echo "Installing numpy..."
pip install numpy --prefer-binary

# Install core dependencies
echo "Installing core dependencies..."
pip install PyYAML python-dotenv beautifulsoup4 lxml tqdm

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA 11.8 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install transformers and related packages
echo "Installing transformers ecosystem..."
pip install transformers accelerate sentencepiece tokenizers

# Install bitsandbytes for quantization
echo "Installing bitsandbytes for quantization..."
pip install bitsandbytes

# Try to install datasets with fallback
echo "Installing datasets package..."
pip install datasets || {
    echo "Standard datasets install failed, trying without dependencies..."
    pip install datasets --no-deps
    echo "Installing datasets dependencies manually..."
    pip install fsspec pandas requests xxhash multiprocess dill huggingface-hub
    # Try pyarrow separately as it often causes issues
    pip install pyarrow || echo "Warning: pyarrow installation failed, some features may not work"
}

# Optional: vast.ai CLI
pip install vastai || echo "vastai CLI installation failed (optional)"

echo ""
echo "✅ Dependencies installed successfully!"
echo ""
