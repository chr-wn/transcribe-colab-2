#!/bin/bash
# Setup script for Audio Transcription Tool (Google Colab with GPU acceleration)

set -e

echo "Setting up Audio Transcription Tool for Google Colab with GPU acceleration..."

# Set up CUDA environment variables
export CUDA_HOME=/usr/local/cuda-12.1
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

echo "CUDA environment:"
echo "  CUDA_HOME: $CUDA_HOME"
echo "  PATH: $PATH"
echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Check GPU availability
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    GPU_AVAILABLE=true
    echo "NVIDIA GPU detected!"
else
    echo "No NVIDIA GPU detected, using CPU mode"
    GPU_AVAILABLE=false
fi

# Update package list
apt-get update -qq

# Install required system packages
echo "Installing system dependencies..."
if [ "$GPU_AVAILABLE" = true ]; then
    # Install CUDA development tools for GPU acceleration
    apt-get install -y -qq cmake build-essential git curl nvidia-cuda-toolkit
else
    apt-get install -y -qq cmake build-essential git curl
fi

# Clone whisper.cpp if not exists
if [ ! -d "whisper.cpp" ]; then
    echo "Cloning whisper.cpp..."
    git clone https://github.com/ggml-org/whisper.cpp.git
else
    echo "whisper.cpp already exists, updating..."
    cd whisper.cpp
    git pull
    cd ..
fi

# Build whisper.cpp with GPU acceleration
echo "Building whisper.cpp with optimizations..."
cd whisper.cpp

# Create build directory
mkdir -p build
cd build

# Configure with CMake (GPU-optimized for Colab)
if [ "$GPU_AVAILABLE" = true ]; then
    echo "Building with CUDA support for GPU acceleration..."
    echo "Using CUDA compiler: $CUDA_HOME/bin/nvcc"
    
    # Use CUDA for T4/A100 GPUs in Colab with conservative settings
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DGGML_CUDA=1 \
             -DCMAKE_CUDA_ARCHITECTURES="75" \
             -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
             -DGGML_CUDA_F16=OFF \
             -DGGML_NATIVE=ON
else
    echo "Building with CPU optimizations..."
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DGGML_NATIVE=ON \
             -DGGML_BLAS=ON
fi

# Build with all available cores
echo "Compiling (this may take a few minutes)..."
cmake --build . -j$(nproc) --config Release

cd ../..

# Verify build
if [ ! -f "whisper.cpp/build/bin/whisper-cli" ]; then
    echo "Error: whisper-cli not found after build"
    exit 1
fi

# Test the build
echo "Testing whisper.cpp build..."
if ./whisper.cpp/build/bin/whisper-cli --help > /dev/null 2>&1; then
    echo "✅ whisper-cli is working"
    
    # Test GPU acceleration
    if [ "$GPU_AVAILABLE" = true ]; then
        echo "Testing GPU acceleration..."
        ./whisper.cpp/build/bin/whisper-cli --help | grep -i cuda || echo "CUDA support may not be enabled"
    fi
else
    echo "❌ whisper-cli test failed"
    exit 1
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install mutagen

# Make transcribe.py executable
chmod +x transcribe.py

# Create models directory
mkdir -p models

# Pre-download base model for faster first use
echo "Pre-downloading base model..."
mkdir -p models
curl -L -o models/ggml-base.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin

echo ""
echo "Colab setup complete!"
if [ "$GPU_AVAILABLE" = true ]; then
    echo "GPU acceleration is enabled for faster transcription!"
else
    echo "CPU-optimized build completed."
fi
echo ""
echo "Usage in Colab:"
echo "  !./transcribe.py audio.mp3                    # Basic transcription"
echo "  !./transcribe.py -t audio.mp3                 # With timestamps"
echo "  !./transcribe.py -m large audio.mp3           # Use large model (recommended with GPU)"
echo "  !./transcribe.py -o transcript.txt audio.mp3  # Custom output"
echo ""
echo "With GPU acceleration, you can use larger models for better accuracy!"
echo "Upload your audio files using the file browser on the left."