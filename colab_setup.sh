#!/bin/bash
# Setup script for Audio Transcription Tool (Google Colab with GPU acceleration)

set -e

echo "Setting up Audio Transcription Tool for Google Colab with GPU acceleration..."

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
    # Use CUDA for T4/A100 GPUs in Colab
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DGGML_CUDA=1 \
             -DCMAKE_CUDA_ARCHITECTURES="75;80;86" \
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

# Test GPU acceleration
if [ "$GPU_AVAILABLE" = true ]; then
    echo "Testing GPU acceleration..."
    ./whisper.cpp/build/bin/whisper-cli --help | grep -i cuda || echo "CUDA support may not be enabled"
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