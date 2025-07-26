#!/bin/bash
# Setup script for Audio Transcription Tool (Google Colab with GPU acceleration)

set -e

echo "Setting up Audio Transcription Tool for Google Colab with GPU acceleration..."

# Check GPU and CUDA compatibility
echo "Checking GPU and CUDA compatibility..."
python3 check_cuda_compatibility.py

# Get build configuration
eval $(python3 check_cuda_compatibility.py --shell)
echo "Build configuration: CUDA_ENABLED=$CUDA_ENABLED"

# Update package list
apt-get update -qq

# Install required system packages
echo "Installing system dependencies..."
if [ "$CUDA_ENABLED" = "true" ]; then
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

# Configure with CMake using compatibility-checked flags
echo "Configuring build with recommended settings..."
cmake .. $CMAKE_FLAGS

# Build with all available cores
echo "Compiling (this may take a few minutes)..."
if ! cmake --build . -j$(nproc) --config Release; then
    if [ "$CUDA_ENABLED" = "true" ]; then
        echo "CUDA build failed, trying fallback CPU build..."
        cd ..
        rm -rf build
        mkdir build
        cd build
        
        # Fallback to CPU-only build
        cmake .. -DCMAKE_BUILD_TYPE=Release \
                 -DGGML_NATIVE=ON \
                 -DGGML_BLAS=ON
        
        cmake --build . -j$(nproc) --config Release
        
        if [ $? -eq 0 ]; then
            echo "Fallback CPU build successful"
            CUDA_ENABLED=false
        else
            echo "Both GPU and CPU builds failed"
            exit 1
        fi
    else
        echo "CPU build failed"
        exit 1
    fi
fi

cd ../..

# Verify build
if [ ! -f "whisper.cpp/build/bin/whisper-cli" ]; then
    echo "Error: whisper-cli not found after build"
    exit 1
fi

# Test GPU acceleration
if [ "$CUDA_ENABLED" = "true" ]; then
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
if [ "$CUDA_ENABLED" = "true" ]; then
    echo "🚀 GPU acceleration is enabled for faster transcription!"
else
    echo "💻 CPU-optimized build completed."
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