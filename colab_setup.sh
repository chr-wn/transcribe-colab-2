#!/bin/bash
# Setup script for Audio Transcription Tool (Google Colab)

set -e

echo "Setting up Audio Transcription Tool for Google Colab..."

# Update package list
apt-get update -qq

# Install required system packages
echo "Installing system dependencies..."
apt-get install -y -qq cmake build-essential git curl

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

# Build whisper.cpp
echo "Building whisper.cpp..."
cd whisper.cpp

# Create build directory
mkdir -p build
cd build

# Configure with CMake (optimized for Colab)
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build with all available cores
cmake --build . -j$(nproc) --config Release

cd ../..

# Verify build
if [ ! -f "whisper.cpp/build/bin/whisper-cli" ]; then
    echo "Error: whisper-cli not found after build"
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
./transcribe.py --help > /dev/null 2>&1 || true  # This will trigger model download

echo ""
echo "Colab setup complete!"
echo ""
echo "Usage in Colab:"
echo "  !./transcribe.py audio.mp3                    # Basic transcription"
echo "  !./transcribe.py -t audio.mp3                 # With timestamps"
echo "  !./transcribe.py -m large audio.mp3           # Use large model"
echo "  !./transcribe.py -o transcript.txt audio.mp3  # Custom output"
echo ""
echo "Upload your audio files using the file browser on the left."