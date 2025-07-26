#!/bin/bash
# Setup script for Audio Transcription Tool (Local)

set -e

echo "Setting up Audio Transcription Tool..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    echo "Please install Python 3 and try again."
    exit 1
fi

# Check if cmake is available
if ! command -v cmake &> /dev/null; then
    echo "Error: CMake is required but not installed."
    echo "Please install CMake and try again."
    exit 1
fi

# Check if git is available
if ! command -v git &> /dev/null; then
    echo "Error: Git is required but not installed."
    echo "Please install Git and try again."
    exit 1
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

# Build whisper.cpp
echo "Building whisper.cpp..."
cd whisper.cpp

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . -j --config Release

cd ../..

# Verify build
if [ ! -f "whisper.cpp/build/bin/whisper-cli" ]; then
    echo "Error: whisper-cli not found after build"
    exit 1
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install --user mutagen

# Make transcribe.py executable
chmod +x transcribe.py

# Create models directory
mkdir -p models

echo ""
echo "Setup complete!"
echo ""
echo "Usage:"
echo "  ./transcribe.py audio.mp3                    # Basic transcription"
echo "  ./transcribe.py -t audio.mp3                 # With timestamps"
echo "  ./transcribe.py -m large audio.mp3           # Use large model"
echo "  ./transcribe.py -o transcript.txt audio.mp3  # Custom output"
echo "  ./transcribe.py *.mp3                        # Batch processing"
echo ""
echo "The first time you use a model, it will be downloaded automatically."