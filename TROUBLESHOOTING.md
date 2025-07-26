# Troubleshooting Guide

## Common Issues and Solutions

### Setup Issues

#### "cmake not found"
```bash
# Ubuntu/Debian
sudo apt-get install cmake

# macOS
brew install cmake

# Or download from https://cmake.org/download/
```

#### "whisper-cli not found after build"
```bash
# Check if build succeeded
ls -la whisper.cpp/build/bin/

# If empty, try rebuilding
cd whisper.cpp
rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j --config Release
```

#### "Permission denied" when running scripts
```bash
chmod +x setup.sh colab_setup.sh transcribe.py test_setup.py
```

### Runtime Issues

#### "Model not found" error
The tool should download models automatically. If this fails:
```bash
# Manual download
mkdir -p models
cd models
curl -L -o ggml-base.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin
```

#### "File format not supported"
Convert your audio file:
```bash
# Using ffmpeg
ffmpeg -i input.m4a -ar 16000 -ac 1 output.wav

# Supported formats: MP3, WAV, M4A, FLAC, OGG, WMA
```

#### Slow transcription
- Use a smaller model (`tiny` or `base`)
- Ensure you have enough RAM
- Check if GPU acceleration is working

#### "Out of memory" error
- Use a smaller model
- Process files one at a time instead of batch mode
- Close other applications

### Google Colab Issues

#### "Runtime disconnected"
- Colab has usage limits
- Save your work frequently
- Consider using smaller models for long audio files

#### "Disk space full"
```python
# Check disk usage
!df -h

# Clean up if needed
!rm -rf whisper.cpp/build/*.o
!rm -rf models/ggml-large*.bin  # Remove large models if not needed
```

#### Upload fails
- Check file size limits (Colab has limits)
- Try uploading smaller files
- Use Google Drive integration for large files

### Performance Issues

#### Very slow transcription
1. Check if you're using the right model size
2. Verify whisper.cpp built with optimizations:
   ```bash
   cd whisper.cpp/build
   cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
   cmake --build . -j --config Release
   ```

#### GPU not being used
whisper.cpp will automatically use available GPU acceleration. To verify:
```bash
# Check if CUDA is available (NVIDIA)
nvidia-smi

# Check CUDA compatibility and build status
python check_cuda_compatibility.py
```

#### CUDA Build Errors (PTX/movmatrix errors)
If you see errors like "Feature 'movmatrix' requires PTX ISA .version 7.8":

```bash
# Check CUDA compatibility first
!python check_cuda_compatibility.py

# If CUDA build fails, force CPU build
!cd whisper.cpp && rm -rf build && mkdir build && cd build
!cd whisper.cpp/build && cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_BLAS=ON
!cd whisper.cpp/build && cmake --build . -j --config Release
```

The setup script automatically falls back to CPU build if CUDA compilation fails.

### File Issues

#### "File not readable"
```bash
# Check file permissions
ls -la your_audio_file.mp3

# Fix permissions
chmod 644 your_audio_file.mp3
```

#### "Unsupported file format"
The tool supports: MP3, WAV, M4A, FLAC, OGG, WMA

Convert unsupported formats:
```bash
ffmpeg -i input.mov -vn -acodec mp3 output.mp3
```

### Output Issues

#### Empty transcript
- Check if the audio file has speech
- Try a different model (larger models are more accurate)
- Verify the audio file isn't corrupted

#### Garbled output
- The audio might be in a different language
- Try using the `large` model for better accuracy
- Check if the audio quality is good

## Getting Help

If you're still having issues:

1. Run the test script: `./test_setup.py`
2. Check the whisper.cpp logs for detailed error messages
3. Verify your audio file plays correctly in other applications
4. Try with a different, known-good audio file

## Debug Mode

For detailed debugging information:
```bash
# Run with verbose output
./transcribe.py -v your_audio.mp3

# Check whisper.cpp directly
./whisper.cpp/build/bin/whisper-cli --help
./whisper.cpp/build/bin/whisper-cli -m models/ggml-base.bin -f your_audio.mp3
```