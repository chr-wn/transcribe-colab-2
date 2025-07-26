# GPU Optimization Guide for Google Colab

This guide explains how to maximize transcription performance using GPU acceleration in Google Colab.

## GPU Types in Colab

Google Colab provides different GPU types:

- **T4** - Most common, good performance (16GB VRAM)
- **A100** - Highest performance (40GB VRAM) 
- **V100** - High performance (16GB VRAM)

## Enabling GPU Acceleration

1. **Set Runtime Type:**
   - Go to `Runtime` → `Change runtime type`
   - Set `Hardware accelerator` to `GPU`
   - Click `Save`

2. **Verify GPU Access:**
   ```bash
   !nvidia-smi
   ```

## Performance Optimizations

### Model Selection for GPU

With GPU acceleration, you can use larger models efficiently:

| Model  | CPU Time | GPU Time | Accuracy | Recommended Use |
|--------|----------|----------|----------|-----------------|
| tiny   | 10s      | 3s       | Basic    | Quick tests     |
| base   | 30s      | 8s       | Good     | General use     |
| small  | 60s      | 15s      | Better   | Quality content |
| medium | 120s     | 25s      | High     | Professional    |
| large  | 240s     | 40s      | Highest  | Best quality    |

**Recommendation:** Use `large` model with GPU for best quality without significant speed penalty.

### Batch Processing Optimization

For multiple files, process them individually rather than concatenating:

```bash
# Efficient: Process separately (parallel GPU usage)
!./transcribe.py -m large file1.mp3
!./transcribe.py -m large file2.mp3
!./transcribe.py -m large file3.mp3

# Less efficient: Batch mode
!./transcribe.py -b -o combined.txt -m large *.mp3
```

### Memory Management

Monitor GPU memory usage:

```bash
# Check available memory
!nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits

# Clear GPU memory if needed
import torch
torch.cuda.empty_cache()
```

## Performance Tips

### 1. Use Appropriate Model Size

- **T4 GPU (16GB):** Can handle `large` model efficiently
- **A100 GPU (40GB):** Can handle multiple `large` models simultaneously

### 2. Audio File Optimization

Preprocess audio for better performance:

```bash
# Convert to optimal format (16kHz, mono)
!ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav
!./transcribe.py -m large output.wav
```

### 3. Concurrent Processing

For multiple files, use shell backgrounding:

```bash
# Process multiple files concurrently
!./transcribe.py -m large file1.mp3 &
!./transcribe.py -m large file2.mp3 &
!./transcribe.py -m large file3.mp3 &
wait
```

## Benchmarking Results

Typical performance on different Colab GPUs:

### T4 GPU (16GB VRAM)
- **10-minute audio file:**
  - tiny: ~15 seconds
  - base: ~25 seconds  
  - small: ~45 seconds
  - medium: ~75 seconds
  - large: ~120 seconds

### A100 GPU (40GB VRAM)
- **10-minute audio file:**
  - tiny: ~8 seconds
  - base: ~12 seconds
  - small: ~20 seconds
  - medium: ~35 seconds
  - large: ~55 seconds

## Troubleshooting GPU Issues

### "CUDA out of memory"
```bash
# Use smaller model
!./transcribe.py -m medium audio.mp3

# Or process shorter segments
!ffmpeg -i long_audio.mp3 -t 600 -c copy segment1.mp3
!./transcribe.py -m large segment1.mp3
```

### "No GPU detected"
1. Check runtime type is set to GPU
2. Restart runtime: `Runtime` → `Restart runtime`
3. Verify with `!nvidia-smi`

### Slow performance despite GPU
1. Ensure CUDA build was successful during setup
2. Check GPU utilization: `!nvidia-smi -l 1`
3. Try rebuilding with CUDA support

## Advanced Optimizations

### Custom Build Flags

For maximum performance, the setup script uses:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DGGML_CUDA=1 \
         -DCMAKE_CUDA_ARCHITECTURES="75;80;86" \
         -DGGML_NATIVE=ON
```

### Model Caching

Models are cached after first download:
- Location: `./models/`
- Reused across sessions if persistent storage is enabled

### Persistent Storage

To keep models between sessions:

```python
from google.colab import drive
drive.mount('/content/drive')

# Symlink models to Drive
!ln -sf /content/drive/MyDrive/whisper_models ./models
```

## Expected Performance Gains

With GPU acceleration, expect:

- **3-5x faster** transcription compared to CPU
- Ability to use `large` model in reasonable time
- Better accuracy without significant time penalty
- Efficient processing of long audio files (>1 hour)

## Best Practices

1. **Always use GPU runtime** for transcription tasks
2. **Start with `large` model** - the quality improvement is worth it
3. **Process files individually** for better GPU utilization
4. **Monitor GPU memory** to avoid out-of-memory errors
5. **Use optimal audio formats** (16kHz, mono) when possible