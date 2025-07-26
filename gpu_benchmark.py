#!/usr/bin/env python3
"""
GPU Performance Benchmark for Audio Transcription Tool
"""

import subprocess
import time
import sys
from pathlib import Path

def check_gpu():
    """Check if GPU is available."""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ GPU detected:")
            # Extract GPU name
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Tesla' in line or 'RTX' in line or 'GTX' in line or 'A100' in line or 'T4' in line:
                    gpu_info = line.split('|')[1].strip() if '|' in line else line.strip()
                    print(f"   {gpu_info}")
                    break
            return True
        else:
            print("❌ No GPU detected")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found - no GPU available")
        return False

def benchmark_transcription(audio_file: str, model: str = 'base') -> float:
    """Benchmark transcription speed."""
    if not Path(audio_file).exists():
        print(f"❌ Audio file not found: {audio_file}")
        return 0
    
    print(f"Benchmarking {model} model with {audio_file}...")
    
    start_time = time.time()
    
    try:
        result = subprocess.run([
            './transcribe.py', 
            '-m', model, 
            audio_file
        ], capture_output=True, text=True, check=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ {model} model completed in {duration:.1f} seconds")
        return duration
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Transcription failed: {e}")
        return 0

def create_test_audio():
    """Create a test audio file if none exists."""
    test_file = "test_audio.wav"
    
    if Path(test_file).exists():
        return test_file
    
    print("Creating test audio file...")
    
    # Try to create a simple test audio file
    try:
        # Create 30 seconds of silence for testing
        subprocess.run([
            'ffmpeg', '-f', 'lavfi', '-i', 'anullsrc=r=16000:cl=mono', 
            '-t', '30', '-y', test_file
        ], capture_output=True, check=True)
        
        print(f"✅ Created test audio: {test_file}")
        return test_file
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Could not create test audio file")
        print("Please provide your own audio file for benchmarking")
        return None

def main():
    """Run GPU benchmark."""
    print("🚀 GPU Performance Benchmark for Audio Transcription")
    print("=" * 50)
    
    # Check GPU
    has_gpu = check_gpu()
    print()
    
    # Check if transcribe.py exists
    if not Path('./transcribe.py').exists():
        print("❌ transcribe.py not found. Run setup first.")
        sys.exit(1)
    
    # Get audio file
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        audio_file = create_test_audio()
        if not audio_file:
            print("Usage: python gpu_benchmark.py <audio_file>")
            sys.exit(1)
    
    print(f"Using audio file: {audio_file}")
    print()
    
    # Benchmark different models
    models = ['tiny', 'base', 'small']
    if has_gpu:
        models.extend(['medium', 'large'])  # Add larger models for GPU
    
    results = {}
    
    for model in models:
        duration = benchmark_transcription(audio_file, model)
        if duration > 0:
            results[model] = duration
        print()
    
    # Show results
    if results:
        print("📊 BENCHMARK RESULTS")
        print("=" * 30)
        
        for model, duration in results.items():
            print(f"{model:8s}: {duration:6.1f}s")
        
        if has_gpu and len(results) > 1:
            print()
            print("💡 GPU Performance Tips:")
            print("- Use 'large' model for best accuracy")
            print("- GPU makes larger models practical")
            print("- Process multiple files individually for best performance")
    
    else:
        print("❌ No successful benchmarks completed")

if __name__ == "__main__":
    main()