#!/usr/bin/env python3
"""
Test GPU acceleration for whisper.cpp transcription
"""

import subprocess
import time
import tempfile
import os
from pathlib import Path

def create_test_audio():
    """Create a short test audio file."""
    test_file = "test_gpu.wav"
    
    if Path(test_file).exists():
        return test_file
    
    try:
        # Create 10 seconds of sine wave for testing
        subprocess.run([
            'ffmpeg', '-f', 'lavfi', 
            '-i', 'sine=frequency=440:duration=10:sample_rate=16000',
            '-y', test_file
        ], capture_output=True, check=True)
        
        return test_file
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Could not create test audio (ffmpeg not available)")
        return None

def test_transcription_speed():
    """Test transcription speed to detect GPU acceleration."""
    print("🧪 Testing GPU acceleration...")
    print("=" * 40)
    
    # Check if whisper-cli exists
    whisper_cli = Path("whisper.cpp/build/bin/whisper-cli")
    if not whisper_cli.exists():
        print("❌ whisper-cli not found. Run setup first.")
        return False
    
    # Create test audio
    test_audio = create_test_audio()
    if not test_audio:
        print("❌ Could not create test audio file")
        return False
    
    print(f"📁 Using test audio: {test_audio}")
    
    # Check if base model exists
    base_model = Path("models/ggml-base.bin")
    if not base_model.exists():
        print("📥 Downloading base model...")
        os.makedirs("models", exist_ok=True)
        try:
            subprocess.run([
                'curl', '-L', '-o', str(base_model),
                'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin'
            ], check=True)
        except subprocess.CalledProcessError:
            print("❌ Failed to download model")
            return False
    
    # Test transcription
    print("⏱️  Running transcription test...")
    start_time = time.time()
    
    try:
        result = subprocess.run([
            str(whisper_cli),
            '-m', str(base_model),
            '-f', test_audio,
            '--output-txt'
        ], capture_output=True, text=True, check=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ Transcription completed in {duration:.1f} seconds")
        
        # Analyze performance
        audio_duration = 10  # seconds
        realtime_factor = duration / audio_duration
        
        print(f"📊 Performance analysis:")
        print(f"   Audio duration: {audio_duration}s")
        print(f"   Processing time: {duration:.1f}s")
        print(f"   Realtime factor: {realtime_factor:.2f}x")
        
        if realtime_factor < 0.5:
            print("🚀 Excellent performance - likely GPU accelerated!")
        elif realtime_factor < 1.0:
            print("⚡ Good performance - possible GPU acceleration")
        elif realtime_factor < 2.0:
            print("💻 Moderate performance - likely CPU processing")
        else:
            print("🐌 Slow performance - check setup")
        
        # Check for CUDA in output
        if "CUDA" in result.stderr or "cuda" in result.stderr:
            print("🎯 CUDA detected in output - GPU acceleration confirmed!")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Transcription failed: {e}")
        print(f"   stderr: {e.stderr}")
        return False
    
    finally:
        # Clean up test file
        if Path(test_audio).exists():
            Path(test_audio).unlink()

def main():
    """Run GPU acceleration test."""
    print("🔬 GPU Acceleration Test for whisper.cpp")
    print("=" * 50)
    
    # Check GPU availability
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected")
            # Extract GPU info
            lines = result.stdout.split('\n')
            for line in lines:
                if any(gpu in line for gpu in ['Tesla', 'RTX', 'GTX', 'A100', 'T4']):
                    gpu_info = line.split('|')[1].strip() if '|' in line else line.strip()
                    print(f"   {gpu_info}")
                    break
        else:
            print("❌ No NVIDIA GPU detected")
    except FileNotFoundError:
        print("❌ nvidia-smi not found")
    
    print()
    
    # Run transcription test
    success = test_transcription_speed()
    
    if success:
        print("\n🎉 Test completed successfully!")
        print("\nTo benchmark with your own audio:")
        print("   python gpu_benchmark.py your_audio.mp3")
    else:
        print("\n❌ Test failed. Check troubleshooting guide.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())