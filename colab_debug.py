#!/usr/bin/env python3
"""
Colab-specific debug script to diagnose transcription issues
"""

import subprocess
import os
from pathlib import Path

def check_environment():
    """Check Colab environment setup."""
    print("🔍 Checking Colab Environment")
    print("=" * 40)
    
    # Check CUDA environment
    cuda_home = os.environ.get('CUDA_HOME', 'Not set')
    print(f"CUDA_HOME: {cuda_home}")
    
    ld_library_path = os.environ.get('LD_LIBRARY_PATH', 'Not set')
    print(f"LD_LIBRARY_PATH: {ld_library_path}")
    
    # Check GPU
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ GPU available")
            # Extract GPU info
            for line in result.stdout.split('\n'):
                if 'Tesla' in line or 'T4' in line or 'A100' in line:
                    print(f"   {line.strip()}")
                    break
        else:
            print("❌ No GPU detected")
    except:
        print("❌ nvidia-smi failed")
    
    print()

def test_whisper_direct():
    """Test whisper.cpp directly with a simple command."""
    print("🧪 Testing whisper.cpp directly")
    print("=" * 40)
    
    whisper_cli = Path("whisper.cpp/build/bin/whisper-cli")
    if not whisper_cli.exists():
        print(f"❌ whisper-cli not found: {whisper_cli}")
        return False
    
    print(f"✅ Found whisper-cli: {whisper_cli}")
    
    # Test help
    try:
        result = subprocess.run([str(whisper_cli), "--help"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Help command works")
            # Show available options
            if "--output-txt" in result.stdout:
                print("   --output-txt option available")
            if "-ot" in result.stdout:
                print("   -ot option available")
            if "-f" in result.stdout:
                print("   -f option available")
        else:
            print("❌ Help command failed")
            print(f"   stderr: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Help test failed: {e}")
        return False
    
    print()
    return True

def test_with_sample_audio(audio_file):
    """Test transcription with provided audio file."""
    print(f"🎵 Testing with audio file: {audio_file}")
    print("=" * 40)
    
    audio_path = Path(audio_file)
    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_file}")
        return False
    
    size_mb = audio_path.stat().st_size / (1024 * 1024)
    print(f"📁 Audio file: {audio_path.name} ({size_mb:.1f} MB)")
    
    # Check model
    model_path = Path("models/ggml-base.bin")
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return False
    
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"🧠 Model: {model_path.name} ({model_size_mb:.1f} MB)")
    
    # Test direct whisper.cpp command
    whisper_cli = Path("whisper.cpp/build/bin/whisper-cli")
    cmd = [
        str(whisper_cli),
        "-m", str(model_path),
        "-f", str(audio_path),
        "-t", "4",
        "-l", "auto"
    ]
    
    print(f"🚀 Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        print(f"📊 Return code: {result.returncode}")
        
        if result.stdout:
            print("📝 Stdout:")
            print(result.stdout)
        
        if result.stderr:
            print("⚠️  Stderr:")
            print(result.stderr)
        
        # Check for output files
        txt_file = audio_path.with_suffix('.txt')
        if txt_file.exists():
            content = txt_file.read_text(encoding='utf-8').strip()
            print(f"✅ Generated text file: {txt_file}")
            print(f"📄 Content ({len(content)} chars):")
            print(f"   {content[:500]}{'...' if len(content) > 500 else ''}")
            return True
        else:
            print("❌ No text file generated")
            
            # List all files in directory to see what was created
            print("📂 Files in directory:")
            for f in audio_path.parent.glob(f"{audio_path.stem}.*"):
                print(f"   {f}")
            
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Command timed out (5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Command failed: {e}")
        return False

def main():
    """Run all debug tests."""
    print("🐛 Colab Debug Tool for whisper.cpp")
    print("=" * 50)
    
    check_environment()
    
    if not test_whisper_direct():
        print("❌ whisper.cpp binary test failed")
        return 1
    
    # Test with audio file if provided
    import sys
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        if test_with_sample_audio(audio_file):
            print("\n✅ All tests passed!")
            return 0
        else:
            print("\n❌ Audio test failed")
            return 1
    else:
        print("💡 To test with audio: python colab_debug.py <audio_file>")
        print("✅ Basic tests passed!")
        return 0

if __name__ == "__main__":
    exit(main())