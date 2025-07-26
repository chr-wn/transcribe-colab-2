#!/usr/bin/env python3
"""
Debug script to test whisper.cpp directly and diagnose issues
"""

import subprocess
import sys
from pathlib import Path

def test_whisper_binary():
    """Test the whisper.cpp binary directly."""
    print("🔍 Testing whisper.cpp binary...")
    print("=" * 40)
    
    # Check if binary exists
    whisper_cli = Path("whisper.cpp/build/bin/whisper-cli")
    if not whisper_cli.exists():
        print("❌ whisper-cli not found at expected location")
        print(f"   Expected: {whisper_cli}")
        return False
    
    print(f"✅ Found whisper-cli at: {whisper_cli}")
    
    # Test help command
    print("\n📖 Testing help command...")
    try:
        result = subprocess.run([str(whisper_cli), "--help"], 
                              capture_output=True, text=True, timeout=10)
        print(f"   Return code: {result.returncode}")
        if result.stdout:
            print("   Help output (first 500 chars):")
            print(f"   {result.stdout[:500]}...")
        if result.stderr:
            print("   Stderr:")
            print(f"   {result.stderr}")
        
        if result.returncode != 0:
            print("❌ Help command failed")
            return False
        else:
            print("✅ Help command successful")
            
    except Exception as e:
        print(f"❌ Error running help: {e}")
        return False
    
    return True

def test_model_file():
    """Test if model file exists and is readable."""
    print("\n📁 Testing model file...")
    
    model_path = Path("models/ggml-base.bin")
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return False
    
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"✅ Model found: {model_path} ({size_mb:.1f} MB)")
    
    return True

def test_audio_file(audio_file):
    """Test transcription with a specific audio file."""
    print(f"\n🎵 Testing transcription with: {audio_file}")
    
    audio_path = Path(audio_file)
    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_file}")
        return False
    
    whisper_cli = Path("whisper.cpp/build/bin/whisper-cli")
    model_path = Path("models/ggml-base.bin")
    
    # Simple command
    cmd = [
        str(whisper_cli),
        "-m", str(model_path),
        "-f", str(audio_path),
        "-t", "4"  # 4 threads
    ]
    
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        print("   Running transcription...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        print(f"   Return code: {result.returncode}")
        
        if result.stdout:
            print("   Stdout:")
            print(f"   {result.stdout}")
        
        if result.stderr:
            print("   Stderr:")
            print(f"   {result.stderr}")
        
        # Check for output files
        txt_file = audio_path.with_suffix('.txt')
        if txt_file.exists():
            content = txt_file.read_text(encoding='utf-8')
            print(f"   Generated text file: {txt_file}")
            print(f"   Content ({len(content)} chars): {content[:200]}...")
            return True
        else:
            print("   No text file generated")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Transcription timed out (5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error during transcription: {e}")
        return False

def main():
    """Run debug tests."""
    print("🐛 whisper.cpp Debug Tool")
    print("=" * 50)
    
    # Test binary
    if not test_whisper_binary():
        print("\n❌ Binary test failed. Check setup.")
        return 1
    
    # Test model
    if not test_model_file():
        print("\n❌ Model test failed. Download model first.")
        return 1
    
    # Test with audio file if provided
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        if not test_audio_file(audio_file):
            print(f"\n❌ Audio test failed with {audio_file}")
            return 1
        else:
            print(f"\n✅ Audio test successful with {audio_file}")
    else:
        print("\n💡 To test with audio file: python debug_whisper.py <audio_file>")
    
    print("\n🎉 All tests passed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())