#!/usr/bin/env python3
"""
Test script to verify the transcription tool setup.
"""

import sys
import subprocess
from pathlib import Path

def test_whisper_cpp_build():
    """Test if whisper.cpp is built correctly."""
    whisper_cli = Path("whisper.cpp/build/bin/whisper-cli")
    
    if not whisper_cli.exists():
        print("❌ whisper-cli not found. Run setup.sh first.")
        return False
    
    try:
        result = subprocess.run([str(whisper_cli), "--help"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ whisper-cli is working")
            return True
        else:
            print("❌ whisper-cli failed to run")
            return False
    except Exception as e:
        print(f"❌ Error testing whisper-cli: {e}")
        return False

def test_python_dependencies():
    """Test if Python dependencies are available."""
    try:
        import mutagen
        print("✅ mutagen is available")
        return True
    except ImportError:
        print("❌ mutagen not found. Install with: pip install mutagen")
        return False

def test_transcribe_script():
    """Test if the transcribe script can run."""
    try:
        result = subprocess.run([sys.executable, "transcribe.py", "--help"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ transcribe.py is working")
            return True
        else:
            print("❌ transcribe.py failed to run")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error testing transcribe.py: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Audio Transcription Tool setup...\n")
    
    tests = [
        ("whisper.cpp build", test_whisper_cpp_build),
        ("Python dependencies", test_python_dependencies),
        ("transcribe script", test_transcribe_script),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"Testing {name}...")
        result = test_func()
        results.append(result)
        print()
    
    if all(results):
        print("🎉 All tests passed! The tool is ready to use.")
        print("\nTry it with:")
        print("  ./transcribe.py --help")
    else:
        print("❌ Some tests failed. Please check the setup.")
        sys.exit(1)

if __name__ == "__main__":
    main()