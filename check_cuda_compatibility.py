#!/usr/bin/env python3
"""
Check CUDA compatibility for whisper.cpp build in Google Colab
"""

import subprocess
import sys
import re

def check_nvidia_gpu():
    """Check if NVIDIA GPU is available."""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            # Extract GPU model
            gpu_info = result.stdout
            if 'Tesla T4' in gpu_info:
                return 'T4', '7.5'
            elif 'Tesla V100' in gpu_info:
                return 'V100', '7.0'
            elif 'A100' in gpu_info:
                return 'A100', '8.0'
            else:
                # Try to extract compute capability
                return 'Unknown', '7.5'  # Default to T4 capability
        return None, None
    except FileNotFoundError:
        return None, None

def check_cuda_version():
    """Check CUDA toolkit version."""
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            # Extract version from output
            match = re.search(r'release (\d+\.\d+)', result.stdout)
            if match:
                return match.group(1)
        return None
    except FileNotFoundError:
        return None

def get_recommended_build_flags(gpu_model, cuda_version):
    """Get recommended build flags based on GPU and CUDA version."""
    flags = {
        'cmake_flags': ['-DCMAKE_BUILD_TYPE=Release'],
        'cuda_enabled': False,
        'reason': ''
    }
    
    if not gpu_model or not cuda_version:
        flags['cmake_flags'].extend(['-DGGML_NATIVE=ON', '-DGGML_BLAS=ON'])
        flags['reason'] = 'No GPU or CUDA detected, using CPU optimizations'
        return flags
    
    cuda_ver = float(cuda_version)
    
    # Conservative settings for older CUDA versions
    if cuda_ver < 11.0:
        flags['cmake_flags'].extend(['-DGGML_NATIVE=ON', '-DGGML_BLAS=ON'])
        flags['reason'] = f'CUDA {cuda_version} too old, using CPU build'
    elif cuda_ver < 11.8:
        # Conservative CUDA build for older versions
        flags['cmake_flags'].extend([
            '-DGGML_CUDA=1',
            '-DCMAKE_CUDA_ARCHITECTURES=75',
            '-DGGML_CUDA_F16=OFF'
        ])
        flags['cuda_enabled'] = True
        flags['reason'] = f'CUDA {cuda_version} with conservative settings'
    else:
        # Full CUDA build for newer versions
        if gpu_model == 'T4':
            arch = '75'
        elif gpu_model == 'V100':
            arch = '70'
        elif gpu_model == 'A100':
            arch = '80'
        else:
            arch = '75'  # Default to T4
            
        flags['cmake_flags'].extend([
            '-DGGML_CUDA=1',
            f'-DCMAKE_CUDA_ARCHITECTURES={arch}',
            '-DGGML_CUDA_F16=ON'
        ])
        flags['cuda_enabled'] = True
        flags['reason'] = f'CUDA {cuda_version} with {gpu_model} optimizations'
    
    return flags

def main():
    """Main compatibility check."""
    print("🔍 Checking CUDA compatibility for whisper.cpp...")
    print("=" * 50)
    
    # Check GPU
    gpu_model, compute_cap = check_nvidia_gpu()
    if gpu_model:
        print(f"✅ GPU detected: {gpu_model} (Compute {compute_cap})")
    else:
        print("❌ No NVIDIA GPU detected")
    
    # Check CUDA
    cuda_version = check_cuda_version()
    if cuda_version:
        print(f"✅ CUDA toolkit: {cuda_version}")
    else:
        print("❌ CUDA toolkit not found")
    
    print()
    
    # Get recommendations
    flags = get_recommended_build_flags(gpu_model, cuda_version)
    
    print("📋 Recommended build configuration:")
    print(f"   Reason: {flags['reason']}")
    print(f"   CUDA enabled: {flags['cuda_enabled']}")
    print("   CMake flags:")
    for flag in flags['cmake_flags']:
        print(f"     {flag}")
    
    # Output for shell script consumption
    if len(sys.argv) > 1 and sys.argv[1] == '--shell':
        print("\n# Shell variables:")
        print(f"CUDA_ENABLED={str(flags['cuda_enabled']).lower()}")
        print(f"CMAKE_FLAGS=\"{' '.join(flags['cmake_flags'])}\"")
    
    return 0 if flags['cuda_enabled'] or not gpu_model else 1

if __name__ == "__main__":
    sys.exit(main())