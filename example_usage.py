#!/usr/bin/env python3
"""
Example usage of the transcription tool as a Python module.
"""

import sys
from pathlib import Path
from transcribe import TranscriptionService, TranscriptionConfig

def example_basic_transcription():
    """Example of basic transcription."""
    print("=== Basic Transcription Example ===")
    
    # Create configuration
    config = TranscriptionConfig(
        model_name='base',
        include_timestamps=False,
        batch_mode=False
    )
    
    # Create service
    service = TranscriptionService(config, verbose=True)
    
    # Example file (you would replace this with your actual file)
    audio_files = ['example_audio.mp3']
    
    try:
        # Process files
        results = service.process_files(audio_files)
        
        # Save results
        service.save_results(results)
        
        print("Transcription completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")

def example_batch_transcription():
    """Example of batch transcription with timestamps."""
    print("=== Batch Transcription Example ===")
    
    # Create configuration for batch processing
    config = TranscriptionConfig(
        model_name='small',
        include_timestamps=True,
        batch_mode=True,
        custom_output='combined_transcript.txt'
    )
    
    # Create service
    service = TranscriptionService(config, verbose=True)
    
    # Example files (you would replace these with your actual files)
    audio_files = ['audio1.mp3', 'audio2.mp3', 'audio3.mp3']
    
    try:
        # Process files
        results = service.process_files(audio_files)
        
        # Save results
        service.save_results(results)
        
        print("Batch transcription completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")

def main():
    """Run examples."""
    if len(sys.argv) > 1 and sys.argv[1] == 'batch':
        example_batch_transcription()
    else:
        example_basic_transcription()

if __name__ == "__main__":
    main()