#!/usr/bin/env python3
"""
Audio Transcription Tool using whisper.cpp
A command-line tool for converting audio files to text transcripts.
"""

import argparse
import sys
import os
import subprocess
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any
import os
import shutil

try:
    from mutagen import File as MutagenFile
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False


class ProgressSpinner:
    """Simple progress spinner for long-running operations."""
    
    def __init__(self, message: str = "Processing"):
        self.message = message
        self.spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self.running = False
        self.thread = None
    
    def start(self):
        """Start the spinner in a separate thread."""
        self.running = True
        self.thread = threading.Thread(target=self._spin)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop the spinner."""
        self.running = False
        if self.thread:
            self.thread.join()
        # Clear the spinner line
        print("\r" + " " * (len(self.message) + 10) + "\r", end="", flush=True)
    
    def _spin(self):
        """Internal method to animate the spinner."""
        i = 0
        while self.running:
            char = self.spinner_chars[i % len(self.spinner_chars)]
            print(f"\r   {char} {self.message}...", end="", flush=True)
            time.sleep(0.1)
            i += 1


class TranscriptionError(Exception):
    """Custom exception for transcription-related errors."""
    pass


@dataclass
class TranscriptionConfig:
    """Configuration settings for transcription operations."""
    model_name: str = 'base'
    include_timestamps: bool = False
    batch_mode: bool = False
    custom_output: Optional[str] = None


class FileManager:
    """Handles file validation and output path generation."""
    
    SUPPORTED_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma'}
    
    def get_audio_info(self, file_path: Path) -> Dict[str, Any]:
        """Get audio file information including duration and size."""
        info = {
            'size_mb': file_path.stat().st_size / (1024 * 1024),
            'duration_seconds': None,
            'duration_formatted': None,
            'bitrate': None,
            'sample_rate': None
        }
        
        if MUTAGEN_AVAILABLE:
            try:
                audio_file = MutagenFile(file_path)
                if audio_file and hasattr(audio_file, 'info'):
                    if hasattr(audio_file.info, 'length'):
                        info['duration_seconds'] = audio_file.info.length
                        info['duration_formatted'] = self._format_duration(audio_file.info.length)
                    if hasattr(audio_file.info, 'bitrate'):
                        info['bitrate'] = audio_file.info.bitrate
                    if hasattr(audio_file.info, 'sample_rate'):
                        info['sample_rate'] = audio_file.info.sample_rate
            except Exception:
                pass
        
        return info
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in MM:SS or HH:MM:SS format."""
        if seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes:02d}:{secs:02d}"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def validate_input_files(self, file_paths: List[str]) -> List[Path]:
        """Validates input files exist, are readable, and have correct format."""
        validated_files = []
        
        for file_path in file_paths:
            path = Path(file_path)
            
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if not path.is_file():
                raise ValueError(f"Path is not a file: {file_path}")
            
            if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                raise ValueError(
                    f"Unsupported file format: {path.suffix}. "
                    f"Supported formats: {', '.join(self.SUPPORTED_EXTENSIONS)}"
                )
            
            if not os.access(path, os.R_OK):
                raise ValueError(f"File is not readable: {file_path}")
            
            validated_files.append(path)
        
        return validated_files
    
    def generate_output_path(self, input_path: Path, custom_output: Optional[str] = None, 
                           model_name: Optional[str] = None, include_timestamps: bool = False) -> Path:
        """Generates appropriate output filename for a given input file."""
        if custom_output:
            return Path(custom_output)
        
        stem = input_path.stem
        suffixes = []
        
        if model_name:
            suffixes.append(model_name)
        
        if include_timestamps:
            suffixes.append("timestamps")
        
        if suffixes:
            suffix_str = "-" + "-".join(suffixes)
            return input_path.parent / f"{stem}{suffix_str}.txt"
        else:
            return input_path.with_suffix('.txt')


class WhisperCppTranscriber:
    """Handles audio transcription using whisper.cpp."""
    
    def __init__(self, model_name: str = 'base'):
        """Initialize the transcriber with a Whisper model."""
        self.model_name = model_name
        self.whisper_cpp_path = self._find_whisper_cpp()
        self.model_path = self._ensure_model_exists()
        self.gpu_capabilities = self._detect_gpu_capabilities()
        
        # Print GPU info on first initialization
        if self.gpu_capabilities['cuda']:
            print("GPU acceleration detected (CUDA) - using optimized settings")
    
    def _find_whisper_cpp(self) -> Path:
        """Find the whisper.cpp executable."""
        # Check common locations
        possible_paths = [
            Path("./whisper.cpp/build/bin/whisper-cli"),
            Path("./build/bin/whisper-cli"),
            Path("./whisper-cli"),
            Path("/usr/local/bin/whisper-cli"),
        ]
        
        for path in possible_paths:
            if path.exists() and path.is_file():
                return path
        
        # Try to find in PATH
        try:
            result = subprocess.run(['which', 'whisper-cli'], 
                                  capture_output=True, text=True, check=True)
            return Path(result.stdout.strip())
        except subprocess.CalledProcessError:
            pass
        
        raise TranscriptionError(
            "whisper-cli not found. Please run setup.sh or colab_setup.sh first."
        )
    
    def _detect_gpu_capabilities(self) -> Dict[str, bool]:
        """Detect available GPU acceleration capabilities."""
        capabilities = {
            'cuda': False,
            'vulkan': False,
            'opencl': False
        }
        
        # Check for NVIDIA GPU
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                capabilities['cuda'] = True
        except FileNotFoundError:
            pass
        
        return capabilities
    
    def _ensure_model_exists(self) -> Path:
        """Ensure the model file exists, download if necessary."""
        model_dir = Path("./models")
        model_file = model_dir / f"ggml-{self.model_name}.bin"
        
        if model_file.exists():
            return model_file
        
        # Try to download the model
        print(f"Model {self.model_name} not found. Downloading...")
        
        model_dir.mkdir(exist_ok=True)
        download_script = Path("./models/download-ggml-model.sh")
        
        if not download_script.exists():
            # Create a simple download script
            self._create_download_script(download_script)
        
        try:
            subprocess.run([str(download_script), self.model_name], 
                         check=True, cwd=model_dir.parent)
        except subprocess.CalledProcessError as e:
            raise TranscriptionError(f"Failed to download model {self.model_name}: {e}")
        
        if not model_file.exists():
            raise TranscriptionError(f"Model file not found after download: {model_file}")
        
        return model_file
    
    def _create_download_script(self, script_path: Path):
        """Create a simple model download script."""
        script_content = '''#!/bin/bash
MODEL=${1:-base}
MODEL_FILE="models/ggml-${MODEL}.bin"

if [ -f "$MODEL_FILE" ]; then
    echo "Model $MODEL already exists at $MODEL_FILE"
    exit 0
fi

echo "Downloading model: $MODEL"
mkdir -p models

# Download from Hugging Face
URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-${MODEL}.bin"
curl -L -o "$MODEL_FILE" "$URL"

if [ $? -eq 0 ]; then
    echo "Successfully downloaded $MODEL_FILE"
else
    echo "Failed to download model $MODEL"
    exit 1
fi
'''
        script_path.write_text(script_content)
        script_path.chmod(0o755)

    def transcribe_file(self, input_path: Path, include_timestamps: bool = False, 
                        verbose: bool = False) -> Dict[str, Any]:
        """Transcribe a single audio file using whisper.cpp."""
        print(f"Transcribing: {input_path.name}")
        
        file_manager = FileManager()
        audio_info = file_manager.get_audio_info(input_path)
        
        size_info = f"{audio_info['size_mb']:.1f} MB"
        if audio_info['duration_formatted']:
            duration_info = f", {audio_info['duration_formatted']} duration"
            print(f"   File: {size_info}{duration_info}")
        else:
            print(f"   File: {size_info}")
        
        if verbose and audio_info['bitrate']:
            print(f"   Bitrate: {audio_info['bitrate'] / 1000:.0f} kbps")
            if audio_info['sample_rate']:
                print(f"   Sample rate: {audio_info['sample_rate']} Hz")
        
        start_time = time.time()
        
        cmd = [
            str(self.whisper_cpp_path), "-m", str(self.model_path),
            "-f", str(input_path), "--output-txt"
        ]
        
        if self.gpu_capabilities['cuda']:
            cmd.extend(["-t", str(min(8, os.cpu_count() or 4))])
            try:
                test_result = subprocess.run([str(self.whisper_cpp_path), "--help"], 
                                          capture_output=True, text=True)
                if "--gpu-layers" in test_result.stdout:
                    cmd.extend(["--gpu-layers", "999"])
            except:
                pass
        else:
            cmd.extend(["-t", str(os.cpu_count() or 4)])
            
        if not include_timestamps:
            cmd.extend(["--no-timestamps"])

        cmd.extend(["--no-prints", "--language", "auto"])
        
        if verbose:
            cmd.extend(["--print-progress"])

        spinner = None
        if not verbose:
            spinner = ProgressSpinner("Transcribing")
            spinner.start()
        
        try:
            output_capture = ""

            if verbose:
                print("   Starting transcription...")
                
                if os.name == 'posix' and shutil.which('stdbuf'):
                    cmd = ['stdbuf', '-o0'] + cmd
                
                print(f"   Command: {' '.join(cmd)}")
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    errors='replace'
                )

                if process.stdout:
                    for line in iter(process.stdout.readline, ''):
                        print(line, end='', flush=True)
                        output_capture += line
                    process.stdout.close()
                
                process.wait()
                
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(
                        process.returncode, cmd, output=output_capture
                    )
            else:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
                output_capture = result.stdout + result.stderr

            if spinner:
                spinner.stop()
            
            end_time = time.time()
            duration = end_time - start_time
            completion_message = f"   Transcription completed in {self._format_duration(duration)}"
            print(f"\n{completion_message}" if verbose and output_capture else completion_message)
            
            # ✨ KEY FIX: Clean the progress indicator string from the end of lines.
            lines = output_capture.strip().split('\n')
            log_prefixes = ('whisper_', 'ggml_', 'main:', 'output_', 'log_mel_spectrogram', 'encode_fallback')
            progress_string_marker = 'whisper_print_progress_callback'
            
            transcript_lines = []
            for line in lines:
                # First, discard lines that are exclusively log messages
                if line.strip().startswith(log_prefixes):
                    continue

                # Next, clean any appended progress indicators from the remaining lines
                # This splits the line at the marker and takes the first part (the actual text)
                cleaned_line = line.split(progress_string_marker)[0]

                # Append the cleaned line only if it's not empty
                if cleaned_line.strip():
                    transcript_lines.append(cleaned_line)
            
            text = '\n'.join(transcript_lines).strip()

            language = "unknown"
            if "detected language:" in output_capture.lower():
                for line in output_capture.split('\n'):
                    if "detected language:" in line.lower():
                        language = line.split(':')[-1].strip()
                        break
            
            if verbose:
                print(f"   Language detected: {language}")

            return {
                'text': text,
                'language': language,
                'duration': duration
            }
            
        except subprocess.CalledProcessError as e:
            if spinner:
                spinner.stop()
            error_msg = f"whisper.cpp failed with return code {e.returncode}"
            if e.output and e.output.strip():
                error_msg += f":\n{e.output.strip()}"
            raise TranscriptionError(error_msg)
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in a human-readable way."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.1f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}h {minutes}m {secs:.1f}s"

class TranscriptionService:
    """Main service class that orchestrates the transcription process."""
    
    def __init__(self, config: TranscriptionConfig, verbose: bool = False):
        """Initialize the transcription service with configuration."""
        self.config = config
        self.verbose = verbose
        self.file_manager = FileManager()
        self.transcriber = WhisperCppTranscriber(config.model_name)
    
    def process_files(self, input_files: List[str]) -> List[Dict[str, Any]]:
        """Process multiple audio files for transcription."""
        overall_start_time = time.time()
        
        # Validate input files
        validated_files = self.file_manager.validate_input_files(input_files)
        
        if self.verbose:
            print(f"\nProcessing {len(validated_files)} file(s) with {self.config.model_name} model")
            total_size = sum(f.stat().st_size for f in validated_files) / (1024 * 1024)
            print(f"Total size: {total_size:.1f} MB")
            print()
        
        results = []
        
        # Process each file
        for i, input_path in enumerate(validated_files):
            try:
                if len(validated_files) > 1:
                    print(f"\n[{i+1}/{len(validated_files)}]", end=" ")
                
                result = self.transcriber.transcribe_file(
                    input_path, 
                    self.config.include_timestamps,
                    self.verbose
                )
                
                # Add file info to result
                result['input_file'] = input_path
                result['output_file'] = self.file_manager.generate_output_path(
                    input_path,
                    self.config.custom_output if not self.config.batch_mode else None,
                    self.config.model_name,
                    self.config.include_timestamps
                )
                
                results.append(result)
                
            except TranscriptionError as e:
                print(f"Error: {e}", file=sys.stderr)
                if not self.config.batch_mode:
                    raise
                continue
        
        overall_end_time = time.time()
        overall_duration = overall_end_time - overall_start_time
        
        if len(validated_files) > 1 or self.verbose:
            print(f"\nTotal processing time: {self.transcriber._format_duration(overall_duration)}")
            if results:
                avg_time = overall_duration / len(results)
                print(f"Average time per file: {self.transcriber._format_duration(avg_time)}")
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]]):
        """Save transcription results to output files."""
        if not results:
            print("No results to save.")
            return
        
        try:
            if self.config.batch_mode and self.config.custom_output:
                self._save_concatenated_results(results)
            else:
                self._save_individual_results(results)
        except Exception as e:
            raise TranscriptionError(f"Failed to save results: {e}")
    
    def _save_individual_results(self, results: List[Dict[str, Any]]):
        """Save each result to its own output file."""
        for result in results:
            try:
                if result['text']:
                    # Directly write the cleaned text from the result dictionary.
                    # This is now the single source of truth for the output.
                    with open(result['output_file'], 'w', encoding='utf-8') as f:
                        f.write(result['text'])
                        f.write('\n') # Add a trailing newline for consistency
                    print(f"Saved: {result['output_file']}")
                else:
                    print(f"Warning: No text to save for {result['input_file']}")
            except Exception as e:
                raise TranscriptionError(f"Failed to save {result['output_file']}: {e}")

    def _save_concatenated_results(self, results: List[Dict[str, Any]]):
        """Save all results concatenated into a single file."""
        output_file = Path(self.config.custom_output)
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for i, result in enumerate(results):
                    f.write(f"=== {result['input_file'].name} ===\n")
                    f.write(result['text'])
                    f.write('\n')
                    
                    if i < len(results) - 1:
                        f.write('\n' + '='*50 + '\n\n')
            
            print(f"Saved concatenated results: {output_file}")
            
        except Exception as e:
            raise TranscriptionError(f"Failed to save concatenated results to {output_file}: {e}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert audio files to text transcripts using whisper.cpp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s audio.mp3                    # Basic transcription
  %(prog)s -m large audio.mp3           # Use large model
  %(prog)s -t audio.mp3                 # Include timestamps
  %(prog)s -o transcript.txt audio.mp3  # Custom output filename
  %(prog)s *.mp3                        # Batch process multiple files
  %(prog)s -b -o all.txt *.mp3          # Concatenate all into single file
  %(prog)s -v audio.mp3                 # Verbose output

Supported Models:
  tiny   - Fastest, least accurate (~39MB)
  base   - Good balance (default) (~74MB)
  small  - Better accuracy (~244MB)
  medium - High accuracy (~769MB)
  large  - Best accuracy (~1550MB)
        """
    )
    
    parser.add_argument(
        'files',
        nargs='+',
        help='Audio file(s) to transcribe'
    )
    
    parser.add_argument(
        '-m', '--model',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        default='base',
        help='Whisper model to use (default: base)'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output filename (default: replace extension with .txt)'
    )
    
    parser.add_argument(
        '-t', '--timestamps',
        action='store_true',
        help='Include timestamps in output'
    )
    
    parser.add_argument(
        '-b', '--batch',
        action='store_true',
        help='Batch mode: concatenate multiple files into single output'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed information'
    )
    
    return parser


def main():
    """Main entry point for the transcription tool."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        # Create configuration
        config = TranscriptionConfig(
            model_name=args.model,
            include_timestamps=args.timestamps,
            batch_mode=args.batch,
            custom_output=args.output
        )
        
        if args.verbose:
            print(f"Configuration:")
            print(f"  Model: {config.model_name}")
            print(f"  Timestamps: {config.include_timestamps}")
            print(f"  Batch mode: {config.batch_mode}")
            print(f"  Output: {config.custom_output or 'auto-generate'}")
            print()
        
        # Validate batch mode usage
        if args.batch and len(args.files) == 1:
            print("Warning: Batch mode with single file. Consider using regular mode.")
        
        if args.batch and not args.output:
            print("Error: Batch mode requires -o/--output option")
            sys.exit(1)
        
        # Create service and process files
        service = TranscriptionService(config, args.verbose)
        results = service.process_files(args.files)
        
        if results:
            service.save_results(results)
            print(f"\nTranscription complete! Processed {len(results)} file(s).")
        else:
            print("No files were successfully transcribed.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nTranscription cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()