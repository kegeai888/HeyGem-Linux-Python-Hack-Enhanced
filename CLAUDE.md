# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HeyGem-Linux-Python-Hack is a Python-based digital human project extracted from HeyGem.ai. It creates digital humans by combining audio lip-sync with video faces, running directly on Linux systems without Docker dependencies.

## System Requirements

- **Python Version**: Strictly Python 3.8 only (run.py:17-19 enforces this)
- **OS**: Linux only
- **GPU**: CUDA-enabled NVIDIA GPU with compatible drivers
- **Dependencies**: ONNX Runtime GPU, OpenCV, FFmpeg

## Core Architecture

### Main Components

1. **Digital Human Service** (`service/trans_dh_service.cpython-38-x86_64-linux-gnu.so`)
   - Core face-to-face transformation engine
   - Compiled Python extension (binary)
   - Main processing pipeline for digital human generation

2. **Face Processing Pipeline**:
   - Face detection: `face_detect_utils/` with SCRFD models
   - Face attribute detection: `face_attr_detect/`
   - Face restoration: `pretrain_models/face_lib/face_restore/gfpgan/`
   - Face parsing: `pretrain_models/face_lib/face_parsing/`

3. **Model Infrastructure**:
   - Landmark-to-face generation: `landmark2face_wy/`
   - ONNX models for GPU acceleration
   - WeNet for audio processing: `wenet/`

4. **Utility Modules** (all compiled .so files):
   - `y_utils/`: Core configuration, logging, time utilities
   - `h_utils/`: Custom utilities, request handling
   - `service/`: Main service orchestration

### Application Entry Points

- **Command Line**: `run.py` - Main processing script
- **Web Interface**: `app.py` - Gradio-based web UI
- **Full Pipeline**: `inference_from_text.sh` - Text-to-speech + face synthesis

## Common Development Commands

### Environment Setup
```bash
# Download required models (REQUIRED first step)
bash download.sh

# Check ONNX CUDA compatibility
python check_env/check_onnx_cuda.py
```

### Basic Usage
```bash
# Run with default demo files
python run.py

# Run with custom audio/video (relative paths only)
python run.py --audio_path example/audio.wav --video_path example/video.mp4

# Launch web interface
python app.py
```

### Full Text-to-Speech Pipeline
```bash
# Requires separate tts-fish-speech repository
bash inference_from_text.sh <audio_file> <text_file> <video_file>
```

## Key Technical Details

### Video Processing Pipeline
- Input: Audio file + Video file with face
- Output: Video with lip-sync applied (`/root/outputs/results/`)
- Processing: Frame-by-frame face detection → landmark extraction → face generation → video assembly

### Model Dependencies
All models downloaded via `download.sh` from GitHub releases:
- Face detection: SCRFD, PFPLD models
- Face restoration: GFPGAN
- Digital human: DiNet model
- Audio processing: WeNet model

### Environment Validation
- Use `check_env/check_onnx_cuda.py` to verify ONNX Runtime GPU setup
- CUDA version compatibility is critical (tested: CUDA 11.8.0 + onnxruntime-gpu 1.16.0)

## Common Issues and Solutions

### Multiple Faces Error
Replace face detection model:
```bash
wget https://github.com/Holasyb918/HeyGem-Linux-Python-Hack/releases/download/ckpts_and_onnx/scrfd_10g_kps.onnx
mv face_detect_utils/resources/scrfd_500m_bnkps_shape640x640.onnx face_detect_utils/resources/scrfd_500m_bnkps_shape640x640.onnx.bak
mv scrfd_10g_kps.onnx face_detect_utils/resources/scrfd_500m_bnkps_shape640x640.onnx
```

### CUDA Library Issues
Add CUDA libraries to path:
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Missing Dependencies
```bash
pip install typeguard  # For check_argument_types import error
```

## File Structure Notes

- `example/`: Demo audio/video files for testing
- `result/`: Output directory for generated videos
- Most utility modules are compiled .so files (not editable Python source)
- Configuration is handled through compiled modules in `y_utils/config.cpython-38-x86_64-linux-gnu.so`

## Development Constraints

- Python 3.8 is strictly enforced - other versions will cause runtime failures
- Only relative paths supported for input files
- Most core functionality is in compiled extensions - limited source code modification possible
- GPU acceleration is essential for reasonable performance
- FFmpeg required for final video assembly with audio
- 始终使用中文回复。