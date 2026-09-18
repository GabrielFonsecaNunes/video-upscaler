---
name: video-upscaling
description: Improve a local video with Real-ESRGAN on macOS, Windows, or Linux.
---

# Video Upscaling

Process videos locally. Do not overwrite the source file, and always inspect the input before rendering.

## Low-cost workflow

1. Run `python scripts/video_upscale.py --check`.
2. Check duration, resolution, and available disk space with FFprobe.
3. Default to a 2x scale. For lengthy, compressed, or unknown inputs, first render ten seconds:
   `python scripts/video_upscale.py input.mp4 preview-2x.mp4 --scale 2 --limit-seconds 10`
4. After approval, process the full file:
   `python scripts/video_upscale.py input.mp4 output-2x.mp4 --scale 2`

## Cross-platform setup

- macOS: install FFmpeg with Homebrew and place the official macOS Real-ESRGAN NCNN executable in `tools/` or add it to PATH.
- Windows: install FFmpeg and place `realesrgan-ncnn-vulkan.exe` plus its `models/` folder in `tools/`, or add it to PATH.
- Linux: install FFmpeg from the system package manager and add the Real-ESRGAN NCNN executable to PATH or `tools/`.

Never silently replace AI enhancement with ordinary resizing. Audio is copied when the output container supports it.
