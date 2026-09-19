# Video Upscaler

An offline, universal video-upscaling plugin for macOS, Windows, and Linux. It is designed to work across desktop and mobile browsing experiences through the main modern browsers, while keeping the original file unchanged, preserving audio, and using the free Real-ESRGAN NCNN engine instead of metered cloud services.

The project provides one browser extension experience for the major browsers and a portable local processing layer for different device types. Processing remains on the user's computer whenever the native host is available, so videos do not need to be uploaded to a third-party service.

## Why it is low-cost

- Local processing: video frames never leave the computer.
- No accounts, subscriptions, or API billing.
- A 10-second preview avoids spending time and electricity on an unsuitable full render.
- The default is 2x, which is substantially cheaper than 4x and is appropriate for most 360p–1080p source material.

## Browser extension

The `browser-extension/` folder is a universal Manifest V3 extension for Google Chrome, Microsoft Edge, Mozilla Firefox, Brave, Opera, and other Chromium-based browsers. It detects `<video>` elements, including the YouTube player, and displays **Prévia 2×** and **2× completo** directly above the video.

The extension is intended for laptops, desktops, and supported tablet or mobile browser environments. Chromium browsers share the same package, while Firefox uses its compatible Manifest V3 installation flow. Devices that do not support browser extensions can still use the local Python and command-line processing tools directly.

The extension never sends frames to a paid web service. It uses the included `native-host/` companion to download the video only after the user clicks a button, then processes it on the same computer. Results are saved in `Videos/Video Upscaler`.

On Apple Silicon, the companion automatically uses ONNX Runtime with the Real-ESRGAN x4 model in `scripts/video_upscale_onnx.py`, using CoreML when available and CPU as fallback. The model is downloaded once to `tools/onnx-models/`. The 2× option runs the x4 model and downsamples its result; frames are decoded and encoded through FFmpeg pipes without a temporary PNG sequence.

## Hybrid streaming branch

The `streaming-hybrid` branch adds a portable C++ streaming engine under
`streaming-engine/`. It uses a framed stdin/stdout protocol so a JavaScript or
Python coordinator can send one RGB frame at a time and receive its processed
2× frame without temporary files. The current engine uses nearest-neighbor
scaling as a transport smoke test; the neural backend is intentionally isolated
behind the same protocol for later ONNX Runtime, NCNN, Vulkan, or WebGPU
integration.

Build it with CMake when available, or compile `streaming-engine/main.cpp`
directly with a C++17 compiler. The Python adapter reads raw RGB frames from
stdin and writes processed RGB frames to stdout:

```sh
clang++ -std=c++17 streaming-engine/main.cpp -o /tmp/video-upscaler-engine
python3 streaming-engine/stream_frames.py /tmp/video-upscaler-engine \
  --width 640 --height 360 --frames 30 < frames.rgb > frames-2x.rgb
```

### Install for development

1. Chrome, Edge, Brave, Opera, and other Chromium browsers: open the browser's extensions page, activate **Developer mode**, choose **Load unpacked**, and select `browser-extension/`.
2. Firefox: open `about:debugging#/runtime/this-firefox`, choose **Load Temporary Add-on**, and select `browser-extension/manifest.json`.
3. On supported mobile browsers, install the extension through the browser's extension store or compatible add-on flow.
4. Register the native host after loading the extension. In Chrome or Edge, copy its extension ID from the extensions page and run:

   ```sh
   python native-host/install_native_host.py --browser chrome --extension-id YOUR_EXTENSION_ID
   ```

   Replace `chrome` with `edge`, `brave`, or `opera` as appropriate. For Firefox, run `python native-host/install_native_host.py --browser firefox`. On Windows, add the registry entry printed by the same command with `--print-windows-registry`.

Native Messaging requires this browser-specific registration step so a web page can never run programs on the computer directly.

On YouTube, the companion receives the regular page address rather than a short-lived media address, so it can obtain the selected source through the local `yt-dlp` installation.

## Requirements

- Python 3.10+
- FFmpeg and FFprobe
- `realesrgan-ncnn-vulkan` (or `realesrgan-ncnn-py`) with its `models` folder
- `onnxruntime`, `onnx`, and `Pillow` for the Apple Silicon ONNX backend

Place executables in the system PATH, or use this layout:

```
tools/
  realesrgan-ncnn-vulkan   # .exe on Windows
  models/
```

## Usage

```sh
python scripts/video_upscale.py --check
python scripts/video_upscale.py input.mp4 preview-2x.mp4 --scale 2 --limit-seconds 10
python scripts/video_upscale.py input.mp4 output-2x.mp4 --scale 2
```

The output must have a different name from the input. For the included 360p trailer, 2x produces 1280×720.
