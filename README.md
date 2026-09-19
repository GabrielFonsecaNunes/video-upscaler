# Video Upscaler

An offline Codex plugin for improving local videos on macOS, Windows, and Linux. It keeps the original file unchanged, preserves audio, and uses the free Real-ESRGAN NCNN engine instead of metered cloud services.

## Why it is low-cost

- Local processing: video frames never leave the computer.
- No accounts, subscriptions, or API billing.
- A 10-second preview avoids spending time and electricity on an unsuitable full render.
- The default is 2x, which is substantially cheaper than 4x and is appropriate for most 360p–1080p source material.

## Browser extension

The `browser-extension/` folder is a Manifest V3 extension for Chrome, Edge, and recent Firefox versions. It detects `<video>` elements, including the YouTube player, and displays **Prévia 2×** and **2× completo** directly above the video.

The extension never sends frames to a paid web service. It uses the included `native-host/` companion to download the video only after the user clicks a button, then processes it on the same computer. Results are saved in `Videos/Video Upscaler`.

On Apple Silicon, the companion automatically uses the MLX backend in `scripts/video_upscale_mlx.py`. This is the native Metal path and avoids Vulkan. It currently supports the economical 2× profile.

### Install for development

1. Chrome/Edge: open `chrome://extensions`, activate **Developer mode**, choose **Load unpacked**, and select `browser-extension/`.
2. Firefox: open `about:debugging#/runtime/this-firefox`, choose **Load Temporary Add-on**, and select `browser-extension/manifest.json`.
3. Register the native host after loading the extension. In Chrome or Edge, copy its extension ID from the extensions page and run:

   ```sh
   python native-host/install_native_host.py --browser chrome --extension-id YOUR_EXTENSION_ID
   ```

   Replace `chrome` with `edge` for Edge. For Firefox, run `python native-host/install_native_host.py --browser firefox`. On Windows, add the registry entry printed by the same command with `--print-windows-registry`.

Native Messaging requires this browser-specific registration step so a web page can never run programs on the computer directly.

On YouTube, the companion receives the regular page address rather than a short-lived media address, so it can obtain the selected source through the local `yt-dlp` installation.

## Requirements

- Python 3.10+
- FFmpeg and FFprobe
- `realesrgan-ncnn-vulkan` (or `realesrgan-ncnn-py`) with its `models` folder

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

## WebGPU branch

The `webgpu` branch adds a browser-only path. It copies the current video frame into a
WebGPU texture and renders a 2x canvas overlay with linear filtering and a lightweight
sharpening shader. It keeps the original video as the audio and playback clock, so no
download or Native Messaging host is required for this mode.

This first browser path is a GPU shader baseline, not the full Real-ESRGAN neural model.
The model can be added later by converting its weights and convolution layers to an
ONNX/WebGPU or WGSL representation. WebGPU must be enabled in the browser, and DRM
protected media cannot be read by the extension.
