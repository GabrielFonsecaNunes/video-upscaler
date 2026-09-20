# Video Upscaler

An offline, universal video-upscaling plugin for macOS, Windows, and Linux. It is designed to work across desktop and mobile browsing experiences through the main modern browsers, while keeping the original file unchanged, preserving audio, and using the free Real-ESRGAN NCNN engine instead of metered cloud services.

The project provides one browser extension experience for the major browsers and a portable local processing layer for different device types. Processing remains on the user's computer whenever the native host is available, so videos do not need to be uploaded to a third-party service.

## Why it is low-cost

- Local processing: video frames never leave the computer.
- No accounts, subscriptions, or API billing.
- A 10-second preview avoids spending time and electricity on an unsuitable full render.
- For direct (non-YouTube) video URLs, previews are trimmed with FFmpeg directly from the remote stream whenever the source supports it, instead of downloading the entire file first; a full download is only used as a fallback or for the full-length render.
- The default is 2x, which is substantially cheaper than 4x and is appropriate for most 360p–1080p source material.

## Browser extension

The `browser-extension/` folder is a universal Manifest V3 extension for Google Chrome, Microsoft Edge, Mozilla Firefox, Brave, Opera, and other Chromium-based browsers. It detects `<video>` elements, including the YouTube player, and displays **Prévia 2×** and **2× completo** directly above the video.

The extension is intended for laptops, desktops, and supported tablet or mobile browser environments. Chromium browsers share the same package, while Firefox uses its compatible Manifest V3 installation flow. Devices that do not support browser extensions can still use the local Python and command-line processing tools directly.

The extension never sends frames to a paid web service. It uses the included `native-host/` companion to download the video only after the user clicks a button, then processes it on the same computer. Results are saved in `Videos/Video Upscaler`.

On Apple Silicon, the companion uses the converted CoreML model in `scripts/video_upscale_coreml.py` when `.coreml-env/` and the `.mlpackage` are available. It falls back to ONNX Runtime in `scripts/video_upscale_onnx.py` otherwise. The model is downloaded once to `tools/onnx-models/`. Frames are decoded and encoded through FFmpeg pipes without a temporary PNG sequence.

### Optional fast mode (EfRLFN, RLFN, NanoVSR)

Selecting **"EfRLFN x2 (turbo)"** or **"EfRLFN x4 (turbo)"** in the model dropdown switches to `scripts/video_upscale_efrlfn.py`, which runs the vendored [EfRLFN](https://github.com/EvgeneyBogatyrev/EfRLFN) model (MIT License, ICLR 2026) via PyTorch/MPS. It processes each frame whole instead of tiling it, so it needs no overlap/stitch logic. Benchmarked on a 640×480 anime frame: ~135ms/frame at 2x (vs. ~970ms for Real-ESRGAN CoreML, ~7x faster) and ~165ms/frame at 4x (vs. ~880ms, ~5x faster) — at the cost of visibly softer, less detailed output at both scales, a real quality/speed trade-off rather than a drop-in replacement. Pretrained weights are downloaded once to `tools/onnx-models/efrlfn-x{scale}.pt`.

Selecting **"RLFN x2 (turbo nítido)"** or **"RLFN x4 (turbo nítido)"** switches to `scripts/video_upscale_rlfn.py`, which runs the vendored [RLFN](https://github.com/bytedance/RLFN) model (Apache License 2.0; 1st place, Runtime track, NTIRE 2022 Efficient SR Challenge), also via PyTorch/MPS on the whole frame. It runs at a similar speed to EfRLFN (~142ms/frame at 2x, ~145ms/frame at 4x on the same test frame — about as fast, ~6x faster than Real-ESRGAN CoreML) but produces noticeably sharper output in side-by-side crops, close to the Real-ESRGAN backend's detail level. It is the better fast-mode default when both speed and sharpness matter. Pretrained weights are downloaded once to `tools/onnx-models/rlfn-x{scale}.pth`.

Selecting **"NanoVSR x4 (turbo temporal)"** switches to `scripts/video_upscale_nanovsr.py`, which runs the vendored [NanoVSR](https://github.com/filippawlicki/nanovsr) model (MIT License, ECCV 2026). Unlike EfRLFN/RLFN, NanoVSR is a genuine *video* super-resolution model: it processes 15-frame chunks with bidirectional recurrent propagation instead of upscaling each frame in isolation, so the FFmpeg pipe buffers frames into chunks rather than streaming one at a time. It was the fastest backend tested (~93ms/frame on the same 640×480 test frame, ~9x faster than Real-ESRGAN CoreML — even faster than EfRLFN/RLFN), with quality close to RLFN's in both a static crop test and a real 3-second clip with genuine motion. x4 only (the released checkpoint doesn't support x2). Pretrained weights (the 226k-parameter variant) are downloaded once to `tools/onnx-models/nanovsr-226k.pth`.

For a direct HTTP(S) media URL, the native host passes the source straight to FFmpeg instead of downloading it first: NanoVSR begins decoding and processing temporal chunks while the remote transfer continues. The final MP4 is still created after encoding finishes. This requires a server that supports HTTP byte-range requests (the usual requirement for browser-playable MP4 files). YouTube keeps its existing `yt-dlp` download path because the page URL is not a playable media stream.

### Experimental in-browser pipeline

The extension also has a **"No navegador"** button that upscales entirely inside the tab, with no native host, FFmpeg, or Real-ESRGAN involved. It is implemented in `browser-extension/browser-pipeline.js`: it re-renders the `<video>` onto a canvas at 2×/4× (WebGPU when available, Canvas 2D otherwise) and records the result with `MediaRecorder`, reusing the original audio track.

This is a resampling pipeline, not a neural super-resolution model, so quality is lower than the native Real-ESRGAN backends. It exists as an alternative "Camada 3" for users without the native companion installed, and as a base for a future WebGPU compute-shader or WebNN/ONNX Runtime Web model — see `upscaleFrameWebGPU`-equivalent code in that file. The native pipeline remains the default; this backend is opt-in per click.

## Hybrid streaming branch

The `streaming-hybrid` branch adds a portable C++ streaming engine under
`streaming-engine/`. It uses a framed stdin/stdout protocol so a JavaScript or
Python coordinator can send one RGB frame at a time and receive its processed
2× frame without temporary files. The current engine uses nearest-neighbor
scaling as a transport smoke test; the neural backend is intentionally isolated
behind the same protocol for later ONNX Runtime, NCNN, Vulkan, or WebGPU
integration.

The extension sends `enableNativeEngine: true` on every request. The native
host only honors it for the `x2plus` model on Apple Silicon **when the CoreML
and ONNX Runtime backends are unavailable**, since only the direct MLX script
(`scripts/video_upscale_mlx.py`) accepts `--stream-engine`; see the
`use_mlx_direct` check in `native-host/video_upscaler_host.py`. When that
path runs, frames flow from FFmpeg through the C++ engine and MLX
(`realesrgan-x2plus`) without ever touching disk as a PNG sequence. On a
machine where CoreML is configured (the default once `.coreml-env/` and the
converted `.mlpackage` exist), the CoreML backend is used instead and the
streaming engine is not invoked.

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

YouTube requires yt-dlp to solve a JS-based signature challenge before it will return a working video URL. The native host automatically adds `--js-runtimes node --remote-components ejs:github` when a Node.js binary is available, since without it yt-dlp fails with "This video is not available" even for public videos. Install [Node.js](https://nodejs.org/) locally if `yt-dlp` still cannot fetch YouTube sources.

## Requirements

- Python 3.10+
- FFmpeg and FFprobe
- `realesrgan-ncnn-vulkan` (or `realesrgan-ncnn-py`) with its `models` folder
- `onnxruntime`, `onnx`, and `Pillow` for the Apple Silicon ONNX backend
- `coremltools`, `onnx2torch`, and PyTorch in Python 3.12 for optional CoreML conversion

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

### CoreML conversion on Apple Silicon

CoreML conversion is a one-time build step. Use a Python 3.12 environment because
the native `coremltools` runtime is not available in every Python version:

```sh
python scripts/convert_onnx_to_coreml.py \
  tools/onnx-models/real-esrgan-x4plus-128.onnx \
  tools/onnx-models/real-esrgan-x4plus-128.mlpackage
```

The generated `.mlpackage` uses FP16 weights and a fixed 128×128 tile input.
