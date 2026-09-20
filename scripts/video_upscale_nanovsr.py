#!/usr/bin/env python3
"""Upscale videos with NanoVSR ("fast mode", temporal) and FFmpeg pipes.

NanoVSR (https://github.com/filippawlicki/nanovsr, MIT License, ECCV 2026) is
a bidirectional recurrent video super-resolution model designed for edge
devices. Unlike the other fast-mode backends (EfRLFN, RLFN), it is a real
*video* super-resolution model: it processes a temporal chunk of frames at
once (default T=15) and propagates features forward/backward across the
chunk instead of upscaling each frame in isolation, so it can exploit
inter-frame continuity.

Benchmarked on a 640x480 anime frame (repeated to fill a 15-frame chunk,
so no genuine motion to exploit): ~93ms/frame on MPS with the 226k-parameter
variant -- faster than RLFN (~145ms) and EfRLFN (~165ms), and about 9x
faster than the Real-ESRGAN CoreML backend (~880ms), with quality close to
RLFN's in a static side-by-side crop. Real video with motion was not
benchmarked here; NanoVSR's temporal design may do comparatively better on
genuine motion than the per-frame backends, or worse if scene cuts fall
inside a chunk (each chunk's forward/backward propagation resets at
chunk boundaries, so very short chunk_size values reduce temporal context).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch

from nanovsr.utils import load_model

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "tools" / "onnx-models"
MODEL_SCALE = 4
WEIGHTS_URL = "https://github.com/filippawlicki/nanovsr/releases/download/v1.0/nanovsr_226k.pth"
CHUNK_SIZE = 15


def video_info(ffprobe: str, source: Path) -> tuple[int, int, str]:
    result = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate", "-of", "json", str(source)],
        capture_output=True, check=True, text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    rate = stream["r_frame_rate"]
    return int(stream["width"]), int(stream["height"]), rate if rate != "0/0" else "30"


def weights_path() -> Path:
    return MODEL_DIR / "nanovsr-226k.pth"


def ensure_weights() -> Path:
    path = weights_path()
    if path.is_file():
        return path
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading NanoVSR weights: {WEIGHTS_URL}", file=sys.stderr, flush=True)
    urlretrieve(WEIGHTS_URL, path)
    return path


def load_network() -> tuple[torch.nn.Module, str]:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = load_model(str(ensure_weights()), device=device)
    return model, device


def upscale_chunk(network: torch.nn.Module, device: str, frames: list[np.ndarray]) -> list[bytes]:
    """frames: list of HxWx3 uint8 RGB arrays (already RGB, no BGR<->RGB swap needed
    since these come straight from the FFmpeg rgb24 pipe). Returns list of raw
    RGB bytes at 4x resolution."""
    batch = np.stack(frames, axis=0).transpose(0, 3, 1, 2)
    batch = np.ascontiguousarray(batch)
    tensor = torch.from_numpy(batch).float().div(255.0).unsqueeze(0).to(device)
    with torch.no_grad():
        out = network(tensor)
    out = out[0].clamp(0, 1)
    results = []
    for i in range(out.shape[0]):
        img = (out[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        results.append(img.tobytes())
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?")
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--scale", type=int, choices=(4,), default=4)
    parser.add_argument("--limit-seconds", type=float)
    parser.add_argument("--preview-fps", type=int)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if args.check:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Torch device: {device}")
        print(f"FFmpeg: {ffmpeg or 'missing'}\nFFprobe: {ffprobe or 'missing'}")
        return 0 if ffmpeg and ffprobe else 2
    if not args.input or not args.output or not ffmpeg or not ffprobe:
        print("Error: input/output, FFmpeg and FFprobe are required.", file=sys.stderr)
        return 2
    source, destination = args.input.resolve(), args.output.resolve()
    if not source.is_file() or source == destination or destination.exists():
        print("Error: source must exist and output must be new and different.", file=sys.stderr)
        return 2
    width, height, rate = video_info(ffprobe, source)
    if args.preview_fps:
        rate = str(args.preview_fps)
    decode = [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(source)]
    if args.limit_seconds:
        decode += ["-t", str(args.limit_seconds)]
    if args.preview_fps:
        decode += ["-vf", f"fps={args.preview_fps}"]
    decode += ["-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    output_width, output_height = width * MODEL_SCALE, height * MODEL_SCALE
    encode = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
              "-pix_fmt", "rgb24", "-s", f"{output_width}x{output_height}", "-framerate", rate,
              "-i", "pipe:0", "-i", str(source), "-map", "0:v:0", "-map", "1:a?",
              "-c:v", "libx264", "-crf", "17", "-preset", "medium", "-c:a", "copy",
              "-shortest", str(destination)]
    frame_size = width * height * 3
    network, device = load_network()
    decoder = subprocess.Popen(decode, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    encoder = subprocess.Popen(encode, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        number = 0
        chunk: list[np.ndarray] = []

        def flush_chunk():
            nonlocal number
            if not chunk:
                return
            for output in upscale_chunk(network, device, chunk):
                encoder.stdin.write(output)
                number += 1
                print(f"Frame {number}", flush=True)
            chunk.clear()

        while raw := decoder.stdout.read(frame_size):
            if len(raw) != frame_size:
                raise RuntimeError("FFmpeg returned an incomplete RGB frame")
            image = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
            chunk.append(image)
            if len(chunk) == args.chunk_size:
                flush_chunk()
        flush_chunk()
        decoder.wait()
        if decoder.returncode:
            raise subprocess.CalledProcessError(decoder.returncode, decode, stderr=decoder.stderr.read())
        encoder.stdin.close()
        encoder.wait()
        if encoder.returncode:
            raise subprocess.CalledProcessError(encoder.returncode, encode, stderr=encoder.stderr.read())
    finally:
        for process in (decoder, encoder):
            if process.poll() is None:
                process.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
