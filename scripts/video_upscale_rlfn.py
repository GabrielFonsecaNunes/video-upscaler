#!/usr/bin/env python3
"""Upscale videos with the RLFN model ("fast mode") and FFmpeg pipes.

RLFN (https://github.com/bytedance/RLFN, Apache License 2.0) won 1st place
in the Runtime track of NTIRE 2022 Efficient Super-Resolution Challenge.
Like EfRLFN it processes the whole frame at once (no 128x128 tiling needed),
but empirically produces noticeably sharper output at a similar speed:
benchmarked on a 640x480 anime frame, ~142ms/frame (x2) and ~145ms/frame
(x4) on MPS -- about as fast as EfRLFN (~135-165ms) but visually close to
the Real-ESRGAN CoreML backend's sharpness in side-by-side crops, unlike
EfRLFN's noticeably softer output.
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

from rlfn.rlfn import RLFN

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "tools" / "onnx-models"
WEIGHTS_URLS = {
    2: "https://github.com/bytedance/RLFN/raw/main/model_zoo/rlfn_x2.pth",
    4: "https://github.com/bytedance/RLFN/raw/main/model_zoo/rlfn_x4.pth",
}


def video_info(ffprobe: str, source: Path) -> tuple[int, int, str]:
    result = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate", "-of", "json", str(source)],
        capture_output=True, check=True, text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    rate = stream["r_frame_rate"]
    return int(stream["width"]), int(stream["height"]), rate if rate != "0/0" else "30"


def weights_path(scale: int) -> Path:
    return MODEL_DIR / f"rlfn-x{scale}.pth"


def ensure_weights(scale: int) -> Path:
    path = weights_path(scale)
    if path.is_file():
        return path
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading RLFN x{scale} weights: {WEIGHTS_URLS[scale]}", file=sys.stderr, flush=True)
    urlretrieve(WEIGHTS_URLS[scale], path)
    return path


def load_network(scale: int) -> tuple[torch.nn.Module, str]:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    network = RLFN(upscale=scale)
    state = torch.load(ensure_weights(scale), map_location="cpu")
    network.load_state_dict(state, strict=True)
    network.eval()
    return network.to(device), device


def upscale_frame(network: torch.nn.Module, device: str, raw: bytes, width: int, height: int) -> bytes:
    image = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
    tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    with torch.no_grad():
        out = network(tensor).clamp(0, 1)
    out = (out[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return out.tobytes()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?")
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--scale", type=int, choices=(2, 4), default=2)
    parser.add_argument("--limit-seconds", type=float)
    parser.add_argument("--preview-fps", type=int)
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
    output_width, output_height = width * args.scale, height * args.scale
    encode = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
              "-pix_fmt", "rgb24", "-s", f"{output_width}x{output_height}", "-framerate", rate,
              "-i", "pipe:0", "-i", str(source), "-map", "0:v:0", "-map", "1:a?",
              "-c:v", "libx264", "-crf", "17", "-preset", "medium", "-c:a", "copy",
              "-shortest", str(destination)]
    frame_size = width * height * 3
    network, device = load_network(args.scale)
    decoder = subprocess.Popen(decode, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    encoder = subprocess.Popen(encode, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        number = 0
        while raw := decoder.stdout.read(frame_size):
            if len(raw) != frame_size:
                raise RuntimeError("FFmpeg returned an incomplete RGB frame")
            encoder.stdin.write(upscale_frame(network, device, raw, width, height))
            number += 1
            print(f"Frame {number}", flush=True)
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
