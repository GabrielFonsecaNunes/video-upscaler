#!/usr/bin/env python3
"""Upscale videos with the ONNX Real-ESRGAN model and FFmpeg pipes."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except ImportError:
    ort = None

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "tools" / "onnx-models"
MODEL_PATH = MODEL_DIR / "real-esrgan-x4plus-128.onnx"
MODEL_URL = (
    "https://huggingface.co/bukuroo/RealESRGAN-ONNX/resolve/main/"
    "real-esrgan-x4plus-128.onnx"
)
MODEL_TILE = 128
MODEL_SCALE = 4


def video_info(ffprobe: str, source: Path) -> tuple[int, int, str]:
    result = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate", "-of", "json", str(source)],
        capture_output=True, check=True, text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    rate = stream["r_frame_rate"]
    return int(stream["width"]), int(stream["height"]), rate if rate != "0/0" else "30"


def ensure_model() -> Path:
    if MODEL_PATH.is_file():
        return MODEL_PATH
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading ONNX model: {MODEL_URL}", file=sys.stderr, flush=True)
    urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def session() -> ort.InferenceSession:
    if ort is None:
        raise RuntimeError("onnxruntime is required for the ONNX backend")
    providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(str(ensure_model()), providers=providers)


def upscale_frame(runner: ort.InferenceSession, raw: bytes, width: int, height: int,
                  output_scale: int) -> bytes:
    image = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).astype(np.float32) / 255.0
    output = np.zeros((height * MODEL_SCALE, width * MODEL_SCALE, 3), dtype=np.float32)
    overlap = 8
    step = MODEL_TILE - overlap * 2
    for top in range(0, height, step):
        for left in range(0, width, step):
            bottom, right = min(top + step, height), min(left + step, width)
            tile_top = max(0, top - overlap)
            tile_left = max(0, left - overlap)
            tile_bottom = min(height, bottom + overlap)
            tile_right = min(width, right + overlap)
            tile = image[tile_top:tile_bottom, tile_left:tile_right]
            pad_h = MODEL_TILE - tile.shape[0]
            pad_w = MODEL_TILE - tile.shape[1]
            pad_mode = "reflect" if tile.shape[0] > 1 and tile.shape[1] > 1 else "edge"
            tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode=pad_mode)
            enhanced = runner.run(None, {"image": tile.transpose(2, 0, 1)[None]})[0][0].transpose(1, 2, 0)
            crop_top = (top - tile_top) * MODEL_SCALE
            crop_left = (left - tile_left) * MODEL_SCALE
            crop_bottom = crop_top + (bottom - top) * MODEL_SCALE
            crop_right = crop_left + (right - left) * MODEL_SCALE
            output[top * MODEL_SCALE:bottom * MODEL_SCALE,
                   left * MODEL_SCALE:right * MODEL_SCALE] = enhanced[
                       crop_top:crop_bottom, crop_left:crop_right]
    if output_scale != MODEL_SCALE:
        resized = Image.fromarray(np.clip(output * 255, 0, 255).astype(np.uint8))
        resized = resized.resize((width * output_scale, height * output_scale), Image.Resampling.LANCZOS)
        return np.asarray(resized).tobytes()
    return np.clip(output * 255, 0, 255).astype(np.uint8).tobytes()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?")
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--scale", type=int, choices=(2, 4), default=4)
    parser.add_argument("--limit-seconds", type=float)
    parser.add_argument("--preview-fps", type=int)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if args.check:
        print(f"ONNX Runtime providers: {ort.get_available_providers()}")
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
    runner = session()
    decoder = subprocess.Popen(decode, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    encoder = subprocess.Popen(encode, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        number = 0
        while raw := decoder.stdout.read(frame_size):
            if len(raw) != frame_size:
                raise RuntimeError("FFmpeg returned an incomplete RGB frame")
            encoder.stdin.write(upscale_frame(runner, raw, width, height, args.scale))
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
