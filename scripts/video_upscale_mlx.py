#!/usr/bin/env python3
"""Apple Silicon video upscaling using MLX and the local Real-ESRGAN weights."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MLX_SOURCE = ROOT / "tools" / "real-esrgan-mlx"
sys.path.insert(0, str(MLX_SOURCE))

import mlx.core as mx
import numpy as np
from upscale import load_model, upscale_image


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def video_info(ffprobe: str, source: Path) -> tuple[int, int, str]:
    result = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate", "-of", "json", str(source)],
        capture_output=True, check=True, text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    rate = stream["r_frame_rate"]
    return int(stream["width"]), int(stream["height"]), rate if rate != "0/0" else "30"


def preview_rate(rate: str, target: int) -> tuple[str, bool]:
    return (str(target), True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?")
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--scale", type=int, choices=(2,), default=2)
    parser.add_argument("--limit-seconds", type=float)
    parser.add_argument("--preview-fps", type=int)
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if args.check:
        print(f"MLX source: {MLX_SOURCE}\nFFmpeg: {ffmpeg or 'missing'}\nFFprobe: {ffprobe or 'missing'}")
        return 0 if MLX_SOURCE.is_dir() and ffmpeg and ffprobe else 2
    if not args.input or not args.output or not ffmpeg or not ffprobe:
        print("Error: provide input/output; FFmpeg and FFprobe are required.", file=sys.stderr)
        return 2
    source, destination = args.input.resolve(), args.output.resolve()
    if not source.is_file() or source == destination or destination.exists():
        print("Error: source must exist and output must be new and different.", file=sys.stderr)
        return 2
    destination.parent.mkdir(parents=True, exist_ok=True)
    width, height, rate = video_info(ffprobe, source)
    output_rate, sample_to_30 = preview_rate(rate, args.preview_fps) if args.preview_fps else (rate, False)
    frame_size = width * height * 3
    decode = [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(source)]
    if args.limit_seconds:
        decode += ["-t", str(args.limit_seconds)]
    if sample_to_30:
        decode += ["-vf", f"fps={args.preview_fps}"]
    decode += ["-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    encode = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width * 2}x{height * 2}",
        "-framerate", output_rate, "-i", "pipe:0",
        "-i", str(source), "-map", "0:v:0", "-map", "1:a?",
        "-c:v", "libx264", "-crf", "17", "-preset", "medium",
        "-c:a", "copy", "-shortest", str(destination),
    ]
    decoder = encoder = None
    try:
        model, native_scale = load_model("x2plus", dtype=mx.float16)
        decoder = subprocess.Popen(decode, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        encoder = subprocess.Popen(encode, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        number = 0
        while True:
            raw = decoder.stdout.read(frame_size)
            if not raw:
                break
            if len(raw) != frame_size:
                raise RuntimeError(f"Frame incompleto recebido do FFmpeg: {len(raw)} de {frame_size} bytes")
            array = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).astype(np.float32) / 255.0
            result = upscale_image(model, array, native_scale, tile_size=args.tile, dtype=mx.float16)
            output = np.clip(result * 255, 0, 255).astype(np.uint8)
            encoder.stdin.write(output.tobytes())
            number += 1
            print(f"Frame {number}", flush=True)
        decoder.stdout.close()
        decoder.wait()
        if decoder.returncode:
            raise subprocess.CalledProcessError(decoder.returncode, decode, stderr=decoder.stderr.read())
        encoder.stdin.close()
        encoder.wait()
        if encoder.returncode:
            raise subprocess.CalledProcessError(encoder.returncode, encode, stderr=encoder.stderr.read())
    except subprocess.CalledProcessError as error:
        print(f"Error: command failed ({error.returncode}).", file=sys.stderr)
        return error.returncode or 1
    finally:
        for process in (decoder, encoder):
            if process and process.poll() is None:
                process.kill()
    print(f"Complete: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
