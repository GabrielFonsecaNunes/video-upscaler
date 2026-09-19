#!/usr/bin/env python3
"""Apple Silicon video upscaling using MLX and the local Real-ESRGAN weights."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MLX_SOURCE = ROOT / "tools" / "real-esrgan-mlx"
sys.path.insert(0, str(MLX_SOURCE))

import mlx.core as mx
import numpy as np
from PIL import Image
from upscale import load_model, upscale_image


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def frame_rate(ffprobe: str, source: Path) -> str:
    result = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "json", str(source)], capture_output=True, check=True, text=True)
    value = json.loads(result.stdout)["streams"][0]["r_frame_rate"]
    return value if value != "0/0" else "30"


def supports_encoder(ffmpeg: str, encoder: str) -> bool:
    result = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, check=True, text=True)
    return any(encoder in line.split() for line in (result.stdout + result.stderr).splitlines())


def encode_command(ffmpeg: str, rate: str, enlarged: Path, source: Path, destination: Path, encoder: str) -> list[str]:
    command = [ffmpeg, "-hide_banner", "-y", "-threads", "0", "-framerate", rate, "-i", str(enlarged / "frame_%08d.png"), "-i", str(source), "-map", "0:v:0", "-map", "1:a?", "-pix_fmt", "yuv420p"]
    use_videotoolbox = encoder == "videotoolbox" or (encoder == "auto" and platform.system() == "Darwin" and supports_encoder(ffmpeg, "h264_videotoolbox"))
    if use_videotoolbox:
        command += ["-c:v", "h264_videotoolbox", "-q:v", "75", "-allow_sw", "1"]
    else:
        command += ["-c:v", "libx264", "-crf", "17", "-preset", "medium"]
    command += ["-c:a", "copy", "-shortest"]
    if destination.suffix.lower() in {".mp4", ".m4v", ".mov"}:
        command += ["-movflags", "+faststart"]
    return command + [str(destination)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?")
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--scale", type=int, choices=(2,), default=2)
    parser.add_argument("--limit-seconds", type=float)
    parser.add_argument("--tile", type=int, default=512, help="Larger tiles are faster but use more unified memory")
    parser.add_argument("--encoder", choices=("auto", "videotoolbox", "x264"), default="auto")
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
    rate = frame_rate(ffprobe, source)
    work = Path(tempfile.mkdtemp(prefix="video-upscaler-mlx-"))
    frames, enlarged = work / "frames", work / "enlarged"
    frames.mkdir(); enlarged.mkdir()
    try:
        extract = [ffmpeg, "-hide_banner", "-y", "-threads", "0", "-i", str(source)]
        if args.limit_seconds:
            extract += ["-t", str(args.limit_seconds)]
        run(extract + ["-map", "0:v:0", "-fps_mode", "passthrough", str(frames / "frame_%08d.png")])
        model, native_scale = load_model("x2plus", dtype=mx.float16)
        for number, frame in enumerate(sorted(frames.glob("*.png")), start=1):
            with Image.open(frame) as image:
                array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            result = upscale_image(model, array, native_scale, tile_size=args.tile, dtype=mx.float16)
            Image.fromarray(np.clip(result * 255, 0, 255).astype(np.uint8), "RGB").save(enlarged / frame.name)
            print(f"Frame {number}", flush=True)
        run(encode_command(ffmpeg, rate, enlarged, source, destination, args.encoder))
    except subprocess.CalledProcessError as error:
        print(f"Error: command failed ({error.returncode}).", file=sys.stderr)
        return error.returncode or 1
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"Complete: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
