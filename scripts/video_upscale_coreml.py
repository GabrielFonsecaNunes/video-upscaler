#!/usr/bin/env python3
"""Run the converted Real-ESRGAN model with Apple's CoreML runtime."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import coremltools as ct
import numpy as np

# Reuse the tile and FFmpeg pipeline when launched from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from video_upscale_onnx import MODEL_SCALE, upscale_frame, video_info

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "tools" / "onnx-models" / "real-esrgan-x4plus-128.mlpackage"


class CoreMLRunner:
    def __init__(self) -> None:
        self.model = ct.models.MLModel(str(MODEL_PATH), compute_units=ct.ComputeUnit.ALL)

    def run(self, _outputs: None, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        result = self.model.predict({"image": inputs["image"]})
        return [result["upscaled_image"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scale", type=int, choices=(2, 4), default=4)
    parser.add_argument("--limit-seconds", type=float)
    parser.add_argument("--preview-fps", type=int)
    args = parser.parse_args()

    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe or not MODEL_PATH.is_dir():
        print("Error: FFmpeg, FFprobe and the converted CoreML model are required.", file=sys.stderr)
        return 2

    width, height, rate = video_info(ffprobe, args.input)
    if args.preview_fps:
        rate = str(args.preview_fps)
    decode = [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(args.input)]
    if args.limit_seconds:
        decode += ["-t", str(args.limit_seconds)]
    if args.preview_fps:
        decode += ["-vf", f"fps={args.preview_fps}"]
    decode += ["-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    output_width, output_height = width * args.scale, height * args.scale
    encode = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{output_width}x{output_height}",
        "-framerate", rate, "-i", "pipe:0", "-i", str(args.input),
        "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-crf", "17",
        "-preset", "medium", "-c:a", "copy", "-shortest", str(args.output),
    ]
    frame_size = width * height * 3
    decoder = subprocess.Popen(decode, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    encoder = subprocess.Popen(encode, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    runner = CoreMLRunner()
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
