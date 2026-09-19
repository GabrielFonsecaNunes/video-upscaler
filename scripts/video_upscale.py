#!/usr/bin/env python3
"""Upscale local videos with Real-ESRGAN while preserving the original and audio."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def find_executable(*names: str) -> str | None:
    """Look on PATH first, then in the plugin's optional tools directory."""
    tools = Path(__file__).resolve().parents[1] / "tools"
    suffixes = (".exe", "") if platform.system() == "Windows" else ("",)
    for name in names:
        if found := shutil.which(name):
            return found
        for suffix in suffixes:
            candidate = tools / f"{name}{suffix}"
            if candidate.is_file():
                return str(candidate)
    return None


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def dependencies() -> tuple[str, str, str]:
    ffmpeg = find_executable("ffmpeg")
    ffprobe = find_executable("ffprobe")
    realesrgan = find_executable("realesrgan-ncnn-vulkan", "realesrgan-ncnn-py")
    missing = [label for label, value in {
        "FFmpeg": ffmpeg, "FFprobe": ffprobe, "Real-ESRGAN": realesrgan
    }.items() if not value]
    if missing:
        system = platform.system()
        raise RuntimeError(
            "Missing: " + ", ".join(missing) + ". Install FFmpeg and a local "
            f"Real-ESRGAN build for {system}, or place its executable in tools/."
        )
    return ffmpeg, ffprobe, realesrgan


def frame_rate(ffprobe: str, source: Path) -> str:
    result = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=r_frame_rate", "-of", "json", str(source)],
        check=True, text=True, capture_output=True,
    )
    value = json.loads(result.stdout)["streams"][0]["r_frame_rate"]
    return value if value != "0/0" else "30"


def preview_rate(rate: str, target: int) -> tuple[str, bool]:
    return (str(target), True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?", help="Source video")
    parser.add_argument("output", type=Path, nargs="?", help="New output video")
    parser.add_argument("--scale", choices=(2, 3, 4), type=int, default=2)
    parser.add_argument("--model", default="realesrgan-x4plus")
    parser.add_argument("--limit-seconds", type=float, help="Process only an initial preview")
    parser.add_argument("--preview-fps", type=int)
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        ffmpeg, ffprobe, realesrgan = dependencies()
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    if args.check:
        print(f"Platform: {platform.system()}\nFFmpeg: {ffmpeg}\nFFprobe: {ffprobe}\nReal-ESRGAN: {realesrgan}")
        return 0
    if not args.input or not args.output:
        print("Error: provide input and output, or use --check.", file=sys.stderr)
        return 2

    source, destination = args.input.resolve(), args.output.resolve()
    if not source.is_file():
        print(f"Error: source not found: {source}", file=sys.stderr)
        return 2
    if source == destination:
        print("Error: output must be different from the source.", file=sys.stderr)
        return 2
    if destination.exists():
        print(f"Error: output already exists: {destination}", file=sys.stderr)
        return 2
    destination.parent.mkdir(parents=True, exist_ok=True)

    rate = frame_rate(ffprobe, source)
    output_rate, sample_to_30 = preview_rate(rate, args.preview_fps) if args.preview_fps else (rate, False)
    work = Path(tempfile.mkdtemp(prefix="video-upscaler-"))
    frames, enlarged = work / "frames", work / "enlarged"
    frames.mkdir(); enlarged.mkdir()
    try:
        extract = [ffmpeg, "-hide_banner", "-y", "-i", str(source)]
        if args.limit_seconds:
            extract += ["-t", str(args.limit_seconds)]
        if sample_to_30:
            extract += ["-vf", "fps=30"]
        run(extract + ["-map", "0:v:0", "-fps_mode", "passthrough", str(frames / "frame_%08d.png")])
        model_dir = Path(realesrgan).resolve().parent / "models"
        enhance = [realesrgan, "-i", str(frames), "-o", str(enlarged), "-n", args.model,
                   "-s", str(args.scale), "-f", "png"]
        if model_dir.is_dir():
            enhance += ["-m", str(model_dir)]
        run(enhance)
        run([ffmpeg, "-hide_banner", "-y", "-framerate", output_rate, "-i",
             str(enlarged / "frame_%08d.png"), "-i", str(source), "-map", "0:v:0",
             "-map", "1:a?", "-c:v", "libx264", "-crf", "17", "-preset", "medium",
             "-c:a", "copy", "-shortest", str(destination)])
    except subprocess.CalledProcessError as error:
        print(f"Error: command failed ({error.returncode}).", file=sys.stderr)
        return error.returncode or 1
    finally:
        if args.keep_frames:
            print(f"Temporary frames: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)
    print(f"Complete: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
