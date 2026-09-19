#!/usr/bin/env python3
"""Feed RGB frames to the portable C++ streaming engine."""

from __future__ import annotations

import argparse
import contextlib
import struct
import sys
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", type=Path)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--model", choices=("nearest", "x2plus"), default="nearest")
    parser.add_argument("--tile", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame_size = args.width * args.height * 3
    model = None
    upscale_image = None
    native_scale = 2
    if args.model == "x2plus":
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "tools" / "real-esrgan-mlx"))
        import mlx.core as mx
        import numpy as np
        from upscale import load_model, upscale_image as mlx_upscale_image

        with contextlib.redirect_stdout(sys.stderr):
            model, native_scale = load_model("x2plus", dtype=mx.float16)
        upscale_image = mlx_upscale_image
    engine_command = [str(args.engine)]
    if args.model == "x2plus":
        engine_command.append("--passthrough")
    process = subprocess.Popen(engine_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    assert process.stdin is not None
    assert process.stdout is not None
    try:
        for _ in range(args.frames):
            frame = sys.stdin.buffer.read(frame_size)
            if len(frame) != frame_size:
                raise RuntimeError(f"expected {frame_size} input bytes")
            process.stdin.write(struct.pack("<II", args.width, args.height) + frame)
            process.stdin.flush()
            header = process.stdout.read(12)
            if len(header) != 12:
                raise RuntimeError("engine closed before returning a frame")
            output_width, output_height, output_size = struct.unpack("<III", header)
            output = process.stdout.read(output_size)
            if len(output) != output_size:
                raise RuntimeError("incomplete frame returned by engine")
            if args.model == "x2plus":
                array = np.frombuffer(output, dtype=np.uint8).reshape(
                    output_height, output_width, 3
                ).astype(np.float32) / 255.0
                enhanced = upscale_image(
                    model,
                    array,
                    native_scale,
                    tile_size=args.tile,
                    pre_pad=0,
                    dtype=mx.float16,
                )
                enhanced_uint8 = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
                output_height, output_width = enhanced_uint8.shape[:2]
                output = enhanced_uint8.tobytes()
            sys.stdout.buffer.write(output)
            sys.stdout.buffer.flush()
            expected_scale = 2
            if output_width != args.width * expected_scale or output_height != args.height * expected_scale:
                raise RuntimeError(
                    f"engine returned unexpected dimensions: "
                    f"{output_width}x{output_height}, expected "
                    f"{args.width * expected_scale}x{args.height * expected_scale}"
                )
    finally:
        process.stdin.close()
        process.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
