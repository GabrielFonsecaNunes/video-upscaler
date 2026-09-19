#!/usr/bin/env python3
"""Feed RGB frames to the portable C++ streaming engine."""

from __future__ import annotations

import argparse
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame_size = args.width * args.height * 3
    process = subprocess.Popen([str(args.engine)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
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
            sys.stdout.buffer.write(output)
            sys.stdout.buffer.flush()
            if output_width != args.width * 2 or output_height != args.height * 2:
                raise RuntimeError("engine returned unexpected dimensions")
    finally:
        process.stdin.close()
        process.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
