#!/usr/bin/env python3
"""Export an MLX safetensors model to a WebGPU tensor package.

The exporter keeps tensors in the NHWC/OHWI layout used by the browser
runtime and writes one contiguous float32 buffer plus a JSON index.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def read_safetensors(path: Path) -> dict[str, tuple[dict, bytes]]:
    with path.open("rb") as source:
        header_size = struct.unpack("<Q", source.read(8))[0]
        header = json.loads(source.read(header_size))
        payload = source.read()
    tensors = {}
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        start, end = metadata["data_offsets"]
        tensors[name] = (metadata, payload[start:end])
    return tensors


def export(source: Path, destination: Path) -> None:
    tensors = read_safetensors(source)
    destination.mkdir(parents=True, exist_ok=True)
    binary = bytearray()
    manifest_tensors = {}

    for name in sorted(tensors):
        metadata, raw = tensors[name]
        if metadata["dtype"] != "F32":
            raise ValueError(f"{name}: expected F32, got {metadata['dtype']}")
        array = np.frombuffer(raw, dtype="<f4")
        aligned_offset = (len(binary) + 3) & ~3
        binary.extend(b"\0" * (aligned_offset - len(binary)))
        offset = len(binary)
        binary.extend(array.tobytes())
        manifest_tensors[name] = {
            "offset": offset,
            "length": len(array),
            "shape": metadata["shape"],
            "layout": "OHWI" if len(metadata["shape"]) == 4 else "vector",
        }

    (destination / "weights.bin").write_bytes(binary)
    manifest = {
        "format": "video-upscaler-webgpu",
        "version": 1,
        "model": source.stem,
        "dtype": "f32",
        "architecture": {
            "name": "SRVGGNetCompact",
            "numInCh": 3,
            "numOutCh": 3,
            "numFeat": 64,
            "numConv": 32,
            "upscale": 4,
        },
        "weights": manifest_tensors,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Exported {len(manifest_tensors)} tensors to {destination}")
    print(f"Binary size: {len(binary):,} bytes")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="MLX safetensors file")
    parser.add_argument("destination", type=Path, help="WebGPU package directory")
    args = parser.parse_args()
    export(args.source, args.destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
