#!/usr/bin/env python3
"""Convert the fixed-tile Real-ESRGAN ONNX model to a CoreML package."""

from __future__ import annotations

import argparse
from pathlib import Path

import coremltools as ct
import onnx
import torch
from onnx2torch import convert


def convert_model(source: Path, destination: Path) -> None:
    model = onnx.load(str(source))
    # onnx2torch currently supports Resize through opset 13. The model's
    # Resize nodes use the same attributes and semantics in this graph.
    for opset in model.opset_import:
        opset.version = 13
    temporary = destination.with_suffix(".op13.onnx")
    onnx.save(model, str(temporary))
    try:
        network = convert(str(temporary)).eval()
        example = torch.zeros((1, 3, 128, 128), dtype=torch.float32)
        traced = torch.jit.trace(network, example)
        package = ct.convert(
            traced,
            source="pytorch",
            inputs=[ct.TensorType(name="image", shape=example.shape)],
            outputs=[ct.TensorType(name="upscaled_image")],
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        package.save(str(destination))
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if not args.source.is_file():
        parser.error(f"ONNX model not found: {args.source}")
    convert_model(args.source, args.destination)
    print(f"CoreML model written to {args.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
