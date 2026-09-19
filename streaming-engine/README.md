# Portable streaming engine

This directory contains the cross-platform C++ transport layer for chunked
video processing. It receives raw RGB frames on stdin and emits processed
frames on stdout, one frame at a time.

The default C++ mode uses nearest-neighbor 2x scaling as a deterministic
transport smoke test. The Python adapter also supports the local Real-ESRGAN
`x2plus` MLX model. In that mode C++ only transports the frame and the Python
coordinator applies the neural model, producing a real 2x output.

## Protocol

For every input frame:

```text
uint32 width (little-endian)
uint32 height (little-endian)
width * height * 3 bytes RGB
```

The output is:

```text
uint32 output_width
uint32 output_height
uint32 payload_size
payload_size bytes RGB
```

## Neural mode

Build the engine and pass `--model x2plus` to the adapter. The MLX environment
must contain the model dependencies and weights are downloaded/converted on
the first run:

```sh
clang++ -std=c++17 streaming-engine/main.cpp -o /tmp/video-upscaler-engine
.apple-silicon-env/bin/python streaming-engine/stream_frames.py \
  /tmp/video-upscaler-engine --width 640 --height 360 \
  --frames 30 --model x2plus < frames.rgb > frames-2x.rgb
```
