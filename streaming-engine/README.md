# Portable streaming engine

This directory contains the cross-platform C++ transport layer for chunked
video processing. It receives raw RGB frames on stdin and emits 2x frames on
stdout, one frame at a time.

The current implementation uses nearest-neighbor scaling as a deterministic
transport smoke test. The neural backend can replace that operation without
changing the framing protocol.

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
