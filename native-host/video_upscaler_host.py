#!/usr/bin/env python3
"""Native Messaging companion for the Video Upscaler browser extension."""

from __future__ import annotations

import json
import os
import platform
import re
import socket
import struct
import subprocess
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
UPSCALE = ROOT / "scripts" / "video_upscale.py"
MLX_UPSCALE = ROOT / "scripts" / "video_upscale_mlx.py"
MLX_PYTHON = ROOT / ".apple-silicon-env" / "bin" / "python"
OUTPUT = Path.home() / "Videos" / "Video Upscaler"

class OutputHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return

def serve_output(directory: Path, token: str, port: int) -> None:
    class TokenHandler(OutputHandler):
        def translate_path(self, path: str) -> str:
            prefix = f"/{token}/"
            if not path.startswith(prefix):
                return str(directory / "__not_found__")
            return str(directory / unquote(path[len(prefix):]))

    ThreadingHTTPServer(("127.0.0.1", port), TokenHandler).serve_forever()


def output_url(path: Path) -> str:
    token = os.urandom(16).hex()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--serve",
         str(path.parent), token, str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    for _ in range(40):
        with socket.socket() as ready:
            try:
                ready.connect(("127.0.0.1", port))
                break
            except ConnectionRefusedError:
                import time
                time.sleep(0.05)
    return f"http://127.0.0.1:{port}/{token}/{quote(path.name)}"


def read_message() -> dict | None:
    raw_length = sys.stdin.buffer.read(4)
    if not raw_length:
        return None
    length = struct.unpack("<I", raw_length)[0]
    return json.loads(sys.stdin.buffer.read(length).decode("utf-8"))


def send_message(payload: dict) -> None:
    encoded = json.dumps(payload).encode("utf-8")
    sys.stdout.buffer.write(struct.pack("<I", len(encoded)) + encoded)
    sys.stdout.buffer.flush()


def safe_stem(title: str) -> str:
    return re.sub(r"[^A-Za-z0-9._ -]+", "_", title).strip(" ._")[:80] or "upscaled-video"


def download(message: dict, target: Path) -> Path:
    url = message["videoUrl"]
    page = message.get("pageUrl", "")
    if "youtube.com" in page or "youtu.be" in page:
        subprocess.run(
            [sys.executable, "-m", "yt_dlp", "--no-playlist", "-f",
             "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
             "--merge-output-format", "mp4", "-o", str(target), page],
            check=True, stdout=sys.stderr,
        )
        return target.with_suffix(".mp4")
    request = Request(url, headers={"User-Agent": "Mozilla/5.0", "Referer": page})
    with urlopen(request, timeout=60) as response, target.open("wb") as file:
        while chunk := response.read(1024 * 1024):
            file.write(chunk)
    return target


def main() -> None:
    if len(sys.argv) == 5 and sys.argv[1] == "--serve":
        serve_output(Path(sys.argv[2]).resolve(), sys.argv[3], int(sys.argv[4]))
        return
    message = read_message()
    if not message or message.get("type") != "UPSCALE_VIDEO":
        send_message({"error": "Unsupported request"})
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = safe_stem(message.get("title", "upscaled-video"))
    source = OUTPUT / f"{stem}-source.mp4"
    suffix = "preview" if message.get("limitSeconds") else "2x"
    destination = OUTPUT / f"{stem}-{suffix}.mp4"
    try:
        download(message, source)
        use_mlx = platform.system() == "Darwin" and platform.machine() == "arm64" and MLX_PYTHON.is_file()
        command = [str(MLX_PYTHON) if use_mlx else sys.executable, str(MLX_UPSCALE if use_mlx else UPSCALE), str(source), str(destination), "--scale", str(message.get("scale", 2))]
        if message.get("limitSeconds"):
            command += ["--limit-seconds", str(message["limitSeconds"])]
        subprocess.run(command, check=True, stdout=sys.stderr)
        send_message({"output": output_url(destination), "path": str(destination)})
    except Exception as error:
        send_message({"error": str(error)})


if __name__ == "__main__":
    main()
