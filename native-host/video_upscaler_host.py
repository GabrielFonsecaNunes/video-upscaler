#!/usr/bin/env python3
"""Native Messaging companion for the Video Upscaler browser extension."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
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
            requested = (directory / unquote(path[len(prefix):])).resolve()
            if requested.parent != directory:
                return str(directory / "__not_found__")
            return str(requested)

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


def output_directory() -> Path:
    """Prefer Videos, but Chrome may not be granted access to that macOS folder."""
    try:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        return OUTPUT
    except OSError:
        fallback = Path.home() / "Downloads" / "Video Upscaler"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def yt_dlp_command() -> list[str]:
    """Find yt-dlp even when Chrome starts the native host with a minimal PATH."""
    candidates = [
        shutil.which("yt-dlp"),
        "/opt/homebrew/bin/yt-dlp",
        "/usr/local/bin/yt-dlp",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return [str(candidate)]
    return [sys.executable, "-m", "yt_dlp"]


def download(message: dict, target: Path) -> Path:
    url = message["videoUrl"]
    page = message.get("pageUrl", "")
    if "youtube.com" in page or "youtu.be" in page:
        limit_seconds = message.get("limitSeconds")
        sections = ["--download-sections", f"*0-{limit_seconds}"] if limit_seconds else []
        subprocess.run(
            [*yt_dlp_command(), "--no-playlist", *sections, "-f",
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
    if len(sys.argv) == 3 and sys.argv[1] == "--serve-file":
        path = Path(sys.argv[2]).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"Arquivo não encontrado: {path}")
        token = os.urandom(16).hex()
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        print(f"http://127.0.0.1:{port}/{token}/{quote(path.name)}", flush=True)
        serve_output(path.parent, token, port)
        return
    if len(sys.argv) == 5 and sys.argv[1] == "--serve":
        serve_output(Path(sys.argv[2]).resolve(), sys.argv[3], int(sys.argv[4]))
        return
    message = read_message()
    if not message or message.get("type") != "UPSCALE_VIDEO":
        send_message({"error": "Unsupported request"})
        return
    try:
        output = output_directory()
        stem = safe_stem(message.get("title", "upscaled-video"))
        job_id = os.urandom(6).hex()
        source = output / f"{stem}-{job_id}-source.mp4"
        limit_seconds = message.get("limitSeconds")
        suffix = f"{int(limit_seconds)}s" if limit_seconds else "2x"
        destination = output / f"{stem}-{suffix}-{job_id}.mp4"
        download(message, source)
        use_mlx = platform.system() == "Darwin" and platform.machine() == "arm64" and MLX_PYTHON.is_file()
        command = [str(MLX_PYTHON) if use_mlx else sys.executable, str(MLX_UPSCALE if use_mlx else UPSCALE), str(source), str(destination), "--scale", str(message.get("scale", 2))]
        if limit_seconds:
            command += ["--limit-seconds", str(limit_seconds)]
        subprocess.run(command, check=True, stdout=sys.stderr)
        send_message({"output": output_url(destination), "path": str(destination)})
    except Exception as error:
        send_message({"error": str(error)})


if __name__ == "__main__":
    main()
