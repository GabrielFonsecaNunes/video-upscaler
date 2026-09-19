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
import shutil
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
UPSCALE = ROOT / "scripts" / "video_upscale.py"
MLX_UPSCALE = ROOT / "scripts" / "video_upscale_mlx.py"
MLX_PYTHON = ROOT / ".apple-silicon-env" / "bin" / "python"
OUTPUT = Path.home() / "Videos" / "Video Upscaler"
STREAM_ENGINE = ROOT / "streaming-engine" / "video-upscaler-engine"
MAX_MESSAGE_SIZE = 32 * 1024 * 1024
LOG_FILE = OUTPUT / "native-host.log"
TOOL_DIRECTORIES = (
    ROOT / ".apple-silicon-env" / "bin",
    Path("/opt/homebrew/bin"),
    Path("/usr/local/bin"),
    Path("/Library/Frameworks/Python.framework/Versions/3.14/bin"),
    Path("/usr/bin"),
    Path("/bin"),
)


def configure_environment() -> None:
    paths = [str(path) for path in TOOL_DIRECTORIES if path.is_dir()]
    current = os.environ.get("PATH", "")
    os.environ["PATH"] = os.pathsep.join(paths + ([current] if current else []))


def find_executable(name: str) -> str | None:
    for directory in TOOL_DIRECTORIES:
        candidate = directory / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return shutil.which(name)


def log_event(message: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as log:
        log.write(message + "\n")

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
    raw_length = bytearray()
    while len(raw_length) < 4:
        chunk = sys.stdin.buffer.read(4 - len(raw_length))
        if not chunk:
            break
        raw_length.extend(chunk)
    if not raw_length:
        return None
    if len(raw_length) != 4:
        raise RuntimeError("Native Messaging header incompleto")
    length = struct.unpack("<I", raw_length)[0]
    if length > MAX_MESSAGE_SIZE:
        raise RuntimeError(f"Mensagem Native Messaging excede {MAX_MESSAGE_SIZE} bytes")
    payload = bytearray()
    while len(payload) < length:
        chunk = sys.stdin.buffer.read(length - len(payload))
        if not chunk:
            break
        payload.extend(chunk)
    if len(payload) != length:
        raise RuntimeError("Mensagem Native Messaging incompleta")
    return json.loads(bytes(payload).decode("utf-8"))


def send_message(payload: dict) -> None:
    encoded = json.dumps(payload).encode("utf-8")
    if len(encoded) > MAX_MESSAGE_SIZE:
        raise RuntimeError("Resposta Native Messaging excede o limite permitido")
    sys.stdout.buffer.write(struct.pack("<I", len(encoded)) + encoded)
    sys.stdout.buffer.flush()


def safe_stem(title: str) -> str:
    return re.sub(r"[^A-Za-z0-9._ -]+", "_", title).strip(" ._")[:80] or "upscaled-video"


def unique_output(directory: Path, stem: str, suffix: str) -> Path:
    candidate = directory / f"{stem}-{suffix}.mp4"
    if not candidate.exists():
        return candidate
    for number in range(2, 1000):
        candidate = directory / f"{stem}-{suffix}-{number}.mp4"
        if not candidate.exists():
            return candidate
    raise RuntimeError("Não foi possível criar um nome de saída disponível")


def download(message: dict, target: Path) -> Path:
    url = message["videoUrl"]
    page = message.get("pageUrl", "")
    if "youtube.com" in page or "youtu.be" in page:
        yt_dlp = find_executable("yt-dlp")
        if not yt_dlp:
            raise RuntimeError("yt-dlp não encontrado. Verifique a instalação local.")
        subprocess.run(
            [yt_dlp, "--no-playlist", "-f",
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


def ensure_stream_engine() -> Path | None:
    if not STREAM_ENGINE.exists():
        source = ROOT / "streaming-engine" / "main.cpp"
        compiler = shutil.which("clang++") or shutil.which("g++")
        if compiler and source.exists():
            subprocess.run([compiler, "-std=c++17", str(source), "-o", str(STREAM_ENGINE)], check=True)
    if STREAM_ENGINE.exists() and platform.system() == "Darwin":
        subprocess.run(["xattr", "-d", "com.apple.quarantine", str(STREAM_ENGINE)],
                       check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["codesign", "--force", "--sign", "-", str(STREAM_ENGINE)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return STREAM_ENGINE if STREAM_ENGINE.exists() else None


def main() -> None:
    configure_environment()
    log_event(f"start argv={sys.argv!r}")
    if len(sys.argv) == 5 and sys.argv[1] == "--serve":
        serve_output(Path(sys.argv[2]).resolve(), sys.argv[3], int(sys.argv[4]))
        return
    message = read_message()
    log_event(f"message type={message.get('type') if message else None!r}")
    if not message or message.get("type") != "UPSCALE_VIDEO":
        send_message({"error": "Unsupported request"})
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = safe_stem(message.get("title", "upscaled-video"))
    source = OUTPUT / f"{stem}-source.mp4"
    suffix = "preview" if message.get("limitSeconds") else "2x"
    destination = unique_output(OUTPUT, stem, suffix)
    try:
        download(message, source)
        use_mlx = platform.system() == "Darwin" and platform.machine() == "arm64" and MLX_PYTHON.is_file()
        command = [str(MLX_PYTHON) if use_mlx else sys.executable, str(MLX_UPSCALE if use_mlx else UPSCALE), str(source), str(destination), "--scale", str(message.get("scale", 2))]
        if use_mlx:
            stream_engine = ensure_stream_engine()
            if stream_engine is not None:
                command += ["--stream-engine", str(stream_engine)]
        if message.get("limitSeconds"):
            command += ["--limit-seconds", str(message["limitSeconds"])]
            command += ["--preview-fps", "30"]
        subprocess.run(command, check=True, stdout=sys.stderr)
        log_event(f"completed output={destination}")
        send_message({"output": output_url(destination), "path": str(destination)})
    except Exception as error:
        log_event(traceback.format_exc())
        send_message({"error": str(error)})


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Native Host fatal error: {error}", file=sys.stderr, flush=True)
        log_event(traceback.format_exc())
        try:
            send_message({"error": str(error)})
        except Exception:
            pass
