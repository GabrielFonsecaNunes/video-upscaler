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
ONNX_UPSCALE = ROOT / "scripts" / "video_upscale_onnx.py"
COREML_UPSCALE = ROOT / "scripts" / "video_upscale_coreml.py"
EFRLFN_UPSCALE = ROOT / "scripts" / "video_upscale_efrlfn.py"
MLX_PYTHON = ROOT / ".apple-silicon-env" / "bin" / "python"
COREML_PYTHON = ROOT / ".coreml-env" / "bin" / "python"
COREML_MODEL = ROOT / "tools" / "onnx-models" / "real-esrgan-x4plus-128.mlpackage"
OUTPUT = Path.home() / "Videos" / "Video Upscaler"
STREAM_ENGINE = ROOT / "streaming-engine" / "video-upscaler-engine"
# "fast mode": ~7x faster than Real-ESRGAN CoreML (no tiling, ~487K params)
# at the cost of softer/less detailed output. See scripts/video_upscale_efrlfn.py.
EFRLFN_MODELS = {"efrlfn_x2": 2, "efrlfn_x4": 4}
MODELS = {"x2plus": 2, "x4plus": 4, "anime_6B": 4, "animevideo": 4, "general": 4, **EFRLFN_MODELS}
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
    limit_seconds = message.get("limitSeconds")
    if "youtube.com" in page or "youtu.be" in page:
        yt_dlp = find_executable("yt-dlp")
        if not yt_dlp:
            raise RuntimeError("yt-dlp não encontrado. Verifique a instalação local.")
        command = [
            yt_dlp, "--no-playlist", "--concurrent-fragments", "8",
            "--newline", "-f",
            "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/"
            "best[ext=mp4][height<=720]/best[ext=mp4]",
            "--merge-output-format", "mp4", "-o", str(target),
        ]
        command += youtube_js_challenge_args()
        if limit_seconds:
            command += ["--download-sections", f"*0-{float(limit_seconds):g}"]
        command.append(page)
        subprocess.run(command, check=True, stdout=sys.stderr)
        return target.with_suffix(".mp4")
    if limit_seconds and download_preview_with_ffmpeg(url, page, float(limit_seconds), target):
        return target
    request = Request(url, headers={"User-Agent": "Mozilla/5.0", "Referer": page})
    with urlopen(request, timeout=60) as response, target.open("wb") as file:
        while chunk := response.read(1024 * 1024):
            file.write(chunk)
    return target


def youtube_js_challenge_args() -> list[str]:
    """YouTube now requires solving a JS "signature/n" challenge before it
    hands out a working videoplayback URL. yt-dlp needs a JS runtime (node,
    since deno is rarely preinstalled) plus its remote challenge-solver
    script; without both, extraction fails with "This video is not
    available" even though the video is public. See
    https://github.com/yt-dlp/yt-dlp/wiki/EJS
    """
    if find_executable("node"):
        return ["--js-runtimes", "node", "--remote-components", "ejs:github"]
    return []


def download_preview_with_ffmpeg(url: str, page: str, limit_seconds: float, target: Path) -> bool:
    """Trim a direct (non-YouTube) video source to a short preview without
    downloading the full file first. FFmpeg stops reading the network stream
    once it has enough packets for --limit-seconds, unlike urlopen() which
    always reads the whole response body.

    Some MP4s store their moov atom at the end of the file (no
    "faststart"), which forces FFmpeg to read the whole stream before it can
    even start demuxing — in that case FFmpeg still exits 0 but writes a
    tiny/invalid file, so success is verified by probing the actual output
    duration rather than trusting the return code alone.

    Returns True on success; False means the caller should fall back to a
    full download (older FFmpeg builds, servers without partial-read
    support, or a moov-at-end / copy-incompatible container).
    """
    ffmpeg = find_executable("ffmpeg")
    ffprobe = find_executable("ffprobe")
    if not ffmpeg or not ffprobe:
        return False
    headers = "User-Agent: Mozilla/5.0\r\n"
    if page:
        headers += f"Referer: {page}\r\n"
    base = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-headers", headers, "-i", url,
            "-t", f"{limit_seconds:g}"]
    for extra in (["-c", "copy"], []):
        subprocess.run(base + extra + [str(target)], capture_output=True, text=True)
        if target.is_file() and _probe_duration(ffprobe, target) > 0:
            return True
        target.unlink(missing_ok=True)
    return False


def _probe_duration(ffprobe: str, path: Path) -> float:
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=15,
        )
        return float(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError):
        return 0.0


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
        use_coreml = use_mlx and COREML_PYTHON.is_file() and COREML_UPSCALE.is_file() and COREML_MODEL.is_dir()
        use_onnx = use_mlx and ONNX_UPSCALE.is_file() and not use_coreml
        model = message.get("model", "x2plus")
        if model not in MODELS:
            raise ValueError(f"Modelo inválido: {model}")
        use_efrlfn = model in EFRLFN_MODELS and EFRLFN_UPSCALE.is_file() and (
            COREML_PYTHON.is_file() or MLX_PYTHON.is_file())
        backend_model = model if use_mlx else {
            "x2plus": "realesrgan-x2plus",
            "x4plus": "realesrgan-x4plus",
            "anime_6B": "realesrgan-x4plus-anime",
            "animevideo": "realesr-animevideov3",
            "general": "realesr-general-x4v3",
        }.get(model)
        if use_efrlfn:
            efrlfn_python = COREML_PYTHON if COREML_PYTHON.is_file() else MLX_PYTHON
            command = [str(efrlfn_python), str(EFRLFN_UPSCALE), str(source), str(destination),
                       "--scale", str(MODELS[model])]
        elif use_coreml:
            command = [str(COREML_PYTHON), str(COREML_UPSCALE), str(source), str(destination),
                       "--scale", str(MODELS[model])]
        elif use_onnx:
            command = [str(MLX_PYTHON), str(ONNX_UPSCALE), str(source), str(destination),
                       "--scale", str(MODELS[model])]
        else:
            command = [str(MLX_PYTHON) if use_mlx else sys.executable,
                       str(MLX_UPSCALE if use_mlx else UPSCALE), str(source), str(destination),
                       "--scale", str(MODELS[model]), "--model", backend_model]
        # streaming-engine/ only wires into the direct MLX invocation below
        # (video_upscale_mlx.py); the CoreML, ONNX and EfRLFN scripts don't
        # accept --stream-engine, so this must stay mutually exclusive with them.
        use_mlx_direct = use_mlx and not use_coreml and not use_onnx and not use_efrlfn
        if use_mlx_direct and model == "x2plus" and message.get("enableNativeEngine") is True:
            stream_engine = ensure_stream_engine()
            if stream_engine is not None:
                command += ["--stream-engine", str(stream_engine)]
        if message.get("limitSeconds"):
            command += ["--limit-seconds", str(message["limitSeconds"])]
            command += ["--preview-fps", "30"]
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.stdout:
            log_event(result.stdout[-8000:])
        if result.stderr:
            log_event(result.stderr[-12000:])
        if result.returncode:
            raise subprocess.CalledProcessError(
                result.returncode, command, output=result.stdout, stderr=result.stderr
            )
        log_event(f"completed output={destination}")
        send_message({"output": output_url(destination), "path": str(destination)})
    except Exception as error:
        log_event(traceback.format_exc())
        detail = str(error)
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            detail = f"{detail}: {error.stderr[-4000:]}"
        send_message({"error": detail})


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
