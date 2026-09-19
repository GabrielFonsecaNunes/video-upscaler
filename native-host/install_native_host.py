#!/usr/bin/env python3
"""Register the Video Upscaler Native Messaging host for one local browser."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import shutil
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOST = ROOT / "native-host" / "video_upscaler_host.py"
LAUNCHER = ROOT / "native-host" / "video_upscaler_host_launcher"
NAME = "com.local_video_upscaler"
FIREFOX_ID = "video-upscaler-local@examp3le.com"


def destination(browser: str) -> Path:
    system = platform.system()
    home = Path.home()
    if system == "Darwin":
        bases = {
            "chrome": home / "Library/Application Support/Google/Chrome/NativeMessagingHosts",
            "edge": home / "Library/Application Support/Microsoft Edge/NativeMessagingHosts",
            "firefox": home / "Library/Application Support/Mozilla/NativeMessagingHosts",
        }
    elif system == "Linux":
        bases = {
            "chrome": home / ".config/google-chrome/NativeMessagingHosts",
            "edge": home / ".config/microsoft-edge/NativeMessagingHosts",
            "firefox": home / ".mozilla/native-messaging-hosts",
        }
    else:
        raise RuntimeError("On Windows, run this tool with --print-windows-registry and add the registry value shown.")
    return bases[browser] / f"{NAME}.json"


def manifest(browser: str, extension_id: str) -> dict:
    host_path = LAUNCHER if platform.system() != "Windows" else HOST
    payload = {"name": NAME, "description": "Local Video Upscaler companion", "path": str(host_path), "type": "stdio"}
    payload["allowed_extensions" if browser == "firefox" else "allowed_origins"] = [
        FIREFOX_ID if browser == "firefox" else f"chrome-extension://{extension_id}/"
    ]
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--browser", choices=("chrome", "edge", "firefox"), required=True)
    parser.add_argument("--extension-id", help="Required for Chrome and Edge after loading the extension")
    parser.add_argument("--print-windows-registry", action="store_true")
    args = parser.parse_args()
    if args.browser != "firefox" and not args.extension_id:
        parser.error("--extension-id is required for Chrome and Edge")
    if not HOST.is_file():
        raise RuntimeError(f"Host program not found: {HOST}")
    HOST.chmod(HOST.stat().st_mode | stat.S_IXUSR)
    if platform.system() != "Windows":
        LAUNCHER.write_text(
            f"#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(HOST))} \"$@\"\n"
        )
        LAUNCHER.chmod(0o755)
    if platform.system() == "Windows":
        if not args.print_windows_registry:
            raise RuntimeError("Use --print-windows-registry on Windows.")
        manifest_path = ROOT / "native-host" / f"{NAME}.{args.browser}.json"
        manifest_path.write_text(json.dumps(manifest(args.browser, args.extension_id or ""), indent=2))
        key = {"chrome": "Google\\Chrome", "edge": "Microsoft\\Edge", "firefox": "Mozilla\\NativeMessagingHosts"}[args.browser]
        print(f"Create REG_SZ value HKCU\\Software\\{key}\\NativeMessagingHosts\\{NAME} = {manifest_path}")
        return 0
    target = destination(args.browser)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest(args.browser, args.extension_id or ""), indent=2) + "\n")
    print(f"Registered: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
