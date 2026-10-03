#!/usr/bin/env python3
"""Build release/OmniBrain-Setup.zip  (needs Pillow for the icon).   python installer/build_zip.py"""
import io
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "omni_brain"
INST = ROOT / "installer"
OUT = ROOT / "release" / "OmniBrain-Setup.zip"
TOP = "OmniBrain-Setup"
APP_FILES = ["omni_brain_mcp.py", "agent.py", "app.py", "browsers.py", "wiki_kb.py", "wiki_ingest.py", "requirements.txt", "README.md", "ui/index.html"]
INSTALLER_FILES = ["Install.bat", "Uninstall.bat", "setup.py", "README-ติดตั้ง.txt"]


def make_icon() -> bytes:
    """Orbit logo (same idea as the app header) on a dark rounded square, as a multi-size .ico."""
    from PIL import Image, ImageDraw

    S = 512
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, S - 1, S - 1), radius=112, fill=(10, 12, 16, 255))
    cyan = (98, 224, 245, 255)
    c = S // 2
    d.ellipse((c - 46, c - 46, c + 46, c + 46), fill=cyan)
    for ang in (60, -60):  # two tilted rings, drawn on a transparent layer then rotated
        ring = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        ImageDraw.Draw(ring).ellipse((40, c - 92, S - 40, c + 92), outline=cyan, width=22)
        img.alpha_composite(ring.rotate(ang, resample=Image.BICUBIC))
    buf = io.BytesIO()
    img.save(buf, format="ICO", sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    return buf.getvalue()


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for name in INSTALLER_FILES:
            z.write(INST / name, f"{TOP}/{name}")
        for name in APP_FILES:
            z.write(SRC / name, f"{TOP}/app/{name}")
        for snap in sorted((SRC / "knowledge").glob("*.json")):  # optional game-knowledge snapshots (not kept in git)
            z.write(snap, f"{TOP}/app/knowledge/{snap.name}")
        z.writestr(f"{TOP}/app/omnibrain.ico", make_icon())
        z.writestr(f"{TOP}/app/BUILD.txt", f"OmniBrain build {git_rev()}\n")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    sys.exit(main())
