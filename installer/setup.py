#!/usr/bin/env python3
"""OmniBrain setup (stdlib only; Python 3.10+).

    python setup.py                     # Thai wizard window (falls back to console if Tk is missing)
    python setup.py --silent            # no window, default options
    python setup.py --uninstall [--purge]

Installs per-user (no administrator rights): app files + a private virtualenv, Desktop / Start-menu
shortcuts, an "Apps & features" uninstall entry, and - optionally - registers the MCP server in
Claude Desktop's config WITHOUT touching any other server already there.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import venv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

APP_NAME = "OmniBrain"
VERSION = "1.0.0"
SERVER_NAME = "omni-brain"
HERE = Path(__file__).resolve().parent
IS_WIN = sys.platform == "win32"

CORE_PACKAGES = ["mcp", "pyautogui", "mss", "pillow", "pyperclip"]   # the app cannot work without these
OPTIONAL_PACKAGES = ["pywebview", "keyring", "playwright"]                         # native window / Credential Manager
VERIFY_IMPORTS = ["mss", "PIL", "mcp", "pyperclip", "pyautogui"]

Log = Callable[[str], None]


# ------------------------------------------------------------------ paths
def default_prefix(env: Optional[Dict[str, str]] = None) -> Path:
    env = env if env is not None else os.environ
    if IS_WIN and env.get("LOCALAPPDATA"):
        return Path(env["LOCALAPPDATA"]) / "Programs" / APP_NAME
    return Path.home() / ".local" / "share" / APP_NAME


def data_dir(env: Optional[Dict[str, str]] = None) -> Path:
    """Same rule as app.py, so the app and the Claude Desktop server share ONE memory database."""
    env = env if env is not None else os.environ
    if env.get("OMNI_HOME"):
        return Path(env["OMNI_HOME"])
    root = env.get("APPDATA") if IS_WIN else str(Path.home() / ".config")
    return Path(root or Path.home()) / APP_NAME


def venv_python(prefix: Path, windowed: bool = False) -> Path:
    if IS_WIN:
        return prefix / "venv" / "Scripts" / ("pythonw.exe" if windowed else "python.exe")
    return prefix / "venv" / "bin" / "python"


# ------------------------------------------------------------------ Claude Desktop
class ConfigError(Exception):
    pass


def find_claude_configs(env: Optional[Dict[str, str]] = None) -> List[Path]:
    """Config files of installed Claude Desktop builds (classic installer and Microsoft Store/MSIX).
    A candidate counts if the file exists or Claude's own folder does (config not created yet)."""
    env = env if env is not None else os.environ
    cands: List[Path] = []
    if env.get("APPDATA"):
        cands.append(Path(env["APPDATA"]) / "Claude" / "claude_desktop_config.json")
    if env.get("LOCALAPPDATA"):
        pk = os.path.join(env["LOCALAPPDATA"], "Packages", "Claude_*", "LocalCache", "Roaming", "Claude")
        cands += [Path(d) / "claude_desktop_config.json" for d in glob.glob(pk)]  # folder may exist before the file does
    if sys.platform == "darwin":
        cands.append(Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json")
    seen, out = set(), []
    for c in cands:
        if (c.exists() or c.parent.exists()) and str(c) not in seen:
            seen.add(str(c))
            out.append(c)
    return out


def claude_running() -> bool:
    if not IS_WIN:
        return False
    try:
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq claude.exe", "/NH"], capture_output=True, text=True, timeout=10).stdout
        return "claude.exe" in out.lower()
    except Exception:
        return False


def mcp_entry(prefix: Path, env: Optional[Dict[str, str]] = None) -> Dict[str, object]:
    return {
        "command": str(venv_python(prefix)),
        "args": [str(prefix / "app" / "omni_brain_mcp.py")],
        "env": {"OMNI_BRAIN_DB": str(data_dir(env) / "agent_brain.db")},
    }


def _read_config(path: Path) -> dict:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        return {}
    try:
        data = json.loads(text)
    except ValueError as e:
        raise ConfigError(f"{path} is not valid JSON ({e}); left untouched") from e
    if not isinstance(data, dict):
        raise ConfigError(f"{path} does not contain a JSON object; left untouched")
    return data


def _write_config(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def register_claude(path: Path, entry: Dict[str, object]) -> str:
    data = _read_config(path)
    servers = data.setdefault("mcpServers", {})
    if not isinstance(servers, dict):
        raise ConfigError(f"{path}: 'mcpServers' is not an object; left untouched")
    action = "updated" if SERVER_NAME in servers else "added"
    if path.exists():
        bak = path.with_name(path.name + ".omnibrain.bak")
        if bak.exists():
            bak = path.with_name(f"{path.name}.omnibrain.{int(time.time())}.bak")
        shutil.copy2(path, bak)
    servers[SERVER_NAME] = entry
    _write_config(path, data)
    return action


def unregister_claude(path: Path) -> bool:
    try:
        data = _read_config(path)
    except ConfigError:
        return False
    servers = data.get("mcpServers")
    if isinstance(servers, dict) and SERVER_NAME in servers:
        del servers[SERVER_NAME]
        _write_config(path, data)
        return True
    return False


# ------------------------------------------------------------------ options
@dataclass
class Options:
    prefix: Path = field(default_factory=default_prefix)
    desktop_shortcut: bool = True
    start_menu: bool = True
    register_claude: bool = True
    claude_configs: List[Path] = field(default_factory=find_claude_configs)
    launch_after: bool = True
    skip_pip: bool = False


# ------------------------------------------------------------------ install steps
def _run(cmd: List[str], log: Log, timeout: int = 900) -> int:
    kw = {"creationflags": 0x08000000} if IS_WIN else {}  # CREATE_NO_WINDOW
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", **kw)
    start = time.time()
    assert p.stdout is not None
    for line in p.stdout:
        line = line.rstrip()
        if line:
            log("    " + line[:160])
        if time.time() - start > timeout:
            p.kill()
            log("    (timeout)")
            return 1
    return p.wait()


def _ps_quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def make_shortcuts(opts: Options, log: Log) -> None:
    if not IS_WIN:
        log("ข้ามการสร้างทางลัด (ไม่ใช่ Windows)")
        return
    target = str(venv_python(opts.prefix, windowed=True))
    app_py = str(opts.prefix / "app" / "app.py")
    icon = str(opts.prefix / "app" / "omnibrain.ico")
    where = []
    if opts.desktop_shortcut:
        where.append("Desktop")
    if opts.start_menu:
        where.append("Programs")
    if not where:
        return
    ps = (
        "$ws = New-Object -ComObject WScript.Shell; "
        f"foreach ($k in @({','.join(_ps_quote(w) for w in where)})) {{ "
        "$dir = [Environment]::GetFolderPath($k); "
        f"$l = $ws.CreateShortcut((Join-Path $dir '{APP_NAME}.lnk')); "
        f"$l.TargetPath = {_ps_quote(target)}; $l.Arguments = {_ps_quote(chr(34) + app_py + chr(34))}; "
        f"$l.WorkingDirectory = {_ps_quote(str(opts.prefix / 'app'))}; $l.IconLocation = {_ps_quote(icon)}; "
        f"$l.Description = 'OmniBrain'; $l.Save() }}"
    )
    rc = _run(["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", ps], log, 60)
    log("สร้างทางลัดแล้ว" if rc == 0 else "สร้างทางลัดไม่สำเร็จ (เปิดแอปจาก Uninstall/โฟลเดอร์ติดตั้งแทนได้)")


def register_uninstall_entry(opts: Options, base_python: str, log: Log) -> None:
    if not IS_WIN:
        return
    try:
        import winreg

        cmd = f'"{base_python}" "{opts.prefix / "setup.py"}" --uninstall --prefix "{opts.prefix}"'
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, rf"Software\Microsoft\Windows\CurrentVersion\Uninstall\{APP_NAME}") as k:
            for name, val in (("DisplayName", APP_NAME), ("DisplayVersion", VERSION), ("InstallLocation", str(opts.prefix)),
                              ("UninstallString", cmd), ("Publisher", "OmniBrain"), ("DisplayIcon", str(opts.prefix / "app" / "omnibrain.ico"))):
                winreg.SetValueEx(k, name, 0, winreg.REG_SZ, val)
            winreg.SetValueEx(k, "NoModify", 0, winreg.REG_DWORD, 1)
        log("ลงทะเบียนในรายการ Apps & features แล้ว")
    except Exception as e:
        log(f"ลงทะเบียน Apps & features ไม่สำเร็จ: {e}")


def write_uninstaller(opts: Options, base_python: str) -> None:
    shutil.copy2(HERE / "setup.py", opts.prefix / "setup.py")
    bat = f'@echo off\r\n"{base_python}" "%~dp0setup.py" --uninstall --prefix "%~dp0."\r\n'
    (opts.prefix / "Uninstall.bat").write_text(bat, encoding="ascii", errors="replace")


def copy_app(opts: Options, log: Log) -> None:
    src = HERE / "app"
    if not (src / "app.py").exists():
        raise RuntimeError(f"ไม่พบโฟลเดอร์ app ข้างไฟล์ติดตั้ง ({src}) — แตกไฟล์ Zip ให้ครบก่อน")
    dst = opts.prefix / "app"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "agent_brain.db*"))
    log(f"คัดลอกไฟล์โปรแกรมไปที่ {dst}")


def install(opts: Options, log: Log = print) -> Dict[str, object]:
    """Returns a summary dict; raises on a fatal error."""
    base_python = sys.executable
    opts.prefix.mkdir(parents=True, exist_ok=True)
    log(f"กำลังติดตั้งที่ {opts.prefix}")
    copy_app(opts, log)

    vpy = venv_python(opts.prefix)
    if not vpy.exists():
        log("สร้างสภาพแวดล้อม Python แยกของโปรแกรม (venv)…")
        venv.create(opts.prefix / "venv", with_pip=not opts.skip_pip, clear=False)
    warnings: List[str] = []
    if not opts.skip_pip:
        pip = [str(vpy), "-m", "pip", "install", "--disable-pip-version-check", "--no-input", "--upgrade"]
        log("ติดตั้งไลบรารีหลัก (ต้องต่ออินเทอร์เน็ต อาจใช้เวลา 1-3 นาที)…")
        if _run(pip + CORE_PACKAGES, log) != 0:
            raise RuntimeError("ติดตั้งไลบรารีหลักไม่สำเร็จ — ตรวจอินเทอร์เน็ต/ไฟร์วอลล์ แล้วกดติดตั้งใหม่")
        for pkg in OPTIONAL_PACKAGES:
            log(f"ติดตั้งตัวเสริม {pkg}…")
            if _run(pip + [pkg], log) != 0:
                warnings.append(f"ติดตั้ง {pkg} ไม่สำเร็จ (ข้ามได้: " + ({"pywebview": "แอปจะเปิดในเบราว์เซอร์แทนหน้าต่างเฉพาะ", "keyring": "คีย์ API จะถูกเก็บในไฟล์แทน", "playwright": "ปุ่ม “เรียนรู้จาก Wiki” จะใช้อัปเดตไม่ได้ แต่ความรู้ที่มากับตัวติดตั้งยังใช้ได้"}[pkg]) + ")")
        log("ตรวจสอบไลบรารี…")
        code = "import " + ", ".join(VERIFY_IMPORTS)
        if _run([str(vpy), "-c", code], log, 120) != 0:
            raise RuntimeError("ไลบรารีติดตั้งแล้วแต่เรียกใช้ไม่ได้ (ดูข้อความด้านบน)")

    data_dir().mkdir(parents=True, exist_ok=True)
    write_uninstaller(opts, base_python)
    make_shortcuts(opts, log)
    register_uninstall_entry(opts, base_python, log)

    claude_done: List[str] = []
    if opts.register_claude:
        for cfg in opts.claude_configs:
            try:
                action = register_claude(cfg, mcp_entry(opts.prefix))
                claude_done.append(str(cfg))
                log(f"เชื่อม Claude Desktop ({action}): {cfg}")
            except (ConfigError, OSError) as e:
                warnings.append(f"เชื่อม Claude Desktop ไม่สำเร็จ: {e}")
    (opts.prefix / "install.json").write_text(json.dumps({
        "version": VERSION, "claude_configs": claude_done, "base_python": base_python, "installed_at": int(time.time())}, indent=2), encoding="utf-8")
    log("ติดตั้งเสร็จสมบูรณ์")
    return {"prefix": str(opts.prefix), "claude_configs": claude_done, "warnings": warnings}


def uninstall(prefix: Path, purge: bool = False, log: Log = print, env: Optional[Dict[str, str]] = None) -> None:
    meta: dict = {}
    try:
        meta = json.loads((prefix / "install.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    cfgs = [Path(p) for p in meta.get("claude_configs", [])] or find_claude_configs(env)
    for cfg in cfgs:
        if unregister_claude(cfg):
            log(f"ถอนออกจาก Claude Desktop: {cfg}")
    if IS_WIN:
        ps = ("foreach ($k in 'Desktop','Programs') { $p = Join-Path ([Environment]::GetFolderPath($k)) 'OmniBrain.lnk'; "
              "if (Test-Path $p) { Remove-Item $p -Force } }")
        _run(["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", ps], log, 30)
        try:
            import winreg

            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, rf"Software\Microsoft\Windows\CurrentVersion\Uninstall\{APP_NAME}")
        except Exception:
            pass
    if purge:
        shutil.rmtree(data_dir(env), ignore_errors=True)
        log(f"ลบข้อมูลและการตั้งค่า: {data_dir(env)}")
    if IS_WIN:
        # this script lives inside the folder being deleted: finish the job from a detached helper
        subprocess.Popen(f'cmd /c ping -n 3 127.0.0.1 >nul & rmdir /s /q "{prefix}"', shell=True,
                         creationflags=0x00000008 | 0x08000000, close_fds=True)
    else:
        shutil.rmtree(prefix, ignore_errors=True)
    log("ถอนการติดตั้งเสร็จแล้ว (ข้อมูลความจำของคุณยังอยู่ ถ้าไม่ได้เลือกลบ)")


def launch_app(prefix: Path) -> None:
    kw = {"creationflags": 0x00000008 | 0x08000000} if IS_WIN else {}
    subprocess.Popen([str(venv_python(prefix, windowed=True)), str(prefix / "app" / "app.py")], cwd=str(prefix / "app"), **kw)


# ------------------------------------------------------------------ wizard (Thai)
def run_wizard(opts: Options, on_ready: Optional[Callable] = None) -> int:
    import queue
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    if IS_WIN:
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

    BG, FG, ACC = "#0a0c10", "#e6e9ef", "#62e0f5"
    _style = ttk.Style()
    _style.configure("TCheckbutton", background="#f4f5f7", font=("Segoe UI", 10))

    class Wizard(tk.Tk):
        def __init__(self) -> None:
            super().__init__()
            self.title(f"ติดตั้ง {APP_NAME}")
            self.geometry("680x520")
            self.minsize(640, 480)
            self.configure(bg="#f4f5f7")
            self.q: "queue.Queue" = queue.Queue()
            self.result: Dict[str, object] = {}
            self.error: Optional[str] = None
            self.page = 0
            self.v_prefix = tk.StringVar(value=str(opts.prefix))
            self.v_desktop = tk.BooleanVar(value=opts.desktop_shortcut)
            self.v_menu = tk.BooleanVar(value=opts.start_menu)
            self.v_claude = tk.BooleanVar(value=opts.register_claude and bool(opts.claude_configs))
            self.v_launch = tk.BooleanVar(value=opts.launch_after)
            head = tk.Frame(self, bg=BG, height=74)
            head.pack(fill="x")
            head.pack_propagate(False)
            tk.Label(head, text="◉  OmniBrain", bg=BG, fg=ACC, font=("Segoe UI", 18, "bold")).pack(side="left", padx=24)
            tk.Label(head, text=f"ตัวติดตั้ง v{VERSION}", bg=BG, fg="#8b93a3", font=("Segoe UI", 10)).pack(side="right", padx=24)
            self.body = tk.Frame(self, bg="#f4f5f7")
            self.body.pack(fill="both", expand=True, padx=28, pady=(18, 6))
            nav = tk.Frame(self, bg="#f4f5f7")
            nav.pack(fill="x", padx=28, pady=(0, 18))
            self.b_cancel = ttk.Button(nav, text="ยกเลิก", command=self.destroy)
            self.b_cancel.pack(side="left")
            self.b_next = ttk.Button(nav, text="ถัดไป", command=self.next)
            self.b_next.pack(side="right")
            self.b_back = ttk.Button(nav, text="ย้อนกลับ", command=self.back)
            self.b_back.pack(side="right", padx=8)
            self.show(0)

        def clear(self) -> None:
            for w in self.body.winfo_children():
                w.destroy()

        def label(self, text: str, size: int = 10, bold: bool = False, color: str = "#1b2030", wrap: int = 620, pady=(0, 6)) -> tk.Label:
            lb = tk.Label(self.body, text=text, bg="#f4f5f7", fg=color, justify="left", anchor="w", wraplength=wrap,
                          font=("Segoe UI", size, "bold" if bold else "normal"))
            lb.pack(fill="x", pady=pady)
            return lb

        def show(self, n: int) -> None:
            self.page = n
            self.clear()
            self.b_back.state(["!disabled"] if n in (1,) else ["disabled"])
            self.b_next.state(["!disabled"])
            self.b_cancel.state(["!disabled"])
            [self.p_welcome, self.p_options, self.p_progress, self.p_done][n]()

        # -- pages
        def p_welcome(self) -> None:
            self.label("ยินดีต้อนรับ", 16, True)
            self.label("OmniBrain คือผู้ช่วย AI ที่จำงานได้ ดูหน้าจอได้ และควบคุมเมาส์/คีย์บอร์ดของคุณอย่างมีขอบเขต "
                       "เริ่มต้นเป็นโหมดจำลองเสมอ และจะควบคุมเครื่องจริงเฉพาะเมื่อคุณกดอนุญาตเองเท่านั้น", 11, pady=(0, 12))
            self.label("• ติดตั้งเฉพาะผู้ใช้ปัจจุบัน ไม่ต้องใช้สิทธิ์ผู้ดูแลระบบ\n"
                       "• ต้องต่ออินเทอร์เน็ตเพื่อดาวน์โหลดไลบรารี (ประมาณ 1-3 นาที)\n"
                       "• ถอนการติดตั้งได้จาก Settings ▸ Apps", 10, color="#4a5160")
            self.b_next.config(text="ถัดไป", command=self.next)

        def p_options(self) -> None:
            self.label("ตัวเลือกการติดตั้ง", 14, True)
            self.label("ติดตั้งที่", 10, color="#4a5160", pady=(4, 2))
            row = tk.Frame(self.body, bg="#f4f5f7")
            row.pack(fill="x")
            ttk.Entry(row, textvariable=self.v_prefix).pack(side="left", fill="x", expand=True)
            ttk.Button(row, text="เลือก…", command=lambda: self.v_prefix.set(filedialog.askdirectory() or self.v_prefix.get())).pack(side="left", padx=(8, 0))
            self.label("", pady=(2, 0))
            ttk.Checkbutton(self.body, text="สร้างไอคอนบน Desktop", variable=self.v_desktop).pack(anchor="w", pady=2)
            ttk.Checkbutton(self.body, text="เพิ่มลงในเมนู Start", variable=self.v_menu).pack(anchor="w", pady=2)
            if opts.claude_configs:
                ttk.Checkbutton(self.body, text="เชื่อมกับ Claude Desktop (ใช้ความจำร่วมกับแอปนี้)", variable=self.v_claude).pack(anchor="w", pady=2)
                self.label("พบ Claude Desktop: " + "; ".join(str(c) for c in opts.claude_configs) +
                           "\nระบบจะสำรองไฟล์ตั้งค่าเดิมไว้ และไม่แตะเซิร์ฟเวอร์ MCP อื่นที่มีอยู่", 9, color="#6a7282", pady=(0, 6))
            else:
                self.v_claude.set(False)
                self.label("ไม่พบ Claude Desktop ในเครื่องนี้ — ข้ามการเชื่อม (ติดตั้งใหม่ภายหลังได้)", 9, color="#6a7282")
            ttk.Checkbutton(self.body, text="เปิด OmniBrain เมื่อติดตั้งเสร็จ", variable=self.v_launch).pack(anchor="w", pady=2)
            self.b_next.config(text="ติดตั้ง", command=self.start_install)

        def p_progress(self) -> None:
            self.label("กำลังติดตั้ง…", 14, True)
            self.bar = ttk.Progressbar(self.body, mode="indeterminate")
            self.bar.pack(fill="x", pady=(2, 8))
            self.bar.start(12)
            box = tk.Frame(self.body)
            box.pack(fill="both", expand=True)
            self.txt = tk.Text(box, height=12, bg="#0f131a", fg="#cdd3df", relief="flat", font=("Consolas", 9), wrap="word")
            sb = ttk.Scrollbar(box, command=self.txt.yview)
            self.txt.config(yscrollcommand=sb.set, state="disabled")
            sb.pack(side="right", fill="y")
            self.txt.pack(side="left", fill="both", expand=True)
            for b in (self.b_back, self.b_next, self.b_cancel):
                b.state(["disabled"])

        def p_done(self) -> None:
            ok = self.error is None
            self.label("ติดตั้งเสร็จแล้ว ✓" if ok else "ติดตั้งไม่สำเร็จ", 16, True, color="#0f7a4f" if ok else "#c0392b")
            if ok:
                claude = self.result.get("claude_configs")
                if claude:
                    run = claude_running()
                    self.label(("⚠ Claude Desktop กำลังเปิดอยู่ — " if run else "") +
                               "เชื่อม OmniBrain เข้ากับ Claude Desktop แล้ว ต้องปิด Claude Desktop ให้สนิท "
                               "(คลิกขวาไอคอนที่ถาดระบบ ▸ Quit) แล้วเปิดใหม่ จึงจะเห็นเครื่องมือ omni-brain", 10, pady=(0, 8))
                self.label("วิธีปลดล็อกโหมดควบคุมจริง:\n"
                           "• ในแอป OmniBrain: กดปุ่ม “ปลดล็อก”\n"
                           "• ใน Claude Desktop: ให้ AI เรียก request_live_unlock จะมีกล่องถามบนหน้าจอให้คุณกด Yes\n"
                           "หยุดฉุกเฉิน: ชักเมาส์ไปมุมซ้ายบนของจอใดจอหนึ่ง (ทุกจอใช้ได้)", 10, color="#4a5160", pady=(0, 8))
                for w in self.result.get("warnings", []):  # type: ignore[union-attr]
                    self.label("• " + str(w), 9, color="#9a6700", pady=(0, 2))
                self.b_next.config(text="เสร็จสิ้น", command=self.finish)
            else:
                self.label(str(self.error), 10, color="#c0392b")
                self.b_next.config(text="ปิด", command=self.destroy)
                self.b_back.config(command=lambda: self.show(1))
                self.b_back.state(["!disabled"])
            self.b_cancel.state(["disabled"])

        # -- actions
        def next(self) -> None:
            self.show(self.page + 1)

        def back(self) -> None:
            self.show(max(0, self.page - 1))

        def start_install(self) -> None:
            opts.prefix = Path(self.v_prefix.get()).expanduser()
            opts.desktop_shortcut, opts.start_menu = self.v_desktop.get(), self.v_menu.get()
            opts.register_claude, opts.launch_after = self.v_claude.get(), self.v_launch.get()
            self.show(2)
            threading.Thread(target=self.work, daemon=True).start()
            self.after(100, self.poll)

        def work(self) -> None:
            try:
                self.result = install(opts, lambda m: self.q.put(m))
            except Exception as e:
                self.error = f"{e}"
            self.q.put(None)

        def poll(self) -> None:
            try:
                while True:
                    m = self.q.get_nowait()
                    if m is None:
                        self.bar.stop()
                        self.show(3)
                        return
                    self.txt.config(state="normal")
                    self.txt.insert("end", m + "\n")
                    self.txt.see("end")
                    self.txt.config(state="disabled")
            except queue.Empty:
                pass
            self.after(100, self.poll)

        def finish(self) -> None:
            if opts.launch_after and self.error is None:
                launch_app(opts.prefix)
            self.destroy()

    w = Wizard()
    w.update_idletasks()
    w.geometry(f"+{max(0, (w.winfo_screenwidth() - 680) // 2)}+{max(0, (w.winfo_screenheight() - 520) // 3)}")
    if on_ready:  # test hook: lets a script drive the pages and take screenshots
        w.after(300, lambda: on_ready(w))
    w.mainloop()
    return 0 if w.error is None else 1


def run_uninstall_gui(prefix: Path, purge: bool, assume_yes: bool) -> int:
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        if not assume_yes:
            if not messagebox.askyesno(f"ถอนการติดตั้ง {APP_NAME}", f"ถอนการติดตั้ง {APP_NAME} ออกจากเครื่องนี้?"):
                return 0
            purge = purge or messagebox.askyesno(f"{APP_NAME}", "ลบข้อมูลความจำและการตั้งค่าของคุณด้วยไหม?\n(เลือก “ไม่” เพื่อเก็บไว้ใช้ภายหลัง)")
        uninstall(prefix, purge)
        messagebox.showinfo(APP_NAME, "ถอนการติดตั้งเสร็จแล้ว")
        return 0
    except ImportError:
        uninstall(prefix, purge)
        return 0


def main(argv: Optional[List[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=f"{APP_NAME} setup")
    ap.add_argument("--silent", action="store_true", help="no window, use the options below")
    ap.add_argument("--prefix", type=Path)
    ap.add_argument("--no-desktop", action="store_true")
    ap.add_argument("--no-startmenu", action="store_true")
    ap.add_argument("--no-claude", action="store_true", help="do not register with Claude Desktop")
    ap.add_argument("--claude-config", type=Path, action="append", help="claude_desktop_config.json to update (repeatable)")
    ap.add_argument("--skip-pip", action="store_true", help="do not download libraries (testing / offline)")
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--uninstall", action="store_true")
    ap.add_argument("--purge", action="store_true", help="with --uninstall: also delete memory + settings")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args(argv)

    prefix = a.prefix or default_prefix()
    if a.uninstall:
        if a.silent or a.yes:
            uninstall(prefix, a.purge)
            return 0
        return run_uninstall_gui(prefix, a.purge, False)

    opts = Options(prefix=prefix, desktop_shortcut=not a.no_desktop, start_menu=not a.no_startmenu, register_claude=not a.no_claude,
                   launch_after=a.launch, skip_pip=a.skip_pip)
    if a.claude_config:
        opts.claude_configs = list(a.claude_config)
    use_gui = not a.silent
    if use_gui:
        try:
            import tkinter  # noqa: F401
        except ImportError:
            use_gui = False
            print("ไม่พบ Tk — ใช้โหมดข้อความ")
    if use_gui:
        return run_wizard(opts)
    try:
        summary = install(opts, print)
    except Exception as e:
        print(f"ติดตั้งไม่สำเร็จ: {e}")
        return 1
    for w in summary["warnings"]:  # type: ignore[union-attr]
        print("คำเตือน:", w)
    if a.launch:
        launch_app(opts.prefix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
