"""Find the web browsers installed on this machine and open URLs with them (stdlib only).

Used twice:
  * the app's own window: open the UI in Edge / Chrome / Firefox / Brave ... , or in several at once
  * the wiki learner (wiki_ingest): which Chromium-family browser renders the wiki

Detection looks at the usual install folders (Windows: Program Files and %LOCALAPPDATA%; macOS: /Applications;
Linux: PATH). It only reports files that exist. Unknown browsers are simply not offered.
"""
import os
import shutil
import subprocess
import sys
from typing import Any, Callable, Dict, List, Optional

CHROMIUM = "chromium"   # can be driven by Playwright and supports --app=URL (a window without browser chrome)
GECKO = "gecko"

NAMES = {"edge": "Microsoft Edge", "chrome": "Google Chrome", "brave": "Brave", "vivaldi": "Vivaldi", "opera": "Opera",
         "chromium": "Chromium", "firefox": "Mozilla Firefox"}
FAMILY = {"edge": CHROMIUM, "chrome": CHROMIUM, "brave": CHROMIUM, "vivaldi": CHROMIUM, "opera": CHROMIUM,
          "chromium": CHROMIUM, "firefox": GECKO}
# best first: used by "auto"
PREFERENCE = ["edge", "chrome", "brave", "vivaldi", "chromium", "opera", "firefox"]

# Windows: path templates; {PF}, {PF86}, {LOCAL} are expanded from the environment
_WIN = {
    "edge": [r"{PF86}\Microsoft\Edge\Application\msedge.exe", r"{PF}\Microsoft\Edge\Application\msedge.exe"],
    "chrome": [r"{PF}\Google\Chrome\Application\chrome.exe", r"{PF86}\Google\Chrome\Application\chrome.exe",
               r"{LOCAL}\Google\Chrome\Application\chrome.exe"],
    "brave": [r"{PF}\BraveSoftware\Brave-Browser\Application\brave.exe", r"{PF86}\BraveSoftware\Brave-Browser\Application\brave.exe",
              r"{LOCAL}\BraveSoftware\Brave-Browser\Application\brave.exe"],
    "vivaldi": [r"{LOCAL}\Vivaldi\Application\vivaldi.exe", r"{PF}\Vivaldi\Application\vivaldi.exe"],
    "opera": [r"{LOCAL}\Programs\Opera\opera.exe", r"{PF}\Opera\opera.exe", r"{LOCAL}\Programs\Opera GX\opera.exe"],
    "chromium": [r"{LOCAL}\Chromium\Application\chrome.exe"],
    "firefox": [r"{PF}\Mozilla Firefox\firefox.exe", r"{PF86}\Mozilla Firefox\firefox.exe"],
}
_MAC = {
    "edge": ["/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"],
    "chrome": ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"],
    "brave": ["/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"],
    "vivaldi": ["/Applications/Vivaldi.app/Contents/MacOS/Vivaldi"],
    "opera": ["/Applications/Opera.app/Contents/MacOS/Opera"],
    "chromium": ["/Applications/Chromium.app/Contents/MacOS/Chromium"],
    "firefox": ["/Applications/Firefox.app/Contents/MacOS/firefox"],
}
_LINUX = {
    "edge": ["microsoft-edge", "microsoft-edge-stable"],
    "chrome": ["google-chrome", "google-chrome-stable"],
    "brave": ["brave-browser", "brave"],
    "vivaldi": ["vivaldi", "vivaldi-stable"],
    "opera": ["opera"],
    "chromium": ["chromium", "chromium-browser"],
    "firefox": ["firefox"],
}


def detect(env: Optional[Dict[str, str]] = None, platform: Optional[str] = None,
           exists: Callable[[str], bool] = os.path.isfile, which: Callable[[str], Optional[str]] = shutil.which) -> List[Dict[str, str]]:
    """[{id, name, family, path}] for every supported browser found, in PREFERENCE order."""
    env = env if env is not None else os.environ
    platform = platform or sys.platform
    found: Dict[str, str] = {}
    if platform == "win32":
        sub = {"PF": env.get("ProgramFiles", r"C:\Program Files"), "PF86": env.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
               "LOCAL": env.get("LOCALAPPDATA", "")}
        for bid, templates in _WIN.items():
            for t in templates:
                if "{LOCAL}" in t and not sub["LOCAL"]:
                    continue
                path = t.format(**sub)
                if exists(path):
                    found[bid] = path
                    break
    elif platform == "darwin":
        for bid, paths in _MAC.items():
            hit = next((p for p in paths if exists(p)), None)
            if hit:
                found[bid] = hit
    else:
        for bid, names in _LINUX.items():
            hit = next((w for w in (which(n) for n in names) if w), None)
            if hit:
                found[bid] = hit
    return [{"id": b, "name": NAMES[b], "family": FAMILY[b], "path": found[b]} for b in PREFERENCE if b in found]


def build_command(browser: Dict[str, str], url: str, app_mode: bool = False) -> List[str]:
    exe = browser["path"]
    if browser["family"] == CHROMIUM:
        if app_mode:  # a standalone window with no tabs/address bar: feels like a native app
            return [exe, f"--app={url}", "--window-size=1120,780"]
        return [exe, "--new-window", url]
    return [exe, "-new-window", url]


def open_url(browser_id: str, url: str, detected: Optional[List[Dict[str, str]]] = None, app_mode: bool = False,
             popen: Callable[..., Any] = subprocess.Popen) -> bool:
    """Open `url` with the given browser id ('default' = the system default). Returns False if it can't."""
    if browser_id in ("default", "", None):
        import webbrowser

        return bool(webbrowser.open(url))
    detected = detected if detected is not None else detect()
    b = next((x for x in detected if x["id"] == browser_id), None)
    if not b:
        return False
    kw: Dict[str, Any] = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL, "stdin": subprocess.DEVNULL}
    if sys.platform == "win32":
        kw["creationflags"] = 0x00000008 | 0x08000000  # DETACHED_PROCESS | CREATE_NO_WINDOW
    else:
        kw["start_new_session"] = True
    try:
        popen(build_command(b, url, app_mode), **kw)
        return True
    except OSError:
        return False


def choose_ui_launch(pref: str, detected: List[Dict[str, str]], has_pywebview: bool) -> Dict[str, Any]:
    """Decide how to show the app's own window.
    pref: 'auto' | 'pywebview' | 'default' | a browser id.  Returns {"mode": pywebview|app|tab|default, "browser": id|None}."""
    ids = {b["id"]: b for b in detected}
    pref = (pref or "auto").lower()
    if pref == "default":
        return {"mode": "default", "browser": None}
    if pref == "pywebview" and has_pywebview:
        return {"mode": "pywebview", "browser": None}
    if pref in ids:
        return {"mode": "app" if ids[pref]["family"] == CHROMIUM else "tab", "browser": pref}
    # auto, or the requested thing is not available here
    if has_pywebview:
        return {"mode": "pywebview", "browser": None}
    best = next((b for b in detected if b["family"] == CHROMIUM), None)
    if best:
        return {"mode": "app", "browser": best["id"]}
    return {"mode": "default", "browser": None}
