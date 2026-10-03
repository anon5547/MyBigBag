"""Tests for browser detection / launching and its wiring (ingester, app API, launch plan)."""
import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.modules.setdefault("mouseinfo", types.ModuleType("mouseinfo"))
import browsers  # noqa: E402
import wiki_ingest  # noqa: E402
from test_app import http_req, llm, sandbox, server  # noqa: E402,F401  (fixtures)

REAL_DETECT = browsers.detect
WIN_ENV = {"ProgramFiles": r"C:\Program Files", "ProgramFiles(x86)": r"C:\Program Files (x86)", "LOCALAPPDATA": r"C:\Users\u\AppData\Local"}
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
BRAVE = r"C:\Users\u\AppData\Local\BraveSoftware\Brave-Browser\Application\brave.exe"
FIREFOX = r"C:\Program Files\Mozilla Firefox\firefox.exe"


def det(*present, platform="win32"):
    return REAL_DETECT(WIN_ENV, platform, exists=lambda p: p in present)


def test_windows_detection_finds_installed_browsers_in_preference_order():
    found = det(FIREFOX, BRAVE, CHROME, EDGE)
    assert [b["id"] for b in found] == ["edge", "chrome", "brave", "firefox"]
    assert found[0]["path"] == EDGE and found[0]["family"] == "chromium" and found[3]["family"] == "gecko"
    assert det() == []                                         # nothing installed -> nothing offered


def test_per_user_install_paths_and_missing_localappdata():
    assert [b["id"] for b in det(BRAVE)] == ["brave"]
    no_local = {k: v for k, v in WIN_ENV.items() if k != "LOCALAPPDATA"}
    assert REAL_DETECT(no_local, "win32", exists=lambda p: p == BRAVE) == []   # must not probe "\BraveSoftware..."


def test_linux_and_mac_detection():
    lin = REAL_DETECT({}, "linux", which=lambda n: {"firefox": "/usr/bin/firefox", "google-chrome": "/usr/bin/google-chrome"}.get(n))
    assert [b["id"] for b in lin] == ["chrome", "firefox"]
    mac = REAL_DETECT({}, "darwin", exists=lambda p: p.endswith("Google Chrome"))
    assert [b["id"] for b in mac] == ["chrome"]


def test_command_lines():
    chrome, ff = det(CHROME, FIREFOX)
    assert browsers.build_command(chrome, "http://x/", app_mode=True) == [CHROME, "--app=http://x/", "--window-size=1120,780"]
    assert browsers.build_command(chrome, "http://x/") == [CHROME, "--new-window", "http://x/"]
    assert browsers.build_command(ff, "http://x/", app_mode=True) == [FIREFOX, "-new-window", "http://x/"]  # no app mode in Firefox


def test_open_url_spawns_detached_and_reports_failure():
    found = det(EDGE)
    calls = []
    assert browsers.open_url("edge", "http://x/", found, popen=lambda cmd, **k: calls.append((cmd, k))) is True
    assert calls[0][0][0] == EDGE and calls[0][1]["stdin"] is not None
    assert browsers.open_url("firefox", "http://x/", found, popen=lambda *a, **k: calls.append(1)) is False   # not installed
    assert len(calls) == 1

    def boom(*a, **k):
        raise OSError("blocked")
    assert browsers.open_url("edge", "http://x/", found, popen=boom) is False


@pytest.mark.parametrize("pref,has_webview,found,expect", [
    ("auto", True, ("edge",), ("pywebview", None)),            # native window wins when available
    ("auto", False, ("edge", "chrome"), ("app", "edge")),      # otherwise a chromeless Edge window, not a tab
    ("auto", False, ("firefox",), ("default", None)),          # Firefox has no app mode: use the system default
    ("auto", False, (), ("default", None)),
    ("chrome", True, ("edge", "chrome"), ("app", "chrome")),   # explicit choice beats pywebview
    ("firefox", True, ("edge", "firefox"), ("tab", "firefox")),
    ("brave", True, ("edge",), ("pywebview", None)),           # asked for a browser that isn't installed: fall back
    ("pywebview", False, ("edge",), ("app", "edge")),
    ("default", True, ("edge",), ("default", None)),
])
def test_ui_launch_plan(pref, has_webview, found, expect):
    paths = {"edge": EDGE, "chrome": CHROME, "firefox": FIREFOX, "brave": BRAVE}
    detected = det(*[paths[f] for f in found])
    plan = browsers.choose_ui_launch(pref, detected, has_webview)
    assert (plan["mode"], plan["browser"]) == expect


# ---------------- ingester: which browser renders the wiki
class FakePW:
    def __init__(self, fail=()):
        self.calls, self.fail = [], set(fail)
        outer = self

        class Chromium:
            @staticmethod
            def launch(**k):
                outer.calls.append(("chromium", k))
                key = k.get("channel") or k.get("executable_path") or "bundled"
                if key in outer.fail:
                    raise RuntimeError("not installed")
                return "BROWSER"

        class Firefox:
            @staticmethod
            def launch(**k):
                outer.calls.append(("firefox", k))
                if "firefox" in outer.fail:
                    raise RuntimeError("Executable doesn't exist")
                return "FF"

        self.chromium, self.firefox = Chromium, Firefox


@pytest.fixture(autouse=True)
def _no_env_override(monkeypatch):
    monkeypatch.delenv("OMNI_BROWSER_PATH", raising=False)


def test_auto_prefers_edge_then_chrome_then_other_chromium_then_bundled():
    pw = FakePW(fail={"msedge"})
    assert wiki_ingest.launch_browser(pw, "auto", det(BRAVE)) == "BROWSER"
    assert [c[1].get("channel") for c in pw.calls[:2]] == ["msedge", "chrome"]      # chrome channel works -> stop
    pw = FakePW(fail={"msedge", "chrome"})
    wiki_ingest.launch_browser(pw, "auto", det(BRAVE))
    assert pw.calls[-1][1]["executable_path"] == BRAVE                                # Brave picked up as the next option
    pw = FakePW(fail={"msedge", "chrome", BRAVE})
    wiki_ingest.launch_browser(pw, "auto", det(BRAVE))
    assert pw.calls[-1][1].get("executable_path") is None and pw.calls[-1][0] == "chromium"   # Playwright's own build


def test_explicit_choice_is_used_and_not_silently_replaced():
    pw = FakePW()
    wiki_ingest.launch_browser(pw, "brave", det(EDGE, BRAVE))
    assert len(pw.calls) == 1 and pw.calls[0][1]["executable_path"] == BRAVE
    with pytest.raises(wiki_ingest.IngestError, match="ไม่พบเบราว์เซอร์ “opera”"):
        wiki_ingest.launch_browser(FakePW(), "opera", det(EDGE))
    with pytest.raises(wiki_ingest.IngestError, match="ไม่พบเบราว์เซอร์ “firefox-system”"):
        wiki_ingest.launch_browser(FakePW(), "firefox-system", det(FIREFOX))    # Gecko isn't a Chromium-family pick


def test_firefox_uses_playwrights_own_build_with_a_clear_error():
    assert wiki_ingest.launch_browser(FakePW(), "firefox", []) == "FF"
    with pytest.raises(wiki_ingest.IngestError, match="playwright install firefox"):
        wiki_ingest.launch_browser(FakePW(fail={"firefox"}), "firefox", [])


def test_nothing_available_lists_what_was_tried():
    with pytest.raises(wiki_ingest.IngestError, match="msedge, chrome"):
        wiki_ingest.launch_browser(FakePW(fail={"msedge", "chrome", "bundled"}), "auto", [])


# ---------------- app wiring
def test_state_lists_detected_browsers_and_settings_roundtrip(server, monkeypatch):
    import app as appmod

    monkeypatch.setattr(browsers, "detect", lambda *a, **k: det(EDGE, FIREFOX))
    st = http_req("GET", "/api/state")[1]
    assert [b["id"] for b in st["browsers"]] == ["edge", "firefox"] and "path" not in st["browsers"][0]   # no filesystem paths to the UI
    assert st["settings"]["ui_browser"] == "auto" and st["settings"]["ingest_browser"] == "auto"
    r = http_req("POST", "/api/settings", {"ui_browser": "edge", "ingest_browser": "brave"})[1]["state"]["settings"]
    assert r["ui_browser"] == "edge" and r["ingest_browser"] == "brave"
    assert json.load(open(appmod.SETTINGS_PATH))["ingest_browser"] == "brave"


def test_open_browser_only_opens_the_apps_own_address(server, monkeypatch):
    import app as appmod

    opened = []
    monkeypatch.setattr(browsers, "open_url", lambda bid, url, *a, **k: opened.append((bid, url)) or True)
    assert http_req("POST", "/api/open-browser", {"id": "edge", "url": "http://evil.example/"})[1]["ok"] is True
    assert opened == [("edge", f"http://127.0.0.1:{appmod.APP.port}/")]               # a caller-supplied url is ignored
    assert http_req("POST", "/api/open-browser", {"id": "edge"}, token=False)[0] == 403
    monkeypatch.setattr(browsers, "open_url", lambda *a, **k: False)
    assert http_req("POST", "/api/open-browser", {"id": "nope"})[0] == 400


def test_learning_uses_the_chosen_ingest_browser(server, monkeypatch):
    import time
    import app as appmod

    seen = {}

    def fake_capture(urls, progress, name=None, **k):
        seen.update(k)
        return {"source": "s", "url": urls[0], "docs": [{"tab": "t", "title": "T", "text": "ข้อความทดสอบยาวพอสมควร"}]}

    monkeypatch.setattr(wiki_ingest, "capture", fake_capture)
    http_req("POST", "/api/settings", {"ingest_browser": "brave"})
    http_req("POST", "/api/kb/learn", {})
    for _ in range(100):
        if not http_req("GET", "/api/state")[1]["kb"]["running"]:
            break
        time.sleep(0.05)
    assert seen["browser"] == "brave"
