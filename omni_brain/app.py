#!/usr/bin/env python3
"""OmniBrain desktop app (Windows 10/11, also runs on macOS/Linux).

    python omni_brain/app.py            # native window if `pywebview` is installed, else your browser
    python omni_brain/app.py --browser  # force browser
Local-only web UI (127.0.0.1, random port). Every API call needs a per-launch token and a matching
Host/Origin, so a web page you happen to visit cannot drive your mouse through this server.
"""
import argparse
import glob
import json
import os
import secrets
import socket
import sys
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Deque, Dict, Optional
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))


def home_dir() -> str:
    base = os.environ.get("OMNI_HOME")
    if not base:
        root = os.environ.get("APPDATA") if sys.platform == "win32" else os.path.join(os.path.expanduser("~"), ".config")
        base = os.path.join(root or os.path.expanduser("~"), "OmniBrain")
    os.makedirs(base, exist_ok=True)
    return base


HOME = home_dir()
os.environ.setdefault("OMNI_BRAIN_DB", os.path.join(HOME, "agent_brain.db"))
sys.path.insert(0, HERE)
import omni_brain_mcp as ob  # noqa: E402  (after OMNI_BRAIN_DB is set)
ob.DB_PATH = os.environ["OMNI_BRAIN_DB"]  # also correct if the module was imported earlier
from agent import DEFAULTS, PRESETS, Agent, selftest  # noqa: E402

SETTINGS_PATH = os.path.join(HOME, "settings.json")
KEYRING_SERVICE = "OmniBrain"
USE_KEYRING = os.environ.get("OMNI_NO_KEYRING") != "1"


# ---------------------------------------------------------------- settings
class Settings:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.data: Dict[str, Any] = dict(DEFAULTS)
        self._key = os.environ.get("OMNI_API_KEY", "")
        try:
            with open(SETTINGS_PATH, encoding="utf-8") as f:
                saved = json.load(f)
            self._key = self._key or saved.pop("api_key", "")
            self.data.update({k: saved[k] for k in DEFAULTS if k in saved})
            self.preset = saved.get("preset", "deepseek")
        except (OSError, ValueError):
            self.preset = "deepseek"
        if not self._key and USE_KEYRING:
            try:
                import keyring

                self._key = keyring.get_password(KEYRING_SERVICE, "api_key") or ""
            except Exception:
                pass

    def cfg(self) -> Dict[str, Any]:
        return dict(self.data)

    def key(self) -> str:
        return self._key

    def update(self, new: Dict[str, Any]) -> None:
        with self.lock:
            for k, default in DEFAULTS.items():
                if k in new and new[k] is not None:
                    try:
                        self.data[k] = type(default)(new[k]) if not isinstance(default, str) else str(new[k]).strip()
                    except (TypeError, ValueError):
                        pass
            self.data["image_width"] = max(320, min(int(self.data["image_width"]), 2000))
            self.data["history_turns"] = max(1, min(int(self.data["history_turns"]), 30))
            self.data["max_steps"] = max(1, min(int(self.data["max_steps"]), 40))
            self.preset = str(new.get("preset", self.preset))
            if new.get("api_key"):
                self._key = str(new["api_key"]).strip()
            self._save()

    def _save(self) -> None:
        stored_in_keyring = False
        if self._key and USE_KEYRING:
            try:
                import keyring

                keyring.set_password(KEYRING_SERVICE, "api_key", self._key)
                stored_in_keyring = True
            except Exception:
                pass
        body = {**self.data, "preset": self.preset}
        if self._key and not stored_in_keyring:
            body["api_key"] = self._key  # plaintext fallback; file is chmod 600 where supported
        tmp = SETTINGS_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(body, f, ensure_ascii=False, indent=2)
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, SETTINGS_PATH)


# ---------------------------------------------------------------- app state
class App:
    LEARN_COMMANDS = ("/learn", "/เรียนรู้")

    def __init__(self) -> None:
        self.settings = Settings()
        self.events: Deque[Dict[str, Any]] = deque(maxlen=600)
        self.next_id = 1
        self.ev_lock = threading.Lock()
        self.busy = threading.Lock()
        self.agent = Agent(self.settings.cfg, self.settings.key, self.emit)
        self.token = secrets.token_urlsafe(24)
        self.port = 0
        self.kb_lock = threading.Lock()
        self.kb_state: Dict[str, Any] = {"running": False, "msg": "", "error": None}
        self.last_usage: Dict[str, Any] = {"prompt": 0, "completion": 0, "cached": 0, "calls": 0, "cost": 0.0}

    def emit(self, ev: Dict[str, Any]) -> None:
        if ev.get("type") == "usage":
            self.last_usage = {k: ev[k] for k in ("prompt", "completion", "cached", "calls", "cost")}
            return  # usage is polled through /api/state, no need to flood the feed
        with self.ev_lock:
            ev = {"id": self.next_id, "t": time.time(), **ev}
            self.next_id += 1
            self.events.append(ev)

    def events_after(self, after: int) -> list:
        with self.ev_lock:
            return [e for e in self.events if e["id"] > after]

    # ---- game knowledge base (wiki) ------------------------------------------
    def start_learn(self, url: str = "") -> bool:
        """Human-triggered (button or /เรียนรู้): render the wiki in a browser and replace the stored copy."""
        if not self.kb_lock.acquire(blocking=False):
            return False
        url = (url or self.settings.cfg().get("kb_url") or "").strip()
        self.kb_state = {"running": True, "msg": "เริ่มเรียนรู้…", "error": None}

        def progress(m: str) -> None:
            self.kb_state["msg"] = m

        def run() -> None:
            try:
                import wiki_ingest

                snap = wiki_ingest.capture([url], progress)
                n = ob.kb.replace_source(snap["source"], snap["url"], snap["docs"])
                self.kb_state = {"running": False, "error": None, "msg": f"เรียนรู้เสร็จ: {n['docs']} รายการ ({n['chunks']} ส่วน)"}
                self.emit({"type": "notice", "text": "📚 " + self.kb_state["msg"]})
            except Exception as e:
                self.kb_state = {"running": False, "msg": "", "error": f"{e}"[:300]}
                self.emit({"type": "error", "text": f"เรียนรู้ไม่สำเร็จ: {e}"[:400]})
            finally:
                self.kb_lock.release()

        threading.Thread(target=run, daemon=True).start()
        return True

    def import_bundled_snapshot(self) -> int:
        """First start (or empty base): load the knowledge shipped inside the installer, no browser needed."""
        if ob.kb.status():
            return 0
        n = 0
        for path in sorted(glob.glob(os.path.join(HERE, "knowledge", "*.json"))):
            try:
                n += ob.kb.import_snapshot(path)["docs"]
            except Exception:
                pass
        return n

    def start_chat(self, text: str) -> bool:
        cmd = text.strip().split(None, 1)
        if cmd and cmd[0].lower() in self.LEARN_COMMANDS:  # handled locally: zero model tokens
            self.emit({"type": "user", "text": text})
            if not self.start_learn(cmd[1] if len(cmd) > 1 and cmd[1].startswith("http") else ""):
                self.emit({"type": "notice", "text": "กำลังเรียนรู้อยู่แล้ว รอให้เสร็จก่อน"})
            else:
                self.emit({"type": "notice", "text": "📚 เริ่มเรียนรู้จาก Wiki (ดูความคืบหน้าในแท็บ “ความรู้”)"})
            return True
        if not self.busy.acquire(blocking=False):
            return False
        self.emit({"type": "user", "text": text})

        def run() -> None:
            try:
                self.agent.run_turn(text)
            finally:
                self.emit({"type": "done"})
                self.busy.release()

        threading.Thread(target=run, daemon=True).start()
        return True

    def reset(self) -> None:
        self.agent.reset()
        self.last_usage = {"prompt": 0, "completion": 0, "cached": 0, "calls": 0, "cost": 0.0}
        with self.ev_lock:
            self.events.clear()

    def stop(self) -> None:
        self.agent.cancel.set()
        with ob.S.mutex:
            ob.S.lock("ui stop button")
        self.emit({"type": "notice", "text": "⏹ หยุดฉุกเฉิน: ล็อกกลับโหมดจำลอง ยกเลิก ticket ทั้งหมด"})

    def skills(self) -> list:
        with ob.db() as c:
            return [dict(r) for r in c.execute(
                "SELECT skill_name, app_context, json_array_length(steps_json) AS steps, success_count, fail_count "
                "FROM procedural_skills ORDER BY updated_at DESC LIMIT 30")]

    def run_skill_by_human(self, name: str) -> Dict[str, Any]:
        """The human pressed the button, so that press IS the approval: no model, no tokens spent."""
        with ob.db() as c:
            row = c.execute("SELECT json_array_length(steps_json) AS n FROM procedural_skills WHERE skill_name=?", (name,)).fetchone()
        if not row:
            return {"ok": False, "text": "ไม่พบสกิล"}
        with ob.S.mutex:
            t = ob.Ticket(f"human-run:{name}", int(row["n"]))
            ob.S.tickets[t.id] = t
        res = json.loads(ob.run_skill_macro(name, t.id))
        self.emit({"type": "notice", "text": f"▶ สกิล '{name}' — {'สำเร็จ' if res.get('completed') else 'ไม่จบ'} ({res.get('mode')})"})
        return {"ok": True, "result": res}

    def state(self) -> Dict[str, Any]:
        st = json.loads(ob.safety_status())
        return {
            "settings": {**self.settings.cfg(), "preset": self.settings.preset, "has_key": bool(self.settings.key())},
            "presets": PRESETS, "status": st, "usage": self.last_usage, "busy": self.busy.locked(),
            "skills": self.skills(), "max_lease": ob.MAX_LEASE_MIN,
            "kb": {"sources": ob.kb.status(), **self.kb_state},
        }


APP: Optional[App] = None


# ---------------------------------------------------------------- http
class Handler(BaseHTTPRequestHandler):
    server_version = "OmniBrain"

    def log_message(self, fmt: str, *args: Any) -> None:  # keep the console quiet
        pass

    # -- security gate --
    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").lower()
        return host in (f"127.0.0.1:{APP.port}", f"localhost:{APP.port}")

    def _authed(self) -> bool:
        if not self._host_ok():
            return False
        origin = self.headers.get("Origin")
        if origin and origin not in (f"http://127.0.0.1:{APP.port}", f"http://localhost:{APP.port}"):
            return False
        return secrets.compare_digest(self.headers.get("X-Omni-Token", ""), APP.token)

    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith("text") or "json" in ctype else ""))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; connect-src 'self'")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: Any, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode())

    def do_GET(self) -> None:
        u = urlparse(self.path)
        if u.path == "/":
            if not self._host_ok():
                return self._send(403, b"forbidden", "text/plain")
            with open(os.path.join(HERE, "ui", "index.html"), encoding="utf-8") as f:
                page = f.read().replace("__OMNI_TOKEN__", APP.token)
            return self._send(200, page.encode(), "text/html")
        if not self._authed():
            return self._send(403, b'{"error":"forbidden"}')
        if u.path == "/api/state":
            return self._json(APP.state())
        if u.path == "/api/events":
            after = int((parse_qs(u.query).get("after") or ["0"])[0] or 0)
            return self._json({"events": APP.events_after(after)})
        self._send(404, b'{"error":"not found"}')

    def do_POST(self) -> None:
        if not self._authed():
            return self._send(403, b'{"error":"forbidden"}')
        try:
            n = int(self.headers.get("Content-Length") or 0)
            data = json.loads(self.rfile.read(min(n, 1_000_000)) or b"{}")
        except ValueError:
            return self._json({"error": "bad json"}, 400)
        p = urlparse(self.path).path
        try:
            if p == "/api/chat":
                text = str(data.get("text", "")).strip()[:4000]
                if not text:
                    return self._json({"error": "empty"}, 400)
                if not APP.start_chat(text):
                    return self._json({"ok": False, "busy": True}, 409)
                return self._json({"ok": True})
            if p == "/api/stop":
                APP.stop()
                return self._json({"ok": True})
            if p == "/api/unlock":  # the ONLY way to open a LIVE lease in the app; the model has no such tool
                minutes = max(1, min(int(data.get("minutes", ob.DEFAULT_LEASE_MIN)), ob.MAX_LEASE_MIN))
                with ob.S.mutex:
                    res = ob._open_lease(minutes, "app-ui")
                APP.emit({"type": "notice", "text": f"🔓 เปิดโหมดควบคุมจริง {minutes} นาที"})
                return self._json({"ok": True, "status": json.loads(res)})
            if p == "/api/lock":
                with ob.S.mutex:
                    ob.S.lock("ui lock")
                APP.emit({"type": "notice", "text": "🔒 กลับโหมดจำลอง"})
                return self._json({"ok": True})
            if p == "/api/settings":
                APP.settings.update(data)
                return self._json({"ok": True, "state": APP.state()})
            if p == "/api/reset":
                APP.reset()
                return self._json({"ok": True})
            if p == "/api/skill/run":
                return self._json(APP.run_skill_by_human(str(data.get("name", ""))))
            if p == "/api/kb/learn":
                ok = APP.start_learn(str(data.get("url", "")))
                return self._json({"ok": ok, "running": not ok})
            if p == "/api/kb/search":
                import wiki_kb

                q, tab = str(data.get("q", ""))[:200], str(data.get("tab", ""))[:40]
                res = ob.kb.search(q, tab, int(data.get("n", 4))) if q.strip() else []
                return self._json({"results": res, "chars": len(wiki_kb.format_results(res))})
            if p == "/api/selftest":
                return self._json(selftest(APP.settings.cfg(), APP.settings.key()))
        except Exception as e:
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)
        self._send(404, b'{"error":"not found"}')


def make_server(port: int = 0) -> ThreadingHTTPServer:
    global APP
    APP = App()
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    APP.port = srv.server_address[1]
    return srv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--browser", action="store_true", help="open in the default browser instead of a native window")
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--port", type=int, default=0)
    args = ap.parse_args()
    ob.init_db()
    srv = make_server(args.port)
    got = APP.import_bundled_snapshot()
    if got:
        print(f"นำเข้าความรู้ที่มากับตัวติดตั้ง: {got} รายการ")
    url = f"http://127.0.0.1:{APP.port}/"
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"OmniBrain running at {url}  (data: {HOME})")
    if args.no_open:
        threading.Event().wait()
    try:
        if args.browser:
            raise ImportError
        import webview  # pip install pywebview  (uses Edge WebView2 on Windows 10/11)

        webview.create_window("OmniBrain", url, width=1120, height=780, min_size=(760, 560), background_color="#0b0d10")
        webview.start()
    except ImportError:
        webbrowser.open(url)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
    finally:
        with ob.S.mutex:
            ob.S.lock("app exit")


if __name__ == "__main__":
    main()
