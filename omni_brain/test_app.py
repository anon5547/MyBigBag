"""Tests for agent.py + app.py against a fake OpenAI-compatible server (no network, no real API key)."""
import http.client
import json
import os
import sys
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

os.environ["OMNI_HOME"] = os.path.join(os.environ.get("TMPDIR", "/tmp"), "omni_test_home")
os.environ["OMNI_NO_KEYRING"] = "1"
sys.path.insert(0, os.path.dirname(__file__))
sys.modules.setdefault("mouseinfo", types.ModuleType("mouseinfo"))
import agent as ag  # noqa: E402
import app as appmod  # noqa: E402
import omni_brain_mcp as ob  # noqa: E402

JPEG = b"\xff\xd8\xff\xe0fakejpegbytes"


def call(name, args, cid):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def reply(content=None, calls=None, usage=(100, 10)):
    msg = {"role": "assistant", "content": content}
    if calls:
        msg["tool_calls"] = calls
    return {"choices": [{"message": msg}], "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]}}


class FakeLLM:
    """Serves scripted replies and records every request body."""

    def __init__(self):
        self.script, self.requests, self.default = [], [], None
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append({"body": body, "auth": self.headers.get("Authorization")})
                r = outer.script.pop(0) if outer.script else (outer.default or reply("จบ"))
                data = json.dumps(r).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/v1"


@pytest.fixture
def llm():
    f = FakeLLM()
    yield f
    f.srv.shutdown()


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch, llm):
    monkeypatch.setattr(ob, "DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setattr(appmod, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(ob, "S", ob.Safety())
    monkeypatch.setattr(ob, "_gui_mod", None)
    ob.init_db()


def make_agent(llm, **over):
    cfg = {**ag.DEFAULTS, "base_url": llm.url, "model": "m", "price_in": 1.0, "price_out": 2.0, **over}
    events = []
    return ag.Agent(lambda: cfg, lambda: "sk-secret", events.append), events, cfg


def fake_observe(monkeypatch):
    monkeypatch.setattr(ob, "observe_screen", lambda *a, **k: [ob.Image(data=JPEG, format="jpeg"), json.dumps({"mode": "DRY_RUN"})])


# ------------------------------------------------------------------ agent
def test_full_loop_tools_images_and_usage(llm, monkeypatch):
    fake_observe(monkeypatch)
    llm.script = [
        reply(calls=[call("recall", {"keyword": "notepad"}, "c1")]),
        reply(calls=[call("look", {}, "c2")]),
        reply(calls=[call("look", {}, "c3")]),
        reply("เห็นแล้ว"),
    ]
    a, events, cfg = make_agent(llm)
    a.run_turn("ดูจอ")
    assert [e["text"] for e in events if e["type"] == "assistant"] == ["เห็นแล้ว"]
    first = llm.requests[0]["body"]
    assert first["messages"][0]["role"] == "system" and first["tool_choice"] == "auto"
    names = {t["function"]["name"] for t in first["tools"]}
    assert not names & {"unlock_live", "request_live_unlock", "lock_now", "set_safety_lock"}
    assert llm.requests[0]["auth"] == "Bearer sk-secret"
    # only the newest screenshot stays in context; the older one became a text note
    last = llm.requests[-1]["body"]["messages"]
    imgs = [m for m in last if isinstance(m["content"], list) and any(p["type"] == "image_url" for p in m["content"])]
    assert len(imgs) == 1
    assert any(m["content"] == ag.OLD_IMAGE_NOTE for m in last if isinstance(m["content"], str))
    assert all(not any(k.startswith("_") for k in m) for m in last)
    # tool replies stay adjacent to their assistant tool_calls
    for i, m in enumerate(last):
        if m.get("tool_calls"):
            assert last[i + 1]["role"] == "tool" and last[i + 1]["tool_call_id"] == m["tool_calls"][0]["id"]
    assert a.usage["calls"] == 4 and a.usage["prompt"] == 400 and abs(a.cost() - (400 * 1 + 40 * 2) / 1e6) < 1e-9
    assert any(e["type"] == "image" for e in events)


def test_model_cannot_call_unlock_even_if_it_tries(llm):
    llm.script = [reply(calls=[call("unlock_live", {"code": "x"}, "c1")]), reply("ok")]
    a, events, _ = make_agent(llm)
    a.run_turn("ปลดล็อกให้หน่อย")
    tool_msgs = [m for m in a.messages if m["role"] == "tool"]
    assert "unknown tool" in tool_msgs[0]["content"]
    assert json.loads(ob.safety_status())["mode"] == "DRY_RUN"


def test_step_cap_stops_runaway_loop(llm):
    llm.default = reply(calls=[call("recall", {"keyword": "x"}, "c")])
    a, events, _ = make_agent(llm, max_steps=3)
    a.run_turn("วนไปเรื่อยๆ")
    assert len(llm.requests) == 3
    assert "ครบ 3 ขั้น" in [e for e in events if e["type"] == "assistant"][-1]["text"]


def test_history_window_drops_whole_old_turns(llm):
    a, _, _ = make_agent(llm, history_turns=2)
    for i in range(5):
        a.run_turn(f"ข้อความ {i}")
    sent = [m["content"] for m in llm.requests[-1]["body"]["messages"] if m["role"] == "user"]
    assert sent == ["ข้อความ 3", "ข้อความ 4"]


def test_cancel_stops_before_next_call(llm):
    a, events, _ = make_agent(llm)
    # run_turn clears the flag at start (a new message is a new intent), so cancel mid-turn instead:
    orig = a._exec
    llm.default = reply(calls=[call("recall", {"keyword": "x"}, "c")])
    a._exec = lambda c: (a.cancel.set(), orig(c))[1]
    a.run_turn("go")
    assert len(llm.requests) == 1 and "หยุดตามคำสั่ง" in [e for e in events if e["type"] == "assistant"][-1]["text"]


def test_http_error_is_reported_without_leaking_key(llm, monkeypatch):
    def boom(*a, **k):
        import urllib.error, io
        raise urllib.error.HTTPError("u", 401, "x", {}, io.BytesIO(b'{"error":"bad key sk-secret"}'))
    monkeypatch.setattr(ag.urllib.request, "urlopen", boom)
    a, events, _ = make_agent(llm)
    a.run_turn("hi")
    err = [e for e in events if e["type"] == "error"][0]["text"]
    assert "401" in err and "sk-secret" not in err


def test_missing_model_gives_readable_error(llm):
    a, events, _ = make_agent(llm, model="")
    a.run_turn("hi")
    assert "ชื่อโมเดล" in [e for e in events if e["type"] == "error"][0]["text"]


def test_selftest_detects_vision(llm, monkeypatch):
    monkeypatch.setattr(ag.random, "choice", lambda s: "A")
    llm.script = [reply("OK"), reply("AAAAA")]
    cfg = {**ag.DEFAULTS, "base_url": llm.url, "model": "m"}
    r = ag.selftest(cfg, "k")
    assert r["connect"] and r["vision"]
    llm.script = [reply("OK"), reply("ขออภัย ฉันไม่เห็นภาพ")]
    assert ag.selftest(cfg, "k")["vision"] is False


def test_compact_tools_are_small():
    # the tool list is resent on every request; keep it well under the MCP docstring version
    assert len(json.dumps(ag.TOOLS, ensure_ascii=False)) < 5200


# ------------------------------------------------------------------ app / http
@pytest.fixture
def server(llm):
    srv = appmod.make_server(0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    appmod.APP.settings.update({"base_url": llm.url, "model": "m", "api_key": "sk-secret", "preset": "custom"})
    yield srv
    srv.shutdown()


def http_req(method, path, body=None, token=True, host=None, origin=None):
    port = appmod.APP.port
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    h = {"Host": host or f"127.0.0.1:{port}", "Content-Type": "application/json"}
    if token:
        h["X-Omni-Token"] = appmod.APP.token
    if origin:
        h["Origin"] = origin
    c.request(method, path, json.dumps(body) if body is not None else None, h)
    r = c.getresponse()
    raw = r.read()
    try:
        return r.status, json.loads(raw)
    except ValueError:
        return r.status, raw


def test_api_requires_token_host_and_origin(server):
    assert http_req("GET", "/api/state", token=False)[0] == 403
    assert http_req("POST", "/api/unlock", {"minutes": 5}, token=False)[0] == 403
    assert http_req("GET", "/api/state", host="evil.example:80")[0] == 403           # DNS rebinding
    assert http_req("POST", "/api/unlock", {}, origin="http://evil.example")[0] == 403  # cross-site page
    assert http_req("GET", "/api/state")[0] == 200
    assert json.loads(ob.safety_status())["mode"] == "DRY_RUN"


def test_index_embeds_token_only_for_correct_host(server):
    s, page = http_req("GET", "/", token=False)
    assert s == 200 and appmod.APP.token.encode() in page and b"__OMNI_TOKEN__" not in page
    assert http_req("GET", "/", token=False, host="evil.example")[0] == 403


def test_state_never_returns_the_api_key(server):
    s, st = http_req("GET", "/api/state")
    assert st["settings"]["has_key"] is True and "sk-secret" not in json.dumps(st)
    assert "api_key" not in st["settings"]


def test_settings_roundtrip_and_clamping(server):
    s, r = http_req("POST", "/api/settings", {"model": " abc ", "image_width": 99999, "max_steps": 0, "history_turns": "x"})
    st = r["state"]["settings"]
    assert st["model"] == "abc" and st["image_width"] == 2000 and st["max_steps"] == 1 and st["history_turns"] == 6
    saved = json.load(open(appmod.SETTINGS_PATH))
    assert saved["api_key"] == "sk-secret"  # plaintext fallback only because keyring is disabled in tests
    assert oct(os.stat(appmod.SETTINGS_PATH).st_mode & 0o777) == "0o600"


def test_unlock_stop_lock_flow(server):
    s, r = http_req("POST", "/api/unlock", {"minutes": 9999})
    assert r["status"]["status"] == "LIVE" and r["status"]["minutes"] == ob.MAX_LEASE_MIN
    assert http_req("GET", "/api/state")[1]["status"]["mode"] == "LIVE"
    http_req("POST", "/api/stop", {})
    assert http_req("GET", "/api/state")[1]["status"]["mode"] == "DRY_RUN"
    http_req("POST", "/api/unlock", {"minutes": 1})
    http_req("POST", "/api/lock", {})
    assert http_req("GET", "/api/state")[1]["status"]["mode"] == "DRY_RUN"


def wait_done(after=0, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        evs = http_req("GET", f"/api/events?after={after}")[1]["events"]
        if any(e["type"] == "done" for e in evs):
            return evs
        time.sleep(0.05)
    raise AssertionError("chat did not finish")


def test_chat_end_to_end_and_busy_guard(server, llm):
    llm.script = [reply("สวัสดีครับ", usage=(50, 5))]
    assert http_req("POST", "/api/chat", {"text": ""})[0] == 400
    assert http_req("POST", "/api/chat", {"text": "ทักหน่อย"})[0] == 200
    evs = wait_done()
    assert [e["type"] for e in evs][:2] == ["user", "assistant"] and evs[1]["text"] == "สวัสดีครับ"
    u = http_req("GET", "/api/state")[1]["usage"]
    assert u["prompt"] == 50 and u["completion"] == 5 and u["calls"] == 1
    http_req("POST", "/api/reset", {})
    assert http_req("GET", "/api/events?after=0")[1]["events"] == []
    assert http_req("GET", "/api/state")[1]["usage"]["calls"] == 0


def test_human_runs_macro_without_any_llm_call(server, llm):
    steps = json.dumps([{"action": "press", "key": "a"}])
    ob.save_skill_macro("demo", "notepad", steps)
    assert [s["skill_name"] for s in http_req("GET", "/api/state")[1]["skills"]] == ["demo"]
    r = http_req("POST", "/api/skill/run", {"name": "demo"})[1]
    assert r["ok"] and r["result"]["mode"] == "DRY_RUN" and llm.requests == []
    assert http_req("POST", "/api/skill/run", {"name": "nope"})[1]["ok"] is False


def test_selftest_endpoint(server, llm, monkeypatch):
    monkeypatch.setattr(ag.random, "choice", lambda s: "Z")
    llm.script = [reply("OK"), reply("ZZZZZ")]
    r = http_req("POST", "/api/selftest", {})[1]
    assert r["connect"] and r["vision"]
