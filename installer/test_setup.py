"""Tests for the installer. They unzip the REAL release zip and run the installer from the unzipped copy."""
import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    subprocess.run([sys.executable, str(ROOT / "installer" / "build_zip.py")], check=True, capture_output=True)
    dst = tmp_path_factory.mktemp("unzipped")
    with zipfile.ZipFile(ROOT / "release" / "OmniBrain-Setup.zip") as z:
        z.extractall(dst)
    return dst / "OmniBrain-Setup"


@pytest.fixture
def S(release):
    spec = importlib.util.spec_from_file_location("omni_setup", release / "setup.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------- zip contents ----------------
def test_zip_has_everything_and_nothing_extra(release):
    names = {p.relative_to(release).as_posix() for p in release.rglob("*") if p.is_file()}
    assert {"Install.bat", "Uninstall.bat", "setup.py", "README-ติดตั้ง.txt", "app/app.py", "app/agent.py",
            "app/omni_brain_mcp.py", "app/ui/index.html", "app/omnibrain.ico", "app/requirements.txt"} <= names
    assert not [n for n in names if "test_" in n or n.endswith((".db", ".pyc"))]


def test_bat_files_are_ascii_crlf_and_readme_has_bom(release):
    for bat in ("Install.bat", "Uninstall.bat"):
        raw = (release / bat).read_bytes()
        raw.decode("ascii")  # cmd.exe mangles UTF-8: keep the batch files pure ASCII
        assert raw.count(b"\r\n") == raw.count(b"\n")
    assert (release / "README-ติดตั้ง.txt").read_bytes().startswith(b"\xef\xbb\xbf")


def test_icon_is_a_valid_multisize_ico(release):
    from PIL import Image

    im = Image.open(release / "app" / "omnibrain.ico")
    assert im.format == "ICO" and {(16, 16), (256, 256)} <= set(im.ico.sizes())


# ---------------- Claude Desktop config ----------------
ENTRY = {"command": "C:/x/python.exe", "args": ["C:/x/omni_brain_mcp.py"], "env": {"OMNI_BRAIN_DB": "C:/d.db"}}


def test_register_creates_file_when_missing(S, tmp_path):
    cfg = tmp_path / "Claude" / "claude_desktop_config.json"
    assert S.register_claude(cfg, ENTRY) == "added"
    assert json.loads(cfg.read_text())["mcpServers"]["omni-brain"] == ENTRY


def test_register_preserves_everything_else_and_backs_up(S, tmp_path):
    cfg = tmp_path / "c.json"
    orig = {"theme": "dark", "mcpServers": {"filesystem": {"command": "npx", "args": ["-y", "fs"]}}, "x": [1, 2]}
    cfg.write_text("\ufeff" + json.dumps(orig))  # Windows editors like to add a BOM
    assert S.register_claude(cfg, ENTRY) == "added"
    new = json.loads(cfg.read_text(encoding="utf-8"))
    assert new["mcpServers"]["filesystem"] == orig["mcpServers"]["filesystem"] and new["theme"] == "dark" and new["x"] == [1, 2]
    assert json.loads(next(tmp_path.glob("c.json.omnibrain*.bak")).read_text(encoding="utf-8-sig")) == orig


def test_register_is_idempotent_and_updates(S, tmp_path):
    cfg = tmp_path / "c.json"
    S.register_claude(cfg, ENTRY)
    assert S.register_claude(cfg, {**ENTRY, "command": "D:/new.exe"}) == "updated"
    data = json.loads(cfg.read_text())
    assert list(data["mcpServers"]) == ["omni-brain"] and data["mcpServers"]["omni-brain"]["command"] == "D:/new.exe"


@pytest.mark.parametrize("bad", ["{not json", "[1,2,3]", '{"mcpServers": 5}'])
def test_register_refuses_to_touch_broken_config(S, tmp_path, bad):
    cfg = tmp_path / "c.json"
    cfg.write_text(bad)
    with pytest.raises(S.ConfigError):
        S.register_claude(cfg, ENTRY)
    assert cfg.read_text() == bad and not list(tmp_path.glob("*.bak"))


def test_unregister_removes_only_ours(S, tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"mcpServers": {"filesystem": {"command": "npx"}, "omni-brain": ENTRY}}))
    assert S.unregister_claude(cfg) is True
    assert json.loads(cfg.read_text())["mcpServers"] == {"filesystem": {"command": "npx"}}
    assert S.unregister_claude(cfg) is False and S.unregister_claude(tmp_path / "missing.json") is False


def test_find_claude_configs(S, tmp_path):
    (tmp_path / "Roaming" / "Claude").mkdir(parents=True)
    msix = tmp_path / "Local" / "Packages" / "Claude_abc123" / "LocalCache" / "Roaming" / "Claude"
    msix.mkdir(parents=True)
    found = S.find_claude_configs({"APPDATA": str(tmp_path / "Roaming"), "LOCALAPPDATA": str(tmp_path / "Local")})
    assert len(found) == 2 and len(set(found)) == 2
    assert any("Packages" in str(p) for p in found) and any(p.parent.parent.name == "Roaming" and "Packages" not in str(p) for p in found)
    assert S.find_claude_configs({"APPDATA": str(tmp_path / "nope")}) == []


def test_mcp_entry_points_at_venv_and_shared_database(S, tmp_path):
    e = S.mcp_entry(tmp_path / "OmniBrain", {"OMNI_HOME": str(tmp_path / "data")})
    assert e["command"].startswith(str(tmp_path / "OmniBrain" / "venv"))
    assert e["args"] == [str(tmp_path / "OmniBrain" / "app" / "omni_brain_mcp.py")]
    assert e["env"]["OMNI_BRAIN_DB"] == str(tmp_path / "data" / "agent_brain.db")


def test_app_and_installer_agree_on_the_data_folder(S, release, tmp_path, monkeypatch):
    monkeypatch.setenv("OMNI_HOME", str(tmp_path / "d"))
    monkeypatch.setenv("OMNI_NO_KEYRING", "1")
    sys.path.insert(0, str(release / "app"))
    try:
        spec = importlib.util.spec_from_file_location("app_under_test", release / "app" / "app.py")
        sys.modules.setdefault("mouseinfo", type(sys)("mouseinfo"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        assert Path(m.home_dir()) == S.data_dir()
    finally:
        sys.path.remove(str(release / "app"))


# ---------------- install / uninstall ----------------
def opts_for(S, tmp_path, **kw):
    cfg = tmp_path / "Claude" / "claude_desktop_config.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}}))
    return S.Options(prefix=tmp_path / "Programs" / "OmniBrain", claude_configs=[cfg], skip_pip=True, launch_after=False, **kw), cfg


def test_full_install_then_uninstall_keeps_user_data(S, tmp_path, monkeypatch):
    monkeypatch.setenv("OMNI_HOME", str(tmp_path / "data"))
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "agent_brain.db").write_text("precious memories")
    opts, cfg = opts_for(S, tmp_path)
    logs = []
    res = S.install(opts, logs.append)
    p = opts.prefix
    assert (p / "app" / "app.py").exists() and (p / "app" / "ui" / "index.html").exists()
    assert S.venv_python(p).exists() and (p / "setup.py").exists() and "--uninstall" in (p / "Uninstall.bat").read_text()
    servers = json.loads(cfg.read_text())["mcpServers"]
    assert set(servers) == {"other", "omni-brain"} and servers["omni-brain"]["command"] == str(S.venv_python(p))
    assert res["claude_configs"] == [str(cfg)] and json.loads((p / "install.json").read_text())["version"] == S.VERSION
    # re-install over an existing copy works and doesn't duplicate anything
    S.install(opts, logs.append)
    assert list(json.loads(cfg.read_text())["mcpServers"]) == ["other", "omni-brain"]
    # uninstall
    S.uninstall(p, purge=False, log=logs.append)
    assert not p.exists()
    assert list(json.loads(cfg.read_text())["mcpServers"]) == ["other"]
    assert (tmp_path / "data" / "agent_brain.db").read_text() == "precious memories"


def test_uninstall_purge_deletes_data(S, tmp_path, monkeypatch):
    monkeypatch.setenv("OMNI_HOME", str(tmp_path / "data"))
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "settings.json").write_text("{}")
    opts, _ = opts_for(S, tmp_path)
    S.install(opts, lambda m: None)
    S.uninstall(opts.prefix, purge=True, log=lambda m: None)
    assert not (tmp_path / "data").exists()


def test_unregistering_respects_other_people_editing_the_config(S, tmp_path):
    opts, cfg = opts_for(S, tmp_path)
    S.install(opts, lambda m: None)
    cfg.write_text("{ broken by hand")  # user broke their own file between install and uninstall
    S.uninstall(opts.prefix, log=lambda m: None)  # must not crash or overwrite it
    assert cfg.read_text() == "{ broken by hand"


def test_broken_claude_config_is_a_warning_not_a_failed_install(S, tmp_path):
    opts, cfg = opts_for(S, tmp_path)
    cfg.write_text("{ nope")
    res = S.install(opts, lambda m: None)
    assert res["claude_configs"] == [] and "Claude Desktop" in res["warnings"][0] and cfg.read_text() == "{ nope"
    assert (opts.prefix / "app" / "app.py").exists()


def test_install_without_claude_leaves_config_alone(S, tmp_path):
    opts, cfg = opts_for(S, tmp_path, register_claude=False)
    before = cfg.read_text()
    S.install(opts, lambda m: None)
    assert cfg.read_text() == before


def test_incomplete_extraction_gives_clear_error(S, tmp_path, monkeypatch):
    monkeypatch.setattr(S, "HERE", tmp_path)  # no app/ folder next to setup.py
    opts, _ = opts_for(S, tmp_path)
    with pytest.raises(RuntimeError, match="แตกไฟล์"):
        S.install(opts, lambda m: None)


def test_silent_cli(S, tmp_path, capsys):
    prefix = tmp_path / "P"
    assert S.main(["--silent", "--skip-pip", "--no-claude", "--prefix", str(prefix)]) == 0
    assert (prefix / "app" / "app.py").exists()
    assert S.main(["--uninstall", "--yes", "--prefix", str(prefix)]) == 0 and not prefix.exists()
