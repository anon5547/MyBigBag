"""Collect a game wiki into the local knowledge base by rendering it in a real browser.

Many wikis are single-page apps: the HTML is nearly empty and the data is built by JavaScript, so a plain
HTTP fetch sees nothing. We render the page (Microsoft Edge on Windows 10/11 via Playwright - nothing to
download) and read what a visitor would see.

Politeness (this is a personal-use tool, not a crawler):
  * robots.txt is honoured; the identifying User-Agent says what we are
  * one page load per URL; opening tabs / detail pop-ups happens inside the already loaded page, so it costs
    the site no extra requests
  * /api/ and other disallowed paths are never requested
Whatever is captured is stored as untrusted reference text (see wiki_kb).

CLI:  python wiki_ingest.py https://example.com/wiki/ --out knowledge/example.json
"""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Optional
from urllib import robotparser

from wiki_kb import clean_text

UA = "OmniBrain/1.0 (personal game-assistant; reads public wiki pages like a visitor)"
Progress = Callable[[str], None]


class IngestError(Exception):
    pass


# ---------------------------------------------------------------- per-site profile (Lumivara-style wiki)
# Detected by structure (".nav-item[data-tab]" + "#panel-<tab>"), not by hostname, so a sibling wiki built the same
# way works too. Anything else goes through the generic heading-splitter below.
CARD_RULES: Dict[str, Dict[str, Any]] = {
    "monsters": {"item": ".monster-card", "modal": True},
    "equipment": {"item": ".equipment-card"},
    "items": {"item": ".item-card"},
    "cards": {"item": ".monster-card-item"},
    "jobs": {"item": ".job-card"},
    "maps": {"item": ".map-card"},
    "skills": {"item": ".skill-card", "levels": True},
}
SITE_INIT_SCRIPTS = {"lumivaraonline.com": "localStorage.setItem('pixelrpg.language','th')"}  # Thai = source language

JS_CARDS = """els => els.map(e => {
  const t = e.querySelector('h1,h2,h3,h4,.eq-title,.item-name,.skill-name,[class*="name"],[class*="title"]');
  return {title: (t ? t.innerText : e.innerText.split('\\n')[0]).trim(), text: e.innerText};
})"""

JS_SKILL_LEVELS = r"""card => {
  const sel = card.querySelector('select.skill-level-select');
  if (!sel) return [];
  const out = [], keep = sel.value;
  for (const o of Array.from(sel.options)) {
    sel.value = o.value;
    sel.dispatchEvent(new Event('change', {bubbles: true}));
    const bar = card.querySelector('.skill-stats-bar'), desc = card.querySelector('.skill-desc');
    out.push([o.textContent.trim(), ((bar ? bar.innerText : '') + ' | ' + (desc ? desc.innerText : '')).replace(/\s+/g, ' ')]);
  }
  sel.value = keep;
  sel.dispatchEvent(new Event('change', {bubbles: true}));
  return out;
}"""


def launch_browser(p: Any, choice: str = "auto", detected: Optional[List[Dict[str, str]]] = None) -> Any:
    """Start a browser for rendering.
    choice: 'auto' (Edge -> Chrome -> any other installed Chromium-family browser -> Playwright's own Chromium),
    a browser id from browsers.detect() ('edge', 'chrome', 'brave', 'vivaldi', 'opera', 'chromium'),
    or 'firefox' (needs Playwright's own Firefox build: `playwright install firefox`).
    OMNI_BROWSER_PATH overrides everything (testing / unusual installs)."""
    import browsers

    args = os.environ.get("OMNI_BROWSER_ARGS", "").split()
    path = os.environ.get("OMNI_BROWSER_PATH")
    if path:
        return p.chromium.launch(executable_path=path, args=args)
    choice = (choice or "auto").lower()
    found = detected if detected is not None else browsers.detect()
    chromium_family = [b for b in found if b["family"] == browsers.CHROMIUM]
    tried: List[str] = []
    if choice == "firefox":
        try:
            return p.firefox.launch()
        except Exception as e:
            raise IngestError("Firefox ต้องใช้ตัว Firefox ของ Playwright ติดตั้งด้วย `playwright install firefox` "
                              f"(Playwright ควบคุม Firefox ที่ติดตั้งในเครื่องโดยตรงไม่ได้) — {str(e)[:120]}") from e
    if choice != "auto":
        pick = next((b for b in chromium_family if b["id"] == choice), None)
        if not pick:
            raise IngestError(f"ไม่พบเบราว์เซอร์ “{choice}” ในเครื่องนี้ (หรือไม่ใช่ตระกูล Chromium) — เลือก “อัตโนมัติ” หรือเบราว์เซอร์อื่น")
        return p.chromium.launch(executable_path=pick["path"], args=args)
    for channel in ("msedge", "chrome"):  # Playwright knows where these live
        try:
            return p.chromium.launch(channel=channel, args=args)
        except Exception as e:
            tried.append(channel)
    for b in chromium_family:  # Brave, Vivaldi, Opera, Chromium ...
        if b["id"] in ("edge", "chrome"):
            continue
        try:
            return p.chromium.launch(executable_path=b["path"], args=args)
        except Exception:
            tried.append(b["id"])
    try:
        return p.chromium.launch(args=args)
    except Exception as e:
        raise IngestError("ไม่พบเบราว์เซอร์ที่ใช้เรนเดอร์ได้ (ลองแล้ว: " + ", ".join(tried + ["chromium ของ Playwright"]) + "). "
                          "Windows 10/11 มี Edge มาให้ปกติ หรือติดตั้ง Chrome/Brave หรือรัน `playwright install chromium` "
                          f"— {str(e)[:120]}") from e


def robots_allows(url: str) -> bool:
    parts = urllib.parse.urlsplit(url)
    rp = robotparser.RobotFileParser()
    try:
        req = urllib.request.Request(f"{parts.scheme}://{parts.netloc}/robots.txt", headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=10) as r:
            rp.parse(r.read().decode("utf-8", "ignore").splitlines())
    except urllib.error.HTTPError as e:
        return e.code != 401 and e.code != 403  # 404 = no rules; 401/403 = site says keep out
    except Exception:
        return True  # unreachable robots.txt: the page fetch itself will fail loudly if the site is down
    return rp.can_fetch(UA, url)


def _slug(url: str) -> str:
    p = urllib.parse.urlsplit(url)
    return re.sub(r"[^a-z0-9]+", "-", (p.netloc + p.path).lower()).strip("-")


def _split_by_headings(tab: str, tab_title: str, text: str, headings: List[str]) -> List[Dict[str, str]]:
    hs = {h.strip() for h in headings if h.strip()}
    docs, title, buf = [], tab_title, []
    for line in text.split("\n"):
        if line.strip() in hs:
            if buf:
                docs.append({"tab": tab, "title": title, "text": "\n".join(buf)})
            title, buf = f"{tab_title} › {line.strip()}", []
        else:
            buf.append(line)
    if buf:
        docs.append({"tab": tab, "title": title, "text": "\n".join(buf)})
    return [d for d in docs if len(d["text"].strip()) > 20]


def _capture_wiki_app(page: Any, progress: Progress) -> List[Dict[str, str]]:
    tabs = page.eval_on_selector_all(".nav-item[data-tab]", "els => [...new Set(els.map(e => e.dataset.tab))]")
    titles = page.evaluate("() => Object.fromEntries([...document.querySelectorAll('.nav-item[data-tab]')]"
                           ".map(e => [e.dataset.tab, e.innerText.replace(/\\s+/g,' ').trim().replace(/\\s*\\d+$/, '')]))")
    docs: List[Dict[str, str]] = []
    for i, tab in enumerate(tabs, 1):
        progress(f"อ่านหมวด {tab} ({i}/{len(tabs)})")
        page.evaluate("t => { location.hash = '#' + t }", tab)
        try:
            page.wait_for_selector(f"#panel-{tab}.active", timeout=8000)
        except Exception:
            progress(f"  ข้ามหมวด {tab} (หน้าไม่แสดง)")
            continue
        page.wait_for_timeout(250)
        rule = CARD_RULES.get(tab)
        panel = f"#panel-{tab}"
        if rule:
            cards = page.eval_on_selector_all(f"{panel} {rule['item']}", JS_CARDS)
            if rule.get("modal"):  # monster detail (full drop table) lives in a pop-up
                handles = page.query_selector_all(f"{panel} {rule['item']}")
                for card, h in zip(cards, handles):
                    try:
                        h.click(timeout=3000)
                        page.wait_for_selector("dialog[open]", timeout=3000)
                        txt = page.inner_text("dialog[open]")
                        first = txt.split("\n", 1)[0]
                        if ":" in first:
                            card["title"] = first.split(":", 1)[1].strip() or card["title"]
                        card["text"] = txt
                        page.keyboard.press("Escape")
                        page.wait_for_selector("dialog[open]", state="detached", timeout=3000)
                    except Exception:
                        pass  # keep the card's own text
            if rule.get("levels"):
                handles = page.query_selector_all(f"{panel} {rule['item']}")
                for card, h in zip(cards, handles):
                    try:
                        lv = h.evaluate(JS_SKILL_LEVELS)
                        if lv:
                            base = [ln for ln in card["text"].split("\n") if not re.fullmatch(r"Lv\.\d+", ln.strip()) and "ดูผลที่เลเวลสกิล" not in ln]
                            card["text"] = "\n".join(base) + "\nค่าตามเลเวลสกิล:\n" + "\n".join(f"{a}: {b}" for a, b in lv)
                    except Exception:
                        pass
            docs += [{"tab": tab, "title": c["title"] or tab, "text": c["text"]} for c in cards]
            progress(f"  {tab}: {len(cards)} รายการ")
        else:
            text = page.inner_text(panel)
            heads = page.eval_on_selector_all(f"{panel} h2, {panel} h3, {panel} h4", "els => els.map(e => e.innerText)")
            part = _split_by_headings(tab, titles.get(tab, tab), text, heads)
            docs += part
            progress(f"  {tab}: {len(part)} หัวข้อ")
    return docs


def _capture_generic(page: Any, url: str, progress: Progress) -> List[Dict[str, str]]:
    title = page.title() or url
    heads = page.eval_on_selector_all("h1, h2, h3", "els => els.map(e => e.innerText)")
    text = page.inner_text("body")
    docs = _split_by_headings("page", title, text, heads)
    progress(f"  {title}: {len(docs)} หัวข้อ")
    return docs


def capture(urls: List[str], progress: Progress = print, name: Optional[str] = None, nav_timeout_ms: int = 45000,
            init_script: Optional[str] = None, browser: str = "auto") -> Dict[str, Any]:
    """Render each URL once and return {"source", "url", "docs": [{tab, title, text}], "captured_at"}."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        raise IngestError("ต้องติดตั้ง playwright ก่อน:  pip install playwright  (ใช้ Microsoft Edge ที่มีในเครื่อง ไม่ต้องโหลดเบราว์เซอร์เพิ่ม)") from e
    urls = [u.strip() for u in urls if u.strip()]
    if not urls:
        raise IngestError("ไม่มี URL")
    for u in urls:
        if urllib.parse.urlsplit(u).scheme not in ("http", "https"):
            raise IngestError(f"รองรับเฉพาะ http/https: {u}")
        if not robots_allows(u):
            raise IngestError(f"robots.txt ของเว็บนี้ไม่อนุญาตให้เข้า {u} — จึงไม่เก็บข้อมูล")
    docs: List[Dict[str, str]] = []
    with sync_playwright() as p:
        browser_ = launch_browser(p, browser)
        try:
            host = urllib.parse.urlsplit(urls[0]).hostname or ""
            ctx = browser_.new_context(user_agent=UA, viewport={"width": 1400, "height": 900})
            init = init_script or next((s for h, s in SITE_INIT_SCRIPTS.items() if host.endswith(h)), None)
            if init:
                ctx.add_init_script(init)
            for n, url in enumerate(urls):
                if n:
                    time.sleep(1.0)  # one request per second at most
                progress(f"เปิด {url}")
                page = ctx.new_page()
                # text only: skip images/fonts/media (also keeps load on the site minimal)
                page.route("**/*", lambda r: r.abort() if r.request.resource_type in ("image", "font", "media") else r.continue_())
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=nav_timeout_ms)
                    page.wait_for_timeout(1500)
                except Exception as e:
                    raise IngestError(f"เปิด {url} ไม่ได้: {str(e)[:160]}") from e
                if page.query_selector(".nav-item[data-tab]") and page.query_selector(".wiki-panel"):
                    docs += _capture_wiki_app(page, progress)
                else:
                    docs += _capture_generic(page, url, progress)
                page.close()
        finally:
            browser_.close()
    docs = [{"tab": d["tab"], "title": d["title"], "text": clean_text(d["text"])} for d in docs]
    docs = [d for d in docs if d["text"]]
    if not docs:
        raise IngestError("เปิดหน้าได้แต่ไม่พบข้อความ (เว็บอาจต้องล็อกอิน หรือโครงสร้างไม่รองรับ)")
    return {"source": name or _slug(urls[0]), "url": urls[0], "captured_at": int(time.time()), "docs": docs}


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Render a wiki and save a knowledge snapshot")
    ap.add_argument("urls", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name")
    ap.add_argument("--browser", default="auto", help="auto | edge | chrome | brave | vivaldi | opera | chromium | firefox")
    ap.add_argument("--init-script", help="JS run before the page loads (e.g. to pick a language); mainly for local mirrors")
    a = ap.parse_args(argv)
    snap = capture(a.urls, print, a.name, init_script=a.init_script, browser=a.browser)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False)
    print(f"saved {len(snap['docs'])} entries -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
