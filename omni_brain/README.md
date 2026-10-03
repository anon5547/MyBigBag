# OmniBrain

PC agent with persistent memory and a guarded hand. Two ways to use it:

| | File | For |
|---|---|---|
| Desktop app (Thai UI, chat) | `app.py` / `start_omnibrain.bat` | Windows 10/11, your own API key |
| MCP server | `omni_brain_mcp.py` | Claude Desktop / Cursor / any MCP client |

## Run the app
```
pip install -r requirements.txt
python app.py            # native window if pywebview is installed, else your browser
```
Settings (gear icon): pick a provider, paste a model name and key, press **ทดสอบโมเดล**. The test
reads a generated image, so you learn in seconds whether the model can really see the screen.

## Browsers
* `browsers.py` finds Edge, Chrome, Brave, Vivaldi, Opera, Chromium and Firefox (Program Files / %LOCALAPPDATA% on
  Windows, /Applications on macOS, PATH on Linux).
* Window: `python app.py` uses the native window if `pywebview` is installed, otherwise a chromeless Edge/Chrome
  window (`--app=`), otherwise the system default. Pick one with Settings > "เปิดหน้าแอปด้วย" or
  `--browser edge|chrome|firefox|brave|default`. Open it in several at once with `--also chrome,firefox` or the
  buttons in Settings (all of them show the same session).
* Wiki learning: Settings > "เบราว์เซอร์ที่ใช้อ่าน Wiki" (or `wiki_ingest.py --browser brave`). Auto = Edge, Chrome,
  other Chromium browsers, then Playwright's own Chromium. Firefox works only through Playwright's own build
  (`playwright install firefox`); Playwright cannot drive an ordinary Firefox install.

## Game knowledge (wiki)
* `wiki_kb.py` stores a game wiki in the same SQLite file (FTS5 trigram index: Thai works without word
  segmentation). `wiki_ingest.py` renders a JS wiki in Edge/Chrome via Playwright and saves it - one page load,
  robots.txt honoured, `/api/` never touched.
* The model gets one tool, `wiki(query, tab)`, returning ~1-3k tokens of the best chunks. The whole wiki is far too
  big for a prompt (Lumivara: ~550k characters, ~300k tokens), so search is the only workable design.
* App: tab **ความรู้** (status, learn button, test search) or type `/เรียนรู้` in the chat (zero model tokens).
  Claude Desktop gets the same data through the MCP tool `wiki_search`.
* A snapshot in `knowledge/*.json` is imported on first start so it works before any live learning.
* Wiki text is untrusted reference data and is labelled so in every result.
* Lumivara's terms forbid bots/macros other than the in-game Auto Play: use this for Q&A and planning only.

## Several monitors
* The model sees **one** monitor per screenshot (default: the one under the mouse) to save tokens; it can ask for
  monitor 1..N or all stitched together. Coordinates on monitors left of / above the primary are negative - handled.
* Windows is switched to per-monitor DPI awareness at start-up so screenshot pixels equal cursor coordinates even
  when the monitors use different scaling.
* **Emergency stop**: pyautogui alone only watches the *primary* monitor's corners, which does nothing useful when
  the primary is the middle screen. OmniBrain arms the fail-safe on the top-left corner of **every** monitor.
* Not yet tried on a real 3-monitor Windows desk (only simulated layouts + a real single X screen).

## Safety model
* Starts in **dry-run**. Only the **ปลดล็อก** button in the app opens a LIVE lease (5/10/30 min, auto-expires).
  The model has no tool that unlocks, so it cannot do it itself.
* Every action needs a single-use ticket (`deliberate`), bounds/hotkey/rate checks, and yields if you touch the mouse.
* **หยุด** button or mouse to the top-left corner: locks immediately and revokes all tickets.
* The local server binds to 127.0.0.1, needs a per-launch token and a matching Host/Origin, so a web page
  you visit cannot call it.
* Not covered: other tools on your machine that can click or run shell commands for the model
  (auto-run terminals, other computer-use tools). Keep those off while LIVE.
* The API key goes to Windows Credential Manager if `keyring` is installed, otherwise to `settings.json`
  (plaintext, in `%APPDATA%\OmniBrain`).

## Keeping token cost low
Compact tool schemas · screenshot only on request and only the newest one kept · sliding history window ·
step cap per message · saved skills replay with **zero** model calls (press เล่น in the skills tab) ·
live token/cost meter at the bottom.

## Tests
`pytest omni_brain -q` (add `xvfb-run -a` on Linux to include the real-screen test).
