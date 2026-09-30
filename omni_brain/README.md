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
