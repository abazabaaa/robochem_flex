"""
Author : Elia Savino – github.com/EliaSavino
Desc   : Ultra-light WebSocket log streamer with 1 000-line history buffer.
"""

import asyncio
import contextlib
from contextlib import asynccontextmanager
from typing import List, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
# --------------------------------------------------------------------------- #
#                          ----------  HTML  ----------                       #
# --------------------------------------------------------------------------- #
HTML_PAGE = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>RoBrains Live Logs</title>

  <!-- favicon -->
  <link rel="icon" type="image/jpeg" href="/static/robrains.jpg"/>

  <!-- ANSI-to-HTML converter -->
  <script src="https://unpkg.com/ansi_up@4.0.4/ansi_up.js"></script>

  <style>
    :root {
      --bg: #111;
      --panel: #1b1b1b;
      --text: #e5e5e5;
      --accent: #007aff;
    }
    *        { box-sizing:border-box; }
    body     { margin:0; height:100vh; display:flex; flex-direction:column;
               font-family:system-ui, sans-serif; background:var(--bg); color:var(--text);}
    header   { display:flex; align-items:center; gap:1rem; padding:.6rem 1rem;
               background:var(--panel); box-shadow:0 0 6px #000a; flex-wrap:wrap; }
    header img.logo { height:32px; border-radius:4px; }
    header h1 { font-size:1.1rem; margin:0; letter-spacing:.5px; margin-right:auto; }
    header label, header select { font-size:.9rem; }
    header input[type="checkbox"] { margin-right:.3rem; }
    #logs    { flex:1; overflow-y:auto; white-space:pre-wrap; padding:1rem;
               font-family:monospace; background:var(--bg);
               display:flex; flex-direction:column; gap:.25rem; }

    #logs div { animation:fadeIn .3s ease-out both; }
    @keyframes fadeIn { from {opacity:0; transform:translateY(2px);}
                        to   {opacity:1; transform:none;} }

  </style>
</head>
<body>

  <!-- top banner & controls -->
  <header>
    <img src="/static/robrains.jpg" alt="logo" class="logo"/>
    <h1>RoBrains Live Logs</h1>

    <!-- Auto-scroll toggle -->
    <label>
      <input type="checkbox" id="autoscroll" checked>
      Auto-scroll
    </label>

    <!-- Level filter -->
    <label>
      Level:
      <select id="levelFilter">
        <option value="all">All</option>
        <option value="OK">OK</option>
        <option value="ERROR">ERROR</option>
        <option value="WARNING">WARNING</option>
      </select>
    </label>

    <!-- Origin filter -->
    <label>
      Origin:
      <select id="originFilter">
        <option value="all">All</option>
        <!-- populated dynamically -->
      </select>
    </label>
  </header>

  <!-- scrolling log container -->
  <div id="logs"></div>

<script type="module">
    const logDiv = document.getElementById("logs");
    const ansi   = new AnsiUp();
    const proto  = location.protocol === "https:" ? "wss" : "ws";
    const ws     = new WebSocket(`${proto}://${location.host}/ws/logs`);

    // State
    let origins = new Set();

    const autoscrollCheckbox = document.getElementById("autoscroll");
    const levelFilter        = document.getElementById("levelFilter");
    const originFilter       = document.getElementById("originFilter");

    // Re-filter all existing entries
    function applyFilters() {
      const lvl    = levelFilter.value;
      const orig   = originFilter.value;
      for (const entry of logDiv.children) {
        const entryLvl  = entry.dataset.level || "";
        const entryOrig = entry.dataset.origin || "";
        const passLevel = (lvl === "all") || (entryLvl === lvl);
        const passOrig  = (orig === "all") || (entryOrig === orig);
        entry.style.display = (passLevel && passOrig) ? "" : "none";
      }
    }

    levelFilter.addEventListener("change", applyFilters);
    originFilter.addEventListener("change", applyFilters);

    ws.onmessage = e => {
      const raw = e.data;

      // 1) Remove any ANSI color codes (ESC [ … m)
      const clean = raw.replace(/\x1b\[[0-9;]*m/g, '');

      // 2) Extract level badge (e.g. [OK], [WARNING], [ERROR])
      const levelMatch = clean.match(/^\s*\[([A-Z]+)\]/);
      const level      = levelMatch ? levelMatch[1] : '';

      // 3) Extract all bracketed tokens, pick first non-level as origin
      const allBrackets = [...clean.matchAll(/\[([^\]]+)\]/g)].map(m => m[1]);
      const origin      = allBrackets.find(tok =>
        !['OK','ERROR','WARNING'].includes(tok)
      ) || '';

      // 4) Create the DOM element and tag it
      const el = document.createElement("div");
      el.dataset.level  = level;    // now “OK”, “WARNING”, “ERROR” or empty
      el.dataset.origin = origin;   // e.g. “SessionContainer” or “UFO_V1”

      // 5) Render the colored HTML _after_ tagging
      el.innerHTML = ansi.ansi_to_html(raw);
      logDiv.appendChild(el);

      // 6) If this is a new origin, add it to the origin <select>
      if (origin && !origins.has(origin)) {
        origins.add(origin);
        const opt = document.createElement("option");
        opt.value = origin;
        opt.textContent = origin;
        originFilter.appendChild(opt);
      }

      // 7) Re-apply filters & maybe scroll
      applyFilters();
      if (autoscrollCheckbox.checked) {
        logDiv.scrollTop = logDiv.scrollHeight;
      }
    };
    ws.onopen  = ()=> console.log("WebSocket connected");
    ws.onclose = ()=> console.warn("WebSocket closed");
  </script>
</body>
</html>
"""  # --------------------------------------------------------------------------- #
#                             ----  FastAPI app  ----                         #
# --------------------------------------------------------------------------- #

HISTORY: List[str] = []  # ring-buffer
MAX_HISTORY: int = 1000
LOG_QUEUE: asyncio.Queue = asyncio.Queue()
CLIENTS: Set[WebSocket] = set()


async def _broadcaster() -> None:
    """Background task: pop from LOG_QUEUE, push to every client."""
    try:
        while True:
            msg = await LOG_QUEUE.get()
            # Fan-out
            disconnected = []
            for ws in CLIENTS:
                try:
                    await ws.send_text(msg)
                except Exception:
                    disconnected.append(ws)
            for ws in disconnected:
                CLIENTS.discard(ws)
    except asyncio.CancelledError:
        pass


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Start broadcaster on startup, cancel on shutdown."""
    task = asyncio.create_task(_broadcaster())
    yield
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


app = FastAPI(lifespan=_lifespan)

app.mount("/static", StaticFiles(directory=base_dir / "static"), name="static")

# -------------------------------  Routes  ---------------------------------- #


@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_PAGE


@app.websocket("/ws/logs")
async def ws_logs(ws: WebSocket):
    await ws.accept()

    # 1) dump history
    for line in HISTORY:
        await ws.send_text(line)

    # 2) register for live updates
    CLIENTS.add(ws)
    try:
        while True:
            await ws.receive_text()  # keep connection alive
    except WebSocketDisconnect:
        CLIENTS.discard(ws)


@app.post("/log")
async def post_log(req: Request):
    """
    POST JSON { "message": "text" } to broadcast & store.
    """
    data = await req.json()
    msg = data.get("message")
    if not isinstance(msg, str):
        return {"status": "error", "detail": "no 'message' key"}, 400

    # enqueue for broadcast
    await LOG_QUEUE.put(msg)

    # ring-buffer
    HISTORY.append(msg)
    if len(HISTORY) > MAX_HISTORY:
        HISTORY.pop(0)

    return {"status": "queued"}


# --------------------------------------------------------------------------- #
#                               ----  main  ----                              #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    uvicorn.run(
        "websocket_logger:app", host="127.0.0.1", port=6999, log_level="warning"
    )
