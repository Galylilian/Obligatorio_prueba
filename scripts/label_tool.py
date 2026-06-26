#!/usr/bin/env python3
"""
scripts/label_tool.py

PASO 2: Herramienta visual de etiquetado manual (servidor HTTP, sin Streamlit).

Muestra las imagenes del pool una por una en el navegador y permite
moverlas a las carpetas de cada clase:

    data/raw/fall/
    data/raw/no_fall/

Atajos de teclado:
    F -> fall
    N -> no_fall
    S -> saltar
    D -> borrar

Uso:
    python scripts/label_tool.py
    python scripts/label_tool.py --port 8765 --no-browser
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

PORT = 8765

ROOT       = Path(__file__).resolve().parents[1]
POOL_DIR   = ROOT / "data" / "raw" / "pool"
FALL_DIR   = ROOT / "data" / "raw" / "fall"
NO_FALL_DIR = ROOT / "data" / "raw" / "no_fall"

_cursor: list[int] = [0]


def image_files() -> list[Path]:
    if not POOL_DIR.exists():
        return []
    return sorted(
        f for f in POOL_DIR.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    )


def stats() -> dict:
    inbox   = len(image_files())
    fall    = len(list(FALL_DIR.glob("*.*")))   if FALL_DIR.exists()    else 0
    no_fall = len(list(NO_FALL_DIR.glob("*.*"))) if NO_FALL_DIR.exists() else 0
    return {"inbox": inbox, "fall": fall, "no_fall": no_fall, "labeled": fall + no_fall}


def current_item() -> dict | None:
    files = image_files()
    if not files:
        return None
    idx = _cursor[0] % len(files)
    s = stats()
    return {
        "filename":        files[idx].name,
        "position":        idx + 1,
        "inbox_remaining": len(files),
        **s,
    }


def apply_action(filename: str, action: str) -> dict:
    files = image_files()
    action = action.strip().lower()

    if action == "skip":
        if files:
            _cursor[0] = (_cursor[0] + 1) % max(len(files), 1)
        return {"ok": True, "action": "skip", **stats()}

    src = POOL_DIR / Path(filename).name
    if not src.exists():
        return {"ok": False, "error": f"No existe: {filename}"}

    if action == "fall":
        FALL_DIR.mkdir(parents=True, exist_ok=True)
        src.rename(FALL_DIR / src.name)
    elif action == "no_fall":
        NO_FALL_DIR.mkdir(parents=True, exist_ok=True)
        src.rename(NO_FALL_DIR / src.name)
    elif action == "delete":
        src.unlink()
    else:
        return {"ok": False, "error": f"Accion desconocida: {action}"}

    files = image_files()
    if files and _cursor[0] >= len(files):
        _cursor[0] = 0

    return {"ok": True, "action": action, **stats()}


HTML = r"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Etiquetar dataset</title>
  <style>
    :root {
      --bg: #0f1117; --panel: #1a1d27; --border: #2a2f3d;
      --text: #e8eaef; --muted: #9aa3b2;
      --fall: #e85d5d; --no-fall: #4caf82; --skip: #5b8def;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: "Segoe UI", system-ui, sans-serif;
           background: var(--bg); color: var(--text); min-height: 100vh; }
    header {
      padding: 1rem 1.5rem; border-bottom: 1px solid var(--border);
      display: flex; justify-content: space-between; align-items: center;
      gap: 1rem; flex-wrap: wrap;
    }
    h1 { margin: 0; font-size: 1.2rem; font-weight: 600; }
    .stats { display: flex; gap: 1rem; flex-wrap: wrap; font-size: 0.9rem; }
    .stat { background: var(--panel); padding: 0.4rem 0.75rem; border-radius: 8px; }
    .stat b { color: #fff; }
    main {
      display: grid; grid-template-columns: 1fr 300px;
      gap: 1rem; padding: 1rem 1.5rem 2rem;
      max-width: 1300px; margin: 0 auto;
    }
    @media (max-width: 860px) { main { grid-template-columns: 1fr; } }
    .viewer {
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 12px; min-height: 60vh;
      display: flex; align-items: center; justify-content: center; overflow: hidden;
    }
    .viewer img { max-width: 100%; max-height: 75vh; object-fit: contain; }
    .empty { color: var(--muted); text-align: center; padding: 3rem; }
    .sidebar { display: flex; flex-direction: column; gap: 1rem; }
    .card {
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 12px; padding: 1rem;
    }
    .card h2 { margin: 0 0 0.75rem; font-size: 0.8rem;
               text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); }
    .meta { font-size: 0.9rem; line-height: 1.6; }
    .meta dt { color: var(--muted); }
    .meta dd { margin: 0 0 0.25rem; }
    .actions { display: grid; gap: 0.6rem; }
    button {
      border: none; border-radius: 10px; padding: 0.9rem 1rem;
      font-size: 1rem; font-weight: 600; cursor: pointer; color: #fff;
      transition: transform 0.1s, opacity 0.15s;
    }
    button:hover { opacity: 0.9; }
    button:active { transform: scale(0.97); }
    .btn-fall    { background: var(--fall); }
    .btn-no-fall { background: var(--no-fall); }
    .btn-skip    { background: var(--skip); }
    .btn-delete  { background: #3a3f4d; color: #ccc; }
    .progress { height: 6px; background: #2a2f3d; border-radius: 3px; overflow: hidden; margin-top: 0.5rem; }
    .progress-bar { height: 100%; background: linear-gradient(90deg, var(--fall), var(--no-fall)); transition: width 0.3s; }
    .hint { font-size: 0.82rem; color: var(--muted); line-height: 1.7; }
    kbd { background: #2a2f3d; border: 1px solid #3d4455; border-radius: 4px; padding: 0.1rem 0.35rem; font-size: 0.75rem; }
    .toast { position: fixed; bottom: 1.5rem; right: 1.5rem; background: #2d3344;
             padding: 0.75rem 1rem; border-radius: 8px; opacity: 0;
             transition: opacity 0.2s; pointer-events: none; }
    .toast.show { opacity: 1; }
  </style>
</head>
<body>
  <header>
    <h1>Etiquetador &mdash; Fall Detector Dataset</h1>
    <div class="stats" id="stats"></div>
  </header>
  <main>
    <section class="viewer" id="viewer">
      <div class="empty">Cargando...</div>
    </section>
    <aside class="sidebar">
      <div class="card">
        <h2>Imagen</h2>
        <dl class="meta" id="meta"></dl>
        <div class="progress"><div class="progress-bar" id="progress"></div></div>
      </div>
      <div class="card">
        <h2>Acciones</h2>
        <div class="actions">
          <button class="btn-fall"    data-action="fall">Caida (F)</button>
          <button class="btn-no-fall" data-action="no_fall">No caida (N)</button>
          <button class="btn-skip"    data-action="skip">Saltar (S)</button>
          <button class="btn-delete"  data-action="delete">Borrar (D)</button>
        </div>
      </div>
      <div class="card hint">
        <p><kbd>F</kbd> caida &nbsp; <kbd>N</kbd> no caida &nbsp; <kbd>S</kbd> saltar &nbsp; <kbd>D</kbd> borrar</p>
        <p>Las imagenes etiquetadas se mueven a <code>data/raw/fall/</code> o <code>data/raw/no_fall/</code>.</p>
      </div>
    </aside>
  </main>
  <div class="toast" id="toast"></div>
  <script>
    let current = null;

    function toast(msg) {
      const el = document.getElementById("toast");
      el.textContent = msg;
      el.classList.add("show");
      setTimeout(() => el.classList.remove("show"), 1000);
    }

    function renderStats(s) {
      document.getElementById("stats").innerHTML =
        `<span class="stat">Pool: <b>${s.inbox}</b></span>
         <span class="stat">Fall: <b>${s.fall}</b></span>
         <span class="stat">No caida: <b>${s.no_fall}</b></span>
         <span class="stat">Etiquetadas: <b>${s.labeled}</b></span>`;
      const total = s.labeled + s.inbox;
      const pct = total > 0 ? (s.labeled / total) * 100 : 0;
      document.getElementById("progress").style.width = pct + "%";
    }

    async function loadCurrent() {
      const res = await fetch("/api/current");
      const data = await res.json();
      current = data.item;
      renderStats(data.stats);

      const viewer = document.getElementById("viewer");
      const meta   = document.getElementById("meta");

      if (!current) {
        viewer.innerHTML = '<div class="empty"><h2>Listo</h2><p>No quedan imagenes en el pool.</p><p>Corre: python scripts/convert_dataset.py</p></div>';
        meta.innerHTML = "";
        return;
      }

      viewer.innerHTML = `<img src="/img/${encodeURIComponent(current.filename)}" alt="imagen">`;
      meta.innerHTML =
        `<dt>Archivo</dt><dd>${current.filename}</dd>
         <dt>Posicion</dt><dd>${current.position} / ${current.inbox_remaining} en pool</dd>`;
    }

    async function doAction(action) {
      if (!current && action !== "skip") return;
      const res = await fetch("/api/action", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename: current?.filename || "", action }),
      });
      const data = await res.json();
      if (!data.ok) { toast(data.error || "Error"); return; }
      const labels = { fall: "Fall", no_fall: "No caida", skip: "Saltada", delete: "Borrada" };
      toast(labels[action] || action);
      await loadCurrent();
    }

    document.querySelectorAll("[data-action]").forEach(btn =>
      btn.addEventListener("click", () => doAction(btn.dataset.action))
    );

    document.addEventListener("keydown", e => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
      const map = { f: "fall", n: "no_fall", s: "skip", d: "delete" };
      const action = map[e.key.toLowerCase()];
      if (action) { e.preventDefault(); doAction(action); }
    });

    loadCurrent();
  </script>
</body>
</html>"""


class LabelHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        pass

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path

        if path == "/":
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif path == "/api/current":
            self._json(200, {"ok": True, "item": current_item(), "stats": stats()})

        elif path.startswith("/img/"):
            name = Path(unquote(path[5:])).name
            p = POOL_DIR / name
            if p.is_file():
                data = p.read_bytes()
                mime, _ = mimetypes.guess_type(str(p))
                self.send_response(200)
                self.send_header("Content-Type", mime or "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_error(404)

        else:
            self.send_error(404)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/action":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
            result = apply_action(payload.get("filename", ""), payload.get("action", ""))
            self._json(200, result)
        except Exception as exc:
            self._json(400, {"ok": False, "error": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Etiquetador manual fall/no_fall")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    s = stats()
    print(f"\nPool: {s['inbox']} imagenes | Fall: {s['fall']} | No caida: {s['no_fall']}")

    if s["inbox"] == 0:
        print("No hay imagenes en el pool.")
        print("Corre primero: python scripts/scrape_dataset.py")
        return

    url = f"http://127.0.0.1:{args.port}/"
    server = ThreadingHTTPServer(("127.0.0.1", args.port), LabelHandler)
    print(f"\nEtiquetador: {url}")
    print("Atajos: F=fall  N=no_fall  S=saltar  D=borrar")
    print("Ctrl+C para salir\n")

    if not args.no_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nCerrado.")
        server.shutdown()


if __name__ == "__main__":
    main()
