#!/usr/bin/env python3
"""
scripts/label_tool.py

PASO 2: Herramienta visual de etiquetado manual (servidor HTTP, sin Streamlit).

Muestra las imagenes del pool una por una en el navegador. Una imagen puede
tener MAS DE UNA PERSONA: por cada persona se dibuja un bounding box y se le
asigna su propio fall/no_fall (una imagen puede tener a la vez una persona
caida y otra parada). Los boxes confirmados quedan registrados en
data/raw/bbox_log.csv (filename, label, x1, y1, x2, y2, timestamp) — una fila
por persona, no por imagen; un mismo filename puede repetirse.

Cuando terminaste de marcar a todas las personas de una imagen, "Listo,
siguiente imagen" la mueve de data/raw/pool/ a data/raw/labeled/ (ya no tiene
sentido una carpeta fall/ y otra no_fall/, porque una misma imagen puede
aportar boxes de ambas clases).

Atajos de teclado:
    F     -> confirmar la persona dibujada como fall
    N     -> confirmar la persona dibujada como no_fall
    S     -> saltar esta imagen (no registra nada, sigue en el pool)
    D     -> borrar esta imagen del pool (descartarla)
    Enter -> Listo, siguiente imagen (requiere al menos un box confirmado)

Uso:
    python scripts/label_tool.py
    python scripts/label_tool.py --port 8765 --no-browser
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

PORT = 8765

ROOT        = Path(__file__).resolve().parents[1]
POOL_DIR    = ROOT / "data" / "raw" / "pool"
LABELED_DIR = ROOT / "data" / "raw" / "labeled"
BBOX_LOG    = ROOT / "data" / "raw" / "bbox_log.csv"

BBOX_FIELDNAMES = ["filename", "label", "x1", "y1", "x2", "y2", "timestamp"]

_cursor: list[int] = [0]


# =============================
# BBOX_LOG.CSV (una fila por persona)
# =============================

def read_bbox_rows() -> list[dict]:
    if not BBOX_LOG.exists():
        return []
    with open(BBOX_LOG, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_bbox_rows(rows: list[dict]) -> None:
    BBOX_LOG.parent.mkdir(parents=True, exist_ok=True)
    tmp = BBOX_LOG.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=BBOX_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(BBOX_LOG)


def boxes_for_filename(filename: str) -> list[dict]:
    return [r for r in read_bbox_rows() if r["filename"] == filename]


def append_box(filename: str, label: str, box: tuple[float, float, float, float]) -> None:
    rows = read_bbox_rows()
    x1, y1, x2, y2 = box
    rows.append({
        "filename": filename,
        "label": label,
        "x1": f"{x1:.6f}",
        "y1": f"{y1:.6f}",
        "x2": f"{x2:.6f}",
        "y2": f"{y2:.6f}",
        "timestamp": datetime.now().isoformat(),
    })
    write_bbox_rows(rows)


def delete_box(filename: str, box_index: int) -> bool:
    """Borra el box en la posicion `box_index` ENTRE los boxes de `filename`."""
    rows = read_bbox_rows()
    matching = [i for i, r in enumerate(rows) if r["filename"] == filename]
    if box_index < 0 or box_index >= len(matching):
        return False
    del rows[matching[box_index]]
    write_bbox_rows(rows)
    return True


def delete_boxes_for(filename: str) -> None:
    """Purga todos los boxes de `filename` (se usa al borrar la imagen entera,
    para no dejar filas huerfanas en bbox_log.csv apuntando a un archivo que
    ya no existe)."""
    rows = read_bbox_rows()
    remaining = [r for r in rows if r["filename"] != filename]
    if len(remaining) != len(rows):
        write_bbox_rows(remaining)


# =============================
# POOL / STATS
# =============================

def image_files() -> list[Path]:
    if not POOL_DIR.exists():
        return []
    return sorted(
        f for f in POOL_DIR.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    )


def stats() -> dict:
    inbox = len(image_files())
    labeled_images = len(list(LABELED_DIR.glob("*.*"))) if LABELED_DIR.exists() else 0
    rows = read_bbox_rows()
    fall = sum(1 for r in rows if r["label"] == "fall")
    no_fall = sum(1 for r in rows if r["label"] == "no_fall")
    return {
        "inbox": inbox,
        "fall": fall,
        "no_fall": no_fall,
        "labeled": labeled_images,
    }


def current_item() -> dict | None:
    files = image_files()
    if not files:
        return None
    idx = _cursor[0] % len(files)
    filename = files[idx].name
    boxes = boxes_for_filename(filename)
    return {
        "filename": filename,
        "position": idx + 1,
        "inbox_remaining": len(files),
        "boxes": boxes,
    }


def apply_action(filename: str, action: str) -> dict:
    """Acciones a nivel de IMAGEN completa: skip / delete / done.
    Confirmar una persona individual es aparte, ver append_box()."""
    files = image_files()
    action = action.strip().lower()

    if action == "skip":
        if files:
            _cursor[0] = (_cursor[0] + 1) % max(len(files), 1)
        return {"ok": True, "action": "skip", **stats()}

    src = POOL_DIR / Path(filename).name
    if not src.exists():
        return {"ok": False, "error": f"No existe: {filename}"}

    if action == "delete":
        src.unlink()
        delete_boxes_for(src.name)
    elif action == "done":
        if not boxes_for_filename(src.name):
            return {"ok": False, "error": "Marca al menos una persona antes de dar por terminada la imagen"}
        LABELED_DIR.mkdir(parents=True, exist_ok=True)
        src.rename(LABELED_DIR / src.name)
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
    .img-wrap {
      position: relative; display: inline-block;
      max-width: 100%; max-height: 75vh; cursor: crosshair;
      user-select: none;
    }
    .img-wrap img { display: block; -webkit-user-drag: none; }
    .bbox-rect {
      position: absolute; display: none; pointer-events: none;
      border: 2px solid #ffcf4d; background: rgba(255, 207, 77, 0.2);
    }
    .bbox-confirmed {
      position: absolute; pointer-events: none;
      border: 2px solid; border-radius: 2px;
    }
    .bbox-confirmed.fall { border-color: var(--fall); background: rgba(232, 93, 93, 0.15); }
    .bbox-confirmed.no_fall { border-color: var(--no-fall); background: rgba(76, 175, 130, 0.15); }
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
    button:disabled { opacity: 0.35; cursor: not-allowed; }
    .btn-fall    { background: var(--fall); }
    .btn-no-fall { background: var(--no-fall); }
    .btn-skip    { background: var(--skip); }
    .btn-delete  { background: #3a3f4d; color: #ccc; }
    .btn-done    { background: #7c5cff; }
    .progress { height: 6px; background: #2a2f3d; border-radius: 3px; overflow: hidden; margin-top: 0.5rem; }
    .progress-bar { height: 100%; background: linear-gradient(90deg, var(--fall), var(--no-fall)); transition: width 0.3s; }
    .hint { font-size: 0.82rem; color: var(--muted); line-height: 1.7; }
    kbd { background: #2a2f3d; border: 1px solid #3d4455; border-radius: 4px; padding: 0.1rem 0.35rem; font-size: 0.75rem; }
    .toast { position: fixed; bottom: 1.5rem; right: 1.5rem; background: #2d3344;
             padding: 0.75rem 1rem; border-radius: 8px; opacity: 0;
             transition: opacity 0.2s; pointer-events: none; }
    .toast.show { opacity: 1; }
    .box-list { list-style: none; margin: 0; padding: 0; display: grid; gap: 0.4rem; }
    .box-list li {
      display: flex; align-items: center; justify-content: space-between;
      background: #12141c; border-radius: 8px; padding: 0.4rem 0.6rem; font-size: 0.85rem;
    }
    .box-list .tag { padding: 0.1rem 0.5rem; border-radius: 6px; font-weight: 600; font-size: 0.75rem; }
    .box-list .tag.fall { background: rgba(232, 93, 93, 0.25); color: var(--fall); }
    .box-list .tag.no_fall { background: rgba(76, 175, 130, 0.25); color: var(--no-fall); }
    .box-list button {
      background: #3a3f4d; color: #ccc; padding: 0.25rem 0.6rem;
      font-size: 0.75rem; border-radius: 6px;
    }
    .empty-boxes { color: var(--muted); font-size: 0.85rem; }
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
        <h2>Personas marcadas</h2>
        <ul class="box-list" id="boxList"></ul>
      </div>
      <div class="card">
        <h2>Persona dibujada</h2>
        <div class="actions">
          <button class="btn-fall"    data-action="fall">Caida (F)</button>
          <button class="btn-no-fall" data-action="no_fall">No caida (N)</button>
        </div>
      </div>
      <div class="card">
        <h2>Imagen completa</h2>
        <div class="actions">
          <button class="btn-done"    data-action="done">Listo, siguiente (Enter)</button>
          <button class="btn-skip"    data-action="skip">Saltar (S)</button>
          <button class="btn-delete"  data-action="delete">Borrar imagen (D)</button>
        </div>
      </div>
      <div class="card hint">
        <p>Dibuja un rectangulo alrededor de una persona, despues <kbd>F</kbd>/<kbd>N</kbd> para esa persona. Repeti si hay mas de una persona en la imagen.</p>
        <p><kbd>Enter</kbd> pasa a la siguiente imagen (requiere al menos una persona marcada). <kbd>S</kbd> salta sin marcar nada. <kbd>D</kbd> descarta la imagen entera.</p>
      </div>
    </aside>
  </main>
  <div class="toast" id="toast"></div>
  <script>
    let current = null;
    let box = null;       // [x1, y1, x2, y2] normalizados 0-1, o null (persona recien dibujada, sin confirmar)
    let dragStart = null; // {x, y} en px relativos a la imagen, mientras se arrastra

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
         <span class="stat">Imagenes listas: <b>${s.labeled}</b></span>`;
      const total = s.labeled + s.inbox;
      const pct = total > 0 ? (s.labeled / total) * 100 : 0;
      document.getElementById("progress").style.width = pct + "%";
    }

    function updateActionButtons() {
      const hasBox = !!box;
      document.querySelectorAll('[data-action="fall"], [data-action="no_fall"]').forEach(btn => {
        btn.disabled = !hasBox;
      });
      const hasConfirmed = !!(current && current.boxes && current.boxes.length);
      const doneBtn = document.querySelector('[data-action="done"]');
      if (doneBtn) doneBtn.disabled = !hasConfirmed;
    }

    function renderConfirmedBoxes() {
      const wrap = document.getElementById("imgWrap");
      const list = document.getElementById("boxList");
      if (!wrap || !list) return;

      wrap.querySelectorAll(".bbox-confirmed").forEach(el => el.remove());
      list.innerHTML = "";

      const boxes = (current && current.boxes) || [];
      if (!boxes.length) {
        list.innerHTML = '<li class="empty-boxes">Todavia no marcaste a nadie en esta imagen.</li>';
        return;
      }

      const img = document.getElementById("theImg");
      boxes.forEach((b, i) => {
        if (img) {
          const el = document.createElement("div");
          el.className = `bbox-confirmed ${b.label}`;
          const x1 = parseFloat(b.x1), y1 = parseFloat(b.y1);
          const x2 = parseFloat(b.x2), y2 = parseFloat(b.y2);
          el.style.left = (x1 * 100) + "%";
          el.style.top = (y1 * 100) + "%";
          el.style.width = ((x2 - x1) * 100) + "%";
          el.style.height = ((y2 - y1) * 100) + "%";
          wrap.appendChild(el);
        }

        const li = document.createElement("li");
        const label = b.label === "fall" ? "Fall" : "No caida";
        li.innerHTML = `<span class="tag ${b.label}">${label}</span>`;
        const del = document.createElement("button");
        del.textContent = "Borrar";
        del.addEventListener("click", () => deleteBox(i));
        li.appendChild(del);
        list.appendChild(li);
      });
    }

    function clientToImagePoint(img, clientX, clientY) {
      const r = img.getBoundingClientRect();
      return {
        x: Math.min(Math.max(clientX - r.left, 0), r.width),
        y: Math.min(Math.max(clientY - r.top, 0), r.height),
        width: r.width,
        height: r.height,
      };
    }

    function paintRect(a, b) {
      const rect = document.getElementById("bboxRect");
      if (!rect) return;
      rect.style.display = "block";
      rect.style.left = Math.min(a.x, b.x) + "px";
      rect.style.top = Math.min(a.y, b.y) + "px";
      rect.style.width = Math.abs(b.x - a.x) + "px";
      rect.style.height = Math.abs(b.y - a.y) + "px";
    }

    // Listeners registrados UNA sola vez sobre `document`, resolviendo los
    // elementos vivos (#imgWrap/#theImg) en cada evento. Si se re-agregaran
    // en cada imagen nueva, los viejos (atados a elementos ya desmontados)
    // quedarian acumulados y pisarian dragStart/box con basura.
    function currentImgElements() {
      return {
        wrap: document.getElementById("imgWrap"),
        img: document.getElementById("theImg"),
      };
    }

    document.addEventListener("mousedown", (e) => {
      const { wrap, img } = currentImgElements();
      if (!wrap || !img || !wrap.contains(e.target)) return;
      dragStart = clientToImagePoint(img, e.clientX, e.clientY);
      box = null;
      updateActionButtons();
      paintRect(dragStart, dragStart);
      e.preventDefault();
    });

    document.addEventListener("mousemove", (e) => {
      if (!dragStart) return;
      const { img } = currentImgElements();
      if (!img) return;
      paintRect(dragStart, clientToImagePoint(img, e.clientX, e.clientY));
    });

    document.addEventListener("mouseup", (e) => {
      if (!dragStart) return;
      const { img } = currentImgElements();
      if (!img) { dragStart = null; return; }

      const cur = clientToImagePoint(img, e.clientX, e.clientY);
      const { width, height } = cur;
      const x1 = Math.min(dragStart.x, cur.x) / width;
      const y1 = Math.min(dragStart.y, cur.y) / height;
      const x2 = Math.max(dragStart.x, cur.x) / width;
      const y2 = Math.max(dragStart.y, cur.y) / height;
      dragStart = null;

      if (x2 - x1 > 0.02 && y2 - y1 > 0.02) {
        box = [x1, y1, x2, y2];
      } else {
        box = null;
        const rect = document.getElementById("bboxRect");
        if (rect) rect.style.display = "none";
      }
      updateActionButtons();
    });

    async function loadCurrent() {
      const res = await fetch("/api/current");
      const data = await res.json();
      current = data.item;
      box = null;
      dragStart = null;
      renderStats(data.stats);

      const viewer = document.getElementById("viewer");
      const meta   = document.getElementById("meta");

      if (!current) {
        viewer.innerHTML = '<div class="empty"><h2>Listo</h2><p>No quedan imagenes en el pool.</p><p>Corre: python scripts/convert_dataset.py</p></div>';
        meta.innerHTML = "";
        document.getElementById("boxList").innerHTML = "";
        updateActionButtons();
        return;
      }

      viewer.innerHTML = `
        <div class="img-wrap" id="imgWrap">
          <img id="theImg" src="/img/${encodeURIComponent(current.filename)}" alt="imagen">
          <div class="bbox-rect" id="bboxRect"></div>
        </div>`;
      meta.innerHTML =
        `<dt>Archivo</dt><dd>${current.filename}</dd>
         <dt>Posicion</dt><dd>${current.position} / ${current.inbox_remaining} en pool</dd>`;

      renderConfirmedBoxes();
      updateActionButtons();
    }

    async function confirmBox(label) {
      if (!current || !box) {
        toast("Dibuja el bounding box primero");
        return;
      }
      const res = await fetch("/api/box", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename: current.filename, label, box }),
      });
      const data = await res.json();
      if (!data.ok) { toast(data.error || "Error"); return; }

      current.boxes = data.boxes;
      box = null;
      dragStart = null;
      document.getElementById("bboxRect").style.display = "none";
      renderStats(data.stats);
      renderConfirmedBoxes();
      updateActionButtons();
      toast(label === "fall" ? "Persona: fall" : "Persona: no_fall");
    }

    async function deleteBox(index) {
      const res = await fetch("/api/box/delete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename: current.filename, box_index: index }),
      });
      const data = await res.json();
      if (!data.ok) { toast(data.error || "Error"); return; }

      current.boxes = data.boxes;
      renderStats(data.stats);
      renderConfirmedBoxes();
      updateActionButtons();
      toast("Box borrado");
    }

    async function doImageAction(action) {
      if (!current) return;
      const res = await fetch("/api/action", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename: current.filename, action }),
      });
      const data = await res.json();
      if (!data.ok) { toast(data.error || "Error"); return; }
      const labels = { done: "Imagen lista", skip: "Saltada", delete: "Borrada" };
      toast(labels[action] || action);
      await loadCurrent();
    }

    document.querySelectorAll('[data-action="fall"], [data-action="no_fall"]').forEach(btn =>
      btn.addEventListener("click", () => confirmBox(btn.dataset.action))
    );
    document.querySelectorAll('[data-action="done"], [data-action="skip"], [data-action="delete"]').forEach(btn =>
      btn.addEventListener("click", () => doImageAction(btn.dataset.action))
    );

    document.addEventListener("keydown", e => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
      if (e.key === "Enter") { e.preventDefault(); doImageAction("done"); return; }
      const map = { f: "fall", n: "no_fall", s: "skip", d: "delete" };
      const action = map[e.key.toLowerCase()];
      if (!action) return;
      e.preventDefault();
      if (action === "fall" || action === "no_fall") {
        confirmBox(action);
      } else {
        doImageAction(action);
      }
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

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8")) if raw else {}

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
        path = urlparse(self.path).path

        try:
            payload = self._read_json_body()

            if path == "/api/action":
                result = apply_action(payload.get("filename", ""), payload.get("action", ""))
                self._json(200, result)

            elif path == "/api/box":
                filename = payload.get("filename", "")
                label = payload.get("label", "")
                box = payload.get("box")

                if label not in ("fall", "no_fall"):
                    self._json(200, {"ok": False, "error": f"Label invalido: {label}"})
                    return
                if not box or len(box) != 4:
                    self._json(200, {"ok": False, "error": "Falta el bounding box"})
                    return
                x1, y1, x2, y2 = (float(v) for v in box)
                if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                    self._json(200, {"ok": False, "error": "Bounding box invalido"})
                    return
                if not (POOL_DIR / Path(filename).name).exists():
                    self._json(200, {"ok": False, "error": f"No existe: {filename}"})
                    return

                append_box(Path(filename).name, label, (x1, y1, x2, y2))
                self._json(200, {
                    "ok": True,
                    "boxes": boxes_for_filename(Path(filename).name),
                    **stats(),
                })

            elif path == "/api/box/delete":
                filename = Path(payload.get("filename", "")).name
                box_index = int(payload.get("box_index", -1))
                if not delete_box(filename, box_index):
                    self._json(200, {"ok": False, "error": "No se encontro ese box"})
                    return
                self._json(200, {
                    "ok": True,
                    "boxes": boxes_for_filename(filename),
                    **stats(),
                })

            else:
                self.send_error(404)

        except Exception as exc:
            self._json(400, {"ok": False, "error": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Etiquetador manual fall/no_fall (multi-persona por imagen)")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    s = stats()
    print(f"\nPool: {s['inbox']} imagenes | Fall: {s['fall']} | No caida: {s['no_fall']} | Imagenes listas: {s['labeled']}")

    if s["inbox"] == 0:
        print("No hay imagenes en el pool.")
        print("Corre primero: python scripts/scrape_dataset.py")
        return

    url = f"http://127.0.0.1:{args.port}/"
    server = ThreadingHTTPServer(("127.0.0.1", args.port), LabelHandler)
    print(f"\nEtiquetador: {url}")
    print("F=fall  N=no_fall (persona dibujada)  Enter=listo, siguiente  S=saltar  D=borrar imagen")
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
