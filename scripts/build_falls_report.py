import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / "data/demo/fall_video_full_results.json").read_text(encoding="utf-8"))
falls = [r for r in data["results"] if r["label"] == "fall"]

segs: list[tuple[float, float]] = []
start = end = None
for row in falls:
    t = row["time_sec"]
    if start is None:
        start = end = t
    elif t <= end + 1.5:
        end = t
    else:
        segs.append((start, end))
        start = end = t
if start is not None:
    segs.append((start, end))

rows = []
for i, (a, b) in enumerate(segs, 1):
    seg = [x for x in falls if a <= x["time_sec"] <= b]
    conf = max(x["confidence"] for x in seg)
    rows.append((i, a, b, len(seg), conf))

html = [
    "<!DOCTYPE html><html><head><meta charset='utf-8'>",
    "<title>Caidas detectadas</title>",
    "<style>body{font-family:sans-serif;background:#111;color:#eee;padding:20px}",
    "table{border-collapse:collapse;width:100%}th,td{border:1px solid #444;padding:8px}",
    "th{background:#333}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}",
    ".card{background:#222;padding:8px;border-radius:8px}img{width:100%;border-radius:4px}</style></head><body>",
    f"<h1>Caidas detectadas</h1><p>{len(falls)} frames | {len(segs)} segmentos</p>",
    "<table><tr><th>#</th><th>Inicio</th><th>Fin</th><th>Frames</th><th>Conf max</th></tr>",
]
for i, a, b, n, conf in rows:
    html.append(f"<tr><td>{i}</td><td>{a:.0f}s</td><td>{b:.0f}s</td><td>{n}</td><td>{conf*100:.1f}%</td></tr>")
html.append("</table><h2>Muestra de frames</h2><div class='grid'>")
frames = sorted((ROOT / "data/demo/frames").glob("fall_*.jpg"))
for p in frames[:: max(len(frames) // 12, 1)][:12]:
    html.append(f"<div class='card'><img src='frames/{p.name}'><p>{p.stem.replace('fall_', '')}</p></div>")
html.append("</div></body></html>")

out = ROOT / "data/demo/caidas_detectadas.html"
out.write_text("".join(html), encoding="utf-8")
print(out)
