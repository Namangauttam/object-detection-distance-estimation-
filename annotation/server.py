"""Local web server for the annotation review tool (standard library only).

Usage:
    python server.py            # then open http://localhost:8765
    python server.py --port 9000
"""
import argparse
import json
import os
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"
IMAGES = DATASET / "images"
LABELS = DATASET / "labels"
CLASSES_FILE = DATASET / "classes.txt"
REVIEW_FILE = DATASET / "review.json"
TOOL = ROOT / "tool" / "index.html"

NAME_RE = re.compile(r"^[\w.-]+$")


def read_classes():
    if not CLASSES_FILE.exists():
        return []
    return [l.strip() for l in CLASSES_FILE.read_text().splitlines() if l.strip()]


def write_classes(names):
    CLASSES_FILE.write_text("\n".join(names) + "\n")
    lines = [f"path: {DATASET.as_posix()}",
             "train: images  # TODO: split into train/val before training",
             "val: images", "names:"] + [f"  {i}: {n}" for i, n in enumerate(names)]
    (DATASET / "data.yaml").write_text("\n".join(lines) + "\n")


def read_review():
    if REVIEW_FILE.exists():
        return json.loads(REVIEW_FILE.read_text())
    return {}


def atomic_write(path, text):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def read_labels(stem):
    f = LABELS / f"{stem}.txt"
    boxes = []
    if f.exists():
        for line in f.read_text().splitlines():
            parts = line.split()
            if len(parts) == 5:
                boxes.append([int(parts[0])] + [float(v) for v in parts[1:]])
    return boxes


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def image_name(self, q):
        name = q.get("img", [""])[0]
        if not NAME_RE.match(name) or not (IMAGES / name).exists():
            return None
        return name

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if url.path in ("/", "/index.html"):
            body = TOOL.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif url.path == "/api/state":
            review = read_review()
            images = []
            for p in sorted(IMAGES.glob("*.jpg")):
                boxes = read_labels(p.stem)
                images.append({"name": p.name, "n": len(boxes),
                               "cls": sorted({b[0] for b in boxes}),
                               "reviewed": bool(review.get(p.name))})
            self.send_json({"classes": read_classes(), "images": images})
        elif url.path == "/api/labels":
            name = self.image_name(q)
            if not name:
                return self.send_json({"error": "bad image"}, 400)
            self.send_json({"boxes": read_labels(Path(name).stem),
                            "reviewed": bool(read_review().get(name))})
        elif url.path.startswith("/images/"):
            name = url.path[len("/images/"):]
            f = IMAGES / name
            if not NAME_RE.match(name) or not f.exists():
                self.send_error(404)
                return
            body = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=3600")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if url.path == "/api/labels":
            name = self.image_name(q)
            if not name:
                return self.send_json({"error": "bad image"}, 400)
            n_cls = len(read_classes())
            lines = []
            for c, cx, cy, w, h in data.get("boxes", []):
                c = int(c)
                if not 0 <= c < n_cls or w <= 0 or h <= 0:
                    continue
                lines.append(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            LABELS.mkdir(parents=True, exist_ok=True)
            atomic_write(LABELS / f"{Path(name).stem}.txt",
                         "\n".join(lines) + ("\n" if lines else ""))
            if "reviewed" in data:
                review = read_review()
                review[name] = bool(data["reviewed"])
                atomic_write(REVIEW_FILE, json.dumps(review, indent=1))
            self.send_json({"ok": True, "n": len(lines)})
        elif url.path == "/api/classes":
            # Append-only: removing or reordering would shift class ids in every label file.
            name = str(data.get("name", "")).strip().replace(" ", "_")
            names = read_classes()
            if not re.match(r"^[\w-]{1,40}$", name) or name in names:
                return self.send_json({"error": "invalid or duplicate class name"}, 400)
            names.append(name)
            write_classes(names)
            self.send_json({"classes": names})
        else:
            self.send_error(404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Annotation tool running at http://localhost:{args.port}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
