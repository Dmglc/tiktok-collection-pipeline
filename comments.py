"""Read the TikTok comment panel from a processed recording's frames.

Run after process.py. For each video group in index.json, it crops the right-hand
comment panel from every 10-second frame, keeps the views that differ, reads them
with `miyo parse` (local OCR), drops repeated lines, and writes comments.md.

Usage: python comments.py "<recording stem>_work" [--limit N]
"""
import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

# Comment panel as fractions of a 2560x1440 Firefox window with the player open.
PANEL = (0.787, 0.100, 0.998, 0.945)
CHANGE = 2.0  # mean pixel difference that counts as a new panel view
NOISE = re.compile(r"^(reply|view \d+ more.*|hide.*|back to top.*|add comment.*|comments? \(.*|creator videos|\d+|\d{1,2}-\d{1,2}|\d+[wdhm] ago)$", re.I)


def diff(a, b):
    import numpy as np
    x = np.asarray(a.convert("L").resize((60, 120)), dtype=np.float32)
    y = np.asarray(b.convert("L").resize((60, 120)), dtype=np.float32)
    return float(np.abs(x - y).mean())


def ocr(path):
    r = subprocess.run(["miyo", "parse", str(path)], capture_output=True, encoding="utf-8", errors="replace", timeout=180)
    return r.stdout if r.returncode == 0 else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--limit", type=int, default=0, help="only the first N videos, for testing")
    a = ap.parse_args()
    work = Path(a.work)
    groups = json.loads((work / "index.json").read_text(encoding="utf-8"))
    if isinstance(groups, dict):
        groups = groups.get("groups") or groups.get("videos")
    out = work / "comment_crops"
    out.mkdir(exist_ok=True)
    jobs = []
    for n, g in enumerate(groups, 1):
        if a.limit and n > a.limit:
            break
        prev = None
        for fr in g["frames"]:
            img = Image.open(work / "frames" / f"{fr['stem']}.jpg")
            w, h = img.size
            l, t, r, b = PANEL
            c = img.crop((int(l * w), int(t * h), int(r * w), int(b * h)))
            if prev is None or diff(prev, c) > CHANGE:
                p = out / f"v{n:03d}_{fr['stem']}.png"
                c.save(p)
                jobs.append((n, p))
                prev = c
    print("panel views", len(jobs), flush=True)
    with ThreadPoolExecutor(4) as ex:
        texts = list(ex.map(lambda j: ocr(j[1]), jobs))
    by = {}
    for (n, _), text in zip(jobs, texts):
        seen, lines = by.setdefault(n, (set(), []))
        for line in text.splitlines():
            s = line.strip(" |>*#-\t")
            key = re.sub(r"\W+", "", s).lower()
            if len(key) < 3 or NOISE.match(s) or key in seen:
                continue
            seen.add(key)
            lines.append(s)
    md = []
    for n, g in enumerate(groups, 1):
        if a.limit and n > a.limit:
            break
        lines = by.get(n, (None, []))[1]
        md.append(f"## Video {n:03d}\n" + ("\n".join(lines) if lines else "(no comments read)") + "\n")
    (work / "comments.md").write_text("\n".join(md), encoding="utf-8")
    print("ok", sum(len(v[1]) for v in by.values()), "lines", flush=True)


if __name__ == "__main__":
    sys.exit(main())
