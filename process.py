"""Prepare a TikTok screen recording for summarizing.

Steps:
  1. ffmpeg: mono 16 kHz mp3 audio + 480p 4 fps video, in 20-minute segments
  2. faster-whisper ("small", CPU int8) transcript with absolute timestamps
  3. one frame every 10 s, cropped to the address bar, creator header and player,
     grouped into videos by address-bar changes, one contact sheet per video

Usage:
  python process.py "<recording.mp4>" [--out DIR] [--segment-minutes 20]
                    [--frame-every 10] [--skip-media] [--skip-transcribe] [--skip-frames]

Output (default: <recording folder>/<recording stem>_work/):
  audio_NN.mp3, video_NN.mp4      step 1
  transcript.txt, transcript.json step 2
  frames/  crops/  sheets/        step 3
  index.json, index.md            per-video groups with transcript excerpts
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Crop boxes as fractions of the full frame (left, top, right, bottom).
# Measured on a 2560x1440 Firefox window with TikTok's collection player open.
CROPS = {
    "url": (0.13, 0.034, 0.40, 0.056),
    "header": (0.795, 0.088, 0.995, 0.192),
    "player": (0.248, 0.078, 0.5395, 1.0),
}
URL_CHANGE = 1.5     # mean pixel difference that counts as a new video (same URL ~0.2-0.6, new ~3.4+)
HEADER_CHANGE = 3.0  # mean pixel difference that counts as a new header view
PLAYERS_PER_SHEET = 4


def run(cmd):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def hms(seconds):
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def step_media(src, out, seg):
    run(["ffmpeg", "-hide_banner", "-v", "error", "-y", "-i", src, "-vn",
         "-ac", "1", "-ar", "16000", "-c:a", "libmp3lame", "-b:a", "48k",
         "-f", "segment", "-segment_time", str(seg), "-reset_timestamps", "1",
         str(out / "audio_%02d.mp3")])
    run(["ffmpeg", "-hide_banner", "-v", "error", "-y", "-i", src, "-an",
         "-vf", "scale=-2:480,fps=4", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
         "-f", "segment", "-segment_time", str(seg), "-reset_timestamps", "1",
         str(out / "video_%02d.mp4")])


def load_audio(path):
    # Decode with ffmpeg instead of faster-whisper's PyAV path, which breaks on newer PyAV.
    import numpy as np
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "s16le",
                          "-ac", "1", "-ar", "16000", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def step_transcribe(out, seg):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "faster-whisper"])
        from faster_whisper import WhisperModel

    model = WhisperModel("small", device="cpu", compute_type="int8")
    lines = []
    for i, audio in enumerate(sorted(out.glob("audio_*.mp3"))):
        offset = i * seg
        print(f"transcribing {audio.name} (offset {hms(offset)})", flush=True)
        segments, info = model.transcribe(load_audio(audio), vad_filter=True, beam_size=5)
        for s in segments:
            lines.append({"start": round(offset + s.start, 1), "end": round(offset + s.end, 1),
                          "text": s.text.strip(), "lang": info.language})
    (out / "transcript.json").write_text(json.dumps(lines, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "transcript.txt").write_text(
        "\n".join(f"[{hms(l['start'])}] {l['text']}" for l in lines), encoding="utf-8")
    print(f"{len(lines)} transcript lines", flush=True)


def crop(img, box):
    w, h = img.size
    l, t, r, b = box
    return img.crop((int(l * w), int(t * h), int(r * w), int(b * h)))


def diff(a, b):
    import numpy as np
    size = (160, 24) if a.width > a.height else (48, 48)
    x = np.asarray(a.convert("L").resize(size), dtype=np.float32)
    y = np.asarray(b.convert("L").resize(size), dtype=np.float32)
    return float(np.abs(x - y).mean())


def contact_sheet(url, headers, players, path):
    # Rows: address bar, up to 3 distinct header views, up to 4 player frames.
    from PIL import Image

    def fit(im, h):
        return im.resize((int(im.width * h / im.height), h))

    rows = [[fit(url, 44)], [fit(h, 150) for h in headers[:3]], [fit(p, 760) for p in players]]
    rows = [r for r in rows if r]
    width = max(sum(im.width for im in r) + 10 * (len(r) - 1) for r in rows)
    height = sum(max(im.height for im in r) for r in rows) + 10 * (len(rows) - 1)
    sheet = Image.new("RGB", (width, height), (24, 24, 24))
    y = 0
    for r in rows:
        x = 0
        for im in r:
            sheet.paste(im, (x, y))
            x += im.width + 10
        y += max(im.height for im in r) + 10
    sheet.save(path, quality=85)


def step_frames(src, out, every):
    from PIL import Image
    frames = out / "frames"
    frames.mkdir(exist_ok=True)
    run(["ffmpeg", "-hide_banner", "-v", "error", "-y", "-i", src,
         "-vf", f"fps=1/{every}", "-q:v", "3", str(frames / "f_%05d.jpg")])

    crops_dir = out / "crops"
    sheets = out / "sheets"
    crops_dir.mkdir(exist_ok=True)
    sheets.mkdir(exist_ok=True)

    groups = []
    prev_url = None
    for i, f in enumerate(sorted(frames.glob("f_*.jpg"))):
        t = i * every
        img = Image.open(f)
        parts = {k: crop(img, box) for k, box in CROPS.items()}
        for k, im in parts.items():
            im.save(crops_dir / f"{f.stem}_{k}.jpg", quality=90)
        if prev_url is None or diff(prev_url, parts["url"]) > URL_CHANGE:
            groups.append({"start": t, "frames": []})
        groups[-1]["frames"].append({"t": t, "stem": f.stem})
        prev_url = parts["url"]

    transcript = []
    tpath = out / "transcript.json"
    if tpath.exists():
        transcript = json.loads(tpath.read_text(encoding="utf-8"))

    md = []
    for n, g in enumerate(groups, 1):
        g["end"] = g["frames"][-1]["t"] + every
        stems = [fr["stem"] for fr in g["frames"]]
        url = Image.open(crops_dir / f"{stems[0]}_url.jpg")
        headers = []
        for s in stems:
            h = Image.open(crops_dir / f"{s}_header.jpg")
            if not headers or diff(headers[-1], h) > HEADER_CHANGE:
                headers.append(h)
        pick = [stems[round(k * (len(stems) - 1) / max(PLAYERS_PER_SHEET - 1, 1))]
                for k in range(min(PLAYERS_PER_SHEET, len(stems)))]
        players = [Image.open(crops_dir / f"{s}_player.jpg") for s in dict.fromkeys(pick)]
        sheet = sheets / f"video_{n:03d}.jpg"
        contact_sheet(url, headers, players, sheet)
        g["sheet"] = str(sheet)
        g["transcript"] = " ".join(l["text"] for l in transcript
                                   if g["start"] <= l["start"] < g["end"])
        md.append(f"## Video {n:03d}  {hms(g['start'])} - {hms(g['end'])}\n"
                  f"Sheet: {sheet.name}\n\n{g['transcript'] or '(no speech)'}\n")

    (out / "index.json").write_text(json.dumps(groups, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "index.md").write_text("\n".join(md), encoding="utf-8")
    print(f"{len(groups)} videos found", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("--out")
    p.add_argument("--segment-minutes", type=float, default=20)
    p.add_argument("--frame-every", type=int, default=10)
    p.add_argument("--skip-media", action="store_true")
    p.add_argument("--skip-transcribe", action="store_true")
    p.add_argument("--skip-frames", action="store_true")
    a = p.parse_args()

    src = Path(a.src)
    out = Path(a.out) if a.out else src.parent / f"{src.stem}_work"
    out.mkdir(parents=True, exist_ok=True)
    seg = int(a.segment_minutes * 60)

    if not a.skip_media:
        step_media(str(src), out, seg)
    if not a.skip_transcribe:
        step_transcribe(out, seg)
    if not a.skip_frames:
        step_frames(str(src), out, a.frame_every)
    print(f"done: {out}")


if __name__ == "__main__":
    main()
