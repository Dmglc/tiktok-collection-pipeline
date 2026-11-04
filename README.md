# TikTok Collection Pipeline

Turns a screen recording of a saved TikTok collection into per-video material you can read and act on: a transcript, contact sheets of what was on screen, and the comments.

## The problem

I saved TikToks I meant to use, like job tips, study methods, money advice, and tools. They piled up in collections I never went back to, and watching them all again wasn't realistic.

## How it works

1. Screen-record yourself scrolling through a collection in the browser.
2. `process.py` makes smaller audio and video copies with ffmpeg and transcribes the audio locally with faster-whisper.
3. It takes a frame every 10 seconds and crops the address bar, creator header, and player. When the address bar changes, a new video has started, so the frames are grouped by video into one contact sheet each.
4. `comments.py` crops the comment panel from the same frames, keeps the views that changed, reads them with local OCR, and removes repeated lines.

Everything runs on your own PC. Nothing is uploaded.

## Usage

```bash
python process.py "recording.mp4"
python comments.py "recording_work"
```

Useful `process.py` flags: `--out DIR`, `--segment-minutes 20`, `--frame-every 10`, `--skip-media`, `--skip-transcribe`, `--skip-frames`. Use `comments.py --limit N` to test on the first N videos.

### Output

Written to `<recording name>_work/` beside the recording:

| File | What it is |
|---|---|
| `transcript.txt`, `transcript.json` | Transcript with timestamps from the start of the recording |
| `sheets/` | One contact sheet per video |
| `index.md`, `index.json` | Each video's time range, sheet, and transcript lines |
| `comments.md` | Comment text for each video |

### Requirements

- Python, with `Pillow` and `numpy`
- [ffmpeg](https://ffmpeg.org/)
- `faster-whisper` (`process.py` installs it if it's missing)
- The `miyo` CLI, which `comments.py` uses for OCR. To use another OCR tool, replace the `ocr()` function.

## Known limits

- **Fixed layout:** the crop boxes were measured on a 2560x1440 Firefox window with TikTok's collection player open. Other layouts need new values for `CROPS` in `process.py` and `PANEL` in `comments.py`.
- **Video boundaries:** they're accurate to about 10 seconds, so a transcript line can land in the neighboring video.
- **Missed content:** it only reads what was on screen in a sampled frame, so comments you scroll past quickly and slides in photo posts can be missed.

## How it was built

Claude Code wrote the code. I decided what it had to do, ran it on real recordings, and sent it back when the output was wrong. The comment step exists because I noticed the useful part of a video was often in its comments.

This repo holds only the code. My own recordings and notes aren't included.
