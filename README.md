# tennis-stroke-comparison

Stick-figure stroke comparisons for tennis. Could and should work for anything else that involves hitting something with a racket probably. 

**[Try it →](https://jihwantime.github.io/tennis-stroke-comparison/viewer/index.html)** — drop in two mp4s, no install.

## Why

I physcially cannot grasp the biomechanics of a serve so maybe this can help me with stuff like the racket drop. My backhand also sucks. 

## How it works

The viewer now runs pose estimation **in the browser** — drop an mp4 into a pane and
the stick-figure overlay is generated automatically. No Python step is required.

1. **Drop a video** — `viewer/index.html` runs MediaPipe Pose Landmarker (WASM) on
   every frame, client-side, and draws the skeleton + approximated racquet on top.
2. **Mark contact** — scrub to the ball-contact frame in each pane and click
   `mark contact`; that frame becomes time 0 so the two clips align on impact.
3. **Compare** — both skeletons play side by side on a synchronized slider centered
   on contact, down to 0.125× speed for the racquet-drop detail.

The old offline path (`src/extract_poses.py` → JSON, then `load json` in the viewer)
still works and is handy for pre-computing poses, but is no longer necessary.

### Runtime and models

The MediaPipe WASM runtime is vendored in `viewer/vendor/`, so it needs no CDN. The
pose model weights (~44MB across three qualities) are **not** in the repo — they're
fetched from Google's MediaPipe storage on first use. A `lite` / `full` / `heavy`
selector in the viewer trades speed for accuracy; accuracy matters most on fast,
blurred frames like the racquet drop. It applies to the *next* video you load.

## Setup

Nothing to install to use the viewer — just serve it (see Usage).

```bash
pip install -r requirements.txt   # only for the optional offline Python scripts
```

To run extraction **fully offline**, download the models locally and point
`MODEL_PATHS` in `viewer/index.html` at `../models/...`:

```bash
mkdir -p models
for q in lite full heavy; do
  curl -L -o models/pose_landmarker_$q.task \
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_$q/float16/latest/pose_landmarker_$q.task
done
```

## Usage

The viewer uses ES modules and WASM, so it **must be served over HTTP** — opening the
file directly with `file://` will not work.

```bash
# Serve the project root, then open the viewer
python3 -m http.server 8000
# → http://localhost:8000/viewer/index.html

# Drop an mp4 into each pane; poses extract automatically.
# Scrub to impact in each pane and click "mark contact" to align them.
```

### Optional: pre-compute poses offline

```bash
python src/extract_poses.py data/raw/federer_fh.mp4 --contact-frame 47 -o data/poses/federer_fh.json
# then use "load json" in the viewer instead of re-extracting in the browser
```

## Roadmap

- [x] Pose extraction pipeline (MediaPipe)
- [x] Static side-by-side viewer
- [x] Synchronized slider with contact at center
- [x] Racquet approximation (extend from wrist)
- [x] In-browser extraction — drop an mp4, overlay is generated automatically
- [x] In-viewer contact marking to align two clips
- [ ] Normalization (hip-center, torso-scale, contact-align) wired into the viewer
- [ ] Mirror toggle for opposite-handedness comparison
- [ ] Trajectory overlays (racquet path, hip-shoulder separation)

## License

MIT — see [LICENSE](LICENSE).
