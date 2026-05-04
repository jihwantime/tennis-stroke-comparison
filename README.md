# tennis-stroke-compare

Side-by-side stick-figure comparison of tennis strokes, time-aligned at ball contact. Drop in a clip of a pro and a clip of yourself, scrub a synchronized slider, and see exactly how the two motions diverge before and after impact.

## Why

Tools like Coaches Eye let you compare swings frame-by-frame, but they're tied to phone apps and proprietary formats. This is an open, scriptable version: pose extraction in Python, an HTML viewer that runs anywhere, and JSON data you actually own.

## How it works

1. **Extract** — `src/extract_poses.py` runs MediaPipe pose estimation on a video clip and writes joint coordinates per frame to JSON.
2. **Normalize** — `src/normalize.py` re-centers each skeleton on its hip midpoint, scales by torso length so heights match, and re-indexes frames so contact = frame 0.
3. **View** — `viewer/index.html` loads two normalized JSONs and renders both skeletons side by side with a synchronized slider centered on contact.

## Status

Early development. See [roadmap](#roadmap) below.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Extract poses from a video
python src/extract_poses.py data/raw/federer_fh.mp4 --contact-frame 47 -o data/poses/federer_fh.json

# Open the viewer
open viewer/index.html
```

## Roadmap

- [ ] Pose extraction pipeline (MediaPipe)
- [ ] Normalization (hip-center, torso-scale, contact-align)
- [ ] Static side-by-side viewer
- [ ] Synchronized slider with contact at center
- [ ] Racquet approximation (extend from wrist)
- [ ] Mirror toggle for opposite-handedness comparison
- [ ] Trajectory overlays (racquet path, hip-shoulder separation)

## License

MIT — see [LICENSE](LICENSE).
