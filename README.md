# tennis-stroke-comparison

Stick-figure stroke comparisons for tennis. Could and should work for anything else that involves hitting something with a racket probably. 

## Why

I physcially cannot grasp the biomechanics of a serve so maybe this can help me with stuff like the racket drop. My backhand also sucks. 

## How it works

1. **Extract** — `src/extract_poses.py` runs MediaPipe pose estimation on a video clip and writes joint coordinates per frame to JSON.
2. **Normalize** — `src/normalize.py` re-centers each skeleton on its hip midpoint, scales by torso length so heights match, and re-indexes frames so contact = frame 0.
3. **View** — `viewer/index.html` loads two normalized JSONs and renders both skeletons side by side with a synchronized slider centered on contact.

## Setup

```bash
pip install -r requirements.txt

# Download the pose model (one-time)
mkdir -p models
curl -L -o models/pose_landmarker_heavy.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
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
