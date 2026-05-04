# data/

## `raw/`
Original video clips. **Gitignored** — these are large, often not redistributable, and reproducible from source.

Naming convention: `{player}_{stroke}_{angle}.mp4`
- e.g. `federer_fh_back.mp4`, `me_fh_back.mp4`

## `poses/`
Extracted joint coordinate JSONs, one per video clip. **Committed to git** — these are small, derived artifacts that the viewer needs.

Each JSON contains:
- `metadata`: source video, fps, contact frame, player name
- `frames`: list of `{time_ms, joints: {...}}` where joints are normalized 2D coordinates

Time is stored relative to the contact frame: contact = `time_ms: 0`, before contact is negative, after contact is positive.
