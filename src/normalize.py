"""
Normalize a raw pose JSON for stroke comparison.

Three transformations:
1. Hip-centering — translate so hip midpoint is at origin in every frame
2. Torso-scaling — divide by torso length so heights are comparable across players
3. Frame interpolation — fill in any frames MediaPipe missed via linear interpolation
                         from neighbors

The output JSON has the same shape as the input, but joint coordinates are
in normalized units (torso-lengths from the hip midpoint) instead of pixels.

Usage:
    python normalize.py INPUT_JSON [-o OUTPUT_JSON]

Example:
    python src/normalize.py data/poses/murray_fh.json
    # writes data/poses/murray_fh.normalized.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# We use these four landmarks to compute the torso reference frame.
TORSO_LANDMARKS = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
MIN_TORSO_VIS = 0.7  # minimum visibility for torso ref frame selection


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path, help="Path to raw pose JSON (output of extract_poses.py)")
    p.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output path (default: <input_stem>.normalized.json in the same dir)",
    )
    return p.parse_args()


def midpoint(j1: dict, j2: dict) -> tuple[float, float]:
    """Average of two joint positions. Visibility is ignored — caller decides
    whether the joints are reliable enough to use."""
    return ((j1["x"] + j2["x"]) / 2, (j1["y"] + j2["y"]) / 2)


def find_torso_reference(frames: list[dict], contact_idx: int) -> tuple[float, int]:
    """Pick the frame nearest to contact where all four torso landmarks are
    well-visible, and compute its torso length (shoulder midpoint to hip midpoint).

    Returns (torso_length_pixels, frame_index_used). Falls back to the contact
    frame even if visibility is poor, with a warning printed.
    """
    n = len(frames)
    # Walk outward from contact: contact, contact-1, contact+1, contact-2, ...
    visit_order = [contact_idx]
    for delta in range(1, n):
        if contact_idx - delta >= 0:
            visit_order.append(contact_idx - delta)
        if contact_idx + delta < n:
            visit_order.append(contact_idx + delta)

    for idx in visit_order:
        joints = frames[idx].get("joints")
        if joints is None:
            continue
        if all(joints[lm]["visibility"] >= MIN_TORSO_VIS for lm in TORSO_LANDMARKS):
            sm = midpoint(joints["left_shoulder"], joints["right_shoulder"])
            hm = midpoint(joints["left_hip"], joints["right_hip"])
            torso_len = ((sm[0] - hm[0]) ** 2 + (sm[1] - hm[1]) ** 2) ** 0.5
            if torso_len > 0:
                return torso_len, idx

    # Fallback: use whatever the contact frame has
    print(
        f"Warning: no frame near contact had all torso landmarks at visibility "
        f">= {MIN_TORSO_VIS}. Falling back to contact frame {contact_idx} as-is.",
        file=sys.stderr,
    )
    joints = frames[contact_idx].get("joints")
    if joints is None:
        raise ValueError(
            f"Contact frame {contact_idx} has no pose detected. Cannot normalize. "
            f"Re-run extraction with a different --contact-frame, or with "
            f"--min-detection-confidence 0.3 to recover more frames."
        )
    sm = midpoint(joints["left_shoulder"], joints["right_shoulder"])
    hm = midpoint(joints["left_hip"], joints["right_hip"])
    torso_len = ((sm[0] - hm[0]) ** 2 + (sm[1] - hm[1]) ** 2) ** 0.5
    return torso_len, contact_idx


def interpolate_missing(frames: list[dict]) -> int:
    """Fill in `joints: null` frames by linearly interpolating from the
    nearest valid frames on either side. Modifies frames in place.

    Returns the number of frames that were filled in.

    Edge frames (missing at the start or end with no anchor on one side)
    are filled by copying from the nearest valid frame — not ideal but
    better than null.
    """
    n = len(frames)
    valid_indices = [i for i, f in enumerate(frames) if f.get("joints") is not None]

    if not valid_indices:
        raise ValueError("No frames have valid pose data. Cannot interpolate.")

    filled = 0
    for i in range(n):
        if frames[i].get("joints") is not None:
            continue

        # Find nearest valid neighbors
        left = max((vi for vi in valid_indices if vi < i), default=None)
        right = min((vi for vi in valid_indices if vi > i), default=None)

        if left is not None and right is not None:
            # Interpolate
            t = (i - left) / (right - left)
            j_left = frames[left]["joints"]
            j_right = frames[right]["joints"]
            interpolated = {}
            for name in j_left:
                interpolated[name] = {
                    "x": j_left[name]["x"] + t * (j_right[name]["x"] - j_left[name]["x"]),
                    "y": j_left[name]["y"] + t * (j_right[name]["y"] - j_left[name]["y"]),
                    "visibility": min(j_left[name]["visibility"], j_right[name]["visibility"]),
                }
            frames[i]["joints"] = interpolated
            frames[i]["interpolated"] = True
        elif left is not None:
            # Edge case: missing at the end. Copy from last valid.
            frames[i]["joints"] = {k: dict(v) for k, v in frames[left]["joints"].items()}
            frames[i]["interpolated"] = True
        elif right is not None:
            # Edge case: missing at the start. Copy from first valid.
            frames[i]["joints"] = {k: dict(v) for k, v in frames[right]["joints"].items()}
            frames[i]["interpolated"] = True

        filled += 1

    return filled


def normalize_frames(
    frames: list[dict], torso_length: float
) -> list[dict]:
    """Apply hip-centering + torso-scaling to every frame.

    For each frame: translate so hip midpoint is at (0, 0), then divide all
    coordinates by torso_length. Visibility is preserved as-is.
    """
    out = []
    for f in frames:
        joints = f["joints"]
        hm = midpoint(joints["left_hip"], joints["right_hip"])
        normalized_joints = {}
        for name, j in joints.items():
            normalized_joints[name] = {
                "x": (j["x"] - hm[0]) / torso_length,
                "y": (j["y"] - hm[1]) / torso_length,
                "visibility": j["visibility"],
            }
        out.append({
            "frame_idx": f["frame_idx"],
            "time_ms": f["time_ms"],
            "joints": normalized_joints,
            "interpolated": f.get("interpolated", False),
        })
    return out


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        print(f"Error: input not found: {args.input}", file=sys.stderr)
        return 1

    output_path = args.output
    if output_path is None:
        output_path = args.input.with_name(args.input.stem + ".normalized.json")

    with args.input.open() as f:
        data = json.load(f)

    metadata = data["metadata"]
    frames = data["frames"]
    contact_frame = metadata["contact_frame"]

    print(f"Loaded {args.input}")
    print(f"  Player: {metadata.get('player', 'unknown')}")
    print(f"  {len(frames)} frames, {metadata.get('missed_frames', 0)} missed by extraction")

    # Step 1: interpolate missing frames
    try:
        filled = interpolate_missing(frames)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(f"  Filled {filled} missing frames via interpolation")

    # Step 2: pick a reference frame and compute torso length
    try:
        torso_len, ref_frame = find_torso_reference(frames, contact_frame)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(f"  Torso reference: frame {ref_frame}, length {torso_len:.1f} px")

    # Step 3: hip-center and torso-scale every frame
    normalized_frames = normalize_frames(frames, torso_len)

    # Build output
    output = {
        "metadata": {
            **metadata,
            "normalized": True,
            "torso_reference_frame": ref_frame,
            "torso_length_pixels": torso_len,
            "interpolated_frames": filled,
            "coord_system": "hip-centered, torso-scaled. y increases downward.",
        },
        "frames": normalized_frames,
    }

    with output_path.open("w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDone. Wrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
