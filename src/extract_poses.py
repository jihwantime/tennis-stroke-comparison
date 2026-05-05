"""
Extract per-frame body pose keypoints from a tennis video.

Runs MediaPipe Pose Landmarker (Tasks API) on every frame of the input video
and writes joint coordinates to JSON. Time is stored relative to a user-specified
contact frame so that downstream alignment is straightforward.

Usage:
    python extract_poses.py VIDEO --contact-frame N [options]

Example:
    python extract_poses.py data/raw/federer_fh.mp4 \\
        --contact-frame 47 \\
        --player federer \\
        --output data/poses/federer_fh.json \\
        --debug-video

Setup (one-time): download the pose model file:
    mkdir -p models
    curl -L -o models/pose_landmarker_heavy.task \\
        https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2  # opencv-python
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# MediaPipe Pose Landmarker returns 33 landmarks. We keep only the ones useful for
# stroke comparison. The Tasks API uses the same indexing as the old solutions API.
# Index reference: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
KEYPOINTS = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

# Skeleton edges for the debug video overlay. Each tuple connects two keypoints.
SKELETON_EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

DEFAULT_MODEL_PATH = Path("models/pose_landmarker_heavy.task")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", type=Path, help="Path to input video file")
    p.add_argument(
        "--contact-frame", type=int, required=True,
        help="Frame number where ball contact occurs (0-indexed). "
             "Used as the time-zero reference for downstream alignment.",
    )
    p.add_argument(
        "--player", type=str, default="unknown",
        help="Player label saved in metadata (e.g. 'federer', 'me')",
    )
    p.add_argument(
        "--handedness", type=str, choices=["right", "left"], default="right",
        help="Which hand holds the racquet (default: right). The viewer "
             "extends the racquet line from this wrist.",
    )
    p.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output JSON path (default: data/poses/<video_stem>.json)",
    )
    p.add_argument(
        "--debug-video", action="store_true",
        help="Also write a video with the skeleton drawn on top, "
             "for visual verification that pose extraction worked.",
    )
    p.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL_PATH,
        help=f"Path to pose_landmarker .task model file (default: {DEFAULT_MODEL_PATH})",
    )
    p.add_argument(
        "--min-detection-confidence", type=float, default=0.5,
        help="Min pose detection confidence (default 0.5)",
    )
    p.add_argument(
        "--min-tracking-confidence", type=float, default=0.5,
        help="Min pose tracking confidence (default 0.5)",
    )
    return p.parse_args()


def extract_landmarks(
    pose_landmarks: list, frame_width: int, frame_height: int
) -> dict[str, dict[str, float]]:
    """Pull just our keypoints from the full 33-landmark output.

    Returns a dict mapping keypoint name to {x, y, visibility}. Coordinates
    are in pixels (top-left origin, x rightward, y downward). Visibility is
    MediaPipe's confidence that the joint is in-frame and unoccluded.
    """
    out = {}
    for name, idx in KEYPOINTS.items():
        lm = pose_landmarks[idx]
        out[name] = {
            "x": lm.x * frame_width,
            "y": lm.y * frame_height,
            "visibility": lm.visibility,
        }
    return out


def draw_skeleton(frame, joints: dict[str, dict[str, float]]) -> None:
    """Mutate `frame` in place: draw skeleton edges and joint dots."""
    # Edges first so dots sit on top
    for a, b in SKELETON_EDGES:
        ja, jb = joints.get(a), joints.get(b)
        if ja is None or jb is None:
            continue
        if ja["visibility"] < 0.3 or jb["visibility"] < 0.3:
            continue
        pa = (int(ja["x"]), int(ja["y"]))
        pb = (int(jb["x"]), int(jb["y"]))
        cv2.line(frame, pa, pb, color=(0, 255, 0), thickness=2)

    for name, j in joints.items():
        if j["visibility"] < 0.3:
            continue
        cv2.circle(frame, (int(j["x"]), int(j["y"])), radius=4,
                   color=(0, 0, 255), thickness=-1)


def build_landmarker(model_path: Path, min_detection: float, min_tracking: float):
    """Construct a PoseLandmarker configured for video processing."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Pose model not found at {model_path}.\n"
            f"Download it with:\n"
            f"  mkdir -p {model_path.parent}\n"
            f"  curl -L -o {model_path} \\\n"
            f"    https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            f"pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
        )

    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=min_detection,
        min_tracking_confidence=min_tracking,
        min_pose_presence_confidence=0.5,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def main() -> int:
    args = parse_args()

    if not args.video.exists():
        print(f"Error: video not found: {args.video}", file=sys.stderr)
        return 1

    # Resolve default output path
    output_path = args.output
    if output_path is None:
        output_path = Path("data/poses") / f"{args.video.stem}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Open video and read metadata
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"Error: could not open video: {args.video}", file=sys.stderr)
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not (0 <= args.contact_frame < total_frames):
        print(
            f"Error: --contact-frame {args.contact_frame} is out of range "
            f"[0, {total_frames}) for this video.",
            file=sys.stderr,
        )
        cap.release()
        return 1

    print(f"Video: {args.video}")
    print(f"  {width}x{height} @ {fps:.1f} fps, {total_frames} frames")
    print(f"  Contact frame: {args.contact_frame} (= time 0)")

    # Set up debug video writer if requested
    debug_writer = None
    if args.debug_video:
        debug_path = output_path.with_suffix(".debug.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        debug_writer = cv2.VideoWriter(str(debug_path), fourcc, fps, (width, height))
        print(f"  Debug video: {debug_path}")

    # Build landmarker
    try:
        landmarker = build_landmarker(
            args.model, args.min_detection_confidence, args.min_tracking_confidence
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        cap.release()
        return 1

    frames_data = []
    frame_idx = 0
    missed_frames = 0

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            # MediaPipe expects RGB; wrap as mp.Image
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # In VIDEO mode, the Tasks API requires a monotonically increasing
            # timestamp in milliseconds. Derive it from frame index and fps.
            timestamp_ms = int((frame_idx / fps) * 1000) if fps > 0 else frame_idx
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            joints = None
            if result.pose_landmarks:
                # pose_landmarks is a list of detections; we configured num_poses=1
                joints = extract_landmarks(result.pose_landmarks[0], width, height)
            else:
                missed_frames += 1

            # Time relative to contact (ms). Negative = before, 0 = contact, positive = after.
            time_ms = ((frame_idx - args.contact_frame) / fps) * 1000.0

            frames_data.append({
                "frame_idx": frame_idx,
                "time_ms": round(time_ms, 2),
                "joints": joints,  # may be None if pose not detected
            })

            # Debug video: draw skeleton overlay if joints found
            if debug_writer is not None:
                if joints is not None:
                    draw_skeleton(frame_bgr, joints)
                # Mark contact frame visually
                if frame_idx == args.contact_frame:
                    cv2.putText(frame_bgr, "CONTACT", (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                debug_writer.write(frame_bgr)

            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"  Processed {frame_idx}/{total_frames} frames...")
    finally:
        cap.release()
        if debug_writer is not None:
            debug_writer.release()
        landmarker.close()

    # Assemble output
    output = {
        "metadata": {
            "source_video": str(args.video),
            "player": args.player,
            "handedness": args.handedness,
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": frame_idx,
            "contact_frame": args.contact_frame,
            "missed_frames": missed_frames,
            "keypoints": list(KEYPOINTS.keys()),
        },
        "frames": frames_data,
    }

    with output_path.open("w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDone. Wrote {output_path}")
    print(f"  {frame_idx} frames processed, {missed_frames} with no pose detected")
    if missed_frames > 0:
        pct = 100 * missed_frames / frame_idx
        print(f"  ({pct:.1f}% miss rate — if high, try --min-detection-confidence 0.3)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
