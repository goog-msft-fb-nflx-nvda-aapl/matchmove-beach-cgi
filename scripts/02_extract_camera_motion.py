#!/usr/bin/env python3
"""Extract approximate temporal/spatial camera-motion context with OpenCV.

This is a planning signal for matchmove, not a final camera solve.
It writes:
  context/camera_motion.json
  context/track_overlay.mp4
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def resize_for_analysis(frame: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame, 1.0
    scale = max_width / width
    resized = cv2.resize(frame, (max_width, int(height * scale)))
    return resized, scale


def affine_to_motion(matrix: np.ndarray | None) -> dict:
    if matrix is None:
        return {"dx": None, "dy": None, "scale": None, "rotation_deg": None}
    a, b, tx = matrix[0]
    c, d, ty = matrix[1]
    scale = math.sqrt(max(0.0, a * d - b * c))
    rotation = math.degrees(math.atan2(c, a))
    return {
        "dx": float(tx),
        "dy": float(ty),
        "scale": float(scale),
        "rotation_deg": float(rotation),
    }


def draw_tracks(frame: np.ndarray, tracks: list[list[tuple[float, float]]]) -> np.ndarray:
    overlay = frame.copy()
    for track in tracks:
        if len(track) < 2:
            continue
        points = np.array(track[-20:], dtype=np.int32)
        color = (0, 255, 180)
        cv2.polylines(overlay, [points], False, color, 1, cv2.LINE_AA)
        cv2.circle(overlay, tuple(points[-1]), 2, (0, 90, 255), -1, cv2.LINE_AA)
    return overlay


def classify_motion(rows: list[dict]) -> dict:
    valid = [row for row in rows if row["inliers"] >= 12 and row["motion"]["dx"] is not None]
    if not valid:
        return {
            "summary": "insufficient reliable local feature tracks",
            "median_dx_per_step": None,
            "median_dy_per_step": None,
            "median_rotation_deg_per_step": None,
            "median_scale_per_step": None,
        }

    dx = np.array([row["motion"]["dx"] for row in valid], dtype=np.float32)
    dy = np.array([row["motion"]["dy"] for row in valid], dtype=np.float32)
    rot = np.array([row["motion"]["rotation_deg"] for row in valid], dtype=np.float32)
    scale = np.array([row["motion"]["scale"] for row in valid], dtype=np.float32)

    abs_x = abs(float(np.median(dx)))
    abs_y = abs(float(np.median(dy)))
    abs_rot = abs(float(np.median(rot)))
    scale_delta = abs(float(np.median(scale)) - 1.0)

    labels = []
    if abs_x > abs_y * 1.5 and abs_x > 2.0:
        labels.append("horizontal pan")
    if abs_y > abs_x * 1.5 and abs_y > 2.0:
        labels.append("vertical tilt")
    if scale_delta > 0.01:
        labels.append("zoom or forward/backward motion")
    if abs_rot > 0.2:
        labels.append("roll/handheld rotation")
    if not labels:
        labels.append("slow handheld or mostly static")

    return {
        "summary": ", ".join(labels),
        "median_dx_per_step": float(np.median(dx)),
        "median_dy_per_step": float(np.median(dy)),
        "median_rotation_deg_per_step": float(np.median(rot)),
        "median_scale_per_step": float(np.median(scale)),
        "reliable_step_count": len(valid),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="video.mp4")
    parser.add_argument("--out", default="context")
    parser.add_argument("--step", type=int, default=10, help="Analyze every Nth frame.")
    parser.add_argument("--max-width", type=int, default=720)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    ok, first = cap.read()
    if not ok:
        raise SystemExit("Could not read first frame")
    first, scale = resize_for_analysis(first, args.max_width)
    prev_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    prev_pts = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=900,
        qualityLevel=0.01,
        minDistance=9,
        blockSize=7,
    )
    tracks: list[list[tuple[float, float]]] = []
    if prev_pts is not None:
        tracks = [[tuple(point.ravel())] for point in prev_pts]

    overlay_path = out_dir / "track_overlay.mp4"
    writer = cv2.VideoWriter(
        str(overlay_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1.0, fps / args.step),
        (first.shape[1], first.shape[0]),
    )
    writer.write(draw_tracks(first, tracks))

    rows = []
    frame_idx = 0
    analyzed_idx = 0
    cumulative_dx = 0.0
    cumulative_dy = 0.0

    while True:
        target = frame_idx + args.step
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx = target
        analyzed_idx += 1

        frame, _ = resize_for_analysis(frame, args.max_width)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_pts is None or len(prev_pts) < 40:
            prev_pts = cv2.goodFeaturesToTrack(
                prev_gray,
                maxCorners=900,
                qualityLevel=0.01,
                minDistance=9,
                blockSize=7,
            )
            tracks = [[tuple(point.ravel())] for point in prev_pts] if prev_pts is not None else []

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            gray,
            prev_pts,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if next_pts is None or status is None:
            prev_gray = gray
            prev_pts = None
            continue

        good_prev = prev_pts[status.ravel() == 1]
        good_next = next_pts[status.ravel() == 1]
        matrix, inlier_mask = cv2.estimateAffinePartial2D(
            good_prev,
            good_next,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )
        inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
        motion = affine_to_motion(matrix)
        if motion["dx"] is not None:
            cumulative_dx += motion["dx"]
            cumulative_dy += motion["dy"]

        new_tracks: list[list[tuple[float, float]]] = []
        for track, good, status_value in zip(tracks, next_pts, status.ravel()):
            if status_value == 1:
                track.append(tuple(good.ravel()))
                new_tracks.append(track)
        tracks = new_tracks

        rows.append(
            {
                "frame": frame_idx,
                "timestamp_sec": frame_idx / fps,
                "tracked_points": int(len(good_next)),
                "inliers": inliers,
                "motion": motion,
                "cumulative_dx": cumulative_dx,
                "cumulative_dy": cumulative_dy,
            }
        )

        writer.write(draw_tracks(frame, tracks))
        prev_gray = gray
        prev_pts = good_next.reshape(-1, 1, 2)

    cap.release()
    writer.release()

    summary = {
        "source_video": args.video,
        "frame_count": frame_count,
        "fps": fps,
        "original_width": width,
        "original_height": height,
        "analysis_scale": scale,
        "analysis_step_frames": args.step,
        "analysis_step_seconds": args.step / fps,
        "motion_classification": classify_motion(rows),
        "tracking_quality": {
            "median_tracked_points": float(np.median([r["tracked_points"] for r in rows]))
            if rows
            else 0.0,
            "median_inliers": float(np.median([r["inliers"] for r in rows])) if rows else 0.0,
            "low_inlier_steps": sum(1 for r in rows if r["inliers"] < 20),
            "analyzed_steps": len(rows),
        },
        "per_step": rows,
        "overlay_video": str(overlay_path),
        "interpretation_notes": [
            "Use this as a rough motion/trackability report only.",
            "Final camera parameters should come from Blender camera tracking, COLMAP/pycolmap, or another SfM solver.",
            "Low inliers around sky/water frames are expected because those areas have weak stable texture.",
        ],
    }

    out_path = out_dir / "camera_motion.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Wrote {overlay_path}")
    print(summary["motion_classification"])


if __name__ == "__main__":
    main()
