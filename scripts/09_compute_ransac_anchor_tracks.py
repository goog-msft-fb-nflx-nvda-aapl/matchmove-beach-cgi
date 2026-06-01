#!/usr/bin/env python3
"""Track CGI anchor points across frames with optical flow + RANSAC affine links."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_gray(cap: cv2.VideoCapture, frame_idx: int, width: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_idx}")
    h, w = frame.shape[:2]
    scale = width / w
    frame = cv2.resize(frame, (width, round(h * scale)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def affine_between(prev_gray: np.ndarray, next_gray: np.ndarray) -> tuple[np.ndarray, int, int]:
    pts = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=1200,
        qualityLevel=0.01,
        minDistance=8,
        blockSize=7,
    )
    if pts is None or len(pts) < 20:
        return np.eye(3, dtype=np.float64), 0, 0

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        next_gray,
        pts,
        None,
        winSize=(25, 25),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01),
    )
    if next_pts is None or status is None:
        return np.eye(3, dtype=np.float64), 0, int(len(pts))

    good_prev = pts[status.ravel() == 1]
    good_next = next_pts[status.ravel() == 1]
    if len(good_prev) < 20:
        return np.eye(3, dtype=np.float64), 0, int(len(good_prev))

    affine, inliers = cv2.estimateAffinePartial2D(
        good_prev,
        good_next,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=3000,
        confidence=0.995,
    )
    if affine is None:
        return np.eye(3, dtype=np.float64), 0, int(len(good_prev))

    mat = np.eye(3, dtype=np.float64)
    mat[:2, :] = affine
    return mat, int(inliers.sum()) if inliers is not None else 0, int(len(good_prev))


def apply(mat: np.ndarray, xy: tuple[float, float]) -> tuple[float, float]:
    point = mat @ np.array([xy[0], xy[1], 1.0], dtype=np.float64)
    return float(point[0] / point[2]), float(point[1] / point[2])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="video.mp4")
    parser.add_argument("--segmentation", default="context/gpu_scene_segmentation/scene_segmentation_context.json")
    parser.add_argument("--out", default="context/ransac_anchor_tracks.json")
    parser.add_argument("--start-frame", type=int, default=300)
    parser.add_argument("--end-frame", type=int, default=540)
    parser.add_argument("--step", type=int, default=3)
    parser.add_argument("--analysis-width", type=int, default=720)
    args = parser.parse_args()

    seg = load_json(Path(args.segmentation))
    anchor_row = min(seg["frames"], key=lambda row: abs(float(row["timestamp_sec"]) - 14.884))
    anchor = anchor_row["candidate_anchors"][0]
    anchor_frame = round(float(anchor_row["timestamp_sec"]) * 30 / args.step) * args.step
    anchor_frame = min(max(anchor_frame, args.start_frame), args.end_frame)
    anchor_xy = (float(anchor["x"]), float(anchor["y"]))

    frame_indices = list(range(args.start_frame, args.end_frame + 1, args.step))
    if anchor_frame not in frame_indices:
        anchor_frame = min(frame_indices, key=lambda idx: abs(idx - anchor_frame))
    anchor_i = frame_indices.index(anchor_frame)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1080)
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1920)
    scale_to_source = source_width / args.analysis_width

    grays = [extract_gray(cap, idx, args.analysis_width) for idx in frame_indices]
    cap.release()

    links = []
    mats = []
    for idx in range(len(frame_indices) - 1):
        mat, inliers, tracked = affine_between(grays[idx], grays[idx + 1])
        mats.append(mat)
        links.append(
            {
                "from_frame": frame_indices[idx],
                "to_frame": frame_indices[idx + 1],
                "tracked_points": tracked,
                "ransac_inliers": inliers,
                "affine_3x3": mat.tolist(),
            }
        )

    transforms = [None] * len(frame_indices)
    transforms[anchor_i] = np.eye(3, dtype=np.float64)
    accum = np.eye(3, dtype=np.float64)
    for i in range(anchor_i, len(frame_indices) - 1):
        accum = mats[i] @ accum
        transforms[i + 1] = accum.copy()
    accum = np.eye(3, dtype=np.float64)
    for i in range(anchor_i - 1, -1, -1):
        try:
            inv = np.linalg.inv(mats[i])
        except np.linalg.LinAlgError:
            inv = np.eye(3, dtype=np.float64)
        accum = inv @ accum
        transforms[i] = accum.copy()

    tracks = []
    for idx, mat in zip(frame_indices, transforms):
        x, y = apply(mat, anchor_xy)
        tracks.append(
            {
                "frame": idx,
                "timestamp_sec": idx / fps,
                "analysis_x": x,
                "analysis_y": y,
                "source_x": x * scale_to_source,
                "source_y": y * scale_to_source,
            }
        )

    result = {
        "source_video": args.video,
        "fps": fps,
        "source_width": source_width,
        "source_height": source_height,
        "analysis_width": args.analysis_width,
        "analysis_height": grays[0].shape[0],
        "anchor_frame": anchor_frame,
        "anchor_xy_analysis": anchor_xy,
        "frame_step": args.step,
        "links": links,
        "tracks": tracks,
        "notes": [
            "Each link maps the previous sampled frame to the next sampled frame.",
            "Tracks are accumulated from the anchor frame using RANSAC-filtered affine transforms.",
        ],
    }
    out = Path(args.out)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    print(f"Anchor frame {anchor_frame}, {len(tracks)} tracked positions")
    print(f"Median inliers {np.median([link['ransac_inliers'] for link in links]):.1f}")


if __name__ == "__main__":
    main()
