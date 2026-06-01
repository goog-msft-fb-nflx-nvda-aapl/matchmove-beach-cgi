#!/usr/bin/env python3
"""GPU semantic/spatial context extraction for sampled matchmove frames.

Outputs per-frame:
  - zero-shot detections for scene classes
  - relative depth map
  - annotated preview image

This intentionally extracts context, not final CGI or final camera solve.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline


DEFAULT_LABELS = [
    "sand beach",
    "ocean water",
    "blue sky",
    "person",
    "scaffolding structure",
    "building",
    "wood pile",
    "railing",
    "island",
    "tower",
    "boat",
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_depth(depth_image: Image.Image) -> np.ndarray:
    depth = np.asarray(depth_image).astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    min_value = float(depth.min())
    max_value = float(depth.max())
    if max_value <= min_value:
        return np.zeros_like(depth, dtype=np.uint8)
    normalized = (depth - min_value) / (max_value - min_value)
    return (normalized * 255.0).astype(np.uint8)


def draw_detections(image: Image.Image, detections: list[dict]) -> Image.Image:
    preview = image.convert("RGB")
    draw = ImageDraw.Draw(preview)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = None

    for det in detections:
        box = det.get("box", {})
        xmin = float(box.get("xmin", 0))
        ymin = float(box.get("ymin", 0))
        xmax = float(box.get("xmax", 0))
        ymax = float(box.get("ymax", 0))
        label = det.get("label", "object")
        score = float(det.get("score", 0.0))
        color = (255, 210, 0)
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
        text = f"{label} {score:.2f}"
        text_box = draw.textbbox((xmin, ymin), text, font=font)
        draw.rectangle(text_box, fill=(0, 0, 0))
        draw.text((xmin, ymin), text, fill=color, font=font)
    return preview


def center_from_box(box: dict) -> tuple[float, float]:
    return (
        (float(box["xmin"]) + float(box["xmax"])) / 2.0,
        (float(box["ymin"]) + float(box["ymax"])) / 2.0,
    )


def propose_anchor_points(detections: list[dict], image_size: tuple[int, int]) -> list[dict]:
    width, height = image_size
    anchors = []

    sand_dets = [
        det for det in detections if "sand" in det.get("label", "").lower() and det.get("box")
    ]
    for det in sand_dets:
        box = det["box"]
        x = (float(box["xmin"]) + float(box["xmax"])) / 2.0
        y = min(height - 1.0, float(box["ymax"]) - 0.18 * (float(box["ymax"]) - float(box["ymin"])))
        anchors.append(
            {
                "type": "sand_ground_anchor",
                "x": x,
                "y": y,
                "reason": "inside detected sand/beach region, biased toward lower part for grounded CGI",
            }
        )

    if not anchors:
        anchors.append(
            {
                "type": "fallback_lower_third_anchor",
                "x": width * 0.5,
                "y": height * 0.72,
                "reason": "fallback based on semantic and OpenCV context: usable sand plane is in lower frame",
            }
        )
    return anchors[:5]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", default="context/media_context.json")
    parser.add_argument("--brief", default="context/context_brief.json")
    parser.add_argument("--out", default="context/gpu_semantic_spatial")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--det-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--depth-model", default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--score-threshold", type=float, default=0.22)
    parser.add_argument("--labels", nargs="*", default=DEFAULT_LABELS)
    args = parser.parse_args()

    out_dir = Path(args.out)
    preview_dir = out_dir / "previews"
    depth_dir = out_dir / "depth"
    preview_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    media = load_json(Path(args.context))
    brief = load_json(Path(args.brief)) if Path(args.brief).exists() else {}
    frame_rows = media["sampled_frames"]

    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    detector = pipeline(
        task="zero-shot-object-detection",
        model=args.det_model,
        device=device,
    )
    depth_estimator = pipeline(
        task="depth-estimation",
        model=args.depth_model,
        device=device,
    )

    results = {
        "models": {
            "detector": args.det_model,
            "depth": args.depth_model,
            "device": device,
        },
        "labels": args.labels,
        "location": brief.get("site", {}),
        "frames": [],
    }

    for row in frame_rows:
        frame_path = Path(row["path"])
        image = Image.open(frame_path).convert("RGB")

        raw_detections = detector(
            image,
            candidate_labels=args.labels,
            threshold=args.score_threshold,
        )
        detections = []
        for det in raw_detections:
            detections.append(
                {
                    "label": det["label"],
                    "score": float(det["score"]),
                    "box": {key: float(value) for key, value in det["box"].items()},
                    "center": center_from_box(det["box"]),
                }
            )

        depth_output = depth_estimator(image)
        depth_norm = normalize_depth(depth_output["depth"])
        depth_path = depth_dir / f"{frame_path.stem}_depth.png"
        cv2.imwrite(str(depth_path), cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO))

        preview = draw_detections(image, detections)
        preview_path = preview_dir / f"{frame_path.stem}_detections.jpg"
        preview.save(preview_path, quality=92)

        results["frames"].append(
            {
                "index": row["index"],
                "timestamp_sec": row["timestamp_sec"],
                "frame_path": str(frame_path),
                "detections": detections,
                "depth_preview": str(depth_path),
                "detection_preview": str(preview_path),
                "candidate_anchors": propose_anchor_points(detections, image.size),
            }
        )
        print(f"processed frame {row['index']} at {row['timestamp_sec']}s")

    out_path = out_dir / "semantic_spatial_context.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
