#!/usr/bin/env python3
"""GPU semantic segmentation for broad matchmove regions.

Uses an ADE20K SegFormer model to produce per-keyframe masks for classes such
as sky, sea/water, sand/earth, person, building, railing/fence, and rock.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation


TARGET_GROUPS = {
    "sky": ["sky"],
    "water": ["water", "sea"],
    "sand_or_ground": ["sand", "earth", "dirt", "field"],
    "people": ["person"],
    "structures": ["building", "house", "skyscraper", "wall"],
    "rail_or_fence": ["railing", "fence"],
    "rocks_or_lumber_proxy": ["rock", "stone"],
}


COLORS = {
    "sky": (90, 170, 255),
    "water": (40, 110, 220),
    "sand_or_ground": (230, 190, 90),
    "people": (255, 60, 80),
    "structures": (120, 120, 120),
    "rail_or_fence": (255, 220, 0),
    "rocks_or_lumber_proxy": (150, 95, 45),
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def labels_for_groups(id2label: dict[int, str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for group, needles in TARGET_GROUPS.items():
        ids = []
        for idx, label in id2label.items():
            label_lower = str(label).lower()
            if any(needle in label_lower for needle in needles):
                ids.append(int(idx))
        groups[group] = sorted(set(ids))
    return groups


def mask_for_ids(segmentation: np.ndarray, ids: list[int]) -> np.ndarray:
    if not ids:
        return np.zeros(segmentation.shape, dtype=np.uint8)
    return np.isin(segmentation, ids).astype(np.uint8) * 255


def coverage(mask: np.ndarray) -> float:
    return float((mask > 0).sum() / mask.size)


def anchor_from_ground(mask: np.ndarray) -> dict | None:
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() < 100:
        return None
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    if num_labels <= 1:
        return None

    height, width = mask.shape
    candidates = []
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[label_id]
        if area < 0.02 * width * height:
            continue
        bottom_bias = (y + h) / height
        candidates.append((area * bottom_bias, area, x, y, w, h, cx, cy))
    if not candidates:
        return None
    _, area, x, y, w, h, cx, _ = max(candidates)
    anchor_y = min(height - 1, y + int(h * 0.72))
    return {
        "type": "segmented_ground_anchor",
        "x": float(cx),
        "y": float(anchor_y),
        "component_box": [x, y, w, h],
        "component_area_px": int(area),
        "reason": "largest lower-frame ground/sand-like semantic component",
    }


def make_overlay(image: np.ndarray, masks: dict[str, np.ndarray]) -> np.ndarray:
    overlay = image.copy()
    color_layer = np.zeros_like(image)
    alpha = np.zeros(image.shape[:2], dtype=np.float32)
    for name, mask in masks.items():
        if name not in COLORS:
            continue
        active = mask > 0
        color_layer[active] = COLORS[name]
        alpha[active] = np.maximum(alpha[active], 0.45)
    return (overlay * (1 - alpha[..., None]) + color_layer * alpha[..., None]).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", default="context/media_context.json")
    parser.add_argument("--out", default="context/gpu_scene_segmentation")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", default="nvidia/segformer-b0-finetuned-ade-512-512")
    args = parser.parse_args()

    out_dir = Path(args.out)
    mask_dir = out_dir / "masks"
    overlay_dir = out_dir / "overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    media = load_json(Path(args.context))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForSemanticSegmentation.from_pretrained(args.model).to(device)
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    groups = labels_for_groups(id2label)

    result = {
        "model": args.model,
        "device": str(device),
        "target_groups": groups,
        "id2label": id2label,
        "frames": [],
    }

    for row in media["sampled_frames"]:
        frame_path = Path(row["path"])
        pil_image = Image.open(frame_path).convert("RGB")
        width, height = pil_image.size

        inputs = processor(images=pil_image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = torch.nn.functional.interpolate(
            outputs.logits,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        segmentation = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)

        bgr = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
        masks = {name: mask_for_ids(segmentation, ids) for name, ids in groups.items()}
        overlay = make_overlay(bgr, masks)

        frame_record = {
            "index": row["index"],
            "timestamp_sec": row["timestamp_sec"],
            "frame_path": str(frame_path),
            "coverage": {},
            "masks": {},
            "overlay": None,
            "candidate_anchors": [],
        }

        for name, mask in masks.items():
            if coverage(mask) <= 0.001:
                continue
            mask_path = mask_dir / f"{frame_path.stem}_{name}.png"
            cv2.imwrite(str(mask_path), mask)
            frame_record["coverage"][name] = coverage(mask)
            frame_record["masks"][name] = str(mask_path)

        ground_anchor = anchor_from_ground(masks["sand_or_ground"])
        if ground_anchor:
            frame_record["candidate_anchors"].append(ground_anchor)

        overlay_path = overlay_dir / f"{frame_path.stem}_semantic_overlay.jpg"
        cv2.imwrite(str(overlay_path), overlay)
        frame_record["overlay"] = str(overlay_path)
        result["frames"].append(frame_record)
        print(f"segmented frame {row['index']} at {row['timestamp_sec']}s")

    out_path = out_dir / "scene_segmentation_context.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
