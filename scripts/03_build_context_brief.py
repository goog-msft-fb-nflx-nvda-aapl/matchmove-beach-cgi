#!/usr/bin/env python3
"""Merge extracted context into one brief for planning CGI and GPU jobs."""

from __future__ import annotations

import json
from pathlib import Path


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def bullet(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "- None recorded"


def main() -> None:
    context_dir = Path("context")
    media = load_json(context_dir / "media_context.json")
    site = load_json(context_dir / "site_context.json")
    camera = load_json(context_dir / "camera_motion.json")
    semantic_files = sorted(Path(".").glob("semantic-video-description*.json"))
    semantic = load_json(semantic_files[-1]) if semantic_files else load_json(
        context_dir / "semantic_video_description.json"
    )

    video = media.get("video", {})
    motion = camera.get("motion_classification", {})
    quality = camera.get("tracking_quality", {})
    location = site.get("location", {})
    spatial = semantic.get("spatial_layout", {})
    lighting = semantic.get("lighting", {})
    trackable = semantic.get("trackable_features", {})

    brief = {
        "source_files": {
            "video": media.get("source_video", "video.mp4"),
            "music": media.get("background_music", "backgroup_music.mp3"),
            "semantic_context": str(semantic_files[-1]) if semantic_files else None,
            "media_context": "context/media_context.json",
            "camera_motion": "context/camera_motion.json",
            "track_overlay": camera.get("overlay_video"),
            "contact_sheet": media.get("contact_sheet"),
        },
        "site": location,
        "video": {
            "duration_sec": video.get("duration_sec"),
            "width": video.get("width"),
            "height": video.get("height"),
            "fps": video.get("fps"),
            "orientation": "vertical portrait",
        },
        "semantic_summary": semantic.get("scene_summary"),
        "temporal_summary": semantic.get("temporal_events", []),
        "camera_motion": {
            "video_understanding": semantic.get("camera_motion", {}),
            "opencv": motion,
            "tracking_quality": quality,
        },
        "spatial_layout": spatial,
        "lighting": lighting,
        "trackable_features": trackable,
        "cgi_recommendations": semantic.get("cgi_recommendations", []),
        "matchmove_risks": semantic.get("matchmove_risks", []),
        "recommended_next_step": {
            "name": "GPU semantic/spatial extraction",
            "why": "We need object/person/water/sky/sand masks and rough depth before deciding occlusion, shadows, and where CGI can be inserted.",
            "outputs": [
                "per-keyframe segmentation masks for sand, ocean, sky, people, scaffold/lumber/railing",
                "depth maps for selected frames",
                "candidate CGI anchor boxes/points on the sand plane",
            ],
        },
    }

    (context_dir / "context_brief.json").write_text(
        json.dumps(brief, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    md = f"""# Matchmove Context Brief

## Source
- Video: `{brief["source_files"]["video"]}`
- Music: `{brief["source_files"]["music"]}`
- Location: {location.get("place", "unknown")}
- Duration: {video.get("duration_sec"):.2f}s
- Resolution/FPS: {video.get("width")}x{video.get("height")} at {video.get("fps"):.2f} fps

## Semantic Summary
{semantic.get("scene_summary", "No semantic summary recorded.")}

## Camera Motion
- Video understanding: {semantic.get("camera_motion", {}).get("overall_motion", "unknown")}
- OpenCV: {motion.get("summary", "unknown")}
- Median motion per {camera.get("analysis_step_seconds", 0):.2f}s: dx={motion.get("median_dx_per_step")}, dy={motion.get("median_dy_per_step")}
- Median tracked points: {quality.get("median_tracked_points")}
- Median inliers: {quality.get("median_inliers")}

## Spatial Layout
- Ground plane: {spatial.get("ground_plane", "unknown")}
- Horizon line: {spatial.get("horizon_line", "unknown")}
- Vertical planes: {spatial.get("walls_or_vertical_planes", "unknown")}
- Foreground occluders: {spatial.get("foreground_occluders", "unknown")}
- Good anchor points: {spatial.get("good_anchor_points", "unknown")}

## Lighting
- Key light: {lighting.get("sun_or_key_light_direction", "unknown")}
- Shadow direction: {lighting.get("shadow_direction", "unknown")}
- Shadow softness: {lighting.get("shadow_softness", "unknown")}
- Color temperature: {lighting.get("color_temperature", "unknown")}

## Trackable Features
Good:
{bullet(trackable.get("good_features", []))}

Bad:
{bullet(trackable.get("bad_features", []))}

Recommended regions:
{bullet(trackable.get("recommended_tracking_regions", []))}

## CGI Direction
{bullet([item.get("idea", str(item)) for item in semantic.get("cgi_recommendations", [])])}

## Risks
{bullet(semantic.get("matchmove_risks", []))}

## Next Step
Run GPU semantic/spatial extraction: SAM2-style masks and depth maps on sampled frames.
"""
    (context_dir / "context_brief.md").write_text(md, encoding="utf-8")
    print("Wrote context/context_brief.json")
    print("Wrote context/context_brief.md")


if __name__ == "__main__":
    main()
