# Beach Matchmove CGI Pipeline

This repository documents a practical matchmove and compositing workflow for inserting CGI into a handheld beach video. The public repo does not include the raw or final video files; the rendered result and tracking overlay are hosted externally.

- Final result: https://www.youtube.com/shorts/hvko7LH6ucQ
- Tracking overlay: https://www.youtube.com/shorts/d1iTP9BZZ9M

## What This Contains

- `scripts/`: preprocessing, tracking, segmentation/depth extraction, RANSAC anchor tracking, Blender scene construction, transparent render, and final mux helpers.
- `assets/`: lightweight public assets used by the Blender scene.
- `report_assets/`: still images showing source-frame sampling, tracking overlays, segmentation/depth previews, RANSAC anchor tracking, and final composite frames.
- `docs/matchmove_report.html`: a self-contained process report with embedded result links and visuals.

## Pipeline Overview

1. Extract source metadata and representative frames.
2. Analyze global camera motion with OpenCV feature tracking.
3. Generate a track overlay video for QA.
4. Run semantic segmentation and relative depth extraction on sampled frames.
5. Attempt sparse SfM reconstruction for camera context.
6. Use RANSAC-filtered affine links to track a stable sand anchor.
7. Build a transparent Blender CGI layer with a zeppelin, flying letter, and seated beach figure.
8. Composite the Blender render over the source footage, add background music, and apply a gradual warm sunset grade.

## Key Implementation Notes

- The footage is a vertical 1080 x 1920, 30 fps beach pan.
- Full SfM was attempted, but the shot contains large sky/water/sand regions and moving people, so the final placement uses a robust 2.5D matchmove strategy.
- The seated figure is pinned to a RANSAC-propagated sand anchor during the stable part of the pan.
- The letter exits the visible frame before being hidden to avoid a final-frame pop.
- The sunset look is applied as a post-process grade so the real footage and CGI warm together.

## Reproducing The Workflow

Install common dependencies as needed:

```bash
python3 -m pip install opencv-python numpy pillow
```

Optional GPU/deeper analysis dependencies depend on your environment:

- PyTorch
- Transformers
- PyCOLMAP
- Blender 4.x
- FFmpeg

Typical local sequence:

```bash
python3 scripts/00_extract_media_context.py
python3 scripts/02_extract_camera_motion.py
python3 scripts/03_build_context_brief.py
python3 scripts/09_compute_ransac_anchor_tracks.py
blender --background --python scripts/07_create_cgi_blender_scene.py
blender --background context/blender_cgi_scene.blend --python scripts/08_render_cgi_preview_segment.py -- --start 1 --end 826 --step 1 --out //cgi_frames_full/frame_
```

The final FFmpeg composite command is environment/project specific because raw footage and music are intentionally not stored in this repository.

## Report

Open `docs/matchmove_report.html` locally to view the process report.

