#!/usr/bin/env python3
"""Render a quick transparent CGI preview segment from the Blender scene."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=300)
    parser.add_argument("--end", type=int, default=540)
    parser.add_argument("--step", type=int, default=3)
    parser.add_argument("--out", default="//preview_cgi_frames/frame_")
    argv = sys.argv
    args = parser.parse_args(argv[argv.index("--") + 1 :] if "--" in argv else [])

    scene = bpy.context.scene
    scene.frame_start = args.start
    scene.frame_end = args.end
    scene.frame_step = args.step
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.filepath = args.out

    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = 16

    Path(bpy.path.abspath(args.out)).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
