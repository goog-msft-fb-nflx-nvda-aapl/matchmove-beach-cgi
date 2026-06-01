#!/usr/bin/env python3
"""Mux background music while preserving the original video duration."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, text=True, capture_output=True)


def duration(path: Path) -> float:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ]
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Rendered/composited video input.")
    parser.add_argument("--reference", default="video.mp4", help="Original video for duration.")
    parser.add_argument("--music", default="backgroup_music.mp3")
    parser.add_argument("--out", default="result.mp4")
    parser.add_argument("--music-volume", type=float, default=0.8)
    args = parser.parse_args()

    video = Path(args.video)
    reference = Path(args.reference)
    music = Path(args.music)
    out = Path(args.out)
    target_duration = duration(reference)

    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-stream_loop",
            "-1",
            "-i",
            str(music),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-t",
            f"{target_duration:.6f}",
            "-af",
            f"volume={args.music_volume},afade=t=out:st={max(0, target_duration - 1.5):.3f}:d=1.5",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(out),
        ]
    )
    actual_duration = duration(out)
    print(f"Wrote {out}")
    print(f"Reference duration: {target_duration:.6f}s")
    print(f"Output duration:    {actual_duration:.6f}s")


if __name__ == "__main__":
    main()
