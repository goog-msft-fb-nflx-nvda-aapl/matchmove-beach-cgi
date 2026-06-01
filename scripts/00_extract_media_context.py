#!/usr/bin/env python3
"""Extract local video/audio context for matchmove planning."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, text=True, capture_output=True)


def ffprobe(path: Path) -> dict:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,bit_rate:stream=index,codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,sample_rate,channels",
            "-of",
            "json",
            str(path),
        ]
    )
    return json.loads(result.stdout)


def parse_rate(rate: str) -> float | None:
    if not rate or rate == "0/0":
        return None
    if "/" in rate:
        num, den = rate.split("/", 1)
        den_float = float(den)
        return float(num) / den_float if den_float else None
    return float(rate)


def video_summary(probe: dict) -> dict:
    streams = probe.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    duration = float(probe.get("format", {}).get("duration") or 0)
    fps = parse_rate(video_stream.get("avg_frame_rate") or "") or parse_rate(
        video_stream.get("r_frame_rate") or ""
    )
    return {
        "duration_sec": duration,
        "width": video_stream.get("width"),
        "height": video_stream.get("height"),
        "fps": fps,
        "video_codec": video_stream.get("codec_name"),
        "audio_stream_count": len(audio_streams),
        "streams": streams,
    }


def extract_even_frames(video: Path, out_dir: Path, duration: float, count: int) -> list[dict]:
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    if count <= 1:
        timestamps = [0.0]
    else:
        end = max(0.0, duration - 0.25)
        timestamps = [round(i * end / (count - 1), 3) for i in range(count)]

    frame_rows: list[dict] = []
    for idx, timestamp in enumerate(timestamps):
        frame_path = frames_dir / f"frame_{idx:03d}_{timestamp:07.3f}s.jpg"
        run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{timestamp:.3f}",
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-vf",
                "scale=720:-2",
                str(frame_path),
            ]
        )
        frame_rows.append({"index": idx, "timestamp_sec": timestamp, "path": str(frame_path)})
    return frame_rows


def extract_scene_frames(video: Path, out_dir: Path) -> list[str]:
    scene_dir = out_dir / "scene_frames"
    scene_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in scene_dir.glob("scene_*.jpg"):
        old_frame.unlink()
    for threshold in ("0.28", "0.18", "0.10"):
        pattern = scene_dir / "scene_%03d.jpg"
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(video),
                "-vf",
                f"select='gt(scene,{threshold})',scale=720:-2",
                "-fps_mode",
                "vfr",
                "-frames:v",
                "24",
                "-q:v",
                "2",
                str(pattern),
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        frames = sorted(scene_dir.glob("scene_*.jpg"))
        if result.returncode == 0 and frames:
            break
    return [str(path) for path in sorted(scene_dir.glob("scene_*.jpg"))]


def make_contact_sheet(frame_paths: list[str], out_dir: Path) -> Path | None:
    if not frame_paths:
        return None
    list_path = out_dir / "contact_sheet_inputs.txt"
    list_path.write_text(
        "".join(f"file '{Path(path).resolve()}'\n" for path in frame_paths),
        encoding="utf-8",
    )
    concat_path = out_dir / "contact_concat.mp4"
    sheet_path = out_dir / "contact_sheet.jpg"
    cols = 4
    rows = max(1, math.ceil(len(frame_paths) / cols))
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-vf",
            f"scale=360:-2,tile={cols}x{rows}",
            "-frames:v",
            "1",
            str(sheet_path),
        ]
    )
    concat_path.unlink(missing_ok=True)
    return sheet_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="video.mp4")
    parser.add_argument("--music", default="backgroup_music.mp3")
    parser.add_argument("--out", default="context")
    parser.add_argument("--frames", type=int, default=12)
    args = parser.parse_args()

    video = Path(args.video)
    music = Path(args.music)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise SystemExit(f"Missing required CLI tool: {tool}")

    video_probe = ffprobe(video)
    music_probe = ffprobe(music) if music.exists() else None
    summary = video_summary(video_probe)

    frames = extract_even_frames(video, out_dir, summary["duration_sec"], args.frames)
    scene_frames = extract_scene_frames(video, out_dir)
    contact_sheet = make_contact_sheet([row["path"] for row in frames], out_dir)

    manifest = {
        "source_video": str(video),
        "background_music": str(music) if music.exists() else None,
        "video": summary,
        "music_probe": music_probe,
        "sampled_frames": frames,
        "scene_change_frames": scene_frames,
        "contact_sheet": str(contact_sheet) if contact_sheet else None,
        "next_context_to_collect": [
            "A semantic description of the scene and safe CGI insertion points.",
            "Approximate ground plane / wall plane / occluding foreground objects.",
            "Camera motion type: pan, tilt, dolly, handheld translation, or mostly static.",
            "Lighting direction, shadow softness, and plausible CGI scale.",
        ],
    }
    manifest_path = out_dir / "media_context.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")
    print(f"Wrote {contact_sheet}")
    print(f"Sampled {len(frames)} frames and {len(scene_frames)} scene-change frames")


if __name__ == "__main__":
    main()
