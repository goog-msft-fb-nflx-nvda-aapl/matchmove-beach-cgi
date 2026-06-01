#!/usr/bin/env python3
"""Create a procedural Blender CGI scene for the matchmove shot.

Run with:
  /Applications/Blender.app/Contents/MacOS/Blender --background --python scripts/07_create_cgi_blender_scene.py

The scene is built as a transparent 2.5D CGI layer aligned to the video frame.
It is intentionally procedural so the project can move forward without external
assets.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import bpy
from mathutils import Vector


ROOT = Path(__file__).resolve().parents[1]
MEDIA_CONTEXT = ROOT / "context" / "media_context.json"
SEGMENTATION_CONTEXT = ROOT / "context" / "gpu_scene_segmentation" / "scene_segmentation_context.json"
RANSAC_TRACKS = ROOT / "context" / "ransac_anchor_tracks_preview.json"
SEATED_WOMAN_CUTOUT = ROOT / "assets" / "seated_woman_cutout.png"
OUT_BLEND = ROOT / "context" / "blender_cgi_scene.blend"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def make_mat(name: str, color: tuple[float, float, float, float], roughness: float = 0.55):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def make_image_material(name: str, image_path: Path):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    try:
        mat.blend_method = "BLEND"
        mat.use_screen_refraction = True
        mat.show_transparent_back = True
    except Exception:
        pass
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    image_node = nodes.new("ShaderNodeTexImage")
    image_node.image = bpy.data.images.load(str(image_path))
    image_node.extension = "CLIP"
    mat.node_tree.links.new(image_node.outputs["Color"], bsdf.inputs["Base Color"])
    mat.node_tree.links.new(image_node.outputs["Alpha"], bsdf.inputs["Alpha"])
    bsdf.inputs["Roughness"].default_value = 0.65
    return mat


def add_uv_sphere(name: str, location, scale, mat):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    obj.data.materials.append(mat)
    return obj


def add_cube(name: str, location, scale, mat):
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    obj.data.materials.append(mat)
    return obj


def add_image_plane_xz(name: str, location, height: float, image_path: Path):
    image = bpy.data.images.load(str(image_path))
    aspect = image.size[0] / image.size[1]
    width = height * aspect
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    # Plane lives in X/Z at constant Y, origin at bottom center for easy grounding.
    verts = [
        (-width / 2, 0, 0),
        (width / 2, 0, 0),
        (width / 2, 0, height),
        (-width / 2, 0, height),
    ]
    faces = [(0, 1, 2, 3)]
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    mesh.uv_layers.new(name="UVMap")
    uv = mesh.uv_layers.active.data
    for loop, co in zip(uv, [(0, 0), (1, 0), (1, 1), (0, 1)]):
        loop.uv = co
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    obj.data.materials.append(make_image_material(f"{name}_mat", image_path))
    return obj


def add_cylinder_between(name: str, a: Vector, b: Vector, radius: float, mat):
    mid = (a + b) / 2
    direction = b - a
    bpy.ops.mesh.primitive_cylinder_add(vertices=20, radius=radius, depth=direction.length, location=mid)
    obj = bpy.context.object
    obj.name = name
    obj.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()
    obj.data.materials.append(mat)
    return obj


def set_key(obj, frame: int, location=None, rotation=None, scale=None) -> None:
    if location is not None:
        obj.location = location
        obj.keyframe_insert(data_path="location", frame=frame)
    if rotation is not None:
        obj.rotation_euler = rotation
        obj.keyframe_insert(data_path="rotation_euler", frame=frame)
    if scale is not None:
        obj.scale = scale
        obj.keyframe_insert(data_path="scale", frame=frame)


def pixel_to_world(x: float, y: float, width: int, height: int, world_h: float) -> tuple[float, float]:
    world_w = world_h * width / height
    return ((x / width - 0.5) * world_w, (0.5 - y / height) * world_h)


def tracked_world_positions(width: int, height: int, world_h: float) -> dict[int, tuple[float, float]]:
    if not RANSAC_TRACKS.exists():
        return {}
    data = load_json(RANSAC_TRACKS)
    return {
        int(row["frame"]): pixel_to_world(float(row["source_x"]), float(row["source_y"]), width, height, world_h)
        for row in data.get("tracks", [])
    }


def lerp(a: float, b: float, t: float) -> float:
    return a * (1 - t) + b * t


def hide_obj(obj, frame: int, hidden: bool) -> None:
    obj.hide_viewport = hidden
    obj.hide_render = hidden
    obj.keyframe_insert(data_path="hide_viewport", frame=frame)
    obj.keyframe_insert(data_path="hide_render", frame=frame)


def make_walker(parent, mats):
    body = add_uv_sphere("beach_walker_body", (0, 0, 0.02), (0.16, 0.09, 0.30), mats["dress"])
    body.parent = parent
    skirt = add_uv_sphere("beach_walker_skirt", (0, 0, -0.20), (0.20, 0.10, 0.14), mats["dress_light"])
    skirt.parent = parent
    head = add_uv_sphere("beach_walker_head", (0, -0.01, 0.45), (0.105, 0.09, 0.115), mats["skin"])
    head.parent = parent
    hair = add_uv_sphere("beach_walker_hair", (0, -0.03, 0.49), (0.125, 0.105, 0.105), mats["hair"])
    hair.parent = parent
    ponytail = add_uv_sphere("beach_walker_ponytail", (-0.10, -0.045, 0.43), (0.07, 0.055, 0.12), mats["hair"])
    ponytail.parent = parent

    left_arm = add_cylinder_between(
        "beach_walker_left_arm",
        Vector((-0.10, -0.02, 0.23)),
        Vector((-0.12, -0.02, -0.08)),
        0.025,
        mats["skin"],
    )
    left_arm.parent = parent
    right_arm = add_cylinder_between(
        "beach_walker_right_arm",
        Vector((0.10, -0.02, 0.23)),
        Vector((0.12, -0.02, -0.08)),
        0.026,
        mats["skin"],
    )
    right_arm.parent = parent

    leg_l = add_cylinder_between(
        "beach_walker_left_leg",
        Vector((-0.06, 0, -0.29)),
        Vector((-0.13, 0, -0.62)),
        0.03,
        mats["skin"],
    )
    leg_l.parent = parent
    leg_r = add_cylinder_between(
        "beach_walker_right_leg",
        Vector((0.06, 0, -0.29)),
        Vector((0.13, 0, -0.62)),
        0.03,
        mats["skin"],
    )
    leg_r.parent = parent
    foot_l = add_cube("beach_walker_left_foot", (-0.14, -0.025, -0.64), (0.08, 0.035, 0.025), mats["sandals"])
    foot_l.parent = parent
    foot_r = add_cube("beach_walker_right_foot", (0.14, -0.025, -0.64), (0.08, 0.035, 0.025), mats["sandals"])
    foot_r.parent = parent

    return {
        "body": body,
        "skirt": skirt,
        "head": head,
        "hair": hair,
        "ponytail": ponytail,
        "right_arm": right_arm,
        "left_arm": left_arm,
        "left_leg": leg_l,
        "right_leg": leg_r,
        "left_foot": foot_l,
        "right_foot": foot_r,
    }


def make_zeppelin(parent, mats):
    hull = add_uv_sphere("zeppelin_hull", (0, 0, 0), (0.85, 0.16, 0.16), mats["airship"])
    hull.parent = parent
    gondola = add_cube("zeppelin_gondola", (0.03, -0.03, -0.20), (0.18, 0.05, 0.035), mats["airship_dark"])
    gondola.parent = parent
    for name, x, z, sx, sz in [
        ("tail_fin_top", -0.78, 0.18, 0.06, 0.13),
        ("tail_fin_bottom", -0.78, -0.18, 0.06, 0.13),
        ("tail_fin_left", -0.78, 0.0, 0.06, 0.03),
    ]:
        fin = add_cube(name, (x, 0, z), (sx, 0.02, sz), mats["airship_dark"])
        fin.parent = parent
    prop = add_cube("zeppelin_propeller", (-0.98, -0.015, 0), (0.018, 0.01, 0.16), mats["airship_dark"])
    prop.parent = parent
    return {"hull": hull, "propeller": prop}


def make_letter(parent, mats):
    letter = add_cube("flying_letter_envelope", (0, 0, 0), (0.20, 0.01, 0.13), mats["letter"])
    letter.parent = parent
    flap = add_cube("letter_flap", (0, -0.012, 0.035), (0.16, 0.008, 0.02), mats["letter_edge"])
    flap.rotation_euler[0] = math.radians(18)
    flap.parent = parent
    return {"letter": letter, "flap": flap}


def main() -> None:
    media = load_json(MEDIA_CONTEXT)
    seg = load_json(SEGMENTATION_CONTEXT)
    width = int(media["video"]["width"])
    height = int(media["video"]["height"])
    fps = round(float(media["video"]["fps"]))
    duration = float(media["video"]["duration_sec"])
    end_frame = int(round(duration * fps))
    world_h = 12.8
    world_w = world_h * width / height

    clear_scene()
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = end_frame
    scene.frame_set(1)
    scene.render.fps = fps
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.film_transparent = True
    scene.eevee.taa_render_samples = 64

    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = (0, -10, 0)
    camera.rotation_euler = (math.radians(90), 0, 0)
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = world_h
    scene.camera = camera

    sun_data = bpy.data.lights.new("warm_sun_from_camera_right", "SUN")
    sun = bpy.data.objects.new("warm_sun_from_camera_right", sun_data)
    bpy.context.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(48), 0, math.radians(-35))
    sun_data.energy = 3.0

    mats = {
        "skin": make_mat("warm_skin", (0.78, 0.50, 0.36, 1)),
        "hair": make_mat("dark_hair", (0.05, 0.035, 0.025, 1)),
        "dress": make_mat("sea_blue_dress", (0.05, 0.28, 0.52, 1)),
        "dress_light": make_mat("beach_skirt_light_blue", (0.12, 0.48, 0.78, 1)),
        "sandals": make_mat("dark_sandals", (0.08, 0.05, 0.035, 1)),
        "letter": make_mat("warm_white_letter", (0.95, 0.88, 0.68, 1)),
        "letter_edge": make_mat("gold_letter_edge", (0.95, 0.67, 0.18, 1)),
        "airship": make_mat("silver_airship", (0.78, 0.78, 0.74, 1)),
        "airship_dark": make_mat("airship_dark_details", (0.20, 0.22, 0.23, 1)),
        "shadow": make_mat("soft_shadow_proxy", (0.03, 0.025, 0.02, 0.35)),
    }

    # Character anchor: use segmentation frame around 14.9s and place on sand.
    anchor_row = min(seg["frames"], key=lambda r: abs(float(r["timestamp_sec"]) - 14.884))
    anchor = anchor_row["candidate_anchors"][0]
    anchor_x, anchor_z = pixel_to_world(anchor["x"] * width / 720.0, anchor["y"] * height / 1280.0, width, height, world_h)
    ransac_positions = tracked_world_positions(width, height, world_h)
    woman = add_image_plane_xz(
        "seated_woman_cutout",
        (anchor_x + 0.18, -0.16, anchor_z - 0.04),
        1.34,
        SEATED_WOMAN_CUTOUT,
    )
    # Do not use a procedural oval shadow under the image cutout. It reads as fake
    # on detailed sand. A contact shadow should be painted/composited later if needed.
    shadow = add_uv_sphere("woman_ground_shadow_disabled", (anchor_x + 0.18, 0.05, anchor_z - 0.08), (0, 0, 0), mats["shadow"])
    shadow.hide_viewport = True
    shadow.hide_render = True

    # No scale-pop and no sudden in-shot materialization: she is visible only
    # while the tracked sand patch is usable.
    hide_obj(woman, 1, True)
    hide_obj(woman, 299, True)
    hide_obj(woman, 300, False)
    hide_obj(woman, 540, False)
    hide_obj(woman, 541, True)
    hide_obj(woman, end_frame, True)
    hide_obj(shadow, 1, True)
    hide_obj(shadow, end_frame, True)

    # Add dense RANSAC-driven keys during the preview span so the seated figure
    # stays linked to the same sand patch while the camera pans.
    for frame, (base_x, base_z) in ransac_positions.items():
        if 300 <= frame <= 540:
            set_key(woman, frame, location=(base_x + 0.18, -0.16, base_z - 0.04))

    letter_root = bpy.data.objects.new("letter_root", None)
    bpy.context.collection.objects.link(letter_root)
    letter_parts = make_letter(letter_root, mats)
    for frame, loc, rot in [
        (1, (world_w * 0.65, -0.25, 2.3), (0, 0, 0)),
        (330, (world_w * 0.65, -0.25, 2.3), (0, 0, 0)),
        (430, (0.7, -0.22, -0.8), (math.radians(20), math.radians(0), math.radians(210))),
        (540, (0.15, -0.22, -1.10), (math.radians(-18), math.radians(0), math.radians(380))),
        (660, (-0.55, -0.22, -2.10), (math.radians(12), 0, math.radians(430))),
        (695, (-1.35, -0.22, -2.55), (math.radians(18), 0, math.radians(500))),
        (720, (-world_w * 0.95, -0.22, -2.35), (math.radians(22), 0, math.radians(555))),
        (end_frame, (-world_w * 1.05, -0.22, -2.20), (math.radians(22), 0, math.radians(600))),
    ]:
        set_key(letter_root, frame, location=loc, rotation=rot)
    for letter_obj in [letter_root, *letter_parts.values()]:
        hide_obj(letter_obj, 1, False)
        hide_obj(letter_obj, 719, False)
        hide_obj(letter_obj, 720, True)
        hide_obj(letter_obj, end_frame, True)

    zeppelin_root = bpy.data.objects.new("zeppelin_root", None)
    bpy.context.collection.objects.link(zeppelin_root)
    make_zeppelin(zeppelin_root, mats)
    # Airship enters from left and exits right with its own motion and scale change.
    # This avoids matching the camera pan speed, which makes it feel pasted to the frame.
    for frame, loc, rot in [
        (330, (-world_w * 0.95, 0.10, 3.18), (0, 0, math.radians(3))),
        (405, (-world_w * 0.35, 0.10, 2.92), (0, 0, math.radians(0))),
        (495, (world_w * 0.35, 0.10, 2.58), (0, 0, math.radians(-2))),
        (570, (world_w * 1.05, 0.10, 2.34), (0, 0, math.radians(-4))),
    ]:
        set_key(zeppelin_root, frame, location=loc, rotation=rot)
    for frame, scale in [
        (330, (0.34, 0.34, 0.34)),
        (405, (0.48, 0.48, 0.48)),
        (495, (0.67, 0.67, 0.67)),
        (570, (0.86, 0.86, 0.86)),
    ]:
        set_key(zeppelin_root, frame, scale=scale)
    hide_obj(zeppelin_root, 1, True)
    hide_obj(zeppelin_root, 329, True)
    hide_obj(zeppelin_root, 330, False)
    hide_obj(zeppelin_root, 570, False)
    hide_obj(zeppelin_root, 571, True)
    hide_obj(zeppelin_root, end_frame, True)
    for frame in range(330, 571, 12):
        zeppelin_root.rotation_euler[1] = math.radians(2.2 * math.sin(frame * 0.11))
        zeppelin_root.keyframe_insert(data_path="rotation_euler", frame=frame)

    note = bpy.data.texts.new("README_scene_notes")
    note.write(
        "Procedural CGI previsualization for video.mp4. Render with transparent film, then composite over original video. "
        "A seated beach figure is anchored to a tracked sand patch; a separate letter flies through; the generic Hindenburg-style zeppelin enters left and exits right."
    )

    bpy.ops.wm.save_as_mainfile(filepath=str(OUT_BLEND))
    print(f"Wrote {OUT_BLEND}")


if __name__ == "__main__":
    main()
