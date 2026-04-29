from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

DEFAULT_FX = 572.411363389757
DEFAULT_FY = 573.5704328585578
DEFAULT_CX = 325.2611083984375
DEFAULT_CY = 242.04899588216654
_NUMPY_LOADED = False


def ensure_numpy() -> None:
    global _NUMPY_LOADED
    global np

    if _NUMPY_LOADED:
        return

    try:
        import numpy as np_module
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Template rendering requires numpy. Install the project vision dependencies before running this script."
        ) from exc

    np = np_module
    _NUMPY_LOADED = True


def parse_args() -> argparse.Namespace:
    module_dir = Path(__file__).resolve().parent
    default_model_dir = module_dir / "examples" / "cad_models"

    parser = argparse.ArgumentParser(
        description="Render multi-view BOP template data for RAM with BlenderProc."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=default_model_dir,
        help="Directory containing obj_<id>.ply files and optional model_meta.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where bop_data/ram_template will be written.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="bowl",
        help="Template category name. Used as the BOP dataset output folder.",
    )
    parser.add_argument(
        "--object-id",
        type=int,
        default=None,
        help="CAD object id. If omitted, it is inferred from model_meta.json by category.",
    )
    parser.add_argument(
        "--num-views",
        type=int,
        default=128,
        help="Number of camera views to render for each object.",
    )
    parser.add_argument(
        "--camera-radius",
        type=float,
        default=5.0,
        help="Radius of the camera sphere around the object, in meters.",
    )
    parser.add_argument(
        "--object-model-unit",
        choices=["m", "dm", "cm", "mm"],
        default="mm",
        help="Unit of the CAD mesh coordinates. Original RAM templates used millimeter meshes.",
    )
    parser.add_argument("--width", type=int, default=640, help="Rendered image width.")
    parser.add_argument("--height", type=int, default=480, help="Rendered image height.")
    parser.add_argument("--fx", type=float, default=DEFAULT_FX)
    parser.add_argument("--fy", type=float, default=DEFAULT_FY)
    parser.add_argument("--cx", type=float, default=DEFAULT_CX)
    parser.add_argument("--cy", type=float, default=DEFAULT_CY)
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Maximum number of samples for BlenderProc color rendering.",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=0.1,
        help="BOP depth_scale written to camera and scene_camera JSON files.",
    )
    parser.add_argument(
        "--color-format",
        choices=["PNG", "JPEG"],
        default="JPEG",
        help="Color image format used by the BOP writer.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing BOP output folder instead of requiring a fresh dataset folder.",
    )
    parser.add_argument(
        "--skip-mask-info",
        action="store_true",
        help="Skip BOP mask, visibility mask, gt_info, and COCO annotation calculation.",
    )
    parser.add_argument("--seed", type=int, default=2, help="NumPy random seed.")
    return parser.parse_args()


def infer_object_id(model_dir: Path, category: str) -> int:
    meta_path = model_dir / "model_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"--object-id was not provided and model metadata was not found: {meta_path}"
        )

    with open(meta_path, "r", encoding="utf-8") as f:
        model_meta = json.load(f)

    for object_id, meta in model_meta.items():
        if meta.get("category_name") == category:
            return int(object_id)

    available = sorted(meta.get("category_name", str(k)) for k, meta in model_meta.items())
    raise KeyError(
        f"Category '{category}' was not found in {meta_path}. Available categories: {available}"
    )


def object_unit_scale(unit: str) -> float:
    return {
        "m": 1.0,
        "dm": 0.1,
        "cm": 0.01,
        "mm": 0.001,
    }[unit]


def sample_camera_locations(num_views: int) -> np.ndarray:
    ensure_numpy()

    indices = np.arange(num_views).astype(np.float32) + 0.5
    phi = math.pi / 2.0 - np.arccos(1.0 - indices / (2.0 * num_views))
    theta = math.pi * (1.0 + 5.0**0.5) * indices
    return np.stack(
        [
            np.cos(theta) * np.sin(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(phi),
        ],
        axis=1,
    )


def load_template_object(
    bproc: Any,
    model_path: Path,
    object_id: int,
    dataset_name: str,
    scale: float,
) -> Any:
    objects = bproc.loader.load_obj(str(model_path))
    if len(objects) != 1:
        raise RuntimeError(f"Expected one mesh in {model_path}, got {len(objects)}.")

    obj = objects[0]
    obj.set_name(f"obj_{object_id:06d}")
    obj.set_scale([scale, scale, scale])
    obj.set_cp("category_id", object_id)
    obj.set_cp("model_path", str(model_path))
    obj.set_cp("is_bop_object", True)
    obj.set_cp("bop_dataset_name", dataset_name)

    materials = obj.get_materials()
    if materials:
        materials[-1].set_name(f"bop_{dataset_name}_vertex_col_material")
    return obj


def configure_camera(bproc: Any, args: argparse.Namespace) -> None:
    ensure_numpy()

    K = np.array(
        [
            [args.fx, 0.0, args.cx],
            [0.0, args.fy, args.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    bproc.camera.set_intrinsics_from_K_matrix(K, args.width, args.height)


def create_scene(bproc: Any) -> tuple[list[Any], Any, Any, Any]:
    room_planes = [
        bproc.object.create_primitive("PLANE", scale=[6, 6, 1]),
        bproc.object.create_primitive("PLANE", scale=[6, 6, 1], location=[0, -6, 6], rotation=[-1.570796, 0, 0]),
        bproc.object.create_primitive("PLANE", scale=[6, 6, 1], location=[0, 6, 6], rotation=[1.570796, 0, 0]),
        bproc.object.create_primitive("PLANE", scale=[6, 6, 1], location=[6, 0, 6], rotation=[0, -1.570796, 0]),
        bproc.object.create_primitive("PLANE", scale=[6, 6, 1], location=[-6, 0, 6], rotation=[0, 1.570796, 0]),
    ]

    light_plane = bproc.object.create_primitive("PLANE", scale=[3, 3, 1], location=[0, 0, 10])
    light_plane.set_name("light_plane")
    light_plane_material = bproc.material.create("light_material")

    light_point = bproc.types.Light()
    light_point.set_energy(200)
    return room_planes, light_plane, light_plane_material, light_point


def set_template_material(obj: Any) -> None:
    materials = obj.get_materials()
    if not materials:
        return

    mat = materials[0]
    mat.set_principled_shader_value("Base Color", [0.5, 0.5, 0.5, 1.0])
    mat.set_principled_shader_value("Roughness", 0.5)
    mat.set_principled_shader_value("Specular", 0.1)


def add_camera_poses(bproc: Any, target_obj: Any, num_views: int, camera_radius: float) -> None:
    locations = sample_camera_locations(num_views)
    poi = bproc.object.compute_poi([target_obj])

    for frame_id, unit_location in enumerate(locations):
        location = camera_radius * unit_location
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=0.0)
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        bproc.camera.add_camera_pose(cam2world_matrix, frame=frame_id)


def render_template(args: argparse.Namespace) -> None:
    ensure_numpy()

    import blenderproc as bproc

    np.random.seed(args.seed)
    bproc.init()
    configure_camera(bproc, args)

    object_id = args.object_id or infer_object_id(args.model_dir, args.category)
    model_path = args.model_dir / f"obj_{object_id:06d}.ply"
    if not model_path.exists():
        raise FileNotFoundError(f"CAD model not found: {model_path}")

    obj = load_template_object(
        bproc=bproc,
        model_path=model_path,
        object_id=object_id,
        dataset_name=args.category,
        scale=object_unit_scale(args.object_model_unit),
    )
    obj.set_shading_mode("auto")
    obj.hide(True)

    _, light_plane, light_plane_material, light_point = create_scene(bproc)
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.set_max_amount_of_samples(args.samples)

    set_template_material(obj)
    obj.set_location([0.0, 0.0, 0.0])
    obj.set_rotation_euler([0.0, 0.0, 0.0])
    obj.hide(False)

    light_plane_material.make_emissive(
        emission_strength=np.random.uniform(2.0, 3.0),
        emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]),
    )
    light_plane.replace_materials(light_plane_material)
    light_point.set_color(np.random.uniform([0.5, 0.5, 0.5], [1.0, 1.0, 1.0]))
    light_point.set_location(
        bproc.sampler.shell(
            center=[0, 0, 0],
            radius_min=8,
            radius_max=10,
            elevation_min=85,
            elevation_max=89,
        )
    )

    add_camera_poses(bproc, obj, args.num_views, args.camera_radius)
    data = bproc.renderer.render()

    bproc.writer.write_bop(
        str(args.output_dir / "bop_data" / "ram_template"),
        target_objects=[obj],
        dataset=args.category,
        depth_scale=args.depth_scale,
        depths=data["depth"],
        colors=data["colors"],
        color_file_format=args.color_format,
        ignore_dist_thres=10,
        frames_per_chunk=args.num_views,
        append_to_existing_output=args.append,
        calc_mask_info_coco=not args.skip_mask_info,
    )


def main() -> None:
    args = parse_args()
    render_template(args)


if __name__ == "__main__":
    main()
