from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

bproc: Any = None
np: Any = None


def parse_args() -> argparse.Namespace:
    module_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Render RAM synthetic training scenes in BOP format with the bundled BlenderProc fork."
    )
    parser.add_argument(
        "--bop-parent-path",
        type=Path,
        default=module_dir / "bop_datasets",
        help="Parent directory containing the ram dataset folder.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="ram",
        help="BOP dataset name to load and write.",
    )
    parser.add_argument(
        "--cc-textures-path",
        type=Path,
        default=module_dir / "assets" / "cc_textures",
        help="Path to ambientCG/CC texture materials.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=module_dir / "outputs" / "training_scenes",
        help="Directory where bop_data/<dataset-name> will be written.",
    )
    parser.add_argument(
        "--object-ids",
        type=int,
        nargs="*",
        default=None,
        help="Object ids to sample. If omitted, all ids in models/model_meta.json are used.",
    )
    parser.add_argument("--num-scenes", type=int, default=400)
    parser.add_argument("--views-per-scene", type=int, default=25)
    parser.add_argument("--objects-per-scene", type=int, default=1)
    parser.add_argument("--frames-per-chunk", type=int, default=10000)
    parser.add_argument("--object-model-unit", choices=["m", "dm", "cm", "mm"], default="mm")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fx", type=float, default=572.411363389757)
    parser.add_argument("--fy", type=float, default=573.5704328585578)
    parser.add_argument("--cx", type=float, default=325.2611083984375)
    parser.add_argument("--cy", type=float, default=242.04899588216654)
    parser.add_argument("--depth-scale", type=float, default=0.1)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_object_ids(ram_dataset_path: Path, object_ids: list[int] | None) -> list[int]:
    if object_ids:
        return sorted(object_ids)

    meta_path = ram_dataset_path / "models" / "model_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"--object-ids was not provided and model metadata was not found: {meta_path}"
        )

    model_meta = load_json(meta_path)
    return sorted(int(object_id) for object_id in model_meta.keys())


def configure_camera(args: argparse.Namespace) -> None:
    k = np.array(
        [
            [args.fx, 0.0, args.cx],
            [0.0, args.fy, args.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    bproc.camera.set_intrinsics_from_K_matrix(k, args.width, args.height)


def create_room() -> list[Any]:
    return [
        bproc.object.create_primitive("PLANE", scale=[4, 4, 1]),
        bproc.object.create_primitive("PLANE", scale=[4, 4, 1], location=[0, -4, 4], rotation=[-1.570796, 0, 0]),
        bproc.object.create_primitive("PLANE", scale=[4, 4, 1], location=[0, 4, 4], rotation=[1.570796, 0, 0]),
        bproc.object.create_primitive("PLANE", scale=[4, 4, 1], location=[4, 0, 4], rotation=[0, -1.570796, 0]),
        bproc.object.create_primitive("PLANE", scale=[4, 4, 1], location=[-4, 0, 4], rotation=[0, 1.570796, 0]),
    ]


def create_lights() -> tuple[Any, Any, Any]:
    light_plane = bproc.object.create_primitive("PLANE", scale=[3, 3, 1], location=[0, 0, 10])
    light_plane.set_name("light_plane")
    light_plane_material = bproc.material.create("light_material")

    light_point = bproc.types.Light()
    light_point.set_energy(200)
    return light_plane, light_plane_material, light_point


def sample_pose_func(obj: Any) -> None:
    lower_bound = np.random.uniform([-0.3, -0.3, 0.0], [-0.2, -0.2, 0.0])
    upper_bound = np.random.uniform([0.2, 0.2, 0.4], [0.3, 0.3, 0.6])
    obj.set_location(np.random.uniform(lower_bound, upper_bound))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())


def sample_initial_pose(room_planes: list[Any]):
    def _sample_initial_pose(obj: Any) -> None:
        obj.set_location(
            bproc.sampler.upper_region(
                objects_to_sample_on=room_planes[0:1],
                min_height=1,
                max_height=4,
                face_sample_range=[0.4, 0.6],
            )
        )
        obj.set_rotation_euler(np.random.uniform([0, 0, 0], [0, 0, np.pi * 2]))

    return _sample_initial_pose


def randomize_object_material(obj: Any) -> None:
    mat = obj.get_materials()[0]
    mat.set_principled_shader_value("Base Color", np.random.uniform(0.1, 0.9, size=3).tolist() + [1])
    mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
    mat.set_principled_shader_value("Specular", np.random.uniform(0, 1.0))
    obj.hide(False)


def randomize_lighting(light_plane: Any, light_plane_material: Any, light_point: Any) -> None:
    light_plane_material.make_emissive(
        emission_strength=np.random.uniform(3, 6),
        emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]),
    )
    light_plane.replace_materials(light_plane_material)
    light_point.set_color(np.random.uniform([0.5, 0.5, 0.5], [1, 1, 1]))
    light_point.set_location(
        bproc.sampler.shell(
            center=[0, 0, 0],
            radius_min=1,
            radius_max=1.5,
            elevation_min=5,
            elevation_max=89,
        )
    )


def add_camera_poses(objects: list[Any], views_per_scene: int) -> None:
    bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(objects)

    cam_poses = 0
    while cam_poses < views_per_scene:
        location = bproc.sampler.shell(
            center=[0, 0, 0],
            radius_min=2.5,
            radius_max=3.5,
            elevation_min=25,
            elevation_max=55,
        )

        poi = bproc.object.compute_poi(np.random.choice(objects, size=1, replace=False))
        rotation_matrix = bproc.camera.rotation_from_forward_vec(
            poi - location,
            inplane_rot=np.random.uniform(-0.7854, 0.7854),
        )
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)

        if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.3}, bop_bvh_tree):
            bproc.camera.add_camera_pose(cam2world_matrix, frame=cam_poses)
            cam_poses += 1


def render_training_scenes(args: argparse.Namespace) -> None:
    global bproc, np
    import blenderproc as _bproc
    import numpy as _np

    bproc = _bproc
    np = _np
    np.random.seed(args.seed)
    ram_dataset_path = args.bop_parent_path / args.dataset_name
    object_ids = resolve_object_ids(ram_dataset_path, args.object_ids)

    bproc.init()
    configure_camera(args)

    target_bop_objs = bproc.loader.load_ram_objs(
        bop_dataset_path=str(ram_dataset_path),
        obj_ids=object_ids,
        object_model_unit=args.object_model_unit,
    )

    for obj in target_bop_objs:
        obj.set_shading_mode("auto")
        obj.hide(True)

    room_planes = create_room()
    light_plane, light_plane_material, light_point = create_lights()
    cc_textures = bproc.loader.load_ccmaterials(str(args.cc_textures_path))

    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.set_max_amount_of_samples(args.samples)

    for _ in range(args.num_scenes):
        sampled_target_bop_objs = list(
            np.random.choice(target_bop_objs, size=args.objects_per_scene, replace=False)
        )

        for obj in sampled_target_bop_objs:
            randomize_object_material(obj)

        randomize_lighting(light_plane, light_plane_material, light_point)

        random_cc_texture = np.random.choice(cc_textures)
        for plane in room_planes:
            plane.replace_materials(random_cc_texture)

        bproc.object.sample_poses(
            objects_to_sample=sampled_target_bop_objs,
            sample_pose_func=sample_pose_func,
            max_tries=1000,
        )

        bproc.object.sample_poses_on_surface(
            objects_to_sample=sampled_target_bop_objs,
            surface=room_planes[0],
            sample_pose_func=sample_initial_pose(room_planes),
            min_distance=0.01,
            max_distance=0.2,
        )

        add_camera_poses(sampled_target_bop_objs, args.views_per_scene)
        data = bproc.renderer.render()

        bproc.writer.write_bop(
            str(args.output_dir / "bop_data"),
            target_objects=sampled_target_bop_objs,
            dataset=args.dataset_name,
            depth_scale=args.depth_scale,
            depths=data["depth"],
            colors=data["colors"],
            color_file_format="JPEG",
            ignore_dist_thres=10,
            frames_per_chunk=args.frames_per_chunk,
        )

        for obj in sampled_target_bop_objs:
            obj.hide(True)


def main() -> None:
    render_training_scenes(parse_args())


if __name__ == "__main__":
    main()
