from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAM_DIR = PROJECT_ROOT / "visions" / "ram"
RAM_LIB_DIR = RAM_DIR / "lib"

FOUNDATION_MODEL = "dinov2-b14"
FEAT_LAYERS = [7, 9, 11]
FEAT_TYPE = "k"
PATCH_NUM = 224 // 14
IMAGE_SIZE = 224
FEATURE_DIM = len(FEAT_LAYERS) * 768
POSITION_ENCODING_DIM = 3 * 384
_DEPENDENCIES_LOADED = False


def ensure_runtime_dependencies() -> None:
    global _DEPENDENCIES_LOADED
    global cv2, np, torch, F, transforms, Image
    global vit_base

    if _DEPENDENCIES_LOADED:
        return

    try:
        import cv2 as cv2_module
        import numpy as np_module
        import torch as torch_module
        import torch.nn.functional as f_module
        import torchvision.transforms as transforms_module
        from PIL import Image as image_module
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Template foundation conversion requires opencv-python, numpy, torch, torchvision, and pillow. "
            "Install the project vision dependencies before running this script."
        ) from exc

    path_str = str(RAM_LIB_DIR)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

    from vision_transformer_dinov2 import vit_base as vit_base_func

    cv2 = cv2_module
    np = np_module
    torch = torch_module
    F = f_module
    transforms = transforms_module
    Image = image_module
    vit_base = vit_base_func
    _DEPENDENCIES_LOADED = True


def parse_args() -> argparse.Namespace:
    module_dir = Path(__file__).resolve().parent
    default_model_dir = module_dir / "examples" / "cad_models"

    parser = argparse.ArgumentParser(
        description="Convert BlenderProc BOP template renders into final RAM foundation template pkl files."
    )
    parser.add_argument(
        "--bop-root",
        type=Path,
        required=True,
        help="Path containing category folders rendered by render_templates.py, e.g. outputs/.../bop_data/ram_template.",
    )
    parser.add_argument(
        "--cad-model-dir",
        type=Path,
        default=default_model_dir,
        help="Directory containing obj_<id>.ply files.",
    )
    parser.add_argument(
        "--category",
        type=str,
        required=True,
        help="Template category name, matching the BOP category folder.",
    )
    parser.add_argument(
        "--object-id",
        type=int,
        default=None,
        help="BOP/CAD object id. If omitted, it is read from the first scene_gt.json entry.",
    )
    parser.add_argument(
        "--template-id",
        type=int,
        default=None,
        help="RAM template id used for the output asset folder and file name. Defaults to template.json, then object id.",
    )
    parser.add_argument(
        "--object-scale",
        type=float,
        default=None,
        help="Object scale used by RAM NOCS normalization. If omitted, it is read from template.json or model_meta.json.",
    )
    parser.add_argument(
        "--template-config",
        type=Path,
        default=PROJECT_ROOT / "visions" / "ram" / "template" / "template.json",
        help="RAM template.json used to infer template id and object_scale.",
    )
    parser.add_argument(
        "--model-meta",
        type=Path,
        default=None,
        help="Optional model_meta.json fallback for object_scale lookup.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "visions" / "ram" / "template" / "objects",
        help="Root where <category>/<template_id>/<template_id>.pkl will be written.",
    )
    parser.add_argument(
        "--dinov2-checkpoint",
        type=Path,
        required=True,
        help="DINOv2 ViT-B/14 pretrained checkpoint. This is separate from ram_model.pth.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n-pts", type=int, default=1024)
    parser.add_argument("--crop-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--num-views", type=int, default=None, help="Optional maximum number of views to process.")
    parser.add_argument(
        "--save-source",
        action="store_true",
        help="Also save the intermediate <template_id>_source.pkl for debugging.",
    )
    parser.add_argument(
        "--copy-mesh",
        action="store_true",
        help="Copy the CAD mesh into the RAM template object directory as obj_<template_id>.ply.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_intrinsic(camera_path: Path) -> tuple[np.ndarray, float]:
    params = load_json(camera_path)
    k = np.eye(3, dtype=np.float32)
    k[0, 0] = params["fx"]
    k[1, 1] = params["fy"]
    k[0, 2] = params["cx"]
    k[1, 2] = params["cy"]
    return k, float(params["depth_scale"])


def transform_coordinates_3d(coordinates: np.ndarray, srt: np.ndarray) -> np.ndarray:
    if coordinates.shape[0] != 3:
        raise ValueError(f"Expected coordinates with shape [3, N], got {coordinates.shape}")
    coordinates = np.vstack([coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)])
    transformed = srt @ coordinates
    return transformed[:3, :] / transformed[3, :]


def get_bbox(points: np.ndarray, img_w: int = 640, img_h: int = 480) -> tuple[int, int, int, int]:
    y1, x1, y2, x2 = points
    window_size = (max(y2 - y1, x2 - x1) // 40 + 1) * 40
    window_size = min(window_size + 20, min(img_h, img_w))
    center = [(y1 + y2) // 2, (x1 + x2) // 2]
    rmin = center[0] - int(window_size / 2)
    rmax = center[0] + int(window_size / 2)
    cmin = center[1] - int(window_size / 2)
    cmax = center[1] + int(window_size / 2)

    if rmin < 0:
        delta = -rmin
        rmin = 0
        rmax += delta
    if cmin < 0:
        delta = -cmin
        cmin = 0
        cmax += delta
    if rmax > img_h:
        delta = rmax - img_h
        rmax = img_h
        rmin -= delta
    if cmax > img_w:
        delta = cmax - img_w
        cmax = img_w
        cmin -= delta

    return int(rmin), int(rmax), int(cmin), int(cmax)


def generate_nocs_map(
    rgb: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    pose: np.ndarray,
    k: np.ndarray,
    depth_scale: float,
    object_scale: float,
    cad_min: np.ndarray,
    cad_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    h, w, _ = rgb.shape
    cad_min = cad_min / object_scale
    cad_max = cad_max / object_scale
    xmin, ymin, zmin = cad_min
    xmax, ymax, zmax = cad_max

    xmap = np.array([[i for i in range(w)] for _ in range(h)])
    ymap = np.array([[j for _ in range(w)] for j in range(h)])

    choose = mask.flatten().nonzero()[0]
    if choose.size == 0:
        raise ValueError("Template mask is empty; cannot generate NOCS map.")

    depth_masked = depth.flatten()[choose][:, np.newaxis]
    xmap_masked = xmap.flatten()[choose][:, np.newaxis]
    ymap_masked = ymap.flatten()[choose][:, np.newaxis]
    pt2 = depth_masked * depth_scale
    pt0 = (xmap_masked - k[0, 2]) * pt2 / k[0, 0]
    pt1 = (ymap_masked - k[1, 2]) * pt2 / k[1, 1]
    points = np.concatenate((pt0, pt1, pt2), axis=1)

    nocs_points = transform_coordinates_3d(points.T, np.linalg.inv(pose)).T
    nocs_points = nocs_points / object_scale

    in_range = (
        (nocs_points[:, 0] >= (xmin - 0.05))
        & (nocs_points[:, 0] <= (xmax + 0.05))
        & (nocs_points[:, 1] >= (ymin - 0.05))
        & (nocs_points[:, 1] <= (ymax + 0.05))
        & (nocs_points[:, 2] >= (zmin - 0.05))
        & (nocs_points[:, 2] <= (zmax + 0.05))
    )
    filtered_points = nocs_points[in_range]
    filtered_choose = choose[in_range]
    if filtered_choose.size == 0:
        raise ValueError("All template mask points were filtered out during NOCS generation.")

    clean_mask = np.zeros(mask.shape, dtype=np.uint8).flatten()
    clean_mask[filtered_choose] = 255
    clean_mask = clean_mask.reshape(h, w)

    new_xmap_masked = xmap.flatten()[filtered_choose][:, np.newaxis]
    new_ymap_masked = ymap.flatten()[filtered_choose][:, np.newaxis]

    nocs_map = np.ones(rgb.shape, dtype=np.uint8) * 255
    nocs_map[new_ymap_masked.flatten(), new_xmap_masked.flatten(), 0] = (
        (filtered_points[:, 2] + 0.5) * 255
    ).astype(np.uint8)
    nocs_map[new_ymap_masked.flatten(), new_xmap_masked.flatten(), 1] = (
        (filtered_points[:, 1] + 0.5) * 255
    ).astype(np.uint8)
    nocs_map[new_ymap_masked.flatten(), new_xmap_masked.flatten(), 2] = (
        (filtered_points[:, 0] + 0.5) * 255
    ).astype(np.uint8)

    return nocs_map, clean_mask


def crop_template_image(
    mask: np.ndarray,
    rgb: np.ndarray,
    depth: np.ndarray,
    nocs: np.ndarray,
    k: np.ndarray,
    crop_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w, _ = rgb.shape
    ys, xs = np.nonzero(mask)
    if ys.size == 0 or xs.size == 0:
        raise ValueError("Clean template mask is empty; cannot crop template image.")

    y1, y2 = int(np.min(ys)), int(np.max(ys))
    x1, x2 = int(np.min(xs)), int(np.max(xs))
    rmin, rmax, cmin, cmax = get_bbox([y1, x1, y2, x2], img_w=w, img_h=h)
    if (rmax - rmin) != (cmax - cmin):
        raise ValueError("Template crop window is not square.")

    center_x = float(cmin + cmax) / 2.0
    center_y = float(rmin + rmax) / 2.0
    crop_scale = float(crop_size) / float(cmax - cmin)

    crop_rgb = rgb[rmin:rmax, cmin:cmax, :].copy()
    crop_depth = depth[rmin:rmax, cmin:cmax].copy()
    crop_mask = mask[rmin:rmax, cmin:cmax].copy()
    crop_nocs = nocs[rmin:rmax, cmin:cmax, :].copy()

    crop_rgb = cv2.resize(crop_rgb, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
    crop_depth = cv2.resize(crop_depth, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)
    crop_mask = cv2.resize(crop_mask, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)
    crop_nocs = cv2.resize(crop_nocs, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)

    crop_k = k.copy()
    crop_k[0, 2] = k[0, 2] + float(crop_size - 1) / 2.0 - center_x
    crop_k[1, 2] = k[1, 2] + float(crop_size - 1) / 2.0 - center_y
    crop_k[0, 0] = k[0, 0] * crop_scale
    crop_k[1, 1] = k[1, 1] * crop_scale

    return crop_rgb, crop_mask, crop_depth, crop_nocs, crop_k


class Dinov2B14FeatureExtractor:
    def __init__(self, checkpoint_path: Path, device_name: str):
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"DINOv2-B/14 checkpoint not found: {checkpoint_path}. "
                "This is the DINOv2 pretrained checkpoint, not visions/ram/checkpoints/ram_model.pth."
            )

        self.device = torch.device(device_name if torch.cuda.is_available() or not device_name.startswith("cuda") else "cpu")
        self.model = vit_base(patch_size=14, img_size=518, block_chunks=0, init_values=1e-5)
        state_dict = self._load_checkpoint_state_dict(checkpoint_path)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    @staticmethod
    def _load_checkpoint_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Unsupported DINOv2 checkpoint format: {checkpoint_path}")

        for key in ("state_dict", "model", "teacher"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

        cleaned_state_dict: dict[str, torch.Tensor] = {}
        for key, value in checkpoint.items():
            if not torch.is_tensor(value):
                raise TypeError(
                    f"Unsupported DINOv2 checkpoint entry '{key}' in {checkpoint_path}; expected tensor weights."
                )

            cleaned_key = key
            for prefix in ("module.", "teacher.", "student.", "backbone."):
                if cleaned_key.startswith(prefix):
                    cleaned_key = cleaned_key[len(prefix):]
            cleaned_state_dict[cleaned_key] = value

        return cleaned_state_dict

    def extract(self, rgb: np.ndarray) -> dict[int, dict[str, np.ndarray]]:
        image = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.get_specific_tokens(image, layers_to_return=FEAT_LAYERS)

        saved_features: dict[int, dict[str, np.ndarray]] = {}
        for layer, layer_features in features.items():
            saved_features[layer] = {}
            for token_type, token_value in layer_features.items():
                if token_type != "attn":
                    saved_features[layer][token_type] = token_value.detach().cpu().numpy().astype(np.float16)
        return saved_features


def resolve_object_scale(args: argparse.Namespace, object_id: int) -> float:
    if args.object_scale is not None:
        return float(args.object_scale)

    if args.template_config.exists():
        template_config = load_json(args.template_config)
        category_info = template_config.get(args.category)
        if category_info and "object_scale" in category_info:
            return float(category_info["object_scale"])

    model_meta_path = args.model_meta
    if model_meta_path is None:
        candidate = args.cad_model_dir / "model_meta.json"
        model_meta_path = candidate if candidate.exists() else None

    if model_meta_path is not None and model_meta_path.exists():
        model_meta = load_json(model_meta_path)
        if str(object_id) in model_meta and "object_scale" in model_meta[str(object_id)]:
            return float(model_meta[str(object_id)]["object_scale"])

    raise ValueError(
        "Object scale is required. Pass --object-scale, add the category to template.json, "
        "or provide model_meta.json with object_scale."
    )


def resolve_template_id(args: argparse.Namespace, object_id: int) -> int:
    if args.template_id is not None:
        return int(args.template_id)

    if args.template_config.exists():
        template_config = load_json(args.template_config)
        category_info = template_config.get(args.category)
        if category_info and "template_id" in category_info:
            return int(category_info["template_id"])

    return int(object_id)


def infer_object_id_from_scene(scene_gt: dict[str, Any]) -> int:
    first_frame_key = sorted(scene_gt.keys(), key=lambda x: int(x))[0]
    return int(scene_gt[first_frame_key][0]["obj_id"])


def read_rgb(path: Path) -> np.ndarray:
    rgb_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise FileNotFoundError(f"Could not read RGB image: {path}")
    return rgb_bgr[:, :, ::-1]


def read_image(path: Path, name: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read {name}: {path}")
    return image


def mesh_bounds(model_path: Path) -> tuple[np.ndarray, np.ndarray]:
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(model_path))
    if len(mesh.vertices) == 0:
        raise ValueError(f"CAD mesh is empty or unreadable: {model_path}")

    points = mesh.sample_points_uniformly(1024)
    points_np = np.asarray(points.points, dtype=np.float32)
    return np.min(points_np, axis=0), np.max(points_np, axis=0)


def reshape_dinov2_b14_feats(
    feats: dict[int, dict[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    feat_maps_with_class_token = []
    feat_maps_no_class_token = []

    for layer in FEAT_LAYERS:
        if layer not in feats or FEAT_TYPE not in feats[layer]:
            raise KeyError(f"Missing DINOv2-B/14 feature layer {layer} token '{FEAT_TYPE}'.")

        token = feats[layer][FEAT_TYPE]
        batch_size, heads, token_count, dim_per_head = token.shape
        expected_token_count = PATCH_NUM * PATCH_NUM + 1
        if token_count != expected_token_count:
            raise ValueError(
                f"Unexpected DINOv2 token count for layer {layer}: {token_count}. "
                f"Expected {expected_token_count} for a {PATCH_NUM}x{PATCH_NUM} patch grid plus class token."
            )

        token = token.transpose(0, 1, 3, 2)
        class_token = token[:, :, :, 0].reshape(batch_size, heads * dim_per_head, 1).astype(np.float32)
        patch_token = token[:, :, :, 1:].reshape(batch_size, heads * dim_per_head, -1).astype(np.float32)
        class_token = np.tile(class_token, (1, 1, patch_token.shape[-1]))

        feat_maps_with_class_token.append(
            (patch_token + class_token).reshape(batch_size, -1, PATCH_NUM, PATCH_NUM)
        )
        feat_maps_no_class_token.append(patch_token.reshape(batch_size, -1, PATCH_NUM, PATCH_NUM))

    return (
        np.concatenate(feat_maps_with_class_token, axis=1),
        np.concatenate(feat_maps_no_class_token, axis=1),
    )


def position_encoding(emb_dim: int, max_length: int = 60000) -> np.ndarray:
    position = np.arange(max_length)[:, np.newaxis]
    positional_encoding = np.zeros((1, max_length, emb_dim), dtype=np.float32)
    two_i = np.arange(0, emb_dim, step=2).astype(np.float32)
    positional_encoding[0, :, 0::2] = np.sin(position / (10000 ** (two_i / emb_dim)))
    positional_encoding[0, :, 1::2] = np.cos(position / (10000 ** (two_i / emb_dim)))
    return positional_encoding[0]


def sample_support_information(
    mask: np.ndarray,
    feat_map_with_class_token: np.ndarray,
    feat_map_no_class_token: np.ndarray,
    nocs_map: np.ndarray,
    depth: np.ndarray,
    k: np.ndarray,
    depth_scale: float = 0.1,
    n_pts: int = 1024,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = mask.shape
    choose = mask.flatten().nonzero()[0]
    if len(choose) == 0:
        raise ValueError("Template mask is empty; cannot sample support information.")

    if len(choose) > n_pts:
        choice_mask = np.zeros(len(choose), dtype=int)
        choice_mask[:n_pts] = 1
        np.random.shuffle(choice_mask)
        choose = choose[choice_mask.nonzero()]
    else:
        choose = np.pad(choose, (0, n_pts - len(choose)), "wrap")

    feat_map_with_class_token = F.interpolate(
        torch.from_numpy(feat_map_with_class_token),
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    )
    _, dim, _, _ = feat_map_with_class_token.shape
    feat_map_with_class_token = feat_map_with_class_token.squeeze().permute(1, 2, 0).numpy()
    feat_with_class_token = feat_map_with_class_token.reshape((-1, dim))[choose, :]

    feat_map_no_class_token = F.interpolate(
        torch.from_numpy(feat_map_no_class_token),
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    )
    _, dim, _, _ = feat_map_no_class_token.shape
    feat_map_no_class_token = feat_map_no_class_token.squeeze().permute(1, 2, 0).numpy()
    feat_no_class_token = feat_map_no_class_token.reshape((-1, dim))[choose, :]

    coord = nocs_map.astype(np.float32) / 255
    nocs = coord.reshape((-1, 3))[choose, :] - 0.5

    xmap = np.array([[i for i in range(w)] for _ in range(h)])
    ymap = np.array([[j for _ in range(w)] for j in range(h)])
    depth_masked = depth.flatten()[choose][:, np.newaxis]
    xmap_masked = xmap.flatten()[choose][:, np.newaxis]
    ymap_masked = ymap.flatten()[choose][:, np.newaxis]
    pt2 = depth_masked * depth_scale
    pt0 = (xmap_masked - k[0, 2]) * pt2 / k[0, 0]
    pt1 = (ymap_masked - k[1, 2]) * pt2 / k[1, 1]
    points = np.concatenate((pt0, pt1, pt2), axis=1)
    pixels = np.concatenate((xmap_masked, ymap_masked), axis=1)

    return feat_with_class_token, feat_no_class_token, nocs, points, choose, pixels


def build_final_template(source_template: list[dict[str, Any]], n_pts: int) -> dict[str, Any]:
    pe_table = position_encoding(POSITION_ENCODING_DIM)
    nocs_maps = []
    feat_vectors_with_class_token = []
    feat_vectors_no_class_token = []
    nocs_vectors = []
    template_points = []
    template_points_2d = []
    poses = []
    rgbs = []
    masks = []
    position_encodings = []
    chooses = []

    for template_entry in source_template:
        rgbs.append(template_entry["rgb"])
        masks.append(template_entry["mask"])
        poses.append(template_entry["object_pose"])

        feat_with_class_token, feat_no_class_token = reshape_dinov2_b14_feats(template_entry["dinov2_b14_feat"])
        feat_vec_with_class_token, feat_vec_no_class_token, nocs_vec, pts, choose, pts2d = sample_support_information(
            template_entry["mask"],
            feat_with_class_token,
            feat_no_class_token,
            template_entry["nocs_map"][:, :, (2, 1, 0)],
            template_entry["depth"],
            template_entry["K"],
            0.1,
            n_pts,
        )

        chooses.append(choose)
        feat_vectors_with_class_token.append(feat_vec_with_class_token[np.newaxis, :, :])
        feat_vectors_no_class_token.append(feat_vec_no_class_token[np.newaxis, :, :])
        nocs_vectors.append(nocs_vec[np.newaxis, :, :])
        template_points.append(pts[np.newaxis, :, :])
        template_points_2d.append(pts2d[np.newaxis, :, :])
        position_encodings.append(pe_table[choose, :][np.newaxis, :, :])

        nocs_map = template_entry["nocs_map"][:, :, (2, 1, 0)]
        nocs_maps.append(nocs_map[np.newaxis, :, :, :])

    return {
        "nocs": np.concatenate(nocs_maps, axis=0),
        "feat_vec_with_class_token": np.concatenate(feat_vectors_with_class_token, axis=0).astype(np.float16),
        "feat_vec_no_class_token": np.concatenate(feat_vectors_no_class_token, axis=0).astype(np.float16),
        "nocs_vec": np.concatenate(nocs_vectors, axis=0),
        "pts": np.concatenate(template_points, axis=0),
        "pts2d": np.concatenate(template_points_2d, axis=0),
        "rgb": rgbs,
        "mask": masks,
        "reference_poses": poses,
        "position_encoding": np.concatenate(position_encodings, axis=0),
        "sample_choose": chooses,
    }


def process_scene_chunk(
    chunk_dir: Path,
    k: np.ndarray,
    depth_scale: float,
    object_scale: float,
    cad_min: np.ndarray,
    cad_max: np.ndarray,
    extractor: Dinov2B14FeatureExtractor,
    crop_size: int,
    max_views: int | None,
) -> tuple[list[dict[str, Any]], int]:
    scene_gt = load_json(chunk_dir / "scene_gt.json")
    object_id = infer_object_id_from_scene(scene_gt)
    rgb_dir = chunk_dir / "rgb"
    depth_dir = chunk_dir / "depth"
    mask_dir = chunk_dir / "mask"

    template_entries: list[dict[str, Any]] = []
    frame_keys = sorted(scene_gt.keys(), key=lambda x: int(x))
    if max_views is not None:
        frame_keys = frame_keys[:max_views]

    for frame_key in frame_keys:
        frame_id = int(frame_key)
        rgb_path = rgb_dir / f"{frame_id:06d}.jpg"
        if not rgb_path.exists():
            rgb_path = rgb_dir / f"{frame_id:06d}.png"
        depth_path = depth_dir / f"{frame_id:06d}.png"
        mask_path = mask_dir / f"{frame_id:06d}_000000.png"

        rgb = read_rgb(rgb_path)
        depth = read_image(depth_path, "depth")
        mask = read_image(mask_path, "mask")

        gt = scene_gt[frame_key][0]
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = np.array(gt["cam_R_m2c"], dtype=np.float32).reshape(3, 3)
        pose[:3, 3] = np.array(gt["cam_t_m2c"], dtype=np.float32).flatten()

        nocs_map, clean_mask = generate_nocs_map(
            rgb=rgb,
            depth=depth,
            mask=mask,
            pose=pose,
            k=k,
            depth_scale=depth_scale,
            object_scale=object_scale,
            cad_min=cad_min,
            cad_max=cad_max,
        )
        crop_rgb, crop_mask, crop_depth, crop_nocs, crop_k = crop_template_image(
            mask=clean_mask,
            rgb=rgb,
            depth=depth,
            nocs=nocs_map,
            k=k,
            crop_size=crop_size,
        )
        features = extractor.extract(crop_rgb)

        template_entries.append(
            {
                "rgb": crop_rgb,
                "mask": crop_mask,
                "depth": crop_depth,
                "object_pose": pose,
                "K": crop_k,
                "nocs_map": crop_nocs,
                "object_scale": object_scale,
                "dinov2_b14_feat": features,
            }
        )

    return template_entries, object_id


def process_bop_templates(args: argparse.Namespace) -> Path:
    ensure_runtime_dependencies()

    category_root = args.bop_root / args.category
    camera_path = category_root / "camera.json"
    train_pbr_dir = category_root / "train_pbr"
    if not train_pbr_dir.exists():
        raise FileNotFoundError(f"BOP train_pbr directory not found: {train_pbr_dir}")

    k, depth_scale = load_intrinsic(camera_path)
    chunk_dirs = sorted(path for path in train_pbr_dir.iterdir() if path.is_dir())
    if not chunk_dirs:
        raise FileNotFoundError(f"No BOP scene chunks found under: {train_pbr_dir}")

    first_scene_gt = load_json(chunk_dirs[0] / "scene_gt.json")
    object_id = args.object_id if args.object_id is not None else infer_object_id_from_scene(first_scene_gt)
    template_id = resolve_template_id(args, object_id)
    object_scale = resolve_object_scale(args, object_id)

    model_path = args.cad_model_dir / f"obj_{object_id:06d}.ply"
    if not model_path.exists():
        raise FileNotFoundError(f"CAD model not found: {model_path}")

    cad_min, cad_max = mesh_bounds(model_path)
    extractor = Dinov2B14FeatureExtractor(args.dinov2_checkpoint, args.device)

    source_template: list[dict[str, Any]] = []
    for chunk_dir in chunk_dirs:
        entries, chunk_object_id = process_scene_chunk(
            chunk_dir=chunk_dir,
            k=k,
            depth_scale=depth_scale,
            object_scale=object_scale,
            cad_min=cad_min,
            cad_max=cad_max,
            extractor=extractor,
            crop_size=args.crop_size,
            max_views=args.num_views,
        )
        if chunk_object_id != object_id:
            raise ValueError(
                f"Object id mismatch. Expected {object_id}, but {chunk_dir / 'scene_gt.json'} contains {chunk_object_id}."
            )
        source_template.extend(entries)
        if args.num_views is not None and len(source_template) >= args.num_views:
            source_template = source_template[: args.num_views]
            break

    if not source_template:
        raise RuntimeError("No template views were processed.")

    final_template = build_final_template(source_template, args.n_pts)

    object_dir = args.output_root / args.category / f"{template_id:06d}"
    object_dir.mkdir(parents=True, exist_ok=True)
    final_path = object_dir / f"{template_id:06d}.pkl"
    with open(final_path, "wb") as f:
        pickle.dump(final_template, f)

    if args.save_source:
        source_path = object_dir / f"{template_id:06d}_source.pkl"
        with open(source_path, "wb") as f:
            pickle.dump(source_template, f)

    if args.copy_mesh:
        shutil.copy2(model_path, object_dir / f"obj_{template_id:06d}.ply")

    return final_path


def main() -> None:
    final_path = process_bop_templates(parse_args())
    print(f"Saved RAM foundation template to {final_path}")


if __name__ == "__main__":
    main()
