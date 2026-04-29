import sys
import open3d as o3d
import cv2
import time
import os 
import numpy as np
import torch  
from os.path import join as pjoin
import json
import argparse
import re
import logging
from contextlib import nullcontext
from easydict import EasyDict

sys.stdout.reconfigure(encoding='utf-8')
class ColoredFormatter(logging.Formatter):
    COLORS = {
        'INFO': '\033[94m',
        'DEBUG': '\033[92m',
        'WARNING': '\033[93m',
        'ERROR': '\033[91m',
        'CRITICAL': '\033[95m',
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.msg = f"{color}{record.msg}{self.RESET}"
        return super().format(record)

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger()
logger.handlers[0].setFormatter(ColoredFormatter('%(levelname)s: %(message)s'))

from controller.extern_camera import ZEDCamera, RealSense455Camera
from utils.image_utils import save_rgb_png, save_depth_npy, turbo_cmap

from visions.ram.ram import RAMIO, RAMTemplate, RAMModel
from visions.vggt.vggt.models.vggt import VGGT
from visions.vggt.vggt.utils.load_fn import load_and_preprocess_images
from visions.vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from utils.config import get_ram_config

def normal_to_rotmat(n):
    n = n / np.linalg.norm(n)
    if abs(n[2]) < 0.99:
        a = np.array([0, 0, 1])
    else:
        a = np.array([0, 1, 0])
    x = np.cross(a, n)
    x = x / np.linalg.norm(x)
    y = np.cross(n, x)
    R = np.stack([x, y, n], axis=1)
    return R

def euler_to_matrix(roll, pitch, yaw):
    rx, ry, rz = np.radians([roll, pitch, yaw])
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx),  np.cos(rx)]
    ])
    Ry = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz),  np.cos(rz), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx

def get_sym_equivalent_pose(T_target, T_init):
    R_sym = np.array([
        [-1, 0, 0],
        [ 0,-1, 0],
        [ 0, 0, 1]
    ])
    T_sym = np.eye(4)
    T_sym[:3,:3] = R_sym

    T_target_sym = np.dot(T_target, T_sym)

    def pose_distance(T1, T2, rot_weight=1.0, trans_weight=0.0):
        d_trans = np.linalg.norm(T1[:3,3] - T2[:3,3])
        R1 = T1[:3,:3]
        R2 = T2[:3,:3]
        R_rel = np.dot(R1.T, R2)
        trace = np.clip(np.trace(R_rel), -1.0, 3.0)
        theta = np.arccos(np.clip((trace - 1)/2, -1.0, 1.0))
        return rot_weight * theta + trans_weight * d_trans

    dist0 = pose_distance(T_init, T_target)
    dist1 = pose_distance(T_init, T_target_sym)

    return T_target if dist0 <= dist1 else T_target_sym

def get_bbox_plane_poses(size, sRT):
    pred_scale = np.cbrt(np.linalg.det(sRT[:3, :3]))

    x, y, z = size
    centers = np.array([
        [ +x/2,   0,    0],
        [ -x/2,   0,    0],
        [   0,  +y/2,   0],
        [   0,  -y/2,   0],
        [   0,    0,  +z/2],
        [   0,    0,  -z/2],
    ])
    normals = np.array([
        [ +1,  0,  0],
        [ -1,  0,  0],
        [  0, +1,  0],
        [  0, -1,  0],
        [  0,  0, +1],
        [  0,  0, -1],
    ])
    planes_poses = []
    for i in range(6):
        R = normal_to_rotmat(normals[i])
        T = np.eye(4)
        T[:3,:3] = R
        T[:3, 3] = centers[i]
        T_cam = sRT @ T
        T_cam[:3, :3] = T_cam[:3, :3] / pred_scale
        planes_poses.append(T_cam)
    return np.stack(planes_poses)

def parse_args():
    parser = argparse.ArgumentParser(description="parse constraints from VLM subtasks")
    parser.add_argument("--prompt", type=str, default="clean the table", help="Text prompt describing the task")
    parser.add_argument("--output", type=str, default="./output/scene_01", help="Directory to save outputs")
    return parser.parse_args()


def resolve_torch_device(device_name):
    if not device_name:
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"

    requested_device = torch.device(device_name)
    if requested_device.type != "cuda":
        return requested_device

    if not torch.cuda.is_available():
        logger.warning(
            "Configured device %s is unavailable because CUDA is not available. Falling back to CPU.",
            device_name,
        )
        return torch.device("cpu")

    cuda_count = torch.cuda.device_count()
    device_index = requested_device.index if requested_device.index is not None else torch.cuda.current_device()
    if device_index >= cuda_count:
        fallback_device = torch.device("cuda:0")
        logger.warning(
            "Configured device %s is unavailable; only %d CUDA device(s) detected. Falling back to %s.",
            device_name,
            cuda_count,
            fallback_device,
        )
        return fallback_device

    return requested_device


def get_autocast_context(device):
    if device.type != "cuda":
        return nullcontext()

    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    return torch.cuda.amp.autocast(dtype=dtype)


if __name__ == "__main__":
    args = parse_args()
    ram_config = get_ram_config()

    promp_name = re.sub(r'\W+', '_', args.prompt.lower())
    logging.info(f"Prompt name: {promp_name}")

    output_dir = os.path.join(args.output)
    debug_dir = os.path.join(args.output, f"{promp_name}")
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    detection_file = os.path.join(output_dir, 'detections.npy')
    detections = np.load(detection_file, allow_pickle=True).item()
    mask = detections['masks']
    class_ids = detections['class_ids']
    xyxys = detections['xyxys']

    vlm_category_json = pjoin(output_dir, 'vlm_categories.json')
    with open(vlm_category_json, 'r') as f:
        vlm_categories = json.load(f)

    config_file = os.path.join(cur_dir, './controller/assets/calibration/external_camera', 'zed_config.json')
    extern_camera = ZEDCamera(config_file=config_file, is_warmup=False)
    rgb_ext_pth = pjoin(output_dir, 'rgb_ext.png')
    rgb_ext_left_pth = pjoin(output_dir, 'rgb_ext_left.png')
    rgb_ext_right_pth = pjoin(output_dir, 'rgb_ext_right.png')
    depth_ext_pth = pjoin(output_dir, 'depth_ext.npy')
    extern_camera.rgb_image = cv2.cvtColor(cv2.imread(rgb_ext_pth), cv2.COLOR_BGR2RGB)
    extern_camera.depth_map = np.load(depth_ext_pth)
    extern_camera.left_image = cv2.cvtColor(cv2.imread(rgb_ext_left_pth), cv2.COLOR_BGR2RGB)
    extern_camera.right_image = cv2.cvtColor(cv2.imread(rgb_ext_right_pth), cv2.COLOR_BGR2RGB)

    extern_camera2 = RealSense455Camera()
    rgb_ext2, depth_ext2 = extern_camera2.capture_image()
    rgb_ext2 = cv2.resize(rgb_ext2, (1280, 960), interpolation=cv2.INTER_LINEAR)
    rgb_ext2 = rgb_ext2[240:, :, :]
    rgb_ext2_pth = pjoin(output_dir, 'rgb_ext2.png')
    save_rgb_png(rgb_ext2, rgb_ext2_pth)

    device = resolve_torch_device(ram_config.device)
    logger.info(f"Using torch device: {device}")
    vggt_preload_start = time.time()
    vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    vggt_preload_time = time.time() - vggt_preload_start
    logger.info(f"VGGT model preload time: {vggt_preload_time:.3f} seconds")

    image_names = [rgb_ext_left_pth, rgb_ext_right_pth, rgb_ext2_pth]
    images = load_and_preprocess_images(image_names).to(device)

    with torch.no_grad():
        with get_autocast_context(device):
            vggt_inference_start = time.time()
            predictions = vggt_model(images)
            vggt_inference_time = time.time() - vggt_inference_start
            logger.info(f"VGGT model inference time: {vggt_inference_time:.3f} seconds")
    
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    vggt_intrinsic = intrinsic.squeeze(0).cpu().numpy()[0]
    vggt_baseline = np.linalg.norm(extrinsic[0, :, 3] - extrinsic[1, :, 3])
    vggt_depth_rescale = 0.120 / vggt_baseline

    depth_map = predictions["depth"].detach().cpu().numpy().squeeze(0)[0, ..., 0] * 1000 * vggt_depth_rescale
    depth_confidence_map = predictions["depth_conf"].detach().cpu().numpy().squeeze(0)[0]

    depth_confidence_map = cv2.resize(depth_confidence_map, (extern_camera.width, extern_camera.height), interpolation=cv2.INTER_NEAREST)
    vggt_depth_map_ext = cv2.resize(depth_map, (extern_camera.width, extern_camera.height), interpolation=cv2.INTER_NEAREST)
    save_depth_npy(vggt_depth_map_ext, os.path.join(output_dir, "vggt_depth_ext.npy"))
    save_rgb_png(turbo_cmap(vggt_depth_map_ext), os.path.join(output_dir, "vggt_depth_ext.png"))

    confidence_threshold = np.percentile(depth_confidence_map.flatten(), 10)
    depth_mask = (depth_confidence_map.flatten() >= confidence_threshold) & (depth_confidence_map.flatten() > 0.1)
    depth_mask = depth_mask.reshape((extern_camera.height, extern_camera.width))

    config_path = pjoin(cur_dir, './visions/ram/assets/ram.json')
    RAM_Model = RAMModel(config_path, device=str(device))
    ObjectTemplate = RAMTemplate(config_path)
    RAM_IO = RAMIO()

    K = np.array([
        [vggt_intrinsic[0,0] * 1280 / 518, 0,   vggt_intrinsic[0,2] * 1280 / 518],
        [0,  vggt_intrinsic[1,1] * 1280 / 518,  vggt_intrinsic[1,2] * 1280 / 518],
        [0,  0,   1 ]
    ])
    np.save(os.path.join(output_dir, "vggt_K.npy"), K)

    objects_meta = EasyDict({})
    for object in vlm_categories:
        obj_name = object['name']
        obj_id = int(object['id'])

        if obj_id < 0:
            logger.warning(f"Object ID {obj_id} is invalid, skipping...")
            continue

        obj_category = object['category']
        objects_meta[str(obj_id)] = EasyDict({})

        obj_mask = mask[obj_id, ...]
        obj_xyxy = xyxys[obj_id, ...]

        inst_mask = np.logical_and(obj_mask, depth_mask)
        save_rgb_png((inst_mask * 255).astype(np.uint8), os.path.join(debug_dir, f'ram_{obj_category}_id{obj_id:02}.png'))

        inst_img = RAM_IO.extract_rgb_patch(extern_camera.rgb_image, obj_mask)

        choose, pts, pts2d = RAM_IO.extract_obj_pts(inst_mask, vggt_depth_map_ext, 1.0, K, True)

        if 'other' in obj_category:
            pred_sRT = np.eye(4)
            pred_sRT[:3, 3] = np.mean(pts, axis=0).flatten()
            objects_meta[str(obj_id)]['centroid'] = pred_sRT
            continue

        logger.info(f">>> Processing object: {obj_name}, ID: {obj_id}, Category: {obj_category}")
        ObjectTemplate.load_template(obj_category)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        o3d.io.write_point_cloud(os.path.join(debug_dir, f'ram_{obj_category}_id{obj_id:02}.ply'), pcd)

        RAM_Model.viewpoint_estimation(ObjectTemplate, inst_img, choose, pts)
        RAM_Model.ramk_nocs_estimation(ObjectTemplate, inst_img, choose, pts)
        pred_sRT = RAM_Model.best_sRT
        unique_nocs = RAM_Model.best_nocs
        unique_pts = RAM_Model.best_pts

        vis_size = RAM_Model.best_size / np.cbrt(np.linalg.det(RAM_Model.best_sRT[:3, :3]))

        dis_img = extern_camera.rgb_image.copy()
        dis_img = RAM_Model.vis_pose(dis_img, pred_sRT, vis_size, K)
        save_rgb_png(dis_img, os.path.join(debug_dir, f'ram_{obj_category}_id{obj_id:02}_pose.png'))

        grasps, is_symmetry = RAM_Model.grasp_estimation(ObjectTemplate, pts, pts2d, K)
        for grasp_name, grasp_meta in grasps.items():
            grasp_pixel = grasp_meta['grasp_pixel']
            grasp_pose = grasp_meta['grasp_pose']

            dis_img = extern_camera.rgb_image.copy()
            dis_img = RAM_Model.vis_grasp_point(dis_img, grasp_pixel)
            dis_img = RAM_Model.vis_pose(dis_img, pred_sRT, vis_size, K)
            save_rgb_png(dis_img, os.path.join(debug_dir, f'ram_{obj_category}_id{obj_id:02}_{grasp_name}.png'))

            if is_symmetry:
                robot_init = np.eye(4)
                robot_init[:3,:3] = euler_to_matrix(-180, 0, -90)
                robot_optimal_grasp_pose = get_sym_equivalent_pose(extern_camera.c2w @ grasp_pose, robot_init)
                objects_meta[str(obj_id)][grasp_name] = np.linalg.inv(extern_camera.c2w) @ robot_optimal_grasp_pose
            else:
                objects_meta[str(obj_id)][grasp_name] = grasp_pose

        planes = RAM_Model.function_plane_estimation(ObjectTemplate, pts)
        for plane_name, plane_meta in planes.items():
            function_point = plane_meta['position']
            function_normal = plane_meta['orientation']

            dis_img = extern_camera.rgb_image.copy()
            function_dis = RAM_Model.vis_function_plane(dis_img, function_point, function_normal, pred_sRT, K)
            save_rgb_png(function_dis, os.path.join(debug_dir, f'ram_{obj_category}_id{obj_id:02}_{plane_name}.png'))

            objects_meta[str(obj_id)][plane_name] = np.eye(4)
            objects_meta[str(obj_id)][plane_name][:3, :3] = normal_to_rotmat(function_normal)
            objects_meta[str(obj_id)][plane_name][:3, 3] = function_point.flatten()
            
            pred_scale = np.cbrt(np.linalg.det(pred_sRT[:3, :3]))
            objects_meta[str(obj_id)][plane_name] = pred_sRT @ objects_meta[str(obj_id)][plane_name]
            objects_meta[str(obj_id)][plane_name][:3, :3] = objects_meta[str(obj_id)][plane_name][:3, :3] / pred_scale

        objects_meta[str(obj_id)]['centroid'] = pred_sRT
        objects_meta[str(obj_id)]['object_size'] = RAM_Model.best_size

        bbox_planes = get_bbox_plane_poses(objects_meta[str(obj_id)]['object_size'] / np.cbrt(np.linalg.det(RAM_Model.best_sRT[:3, :3])), \
                                                    objects_meta[str(obj_id)]['centroid'])
        objects_meta[str(obj_id)]['bbox_plane_x1'] = bbox_planes[0]
        objects_meta[str(obj_id)]['bbox_plane_x2'] = bbox_planes[1]
        objects_meta[str(obj_id)]['bbox_plane_y1'] = bbox_planes[2]
        objects_meta[str(obj_id)]['bbox_plane_y2'] = bbox_planes[3]
        objects_meta[str(obj_id)]['bbox_plane_z1'] = bbox_planes[4]
        objects_meta[str(obj_id)]['bbox_plane_z2'] = bbox_planes[5]

    objects_meta_npy_path = os.path.join(output_dir, "ram_objects_meta.npy")
    np.save(objects_meta_npy_path, objects_meta)
    logger.info(f">>> Save objects meta information to {objects_meta_npy_path}.")



    
