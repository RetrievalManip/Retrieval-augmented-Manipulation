import sys
import cv2
import time
import os 
import numpy as np
from os.path import join as pjoin
import json
import argparse
import logging
from easydict import EasyDict
import supervision as sv
import copy

from controller.extern_camera import ZEDCamera
from controller.hand_camera import RealSense435Camera
from utils.transform_utils import pos_to_mat
from controller.fairino_arm import RobotArm
from controller.viewer import Viewer

from utils.constraint_parser import ConstraintParser
from planner.trajectory_generator import TrajectoryGenerator
from utils.other_utils import set_random_seeds

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


class ConstraintParserFunctions():
    def __init__(self, robot_arm: RobotArm, objects_meta: EasyDict, c2w: np.ndarray):
        self.robot = robot_arm
        self.objects_meta = objects_meta
        self.objects_org_meta = copy.deepcopy(objects_meta)
        self.camera2world_mat = c2w

    def get_gripper_in_world(self):
        cur_pose = self.robot.get_tool_pose()[1]
        tip2tool_dist = 210 - 22
        tip2tool = np.eye(4)
        tip2tool[2, 3] = -tip2tool_dist
        tip2w = pos_to_mat(cur_pose)
        tool2tip = np.linalg.inv(tip2tool) 
        gripper2world_mat = tip2w @ tool2tip 
        return  gripper2world_mat  

    def get_original_primitive_in_world(self, primitive_name, object_id):
        primitive2camera_mat = self.objects_org_meta[str(object_id)][primitive_name]
        primitive2world_mat = self.camera2world_mat @ primitive2camera_mat
        return primitive2world_mat

    def get_primitive_in_world(self, primitive_name, object_id):
        primitive2camera_mat = self.objects_meta[str(object_id)][primitive_name]
        primitive2world_mat = self.camera2world_mat @ primitive2camera_mat
        return primitive2world_mat
    
    def get_original_primitive_in_object(self, primitive_name, object_id):
        primitive2camera_mat = self.objects_org_meta[str(object_id)][primitive_name]
        centroid2camera_mat = self.objects_org_meta[str(object_id)]['centroid']
        primitive2object = np.linalg.inv(centroid2camera_mat) @ primitive2camera_mat
        return primitive2object

    def get_primitive_in_object(self, primitive_name, object_id):
        primitive2camera_mat = self.objects_meta[str(object_id)][primitive_name]
        centroid2camera_mat = self.objects_meta[str(object_id)]['centroid']
        primitive2object = np.linalg.inv(centroid2camera_mat) @ primitive2camera_mat
        return primitive2object
    
    def update_object_meta_in_camera(self, new_centroid2world_mat, object_id, update_centroid=True):
        new_centroid2camera_mat = np.linalg.inv(self.camera2world_mat) @ new_centroid2world_mat
        
        for primitive_name, primtive2camera_mat in self.objects_meta[str(object_id)].items():
            if primitive_name.startswith("grasp_") or primitive_name.startswith("plane_") or primitive_name.startswith("bbox_plane"):
                self.objects_meta[str(object_id)][primitive_name] = \
                    new_centroid2camera_mat @ np.linalg.inv(self.objects_meta[str(object_id)]['centroid']) @ primtive2camera_mat

        if update_centroid:
            self.objects_meta[str(object_id)]['centroid'] = new_centroid2camera_mat


def parse_args():
    parser = argparse.ArgumentParser(description="Parse VLM action constraints and execute robot trajectories")
    parser.add_argument("--output", type=str, default="./output/scene_01", help="Directory to save outputs")
    return parser.parse_args()


if __name__ == "__main__":
    set_random_seeds(42)
    args = parse_args()

    output_dir = os.path.join(args.output)
    cur_dir = os.path.dirname(os.path.abspath(__file__))
   
    config_file = os.path.join(cur_dir, './controller/assets/calibration/external_camera', 'zed_config.json')
    extern_camera = ZEDCamera(config_file=config_file, is_warmup=True)
    rgb_ext_pth = pjoin(output_dir, 'rgb_ext.png')
    rgb_ext_left_pth = pjoin(output_dir, 'rgb_ext_left.png')
    rgb_ext_right_pth = pjoin(output_dir, 'rgb_ext_right.png')
    depth_ext_pth = pjoin(output_dir, "vggt_depth_ext.npy")
    extern_camera.rgb_image = cv2.cvtColor(cv2.imread(rgb_ext_pth), cv2.COLOR_BGR2RGB)
    extern_camera.depth_map = np.load(depth_ext_pth)
    extern_camera.left_image = cv2.cvtColor(cv2.imread(rgb_ext_left_pth), cv2.COLOR_BGR2RGB)
    extern_camera.right_image = cv2.cvtColor(cv2.imread(rgb_ext_right_pth), cv2.COLOR_BGR2RGB)

    vggt_K = np.load(os.path.join(output_dir, 'vggt_K.npy'))
    extern_camera.K = vggt_K

    ram_objects_meta_file = pjoin(output_dir, 'ram_objects_meta.npy')
    with open(ram_objects_meta_file, 'rb') as f:
        ram_objects_meta = np.load(f, allow_pickle=True).item()
    for object_id, meta in ram_objects_meta.items():
        logger.info(f"Object ID: {object_id}, primitives: {meta.keys()}")


    vlm_actions_json = os.path.join(output_dir, "vlm_actions.json")
    with open(vlm_actions_json, 'r') as f:
        vlm_actions = json.load(f)

    config_file = os.path.join(cur_dir, './controller/assets', 'fr5_SHB901.json')
    robot = RobotArm(config_file=config_file)
    robot.reset()

    constraint_functions = ConstraintParserFunctions(
        robot_arm = robot,
        objects_meta = ram_objects_meta,
        c2w = extern_camera.c2w
    )

    constraint_solver = ConstraintParser(
        get_gripper_in_world=constraint_functions.get_gripper_in_world,
        get_primitive_in_world=constraint_functions.get_primitive_in_world,
        get_primitive_in_object=constraint_functions.get_primitive_in_object,
        update_primitive_in_camera=constraint_functions.update_object_meta_in_camera,
        get_original_primitive_in_world=constraint_functions.get_original_primitive_in_world,
        get_original_primitive_in_object=constraint_functions.get_original_primitive_in_object,
    )

    config_file = os.path.join(cur_dir, './controller/assets/calibration/hand_camera', 'realsense_config.json')
    hand_camera = RealSense435Camera(config_file=config_file)

    detection_file = os.path.join(output_dir, 'detections.npy')
    detections = np.load(detection_file, allow_pickle=True).item()

    viewer = Viewer()
    
    traj_gen = TrajectoryGenerator(
        robot=robot,
        hand_camera=hand_camera,
        extern_camera=extern_camera,
        viewer=viewer,
    )

    for subtask_dict in vlm_actions:
        subtask_id = subtask_dict["subtask_id"]
        obstacle_objects = subtask_dict.get("obstacle_objects", [])

        obstacle_detections = []
        for obstacle_object in obstacle_objects:
            object_id = int(obstacle_object['id'])
            if object_id >= 0:
                mask = detections['masks'][object_id, ...]
                xyxy = detections['xyxys'][object_id, ...]
                class_id = detections['class_ids'][object_id, ...]
                obstacle_detection = sv.Detections(
                    xyxy=xyxy[np.newaxis, ...],
                    mask=mask[np.newaxis, ...],
                    class_id=class_id[np.newaxis, ...]
                )
                obstacle_detections.append(obstacle_detection)
        
        gripper_expect_list = constraint_solver.parse_constraints(subtask_dict)

        for gripper_expect in gripper_expect_list:
            logger.warning(f"move type: {gripper_expect.action}")
            logger.warning(f"Subtask {subtask_id} --> Gripper expected pose in world: {gripper_expect.gripper_pos}")
            traj_gen.generatePath_simple(
                grasp_in_world=gripper_expect.gripper_pos,
                move_type=gripper_expect.action,
                constraints=obstacle_detections,
            )

    viewer.update_frame()
    viewer.set_trajectory_generator(traj_gen, capture_system=None, robot=robot)
    while not viewer.trajectory_executed:
        time.sleep(0.01)
