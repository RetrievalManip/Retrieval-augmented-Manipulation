import numpy as np
import json
import os
import torch

from utils.image_utils import *
from utils.camera_utils import *
from utils.geo_utils import *
from utils.transform_utils import *
from controller.fairino_arm import RobotArm
from controller.base_camera import BaseCamera
from controller.viewer import Viewer

import matplotlib.pyplot as plt
import open3d as o3d
from scipy.ndimage import distance_transform_edt, gaussian_filter

from .path_planner import PathPlanner
from .semantic_map import SemanticMap
from utils.config import get_planner_config


def get_object_pcd(robot, extern_camera, detections, class_id=0, use_mask=True, use_detect=False, coord="world"):
    mask = None
    if use_mask and detections is not None and getattr(detections, "mask", None) is not None:
        mask = detections.mask[class_id]
    c2w = extern_camera.c2w if coord == "world" else np.eye(4)
    return get_colored_points_from_depth(
        depths=extern_camera.depth_map,
        rgbs=extern_camera.rgb_image,
        c2w=c2w,
        fx=extern_camera.fx,
        fy=extern_camera.fy,
        cx=extern_camera.cx,
        cy=extern_camera.cy,
        img_size=(extern_camera.width, extern_camera.height),
        mask=mask,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )


class VoxelLinearRotationMap:
    def __init__(self, start_voxel, target_voxel, start_euler, target_euler):
        self.start_voxel = np.asarray(start_voxel, dtype=np.float32)
        self.target_voxel = np.asarray(target_voxel, dtype=np.float32)
        self.start_euler = np.asarray(start_euler, dtype=np.float32)
        self.target_euler = np.asarray(target_euler, dtype=np.float32)
        self.path_vec = self.target_voxel - self.start_voxel
        self.path_len_sq = float(np.dot(self.path_vec, self.path_vec))

    def _alpha(self, voxel_xyz):
        if self.path_len_sq == 0:
            return 1.0
        voxel_xyz = np.asarray(voxel_xyz, dtype=np.float32)
        alpha = np.dot(voxel_xyz - self.start_voxel, self.path_vec) / self.path_len_sq
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return alpha * alpha * (3.0 - 2.0 * alpha)

    def value_at(self, voxel_xyz):
        alpha = self._alpha(voxel_xyz)
        euler = self.start_euler + alpha * (self.target_euler - self.start_euler)
        return euler2quaternion(euler).astype(np.float32)

    def __getitem__(self, key):
        return self.value_at(key)


class VoxelLinearStateMap:
    def __init__(self, start_voxel, target_voxel, initial_state, target_state):
        self.start_voxel = np.asarray(start_voxel, dtype=np.float32)
        self.target_voxel = np.asarray(target_voxel, dtype=np.float32)
        self.initial_state = int(initial_state)
        self.target_state = int(target_state)
        self.path_vec = self.target_voxel - self.start_voxel
        self.path_len_sq = float(np.dot(self.path_vec, self.path_vec))

    def _alpha(self, voxel_xyz):
        if self.path_len_sq == 0:
            return 1.0
        voxel_xyz = np.asarray(voxel_xyz, dtype=np.float32)
        alpha = np.dot(voxel_xyz - self.start_voxel, self.path_vec) / self.path_len_sq
        return float(np.clip(alpha, 0.0, 1.0))

    def value_at(self, voxel_xyz):
        return self.target_state if self._alpha(voxel_xyz) >= 0.5 else self.initial_state

    def __getitem__(self, key):
        return self.value_at(key)


class TrajectoryGenerator():
    def __init__(
            self,
            robot: RobotArm,
            hand_camera: BaseCamera,
            extern_camera: BaseCamera,
            mapsize: int = None,
            viewer: Viewer = None,
    ):
        self.robot = robot
        self.hand_camera = hand_camera
        self.extern_camera = extern_camera
        self.gripper_status = 0

        planner_config = get_planner_config()
        self.map_size = mapsize or planner_config.map_size
        self.workspace = planner_config.workspace or [0, -600, -500, 1200, 600, 700]
        self.voxel_size = (self.workspace[3] - self.workspace[0]) // self.map_size
        self.visual = None
        self.viewer = viewer

        self.full_trajectory = []
        self.full_viewer_path = []

        self.cur_pose = [480, -110, 312, -179, 0, -90]
        self.table_height = planner_config.table_height

    def _normalize_map(self, voxel_map):
        min_value = np.min(voxel_map)
        max_value = np.max(voxel_map)
        if max_value == min_value:
            return voxel_map
        return (voxel_map - min_value) / (max_value - min_value)

    def _points_to_voxel_map(self, points, value=1.0):
        voxel_map = np.zeros((self.map_size, self.map_size, self.map_size), dtype=np.float32)
        for point in points:
            voxel = np.asarray(point).round().astype(int)
            voxel = np.clip(voxel, 0, self.map_size - 1)
            voxel_map[voxel[0], voxel[1], voxel[2]] = value
        return voxel_map

    def _smooth_binary_map(self, voxel_map, sigma):
        if np.max(voxel_map) <= 0:
            return voxel_map
        smooth_map = gaussian_filter(voxel_map.astype(np.float32), sigma=sigma)
        return self._normalize_map(smooth_map)

    def _active_voxel_indices(self, voxel_map, threshold=0.5, max_points=20000):
        if voxel_map.size == 0 or np.max(voxel_map) <= 0:
            return np.empty((0, 3), dtype=np.int64)

        if np.array_equal(voxel_map, voxel_map.astype(bool)):
            indices = np.argwhere(voxel_map > 0)
        else:
            indices = np.argwhere(voxel_map >= np.max(voxel_map) * threshold)

        if len(indices) > max_points:
            sample_idx = np.linspace(0, len(indices) - 1, max_points).astype(int)
            indices = indices[sample_idx]
        return indices

    def _lookup_map_value(self, voxel_map, voxel_xyz):
        voxel_xyz = np.round(voxel_xyz).astype(int)
        voxel_xyz = np.clip(voxel_xyz, 0, self.map_size - 1)
        if hasattr(voxel_map, "value_at"):
            return voxel_map.value_at(voxel_xyz)
        return voxel_map[voxel_xyz[0], voxel_xyz[1], voxel_xyz[2]]
        
    def voxel2base(self, pos):
        return (pos[0] * self.voxel_size + self.workspace[0],
                pos[1] * self.voxel_size + self.workspace[1],
                pos[2] * self.voxel_size + self.workspace[2])
    
    def base2voxel(self, pos):      
        assert pos[0] >= self.workspace[0] and pos[0] <= self.workspace[3], "x out of range"
        assert pos[1] >= self.workspace[1] and pos[1] <= self.workspace[4], "y out of range"
        assert pos[2] >= self.workspace[2] and pos[2] <= self.workspace[5], "z out of range"

        return (int((pos[0] - self.workspace[0]) // self.voxel_size),
                int((pos[1] - self.workspace[1]) // self.voxel_size),
                int((pos[2] - self.workspace[2]) // self.voxel_size))

    def get_target_voxel(self, detection, pts_mode='bbox'):
        assert pts_mode in ['centroid', 'tcbb', 'acc', 'bbox'], "pts_mode should be in ['centroid', 'tcbb', 'acc', 'bbox']"
        points_obj, colors_obj = get_object_pcd(
            self.robot, self.extern_camera,
            detections=detection, class_id=0,
            use_mask=True, use_detect=False,
            coord="world",
        )

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_obj.cpu().numpy())
        nb_neighbors = 20
        std_ratio = 4.0
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                                 std_ratio=std_ratio)
        obj_points = np.asarray(cl.points)

        if pts_mode == 'centroid':
            points_obj = np.mean(obj_points, axis=0)
            
        elif pts_mode == 'tcbb':
            obj_bbox = get_pcd_bbx(obj_points)
            points_obj = np.array([[
                (obj_bbox[0] + obj_bbox[3]) / 2,
                (obj_bbox[1] + obj_bbox[4]) / 2,
                obj_bbox[5] - 6
            ]])
        elif pts_mode == 'acc':
            points_obj = obj_bbox
        elif pts_mode == 'bbox':
            obj_bbox = get_pcd_bbx(obj_points)
            points_obj = self.bbox2voxels(obj_bbox)
            return points_obj, colors_obj

        points_obj = [self.base2voxel(point) for point in np.asarray(points_obj).reshape(-1, 3)]
        
        return points_obj, colors_obj

    def bbox2voxels(self, bbox):
        p1 = self.base2voxel(bbox[:3])
        p2 = self.base2voxel(bbox[3:6])
        return self.bbox2points(np.asarray([*p1, *p2]))
    
    def bbox2points(self, bbox):
        p1 = np.array(bbox[:3])
        p2 = np.array(bbox[3:6])
        
        low = np.minimum(p1, p2)
        high = np.maximum(p1, p2)
        
        xs = np.arange(low[0], high[0] + 1)
        ys = np.arange(low[1], high[1] + 1)
        zs = np.arange(low[2], high[2] + 1)
        
        x_grid, y_grid, z_grid = np.meshgrid(xs, ys, zs, indexing='ij')
        
        points = np.stack([x_grid, y_grid, z_grid], axis=-1).reshape(-1, 3)
        
        return points

    def get_sym_equivalent_pose(self, T_target, T_init):
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
    
    def euler_to_matrix(self, roll, pitch, yaw):
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

    def generatePath(self, grasp_in_world, move_type="grasp", constraints=[]):
        desired_gripper = grasp_in_world[-1]
        gripper_instruction = 1 if desired_gripper == 0 else 0
        grasp_pos = np.array(grasp_in_world[:6])
        start_pose = np.asarray(self.cur_pose, dtype=np.float32)

        final_affordance_map = self.getAffordanceMap_from_pose(grasp_pos)
        self.viewer_map(final_affordance_map.voxel_map, [0, 255, 0])
        self.draw_voxel_map(final_affordance_map.voxel_map)

        avoidance_map = self.getAvoidanceMap(constraints)
        self.viewer_map(avoidance_map.voxel_map, color=[255, 0, 0])
        rotation_map = self.getRotationMap_from_pose(start_pose, grasp_pos)
        velocity_map = self.getVelocityMap(vel=1)
        gripper_state_map = self.getGripperMap_from_pose(
            start_pose=start_pose,
            target_pose=grasp_pos,
            gripper_instruction=gripper_instruction,
        )

        start_pos = self.base2voxel(self.cur_pose)
        start_pos = np.asarray(start_pos).astype(int)

        config_path = os.path.join(os.path.dirname(__file__), "planner_config.json")
        with open(config_path, "r") as f:
            planner_config = json.load(f)
        planner = PathPlanner(planner_config, map_size=self.map_size)

        trajectory = []

        if move_type == "grasp object":
            trajectory2 = []
            grasp_path = []
            if final_affordance_map.voxel_map is not None:
                grasp_path, planner_info2 = planner.optimize(start_pos, final_affordance_map.voxel_map, avoidance_map.voxel_map)
                self.full_viewer_path.extend(grasp_path)
                trajectory2 = self.path2trajectory(grasp_path, rotation_map.voxel_map, velocity_map.voxel_map, gripper_state_map.voxel_map)
            else:
                trajectory2 = []
            trajectory = trajectory2
        else:
            if final_affordance_map.voxel_map is not None:
                path_voxel, planner_info = planner.optimize(start_pos, final_affordance_map.voxel_map, avoidance_map.voxel_map)
                self.view_background()
                self.full_viewer_path.extend(path_voxel)
                segment_trajectory = self.path2trajectory(path_voxel, rotation_map.voxel_map, velocity_map.voxel_map, gripper_state_map.voxel_map)
                trajectory = segment_trajectory
        if trajectory:
            last_waypoint = trajectory[-1]
            self.cur_pose = grasp_in_world[:6].tolist()
            if last_waypoint[3] != self.gripper_status:
                RT_grasp = pos_to_mat(grasp_pos)
                self.viewer.add_grasp(RT_grasp, 0.7)
                self.gripper_status = last_waypoint[3]
                
        self.full_trajectory.extend(trajectory)
        self.viewer_path(self.full_viewer_path)

    def generatePath_simple(self, grasp_in_world, move_type="grasp", constraints=[]):
        return self.generatePath(grasp_in_world=grasp_in_world, move_type=move_type, constraints=constraints)
            
    def path2trajectory(self, path, rotation_map, velocity_map, gripper_map):
        traj = []
        for i in range(len(path)):
            voxel_xyz = path[i]
            world_xyz = voxel_xyz / self.map_size
            voxel_xyz = np.round(voxel_xyz).astype(int)
            rotation = self._lookup_map_value(rotation_map, voxel_xyz)
            velocity = self._lookup_map_value(velocity_map, voxel_xyz)
            gripper = self._lookup_map_value(gripper_map, voxel_xyz)
            if isinstance(gripper_map, np.ndarray) and (i == len(path) - 1) and not (np.all(gripper_map == 1) or np.all(gripper_map == 0)):
                less_common_value = 1 if np.sum(gripper_map == 1) < np.sum(gripper_map == 0) else 0
                less_common_indices = np.where(gripper_map == less_common_value)
                less_common_indices = np.array(less_common_indices).T
                closest_distance = np.min(np.linalg.norm(less_common_indices - voxel_xyz[None, :], axis=0))
                if closest_distance <= 3:
                    gripper = less_common_value
            
            traj.append([world_xyz, rotation, velocity, gripper])
        for _ in range(2):
            traj.append([world_xyz, rotation, velocity, gripper])
        return traj
    
    def getAffordanceMap_from_pose(self, pose=None):
        target_voxel = self.base2voxel(pose[:3])
        points = [target_voxel]
        affordance_map = SemanticMap(map_size=self.map_size)
        target_map = self._points_to_voxel_map(points, value=1.0)
        distance_map = distance_transform_edt(1 - target_map)
        affordance_map.voxel_map = 1 - self._normalize_map(distance_map)

        return affordance_map

    def getAvoidanceMap(self, constraints=[]):
        avoidance_map = SemanticMap(map_size=self.map_size)
        if len(constraints) > 0:
            for detection in constraints:
                points, _ = self.get_target_voxel(detection, pts_mode='bbox')
                obstacle_map = self._points_to_voxel_map(points, value=1.0)
                avoidance_map.voxel_map = np.maximum(
                    avoidance_map.voxel_map,
                    self._smooth_binary_map(obstacle_map, sigma=3),
                )

        return avoidance_map
    
    def getRotationMap_from_pose(self, start_pose=None, target_pose=None):
        rotation_map = SemanticMap(map_size=self.map_size)
        if target_pose is None:
            rotation_map.voxel_map = np.full((self.map_size, self.map_size, self.map_size, 4), euler2quaternion([-178,0,-90]), dtype=np.float32)
            return rotation_map

        if start_pose is None:
            start_pose = np.asarray(self.cur_pose, dtype=np.float32)

        start_voxel = np.asarray(self.base2voxel(start_pose[:3]), dtype=np.float32)
        target_voxel = np.asarray(self.base2voxel(target_pose[:3]), dtype=np.float32)
        rotation_map.voxel_map = VoxelLinearRotationMap(
            start_voxel=start_voxel,
            target_voxel=target_voxel,
            start_euler=start_pose[3:6],
            target_euler=target_pose[3:6],
        )

        return rotation_map

    def getVelocityMap(self, vel):
        velocity_map = SemanticMap(map_size=self.map_size)
        velocity_map.voxel_map.fill(vel)

        return velocity_map

    def getGripperMap_from_pose(self, start_pose=None, target_pose=None, gripper_instruction=0):
        gripper_map = SemanticMap(map_size=self.map_size)

        if self.gripper_status == 0:
            initial_state = 0
        else:
            initial_state = 1

        if target_pose is None:
            gripper_map.voxel_map.fill(initial_state)
            return gripper_map

        if start_pose is None:
            start_pose = np.asarray(self.cur_pose, dtype=np.float32)

        start_voxel = np.asarray(self.base2voxel(start_pose[:3]), dtype=np.float32)
        target_voxel = np.asarray(self.base2voxel(target_pose[:3]), dtype=np.float32)
        gripper_map.voxel_map = VoxelLinearStateMap(
            start_voxel=start_voxel,
            target_voxel=target_voxel,
            initial_state=initial_state,
            target_state=gripper_instruction,
        )
        return gripper_map
        

    
    def draw_waypoints(self, waypoints):
        points = np.asarray(self.visual.points)
        colors = np.asarray(self.visual.colors)
        
        for waypoint in waypoints:
            mask = np.all(points == waypoint, axis=1)
            colors[mask] = [0, 0, 255]
        
        self.visual.colors = o3d.utility.Vector3dVector(colors)
    
    def draw_voxel_map(self, voxel_map):
        points = self._active_voxel_indices(voxel_map, threshold=0.95).astype(np.float64)
        colors = np.full((points.shape[0], 3), [255, 0, 0], dtype=np.float64)

        self.visual = o3d.geometry.PointCloud()
        self.visual.points = o3d.utility.Vector3dVector(points)
        self.visual.colors = o3d.utility.Vector3dVector(colors)
    
    def visualize(self):
        if self.visual is not None:
            o3d.visualization.draw_geometries([self.visual])
        else:
            print("No voxel map to visualize")

    def _perform_action(self, waypoint):
        pos = []
        pos += list(self.voxel2base(waypoint[0] * self.map_size))
        euler = quaternion2euler(waypoint[1])
        pos += euler.tolist()
        if pos[2] < self.table_height:
            pos[2] = self.table_height
        
        print("Moving to position:", pos)
        print("Gripper status:", waypoint[3])
        self.robot.tip_moveto(pos)

        if waypoint[3] != self.gripper_status:
            self.robot.act_gripper(waypoint[3])
            self.gripper_status = waypoint[3]
    
    def viewer_path(self, voxel_path):
        pts = []
        for voxel in voxel_path:
            pts.append(self.voxel2base(voxel))
        pts = np.array(pts)
        blue_color = (0, 0, 255)
        self.viewer.add_trajectory(pts, blue_color)

    def executeFullTrajectory(self):
        print("Starting execution of full accumulated trajectory with", len(self.full_trajectory), "waypoints.")
        for i, waypoint in enumerate(self.full_trajectory):
            self._perform_action(waypoint)
        print("Execution of full trajectory completed.")
        self.full_trajectory = []
        self.full_viewer_path = []

    def executeFullTrajectory_simple(self):
        return self.executeFullTrajectory()
    
    def viewer_map(self, voxel_map, color=[255, 0, 0]):
        points_indices = self._active_voxel_indices(voxel_map, threshold=0.8).astype(np.float64)
        
        if points_indices.size == 0:
            points = np.empty((0, 3))
            colors = np.empty((0, 3))
        else:
            points = np.array([self.voxel2base(point) for point in points_indices])
            if points.ndim == 1:
                points = points.reshape(1, -1)
            
            colors = np.array([color] * points.shape[0])
            if colors.ndim == 1:
                colors = colors.reshape(points.shape[0], -1)
        
        self.viewer.add_pcd(points, colors)

    def view_background(self):
        points_ext, colors_ext = get_colored_points_from_depth(
            depths=self.extern_camera.depth_map,
            rgbs=self.extern_camera.rgb_image,
            c2w=self.extern_camera.c2w,
            fx=self.extern_camera.fx,
            fy=self.extern_camera.fy,
            cx=self.extern_camera.cx,
            cy=self.extern_camera.cy,
            img_size=(self.extern_camera.width, self.extern_camera.height),
            mask=None,
            device=torch.device("cuda")
        )
        err, ret = self.robot.get_tool_pose()
        robot_tool2w = pos_to_mat(ret)
        points_hand, colors_hand = get_colored_points_from_depth(
            depths=self.hand_camera.depth_map,
            rgbs=self.hand_camera.rgb_image,
            c2w= robot_tool2w @ self.hand_camera.c2w,
            fx=self.hand_camera.fx,
            fy=self.hand_camera.fy,
            cx=self.hand_camera.cx,
            cy=self.hand_camera.cy,
            img_size=(self.hand_camera.width, self.hand_camera.height),
            mask=None,
            device=torch.device("cuda")
        )
        self.viewer.add_pcd(points_ext, colors_ext)
        self.viewer.add_camera(self.extern_camera)
        self.viewer.add_camera(self.hand_camera, trans_mat=robot_tool2w)
