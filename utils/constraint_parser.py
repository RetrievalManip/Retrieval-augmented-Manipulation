import os
from os.path import join as opj
cur_pth = os.path.dirname(os.path.abspath(__file__))

import json
import re
import cv2
import copy
import base64
from typing import NamedTuple
from easydict import EasyDict
import numpy as np
from utils.transform_utils import pos_to_mat, mat_to_pos
from scipy.spatial.transform import Rotation as R


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

def compute_gripper_in_object(gripper2world, object2world):
    return np.linalg.inv(object2world) @ gripper2world

def revert_gripper_in_object(gripper2object, object2world):
    return object2world @ gripper2object

def align_vectors(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    
    if np.allclose(a, b):
        return np.eye(3)
    
    if np.allclose(a, -b):
        if abs(a[0]) < 0.9:
            perp = np.array([1, 0, 0])
        else:
            perp = np.array([0, 1, 0])
        v = perp - np.dot(perp, a) * a
        v = v / np.linalg.norm(v)
        return 2 * np.outer(v, v) - np.eye(3)
    
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    
    R = np.eye(3) + vx + np.dot(vx, vx) * (1 - c) / (s * s)
    
    return R

def filter_planes_by_normal(planes, filter_axes, angle_thresh_deg=15):
    AXIS_VEC = {
        'x': np.array([1,0,0]),
        'y': np.array([0,1,0]),
        'z': np.array([0,0,1]),
        '-x': np.array([-1,0,0]),
        '-y': np.array([0,-1,0]),
        '-z': np.array([0,0,-1]),
    }

    keep = []
    angle_thresh = np.deg2rad(angle_thresh_deg)
    for i, T in enumerate(planes):
        normal = T[:3,2]
        normal = normal / np.linalg.norm(normal)
        remove = False
        for ax in filter_axes:
            axis_vec = AXIS_VEC[ax]
            axis_vec = axis_vec / np.linalg.norm(axis_vec)
            angle = np.arccos(np.clip(np.dot(normal, axis_vec), -1, 1))
            if angle < angle_thresh or abs(angle - np.pi) < angle_thresh:
                remove = True
                break
        if not remove:
            keep.append(i)
    return keep

def rotate_pose_around_object_z(T_w_o, center_world, angle_rad):

    T_w_o_tmp = copy.deepcopy(T_w_o) / np.cbrt(np.linalg.det(T_w_o[:3, :3]))
    z_world = T_w_o_tmp[:3, 2]

    rot = R.from_rotvec(z_world * angle_rad)
    R_axis = rot.as_matrix()
    T_axis = np.eye(4)
    T_axis[:3,:3] = R_axis

    T_to_center = np.eye(4)
    T_to_center[:3,3] = -center_world
    T_back = np.eye(4)
    T_back[:3,3] = center_world

    T_rotate = T_back @ T_axis @ T_to_center

    T_w_o_new = T_rotate @ T_w_o
    
    return T_w_o_new


def rotate_pose_around_object_y(T_w_o, center_world, angle_rad):

    T_w_o_tmp = copy.deepcopy(T_w_o) / np.cbrt(np.linalg.det(T_w_o[:3, :3]))
    y_world = T_w_o_tmp[:3, 1]

    rot = R.from_rotvec(y_world * angle_rad)
    R_axis = rot.as_matrix()
    T_axis = np.eye(4)
    T_axis[:3,:3] = R_axis

    T_to_center = np.eye(4)
    T_to_center[:3,3] = -center_world
    T_back = np.eye(4)
    T_back[:3,3] = center_world

    T_rotate = T_back @ T_axis @ T_to_center

    T_w_o_new = T_rotate @ T_w_o
    return T_w_o_new

def find_farthest_plane_from_grasp(planes, grasp_T):
    grasp_pos = grasp_T[:3, 3]
    dists = []
    for T in planes:
        center = T[:3, 3]
        normal = T[:3, 2]
        normal = normal / np.linalg.norm(normal)
        dist = abs(np.dot(grasp_pos - center, normal))
        dists.append(dist)
    idx = np.argmax(dists)
    return idx, dists[idx]

def find_closest_plane_pair(planes_src, planes_tgt):
    centers_src = np.array([T[:3,3] for T in planes_src])
    centers_tgt = np.array([T[:3,3] for T in planes_tgt])
    min_dist = np.inf
    min_pair = (None, None)
    for i, c_src in enumerate(centers_src):
        for j, c_tgt in enumerate(centers_tgt):
            dist = np.linalg.norm(c_src - c_tgt)
            if dist < min_dist:
                min_dist = dist
                min_pair = (i, j)
    return min_pair[0], min_pair[1], min_dist

def find_plane_pair(planes_src, planes_tgt, grasp_src, filter_src_axes=['z', '-z'], \
    filter_tgt_axes=['z', '-z', 'y', '-y'], angle_thresh_deg=15):
    src_keep = filter_planes_by_normal(planes_src, filter_src_axes, angle_thresh_deg)
    tgt_keep = filter_planes_by_normal(planes_tgt, filter_tgt_axes, angle_thresh_deg)
    filtered_src_planes = [planes_src[i] for i in src_keep]
    filtered_tgt_planes = [planes_tgt[i] for i in tgt_keep]
    if len(filtered_src_planes) == 0 or len(filtered_tgt_planes) == 0:
        raise ValueError("No available planes after filtering.")

    src_far_idx_in_filtered, _ = find_farthest_plane_from_grasp(filtered_src_planes, grasp_src)
    src_far_idx = src_keep[src_far_idx_in_filtered]
    src_plane_far = planes_src[src_far_idx]
    src_planes_for_match = [src_plane_far]

    idx_src_final, idx_tgt_final, dist = find_closest_plane_pair(src_planes_for_match, filtered_tgt_planes)
    src_idx = src_far_idx
    tgt_idx = tgt_keep[idx_tgt_final]
    return src_idx, tgt_idx, dist

class ConstraintParser():
    def __init__(
        self, 
        get_gripper_in_world=None, 
        get_primitive_in_world=None, 
        get_primitive_in_object=None, 
        update_primitive_in_camera=None,
        get_original_primitive_in_world=None,
        get_original_primitive_in_object=None, 
        update_primitive_in_camera_directly=None
    ):
        self.get_gripper_in_world = get_gripper_in_world  
        self.get_primitive_in_world = get_primitive_in_world 
        self.get_primitive_in_object = get_primitive_in_object
        self.update_primitive_in_camera = update_primitive_in_camera
        self.get_original_primitive_in_world = get_original_primitive_in_world
        self.get_original_primitive_in_object = get_original_primitive_in_object
        self.update_primitive_in_camera_directly = update_primitive_in_camera_directly

        if self.get_gripper_in_world is None or self.get_primitive_in_world is None or self.get_primitive_in_object is None:
            raise ValueError("get_gripper_in_world function must be provided.")
        self.gripper2world = self.get_gripper_in_world()
        self.gripper_status = 1

    def update_gripper2world(self, gripper2world):
        self.gripper2world = gripper2world

    def update_gripper_status(self, gripper_status):
        if "open" in gripper_status:
            self.gripper_status = 1
        elif "close" in gripper_status:
            self.gripper_status = 0

    def get_gripper2world(self):
        return self.gripper2world
    
    def update_primitive(self, new_centroid2world_mat, object_id, update_centroid=True):
        self.update_primitive_in_camera(new_centroid2world_mat, object_id, update_centroid)
     
    def parse_constraints(self, subtask_dict: dict) -> dict:

        subtask_id = subtask_dict["subtask_id"]
        steps = subtask_dict["constraint_steps"]
        gripper_status = subtask_dict["gripper_status"]
        action = subtask_dict["action_type"]
        grasped_object = subtask_dict["grasped_object"]
        target_objects = subtask_dict["target_objects"]
        obstacle_objects = subtask_dict["obstacle_objects"]


        previous_gripper2world = self.get_gripper2world()
        previous_gripper_6dof = mat_to_pos(previous_gripper2world)
        previous_gripper_status = self.gripper_status
        for i, step in enumerate(steps):
            step_type = step["step_type"]
            step_description = step["step_description"]

            if step_type == 1:
                gripper_in_world, graps_point, grasped_object = self.parse_constraint_1(step_description, 
                    grasped_object=grasped_object, target_objects=target_objects)
            elif step_type == 2:
                gripper_in_world = self.parse_constraint_2(step_description, 
                    grasped_object=grasped_object, target_objects=target_objects)
            elif step_type == 3:
                gripper_in_world = self.parse_constraint_3(step_description, 
                    grasped_object=grasped_object, target_objects=target_objects)
            elif step_type == 4:
                gripper_in_world = self.parse_constraint_4(step_description, 
                    grasped_object=grasped_object, target_objects=target_objects)
            elif step_type == 5:
                gripper_in_world = self.parse_constraint_5(step_description, 
                    grasped_object=grasped_object, target_objects=target_objects)
            elif step_type == 6:
                gripper_in_world = self.parse_constraint_6(step_description, 
                    grasped_object=grasped_object, target_objects=target_objects)
            elif step_type == 7:
                gripper_in_world = self.parse_constraint_7(step_description, 
                    grasped_object=grasped_object, target_objects=target_objects)
            elif step_type == 8:
                gripper_in_world = self.parse_constraint_8(step_description, 
                    grasped_object=grasped_object, target_objects=target_objects)
            elif step_type == 9:
                gripper_in_world = self.parse_constraint_9(step_description, 
                    grasped_object=grasped_object, target_objects=target_objects)
            elif step_type == 10:
                gripper_in_world = self.parse_constraint_10(step_description, 
                    grasped_object=grasped_object, target_objects=target_objects)
            elif step_type == 11:
                gripper_in_world = self.parse_constraint_11(step_description, 
                    grasped_object=grasped_object, target_objects=target_objects)
            elif step_type == 12:
                gripper_in_world = self.parse_constraint_12(step_description, 
                    grasped_object=grasped_object, target_objects=target_objects)
            elif step_type == 13:
                gripper_in_world = self.parse_constraint_13(step_description, 
                    grasped_object=grasped_object, target_objects=target_objects)
            elif step_type == 14:
                gripper_in_world = self.parse_constraint_14(step_description, 
                    grasped_object=grasped_object, target_objects=target_objects)

        if not isinstance(gripper_in_world, list):
            gripper_in_world = [gripper_in_world]
                
        gripper_pose_list = []
        for gripper_in_world_mat in gripper_in_world:

            gripper_6dof = mat_to_pos(gripper_in_world_mat)

            if "open" in gripper_status:
                gripper_jaw = 1
            elif "close" in gripper_status:
                gripper_jaw = 0

            self.update_gripper_status(gripper_status)

            gripper_7dof = np.zeros(7)
            gripper_7dof[:6] = gripper_6dof
            gripper_7dof[6] = gripper_jaw 

            if "reset" in action:                
                lift_gripper_7dof = np.zeros(7)
                lift_gripper_7dof[:6] = copy.deepcopy(previous_gripper_6dof)
                lift_gripper_7dof[2] += 50
                lift_gripper_7dof[6] = 0
                gripper_pose_list.append(
                    EasyDict({
                        "gripper_pos": lift_gripper_7dof, 
                        "action": "lift object",
                    }),
                )

                move_gripper_7dof = np.zeros(7)
                move_gripper_7dof[:6] = copy.deepcopy(gripper_6dof)
                move_gripper_7dof[2] += 50
                move_gripper_7dof[6] = 0
                gripper_pose_list.append(
                    EasyDict({
                        "gripper_pos": move_gripper_7dof, 
                        "action": "move object",
                    }),
                )
                
                release_gripper_7dof = np.zeros(7)
                release_gripper_7dof[:6] = copy.deepcopy(gripper_6dof)
                release_gripper_7dof[6] = 1
                gripper_pose_list.append(
                    EasyDict({
                        "gripper_pos": release_gripper_7dof, 
                        "action": "put and release object",
                    }),
                )

                after_release_gripper_7dof = np.zeros(7)
                after_release_gripper_7dof = copy.deepcopy(gripper_7dof)
                after_release_gripper_7dof[2] += 100
                after_release_gripper_7dof[6] = 1
                gripper_pose_list.append(
                    EasyDict({
                        "gripper_pos": after_release_gripper_7dof, 
                        "action": "move gripper up after release",
                    })
                )
                after_release_gripper2world = pos_to_mat(after_release_gripper_7dof[:6])
                self.update_gripper2world(after_release_gripper2world)

            elif "release" in action:
                gripper_pose_list.append(
                    EasyDict({
                        "gripper_pos": gripper_7dof, 
                        "action": action,
                    })
                )
                after_release_gripper_7dof = np.zeros(7)
                after_release_gripper_7dof = copy.deepcopy(gripper_7dof)
                after_release_gripper_7dof[2] += 100
                gripper_pose_list.append(
                    EasyDict({
                        "gripper_pos": after_release_gripper_7dof, 
                        "action": "move gripper up after release",
                    })
                )
                after_release_gripper2world = pos_to_mat(after_release_gripper_7dof[:6])
                self.update_gripper2world(after_release_gripper2world)

            elif "push" in action:
                gripper_pose_list.append(
                    EasyDict({
                        "gripper_pos": gripper_7dof, 
                        "action": action,
                    }),
                )

            elif "grasp" in action:
                higher_gripper_7dof = copy.deepcopy(gripper_7dof)
                higher_gripper_7dof[2] += 150
                higher_gripper_7dof[6] = 1

                if previous_gripper_6dof[2] < higher_gripper_7dof[2]:
                    lift_gripper_7dof = np.zeros(7)
                    lift_gripper_7dof[:6] = copy.deepcopy(previous_gripper_6dof)
                    lift_gripper_7dof[2] = higher_gripper_7dof[2] 
                    lift_gripper_7dof[6] = previous_gripper_status
                    gripper_pose_list.append(
                        EasyDict({
                            "gripper_pos": lift_gripper_7dof, 
                            "action": "move gripper up before grasp",
                        })
                    )
                else:
                    move_gripper_7dof = np.zeros(7)
                    move_gripper_7dof[:6] = higher_gripper_7dof[:6]
                    move_gripper_7dof[2] = previous_gripper_6dof[2]
                    move_gripper_7dof[6] = previous_gripper_status
                    gripper_pose_list.append(
                        EasyDict({
                            "gripper_pos": move_gripper_7dof, 
                            "action": "move gripper down before grasp",
                        }),
                    )

                gripper_pose_list.append(
                    EasyDict({
                        "gripper_pos": higher_gripper_7dof, 
                        "action": "move gripper up before grasp",
                    })
                )
                gripper_pose_list.append(
                    EasyDict({
                        "gripper_pos": gripper_7dof, 
                        "action": "grasp object",
                        "grasp_point": graps_point,
                        "grasped_object": grasped_object
                    })
                )
            elif "lift" in action:
                gripper_pose_list.append(
                    EasyDict({
                        "gripper_pos": gripper_7dof, 
                        "action": action,
                    }),
                )

            else:
                if previous_gripper_6dof[2] < gripper_6dof[2]:
                    lift_gripper_7dof = np.zeros(7)
                    lift_gripper_7dof[:6] = copy.deepcopy(previous_gripper_6dof)
                    lift_gripper_7dof[2] = gripper_6dof[2] 

                    lift_gripper_7dof[6] = previous_gripper_status
                    gripper_pose_list.append(
                        EasyDict({
                            "gripper_pos": lift_gripper_7dof, 
                            "action": "move object vertically",
                        }),
                    )
                else:
                    move_gripper_7dof = np.zeros(7)
                    move_gripper_7dof[:6] = copy.deepcopy(gripper_6dof)
                    move_gripper_7dof[2] = previous_gripper_6dof[2]
                    move_gripper_7dof[6] = previous_gripper_status
                    gripper_pose_list.append(
                        EasyDict({
                            "gripper_pos": move_gripper_7dof, 
                            "action": "move object horizontally",
                        }),
                    )
                gripper_pose_list.append(
                    EasyDict({
                        "gripper_pos": gripper_7dof, 
                        "action": action,
                    }),
                )
        return gripper_pose_list


    def parse_constraint_1(self, step_description: str, grasped_object: dict, target_objects: list) -> dict:
        pattern = r".*?(?P<graspN>grasp_\d+).*?"
        match = re.match(pattern, step_description)
        assert match is not None, f"Invalid matching: {step_description}"
        match_dict = match.groupdict()
        
        target_object = target_objects[0]
        
        gripper_pos_expect = self.get_primitive_in_world(match_dict["graspN"], target_object["id"])
        gripper2world_new = gripper_pos_expect

        self.update_gripper2world(gripper2world_new)

        graps_point = match_dict["graspN"]
        grasped_object = target_object

        return gripper2world_new, graps_point, grasped_object

        
    def parse_constraint_2(self, step_description: str, grasped_object: dict, target_objects: list) -> dict:
        pattern_with_target_primitive = r".*?(?P<planeN>plane_\d+|centroid).*?(?P<direction>positive|negative).*?(?P<distance>\d+(?:\.\d+)?).*?cm.*?to the \[?(?P<planeM>plane_\d+|centroid)\]?.*?of.*?\[(?P<target_name>[^\]]+)\].*?along.*?(?P<axis>x|y|z)-axis.*?"
        match = re.match(pattern_with_target_primitive, step_description)
        
        target_primitive_name = "centroid"

        if match:
            match_dict = match.groupdict()
            target_primitive_name = match_dict.get("planeM", "centroid")
        else:
            pattern = r".*?(?P<planeN>plane_\d+|centroid).*?(?P<direction>positive|negative).*?(?P<distance>\d+(?:\.\d+)?).*?cm.*?of.*?\[(?P<target_name>[^\]]+)\].*?along.*?(?P<axis>x|y|z)-axis.*?"
            match = re.match(pattern, step_description)
            if match is None:
                pattern = r".*?(?P<planeN>plane_\d+|centroid).*?(?P<distance>\d+(?:\.\d+)?).*?cm.*?of.*?\[(?P<target_name>[^\]]+)\].*?along.*?(?P<axis>x|y|z)-axis.*?"
                match = re.match(pattern, step_description)
                match_dict = match.groupdict()
                match_dict["direction"] = "positive"
            else:
                match_dict = match.groupdict()

        assert match is not None, f"Invalid matching: {step_description}"

        target_object = None
        if len(target_objects) == 1:
            target_object = target_objects[0]
        elif len(target_objects) > 1:
            for item in target_objects:
                if item["name"] in match_dict["target_name"] or match_dict["target_name"] in item["name"]:
                    target_object = item
                    break
            if target_object is None:
                target_object = target_objects[0]

        assert target_object is not None, f"No target object found for constraint: {step_description}, target_objects: {target_objects}"

        direction_zip = dict(zip(["positive", "negative"], [1, -1]))
        distance = direction_zip[match_dict["direction"]] * float(match_dict["distance"]) * 10

        gripper2world = self.get_gripper2world() 
        source_object2world = self.get_primitive_in_world("centroid", grasped_object["id"]) 
        
        target_primitive2world = self.get_primitive_in_world(target_primitive_name, target_object["id"])

        source_primitive2world = self.get_primitive_in_world(match_dict["planeN"], grasped_object["id"])

        axis_zip = dict(zip(["x", "y", "z"], [0, 1, 2]))
        trans_mat = np.eye(4)
        trans_mat[axis_zip[match_dict["axis"]], 3] = distance + target_primitive2world[axis_zip[match_dict["axis"]], 3] \
            - source_primitive2world[axis_zip[match_dict["axis"]], 3]

        gripper2object = compute_gripper_in_object(gripper2world, source_object2world)
        source_object2world_new = trans_mat @ source_object2world
        gripper2world_new = revert_gripper_in_object(gripper2object, source_object2world_new)

        self.update_gripper2world(gripper2world_new)
        self.update_primitive(source_object2world_new, grasped_object["id"])
        return gripper2world_new
    
    
    def parse_constraint_3(self, step_description: str, grasped_object: dict, target_objects: list) -> dict:
        pattern = r".*?(?P<planeN>plane_\d+).*?(?P<direction>positive|negative).*?(?P<distance>\d+(?:\.\d+)?).*?cm.*?(?P<planeM>plane_\d+).*?of.*?\[(?P<target_name>[^\]]+)\].*?along.*?"
        match = re.match(pattern, step_description)
        if match is None:
            pattern = r".*?(?P<planeN>plane_\d+).*?(?P<distance>\d+(?:\.\d+)?).*?cm.*?(?P<planeM>plane_\d+).*?of.*?\[(?P<target_name>[^\]]+)\].*?along.*?"
            match = re.match(pattern, step_description)
            match_dict = match.groupdict()
            match_dict["direction"] = "positive"
        else:
            match_dict = match.groupdict()
        assert match is not None, f"Invalid matching: {step_description}"

        if len(target_objects) == 1:
            target_object = target_objects[0]
        elif len(target_objects) > 1:
            for item in target_objects:
                if item["name"] in match_dict["target_name"]:
                    target_object = item
                    break

        direction_zip = dict(zip(["positive", "negative"], [1, -1]))
        distance = direction_zip[match_dict["direction"]] * float(match_dict["distance"]) * 10

        gripper2world = self.get_gripper2world() 
        source_object2world = self.get_primitive_in_world("centroid", grasped_object["id"]) 

        source_primitive2world = self.get_primitive_in_world(match_dict["planeN"], grasped_object["id"])
        target_primitive2world = self.get_primitive_in_world(match_dict["planeM"], target_object["id"])

        trans_mat = np.eye(4)
        trans_mat[:3, :3] = align_vectors(source_primitive2world[:3, 2], target_primitive2world[:3, 2])

        source_object2world_new = source_object2world @ trans_mat
        source_primitive2world_new = source_object2world_new @ np.linalg.inv(source_object2world) @ source_primitive2world

        trans_mat_translation = np.eye(4)
        primitive_distance = np.dot((source_primitive2world_new[:3, 3] - target_primitive2world[:3, 3]), target_primitive2world[:3, 2])
        trans_mat_translation[:3, 3] = (distance - primitive_distance) * target_primitive2world[:3, 2]

        gripper2object = compute_gripper_in_object(gripper2world, source_object2world)

        source_object2world_new = trans_mat_translation @ source_object2world_new
        gripper2world_new = revert_gripper_in_object(gripper2object, source_object2world_new)

        self.update_gripper2world(gripper2world_new)
        self.update_primitive(source_object2world_new, grasped_object["id"])

        return gripper2world_new


    def parse_constraint_4(self, step_description: str, grasped_object: dict, target_objects: list) -> dict:
        pattern = r".*?(?P<planeN>plane_\d+).*?(?P<direction>positive|negative).*?(?P<axis>x|y|z)-axis.*?"
        match = re.match(pattern, step_description)
        match_dict = match.groupdict()
        assert match is not None, f"Invalid matching: {step_description}"

        direction_zip = dict(zip(["positive", "negative"], [1, -1]))

        gripper2world = self.get_gripper2world()
        source_object2world = self.get_primitive_in_world("centroid", grasped_object["id"]) 
        source_primitive2world = self.get_primitive_in_world(match_dict["planeN"], grasped_object["id"])

        axis_zip = dict(zip(["x", "y", "z"], [0, 1, 2]))
        vec = np.zeros(3)
        vec[axis_zip[match_dict["axis"]]] = direction_zip[match_dict["direction"]]
        target_primitive2world = normal_to_rotmat(vec)

        trans_mat = np.eye(4)
        pseudo_source_primitive2world = copy.deepcopy(source_primitive2world)
        pseudo_target_primitive2world = copy.deepcopy(target_primitive2world)
        pseudo_source_primitive2world[:3, 2] = np.array([source_primitive2world[:3, 2][0], source_primitive2world[:3, 2][1], 0])
        pseudo_target_primitive2world[:3, 2] = np.array([target_primitive2world[:3, 2][0], target_primitive2world[:3, 2][1], 0])
        trans_mat[:3, :3] = align_vectors(pseudo_source_primitive2world[:3, 2], pseudo_target_primitive2world[:3, 2])
        gripper2object = compute_gripper_in_object(gripper2world, source_object2world)

        source_object2world_new = source_object2world @ trans_mat

        gripper2world_new = revert_gripper_in_object(gripper2object, source_object2world_new)

        self.update_gripper2world(gripper2world_new)
        self.update_primitive(source_object2world_new, grasped_object["id"])    

        return gripper2world_new
    

    def parse_constraint_5(self, step_description: str, grasped_object: dict, target_objects: list) -> dict:
        pattern = r".*?closer to.*?\[(?P<target_name>[^\]]+)\].*?gap.*?(?P<distance>\d+(?:\.\d+)?).*?cm"
        match = re.match(pattern, step_description)
        match_dict = match.groupdict()
        assert match is not None, f"Invalid matching: {step_description}"

        if len(target_objects) == 1:
            target_object = target_objects[0]
        elif len(target_objects) > 1:
            for item in target_objects:
                if item["name"] in match_dict["target_name"]:
                    target_object = item
                    break
        
        distance = float(match_dict["distance"]) * 10

        source_bbox_plane2world = []
        target_bbox_plane2world = []

        source_bbox_plane2world.append(self.get_primitive_in_world("bbox_plane_x1", grasped_object["id"]))
        source_bbox_plane2world.append(self.get_primitive_in_world("bbox_plane_x2", grasped_object["id"]))
        source_bbox_plane2world.append(self.get_primitive_in_world("bbox_plane_y1", grasped_object["id"]))
        source_bbox_plane2world.append(self.get_primitive_in_world("bbox_plane_y2", grasped_object["id"]))
        source_bbox_plane2world.append(self.get_primitive_in_world("bbox_plane_z1", grasped_object["id"]))
        source_bbox_plane2world.append(self.get_primitive_in_world("bbox_plane_z2", grasped_object["id"]))

        target_bbox_plane2world.append(self.get_primitive_in_world("bbox_plane_x1", target_object["id"]))
        target_bbox_plane2world.append(self.get_primitive_in_world("bbox_plane_x2", target_object["id"]))
        target_bbox_plane2world.append(self.get_primitive_in_world("bbox_plane_y1", target_object["id"]))
        target_bbox_plane2world.append(self.get_primitive_in_world("bbox_plane_y2", target_object["id"]))
        target_bbox_plane2world.append(self.get_primitive_in_world("bbox_plane_z1", target_object["id"]))
        target_bbox_plane2world.append(self.get_primitive_in_world("bbox_plane_z2", target_object["id"]))
        
        gripper2world = self.get_gripper2world()
        source_object2world = self.get_primitive_in_world("centroid", grasped_object["id"]) 

        src_plane_idx, tgt_plane_idx, _ = find_plane_pair(source_bbox_plane2world, target_bbox_plane2world, gripper2world)

        source_align_plane = source_bbox_plane2world[src_plane_idx]
        target_align_plane = target_bbox_plane2world[tgt_plane_idx]

        trans_mat = np.eye(4)
        trans_mat[:3, :3] = align_vectors(source_align_plane[:3, 2], -target_align_plane[:3, 2])

        gripper2object = compute_gripper_in_object(gripper2world, source_object2world)

        source_object2world_new = source_object2world @ trans_mat
        source_align_plane_new = source_object2world_new @ np.linalg.inv(source_object2world) @ source_align_plane

        trans_mat_translation = np.eye(4)
        primitive_distance = np.dot((source_align_plane_new[:3, 3] - target_align_plane[:3, 3]), target_align_plane[:3, 2])
        trans_mat_translation[:3, 3] = (distance - primitive_distance) * target_align_plane[:3, 2]
        source_object2world_new = trans_mat_translation @ source_object2world_new

        gripper2world_new = revert_gripper_in_object(gripper2object, source_object2world_new)

        self.update_gripper2world(gripper2world_new)
        self.update_primitive(source_object2world_new, grasped_object["id"])
    
        return gripper2world_new
    

    def parse_constraint_6(self, step_description: str, grasped_object: dict, target_objects: list) -> dict:
        gripper2world = self.get_gripper2world()
        source_object2world = self.get_primitive_in_world("centroid", grasped_object["id"]) 
        target_object2world = self.get_original_primitive_in_world("centroid", grasped_object["id"])

        gripper2object = compute_gripper_in_object(gripper2world, source_object2world)
        source_object2world_new = target_object2world  
        gripper2world_new = revert_gripper_in_object(gripper2object, source_object2world_new)

        self.update_gripper2world(gripper2world_new)
        self.update_primitive(source_object2world_new, grasped_object["id"])

        return gripper2world_new

    
    def parse_constraint_7(self, step_description: str, grasped_object: dict, target_objects: list) -> dict:
        pattern = r".*?(?P<planeN>plane_\d+).*?(?P<direction>positive|negative).*?(?P<distance>\d+(?:\.\d+)?).*?cm.*?"
        match = re.match(pattern, step_description)
        match_dict = match.groupdict()

        direction_zip = dict(zip(["positive", "negative"], [1, -1]))
        distance = direction_zip[match_dict["direction"]] * float(match_dict["distance"]) * 10

        gripper2world = self.get_gripper2world() 
        source_object2world = self.get_primitive_in_world("centroid", grasped_object["id"]) 

        source_primitive2world = self.get_primitive_in_world(match_dict["planeN"], grasped_object["id"])
        target_primitive2world = self.get_original_primitive_in_world(match_dict["planeN"], grasped_object["id"])

        trans_mat = np.eye(4)
        trans_mat[:3, :3] = align_vectors(source_primitive2world[:3, 2], target_primitive2world[:3, 2])

        source_object2world_new = source_object2world @ trans_mat
        source_primitive2world_new = source_object2world_new @ np.linalg.inv(source_object2world) @ source_primitive2world

        trans_mat_translation = np.eye(4)
        primitive_distance = np.dot((source_primitive2world_new[:3, 3] - target_primitive2world[:3, 3]), target_primitive2world[:3, 2])
        trans_mat_translation[:3, 3] = (distance - primitive_distance) * target_primitive2world[:3, 2]

        gripper2object = compute_gripper_in_object(gripper2world, source_object2world)

        source_object2world_new = trans_mat_translation @ source_object2world_new
        gripper2world_new = revert_gripper_in_object(gripper2object, source_object2world_new)

        self.update_gripper2world(gripper2world_new)
        self.update_primitive(source_object2world_new, grasped_object["id"])

        return gripper2world_new
    

    def parse_constraint_8(self, step_description: str, grasped_object: dict, target_objects: list) -> dict:
        pattern = r".*?(?P<planeN>plane_\d+).*?(?P<direction>up|down).*?(?P<angle>\d+(?:\.\d+)?).*?degree.*?"
        match = re.match(pattern, step_description)
        assert match is not None, f"Invalid matching: {step_description}"
        match_dict = match.groupdict()

        direction_zip = dict(zip(["up", "down"], [1, -1]))
        angle = direction_zip[match_dict["direction"]] * float(match_dict["angle"])

        gripper2world = self.get_gripper2world()
        source_object2world = self.get_primitive_in_world("centroid", grasped_object["id"]) 
        source_primitive2object = self.get_primitive_in_object(match_dict["planeN"], grasped_object["id"])
        source_primitive2world = self.get_primitive_in_world(match_dict["planeN"], grasped_object["id"])

        source_object2world_new = rotate_pose_around_object_y(source_object2world, source_primitive2world[:3, 3], np.deg2rad(-angle))

        gripper2object = compute_gripper_in_object(gripper2world, source_object2world)
        gripper2world_new = revert_gripper_in_object(gripper2object, source_object2world_new)

        self.update_gripper2world(gripper2world_new)
        self.update_primitive(source_object2world_new, grasped_object["id"])
        
        return gripper2world_new


    def parse_constraint_9(self, step_description: str, grasped_object: dict, target_objects: list) -> dict:
        pattern = r".*?(?P<direction>clockwise|counterclockwise).*?(?P<angle>\d+(?:\.\d+)?).*?degree.*?"
        match = re.match(pattern, step_description)
        assert match is not None, f"Invalid matching: {step_description}"
        match_dict = match.groupdict()

        direction_zip = dict(zip(["clockwise", "counterclockwise"], [1, -1]))
        angle = direction_zip[match_dict["direction"]] * float(match_dict["angle"])

        gripper2world = self.get_gripper2world()
        source_object2world = self.get_primitive_in_world("centroid", grasped_object["id"])

        rotation_center = source_object2world[:3, 3]
        
        source_object2world_new = rotate_pose_around_object_z(
            source_object2world, 
            rotation_center, 
            np.deg2rad(angle)
        )

        gripper2object = compute_gripper_in_object(gripper2world, source_object2world)
        gripper2world_new = revert_gripper_in_object(gripper2object, source_object2world_new)


        self.update_gripper2world(gripper2world_new)
        self.update_primitive(source_object2world_new, grasped_object["id"])

        return gripper2world_new
    

    def parse_constraint_10(self, step_description: str, grasped_object: dict, target_objects: list) -> dict:
        pattern = r".*?(?P<direction>positive|negative).*?(?P<distance>\d+(?:\.\d+)?).*?cm.*?of.*?\[(?P<target_name>[^\]]+)\].*?along.*?(?P<axis>x|y|z)-axis.*?"
        match = re.match(pattern, step_description)
        if match is None:
            pattern = r".*?(?P<distance>\d+(?:\.\d+)?).*?cm.*?of.*?\[(?P<target_name>[^\]]+)\].*?along.*?(?P<axis>x|y|z)-axis.*?"
            match = re.match(pattern, step_description)
            match_dict = match.groupdict()
            match_dict["direction"] = "positive"
        else:
            match_dict = match.groupdict()
        assert match is not None, f"Invalid matching: {step_description}"

        if len(target_objects) == 1:
            target_object = target_objects[0]
        elif len(target_objects) > 1:
            for item in target_objects:
                if item["name"] in match_dict["target_name"]:
                    target_object = item
                    break
        
        direction_zip = dict(zip(["positive", "negative"], [1, -1]))
        distance = direction_zip[match_dict["direction"]] * float(match_dict["distance"]) * 10

        gripper2world = self.get_gripper2world() 
        source_object2world = self.get_primitive_in_world("centroid", grasped_object["id"]) 
        target_object2world = self.get_primitive_in_world("centroid", target_object["id"])

        axis_zip = dict(zip(["x", "y", "z"], [0, 1, 2]))
        trans_mat = np.eye(4)
        trans_mat[axis_zip[match_dict["axis"]], 3] = distance + target_object2world[axis_zip[match_dict["axis"]], 3] \
            - source_object2world[axis_zip[match_dict["axis"]], 3]

        gripper2object = compute_gripper_in_object(gripper2world, source_object2world)
        source_object2world_new = trans_mat @ source_object2world
        gripper2world_new = revert_gripper_in_object(gripper2object, source_object2world_new)

        self.update_gripper2world(gripper2world_new)
        self.update_primitive(source_object2world_new, grasped_object["id"])
        return gripper2world_new


    def parse_constraint_11(self, step_description: str, grasped_object: dict, target_objects: list) -> dict:
        pattern = r".*?at the \[?(?P<contact_point>contact_point_\d+)\]?.*?of.*?\[(?P<target_name>[^\]]+)\].*?rotated \[?(?P<angle>-?\d+(?:\.\d+)?)\]?.*?degrees.*?to \[?(?P<action>close|open)\]?.*?"
        match = re.match(pattern, step_description)
        assert match is not None, f"Invalid matching: {step_description}"
        match_dict = match.groupdict()

        if len(target_objects) == 1:
            target_object = target_objects[0]
        elif len(target_objects) > 1:
            for item in target_objects:
                if item["name"] in match_dict["target_name"]:
                    target_object = item
                    break
        
        angle_deg = -float(match_dict["angle"])
        action = match_dict["action"]
        
        contact_point_world = self.get_primitive_in_world(match_dict["contact_point"], target_object["id"])
        
        hinge_world = self.get_primitive_in_world("hinge_plane", target_object["id"])
        hinge_axis = hinge_world[:3, 2]
        hinge_pivot = hinge_world[:3, 3]

        num_middle_waypoints = 5
        middle_waypoints = []
        
        T_pivot_to_origin = np.eye(4)
        T_pivot_to_origin[:3, 3] = -hinge_pivot
        T_pivot_back = np.eye(4)
        T_pivot_back[:3, 3] = hinge_pivot
        
        angle_rad_total = np.deg2rad(angle_deg)

        for i in range(1, num_middle_waypoints + 1):
            current_angle_rad = angle_rad_total * (i / num_middle_waypoints)
            
            rot_mat = R.from_rotvec(hinge_axis * current_angle_rad).as_matrix()
            
            T_rot = np.eye(4)
            T_rot[:3, :3] = rot_mat
            
            transform_mat = T_pivot_back @ T_rot @ T_pivot_to_origin
            
            contact_point_world_new = transform_mat @ contact_point_world
            
            middle_waypoints.append(contact_point_world_new)
        
        if middle_waypoints:
            gripper2world_new = middle_waypoints[-1]
        else:
            gripper2world_new = contact_point_world

        self.update_gripper2world(gripper2world_new)
        
        return middle_waypoints


    def parse_constraint_12(self, step_description: str, grasped_object: dict, target_objects: list) -> dict:
        pattern = r".*?at the \[?(?P<contact_point>contact_point_\d+)\]?.*?of.*?\[(?P<target_name>[^\]]+)\].*?"
        match = re.match(pattern, step_description)
        assert match is not None, f"Invalid matching: {step_description}"
        match_dict = match.groupdict()

        if len(target_objects) == 1:
            target_object = target_objects[0]
        elif len(target_objects) > 1:
            for item in target_objects:
                if item["name"] in match_dict["target_name"]:
                    target_object = item
                    break
        
        contact_point_world = self.get_primitive_in_world(match_dict["contact_point"], target_object["id"])
        
        gripper2world_new = contact_point_world
        
        self.update_gripper2world(gripper2world_new)
        
        return gripper2world_new

    def parse_constraint_13(self, step_description: str, grasped_object: dict, target_objects: list) -> dict:
        pattern = r".*?\[?(?P<graspN>grasp_\d+)\]?.*?of.*?\[(?P<target_name>[^\]]+)\].*?same position.*?\[?(?P<graspM>grasp_\d+)\]?.*?"
        match = re.match(pattern, step_description)
        assert match is not None, f"Invalid matching: {step_description}"
        match_dict = match.groupdict()

        if len(target_objects) == 1:
            target_object = target_objects[0]
        elif len(target_objects) > 1:
            for item in target_objects:
                if item["name"] in match_dict["target_name"]:
                    target_object = item
                    break
        
        gripper2world = self.get_gripper2world()
        target_object2world = self.get_primitive_in_world("centroid", target_object["id"])
        
        graspN_world = self.get_primitive_in_world(match_dict["graspN"], target_object["id"])
        
        graspM_world = self.get_primitive_in_world(match_dict["graspM"], target_object["id"])
        
        translation = graspM_world[:3, 3] - graspN_world[:3, 3]
        
        trans_mat = np.eye(4)
        trans_mat[:3, 3] = translation

        graspN_world_new = graspM_world
        
        gripper2world_new = graspM_world
        
        self.update_gripper2world(gripper2world_new)
        if self.update_primitive_in_camera_directly is not None:
            self.update_primitive_in_camera_directly(match_dict["graspN"], target_object["id"], graspN_world_new)
        
        return gripper2world_new

    def parse_constraint_14(self, step_description: str, grasped_object: dict, target_objects: list) -> dict:
        pattern = r".*?\[?(?P<graspN>grasp_\d+)\]?.*?of.*?(?P<direction>positive|negative).*?(?P<distance>\d+(?:\.\d+)?).*?cm.*?of.*?\[(?P<target_name>[^\]]+)\].*?along.*?(?P<axis>x|y|z)-axis.*?"
        match = re.match(pattern, step_description)
        if match is None:
            pattern = r".*?\[?(?P<graspN>grasp_\d+)\]?.*?of.*?(?P<distance>\d+(?:\.\d+)?).*?cm.*?of.*?\[(?P<target_name>[^\]]+)\].*?along.*?(?P<axis>x|y|z)-axis.*?"
            match = re.match(pattern, step_description)
            match_dict = match.groupdict()
            match_dict["direction"] = "positive"
        else:
            match_dict = match.groupdict()
        assert match is not None, f"Invalid matching: {step_description}"

        if len(target_objects) == 1:
            target_object = target_objects[0]
        elif len(target_objects) > 1:
            for item in target_objects:
                if item["name"] in match_dict["target_name"]:
                    target_object = item
                    break
        
        direction_zip = dict(zip(["positive", "negative"], [1, -1]))
        distance = direction_zip[match_dict["direction"]] * float(match_dict["distance"]) * 10

        gripper2world = self.get_gripper2world()
        graspN_world = self.get_primitive_in_world(match_dict["graspN"], target_object["id"])

        axis_zip = dict(zip(["x", "y", "z"], [0, 1, 2]))
        trans_mat = np.eye(4)
        trans_mat[axis_zip[match_dict["axis"]], 3] = distance 

        gripper2world_new = trans_mat @ gripper2world

        graspN_world_new = trans_mat @ graspN_world

        self.update_gripper2world(gripper2world_new)

        if self.update_primitive_in_camera_directly is not None:
            self.update_primitive_in_camera_directly(match_dict["graspN"], target_object["id"], graspN_world_new)
        
        
        return gripper2world_new
