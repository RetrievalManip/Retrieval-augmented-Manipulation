from .base_arm import BaseArm
import time
import logging
import os
import sys
import json
import numpy as np

from utils.config import get_robot_config
from utils.transform_utils import mat_to_pos, pos_to_mat

try:
    import Robot
except ImportError:
    Robot = None

def is_zero_list(lst):
    epsilon = 1e-6
    return all(abs(x) < epsilon for x in lst)


class RobotArm(BaseArm):
    def __init__(self, config_file=None):
        super().__init__()

        robot_config = get_robot_config()

        self.reset_pos = robot_config.reset_pos
        self.handidx = 6
        self.ditch = 15
        self.gripper_idx = robot_config.gripper.idx
        self.gripper_offset = robot_config.gripper.offset
        self.ip = robot_config.ip
        self.close_wid = robot_config.gripper.close_width
        self.workspace = robot_config.workspace

        logging.info("==> Robot arm configuration loaded from unified config.yaml")

        if Robot is None:
            raise ImportError(
                "Fairino Robot SDK module 'Robot' is not installed. "
                "Install the vendor SDK before using RobotArm."
            )

        self.robot = Robot.RPC(self.ip)

        time.sleep(1.0)

        err, ret = self.get_tool_pose()
        self.robot.ActGripper(1, 1)

        if is_zero_list(ret):
            logging.error("Robot not connected")
            raise ValueError("Robot not connected")

        ret = self.robot.ActGripper(self.gripper_idx, 0)
        ret = self.robot.ActGripper(self.gripper_idx, 1)
        self.gripper_status_value = 1
        logging.info("==> Robot relink complete")
        time.sleep(1)

    def reset(self):

        self.robot.ActGripper(1, 1)
        self.act_gripper(0)
        err = self.robot.MoveL(self.reset_pos, tool=0, user=0, vel=60)
        if err != 0:
            print("MoveCart error: %d" % err)
        return 0

    def moveto(self, pos_rot):

        err = self.robot.MoveL(pos_rot, tool=0, user=0, vel=60)
        if err != 0:
            logging.error("MoveCart error: %d", err)
            raise ValueError("MoveCart error: %d" % err)
        
        return 0
    
    def move_arc(self, via_pose, target_pose, tool=0, user=0, vel=20, blendR=-1.0):
        err = self.robot.MoveC(
            desc_pos_p=via_pose, tool_p=tool, user_p=user,
            desc_pos_t=target_pose, tool_t=tool, user_t=user,
            vel_p=vel, vel_t=vel, blendR=blendR
        )
        if err != 0:
            logging.error("MoveC error: %d", err)
            raise ValueError("MoveC error: %d" % err)
        return 0

    def tip_moveto(self, pos_rot):
        tip2tool_dist = 210 - self.gripper_offset
        tip2tool = np.eye(4)
        tip2tool[2, 3] = tip2tool_dist

        tip2w = pos_to_mat(pos_rot)
        tool2tip = np.linalg.inv(tip2tool)
        tool2w = tip2w @ tool2tip

        tool_pos_rot = mat_to_pos(tool2w)

        self.moveto(tool_pos_rot)
    
    def tip_move_arc(self, via_pose, target_pose, tool=0, user=0, vel=20, blendR=-1.0):
        via_tool_pose = self.tip2tool(via_pose)
        target_tool_pose = self.tip2tool(target_pose)
        return self.move_arc(
            via_tool_pose, target_tool_pose,
            tool=tool, user=user, vel=vel, blendR=blendR
        )

    def camera_moveto(self, pos_rot, hand_cam_c2w=np.eye(4)):
        cam2w_mat = pos_to_mat(pos_rot)
        tool2cam_mat = np.linalg.inv(hand_cam_c2w) 
        tool2w_mat = cam2w_mat @ tool2cam_mat
        tool_pos_rot = mat_to_pos(tool2w_mat)
        self.moveto(tool_pos_rot)

    def restore_config(self, file):
        config_paras = {
            "reset_pos": self.reset_pos,
            "handidx": self.handidx,
            "ditch": self.ditch,
            "gripper_idx": self.gripper_idx,
            "ip": self.ip,
            "workspace": self.workspace,
        }

        with open(file, 'w') as f:
            json.dump(config_paras, f, indent=4)
        logging.info("Configuration saved to %s", file)

    def read_config(self, file):
        with open(file, 'r') as f:
            config_paras = json.load(f)
        self.reset_pos = config_paras["reset_pos"]
        self.handidx = config_paras["handidx"]
        self.ditch = config_paras["ditch"]
        self.gripper_idx = config_paras["gripper_idx"]
        self.ip = config_paras["ip"]
        self.workspace = config_paras["workspace"]
    
    def get_tool_pose(self, flag=1):
        ret = self.robot.GetActualTCPPose(flag)
        time.sleep(0.5)
        if ret[0] != 0:
            logging.error("GetActualTCPPose error: %d", ret[0])
            return -1, None
        if is_zero_list(ret[1]):
            count = 0
            while count < 10:
                time.sleep(0.1)
                ret = self.robot.GetActualTCPPose(flag)
                if not is_zero_list(ret[1]):
                    break
                count += 1
                print(count)
            raise ValueError("GetActualTCPPose error: %d", ret[0])
        return ret[0], ret[1]
    
    def get_pose_and_joint(self, flag=1):
        err, tcp_pose = self.robot.GetActualTCPPose(flag)
        gripper_status = self.gripper_status_value
        err, joint_pos = self.robot.GetActualJointPosDegree(flag)
        current_time = time.time()
        return tcp_pose, gripper_status, joint_pos, current_time


    def act_gripper(self, status):


        if status == 0:
            self.robot.ActGripper(1, 1)
            error = self.robot.MoveGripper(1, 100, 60, 60, 3000, 0, 0, 0, 0, 0)
            self.gripper_status_value = 0
            if error != 0:
                logging.error("MoveGripper error: %d", error)
                return -1
        elif status == 1:
            self.robot.ActGripper(1, 1)
            error = self.robot.MoveGripper(1, self.close_wid, 60, 60, 3000, 0, 0, 0, 0, 0)
            self.gripper_status_value = 1
            if error != 0:
                logging.error("MoveGripper error: %d", error)
                return -1
        else:
            logging.error("Invalid gripper status: %d", status)
            return -1

        ret = self.robot.GetGripperMotionDone()
        return ret
    
    def get_workspace(self):
        return self.workspace
        
    def tool2tip(self, tool_pose):
        tip2tool_dist = 210 - self.gripper_offset
        tip2tool = np.eye(4)
        tip2tool[2, 3] = -tip2tool_dist

        tip2w = pos_to_mat(tool_pose)
        tool2tip = np.linalg.inv(tip2tool) 
        tool2w = tip2w @ tool2tip 
       
        tool_pos = mat_to_pos(tool2w)
        return tool_pos
    
    def tip2tool(self, tip_pose):
        tip2tool_dist = 210 - self.gripper_offset
        tip2tool = np.eye(4)
        tip2tool[2, 3] = tip2tool_dist

        tip2w = pos_to_mat(tip_pose)
        tool2tip = np.linalg.inv(tip2tool) 
        tool2w = tip2w @ tool2tip 
       
        tool_pos = mat_to_pos(tool2w)
        return tool_pos
