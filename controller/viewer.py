import sys
import os
cur_path = os.path.dirname(os.path.abspath(__file__))

from typing import List
import numpy as np
import math

import viser
import viser.transforms as tf
import imageio

from controller.base_camera import BaseCamera
from .base_viewer import BaseViewer


def euler_from_matrix(R):
    pitch = math.asin(-R[2, 0])
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw

def angle2rotation(x, y, z):
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(x), -math.sin(x)],
                   [0, math.sin(x), math.cos(x)]])
    Ry = np.array([[math.cos(y), 0, math.sin(y)],
                   [0, 1, 0],
                   [-math.sin(y), 0, math.cos(y)]])
    Rz = np.array([[math.cos(z), -math.sin(z), 0],
                   [math.sin(z), math.cos(z), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx
    return R

def gripper2base(rx, ry, rz, x, y, z):
    thetaX = rx / 180.0 * math.pi
    thetaY = ry / 180.0 * math.pi
    thetaZ = rz / 180.0 * math.pi

    R_gripper2base = angle2rotation(thetaX, thetaY, thetaZ)
    T_gripper2base = np.array([[x], [y], [z]])
    return R_gripper2base, T_gripper2base

class Viewer(BaseViewer):
    def __init__(self):
        super().__init__()
        self.ip = "localhost:8099"
        self.server = viser.ViserServer(ip=self.ip)
        self.server.scene.add_frame(
            "/frames",
            wxyz=tf.SO3.exp(np.array([0.0, 0.0, 0.0])).wxyz,
            position=(0, 0, 0),
            show_axes=False,
        )
        self.current_frame_data = None
        self.point_cloud_handle = None
        self.cost_map_handle = None

        self.trajectory_segments = []
        self.trajectory_handle = None
        self.grasp_segments = []

        self.scale = 0.01

        self.trajectory_executed = False

        self.cancelled = False

        self.execution_error = None

        self.start_button = self.server.gui.add_button(
            label="Start",
            disabled=False,
            visible=True,
            hint="Click to execute full trajectory",
            color="green",
            order=1.0,
        )
        self.start_button.on_click(self._on_start_button_clicked)

        self.cancel_button = self.server.gui.add_button(
            label="Cancel",
            disabled=False,
            visible=True,
            hint="Click to cancel and exit the process",
            color="red",
            order=2.0,
        )
        self.cancel_button.on_click(self._on_cancel_button_clicked)
        self.traj_gen = None
        self.capture_system = None
        self.robot = None

    def set_trajectory_generator(self, traj_gen, capture_system=None, robot=None):
        self.traj_gen = traj_gen
        self.capture_system = capture_system
        self.robot = robot

    def _on_start_button_clicked(self, event):
        if self.traj_gen is not None:
            print("Start button clicked, executing full trajectory.")
            try:
                self.traj_gen.executeFullTrajectory_simple()
                if self.robot is not None:
                    self.robot.reset()
                self.execution_error = None
            except Exception as e:
                print(f"Trajectory execution failed: {e}")
                self.execution_error = str(e)
                try:
                    if self.robot is not None:
                        self.robot.reset()
                except Exception:
                    pass

            self.trajectory_executed = True
        else:
            print("Trajectory generator not set!")

    def _on_cancel_button_clicked(self, event):
        print("Cancel button clicked, setting cancelled flag.")
        self.cancelled = True
        self.trajectory_executed = True

    def add_grasp(self, RT_target_in_base, grasp_width):
        R_target_in_base = RT_target_in_base[0:3, 0:3]
        T_target_in_base = RT_target_in_base[0:3, 3]

        R_y180 = np.array([
            [-1, 0,  0],
            [ 0, 1,  0],
            [ 0, 0, -1]
        ])
        R_target_adjusted = R_target_in_base @ R_y180

        T_scaled = T_target_in_base * self.scale

        finger_length = grasp_width * 1.5 * 0.5
        extra_length = finger_length * 1.0

        left_tip_local = np.array([ grasp_width / 2, 0, 0 ])
        right_tip_local = np.array([-grasp_width / 2, 0, 0 ])

        left_finger_top_local = left_tip_local + np.array([0, 0, finger_length])
        right_finger_top_local = right_tip_local + np.array([0, 0, finger_length])

        connection_top_mid = (left_finger_top_local + right_finger_top_local) / 2

        extra_line_local = connection_top_mid + np.array([0, 0, extra_length])

        local_segments = np.array([
            [left_tip_local,       left_finger_top_local],
            [right_tip_local,      right_finger_top_local],
            [left_finger_top_local, right_finger_top_local],
            [connection_top_mid,    extra_line_local]
        ])

        roll, pitch, yaw = euler_from_matrix(R_target_adjusted)
        print("Adjusted Grasp Euler angles (degrees):",
            f"roll: {math.degrees(roll):.2f}, pitch: {math.degrees(pitch):.2f}, yaw: {math.degrees(yaw):.2f}")

        transformed_segments = []
        for seg in local_segments:
            seg_transformed = []
            for point in seg:
                point_world = R_target_adjusted @ point + T_scaled
                seg_transformed.append(point_world)
            transformed_segments.append(seg_transformed)
        transformed_segments = np.array(transformed_segments)

        colors = np.tile(np.array([[[255, 0, 0]]]), (transformed_segments.shape[0], 2, 1))

        self.grasp_segments.append((transformed_segments, colors, 3.0))



    def add_pcd(self, xyzs: np.ndarray, rgbs: np.ndarray):
        if hasattr(xyzs, "cpu"):
            xyzs = xyzs.cpu().numpy()
        if hasattr(rgbs, "cpu"):
            rgbs = rgbs.cpu().numpy()

        xyzs_scaled = xyzs * self.scale

        if self.current_frame_data is None:
            self.current_frame_data = {"position": xyzs_scaled, "color": rgbs}
        else:
            self.current_frame_data["position"] = np.concatenate(
                [self.current_frame_data["position"], xyzs_scaled], axis=0
            )
            self.current_frame_data["color"] = np.concatenate(
                [self.current_frame_data["color"], rgbs], axis=0
            )

    def clear_pcd(self):
        points = np.zeros((1, 3), dtype=np.float32)
        colors = np.zeros((1, 3), dtype=np.float32)
        if self.current_frame_data is None:
            self.current_frame_data = {"position": points, "color": colors}
        else:
            self.current_frame_data["position"] = points
            self.current_frame_data["color"] = colors

    def add_camera(self, camera:BaseCamera, trans_mat:np.ndarray=np.eye(4)):
        rgb = camera.rgb_image
        fx = camera.fx
        type = camera.type

        c2w = trans_mat @ camera.c2w
        T = c2w[:3, 3]
        R = c2w[:3, :3]

        position = T * self.scale
        
        wxyz = tf.SO3.from_matrix(R).wxyz
        
        focal_length = fx
        fov = 2 * np.arctan2(camera.height / 2, focal_length)
        aspect = camera.width / camera.height

        if camera.type == "external":
            if not hasattr(self, "rgb_image_handle_extern") or self.rgb_image_handle_extern is None:
                self.rgb_image_handle_extern = self.server.scene.add_camera_frustum(
                    name="/frames/rgb_display_extern",
                    fov=fov,
                    aspect=aspect,
                    scale=1,
                    image=rgb,
                    wxyz=wxyz,
                    position=position,
                )
            else:
                self.rgb_image_handle_extern.image = rgb

            if not hasattr(self, "rgb_axes_handle_extern") or self.rgb_axes_handle_extern is None:
                self.rgb_axes_handle_extern = self.server.scene.add_frame(
                name="/frames/rgb_display_extern/axes",
                axes_length=0.5,
                axes_radius=0.05,
            )
            else:
                pass

        elif camera.type == "hand": 
            if not hasattr(self, "rgb_image_handle_hand") or self.rgb_image_handle_hand is None:
                self.rgb_image_handle_hand = self.server.scene.add_camera_frustum(
                    name="/frames/rgb_display_hand",
                    fov=fov,
                    aspect=aspect,
                    scale=0.5,
                    image=rgb,
                    wxyz=wxyz,
                    position=position,
                )
            else:
                self.rgb_image_handle_hand.image = rgb
                self.rgb_image_handle_hand.wxyz = wxyz
                self.rgb_image_handle_hand.position = position

        else:
            raise ValueError(f"Unsupported camera type: {camera.type}")

        pass

    def add_bbox(self, bbox: List):
        xmin, ymin, zmin, xmax, ymax, zmax = bbox


    def add_axis(self, pos):
        if len(pos) != 6:
            raise ValueError("pos must contain 6 elements: [x, y, z, rx, ry, rz]")

        x, y, z, rx, ry, rz = pos

        scaled_position = (x * 0.1, y * 0.1, z * 0.1)

        rx_rad = np.deg2rad(rx)
        ry_rad = np.deg2rad(ry)
        rz_rad = np.deg2rad(rz)

        cy = np.cos(rz_rad * 0.5)
        sy = np.sin(rz_rad * 0.5)
        cp = np.cos(ry_rad * 0.5)
        sp = np.sin(ry_rad * 0.5)
        cr = np.cos(rx_rad * 0.5)
        sr = np.sin(rx_rad * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x_q = sr * cp * cy - cr * sp * sy
        y_q = cr * sp * cy + sr * cp * sy
        z_q = cr * cp * sy - sr * sp * cy

        quaternion = np.array([w, x_q, y_q, z_q])
        
        axis_handle = self.server.scene.add_frame(
            name=f"/frames/axis_{int(x)}_{int(y)}_{int(z)}",
            position=scaled_position,
            wxyz=quaternion,
            axes_length=0.5,
            axes_radius=0.05,
        )

        return axis_handle

    def update_frame(self):
        if self.current_frame_data is None:
            print("No point cloud data detected. Call add_pcd first to provide point cloud data.")
            return

        if hasattr(self.current_frame_data["position"], "cpu"):
            points = self.current_frame_data["position"].cpu().numpy()
        else:
            points = self.current_frame_data["position"]

        if hasattr(self.current_frame_data["color"], "cpu"):
            colors = self.current_frame_data["color"].cpu().numpy()
        else:
            colors = self.current_frame_data["color"]

        if self.point_cloud_handle is None:
            self.point_cloud_handle = self.server.scene.add_point_cloud(
                name="/frames/live_point_cloud",
                points=points,
                colors=colors,
                point_size=0.01,
                point_shape="rounded",
            )
        else:
            self.point_cloud_handle.points = points
            self.point_cloud_handle.colors = colors
        if not hasattr(self, "rgb_axes_handle") or self.rgb_axes_handle is None:
            self.rgb_axes_handle = self.server.scene.add_frame(
                name="/frames/rgb_display/axes",
                axes_length=0.5,
                axes_radius=0.05,
            )
        
        if self.trajectory_segments:
            all_segments = np.concatenate([item[0] for item in self.trajectory_segments], axis=0)
            all_colors = np.concatenate([item[1] for item in self.trajectory_segments], axis=0)
            lw = self.trajectory_segments[0][2]
            if self.trajectory_handle is None:
                self.trajectory_handle = self.server.scene.add_line_segments(
                    name="/trajectory",
                    points=all_segments,
                    colors=all_colors,
                    line_width=lw,
                    wxyz=(1.0, 0.0, 0.0, 0.0),
                    position=(0.0, 0.0, 0.0),
                    visible=True,
                )
            else:
                self.trajectory_handle.points = all_segments
                self.trajectory_handle.colors = all_colors
        
        if self.grasp_segments:
            grasp_points = np.concatenate([item[0] for item in self.grasp_segments], axis=0)
            grasp_colors = np.concatenate([item[1] for item in self.grasp_segments], axis=0)
            if not hasattr(self, "grasp_handle") or self.grasp_handle is None:
                self.grasp_handle = self.server.scene.add_line_segments(
                    name="/frames/all_grasps",
                    points=grasp_points,
                    colors=grasp_colors,
                    line_width=self.grasp_segments[0][2]
                )
            else:
                self.grasp_handle.points = grasp_points
                self.grasp_handle.colors = grasp_colors

        self.server.flush()

        self.clear_pcd()

    def add_trajectory(
        self,
        trajectory: np.ndarray,
        color: tuple[int, int, int] = (0, 0, 255),
        line_width: float = 2.0,
    ) -> None:
        if trajectory.shape[0] < 2:
            print("A trajectory needs at least two points to form line segments.")
            return

        scaled_trajectory = trajectory * self.scale
        segments = np.stack([scaled_trajectory[:-1], scaled_trajectory[1:]], axis=1)
        colors = np.tile(np.array(color, dtype=np.uint8), (segments.shape[0], 2, 1))
        self.trajectory_segments.append((segments, colors, line_width))

    def clear_trajectory(self):
        self.trajectory_segments = []
        self.grasp_segments = []

        if self.trajectory_handle is not None:
            try:
                self.trajectory_handle.remove()
            except:
                pass
            self.trajectory_handle = None

        if hasattr(self, "grasp_handle") and self.grasp_handle is not None:
            try:
                self.grasp_handle.remove()
            except:
                pass
            self.grasp_handle = None

        print("Viewer: Cleared all trajectory and grasp visualizations.")
