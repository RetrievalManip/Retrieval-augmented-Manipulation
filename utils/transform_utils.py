import numpy as np
import torch
import math
from scipy.spatial.transform import Rotation as R


def pos_to_mat(pose):
    if not isinstance(pose, np.ndarray):
        pose = np.array(pose, dtype=np.float32)
    assert pose.shape == (6,), "Pose must be of shape (6,)"

    translation = pose[:3]
    rotation = pose[3:]

    rotation_matrix = angle2rotation(rotation[0], rotation[1], rotation[2])

    mat = np.eye(4)
    mat[:3, :3] = rotation_matrix
    mat[:3, 3] = translation

    return mat

def mat_to_pos(mat):
    assert mat.shape == (4, 4), "Matrix must be of shape (4, 4)"

    print("=====>debug mat_to_pos input ****: ", mat)
    translation = mat[:3, 3]
    rotation_matrix = mat[:3, :3]

    rotation = rotation2angle(rotation_matrix)

    print("=====>debug mat_to_pos output ****: ", np.concatenate((translation, rotation)))

    return np.concatenate((translation, rotation))

def angle2rotation(rx, ry, rz):
    x = rx / 180.0 * math.pi
    y = ry / 180.0 * math.pi
    z = rz / 180.0 * math.pi

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

def rotation2angle(R):
    assert R.shape == (3, 3), "Rotation matrix must be of shape (3, 3)"

    sy = math.sqrt(R[0, 0] * R[0, 0] +  R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x * 180. / np.pi,
                     y * 180. / np.pi,
                     z * 180. / np.pi])

def quaternion2euler(quaternion):
    r = R.from_quat(quaternion)
    euler = r.as_euler('xyz', degrees=True)
    return euler
 
def euler2quaternion(euler):
    r = R.from_euler('xyz', euler, degrees=True)
    quaternion = r.as_quat()
    return quaternion

def normalize_angle_continuous(target_angle: float, current_angle: float) -> float:
    diff = target_angle - current_angle
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    return current_angle + diff


def normalize_pose_continuous(target_pose, current_pose):
    if not isinstance(target_pose, np.ndarray):
        target_pose = np.array(target_pose, dtype=np.float64)
    if not isinstance(current_pose, np.ndarray):
        current_pose = np.array(current_pose, dtype=np.float64)

    result = target_pose.copy()

    for i in range(3, 6):
        if i < len(target_pose) and i < len(current_pose):
            result[i] = normalize_angle_continuous(float(target_pose[i]), float(current_pose[i]))

    return result