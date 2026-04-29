import matplotlib.pyplot as plt
import numpy as np

from align import estimateSimilarityTransform


def grasp_point_coord(template_grasp_point, template_scale, ref_points, matching_matrix, query_pts):
    grasp_point_org = template_grasp_point / np.linalg.norm(template_scale)
    ref_pts_in_grasp_frame = ref_points - grasp_point_org[np.newaxis, :]
    query_pts_in_grasp_frame = matching_matrix @ ref_pts_in_grasp_frame
    _, _, grasp_T, _, _, _ = estimateSimilarityTransform(query_pts_in_grasp_frame, query_pts)

    return grasp_T


def find_grasp_point_with_ram(template_grasp_point, template_scale, ref_points, matching_matrix, query_pts):
    return grasp_point_coord(template_grasp_point, template_scale, ref_points, matching_matrix, query_pts)


def visualize_grasp_point_on_image(rgb_img, pixel, star_size=15, color='gold'):
    h, w = rgb_img.shape[:2]
    fig, ax = plt.subplots(figsize=(w / 80, h / 80), dpi=80)
    ax.imshow(rgb_img)
    ax.scatter([pixel[0]], [pixel[1]], marker='*', s=star_size ** 2, c=color, edgecolors='black', linewidths=2, zorder=10)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.canvas.draw()
    vis_img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    vis_img = vis_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, 1:]
    plt.close(fig)
    return vis_img


def euler_to_matrix(roll, pitch, yaw):
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1],
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)],
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)],
    ])
    return Rz @ Ry @ Rx


def get_gripper_lines(
    center,
    roll,
    pitch,
    yaw,
    object_sRT=np.eye(4),
    finger_length=0.10,
    finger_width=0.06,
    approaching_depth=0.03,
):
    object_scale = np.cbrt(np.linalg.det(object_sRT[:3, :3]))
    finger_length = finger_length / object_scale
    finger_width = finger_width / object_scale
    approaching_depth = approaching_depth / object_scale

    T = np.eye(4)
    T[:3, :3] = euler_to_matrix(roll, pitch, yaw)
    T[:3, 3] = np.array(center)
    T = object_sRT @ T

    pts = np.array([
        [0, 0, approaching_depth - 2 * finger_length],
        [0, 0, approaching_depth - finger_length],
        [-finger_width / 2, 0, approaching_depth - finger_length],
        [finger_width / 2, 0, approaching_depth - finger_length],
        [-finger_width / 2, 0, approaching_depth],
        [finger_width / 2, 0, approaching_depth],
    ])

    pts_homo = np.concatenate([pts, np.ones((6, 1))], axis=1)
    pts_world = (T @ pts_homo.T).T[:, :3]

    lines = [
        (pts_world[0], pts_world[1]),
        (pts_world[2], pts_world[3]),
        (pts_world[2], pts_world[4]),
        (pts_world[3], pts_world[5]),
    ]
    return lines, T
