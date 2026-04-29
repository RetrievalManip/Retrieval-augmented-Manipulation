import open3d as o3d
import numpy as np
import torch

def extract_pcd(
        xyz, rgbs,
        down_sample_voxel=None,
        outlier_removal=False,
        std_ratio: float = 1.0,
) -> o3d.geometry.PointCloud:
    points = xyz
    colors = rgbs

    if torch.is_tensor(points):
        points = points.cpu().numpy()
    if torch.is_tensor(colors):
        colors = colors.cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if down_sample_voxel is not None:
        pcd = pcd.voxel_down_sample(voxel_size=down_sample_voxel)

    if outlier_removal:
        cl, ind = pcd.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=std_ratio
        )
        pcd = pcd.select_by_index(ind)

    return pcd 



def get_pcd_centroid(xyz: np.ndarray) -> np.ndarray:
    if xyz.shape[0] == 0:
        return np.zeros((3,))

    centroid = np.mean(xyz, axis=0)
    return centroid


def get_pcd_bbx(xyz: np.ndarray) -> np.ndarray:
    if xyz.shape[0] == 0:
        return np.zeros((6,))

    xmin = np.min(xyz[:, 0])
    ymin = np.min(xyz[:, 1])
    zmin = np.min(xyz[:, 2])
    xmax = np.max(xyz[:, 0])
    ymax = np.max(xyz[:, 1])
    zmax = np.max(xyz[:, 2])

    return np.array([xmin, ymin, zmin, xmax, ymax, zmax])


def get_pcd_bbx_centeroid(xyz: np.ndarray) -> np.ndarray:
    if xyz.shape[0] == 0:
        return np.zeros((3,))

    bbox = get_pcd_bbx(xyz)
    center = np.mean(bbox.reshape(2, 3), axis=0)

    return center


def get_pcd_bounding_radius(xyz: np.ndarray) -> float:
    if xyz.shape[0] == 0:
        return 0.0

    centroid = get_pcd_centroid(xyz)
    distances = np.linalg.norm(xyz - centroid, axis=1)
    radius = np.max(distances)

    return radius

def get_mask_centroid(mask: np.ndarray) -> np.ndarray:
    
    if mask.shape[0] == 0:
        return np.zeros((2,))

    y, x = np.where(mask > 0)
    centroid = np.mean(np.array([x, y]), axis=1)

    return centroid

def angle_clip(angle):
    if angle < -180.0:
        angle += 360.0
    if angle > 180.0:
        angle -= 360.0
    return angle
