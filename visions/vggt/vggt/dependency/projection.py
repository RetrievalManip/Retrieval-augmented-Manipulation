
import torch
import numpy as np
from .distortion import apply_distortion


def img_from_cam_np(
    intrinsics: np.ndarray, points_cam: np.ndarray, extra_params: np.ndarray | None = None, default: float = 0.0
) -> np.ndarray:
    z = points_cam[:, 2:3, :]
    points_cam_norm = points_cam / z
    uv = points_cam_norm[:, :2, :]

    if extra_params is not None:
        uu, vv = apply_distortion(extra_params, uv[:, 0], uv[:, 1])
        uv = np.stack([uu, vv], axis=1)

    ones = np.ones_like(uv[:, :1, :])
    points_cam_h = np.concatenate([uv, ones], axis=1)

    points2D_h = np.einsum("bij,bjk->bik", intrinsics, points_cam_h)
    points2D = np.nan_to_num(points2D_h[:, :2, :], nan=default)

    return points2D.transpose(0, 2, 1)


def project_3D_points_np(
    points3D: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray | None = None,
    extra_params: np.ndarray | None = None,
    *,
    default: float = 0.0,
    only_points_cam: bool = False,
):
    N = points3D.shape[0]
    B = extrinsics.shape[0]

    w_h = np.ones((N, 1), dtype=points3D.dtype)
    points3D_h = np.concatenate([points3D, w_h], axis=1)

    points3D_h_B = np.broadcast_to(points3D_h, (B, N, 4))

    points_cam = np.einsum("bij,bnj->bni", extrinsics, points3D_h_B)
    points_cam = points_cam.transpose(0, 2, 1)

    if only_points_cam:
        return None, points_cam

    if intrinsics is None:
        raise ValueError("`intrinsics` must be provided unless only_points_cam=True")

    points2D = img_from_cam_np(intrinsics, points_cam, extra_params=extra_params, default=default)

    return points2D, points_cam


def project_3D_points(points3D, extrinsics, intrinsics=None, extra_params=None, default=0, only_points_cam=False):
    with torch.cuda.amp.autocast(dtype=torch.double):
        N = points3D.shape[0]
        B = extrinsics.shape[0]
        points3D_homogeneous = torch.cat([points3D, torch.ones_like(points3D[..., 0:1])], dim=1)
        points3D_homogeneous = points3D_homogeneous.unsqueeze(0).expand(B, -1, -1)

        points_cam = torch.bmm(extrinsics, points3D_homogeneous.transpose(-1, -2))

        if only_points_cam:
            return None, points_cam

        points2D = img_from_cam(intrinsics, points_cam, extra_params, default)

        return points2D, points_cam


def img_from_cam(intrinsics, points_cam, extra_params=None, default=0.0):

    points_cam = points_cam / points_cam[:, 2:3, :]
    uv = points_cam[:, :2, :]

    if extra_params is not None:
        uu, vv = apply_distortion(extra_params, uv[:, 0], uv[:, 1])
        uv = torch.stack([uu, vv], dim=1)

    points_cam_homo = torch.cat((uv, torch.ones_like(uv[:, :1, :])), dim=1)
    points2D_homo = torch.bmm(intrinsics, points_cam_homo)

    points2D = points2D_homo[:, :2, :]

    points2D = torch.nan_to_num(points2D, nan=default)

    return points2D.transpose(1, 2)


if __name__ == "__main__":
    B, N = 24, 10240

    for _ in range(100):
        points3D = np.random.rand(N, 3).astype(np.float64)
        extrinsics = np.random.rand(B, 3, 4).astype(np.float64)
        intrinsics = np.random.rand(B, 3, 3).astype(np.float64)

        points3D_torch = torch.tensor(points3D)
        extrinsics_torch = torch.tensor(extrinsics)
        intrinsics_torch = torch.tensor(intrinsics)

        points2D_np, points_cam_np = project_3D_points_np(points3D, extrinsics, intrinsics)

        points2D_torch, points_cam_torch = project_3D_points(points3D_torch, extrinsics_torch, intrinsics_torch)

        points2D_torch_np = points2D_torch.detach().numpy()
        points_cam_torch_np = points_cam_torch.detach().numpy()

        diff = np.abs(points2D_np - points2D_torch_np)
        print("Difference between NumPy and PyTorch implementations:")
        print(diff)

        max_diff = np.max(diff)
        print(f"Maximum difference: {max_diff}")

        if np.allclose(points2D_np, points2D_torch_np, atol=1e-6):
            print("Implementations match closely.")
        else:
            print("Significant differences detected.")

        if points_cam_np is not None:
            points_cam_diff = np.abs(points_cam_np - points_cam_torch_np)
            print("Difference between NumPy and PyTorch camera-space coordinates:")
            print(points_cam_diff)

            max_cam_diff = np.max(points_cam_diff)
            print(f"Maximum camera-space coordinate difference: {max_cam_diff}")

            if np.allclose(points_cam_np, points_cam_torch_np, atol=1e-6):
                print("Camera-space coordinates match closely.")
            else:
                print("Significant differences detected in camera-space coordinates.")
