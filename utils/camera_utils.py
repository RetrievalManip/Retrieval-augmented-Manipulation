from typing import Dict, List, Optional, Sequence, Tuple, Union
from torch import Tensor

import os
from pathlib import Path
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.interpolate import interp1d
from tqdm import tqdm
import cv2

def get_colored_points_from_depth_with_box_promtda(
    depths: Tensor,
    rgbs: Tensor,
    c2w: Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_size: tuple,
    device: torch.device = torch.device("cuda"),
    mask: Optional[Tensor] = None,
    box: List = None,
    promptda = None,
) -> Tuple[Tensor, Tensor]:

    if not torch.is_tensor(depths):
        if isinstance(depths, np.ndarray):
            depths = np.ascontiguousarray(depths)
        depths = torch.tensor(depths, device=device)
    if not torch.is_tensor(rgbs):
        if isinstance(rgbs, np.ndarray):
            rgbs = np.ascontiguousarray(rgbs)
        rgbs = torch.tensor(rgbs, device=device)
    if not torch.is_tensor(c2w):
        if isinstance(c2w, np.ndarray):
            c2w = np.ascontiguousarray(c2w)
        c2w = torch.tensor(c2w, device=device)
    if not torch.is_tensor(mask):
        mask = torch.tensor(mask, device=device)


    if rgbs.max() > 1.0:
        rgbs = rgbs / 255.0

    if box is not None:
        xmin, ymin, xmax, ymax = box
        depths = depths[ymin:ymax, xmin:xmax].contiguous()
        rgbs = rgbs[ymin:ymax, xmin:xmax, ...].contiguous()
        mask = mask[ymin:ymax, xmin:xmax].contiguous()
        cx = cx - xmin
        cy = cy - ymin
        img_size = (xmax - xmin, ymax - ymin)

    if promptda is not None:
        depths = depths.cpu().numpy()
        rgbs = rgbs.cpu().numpy()
        depths = promptda.finetune_depth(
            rgb_image=rgbs,
            depth_map=depths,
        )
        depths = torch.tensor(depths, device=device)
        rgbs = torch.tensor(rgbs, device=device)

    points, _ = get_means3d_backproj(
        depths=depths.float(),
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        img_size=img_size,
        c2w=c2w.float(),
        device=depths.device,
    )
    points = points.squeeze(0)
    if mask is not None:
        if not torch.is_tensor(mask):
            mask = torch.tensor(mask, device=depths.device)
        if mask.dim() == 2:
            mask = mask.view(-1)
        colors = rgbs.view(-1, 3)
        colors = colors[mask]
        points = points[mask]
    else:
        colors = rgbs.view(-1, 3)
        points = points
    return (points, colors)

def get_colored_points_from_depth_with_box(
    depths: Tensor,
    rgbs: Tensor,
    c2w: Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_size: tuple,
    device: torch.device = torch.device("cuda"),
    mask: Optional[Tensor] = None,
    box: List = None,
) -> Tuple[Tensor, Tensor]:

    if not torch.is_tensor(depths):
        if isinstance(depths, np.ndarray):
            depths = np.ascontiguousarray(depths)
        depths = torch.tensor(depths, device=device)
    if not torch.is_tensor(rgbs):
        if isinstance(rgbs, np.ndarray):
            rgbs = np.ascontiguousarray(rgbs)
        rgbs = torch.tensor(rgbs, device=device)
    if not torch.is_tensor(c2w):
        if isinstance(c2w, np.ndarray):
            c2w = np.ascontiguousarray(c2w)
        c2w = torch.tensor(c2w, device=device)
    if not torch.is_tensor(mask):
        mask = torch.tensor(mask, device=device)

    if rgbs.max() > 1.0:
        rgbs = rgbs / 255.0

    if box is not None:
        xmin, ymin, xmax, ymax = box
        depths = depths[ymin:ymax, xmin:xmax].contiguous()
        rgbs = rgbs[ymin:ymax, xmin:xmax, ...].contiguous()
        mask = mask[ymin:ymax, xmin:xmax].contiguous()
        cx = cx - xmin
        cy = cy - ymin
        img_size = (xmax - xmin, ymax - ymin)

    points, _ = get_means3d_backproj(
        depths=depths.float(),
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        img_size=img_size,
        c2w=c2w.float(),
        device=depths.device,
    )
    points = points.squeeze(0)
    if mask is not None:
        if not torch.is_tensor(mask):
            mask = torch.tensor(mask, device=depths.device)
        if mask.dim() == 2:
            mask = mask.view(-1)
        colors = rgbs.view(-1, 3)
        colors = colors[mask]
        points = points[mask]
    else:
        colors = rgbs.view(-1, 3)
        points = points
    return (points, colors)



def get_colored_points_from_depth(
    depths: Tensor,
    rgbs: Tensor,
    c2w: Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_size: tuple,
    device: torch.device = torch.device("cuda"),
    mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:

    if not torch.is_tensor(depths):
        if isinstance(depths, np.ndarray):
            depths = np.ascontiguousarray(depths)
        depths = torch.tensor(depths, device=device)
    if not torch.is_tensor(rgbs):
        if isinstance(rgbs, np.ndarray):
            rgbs = np.ascontiguousarray(rgbs)
        rgbs = torch.tensor(rgbs, device=device)
    if not torch.is_tensor(c2w):
        if isinstance(c2w, np.ndarray):
            c2w = np.ascontiguousarray(c2w)
        c2w = torch.tensor(c2w, device=device)

    if rgbs.max() > 1.0:
        rgbs = rgbs / 255.0

    points, _ = get_means3d_backproj(
        depths=depths.float(),
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        img_size=img_size,
        c2w=c2w.float(),
        device=depths.device,
    )
    points = points.squeeze(0)
    if mask is not None:
        if not torch.is_tensor(mask):
            mask = torch.tensor(mask, device=depths.device)
        if mask.dim() == 2:
            mask = mask.view(-1)
        colors = rgbs.view(-1, 3)[mask]
        points = points[mask]
    else:
        colors = rgbs.view(-1, 3)
        points = points
    return (points, colors)

def get_means3d_backproj(
    depths: Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_size: tuple,
    c2w: Tensor,
    device: torch.device,
    mask: Optional[Tensor] = None,
) -> Tuple[Tensor, List]:

    if depths.dim() == 3:
        depths = depths.view(-1, 1)
    elif depths.shape[-1] != 1:
        depths = depths.unsqueeze(-1).contiguous()
        depths = depths.view(-1, 1)
    if depths.dtype != torch.float:
        depths = depths.float()
        c2w = c2w.float()
    if c2w.device != device:
        c2w = c2w.to(device)

    image_coords = get_camera_coords(img_size)
    image_coords = image_coords.to(device)

    means3d = torch.empty(
        size=(img_size[0], img_size[1], 3), dtype=torch.float32, device=device
    ).view(-1, 3)
    means3d[:, 0] = (image_coords[:, 0] - cx) * depths[:, 0] / fx
    means3d[:, 1] = (image_coords[:, 1] - cy) * depths[:, 0] / fy
    means3d[:, 2] = depths[:, 0]

    if mask is not None:
        if not torch.is_tensor(mask):
            mask = torch.tensor(mask, device=depths.device)
        means3d = means3d[mask]
        image_coords = image_coords[mask]

    if c2w is None:
        c2w = torch.eye((means3d.shape[0], 4, 4), device=device)

    means3d = means3d @ c2w[..., :3, :3].T + c2w[..., :3, 3]
    return means3d, image_coords

def get_camera_coords(img_size: tuple, pixel_offset: float = 0.5) -> Tensor:

    image_coords = torch.meshgrid(
        torch.arange(img_size[0]),
        torch.arange(img_size[1]),
        indexing="xy",
    )
    image_coords = (
        torch.stack(image_coords, dim=-1) + pixel_offset
    )
    image_coords = image_coords.view(-1, 2)
    image_coords = image_coords.float()

    return image_coords


def get_uvs_backproj(
    uvs: np.ndarray,
    depths: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    c2w: np.ndarray,
) -> np.ndarray:
    if len(depths.shape) == 1:
        depths = depths[:, np.newaxis]

    means3d = np.zeros(
        shape=(uvs.shape[0], 3), dtype=np.float32
    )
    means3d[:, 0] = (uvs[:, 0] - cx) * depths[:, 0] / fx
    means3d[:, 1] = (uvs[:, 1] - cy) * depths[:, 0] / fy
    means3d[:, 2] = depths[:, 0]

    if c2w is None:
        c2w = np.eye((means3d.shape[0], 4, 4), device=device)

    means3d = means3d @ c2w[..., :3, :3].T + c2w[..., :3, 3]
    return means3d
