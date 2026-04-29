


import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Union


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: Union[int, Tuple[int, int]], return_grid=False) -> torch.Tensor:
    if isinstance(grid_size, tuple):
        grid_size_h, grid_size_w = grid_size
    else:
        grid_size_h = grid_size_w = grid_size
    grid_h = torch.arange(grid_size_h, dtype=torch.float)
    grid_w = torch.arange(grid_size_w, dtype=torch.float)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
    grid = torch.stack(grid, dim=0)
    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if return_grid:
        return (pos_embed.reshape(1, grid_size_h, grid_size_w, -1).permute(0, 3, 1, 2), grid)
    return pos_embed.reshape(1, grid_size_h, grid_size_w, -1).permute(0, 3, 1, 2)


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: torch.Tensor) -> torch.Tensor:
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])

    emb = torch.cat([emb_h, emb_w], dim=2)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.double)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = torch.einsum("m,d->md", pos, omega)

    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb[None].float()


def get_2d_embedding(xy: torch.Tensor, C: int, cat_coords: bool = True) -> torch.Tensor:
    B, N, D = xy.shape
    assert D == 2

    x = xy[:, :, 0:1]
    y = xy[:, :, 1:2]
    div_term = (torch.arange(0, C, 2, device=xy.device, dtype=torch.float32) * (1000.0 / C)).reshape(1, 1, int(C / 2))

    pe_x = torch.zeros(B, N, C, device=xy.device, dtype=torch.float32)
    pe_y = torch.zeros(B, N, C, device=xy.device, dtype=torch.float32)

    pe_x[:, :, 0::2] = torch.sin(x * div_term)
    pe_x[:, :, 1::2] = torch.cos(x * div_term)

    pe_y[:, :, 0::2] = torch.sin(y * div_term)
    pe_y[:, :, 1::2] = torch.cos(y * div_term)

    pe = torch.cat([pe_x, pe_y], dim=2)
    if cat_coords:
        pe = torch.cat([xy, pe], dim=2)
    return pe


def bilinear_sampler(input, coords, align_corners=True, padding_mode="border"):
    coords = coords.detach().clone()
    coords = coords.to(input.device).to(input.dtype)

    sizes = input.shape[2:]

    assert len(sizes) in [2, 3]

    if len(sizes) == 3:
        coords = coords[..., [1, 2, 0]]

    if align_corners:
        scale = torch.tensor(
            [2 / max(size - 1, 1) for size in reversed(sizes)], device=coords.device, dtype=coords.dtype
        )
    else:
        scale = torch.tensor([2 / size for size in reversed(sizes)], device=coords.device, dtype=coords.dtype)

    coords.mul_(scale)
    coords.sub_(1)

    return F.grid_sample(input, coords, align_corners=align_corners, padding_mode=padding_mode)


def sample_features4d(input, coords):

    B, _, _, _ = input.shape

    coords = coords.unsqueeze(2)

    feats = bilinear_sampler(input, coords)

    return feats.permute(0, 2, 1, 3).view(B, -1, feats.shape[1] * feats.shape[3])
