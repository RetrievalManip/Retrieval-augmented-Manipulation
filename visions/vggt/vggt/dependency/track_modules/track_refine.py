

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

from PIL import Image
import os
from typing import Union, Tuple


def refine_track(
    images, fine_fnet, fine_tracker, coarse_pred, compute_score=False, pradius=15, sradius=2, fine_iters=6, chunk=40960
):

    B, S, N, _ = coarse_pred.shape
    _, _, _, H, W = images.shape

    psize = pradius * 2 + 1

    query_points = coarse_pred[:, 0]



    with torch.no_grad():
        content_to_extract = images.reshape(B * S, 3, H, W)
        C_in = content_to_extract.shape[1]

        content_to_extract = content_to_extract.unfold(2, psize, 1).unfold(3, psize, 1)

    track_int = coarse_pred.floor().int()
    track_frac = coarse_pred - track_int

    topleft = track_int - pradius
    topleft_BSN = topleft.clone()

    topleft = topleft.clamp(0, H - psize)

    topleft = topleft.reshape(B * S, N, 2)

    batch_indices = torch.arange(B * S)[:, None].expand(-1, N).to(content_to_extract.device)

    extracted_patches = content_to_extract[batch_indices, :, topleft[..., 1], topleft[..., 0]]

    if chunk < 0:
        patch_feat = fine_fnet(extracted_patches.reshape(B * S * N, C_in, psize, psize))
    else:
        patches = extracted_patches.reshape(B * S * N, C_in, psize, psize)

        patch_feat_list = []
        for p in torch.split(patches, chunk):
            patch_feat_list += [fine_fnet(p)]
        patch_feat = torch.cat(patch_feat_list, 0)

    C_out = patch_feat.shape[1]

    patch_feat = patch_feat.reshape(B, S, N, C_out, psize, psize)
    patch_feat = rearrange(patch_feat, "b s n c p q -> (b n) s c p q")

    patch_query_points = track_frac[:, 0] + pradius
    patch_query_points = patch_query_points.reshape(B * N, 2).unsqueeze(1)

    fine_pred_track_lists, _, _, query_point_feat = fine_tracker(
        query_points=patch_query_points, fmaps=patch_feat, iters=fine_iters, return_feat=True
    )

    fine_pred_track = fine_pred_track_lists[-1].clone()

    for idx in range(len(fine_pred_track_lists)):
        fine_level = rearrange(fine_pred_track_lists[idx], "(b n) s u v -> b s n u v", b=B, n=N)
        fine_level = fine_level.squeeze(-2)
        fine_level = fine_level + topleft_BSN
        fine_pred_track_lists[idx] = fine_level

    refined_tracks = fine_pred_track_lists[-1].clone()
    refined_tracks[:, 0] = query_points

    score = None

    if compute_score:
        score = compute_score_fn(query_point_feat, patch_feat, fine_pred_track, sradius, psize, B, N, S, C_out)

    return refined_tracks, score


def refine_track_v0(
    images, fine_fnet, fine_tracker, coarse_pred, compute_score=False, pradius=15, sradius=2, fine_iters=6
):

    B, S, N, _ = coarse_pred.shape
    _, _, _, H, W = images.shape

    psize = pradius * 2 + 1

    query_points = coarse_pred[:, 0]



    with torch.no_grad():
        content_to_extract = images.reshape(B * S, 3, H, W)
        C_in = content_to_extract.shape[1]

        content_to_extract = content_to_extract.unfold(2, psize, 1).unfold(3, psize, 1)

    track_int = coarse_pred.floor().int()
    track_frac = coarse_pred - track_int

    topleft = track_int - pradius
    topleft_BSN = topleft.clone()

    topleft = topleft.clamp(0, H - psize)

    topleft = topleft.reshape(B * S, N, 2)

    batch_indices = torch.arange(B * S)[:, None].expand(-1, N).to(content_to_extract.device)

    extracted_patches = content_to_extract[batch_indices, :, topleft[..., 1], topleft[..., 0]]

    patch_feat = fine_fnet(extracted_patches.reshape(B * S * N, C_in, psize, psize))

    C_out = patch_feat.shape[1]


    patch_feat = patch_feat.reshape(B, S, N, C_out, psize, psize)
    patch_feat = rearrange(patch_feat, "b s n c p q -> (b n) s c p q")

    patch_query_points = track_frac[:, 0] + pradius
    patch_query_points = patch_query_points.reshape(B * N, 2).unsqueeze(1)

    fine_pred_track_lists, _, _, query_point_feat = fine_tracker(
        query_points=patch_query_points, fmaps=patch_feat, iters=fine_iters, return_feat=True
    )

    fine_pred_track = fine_pred_track_lists[-1].clone()

    for idx in range(len(fine_pred_track_lists)):
        fine_level = rearrange(fine_pred_track_lists[idx], "(b n) s u v -> b s n u v", b=B, n=N)
        fine_level = fine_level.squeeze(-2)
        fine_level = fine_level + topleft_BSN
        fine_pred_track_lists[idx] = fine_level

    refined_tracks = fine_pred_track_lists[-1].clone()
    refined_tracks[:, 0] = query_points

    score = None

    if compute_score:
        score = compute_score_fn(query_point_feat, patch_feat, fine_pred_track, sradius, psize, B, N, S, C_out)

    return refined_tracks, score




def compute_score_fn(query_point_feat, patch_feat, fine_pred_track, sradius, psize, B, N, S, C_out):

    from kornia.utils.grid import create_meshgrid
    from kornia.geometry.subpix import dsnt

    query_point_feat = query_point_feat.reshape(B, N, C_out)
    query_point_feat = query_point_feat.unsqueeze(1).expand(-1, S - 1, -1, -1)
    query_point_feat = query_point_feat.reshape(B * (S - 1) * N, C_out)

    ssize = sradius * 2 + 1

    patch_feat = rearrange(patch_feat, "(b n) s c p q -> b s n c p q", b=B, n=N)

    patch_feat_unfold = patch_feat.unfold(4, ssize, 1).unfold(5, ssize, 1)

    fine_prediction_floor = fine_pred_track.floor().int()
    fine_level_floor_topleft = fine_prediction_floor - sradius

    fine_level_floor_topleft = fine_level_floor_topleft.clamp(0, psize - ssize)
    fine_level_floor_topleft = fine_level_floor_topleft.squeeze(2)


    batch_indices_score = torch.arange(B)[:, None, None].expand(-1, S, N)
    batch_indices_score = batch_indices_score.reshape(-1).to(patch_feat_unfold.device)
    y_indices = fine_level_floor_topleft[..., 0].flatten()
    x_indices = fine_level_floor_topleft[..., 1].flatten()

    reference_frame_feat = patch_feat_unfold.reshape(
        B * S * N, C_out, psize - sradius * 2, psize - sradius * 2, ssize, ssize
    )

    reference_frame_feat = reference_frame_feat[batch_indices_score, :, x_indices, y_indices]
    reference_frame_feat = reference_frame_feat.reshape(B, S, N, C_out, ssize, ssize)
    reference_frame_feat = reference_frame_feat[:, 1:].reshape(B * (S - 1) * N, C_out, ssize * ssize)

    sim_matrix = torch.einsum("mc,mcr->mr", query_point_feat, reference_frame_feat)
    softmax_temp = 1.0 / C_out**0.5
    heatmap = torch.softmax(softmax_temp * sim_matrix, dim=1)
    heatmap = heatmap.reshape(B * (S - 1) * N, ssize, ssize)

    coords_normalized = dsnt.spatial_expectation2d(heatmap[None], True)[0]
    grid_normalized = create_meshgrid(ssize, ssize, normalized_coordinates=True, device=heatmap.device).reshape(
        1, -1, 2
    )

    var = torch.sum(grid_normalized**2 * heatmap.view(-1, ssize * ssize, 1), dim=1) - coords_normalized**2
    std = torch.sum(torch.sqrt(torch.clamp(var, min=1e-10)), -1)

    score = std.reshape(B, S - 1, N)
    score = torch.cat([torch.ones_like(score[:, 0:1]), score], dim=1)

    return score


def extract_glimpse(
    tensor: torch.Tensor, size: Tuple[int, int], offsets, mode="bilinear", padding_mode="zeros", debug=False, orib=None
):
    B, C, W, H = tensor.shape

    h, w = size
    xs = torch.arange(0, w, dtype=tensor.dtype, device=tensor.device) - (w - 1) / 2.0
    ys = torch.arange(0, h, dtype=tensor.dtype, device=tensor.device) - (h - 1) / 2.0

    vy, vx = torch.meshgrid(ys, xs)
    grid = torch.stack([vx, vy], dim=-1)
    grid = grid[None]

    B, N, _ = offsets.shape

    offsets = offsets.reshape((B * N), 1, 1, 2)
    offsets_grid = offsets + grid

    offsets_grid = (offsets_grid - offsets_grid.new_tensor([W / 2, H / 2])) / offsets_grid.new_tensor([W / 2, H / 2])

    tensor = tensor[:, None]

    tensor = tensor.expand(-1, N, -1, -1, -1)

    tensor = tensor.reshape((B * N), C, W, H)

    sampled = torch.nn.functional.grid_sample(
        tensor, offsets_grid, mode=mode, align_corners=False, padding_mode=padding_mode
    )

    sampled = sampled.reshape(B, N, C, h, w)

    return sampled
