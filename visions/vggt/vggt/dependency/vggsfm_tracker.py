

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

from hydra.utils import instantiate
from omegaconf import OmegaConf

from .track_modules.track_refine import refine_track
from .track_modules.blocks import BasicEncoder, ShallowEncoder
from .track_modules.base_track_predictor import BaseTrackerPredictor


class TrackerPredictor(nn.Module):
    def __init__(self, **extra_args):
        super(TrackerPredictor, self).__init__()
        coarse_stride = 4
        self.coarse_down_ratio = 2

        self.coarse_fnet = BasicEncoder(stride=coarse_stride)
        self.coarse_predictor = BaseTrackerPredictor(stride=coarse_stride)

        self.fine_fnet = ShallowEncoder(stride=1)
        self.fine_predictor = BaseTrackerPredictor(
            stride=1,
            depth=4,
            corr_levels=3,
            corr_radius=3,
            latent_dim=32,
            hidden_size=256,
            fine=True,
            use_spaceatt=False,
        )

    def forward(
        self, images, query_points, fmaps=None, coarse_iters=6, inference=True, fine_tracking=True, fine_chunk=40960
    ):

        if fmaps is None:
            batch_num, frame_num, image_dim, height, width = images.shape
            reshaped_image = images.reshape(batch_num * frame_num, image_dim, height, width)
            fmaps = self.process_images_to_fmaps(reshaped_image)
            fmaps = fmaps.reshape(batch_num, frame_num, -1, fmaps.shape[-2], fmaps.shape[-1])

            if inference:
                torch.cuda.empty_cache()

        coarse_pred_track_lists, pred_vis = self.coarse_predictor(
            query_points=query_points, fmaps=fmaps, iters=coarse_iters, down_ratio=self.coarse_down_ratio
        )
        coarse_pred_track = coarse_pred_track_lists[-1]

        if inference:
            torch.cuda.empty_cache()

        if fine_tracking:
            fine_pred_track, pred_score = refine_track(
                images, self.fine_fnet, self.fine_predictor, coarse_pred_track, compute_score=False, chunk=fine_chunk
            )

            if inference:
                torch.cuda.empty_cache()
        else:
            fine_pred_track = coarse_pred_track
            pred_score = torch.ones_like(pred_vis)

        return fine_pred_track, coarse_pred_track, pred_vis, pred_score

    def process_images_to_fmaps(self, images):
        if self.coarse_down_ratio > 1:
            fmaps = self.coarse_fnet(
                F.interpolate(images, scale_factor=1 / self.coarse_down_ratio, mode="bilinear", align_corners=True)
            )
        else:
            fmaps = self.coarse_fnet(images)

        return fmaps
