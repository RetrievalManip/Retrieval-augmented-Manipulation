
import torch.nn as nn
from .dpt_head import DPTHead
from .track_modules.base_track_predictor import BaseTrackerPredictor


class TrackHead(nn.Module):

    def __init__(
        self,
        dim_in,
        patch_size=14,
        features=128,
        iters=4,
        predict_conf=True,
        stride=2,
        corr_levels=7,
        corr_radius=4,
        hidden_size=384,
    ):
        super().__init__()

        self.patch_size = patch_size

        self.feature_extractor = DPTHead(
            dim_in=dim_in,
            patch_size=patch_size,
            features=features,
            feature_only=True,
            down_ratio=2,
            pos_embed=False,
        )

        self.tracker = BaseTrackerPredictor(
            latent_dim=features,
            predict_conf=predict_conf,
            stride=stride,
            corr_levels=corr_levels,
            corr_radius=corr_radius,
            hidden_size=hidden_size,
        )

        self.iters = iters

    def forward(self, aggregated_tokens_list, images, patch_start_idx, query_points=None, iters=None):
        B, S, _, H, W = images.shape

        feature_maps = self.feature_extractor(aggregated_tokens_list, images, patch_start_idx)

        if iters is None:
            iters = self.iters

        coord_preds, vis_scores, conf_scores = self.tracker(query_points=query_points, fmaps=feature_maps, iters=iters)

        return coord_preds, vis_scores, conf_scores
