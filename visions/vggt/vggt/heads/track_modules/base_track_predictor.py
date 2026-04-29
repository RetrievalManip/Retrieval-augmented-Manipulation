
import torch
import torch.nn as nn
from einops import rearrange, repeat


from .blocks import EfficientUpdateFormer, CorrBlock
from .utils import sample_features4d, get_2d_embedding, get_2d_sincos_pos_embed
from .modules import Mlp


class BaseTrackerPredictor(nn.Module):
    def __init__(
        self,
        stride=1,
        corr_levels=5,
        corr_radius=4,
        latent_dim=128,
        hidden_size=384,
        use_spaceatt=True,
        depth=6,
        max_scale=518,
        predict_conf=True,
    ):
        super(BaseTrackerPredictor, self).__init__()

        self.stride = stride
        self.latent_dim = latent_dim
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.hidden_size = hidden_size
        self.max_scale = max_scale
        self.predict_conf = predict_conf

        self.flows_emb_dim = latent_dim // 2

        self.corr_mlp = Mlp(
            in_features=self.corr_levels * (self.corr_radius * 2 + 1) ** 2,
            hidden_features=self.hidden_size,
            out_features=self.latent_dim,
        )

        self.transformer_dim = self.latent_dim + self.latent_dim + self.latent_dim + 4

        self.query_ref_token = nn.Parameter(torch.randn(1, 2, self.transformer_dim))

        space_depth = depth if use_spaceatt else 0
        time_depth = depth

        self.updateformer = EfficientUpdateFormer(
            space_depth=space_depth,
            time_depth=time_depth,
            input_dim=self.transformer_dim,
            hidden_size=self.hidden_size,
            output_dim=self.latent_dim + 2,
            mlp_ratio=4.0,
            add_space_attn=use_spaceatt,
        )

        self.fmap_norm = nn.LayerNorm(self.latent_dim)
        self.ffeat_norm = nn.GroupNorm(1, self.latent_dim)

        self.ffeat_updater = nn.Sequential(nn.Linear(self.latent_dim, self.latent_dim), nn.GELU())

        self.vis_predictor = nn.Sequential(nn.Linear(self.latent_dim, 1))

        if predict_conf:
            self.conf_predictor = nn.Sequential(nn.Linear(self.latent_dim, 1))

    def forward(self, query_points, fmaps=None, iters=6, return_feat=False, down_ratio=1, apply_sigmoid=True):
        B, N, D = query_points.shape
        B, S, C, HH, WW = fmaps.shape

        assert D == 2, "Input points must be 2D coordinates"

        fmaps = self.fmap_norm(fmaps.permute(0, 1, 3, 4, 2))
        fmaps = fmaps.permute(0, 1, 4, 2, 3)

        if down_ratio > 1:
            query_points = query_points / float(down_ratio)

        query_points = query_points / float(self.stride)

        coords = query_points.clone().reshape(B, 1, N, 2).repeat(1, S, 1, 1)

        query_track_feat = sample_features4d(fmaps[:, 0], coords[:, 0])

        track_feats = query_track_feat.unsqueeze(1).repeat(1, S, 1, 1)
        coords_backup = coords.clone()

        fcorr_fn = CorrBlock(fmaps, num_levels=self.corr_levels, radius=self.corr_radius)

        coord_preds = []

        for _ in range(iters):
            coords = coords.detach()

            fcorrs = fcorr_fn.corr_sample(track_feats, coords)

            corr_dim = fcorrs.shape[3]
            fcorrs_ = fcorrs.permute(0, 2, 1, 3).reshape(B * N, S, corr_dim)
            fcorrs_ = self.corr_mlp(fcorrs_)

            flows = (coords - coords[:, 0:1]).permute(0, 2, 1, 3).reshape(B * N, S, 2)

            flows_emb = get_2d_embedding(flows, self.flows_emb_dim, cat_coords=False)

            flows_emb = torch.cat([flows_emb, flows / self.max_scale, flows / self.max_scale], dim=-1)

            track_feats_ = track_feats.permute(0, 2, 1, 3).reshape(B * N, S, self.latent_dim)

            transformer_input = torch.cat([flows_emb, fcorrs_, track_feats_], dim=2)

            pos_embed = get_2d_sincos_pos_embed(self.transformer_dim, grid_size=(HH, WW)).to(query_points.device)
            sampled_pos_emb = sample_features4d(pos_embed.expand(B, -1, -1, -1), coords[:, 0])

            sampled_pos_emb = rearrange(sampled_pos_emb, "b n c -> (b n) c").unsqueeze(1)

            x = transformer_input + sampled_pos_emb

            query_ref_token = torch.cat(
                [self.query_ref_token[:, 0:1], self.query_ref_token[:, 1:2].expand(-1, S - 1, -1)], dim=1
            )
            x = x + query_ref_token.to(x.device).to(x.dtype)

            x = rearrange(x, "(b n) s d -> b n s d", b=B)

            delta, _ = self.updateformer(x)

            delta = rearrange(delta, " b n s d -> (b n) s d", b=B)
            delta_coords_ = delta[:, :, :2]
            delta_feats_ = delta[:, :, 2:]

            track_feats_ = track_feats_.reshape(B * N * S, self.latent_dim)
            delta_feats_ = delta_feats_.reshape(B * N * S, self.latent_dim)

            track_feats_ = self.ffeat_updater(self.ffeat_norm(delta_feats_)) + track_feats_

            track_feats = track_feats_.reshape(B, N, S, self.latent_dim).permute(0, 2, 1, 3)

            coords = coords + delta_coords_.reshape(B, N, S, 2).permute(0, 2, 1, 3)

            coords[:, 0] = coords_backup[:, 0]

            if down_ratio > 1:
                coord_preds.append(coords * self.stride * down_ratio)
            else:
                coord_preds.append(coords * self.stride)

        vis_e = self.vis_predictor(track_feats.reshape(B * S * N, self.latent_dim)).reshape(B, S, N)
        if apply_sigmoid:
            vis_e = torch.sigmoid(vis_e)

        if self.predict_conf:
            conf_e = self.conf_predictor(track_feats.reshape(B * S * N, self.latent_dim)).reshape(B, S, N)
            if apply_sigmoid:
                conf_e = torch.sigmoid(conf_e)
        else:
            conf_e = None

        if return_feat:
            return coord_preds, vis_e, track_feats, query_track_feat, conf_e
        else:
            return coord_preds, vis_e, conf_e
