import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('.')

from vision_transformer_dinov2 import vit_base

from pct import TransformerEncoder, MapPosTransformer

SUPPORTED_FOUNDATION_MODEL = 'dinov2-b14'

def attention(query, key, value):
    d_k = query.size(-1)
    
    scores = torch.matmul(query, key.transpose(-2, -1).contiguous()) / math.sqrt(d_k)

    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), scores

class RAMNet(nn.Module):
    def __init__(self, foundation_model = SUPPORTED_FOUNDATION_MODEL, feat_layers = [7, 9, 11], \
        feat_type = 'k', img_size = 224):
        super().__init__()

        self.foundation_model_name = foundation_model
        if self.foundation_model_name != SUPPORTED_FOUNDATION_MODEL:
            raise ValueError(
                f"Unsupported foundation model '{self.foundation_model_name}'. "
                f"Only {SUPPORTED_FOUNDATION_MODEL} without register tokens is supported."
            )

        self.layers = feat_layers
        self.feat_type = feat_type
        self.img_size = img_size

        self.patch_size = 14
        self.patch_num = int(self.img_size // self.patch_size)
        self.foundation_model = vit_base(patch_size=self.patch_size, img_size=518, block_chunks=0, init_values=1e-5)
        self.emb_dim = len(self.layers) * 768

        self.view_adapter = TransformerEncoder(embed_dim=self.emb_dim, \
            depth=1, num_heads=4, fetch_list=[0])
        
        self.view_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.emb_dim)
        )
        
        self.shape_encoder = MapPosTransformer(depth=4, num_heads=4, \
            fetch_list = [1,3], output_dim = self.emb_dim, \
            use_pose_embed = True, fix_pose_embed = True)
        
        self.map_adapter = TransformerEncoder(embed_dim=self.emb_dim, \
            depth=4, num_heads=4, fetch_list=[3])
        
        self.deform_decoder = nn.Sequential(
            nn.Conv1d(2*self.emb_dim, 512, 1),
            nn.GELU(),
            nn.Conv1d(512, 256, 1),
            nn.GELU(),
            nn.Conv1d(256, 3, 1)
        )

    def load_pretrain_checkpoint(self, pretrain_path):
        state_dict = torch.load(pretrain_path, map_location='cpu')
        self.foundation_model.load_state_dict(state_dict)
    
    @torch.no_grad()
    def __extract_foundation_feature__(self, img):
        return self.foundation_model.get_specific_tokens(img, layers_to_return = self.layers)
    
    @torch.no_grad()
    def __reshape_single_layer_feats__(self, feat, feat_type):
        b, h, n, d = feat[feat_type].shape
        class_token = feat[feat_type].permute(0, 1, 3, 2)[:, :, :, 0].reshape(b, h*d)
        patch_token = feat[feat_type].permute(0, 1, 3, 2)[:, :, :, 1:].reshape(b, h*d, -1)

        class_token = class_token.contiguous()
        patch_token = patch_token.contiguous()
        return class_token, patch_token
    
    @torch.no_grad()
    def __reshape_multiple_layer_feats__(self, feats):
        feats_with_class_token = []
        feats_no_class_token = []
        feat_type = self.feat_type

        for l in self.layers:
            class_token, patch_token = self.__reshape_single_layer_feats__(feats[l], feat_type)
            b, d, n = patch_token.shape

            class_token = class_token.unsqueeze(-1).repeat(1, 1, patch_token.shape[-1])
            patch_token_with_class = (patch_token+class_token).reshape(b, -1, \
                self.patch_num, self.patch_num).contiguous()

            patch_token_no_class = patch_token.reshape(b, -1, self.patch_num, self.patch_num).contiguous()

            feats_with_class_token.append(patch_token_with_class)
            feats_no_class_token.append(patch_token_no_class)
        
        feats_with_class_token = torch.cat(feats_with_class_token, dim=1)
        feats_no_class_token = torch.cat(feats_no_class_token, dim=1)

        return feats_with_class_token, feats_no_class_token
    
    @torch.no_grad()
    def __sample_feat_vec__(self, feat_map, choose):
        resize_feat_map = F.interpolate(feat_map, size = (self.img_size, self.img_size), \
            mode = 'bilinear', align_corners=True)

        b, c, h, w = resize_feat_map.shape

        resize_feat_map = resize_feat_map.view(b, c, -1)
        choose = choose.unsqueeze(1).repeat(1, c, 1)
        sampled_feat_vec = torch.gather(resize_feat_map, 2, choose).contiguous()

        return sampled_feat_vec
    
    def forward(self, query_img, choose, all_template_feat, all_template_pe, select_template_feat, select_template_nocs, \
        query_partial_shape, query_shape_in_template_view, all_template_partial_shape, select_template_partial_shape, \
        select_template_shape_in_obj_frame, local_template_feat, local_template_pe, local_template_partial_shape, \
        template_complete_nocs):
        
        query_feats = self.__extract_foundation_feature__(query_img)
        query_feat_with_class_token, query_feat_no_class_token = self.__reshape_multiple_layer_feats__(query_feats)
        
        sampled_query_feat_with_class_token = self.__sample_feat_vec__(query_feat_with_class_token, choose)
        sampled_query_feat_no_class_token = self.__sample_feat_vec__(query_feat_no_class_token, choose)
        query_position_embedding = self.view_pos_embed(query_partial_shape).permute(0,2,1)

        query_view_feat = self.view_adapter(sampled_query_feat_with_class_token.permute(0,2,1), \
            query_position_embedding.permute(0,2,1))[0].permute(0,2,1)
        query_view_feat = F.normalize(torch.mean(query_view_feat, dim=-1), dim=-1).unsqueeze(-1)

        b, v, c, n = all_template_feat.shape
        template_view_feat = all_template_feat.view(-1, c, n).contiguous()
        b, v, c, n = all_template_pe.shape
        all_template_pe = all_template_pe.view(-1, c, n).contiguous()

        all_template_pe = self.view_pos_embed(all_template_partial_shape).view(b, v, n, -1).permute(0,1,3,2).contiguous()
        b, v, c, n = all_template_pe.shape
        all_template_pe = all_template_pe.view(-1, c, n).contiguous()

        template_view_feat = self.view_adapter(template_view_feat.permute(0,2,1), \
            all_template_pe.permute(0,2,1))[0].permute(0,2,1).view(b, v, -1, n).contiguous()
        template_view_feat = F.normalize(torch.mean(template_view_feat, dim=-1), dim=-1)

        view_pred = torch.bmm(template_view_feat, query_view_feat).squeeze(-1)
        view_pred = nn.Softmax(dim=1)(view_pred)

        local_view_pred = None
        local_template_view_feat = None
        if self.training:
            b, v, c, n = local_template_feat.shape
            local_template_feat = local_template_feat.view(-1, c, n).contiguous()

            b, v, c, n = local_template_pe.shape
            local_template_pe = self.view_pos_embed(local_template_partial_shape).view(b, v, n, -1).permute(0,1,3,2).contiguous()
            b, v, c, n = local_template_pe.shape
            local_template_pe = local_template_pe.view(-1, c, n).contiguous()

            local_template_view_feat = self.view_adapter(local_template_feat.permute(0,2,1), \
                local_template_pe.permute(0,2,1))[0].permute(0,2,1).view(b, v, -1, n).contiguous()
            local_template_view_feat = F.normalize(torch.mean(local_template_view_feat, dim=-1), dim=-1)

            local_view_pred = torch.bmm(local_template_view_feat, query_view_feat).squeeze(-1)
            local_view_pred = nn.Softmax(dim=1)(local_view_pred)

        query_shape_feat = self.shape_encoder(query_shape_in_template_view)
        query_map_feat = self.map_adapter(query_shape_feat.permute(0,2,1), \
            sampled_query_feat_no_class_token.permute(0,2,1))[0]

        template_shape_feat = self.shape_encoder(select_template_shape_in_obj_frame)
        template_map_feat = self.map_adapter(template_shape_feat.permute(0,2,1), \
            select_template_feat.permute(0,2,1))[0]

        shape_difference = (torch.mean(query_map_feat, dim=1) - torch.mean(template_map_feat, dim=1)).unsqueeze(-1)
        deform_feat = torch.cat((template_map_feat.permute(0,2,1), shape_difference.repeat(1, 1, 1024)), dim=1)

        deformation = self.deform_decoder(deform_feat).permute(0,2,1)
        recons_nocs = select_template_nocs + deformation

        query_nocs, match_matrix = attention(query_map_feat, template_map_feat, recons_nocs)

        return {
            'view_pred': view_pred, 'nocs_pred': query_nocs, 'recons_nocs': recons_nocs, \
            'match_matrix': match_matrix, 'local_view_pred': local_view_pred, 'query_view_feat': query_view_feat.squeeze(-1), \
            'template_view_feat': template_view_feat, 'local_template_view_feat': local_template_view_feat
        }

    def estimate_viewpoint(self, query_img, choose, query_partial_shape, all_template_feat, all_template_partial_shape, all_template_pe):
        query_feats = self.__extract_foundation_feature__(query_img)
        query_feat_with_class_token, query_feat_no_class_token = self.__reshape_multiple_layer_feats__(query_feats)

        sampled_query_feat_with_class_token = self.__sample_feat_vec__(query_feat_with_class_token, choose)
        query_position_embedding = self.view_pos_embed(query_partial_shape).permute(0,2,1)

        query_view_feat = self.view_adapter(sampled_query_feat_with_class_token.permute(0,2,1), \
            query_position_embedding.permute(0,2,1))[0].permute(0,2,1)
        query_view_feat = F.normalize(torch.mean(query_view_feat, dim=-1), dim=-1).unsqueeze(-1)

        b, v, c, n = all_template_feat.shape
        template_view_feat = all_template_feat.view(-1, c, n)
        b, v, c, n = all_template_pe.shape
        all_template_pe = all_template_pe.view(-1, c, n)

        all_template_pe = self.view_pos_embed(all_template_partial_shape).view(b, v, n, -1).permute(0,1,3,2).contiguous()
        b, v, c, n = all_template_pe.shape
        all_template_pe = all_template_pe.view(-1, c, n).contiguous()

        template_view_feat = self.view_adapter(template_view_feat.permute(0,2,1), \
            all_template_pe.permute(0,2,1))[0].permute(0,2,1).view(b, v, -1, n).contiguous()
        template_view_feat = F.normalize(torch.mean(template_view_feat, dim=-1), dim=-1)

        view_pred = torch.bmm(template_view_feat, query_view_feat).squeeze(-1)
        norm_view_pred = nn.Softmax(dim=1)(view_pred)

        return {'view_pred': norm_view_pred, 'view_pred_before_softmax': view_pred}
    
    def estimate_nocs_map(self, query_img, choose, select_template_feat, select_template_nocs, \
        query_shape_in_template_view, select_template_shape_in_obj_frame, template_complete_nocs):
        query_feats = self.__extract_foundation_feature__(query_img)
        query_feat_with_class_token, query_feat_no_class_token = self.__reshape_multiple_layer_feats__(query_feats)

        sampled_query_feat_no_class_token = self.__sample_feat_vec__(query_feat_no_class_token, choose)
        
        query_shape_feat = self.shape_encoder(query_shape_in_template_view)
        query_map_feat = self.map_adapter(query_shape_feat.permute(0,2,1), \
            sampled_query_feat_no_class_token.permute(0,2,1))[0]

        template_shape_feat = self.shape_encoder(select_template_shape_in_obj_frame)
        template_map_feat = self.map_adapter(template_shape_feat.permute(0,2,1), \
            select_template_feat.permute(0,2,1))[0]

        shape_difference = (torch.mean(query_map_feat, dim=1) - torch.mean(template_map_feat, dim=1)).unsqueeze(-1)
        deform_feat = torch.cat((template_map_feat.permute(0,2,1), shape_difference.repeat(1, 1, 1024)), dim=1)

        deformation = self.deform_decoder(deform_feat).permute(0,2,1)
        recons_nocs = select_template_nocs + deformation

        query_nocs, match_matrix = attention(query_map_feat, template_map_feat, recons_nocs)

        return {'nocs_pred': query_nocs, 'recons_nocs': recons_nocs, 'match_matrix': match_matrix}
