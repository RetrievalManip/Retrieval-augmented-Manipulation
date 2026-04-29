import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_

import sys
sys.path.append('..')

from pointnet2_utils_ram import PointNetFeaturePropagation, farthest_point_sample, index_points


def knn(pc, center, n_neighbors=32):  
    dist = torch.cdist(center, pc) 
    neigbhors = dist.topk(k=n_neighbors,dim=2,largest=False) 
    return neigbhors.indices

def fps(data, number):

    fps_idx = farthest_point_sample(data, number)
    return index_points(data, fps_idx)

class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz):
        batch_size, num_points, _ = xyz.shape
        center = fps(xyz, self.num_group)

        idx = knn(xyz, center, self.group_size)

        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center

class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., fetch_list = [3, 7, 11]):
        super().__init__()
        
        self.fetch_list = fetch_list
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    def forward(self, x, pos):
        feature_list = []
        for i, block in enumerate(self.blocks):
            if pos is None:
                x = block(x)
            else:
                x = block(x + pos)
            if i in self.fetch_list:
                feature_list.append(x)
        return feature_list

class MapPosTransformer(nn.Module):
    def __init__(self, depth=12, num_heads=4, group_size = 32, num_group = 128, \
        fetch_list = [3,7,11], output_dim = 768, use_pose_embed = True, fix_pose_embed = False):
        super().__init__()

        self.trans_dim = 384
        self.depth = depth
        self.drop_path_rate = 0.1
        self.num_heads = num_heads
        self.fetch_list = fetch_list
        self.use_pose_embed = use_pose_embed
        self.fix_pose_embed = fix_pose_embed

        self.group_size = group_size
        self.num_group = num_group
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.encoder_dims = 384
        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        if self.use_pose_embed and self.fix_pose_embed:
            self.pos_embed = nn.Sequential(
                nn.Linear(3, 128),
                nn.GELU(),
                nn.Linear(128, self.trans_dim)
            )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
            fetch_list = self.fetch_list
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.propagation_0 = PointNetFeaturePropagation(in_channel=384*(len(self.fetch_list)) + 3,
                                                        mlp=[self.trans_dim * 4, 1024])

        self.convs1 = nn.Conv1d(1024+384*(len(self.fetch_list))*2, 512, 1)
        self.convs2 = nn.Conv1d(512, 512, 1)
        self.convs3 = nn.Conv1d(512, output_dim, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(512)

        self.relu = nn.ReLU()
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def init_from_pretrain(self, pretrain_path):
        if pretrain_path is not None:
            ckpt = torch.load(pretrain_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)
            print(f'Successful Loading the pre-trained point cloud Transformer from {pretrain_path}')

    def forward(self, pts, neighborhood = None, center = None, pos = None):
        B, N, C = pts.shape


        if neighborhood == None or center == None:
            neighborhood, center = self.group_divider(pts)

        group_input_tokens = self.encoder(neighborhood)

        if self.use_pose_embed:
            if self.fix_pose_embed:
                pos = self.pos_embed(center)
            else:
                if pos is None:
                    raise Exception("Sorry, a valid position embedding should be given")
                    return

        x = group_input_tokens
        feature_list = self.blocks(x, pos)

        feature_list = [self.norm(x).transpose(-1, -2).contiguous() for x in feature_list]
        x = torch.cat(feature_list, dim=1)

        x_max = torch.max(x,2)[0]
        x_avg = torch.mean(x,2)
        x_max_feature = x_max.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
        x_avg_feature = x_avg.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
        x_global_feature = torch.cat((x_max_feature, x_avg_feature), 1)

        f_level_0 = self.propagation_0(pts.transpose(-1, -2), center.transpose(-1, -2), pts.transpose(-1, -2), x)

        x = torch.cat((f_level_0, x_global_feature), 1)
        x = self.relu(self.bns1(self.convs1(x)))
        x = self.relu(self.bns2(self.convs2(x)))
        x = self.convs3(x)
        return x
