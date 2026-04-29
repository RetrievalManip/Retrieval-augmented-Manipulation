import os
import cv2
import random
import json
import numpy as np
import _pickle as cPickle
from PIL import Image
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from tqdm import tqdm

from ram_utils import get_bbox, transform_coordinates_3d

TEST_CATEGORY_NAMES = ('bed', 'stove')
SUPPORTED_FOUNDATION_MODEL = 'dinov2-b14'
PATCH_NUM = 224 // 14

def reshape_dinov2_b14_feats(feats, layers, feat_type, patch_num):
    select_feats_with_class_token = []
    select_feats_no_class_token = []
    for l in layers:
        if l not in feats or feat_type not in feats[l]:
            raise KeyError(f"Missing DINOv2-B/14 feature layer {l} token '{feat_type}'.")

        b, h, n, d = feats[l][feat_type].shape
        expected_token_count = patch_num * patch_num + 1
        if n != expected_token_count:
            raise ValueError(
                f"Unexpected DINOv2-B/14 token count for layer {l}: {n}. "
                f"Expected {expected_token_count} for a {patch_num}x{patch_num} patch grid plus class token."
            )

        token = feats[l][feat_type].transpose(0, 1, 3, 2)
        class_token = token[:, :, :, 0].reshape(b, h*d, 1).astype(np.float32)
        patch_token = token[:, :, :, 1:].reshape(b, h*d, -1).astype(np.float32)
        class_token = np.tile(class_token, (1,1,patch_token.shape[-1]))

        feat_with_class_token = (patch_token + class_token).reshape(b, -1, patch_num, patch_num)
        feat_no_class_token = patch_token.reshape(b, -1, patch_num, patch_num)

        select_feats_with_class_token.append(feat_with_class_token)
        select_feats_no_class_token.append(feat_no_class_token)
    
    select_feats_with_class_token = np.concatenate(select_feats_with_class_token, axis=1)
    select_feats_no_class_token = np.concatenate(select_feats_no_class_token, axis=1)

    return select_feats_with_class_token, select_feats_no_class_token

def sample_support_information(mask, feat_map_with_class_token, feat_map_np_class_token, \
    nocs_map, depth, K, depth_scale = 0.1, n_pts = 1024):
    h, w = mask.shape
    choose = mask.flatten().nonzero()[0]
    if len(choose) > n_pts:
        c_mask = np.zeros(len(choose), dtype=int)
        c_mask[:n_pts] = 1
        np.random.shuffle(c_mask)
        choose = choose[c_mask.nonzero()]
    else:
        choose = np.pad(choose, (0, n_pts-len(choose)), 'wrap')

    feat_map_with_class_token = F.interpolate(torch.from_numpy(feat_map_with_class_token), \
        size = (h, w), mode = 'bilinear', align_corners=False)
    
    _, d, _, _ = feat_map_with_class_token.shape
    feat_map_with_class_token = feat_map_with_class_token.squeeze().permute(1,2,0).numpy()
    feat_with_class_token = feat_map_with_class_token.reshape((-1, d))[choose, :]

    feat_map_np_class_token = F.interpolate(torch.from_numpy(feat_map_np_class_token), \
        size = (h, w), mode = 'bilinear', align_corners=False)
    
    _, d, _, _ = feat_map_np_class_token.shape
    feat_map_np_class_token = feat_map_np_class_token.squeeze().permute(1,2,0).numpy()
    feat_no_class_token = feat_map_np_class_token.reshape((-1, d))[choose, :]

    coord = nocs_map.astype(np.float32) / 255
    nocs = coord.reshape((-1, 3))[choose, :] - 0.5

    xmap = np.array([[i for i in range(w)] for j in range(h)])
    ymap = np.array([[j for i in range(w)] for j in range(h)])
    depth_masked = depth.flatten()[choose][:, np.newaxis]
    xmap_masked = xmap.flatten()[choose][:, np.newaxis]
    ymap_masked = ymap.flatten()[choose][:, np.newaxis]
    pt2 = depth_masked * depth_scale
    pt0 = (xmap_masked - K[0,2]) * pt2 / K[0,0]
    pt1 = (ymap_masked - K[1,2]) * pt2 / K[1,1]
    points = np.concatenate((pt0, pt1, pt2), axis=1)
    return feat_with_class_token, feat_no_class_token, nocs, points, choose

def position_encoding(emb_dim, max_length=60000):
    position = np.arange(max_length)[:, np.newaxis]
    positional_encoding = np.zeros((1, max_length, emb_dim)).astype(np.float32)

    _2i = np.arange(0, emb_dim, step=2).astype(np.float32)
    positional_encoding[0, :, 0::2] = np.sin(position / (10000 ** (_2i / emb_dim)))
    positional_encoding[0, :, 1::2] = np.cos(position / (10000 ** (_2i / emb_dim)))

    return positional_encoding[0]

def load_template_information(template_file, pe_table, feat_layers, feat_type, patch_num, n_pts):
    dino_feat_vector_with_class_token = []
    dino_feat_vector_no_class_token = []
    nocs_vector = []
    template_pts = []
    poses = []
    position_encodings = []

    for i in range(len(template_file)):
        poses.append(template_file[i]['object_pose'])
        dino_feat_with_class_token, dino_feat_no_class_token = \
            reshape_dinov2_b14_feats(template_file[i]['dinov2_b14_feat'], feat_layers, feat_type, patch_num)

        dino_vec_with_class_token, dino_vec_no_class_token, nocs_vec, pts, choose = \
            sample_support_information(template_file[i]['mask'], dino_feat_with_class_token, \
            dino_feat_no_class_token, template_file[i]['nocs_map'][:, :, (2, 1, 0)], \
            template_file[i]['depth'], template_file[i]['K'], 0.1, n_pts)
        
        dino_feat_vector_with_class_token.append(dino_vec_with_class_token[np.newaxis, :, :])
        dino_feat_vector_no_class_token.append(dino_vec_no_class_token[np.newaxis, :, :])
        nocs_vector.append(nocs_vec[np.newaxis, :, :])
        template_pts.append(pts[np.newaxis, :, :])
        position_encodings.append(pe_table[choose, :][np.newaxis, :, :])

    dino_vecs_with_class_token = np.concatenate(dino_feat_vector_with_class_token, axis=0)
    dino_vecs_no_class_token = np.concatenate(dino_feat_vector_no_class_token, axis=0)
    nocs_vecs = np.concatenate(nocs_vector, axis=0)
    template_pts = np.concatenate(template_pts, axis=0)
    position_encodings = np.concatenate(position_encodings, axis=0)

    template = {}
    template['feat_vec_with_class_token'] = dino_vecs_with_class_token.astype(np.float16)
    template['feat_vec_no_class_token'] = dino_vecs_no_class_token.astype(np.float16)
    template['nocs_vec'] = nocs_vecs
    template['pts'] = template_pts
    template['reference_poses'] = poses
    template['position_encoding'] = position_encodings

    return template

class RAMBOPDataset(data.Dataset):
    def __init__(self, source, mode, data_dir, n_pts, img_size, feat_layers = [7, 9, 11], \
        feat_type = 'k', foundation_model = SUPPORTED_FOUNDATION_MODEL, template_views = 128):
        self.source = source
        self.mode = mode
        self.data_dir = data_dir
        self.n_pts = n_pts
        self.img_size = img_size

        self.layers = feat_layers
        self.feat_type = feat_type
        self.template_views = template_views

        self.foundation_model = foundation_model
        if self.foundation_model != SUPPORTED_FOUNDATION_MODEL:
            raise ValueError(
                f"Unsupported foundation model '{self.foundation_model}'. "
                f"Training only supports {SUPPORTED_FOUNDATION_MODEL} without register tokens."
            )

        self.pe_table = position_encoding(len(self.layers) * 384)
        self.patch_num = PATCH_NUM
        
        assert source in ['bop', 'real']
        assert mode in ['train', 'test']

        img_list_path = self.source + '_' + self.mode + '_list.txt'
        cad_model_file_path = self.source + '_cad_model.pkl'

        with open(os.path.join(data_dir, img_list_path)) as f:
            img_list = [line.rstrip('\n') for line in f]

        self.img_list = img_list
        self.length = len(self.img_list)

        models = {}
        with open(os.path.join(self.data_dir, cad_model_file_path), 'rb') as f:
            models.update(cPickle.load(f))
        self.models = models

        obj_meta_path = os.path.join(self.data_dir, 'cad_models', 'model_meta.json')
        if not os.path.exists(obj_meta_path):
            obj_meta_path = os.path.join(self.data_dir, 'models', 'model_meta.json')
        with open(obj_meta_path) as f:
            self.objs_meta = json.load(f)
        
        self.intrinsics = [572.411363389757, 573.5704328585578, 325.2611083984375, 242.04899588216654]
        self.depth_scale = 0.1
        self.colorjitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
        
        self.template_dir = os.path.join(self.data_dir, 'dinov2_b14_template')

        self.training_category_config, self.testing_category_config = self.__build_category_configs__()

        self.template = {}
        self.template_id = {}
        self.__preload__template__()

    def __len__(self):
        return self.length
    
    def change_template(self):
        self.__preload__template__()

    def __build_category_configs__(self):
        category_config = {}
        for obj_id, obj_meta in sorted(self.objs_meta.items(), key=lambda item: int(item[0])):
            category_name = obj_meta['category_name']
            category_config.setdefault(category_name, []).append(int(obj_id))

        missing_test_categories = sorted(set(TEST_CATEGORY_NAMES) - set(category_config.keys()))
        if missing_test_categories:
            raise ValueError(
                f"Testing categories {missing_test_categories} are not present in model_meta.json"
            )

        testing_config = {name: category_config[name] for name in TEST_CATEGORY_NAMES}
        training_config = {
            name: ids
            for name, ids in category_config.items()
            if name not in testing_config
        }

        return training_config, testing_config
    
    def __preload__template__(self):
        if self.mode == 'train':
            cate_config = self.training_category_config
        else:
            cate_config = self.testing_category_config

        self.template = {}
        self.template_id = {}
        for cate in tqdm(cate_config.keys()):
            template_obj_id = random.choice(cate_config[cate])
            template_path = os.path.join(self.template_dir, str(template_obj_id).zfill(6) + '.pkl')
            if not os.path.exists(template_path):
                raise FileNotFoundError(
                    f"Template file for category '{cate}' and object id {template_obj_id} not found: {template_path}"
                )
            with open(template_path, 'rb') as f:
                template_pkl = cPickle.load(f)

            self.template_id[cate] = str(template_obj_id)
            self.template[cate] = load_template_information(template_pkl, self.pe_table, self.layers, \
                self.feat_type, self.patch_num, self.n_pts)
    
    def __compute_orientation_error__(self, template_r, query_r, is_symmetry):
        if is_symmetry:
            z = np.array([0, 0, 1])
            z1 = template_r @ z
            z2 = query_r @ z
            cos_theta = z1.dot(z2) / (np.linalg.norm(z1) * np.linalg.norm(z2))
        else:
            R = template_r @ query_r.transpose()
            cos_theta = (np.trace(R) - 1) / 2

        delt_theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

        return delt_theta
    
    def __normalize_shape_pts__(self, pts):
        if len(pts.shape) == 3:
            center = np.mean(pts, axis=1)
            pts = pts - center[:, np.newaxis, :]
            s = np.max(np.sqrt(np.sum(pts ** 2, axis=2)), axis=1)
            pts = pts / s[:, np.newaxis, np.newaxis]
        
        if len(pts.shape) == 2:
            center = np.mean(pts, axis=0)
            pts = pts - center[np.newaxis, :]
            s = np.max(np.sqrt(np.sum(pts ** 2, axis=1)))
            pts = pts / s

        return pts

    def __getitem__(self, index):
        img_path = self.img_list[index]

        root_dir, obj_cate, obj_id, img_idx = img_path.split(',')

        gt_path = os.path.join(root_dir, 'scene_gt.json')
        with open(gt_path) as f:
            gts = json.load(f)
        
        gt_r = np.array(gts[str(int(img_idx))][0]['cam_R_m2c']).reshape(3, 3)
        gt_t = np.array(gts[str(int(img_idx))][0]['cam_t_m2c']).flatten()
        assert (str(gts[str(int(img_idx))][0]['obj_id']) == obj_id)

        rgb_path = os.path.join(root_dir, 'rgb', img_idx + '.jpg')
        mask_path = os.path.join(root_dir, 'mask', img_idx + '_000000.png')
        depth_path = os.path.join(root_dir, 'depth', img_idx + '.png')
        nocs_path = os.path.join(root_dir, 'nocs', img_idx + '.png')

        rgb = cv2.imread(rgb_path)[:, :, :3]
        rgb = rgb[:, :, ::-1]
        h, w, _ = rgb.shape

        cam_fx, cam_fy, cam_cx, cam_cy = self.intrinsics

        xmap = np.array([[i for i in range(w)] for j in range(h)])
        ymap = np.array([[j for i in range(w)] for j in range(h)])

        depth = cv2.imread(depth_path, -1)
        
        mask = cv2.imread(mask_path, -1)

        nocs = cv2.imread(nocs_path, -1)[:, :, ::-1].astype(np.float32) / 255

        ys, xs = np.nonzero(mask)
        y1 = np.min(ys)
        y2 = np.max(ys)
        x1 = np.min(xs)
        x2 = np.max(xs)
        rmin, rmax, cmin, cmax = get_bbox([y1, x1, y2, x2], img_w = w, img_h = h)

        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        if len(choose) > self.n_pts:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.n_pts] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.n_pts-len(choose)), 'wrap')
        
        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
        xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
        ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
        pt2 = depth_masked * self.depth_scale
        pt0 = (xmap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (ymap_masked - cam_cy) * pt2 / cam_fy
        points = np.concatenate((pt0, pt1, pt2), axis=1)
        query_shape_pts = self.__normalize_shape_pts__(points.copy())

        nocs = nocs[rmin:rmax, cmin:cmax, :].reshape((-1, 3))[choose, :] - 0.5
        rgb = rgb[rmin:rmax, cmin:cmax, :]
        rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        crop_w = rmax - rmin
        ratio = self.img_size / crop_w
        col_idx = choose % crop_w
        row_idx = choose // crop_w
        choose = (np.floor(row_idx * ratio) * self.img_size + np.floor(col_idx * ratio)).astype(np.int64)

        if self.mode == 'train':
            rgb = self.colorjitter(Image.fromarray(np.uint8(rgb)))
            rgb = np.array(rgb)
        
        rgb = self.transform(rgb)

        obj_meta = self.objs_meta[obj_id]
        obj_scale = obj_meta['object_scale']
        is_symmetry = (obj_meta['is_symmetric'] == 'True')

        template_complete_nocs = self.models[self.template_id[obj_cate]] / \
            self.objs_meta[self.template_id[obj_cate]]['object_scale']
        query_complete_nocs = self.models[obj_id] / obj_scale
        
        template_orientation_errors = []
        for i in range(self.template_views):
            template_orientation_errors.append(self.__compute_orientation_error__(\
                self.template[obj_cate]['reference_poses'][i][:3, :3], \
                gt_r, is_symmetry))
        
        sorted_view_id = np.array(template_orientation_errors).argsort()
        top_k_closese_view_id = sorted_view_id[:5]
        closest_view = top_k_closese_view_id[0]
        select_template_view = random.choice(list(top_k_closese_view_id))

        select_template_r = self.template[obj_cate]['reference_poses'][select_template_view][:3, :3]

        template_orientation_errors = -10.0 * np.array(template_orientation_errors)
        softmax_template_orientation_errors = np.exp(template_orientation_errors - np.max(template_orientation_errors))
        softmax_template_orientation_errors = softmax_template_orientation_errors / softmax_template_orientation_errors.sum(axis=0)
        softmax_template_orientation_errors = softmax_template_orientation_errors.astype(np.float32)

        sv = 8
        step = self.template_views // sv
        sampled_template_views = []
        for i in range(sv):
            a = int(i * step)
            b = int((i + 1) * step)
            current_view = random.choice(list(sorted_view_id[a:b]))
            if i == 0:
                sampled_template_views.append(closest_view)
            else:
                sampled_template_views.append(current_view)
        
        sampled_orientation_errors = np.array(template_orientation_errors[sampled_template_views])
        sampled_softmax_template_orientation_errors = np.exp(sampled_orientation_errors - np.max(sampled_orientation_errors))
        sampled_softmax_template_orientation_errors = sampled_softmax_template_orientation_errors / sampled_softmax_template_orientation_errors.sum(axis=0)
        sampled_softmax_template_orientation_errors = sampled_softmax_template_orientation_errors.astype(np.float32)

        sampled_hard_template_views = sorted_view_id[:sv]
        sampled_hard_orientation_errors = np.array(template_orientation_errors[sampled_hard_template_views])
        sampled_softmax_hard_template_orientation_errors = np.exp(sampled_hard_orientation_errors - np.max(sampled_hard_orientation_errors))
        sampled_softmax_hard_template_orientation_errors = sampled_softmax_hard_template_orientation_errors / sampled_softmax_hard_template_orientation_errors.sum(axis=0)
        sampled_softmax_hard_template_orientation_errors = sampled_softmax_hard_template_orientation_errors.astype(np.float32)

        template_meta = self.objs_meta[self.template_id[obj_cate]]
        template_obj_scale = template_meta['object_scale']

        template_shape_pts = self.__normalize_shape_pts__(self.template[obj_cate]['pts'])

        template_feat_with_class_token = self.template[obj_cate]['feat_vec_with_class_token'].transpose(0,2,1)
        template_choose = self.template[obj_cate]['position_encoding'].transpose(0,2,1)
        template_feat_no_class_token = self.template[obj_cate]['feat_vec_no_class_token'].transpose(0,2,1)
        template_nocs_vec = self.template[obj_cate]['nocs_vec']
        template_obj_pose = self.template[obj_cate]['reference_poses']

        query_complete_nocs = query_complete_nocs.astype(np.float32)
        template_complete_nocs = template_complete_nocs.astype(np.float32)
        nocs = nocs.astype(np.float32)
        points = points.astype(np.float32)
        query_shape_pts = query_shape_pts.astype(np.float32)

        template_shape_pts = template_shape_pts.astype(np.float32)
        select_template_nocs = template_nocs_vec[select_template_view].astype(np.float32)

        select_template_rt = np.eye(4).astype(np.float32)
        select_template_rt[:3, :3] = select_template_r

        select_template_partial_shape_in_obj_frame = \
            transform_coordinates_3d(template_shape_pts[select_template_view].T, \
            np.linalg.inv(select_template_rt)).T

        query_partial_shape_in_template_view = \
            transform_coordinates_3d(query_shape_pts.T, \
            np.linalg.inv(select_template_rt)).T

        return {
            'query_pts': points, 'query_rgb': rgb, 'query_choose': choose, \
            'query_nocs': nocs, 'query_complete_nocs': query_complete_nocs, \
            'query_partial_shape': query_shape_pts, \
            'template_view_score': softmax_template_orientation_errors, \
            'all_template_feat': template_feat_with_class_token, \
            'position_encoding': template_choose, \
            'select_template_feat': template_feat_no_class_token[select_template_view], \
            'all_template_partial_shape': template_shape_pts, \
            'select_template_partial_shape': template_shape_pts[select_template_view], \
            'select_template_nocs': select_template_nocs, \
            'template_complete_nocs': template_complete_nocs, \
            'r': gt_r, 't': gt_t, 's': obj_scale, \
            'template_r': template_obj_pose[select_template_view][:3, :3], \
            'template_t': template_obj_pose[select_template_view][:3, 3].flatten(), \
            'template_scale': template_obj_scale, \
            'is_symmetry': float(is_symmetry), \
            'sampled_template_view_score': sampled_softmax_template_orientation_errors, \
            'sampled_template_feat': template_feat_with_class_token[sampled_template_views], \
            'sampled_template_partial_shape': template_shape_pts[sampled_template_views], \
            'sampled_position_encoding': template_choose[sampled_template_views], \
            'hard_sampled_template_view_score': sampled_softmax_hard_template_orientation_errors, \
            'hard_sampled_template_feat': template_feat_with_class_token[sampled_hard_template_views], \
            'hard_sampled_template_partial_shape': template_shape_pts[sampled_hard_template_views], \
            'hard_sampled_position_encoding': template_choose[sampled_hard_template_views], \
            'top_1_view_choose': np.array([0]).astype(np.int64), \
            'view_error': np.array(-sampled_orientation_errors / 10.0).copy(), \
            'hard_view_error': np.array(-sampled_hard_orientation_errors / 10.0).copy(), \
            'select_template_shape_in_obj_frame': select_template_partial_shape_in_obj_frame, \
            'query_shape_in_template_view': query_partial_shape_in_template_view
        }
