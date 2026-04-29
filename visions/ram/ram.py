import os
import sys
import cv2
import numpy as np
import open3d as o3d
import pickle
import argparse
import scipy.io as scio
from PIL import Image
import logging
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'lib'))
sys.path.append(os.path.join(ROOT_DIR, 'data'))

from network import RAMNet
from ram_utils import transform_coordinates_3d
from align import estimateSimilarityTransform
from bop_dataset import position_encoding
from grasp import find_grasp_point_with_ram, get_gripper_lines, visualize_grasp_point_on_image
from support_plane import find_support_plane_with_ram, draw_plane_grid_on_image, draw_3d_arrow_on_image
from ram_utils import ram_draw_detections

from utils.config import get_ram_config
from utils.config import resolve_project_path


class BaseRAM():
    def __init__(self):
        ram_config = get_ram_config()

        self.foundation_model = ram_config.foundation_model
        self.feat_layer = list(ram_config.feat_layer)
        self.feat_type = 'k'
        self.img_size = ram_config.img_size
        self.n_pts = ram_config.n_pts
        self.pathc_size = 14
        self.patch_num = int(self.img_size // self.pathc_size)
        self.feature_dim = self._foundation_feature_dim()
        self.position_encoding_dim = 3 * 384
        self.pe_table = position_encoding(self.position_encoding_dim)

    def _foundation_feature_dim(self):
        if self.foundation_model != 'dinov2-b14':
            raise ValueError(
                f"Unsupported RAM foundation model '{self.foundation_model}'. "
                "This project currently supports only dinov2-b14 without register tokens."
            )
        return len(self.feat_layer) * 768


class RAMTemplate(BaseRAM):
    def __init__(self, config_file=None):
        super().__init__()

        self.shape = None
        self.foundation = None
        self.semantic = None
        self.function = None
        self.grasp = None

        if config_file is not None:
            self.read_ram_config(config_file)
            logging.info("==> RAM template configuration loaded from %s", config_file)
        else:
            unified_config = get_ram_config()
            self._load_template_config(unified_config.template_dir)
            logging.info("==> RAM template configuration loaded from unified config.yaml")

    def _load_template_config(self, template_dir):
        if not template_dir:
            raise ValueError("RAM template_dir is not configured.")

        self.template_root = resolve_project_path(template_dir)
        self.template_dir = os.path.join(self.template_root, 'objects')
        template_config_path = os.path.join(self.template_root, 'template.json')
        self.template_config_path = template_config_path

        if not os.path.exists(template_config_path):
            raise FileNotFoundError(
                f"RAM template metadata not found: {template_config_path}. "
                "Keep template.json under visions/ram/template or update template_dir."
            )

        with open(template_config_path, 'r', encoding='utf-8') as f:
            self.template_info = json.load(f)
    
    def read_ram_config(self, config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            ram_config = json.load(f)

        self._load_template_config(ram_config.get('template_dir'))

    def _template_id_name(self):
        return str(self.template_id).zfill(6)

    def _template_asset_dir(self):
        return os.path.join(self.template_dir, self.template_name, self._template_id_name())

    def _ensure_template_asset_dir(self):
        template_asset_dir = self._template_asset_dir()
        if not os.path.isdir(template_asset_dir):
            raise FileNotFoundError(
                f"RAM template assets not found for '{self.template_name}'. "
                f"Expected directory: {template_asset_dir}. "
                "Download or copy the large template payloads into visions/ram/template/objects/."
            )
        return template_asset_dir
    
    def load_template(self, template_category):
        template_info = self.template_info.get(template_category)
        if template_info is None:
            available = ', '.join(sorted(self.template_info.keys()))
            raise KeyError(
                f"RAM template category '{template_category}' is not defined in "
                f"{self.template_config_path}. Available categories: {available}"
            )

        self.template_name = template_category
        self.template_id = template_info['template_id']
        self.template_scale = template_info['object_scale']

        is_symmetric = template_info.get('is_symmetric', False)
        self.is_symmetric = str(is_symmetric).lower() == 'true' if isinstance(is_symmetric, str) else bool(is_symmetric)

        self.symmetry_axis = template_info['symmetry_axis']

        self.symmetry_type = template_info['symmetry_type']

        self._ensure_template_asset_dir()

        self.shape = self.__load_template_ply()
        self.grasp = self.__load_template_grasp()
        self.function = self.__load_template_function()
        self.foundation = self.__load_template_foundation()

    def __load_template_foundation(self):
        template_asset_dir = self._ensure_template_asset_dir()
        template_id = self._template_id_name()
        foundation_path = os.path.join(template_asset_dir, template_id + '.pkl')
        
        if not os.path.exists(foundation_path):
            raise FileNotFoundError(
                f"RAM template foundation not found for '{self.template_name}': {foundation_path}. "
                "Generate it with tools/template_renderer/process_bop_templates.py."
            )

        with open(foundation_path, 'rb') as f:
            template_foundation = pickle.load(f)
        logging.info("Completed loading template foundation for %s", self.template_name)
        return template_foundation
    
    def __load_template_ply(self):
        template_asset_dir = self._ensure_template_asset_dir()
        ply_path = os.path.join(template_asset_dir, 'obj_' + self._template_id_name() + '.ply')
        if not os.path.exists(ply_path):
            raise FileNotFoundError(
                f"RAM template mesh not found for '{self.template_name}': {ply_path}"
            )

        obj_mesh = o3d.io.read_triangle_mesh(ply_path)
        if len(obj_mesh.vertices) == 0:
            raise ValueError(f"RAM template mesh is empty or unreadable: {ply_path}")

        pcd = o3d.geometry.TriangleMesh.sample_points_uniformly(obj_mesh, self.n_pts)
        complete_template_shape = np.array(pcd.points)

        return complete_template_shape
    
    def __load_template_function(self):
        support_info_path = os.path.join(self.template_dir, self.template_name, str(self.template_id).zfill(6), \
            str(self.template_id).zfill(6) + '.json')
        if os.path.exists(support_info_path):
            with open(support_info_path, 'r') as f:
                support_info = json.load(f)
                f.close()

            template_primitives = support_info.get('primitives', {})
            template_function_planes = {}
            for k in template_primitives.keys():
                if 'plane' in k:
                    template_function_planes[k] = {}
                    template_function_planes[k]['position'] = template_primitives[k]['position']
                    template_function_planes[k]['orientation'] = template_primitives[k]['orientation']
                    if 'align_with_axis' in template_primitives[k].keys():
                        template_function_planes[k]['align_with_axis'] = template_primitives[k]['align_with_axis']
                    else:
                        template_function_planes[k]['align_with_axis'] = None
        else:
            template_function_planes = {}
            logging.info("Do not contain template inforamtion for %s", self.template_name)
        
        return template_function_planes

    
    def __load_template_grasp(self):
        support_info_path = os.path.join(self.template_dir, self.template_name, str(self.template_id).zfill(6), \
            str(self.template_id).zfill(6) + '.json')
        
        if os.path.exists(support_info_path):
            with open(support_info_path, 'r') as f:
                support_info = json.load(f)
                f.close()

            template_primitives = support_info.get('primitives', {})
            template_function_grasps = {}
            for k in template_primitives.keys():
                if 'grasp' in k or 'contact_point' in k or 'hinge' in k:
                    template_function_grasps[k] = {}
                    template_function_grasps[k]['position'] = template_primitives[k]['position']
                    template_function_grasps[k]['orientation'] = template_primitives[k]['orientation']
                    if 'rotation_axis_direction' in template_primitives[k]:
                        template_function_grasps[k]['rotation_axis_direction'] = template_primitives[k]['rotation_axis_direction']
                    if 'axis_point' in template_primitives[k]:
                        template_function_grasps[k]['axis_point'] = template_primitives[k]['axis_point']
                    elif 'rotation_axis_point' in template_primitives[k]:
                        template_function_grasps[k]['axis_point'] = template_primitives[k]['rotation_axis_point']
        else:
            template_function_grasps = {}
            logging.info("Do not contain template inforamtion for %s", self.template_name)
        
        return template_function_grasps
    
    def __load_template_semantic(self):
        semantic_info_path = os.path.join(self.template_dir, self.template_name, str(self.template_id).zfill(6), \
            'functions', 'semantics', str(self.template_id).zfill(6) + '_semantic.pkl')
        
        if os.path.exists(semantic_info_path):
            with open(semantic_info_path, 'rb') as f:
                template_semantics = pickle.load(f)
                f.close()
        else:
            template_semantics = None
            logging.info("Do not contain template semantic for %s", self.template_name)

        return template_semantics


class RAMIO(BaseRAM):
    def __init__(self):
        super().__init__()
        self.norm_color = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )
        self.bbox = None
    
    def normalize_pts(self, pts):
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
    
    def __get_bbox(self, bbox, img_w = 640, img_h = 480):
        
        y1, x1, y2, x2 = bbox
        img_height = img_h
        img_width = img_w
        window_size = (max(y2-y1, x2-x1) // 40 + 1) * 40
        window_size = min(window_size, min(img_h, img_w))
        center = [(y1 + y2) // 2, (x1 + x2) // 2]
        rmin = center[0] - int(window_size / 2)
        rmax = center[0] + int(window_size / 2)
        cmin = center[1] - int(window_size / 2)
        cmax = center[1] + int(window_size / 2)
        if rmin < 0:
            delt = -rmin
            rmin = 0
            rmax += delt
        if cmin < 0:
            delt = -cmin
            cmin = 0
            cmax += delt
        if rmax > img_height:
            delt = rmax - img_height
            rmax = img_height
            rmin -= delt
        if cmax > img_width:
            delt = cmax - img_width
            cmax = img_width
            cmin -= delt
        return rmin, rmax, cmin, cmax
    
    def __extract_bbox(self, mask):
        ys, xs = np.nonzero(mask)
        y1 = np.min(ys)
        y2 = np.max(ys)
        x1 = np.min(xs)
        x2 = np.max(xs)
        rmin, rmax, cmin, cmax = self.__get_bbox((y1, x1, y2, x2), \
            mask.shape[1], mask.shape[0])
        
        return rmin, rmax, cmin, cmax

    def extract_rgb_patch(self, rgb, mask):
        self.bbox = self.__extract_bbox(mask)
        rmin, rmax, cmin, cmax = self.bbox
        inst_rgb = rgb[rmin:rmax, cmin:cmax, :]
        inst_rgb = cv2.resize(inst_rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        return self.norm_color(inst_rgb)
    
    def extract_obj_pts(self, mask, depth, depth_scale, K, denoise_pts = True):
        valid_mask = np.logical_and(mask, depth > 0)
        rmin, rmax, cmin, cmax = self.bbox
        choose = valid_mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

        xmap = np.array([[i for i in range(depth.shape[1])] for j in range(depth.shape[0])])
        ymap = np.array([[j for i in range(depth.shape[1])] for j in range(depth.shape[0])])

        if len(choose) > 0:
            if denoise_pts:
                init_n_pts = int(self.n_pts * 1.25)
                if len(choose) > init_n_pts:
                    c_mask = np.zeros(len(choose), dtype=int)
                    c_mask[:init_n_pts] = 1
                    np.random.shuffle(c_mask)
                    choose = choose[c_mask.nonzero()]
                else:
                    choose = np.pad(choose, (0, init_n_pts-len(choose)), 'wrap')
                
                depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
                xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
                ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
                pt2 = depth_masked / depth_scale
                pt0 = (xmap_masked - K[0,2]) * pt2 / K[0,0]
                pt1 = (ymap_masked - K[1,2]) * pt2 / K[1,1]
                points = np.concatenate((pt0, pt1, pt2), axis=1)

                query_pts = self.normalize_pts(points).astype(np.float32)

                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(query_pts)

                labels = point_cloud.cluster_dbscan(eps=0.1, min_points=10)
                unique_labels, counts = np.unique(labels, return_counts=True)

                valid_labels = unique_labels[unique_labels != -1]
                valid_counts = counts[unique_labels != -1]
                sorted_indices = np.argsort(valid_counts)[::-1]
                sorted_labels = valid_labels[sorted_indices]
                sorted_counts = valid_counts[sorted_indices]

                retain_threshold = int(points.shape[0] * 0.8)

                inlier_indices = []
                current_count = 0
                for label in sorted_labels:
                    label_indices = np.where(labels == label)[0]
                    inlier_indices.extend(label_indices)
                    current_count += len(label_indices)     

                    if current_count >= retain_threshold:
                        break
                
                if inlier_indices:
                    inlier_cloud = point_cloud.select_by_index(inlier_indices)
                else:
                    raise Exception(f"cannot find valid point cloud\n")
                
                choose = choose[inlier_indices]

                if len(choose) > 0:
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
                    pt2 = depth_masked / depth_scale
                    pt0 = (xmap_masked - K[0,2]) * pt2 / K[0,0]
                    pt1 = (ymap_masked - K[1,2]) * pt2 / K[1,1]
                    points = np.concatenate((pt0, pt1, pt2), axis=1)

                    crop_w = cmax - cmin
                    crop_h = rmax - rmin
                    ratio_w = self.img_size / crop_w
                    ratio_h = self.img_size / crop_h
                    col_idx = choose % crop_w
                    row_idx = choose // crop_w

                    choose = (np.floor(row_idx * ratio_h) * self.img_size + np.floor(col_idx * ratio_w)).astype(np.int64)

                    return choose, points, np.concatenate((xmap_masked, ymap_masked), axis=1)
                else:
                    raise Exception(f"empty point cloud after denoising\n")
            else:
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
                pt2 = depth_masked / depth_scale
                pt0 = (xmap_masked - K[0,2]) * pt2 / K[0,0]
                pt1 = (ymap_masked - K[1,2]) * pt2 / K[1,1]
                points = np.concatenate((pt0, pt1, pt2), axis=1)

                crop_w = cmax - cmin
                crop_h = rmax - rmin
                ratio_w = self.img_size / crop_w
                ratio_h = self.img_size / crop_h
                col_idx = choose % crop_w
                row_idx = choose // crop_w

                choose = (np.floor(row_idx * ratio_h) * self.img_size + np.floor(col_idx * ratio_w)).astype(np.int64)

                return choose, points, np.concatenate((xmap_masked, ymap_masked), axis=1) 
        else:
            raise Exception(f"empty object mask, need to check the mask image and depth map\n")


class RAMModel(BaseRAM):
    def __init__(self, config_file=None, device=None):
        super().__init__()

        ram_config = get_ram_config()

        if device is not None:
            self.device_name = device
        else:
            self.device_name = ram_config.device if ram_config.device else ("cuda:0" if torch.cuda.is_available() else "cpu")

        self.checkpoint_path = ram_config.checkpoint
        self.ram_k = ram_config.ram_k

        if config_file is not None:
            self.read_config(config_file)
            logging.info("==> RAM model configuration loaded from %s", config_file)
        else:
            logging.info("==> RAM model configuration loaded from unified config.yaml")

        self.checkpoint_path = resolve_project_path(self.checkpoint_path)

        self.device = torch.device(self.device_name)
        self.ram = self.__get_model()
    
    def normalize_pts(self, pts):
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
    
    def read_config(self, config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            ram_config = json.load(f)

        self.checkpoint_path = ram_config.get('ram_checkpoint', self.checkpoint_path)
        self.ram_k = ram_config.get('ram_k', self.ram_k)

    def __get_model(self):
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"RAM checkpoint not found: {self.checkpoint_path}. "
                "Place ram_model.pth under visions/ram/checkpoints/ or update the RAM config."
            )

        self.ram = RAMNet('dinov2-b14', self.feat_layer, \
            self.feat_type, self.img_size)

        self.ram.to(self.device)
        self.ram = nn.DataParallel(self.ram)
        self.ram.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        self.ram.eval()

        return self.ram
    
    def viewpoint_estimation(self, template, img, choose, pts):
        normalized_pts = self.normalize_pts(pts).astype(np.float32)

        f_rgb, f_choose, f_query_pts = [], [], []
        f_rgb.append(img)
        f_choose.append(choose)
        f_query_pts.append(torch.from_numpy(normalized_pts))

        f_rgb = torch.stack(f_rgb, dim=0).to(self.device)
        f_choose = torch.from_numpy(np.array(f_choose)).to(self.device)
        f_query_pts = torch.stack(f_query_pts, dim=0).to(self.device)

        f_all_template_feat, f_all_template_pts, f_all_template_pe = [], [], []
        f_all_template_feat.append(torch.from_numpy(template.foundation['feat_vec_with_class_token'].transpose(0,2,1)))
        f_all_template_pts.append(torch.from_numpy(self.normalize_pts(template.foundation['pts']).astype(np.float32)))
        f_all_template_pe.append(torch.from_numpy(template.foundation['position_encoding'].transpose(0,2,1)))
        
        f_all_template_feat = torch.stack(f_all_template_feat, dim=0).to(self.device)
        f_all_template_pts = torch.stack(f_all_template_pts, dim=0).to(self.device)
        f_all_template_pe = torch.stack(f_all_template_pe, dim=0).to(self.device)

        with torch.no_grad():
            pred = self.ram.module.estimate_viewpoint(f_rgb, f_choose, f_query_pts, f_all_template_feat, f_all_template_pts, f_all_template_pe)
            pred_view = np.argsort(pred['view_pred'].detach().cpu().numpy(), axis=1)[:, ::-1]
        
        self.pred_view = pred_view
        
        return pred_view
    
    def nocs_estimation(self, template, view_idx, img, choose, pts):
        template_r = template.foundation['reference_poses'][view_idx][:3, :3]
        template_pose = np.eye(4).astype(np.float32)
        template_pose[:3, :3] = template_r
        query_pts_in_template_view = transform_coordinates_3d(pts.T, np.linalg.inv(template_pose)).T

        f_rgb, f_choose, f_query_pts = [], [], []
        f_rgb.append(img)
        f_choose.append(choose)
        f_query_pts.append(torch.from_numpy(query_pts_in_template_view.astype(np.float32)))
        f_rgb = torch.stack(f_rgb, dim=0).to(self.device)
        f_choose = torch.from_numpy(np.array(f_choose)).to(self.device)
        f_query_pts = torch.stack(f_query_pts, dim=0).to(self.device)

        f_select_template_feat, f_select_template_nocs,  f_select_template_pts_in_obj_frame= [], [], []
        f_complete_template_nocs = []

        f_select_template_feat.append(template.foundation['feat_vec_no_class_token'].transpose(0,2,1)[view_idx])
        f_select_template_nocs.append(template.foundation['nocs_vec'][view_idx])
        template_pts_in_obj_frame = transform_coordinates_3d(self.normalize_pts(template.foundation['pts']).astype(np.float32)[view_idx].T, \
            np.linalg.inv(template_pose)).T
        f_select_template_pts_in_obj_frame.append(template_pts_in_obj_frame.astype(np.float32))

        f_select_template_feat = torch.from_numpy(np.array(f_select_template_feat)).to(self.device)
        f_select_template_nocs = torch.from_numpy(np.array(f_select_template_nocs)).to(self.device)
        f_select_template_pts_in_obj_frame = torch.from_numpy(np.array(f_select_template_pts_in_obj_frame)).to(self.device)

        f_complete_template_nocs.append(template.shape / template.template_scale)
        f_complete_template_nocs = torch.from_numpy(np.array(f_complete_template_nocs)).to(self.device)

        with torch.no_grad():
            pred = self.ram.module.estimate_nocs_map(f_rgb, f_choose, f_select_template_feat, \
                f_select_template_nocs, f_query_pts, \
                f_select_template_pts_in_obj_frame, f_complete_template_nocs)
        
        pred_nocs = pred['nocs_pred'].detach().cpu().numpy()[0]
        choose_np = f_choose.detach().cpu().numpy()
        match_matrix = F.softmax(pred['match_matrix'], dim=-1).detach().cpu().numpy()[0]

        return pred_nocs, choose_np, match_matrix
    
    def ramk_nocs_estimation(self, template, img, choose, pts):
        normalized_pts = self.normalize_pts(pts).astype(np.float32)
        best_inlier_ratios = 0.0
        
        self.best_view = self.pred_view[0,0]
        self.best_sRT = np.eye(4)
        self.best_R = np.eye(3)
        self.best_T = np.zeros(3)
        self.best_nocs = None
        self.best_pts = None
        self.best_choose = None
        self.best_match_matrix = None

        for i in range(self.ram_k):
            view_idx = self.pred_view[0,i]

            current_pred_nocs, current_choose, current_match_matrix = self.nocs_estimation(template, view_idx, img, choose, normalized_pts)

            _, current_choose = np.unique(current_choose, return_index=True)
            current_nocs_coords = current_pred_nocs[current_choose, :]

            unique_points = pts[current_choose, :]
            scale, nocs_R, nocs_T, pred_sRT, inlier_ratio, _ = estimateSimilarityTransform(current_nocs_coords, unique_points)

            pts1 = transform_coordinates_3d(unique_points.T, np.linalg.inv(pred_sRT)).T
            pts_dis = np.linalg.norm(pts1 - current_nocs_coords, axis=1)
            inlier_idx = list(np.where(pts_dis <= 0.1)[0])
            pred_size = 2 * np.max(abs(current_nocs_coords[inlier_idx]), axis=0)

            if inlier_ratio is not None:
                if inlier_ratio > best_inlier_ratios:
                    best_inlier_ratios = inlier_ratio
                    self.best_inlier_ratios = best_inlier_ratios
                    self.best_view = view_idx
                    self.best_sRT = pred_sRT
                    self.best_R = nocs_R
                    self.best_T = nocs_T
                    
                    self.best_size = pred_size * np.cbrt(np.linalg.det(self.best_sRT[:3, :3]))

                    if template.is_symmetric and template.symmetry_axis == 'z' and template.symmetry_type == 'continuous':
                        self.best_size[:2] = np.max(self.best_size[:2])

                    self.best_nocs = current_nocs_coords
                    self.best_pts = unique_points
                    self.best_choose = current_choose
                    self.best_match_matrix = current_match_matrix
    
    def __pixel_to_camera_point(self, depth_map, K, pixel):
        
        u, v = pixel
        z = depth_map[v, u]
        if z == 0:
            return None
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.array([x, y, z])
    
    def refine_grasp_position(self, grasp_pixel, grasp_pose, target_depth_map, K):
        refine_position = self.__pixel_to_camera_point(target_depth_map, K, grasp_pixel)
        if refine_position is not None:
            grasp_pose[:3, 3] = refine_position.flatten()
            
        return grasp_pose

    def grasp_estimation(self, template, pts, pts2d, intrinsic, grasp_name = None):
        template_nunocs_scale = np.max(template.shape, axis=0) - np.min(template.shape, axis=0)
        if template.grasp is not None:
            if grasp_name is not None:
                if ('grasp' in grasp_name or 'contact_point' in grasp_name) and grasp_name in template.grasp.keys():
                    grasp_position_in_camera = find_grasp_point_with_ram(np.array(template.grasp[grasp_name]['position']), \
                        template_nunocs_scale, template.foundation['nocs_vec'][self.best_view], self.best_match_matrix, pts)
                    grasp_point_obj_frame = transform_coordinates_3d(grasp_position_in_camera[:, np.newaxis], np.linalg.inv(self.best_sRT))

                    roll, pitch, yaw = np.array(template.grasp[grasp_name]['orientation']) * np.pi / 180.0
                    gripper_lines, grasp_pose = get_gripper_lines(grasp_point_obj_frame.flatten(), roll, pitch, yaw, self.best_sRT)
                    grasp_scale = np.cbrt(np.linalg.det(grasp_pose[:3, :3]))
                    grasp_pose[:3, :3] = grasp_pose[:3, :3] / grasp_scale

                    point_2d_homogeneous = intrinsic @ grasp_pose[:3, 3][:, np.newaxis]
                    u = int(point_2d_homogeneous[0, 0] / point_2d_homogeneous[2, 0])
                    v = int(point_2d_homogeneous[1, 0] / point_2d_homogeneous[2, 0])

                    return np.array([u, v]), 1.0, grasp_pose
                else:
                    raise Exception(f"invalid grasp name\n")
            else:
                pred_grasps = {}
                for k in template.grasp.keys():
                    if 'grasp' in k or 'contact_point' in k:
                        pred_grasps[k] = {}
                        grasp_position_in_camera = find_grasp_point_with_ram(np.array(template.grasp[k]['position']), \
                            template_nunocs_scale, template.foundation['nocs_vec'][self.best_view], self.best_match_matrix, pts)
                        grasp_point_obj_frame = transform_coordinates_3d(grasp_position_in_camera[:, np.newaxis], np.linalg.inv(self.best_sRT))                        

                        roll, pitch, yaw = np.array(template.grasp[k]['orientation']) * np.pi / 180.0
                        gripper_lines, grasp_pose = get_gripper_lines(grasp_point_obj_frame.flatten(), roll, pitch, yaw, self.best_sRT)
                        grasp_scale = np.cbrt(np.linalg.det(grasp_pose[:3, :3]))
                        grasp_pose[:3, :3] = grasp_pose[:3, :3] / grasp_scale

                        point_2d_homogeneous = intrinsic @ grasp_position_in_camera[:, np.newaxis]
                        u = int(point_2d_homogeneous[0, 0] / point_2d_homogeneous[2, 0])
                        v = int(point_2d_homogeneous[1, 0] / point_2d_homogeneous[2, 0])

                        pred_grasps[k]['grasp_pixel'] = np.array([u, v])
                        pred_grasps[k]['max_grasp_value'] = 1.0
                        pred_grasps[k]['grasp_pose'] = grasp_pose

                return pred_grasps, template.is_symmetric
        else:
            logging.info("Do not contain template grasp for %s", template.template_name)
    
    def function_plane_estimation(self, template, pts, plane_name = None):
        template_nunocs_scale = np.max(template.shape, axis=0) - np.min(template.shape, axis=0)
        if template.function is not None:
            if plane_name is not None:
                if 'plane' in plane_name and plane_name in template.function.keys():
                    support_point, support_normal = find_support_plane_with_ram(np.array(template.function[plane_name]['position']), \
                        np.array(template.function[plane_name]['orientation']), template.shape, template_nunocs_scale, \
                        template.foundation['nocs_vec'][self.best_view], self.best_match_matrix, pts, self.best_sRT, \
                        template.function[plane_name]['align_with_axis'])
                return support_point, support_normal
            else:
                pred_planes = {}
                for k in template.function.keys():
                    if 'plane' in k:
                        pred_planes[k] = {}
                        support_point, support_normal = find_support_plane_with_ram(np.array(template.function[k]['position']), \
                            np.array(template.function[k]['orientation']), template.shape, template_nunocs_scale, \
                            template.foundation['nocs_vec'][self.best_view], self.best_match_matrix, pts, self.best_sRT, \
                            template.function[k]['align_with_axis'])
                        pred_planes[k]['position'] = support_point
                        pred_planes[k]['orientation'] = support_normal
                return pred_planes
        else:
            logging.info("Do not contain template function plane for %s", template.template_name)
    
    def vis_grasp_point(self, img, grasp_pixel):
        return visualize_grasp_point_on_image(img, grasp_pixel)
    
    def vis_pose(self, img, pred_sRT, pred_size, K):
        return ram_draw_detections(img, K, pred_sRT, pred_size, need_draw_axis=True)
    
    def vis_function_plane(self, img, plane_point, plane_normal, pred_sRT, K):
        img = draw_plane_grid_on_image(img, plane_point, plane_normal, pred_sRT, K)
        img = draw_3d_arrow_on_image(img, plane_point, plane_normal, pred_sRT, K)
        return img
    
    def vis_rotation_axis(self, img, axis_point_obj, axis_dir_obj, pred_sRT, K, length=0.4):
        return draw_3d_arrow_on_image(
            img,
            np.asarray(axis_point_obj, dtype=np.float32),
            np.asarray(axis_dir_obj, dtype=np.float32),
            pred_sRT,
            K,
            length=length
        )

    def semantic_part_estimation(self):
        pass
