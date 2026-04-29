
import torch
import numpy as np
from .vggsfm_utils import *


def predict_tracks(
    images,
    conf=None,
    points_3d=None,
    masks=None,
    max_query_pts=2048,
    query_frame_num=5,
    keypoint_extractor="aliked+sp",
    max_points_num=163840,
    fine_tracking=True,
    complete_non_vis=True,
):

    device = images.device
    dtype = images.dtype
    tracker = build_vggsfm_tracker().to(device, dtype)

    query_frame_indexes = generate_rank_by_dino(images, query_frame_num=query_frame_num, device=device)

    if 0 in query_frame_indexes:
        query_frame_indexes.remove(0)
    query_frame_indexes = [0, *query_frame_indexes]

    keypoint_extractors = initialize_feature_extractors(
        max_query_pts, extractor_method=keypoint_extractor, device=device
    )

    pred_tracks = []
    pred_vis_scores = []
    pred_confs = []
    pred_points_3d = []
    pred_colors = []

    fmaps_for_tracker = tracker.process_images_to_fmaps(images)

    if fine_tracking:
        print("For faster inference, consider disabling fine_tracking")

    for query_index in query_frame_indexes:
        print(f"Predicting tracks for query frame {query_index}")
        pred_track, pred_vis, pred_conf, pred_point_3d, pred_color = _forward_on_query(
            query_index,
            images,
            conf,
            points_3d,
            fmaps_for_tracker,
            keypoint_extractors,
            tracker,
            max_points_num,
            fine_tracking,
            device,
        )

        pred_tracks.append(pred_track)
        pred_vis_scores.append(pred_vis)
        pred_confs.append(pred_conf)
        pred_points_3d.append(pred_point_3d)
        pred_colors.append(pred_color)

    if complete_non_vis:
        pred_tracks, pred_vis_scores, pred_confs, pred_points_3d, pred_colors = _augment_non_visible_frames(
            pred_tracks,
            pred_vis_scores,
            pred_confs,
            pred_points_3d,
            pred_colors,
            images,
            conf,
            points_3d,
            fmaps_for_tracker,
            keypoint_extractors,
            tracker,
            max_points_num,
            fine_tracking,
            min_vis=500,
            non_vis_thresh=0.1,
            device=device,
        )

    pred_tracks = np.concatenate(pred_tracks, axis=1)
    pred_vis_scores = np.concatenate(pred_vis_scores, axis=1)
    pred_confs = np.concatenate(pred_confs, axis=0) if pred_confs else None
    pred_points_3d = np.concatenate(pred_points_3d, axis=0) if pred_points_3d else None
    pred_colors = np.concatenate(pred_colors, axis=0) if pred_colors else None


    return pred_tracks, pred_vis_scores, pred_confs, pred_points_3d, pred_colors


def _forward_on_query(
    query_index,
    images,
    conf,
    points_3d,
    fmaps_for_tracker,
    keypoint_extractors,
    tracker,
    max_points_num,
    fine_tracking,
    device,
):
    frame_num, _, height, width = images.shape

    query_image = images[query_index]
    query_points = extract_keypoints(query_image, keypoint_extractors, round_keypoints=False)
    query_points = query_points[:, torch.randperm(query_points.shape[1], device=device)]

    query_points_long = query_points.squeeze(0).round().long()
    pred_color = images[query_index][:, query_points_long[:, 1], query_points_long[:, 0]]
    pred_color = (pred_color.permute(1, 0).cpu().numpy() * 255).astype(np.uint8)

    if (conf is not None) and (points_3d is not None):
        assert height == width
        assert conf.shape[-2] == conf.shape[-1]
        assert conf.shape[:3] == points_3d.shape[:3]
        scale = conf.shape[-1] / width

        query_points_scaled = (query_points.squeeze(0) * scale).round().long()
        query_points_scaled = query_points_scaled.cpu().numpy()

        pred_conf = conf[query_index][query_points_scaled[:, 1], query_points_scaled[:, 0]]
        pred_point_3d = points_3d[query_index][query_points_scaled[:, 1], query_points_scaled[:, 0]]

        valid_mask = pred_conf > 1.2
        if valid_mask.sum() > 512:
            query_points = query_points[:, valid_mask]
            pred_conf = pred_conf[valid_mask]
            pred_point_3d = pred_point_3d[valid_mask]
            pred_color = pred_color[valid_mask]
    else:
        pred_conf = None
        pred_point_3d = None

    reorder_index = calculate_index_mappings(query_index, frame_num, device=device)

    images_feed, fmaps_feed = switch_tensor_order([images, fmaps_for_tracker], reorder_index, dim=0)
    images_feed = images_feed[None]
    fmaps_feed = fmaps_feed[None]

    all_points_num = images_feed.shape[1] * query_points.shape[1]

    if all_points_num > max_points_num:
        num_splits = (all_points_num + max_points_num - 1) // max_points_num
        query_points = torch.chunk(query_points, num_splits, dim=1)
    else:
        query_points = [query_points]

    pred_track, pred_vis, _ = predict_tracks_in_chunks(
        tracker, images_feed, query_points, fmaps_feed, fine_tracking=fine_tracking
    )

    pred_track, pred_vis = switch_tensor_order([pred_track, pred_vis], reorder_index, dim=1)

    pred_track = pred_track.squeeze(0).float().cpu().numpy()
    pred_vis = pred_vis.squeeze(0).float().cpu().numpy()

    return pred_track, pred_vis, pred_conf, pred_point_3d, pred_color


def _augment_non_visible_frames(
    pred_tracks: list,
    pred_vis_scores: list,
    pred_confs: list,
    pred_points_3d: list,
    pred_colors: list,
    images: torch.Tensor,
    conf,
    points_3d,
    fmaps_for_tracker,
    keypoint_extractors,
    tracker,
    max_points_num: int,
    fine_tracking: bool,
    *,
    min_vis: int = 500,
    non_vis_thresh: float = 0.1,
    device: torch.device = None,
):
    last_query = -1
    final_trial = False
    cur_extractors = keypoint_extractors

    while True:
        vis_array = np.concatenate(pred_vis_scores, axis=1)

        sufficient_vis_count = (vis_array > non_vis_thresh).sum(axis=-1)
        non_vis_frames = np.where(sufficient_vis_count < min_vis)[0].tolist()

        if len(non_vis_frames) == 0:
            break

        print("Processing non visible frames:", non_vis_frames)

        if non_vis_frames[0] == last_query:
            final_trial = True
            cur_extractors = initialize_feature_extractors(2048, extractor_method="sp+sift+aliked", device=device)
            query_frame_list = non_vis_frames
        else:
            query_frame_list = [non_vis_frames[0]]

        last_query = non_vis_frames[0]

        for query_index in query_frame_list:
            new_track, new_vis, new_conf, new_point_3d, new_color = _forward_on_query(
                query_index,
                images,
                conf,
                points_3d,
                fmaps_for_tracker,
                cur_extractors,
                tracker,
                max_points_num,
                fine_tracking,
                device,
            )
            pred_tracks.append(new_track)
            pred_vis_scores.append(new_vis)
            pred_confs.append(new_conf)
            pred_points_3d.append(new_point_3d)
            pred_colors.append(new_color)

        if final_trial:
            break

    return pred_tracks, pred_vis_scores, pred_confs, pred_points_3d, pred_colors
