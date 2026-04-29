import cv2
import numpy as np


def load_depth(img_path):
    depth = cv2.imread(img_path, -1)
    if len(depth.shape) == 3:
        depth16 = depth[:, :, 1] * 256 + depth[:, :, 2]
        depth16 = np.where(depth16 == 32001, 0, depth16)
        depth16 = depth16.astype(np.uint16)
    elif len(depth.shape) == 2 and depth.dtype == 'uint16':
        depth16 = depth
    else:
        assert False, '[ Error ]: Unsupported depth type.'
    return depth16


def aug_get_bbox(bbox, img_w=640, img_h=480):
    y1, x1, y2, x2 = bbox
    img_height = img_h
    img_width = img_w
    window_size = (max(y2 - y1, x2 - x1) // 40 + 1) * 40
    window_size = min(window_size, min(img_h, img_w))
    center = [(y1 + y2) // 2, (x1 + x2) // 2]

    c_dy = np.random.randint(-window_size // 5, window_size // 5)
    c_dx = np.random.randint(-window_size // 5, window_size // 5)
    center[0] += c_dy
    center[1] += c_dx

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


def get_bbox(bbox, img_w=640, img_h=480):
    y1, x1, y2, x2 = bbox
    img_height = img_h
    img_width = img_w
    window_size = (max(y2 - y1, x2 - x1) // 40 + 1) * 40
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


def get_3d_bbox(size, cat_id=0, shift=0):
    bbox_3d = np.array([
        [+size[0] / 2, +size[1] / 2, +size[2] / 2],
        [+size[0] / 2, +size[1] / 2, -size[2] / 2],
        [-size[0] / 2, +size[1] / 2, +size[2] / 2],
        [-size[0] / 2, +size[1] / 2, -size[2] / 2],
        [+size[0] / 2, -size[1] / 2, +size[2] / 2],
        [+size[0] / 2, -size[1] / 2, -size[2] / 2],
        [-size[0] / 2, -size[1] / 2, +size[2] / 2],
        [-size[0] / 2, -size[1] / 2, -size[2] / 2],
    ]) + shift
    bbox_3d = bbox_3d.transpose()
    return bbox_3d


def transform_coordinates_3d(coordinates, sRT):
    assert coordinates.shape[0] == 3
    coordinates = np.vstack([coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)])
    new_coordinates = sRT @ coordinates
    new_coordinates = new_coordinates[:3, :] / new_coordinates[3, :]
    return new_coordinates


def calculate_2d_projections(coordinates_3d, intrinsics):
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)
    return projected_coordinates


def draw_axis(img, img_pts):
    img_pts = np.int32(img_pts).reshape(-1, 2)
    img = cv2.line(img, tuple(img_pts[0]), tuple(img_pts[1]), (255, 0, 0), 3)
    img = cv2.line(img, tuple(img_pts[0]), tuple(img_pts[2]), (0, 255, 0), 3)
    img = cv2.line(img, tuple(img_pts[0]), tuple(img_pts[3]), (0, 0, 255), 3)
    return img


def draw_bboxes(img, img_pts, color):
    img_pts = np.int32(img_pts).reshape(-1, 2)
    color_ground = (int(color[0] * 0.3), int(color[1] * 0.3), int(color[2] * 0.3))
    for i, j in zip([4, 5, 6, 7], [5, 7, 4, 6]):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color_ground, 2)
    color_pillar = (int(color[0] * 0.6), int(color[1] * 0.6), int(color[2] * 0.6))
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color_pillar, 2)
    for i, j in zip([0, 1, 2, 3], [1, 3, 0, 2]):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color, 2)
    return img


def ram_draw_detections(img, intrinsics, pred_sRT, pred_size, need_draw_axis=False):
    img = np.ascontiguousarray(img, dtype=np.uint8)

    bbox_3d = get_3d_bbox(pred_size)
    transformed_bbox_3d = transform_coordinates_3d(bbox_3d, pred_sRT)
    projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
    img = draw_bboxes(img, projected_bbox, (255, 0, 0))

    if need_draw_axis:
        axis_endopoints = np.array([
            [0.0, 0.2, 0.0, 0.0],
            [0.0, 0.0, 0.2, 0.0],
            [0.0, 0.0, 0.0, 0.2],
        ])
        transformed_endpoints = transform_coordinates_3d(axis_endopoints, pred_sRT)
        projected_endpoints = calculate_2d_projections(transformed_endpoints, intrinsics)
        img = draw_axis(img, projected_endpoints)

    return img
