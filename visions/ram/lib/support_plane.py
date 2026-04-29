import numpy as np
import cv2
from ram_utils import transform_coordinates_3d
from align import estimateSimilarityTransform

def points_on_plane(points, template_scale, p, n, epsilon=0.01):
    n = n / np.linalg.norm(n)
    vecs = points - p
    distances = np.dot(vecs, n)
    mask = (np.abs(distances) / np.linalg.norm(template_scale)) < epsilon
    on_plane_points = points[mask]

    return on_plane_points, mask, distances

def plane_point_to_volume(plane_points, template_scale, resolution=0.1):
    origin = np.array([-0.5, -0.5, -0.5])
    vol_length = 1.0
    dims = int(np.ceil(vol_length / resolution))
    volume = np.zeros((dims, dims, dims), dtype=np.uint8)
    if len(plane_points) == 0:
        return volume
    norm_pts = plane_points / template_scale
    norm_pts = np.clip(norm_pts, -0.5, 0.5)
    idxs = ((norm_pts - origin) / resolution).astype(int)
    idxs = np.clip(idxs, 0, dims-1)
    for ix, iy, iz in idxs:
        volume[ix, iy, iz] = 1
    return volume

def query_plane_volume_from_points(points, plane_volume, template_scale, resolution=0.1):
    origin = np.array([-0.5, -0.5, -0.5])
    dims = plane_volume.shape[0]

    nunocs_pts = points * np.linalg.norm(template_scale) / template_scale

    idxs = ((nunocs_pts - origin) / resolution).astype(int)
    idxs = np.clip(idxs, 0, dims-1)
    plane_values = plane_volume[idxs[:, 0], idxs[:, 1], idxs[:, 2]][:, None]
    return plane_values

def propagate_plane_value(plane_value, matching_matrix, query_points, threshold=0.8):
    mapped_plane_value = matching_matrix @ plane_value
    idx = np.where(mapped_plane_value[:, 0] > threshold)[0]
    selected_points = query_points[idx]
    return mapped_plane_value, idx, selected_points

def fit_plane_and_project_origin(points):
    centroid = points.mean(axis=0)
    uu, dd, vv = np.linalg.svd(points - centroid)
    normal = vv[2, :]
    normal = normal / np.linalg.norm(normal)
    
    d = normal.dot(-centroid)
    proj_point = -d * normal
    
    return proj_point, normal

def find_support_plane_coord(template_support_point, template_scale, ref_points, matching_matrix, query_points):
    support_point_org = template_support_point / np.linalg.norm(template_scale)
    ref_pts_in_support_frame = ref_points - support_point_org[np.newaxis, :]
    query_pts_in_support_frame = matching_matrix @ ref_pts_in_support_frame
    _, support_R, support_T, support_sRT, _, _ = estimateSimilarityTransform(query_pts_in_support_frame, query_points)

    return support_T

def find_support_plane_with_ram(template_support_point, template_support_normal, \
    template_model_pts, template_scale, ref_points, matching_matrix, query_points, sRT, align_with_axis = None):
    
    plane_points, _, _ = points_on_plane(template_model_pts, template_scale, template_support_point, template_support_normal)
    support_volume = plane_point_to_volume(plane_points, template_scale)
    
    ref_support_values = query_plane_volume_from_points(ref_points, support_volume, template_scale)

    _, _, selected_support_points = propagate_plane_value(ref_support_values, matching_matrix, query_points)

    if selected_support_points.shape[0] < 3:
        print('Not enough support points found, returning default plane.')
        return template_support_point / (np.linalg.norm(template_scale)), template_support_normal

    selected_support_points_in_object_frame = transform_coordinates_3d(selected_support_points.T, np.linalg.inv(sRT)).T

    _, support_normal = fit_plane_and_project_origin(selected_support_points_in_object_frame)

    if np.dot(support_normal, template_support_normal) < 0:
        support_normal = -support_normal

    if align_with_axis is not None:
        if align_with_axis == 'x':
            support_normal = np.array([1, 0, 0])
        elif align_with_axis == '-x':
            support_normal = np.array([-1, 0, 0])
        elif align_with_axis == 'y':
            support_normal = np.array([0, 1, 0])
        elif align_with_axis == '-y':
            support_normal = np.array([0, -1, 0])
        elif align_with_axis == 'z':
            support_normal = np.array([0, 0, 1])
        elif align_with_axis == '-z':
            support_normal = np.array([0, 0, -1])

    support_point = find_support_plane_coord(template_support_point, template_scale, ref_points, matching_matrix, query_points)
    support_point = transform_coordinates_3d(support_point[:, np.newaxis], np.linalg.inv(sRT)).flatten()
            
    return support_point, support_normal

def get_plane_square_points(plane_point, plane_normal, plane_size=1.2):
    n = plane_normal / np.linalg.norm(plane_normal)
    v = np.array([1, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1, 0])
    basis1 = np.cross(n, v)
    basis1 /= np.linalg.norm(basis1)
    basis2 = np.cross(n, basis1)
    basis2 /= np.linalg.norm(basis2)
    half = plane_size / 2

    corners = [
        plane_point - half * basis1 + half * basis2,
        plane_point + half * basis1 + half * basis2,
        plane_point + half * basis1 - half * basis2,
        plane_point - half * basis1 - half * basis2,
    ]
    return np.stack(corners, axis=0)

def clip_points_to_cube(points, cube_min=-0.5, cube_max=0.5):
    return np.clip(points, cube_min, cube_max)

def apply_pose(points, pose):
    N = points.shape[0]
    homo = np.concatenate([points, np.ones((N,1))], axis=1)
    cam = (pose @ homo.T).T
    return cam[:, :3]

def project_points(points_cam, K):
    x = points_cam[:,0]
    y = points_cam[:,1]
    z = points_cam[:,2]
    u = K[0,0]*x/z + K[0,2]
    v = K[1,1]*y/z + K[1,2]
    return np.stack([u, v], axis=1)

def draw_plane_on_image_obj_coords(
    rgb_img, plane_point, plane_normal, pose, K,
    color=(255,128,0), alpha=0.45, plane_size=1.0, cube_min=-0.5, cube_max=0.5
):
    poly_obj = get_plane_square_points(plane_point, plane_normal, plane_size)
    poly_obj = clip_points_to_cube(poly_obj, cube_min, cube_max)
    poly_cam = apply_pose(poly_obj, pose)
    poly_pix = project_points(poly_cam, K)
    poly_pix = np.round(poly_pix).astype(np.int32)
    vis_img = rgb_img.copy()
    overlay = vis_img.copy()
    cv2.fillPoly(overlay, [poly_pix], color)
    vis_img = cv2.addWeighted(overlay, alpha, vis_img, 1-alpha, 0)
    return vis_img

def draw_plane_grid_on_image(
    img,
    plane_center_obj,
    plane_normal_obj,
    obj_pose_cam,
    K,
    grid_size=4,
    grid_extent=0.5
):
    H, W = img.shape[:2]
    overlay = img.copy()

    n_obj = plane_normal_obj / np.linalg.norm(plane_normal_obj)
    print(f"Plane normal in object coordinates: {n_obj}")
    v = np.array([1,0,0]) if abs(n_obj[0])<0.9 else np.array([0,1,0])
    u_obj = np.cross(n_obj, v)
    u_obj = u_obj / np.linalg.norm(u_obj)
    v_obj = np.cross(n_obj, u_obj)

    ticks = np.linspace(-grid_extent, grid_extent, grid_size+1)
    grid_points_obj = []
    for i in range(grid_size+1):
        for j in range(grid_size+1):
            pt = plane_center_obj + u_obj*ticks[i] + v_obj*ticks[j]
            grid_points_obj.append(pt)
    grid_points_obj = np.array(grid_points_obj).reshape((grid_size+1, grid_size+1, 3))

    def to_cam(pt_obj):
        pt_obj_h = np.concatenate([pt_obj, [1]])
        pt_cam_h = obj_pose_cam @ pt_obj_h
        return pt_cam_h[:3]
    
    grid_points_cam = np.zeros_like(grid_points_obj)
    for i in range(grid_size+1):
        for j in range(grid_size+1):
            grid_points_cam[i,j] = to_cam(grid_points_obj[i,j])

    def proj(pt3d):
        pt3d = np.asarray(pt3d, dtype=np.float64)
        if not np.isfinite(pt3d[2]) or pt3d[2] <= 1e-8:
            return None
        pt = pt3d / pt3d[2]
        if not np.all(np.isfinite(pt)):
            return None
        pix = K @ pt
        if not np.all(np.isfinite(pix)):
            return None
        x, y = int(round(pix[0])), int(round(pix[1]))
        return (x, y)

    grid_points_2d = np.full((grid_size+1, grid_size+1, 2), -1, dtype=int)
    valid_mask = np.zeros((grid_size+1, grid_size+1), dtype=bool)
    for i in range(grid_size+1):
        for j in range(grid_size+1):
            pt2d = proj(grid_points_cam[i,j])
            if pt2d is not None:
                grid_points_2d[i,j] = pt2d
                valid_mask[i,j] = True

    grid_color = (180, 210, 255)
    grid_thickness = 2
    grid_alpha = 0.7
    vertex_color = (60, 120, 255)
    vertex_radius = 7
    center_color = (0, 80, 255)
    center_radius = 10

    for i in range(grid_size+1):
        pts = [tuple(grid_points_2d[i,j]) for j in range(grid_size+1) if valid_mask[i,j]]
        if len(pts) >= 2:
            cv2.polylines(overlay, [np.array(pts)], isClosed=False, color=grid_color, thickness=grid_thickness, lineType=cv2.LINE_AA)
    for j in range(grid_size+1):
        pts = [tuple(grid_points_2d[i,j]) for i in range(grid_size+1) if valid_mask[i,j]]
        if len(pts) >= 2:
            cv2.polylines(overlay, [np.array(pts)], isClosed=False, color=grid_color, thickness=grid_thickness, lineType=cv2.LINE_AA)

    for i in range(grid_size+1):
        for j in range(grid_size+1):
            if valid_mask[i,j]:
                cv2.circle(overlay, tuple(grid_points_2d[i,j]), vertex_radius, vertex_color, -1, lineType=cv2.LINE_AA)

    center_idx = grid_size // 2
    if valid_mask[center_idx,center_idx]:
        cv2.circle(overlay, tuple(grid_points_2d[center_idx,center_idx]), center_radius, center_color, -1, lineType=cv2.LINE_AA)

    out = cv2.addWeighted(overlay, grid_alpha, img, 1-grid_alpha, 0)
    return out

def draw_3d_arrow_on_image(
    img, 
    point,
    normal,
    pose,
    K,
    length=0.3,
    color_main=(255,180,60),
    color_tip=(255,64,0),
    thickness=5,
    tip_length=0.2,
    shadow=True
):
    normal = normal / np.linalg.norm(normal)

    pt3d_start = apply_pose(point[np.newaxis, :], pose).flatten()
    pt3d_end = apply_pose((point + length * normal)[np.newaxis, :], pose).flatten()

    def proj(pt):
        pt = np.asarray(pt, dtype=np.float64)
        if not np.isfinite(pt[2]) or pt[2] <= 1e-8:
            return None
        pt = pt / pt[2]
        if not np.all(np.isfinite(pt)):
            return None
        pix = K @ pt
        if not np.all(np.isfinite(pix)):
            return None
        return (int(round(pix[0])), int(round(pix[1])))

    pt2d_start = proj(pt3d_start)
    pt2d_end = proj(pt3d_end)
    if pt2d_start is None or pt2d_end is None:
        return img

    overlay = img.copy()

    if shadow:
        shadow_offset = (5, 5)
        cv2.arrowedLine(
            overlay, 
            (pt2d_start[0]+shadow_offset[0], pt2d_start[1]+shadow_offset[1]),
            (pt2d_end[0]+shadow_offset[0], pt2d_end[1]+shadow_offset[1]),
            (40, 40, 40), thickness=thickness+4, tipLength=tip_length, line_type=cv2.LINE_AA
        )
    
    num_steps = 24
    for i in range(num_steps):
        alpha = i / (num_steps - 1)
        color = tuple([
            int((1-alpha)*color_main[c] + alpha*color_tip[c])
            for c in range(3)
        ])
        p0 = (
            int((1-alpha)*pt2d_start[0] + alpha*pt2d_end[0]),
            int((1-alpha)*pt2d_start[1] + alpha*pt2d_end[1])
        )
        p1 = (
            int((1-(alpha+1/num_steps))*pt2d_start[0] + (alpha+1/num_steps)*pt2d_end[0]),
            int((1-(alpha+1/num_steps))*pt2d_start[1] + (alpha+1/num_steps)*pt2d_end[1])
        )
        cv2.line(overlay, p0, p1, color, thickness, lineType=cv2.LINE_AA)
    cv2.arrowedLine(overlay, pt2d_start, pt2d_end, color_tip, thickness, tipLength=tip_length, line_type=cv2.LINE_AA)
    
    cv2.circle(overlay, pt2d_start, radius=thickness*2+2, color=(255,255,255), thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(overlay, pt2d_start, radius=thickness*2, color=color_main, thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(overlay, pt2d_start, radius=thickness, color=color_tip, thickness=-1, lineType=cv2.LINE_AA)

    alpha = 0.84
    img_out = cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)
    return img_out
