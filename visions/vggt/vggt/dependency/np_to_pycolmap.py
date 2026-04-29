
import numpy as np
import pycolmap
from .projection import project_3D_points_np


def batch_np_matrix_to_pycolmap(
    points3d,
    extrinsics,
    intrinsics,
    tracks,
    image_size,
    masks=None,
    max_reproj_error=None,
    max_points3D_val=3000,
    shared_camera=False,
    camera_type="SIMPLE_PINHOLE",
    extra_params=None,
    min_inlier_per_frame=64,
    points_rgb=None,
):

    N, P, _ = tracks.shape
    assert len(extrinsics) == N
    assert len(intrinsics) == N
    assert len(points3d) == P
    assert image_size.shape[0] == 2

    if max_reproj_error is not None:
        projected_points_2d, projected_points_cam = project_3D_points_np(points3d, extrinsics, intrinsics)
        projected_diff = np.linalg.norm(projected_points_2d - tracks, axis=-1)
        projected_points_2d[projected_points_cam[:, -1] <= 0] = 1e6
        reproj_mask = projected_diff < max_reproj_error

    if masks is not None and reproj_mask is not None:
        masks = np.logical_and(masks, reproj_mask)
    elif masks is not None:
        masks = masks
    else:
        masks = reproj_mask

    assert masks is not None

    if masks.sum(1).min() < min_inlier_per_frame:
        print(f"Not enough inliers per frame, skip BA.")
        return None, None

    reconstruction = pycolmap.Reconstruction()

    inlier_num = masks.sum(0)
    valid_mask = inlier_num >= 2
    valid_idx = np.nonzero(valid_mask)[0]

    for vidx in valid_idx:
        rgb = points_rgb[vidx] if points_rgb is not None else np.zeros(3)
        reconstruction.add_point3D(points3d[vidx], pycolmap.Track(), rgb)

    num_points3D = len(valid_idx)
    camera = None
    for fidx in range(N):
        if camera is None or (not shared_camera):
            pycolmap_intri = _build_pycolmap_intri(fidx, intrinsics, camera_type, extra_params)

            camera = pycolmap.Camera(
                model=camera_type, width=image_size[0], height=image_size[1], params=pycolmap_intri, camera_id=fidx + 1
            )

            reconstruction.add_camera(camera)

        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(extrinsics[fidx][:3, :3]), extrinsics[fidx][:3, 3]
        )

        image = pycolmap.Image(
            id=fidx + 1, name=f"image_{fidx + 1}", camera_id=camera.camera_id, cam_from_world=cam_from_world
        )

        points2D_list = []

        point2D_idx = 0

        for point3D_id in range(1, num_points3D + 1):
            original_track_idx = valid_idx[point3D_id - 1]

            if (reconstruction.points3D[point3D_id].xyz < max_points3D_val).all():
                if masks[fidx][original_track_idx]:
                    point2D_xy = tracks[fidx][original_track_idx]
                    points2D_list.append(pycolmap.Point2D(point2D_xy, point3D_id))

                    track = reconstruction.points3D[point3D_id].track
                    track.add_element(fidx + 1, point2D_idx)
                    point2D_idx += 1

        assert point2D_idx == len(points2D_list)

        try:
            image.points2D = pycolmap.ListPoint2D(points2D_list)
            image.registered = True
        except:
            print(f"frame {fidx + 1} is out of BA")
            image.registered = False

        reconstruction.add_image(image)

    return reconstruction, valid_mask


def pycolmap_to_batch_np_matrix(reconstruction, device="cpu", camera_type="SIMPLE_PINHOLE"):

    num_images = len(reconstruction.images)
    max_points3D_id = max(reconstruction.point3D_ids())
    points3D = np.zeros((max_points3D_id, 3))

    for point3D_id in reconstruction.points3D:
        points3D[point3D_id - 1] = reconstruction.points3D[point3D_id].xyz

    extrinsics = []
    intrinsics = []

    extra_params = [] if camera_type == "SIMPLE_RADIAL" else None

    for i in range(num_images):
        pyimg = reconstruction.images[i + 1]
        pycam = reconstruction.cameras[pyimg.camera_id]
        matrix = pyimg.cam_from_world.matrix()
        extrinsics.append(matrix)

        calibration_matrix = pycam.calibration_matrix()
        intrinsics.append(calibration_matrix)

        if camera_type == "SIMPLE_RADIAL":
            extra_params.append(pycam.params[-1])

    extrinsics = np.stack(extrinsics)
    intrinsics = np.stack(intrinsics)

    if camera_type == "SIMPLE_RADIAL":
        extra_params = np.stack(extra_params)
        extra_params = extra_params[:, None]

    return points3D, extrinsics, intrinsics, extra_params




def batch_np_matrix_to_pycolmap_wo_track(
    points3d,
    points_xyf,
    points_rgb,
    extrinsics,
    intrinsics,
    image_size,
    shared_camera=False,
    camera_type="SIMPLE_PINHOLE",
):

    N = len(extrinsics)
    P = len(points3d)

    reconstruction = pycolmap.Reconstruction()

    for vidx in range(P):
        reconstruction.add_point3D(points3d[vidx], pycolmap.Track(), points_rgb[vidx])

    camera = None
    for fidx in range(N):
        if camera is None or (not shared_camera):
            pycolmap_intri = _build_pycolmap_intri(fidx, intrinsics, camera_type)

            camera = pycolmap.Camera(
                model=camera_type, width=image_size[0], height=image_size[1], params=pycolmap_intri, camera_id=fidx + 1
            )

            reconstruction.add_camera(camera)

        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(extrinsics[fidx][:3, :3]), extrinsics[fidx][:3, 3]
        )

        image = pycolmap.Image(
            id=fidx + 1, name=f"image_{fidx + 1}", camera_id=camera.camera_id, cam_from_world=cam_from_world
        )

        points2D_list = []

        point2D_idx = 0

        points_belong_to_fidx = points_xyf[:, 2].astype(np.int32) == fidx
        points_belong_to_fidx = np.nonzero(points_belong_to_fidx)[0]

        for point3D_batch_idx in points_belong_to_fidx:
            point3D_id = point3D_batch_idx + 1
            point2D_xyf = points_xyf[point3D_batch_idx]
            point2D_xy = point2D_xyf[:2]
            points2D_list.append(pycolmap.Point2D(point2D_xy, point3D_id))

            track = reconstruction.points3D[point3D_id].track
            track.add_element(fidx + 1, point2D_idx)
            point2D_idx += 1

        assert point2D_idx == len(points2D_list)

        try:
            image.points2D = pycolmap.ListPoint2D(points2D_list)
            image.registered = True
        except:
            print(f"frame {fidx + 1} does not have any points")
            image.registered = False

        reconstruction.add_image(image)

    return reconstruction


def _build_pycolmap_intri(fidx, intrinsics, camera_type, extra_params=None):
    if camera_type == "PINHOLE":
        pycolmap_intri = np.array(
            [intrinsics[fidx][0, 0], intrinsics[fidx][1, 1], intrinsics[fidx][0, 2], intrinsics[fidx][1, 2]]
        )
    elif camera_type == "SIMPLE_PINHOLE":
        focal = (intrinsics[fidx][0, 0] + intrinsics[fidx][1, 1]) / 2
        pycolmap_intri = np.array([focal, intrinsics[fidx][0, 2], intrinsics[fidx][1, 2]])
    elif camera_type == "SIMPLE_RADIAL":
        raise NotImplementedError("SIMPLE_RADIAL is not supported yet")
        focal = (intrinsics[fidx][0, 0] + intrinsics[fidx][1, 1]) / 2
        pycolmap_intri = np.array([focal, intrinsics[fidx][0, 2], intrinsics[fidx][1, 2], extra_params[fidx][0]])
    else:
        raise ValueError(f"Camera type {camera_type} is not supported yet")

    return pycolmap_intri
