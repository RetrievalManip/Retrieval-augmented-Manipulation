
import numpy as np


def randomly_limit_trues(mask: np.ndarray, max_trues: int) -> np.ndarray:
    true_indices = np.flatnonzero(mask)

    if true_indices.size <= max_trues:
        return mask

    sampled_indices = np.random.choice(true_indices, size=max_trues, replace=False)

    limited_flat_mask = np.zeros(mask.size, dtype=bool)
    limited_flat_mask[sampled_indices] = True

    return limited_flat_mask.reshape(mask.shape)


def create_pixel_coordinate_grid(num_frames, height, width):
    y_grid, x_grid = np.indices((height, width), dtype=np.float32)
    x_grid = x_grid[np.newaxis, :, :]
    y_grid = y_grid[np.newaxis, :, :]

    x_coords = np.broadcast_to(x_grid, (num_frames, height, width))
    y_coords = np.broadcast_to(y_grid, (num_frames, height, width))

    f_idx = np.arange(num_frames, dtype=np.float32)[:, np.newaxis, np.newaxis]
    f_coords = np.broadcast_to(f_idx, (num_frames, height, width))

    points_xyf = np.stack((x_coords, y_coords, f_coords), axis=-1)

    return points_xyf
