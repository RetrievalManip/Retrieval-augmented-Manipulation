import numpy as np


def normalize_map(map):
    denom = map.max() - map.min()
    if denom == 0:
        return map
    return (map - map.min()) / denom

def get_target_voxels(target_map):
    if np.max(target_map) <= 0:
        return np.empty((0, 3), dtype=int)
    return np.argwhere(target_map >= np.max(target_map) * 0.99)

def calc_curvature(path):
    dx = np.gradient(path[:, 0])
    dy = np.gradient(path[:, 1])
    dz = np.gradient(path[:, 2])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    ddz = np.gradient(dz)
    curvature = np.sqrt((ddy * dx - ddx * dy)**2 + (ddz * dx - ddx * dz)**2 + (ddz * dy - ddy * dz)**2) / np.power(dx**2 + dy**2 + dz**2, 3/2)
    curvature[np.isnan(curvature)] = 0
    return curvature


class PathPlanner:
    def __init__(self, planner_config, map_size):
        self.config = planner_config
        self.map_size = map_size

    def optimize(self, start_pos: np.ndarray, target_map: np.ndarray, obstacle_map: np.ndarray, object_centric=False):
        info = dict()
        start_pos, raw_start_pos = start_pos.copy(), start_pos
        target_map, raw_target_map = target_map.copy(), target_map
        obstacle_map, raw_obstacle_map = obstacle_map.copy(), obstacle_map

        target_map = normalize_map(target_map)
        obstacle_map = normalize_map(obstacle_map)
        costmap = (
            -self.config['target_map_weight'] * target_map
            + self.config['obstacle_map_weight'] * obstacle_map
        )
        costmap = normalize_map(costmap)
        _costmap = costmap.copy()
        stop_criteria = self._get_stop_criteria()
        path, current_pos = [start_pos], start_pos
        for i in range(self.config['max_steps']):
            all_nearby_voxels = self._calculate_nearby_voxel(current_pos, object_centric=object_centric)
            nearby_score = _costmap[all_nearby_voxels[:, 0], all_nearby_voxels[:, 1], all_nearby_voxels[:, 2]]
            steepest_idx = np.argmin(nearby_score)
            next_pos = all_nearby_voxels[steepest_idx]
            _costmap[current_pos[0].round().astype(int),
                     current_pos[1].round().astype(int),
                     current_pos[2].round().astype(int)] += 1
            path.append(next_pos)
            current_pos = next_pos
            if stop_criteria(current_pos, _costmap, self.config['stop_threshold']):
                break
        raw_path = np.array(path)
        processed_path = self._postprocess_path(raw_path, raw_target_map, object_centric=object_centric)
        info['start_pos'] = start_pos
        info['target_map'] = target_map
        info['obstacle_map'] = obstacle_map
        info['costmap'] = costmap
        info['costmap_altered'] = _costmap
        info['raw_start_pos'] = raw_start_pos
        info['raw_target_map'] = raw_target_map
        info['raw_obstacle_map'] = raw_obstacle_map
        info['planner_raw_path'] = raw_path.copy()
        info['planner_postprocessed_path'] = processed_path.copy()
        info['targets_voxel'] = get_target_voxels(raw_target_map)
        return processed_path, info
    
    def _get_stop_criteria(self):
        def no_nearby_equal_criteria(current_pos, costmap, stop_threshold):
            assert np.isnan(costmap).sum() == 0, 'costmap contains nan'
            current_pos_discrete = current_pos.round().clip(0, self.map_size - 1).astype(int)
            current_cost = costmap[current_pos_discrete[0], current_pos_discrete[1], current_pos_discrete[2]]
            nearby_locs = self._calculate_nearby_voxel(current_pos, object_centric=False)
            nearby_equal = np.any(costmap[nearby_locs[:, 0], nearby_locs[:, 1], nearby_locs[:, 2]] < current_cost + stop_threshold)
            if nearby_equal:
                return False
            return True
        return no_nearby_equal_criteria

    def _calculate_nearby_voxel(self, current_pos, object_centric=False):
        half_size = int(2 * self.map_size / 100)
        offsets = np.arange(-half_size, half_size + 1)
        if object_centric:
            offsets_grid = np.array(np.meshgrid(offsets, offsets, [0])).T.reshape(-1, 3)
            offsets_grid = offsets_grid[np.any(offsets_grid != [0, 0, 0], axis=1)]
        else:
            offsets_grid = np.array(np.meshgrid(offsets, offsets, offsets)).T.reshape(-1, 3)
            offsets_grid = offsets_grid[np.any(offsets_grid != [0, 0, 0], axis=1)]
        all_nearby_voxels = np.clip(current_pos + offsets_grid, 0, self.map_size - 1)
        all_nearby_voxels = np.unique(all_nearby_voxels, axis=0)
        return all_nearby_voxels
    
    def _postprocess_path(self, path, raw_target_map, object_centric=False):
        curvature = calc_curvature(path)
        if len(curvature) > 5:
            high_curvature_idx = np.where(curvature[5:] > self.config['max_curvature'])[0]
            if len(high_curvature_idx) > 0:
                high_curvature_idx += 5
                path = path[:int(0.9 * high_curvature_idx[0])]  
        path_trimmed = path[1:-1]
        skip_ratio = None
        if len(path_trimmed) > 1:
            target_spacing = int(self.config['target_spacing'] * self.map_size / 100)
            length = np.linalg.norm(path_trimmed[1:] - path_trimmed[:-1], axis=1).sum()
            if length > target_spacing:
                curr_spacing = np.linalg.norm(path_trimmed[1:] - path_trimmed[:-1], axis=1).mean()
                skip_ratio = np.round(target_spacing / curr_spacing).astype(int)
                if skip_ratio > 1:
                    path_trimmed = path_trimmed[::skip_ratio]
        path = np.concatenate([path[0:1], path_trimmed, path[-1:]])
        last_waypoint = path[-1].round().clip(0, self.map_size - 1).astype(int)
        if raw_target_map[last_waypoint[0], last_waypoint[1], last_waypoint[2]] < np.max(raw_target_map) * 0.99:
            target_pos = get_target_voxels(raw_target_map)
            closest_target_idx = np.argmin(np.linalg.norm(target_pos - last_waypoint, axis=1))
            closest_target = target_pos[closest_target_idx]
            if object_centric:
                closest_target[2] = last_waypoint[2]
            path = np.append(path, [closest_target], axis=0)
        if object_centric:
            k = self.config['pushing_skip_per_k']
            path = np.concatenate([path[k:-1:k], path[-1:]])
        path = path.clip(0, self.map_size-1)
        return path
