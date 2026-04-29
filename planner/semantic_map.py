import numpy as np
import open3d as o3d

class SemanticMap():
    def __init__(self, map_size):
        self.map_size = map_size
        self.voxel_map = np.zeros((self.map_size, self.map_size, self.map_size))
    
    def assignVoxel(self, points, value):
        for point in points:
            x, y, z = point
            if 0 <= x < self.map_size and 0 <= y < self.map_size and 0 <= z < self.map_size:
                self.voxel_map[x, y, z] = value

    def visualize(self):
        grid = np.indices(self.map_size)

        points = grid.reshape(3, -1).T.astype(np.float64)
    
        
        flat_matrix = self.voxel_map.flatten()
        colors = np.zeros((flat_matrix.size, 3))
        colors[flat_matrix == 1] = [255, 0, 0]
    
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([pcd])
