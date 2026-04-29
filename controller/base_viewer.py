
from typing import List
from abc import ABC, abstractmethod

from controller.base_camera import BaseCamera


class BaseViewer():
    def __init__(self):
        self.ip = "localhost:8099"
        pass
        
    @abstractmethod
    def add_pcd(self, xyzs, rgbs):
        num = xyzs.shape[0]
        pass

    @abstractmethod
    def add_camera(self, camera: BaseCamera):
        fx = camera.fx
        fy = camera.fy
        cx = camera.cx
        cy = camera.cy
        c2w = camera.c2w
        R = c2w[:3, :3]
        T = c2w[:3, 3]

        rbg = camera.rgb_image
        depth= camera.depth_map

        pass
    
    @abstractmethod
    def add_bbox(self, bbox: List):
        xmin, ymin, zmin, xmax, ymax, zmax = bbox

    @abstractmethod
    def add_axis(self, pos):
        pass
    
    @abstractmethod
    def update_frame(self):
        pass
