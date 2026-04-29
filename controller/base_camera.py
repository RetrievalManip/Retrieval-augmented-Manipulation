import json
import logging

import numpy as np


class BaseCamera:
    def __init__(self):
        self.R = np.eye(3)
        self.T = np.zeros((3, 1))
        self.K = np.eye(3)
        self.D = np.zeros((1, 5))
        self.w = 1280
        self.h = 720
        self.undistort = False
        self.max_depth = 2000
        self.type = "external"
        self.fps = 30
        self.camparas_file = ""

        self.rgb_image = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.rgb_right_image = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.depth_map = np.zeros((self.h, self.w), dtype=np.float32)

    def restore_config(self, config_file):
        config_paras = {
            "cam_paras_file": self.camparas_file,
            "fps": self.fps,
            "max_depth": self.max_depth,
            "type": self.type,
        }
        with open(config_file, 'w') as f:
            json.dump(config_paras, f, indent=4)
        logging.info("Configuration saved to %s", config_file)

    def read_config(self, config_file):
        with open(config_file, 'r') as f:
            config_paras = json.load(f)
        self.camparas_file = config_paras["cam_paras_file"]
        self.max_depth = config_paras["max_depth"]
        self.fps = config_paras["fps"]
        self.type = config_paras["type"]

    def capture_image(self):
        RGB_image = self.rgb_image
        depth_map = self.depth_map
        return RGB_image, depth_map
    
    @property
    def w2c(self):
        c2w = np.eye(4)
        c2w[:3, :3] = self.R
        c2w[:3, 3] = self.T.reshape(3, )
        c2w[3, 3] = 1.0
        w2c = np.linalg.inv(c2w)
        return w2c

    @property
    def c2w(self):
        c2w = np.eye(4)
        c2w[:3, :3] = self.R
        c2w[:3, 3] = self.T.reshape(3, )
        c2w[3, 3] = 1.0
        return c2w
    
    @property
    def fx(self):
        return self.K[0, 0]
    
    @property
    def fy(self):
        return self.K[1, 1]
    
    @property
    def cx(self):
        return self.K[0, 2]
    
    @property
    def cy(self):
        return self.K[1, 2]

    @property
    def width(self):
        return self.w
    
    @property
    def height(self):
        return self.h
    
    @property
    def left_image(self):
        return self.rgb_image
    
    @left_image.setter
    def left_image(self, image):
        self.rgb_image = image
    
    @property
    def right_image(self):
        return self.rgb_right_image
    
    @right_image.setter
    def right_image(self, image):
        self.rgb_right_image = image

    @property
    def cam_paras_file(self):
        return self.camparas_file

    @cam_paras_file.setter
    def cam_paras_file(self, file_path):
        self.camparas_file = file_path
    
        
