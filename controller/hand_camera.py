import time
import math
import functools
import numpy as np
import cv2
from PIL import Image
import os
import json

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

import logging
logging.basicConfig(level=logging.INFO)

from .base_camera import BaseCamera

from utils.config import get_camera_config, resolve_project_path


class RealSense435Camera(BaseCamera):
    def __init__(self, config_file=None, target_serial=None):
        super(RealSense435Camera, self).__init__()
        if rs is None:
            raise ImportError("pyrealsense2 is required to use RealSense435Camera.")
        cur_path = os.path.dirname(os.path.abspath(__file__))

        cam_config = get_camera_config('realsense_d435')

        self.camparas_file = cam_config.paras_file if cam_config else os.path.join(cur_path, 'assets/calibration/hand_camera', 'realsense_paras.json')
        self.preset_file = os.path.join(cur_path, 'assets/calibration/hand_camera', 'realsense_HighResHighAccuracyPreset.json')
        self.max_depth = cam_config.max_depth if cam_config else 2000
        self.fps = cam_config.fps if cam_config else 30
        self.type = "hand"

        logging.info("==> Hand camera configuration loaded from unified config.yaml")

        with open(self.camparas_file, 'r') as f:
            cam_paras_dict = json.load(f)
        left_cam_paras = cam_paras_dict["left"]

        self.R = np.array(left_cam_paras["R"])
        self.T = np.array(left_cam_paras["T"])
        self.K = np.array(left_cam_paras["K"])
        self.D = np.array(left_cam_paras["D"])
        self.w = left_cam_paras["w"]
        self.h = left_cam_paras["h"]

        self.pipeline = rs.pipeline()
        config = rs.config()
        
        if target_serial:
            config.enable_device(target_serial)
            logging.info(f"Connecting to specific device with serial: {target_serial}")
        else:
            d435_serial = self._find_d435_serial()
            if d435_serial:
                config.enable_device(d435_serial)
                logging.info(f"Found D435 device with serial: {d435_serial}")
            else:
                raise RuntimeError("No D435 device found")
        
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        advanced_mode = rs.rs400_advanced_mode(device)
        with open(self.preset_file, 'r') as file:
            meta = file.read().strip()
            advanced_mode.load_json(meta)
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        config.enable_stream(rs.stream.color, self.w, self.h, rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, self.w, self.h, rs.format.z16, self.fps)

        profile = self.pipeline.start(config)

        depth_sensor = self.pipeline.get_active_profile().get_device().first_depth_sensor()
        depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
        depth_sensor.set_option(rs.option.exposure, 10000)
        depth_sensor.set_option(rs.option.laser_power, 100.0)
        depth_sensor.set_option(rs.option.gain, 16.0)

        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        logging.info("Depth Scale is: %f" % self.depth_scale)

        self.rgb_image = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.depth_map = np.zeros((self.h, self.w), dtype=np.float32)
        for i in range(30):
            self.rgb_image, self.depth_map = self.capture_image()

    def restore_camparas(self, config_file):
        left_cam_paras = {
            "R": self.R.tolist(),
            "T": self.T.tolist(),
            "K": self.K.tolist(),
            "D": self.D.tolist(),
            "w": self.w,
            "h": self.h,
        }
        cam_paras_dict = {
            "left": left_cam_paras,
        }
        with open(config_file, 'w') as f:
            json.dump(cam_paras_dict, f, indent=4)
        logging.info("Camera parameters saved to %s", config_file)

    def restore_config(self, config_file):
        config_paras = {
            "cam_paras_file": self.camparas_file,
            "preset_file": self.preset_file,
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
        self.camparas_file = resolve_project_path(config_paras["cam_paras_file"])
        self.preset_file = resolve_project_path(config_paras["preset_file"])
        self.max_depth = config_paras["max_depth"]
        self.fps = config_paras["fps"]
        self.type = config_paras["type"]


    def capture_image(self):
        align_to = rs.stream.color
        align = rs.align(align_to)

        frames = self.pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        
        color_frame = frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        CamBGR = np.array(color_frame.get_data())
        CamRGB = cv2.cvtColor(CamBGR, cv2.COLOR_BGR2RGB)
        CamD = np.array(depth_frame.get_data()) * self.depth_scale * 1000
        CamD[CamD > self.max_depth] = 0
        
        self.rgb_image = CamRGB
        self.depth_map = CamD
        return self.rgb_image, self.depth_map
    
    def _find_d435_serial(self):
        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            
            for device in devices:
                device_name = device.get_info(rs.camera_info.name)
                serial_number = device.get_info(rs.camera_info.serial_number)
                product_id = device.get_info(rs.camera_info.product_id)
                
                if ("D435" in device_name or "435" in device_name or 
                    product_id in ['0x0AD1', '0x0AD2', '0x0AD3', '0x0AD4', '0x0AD5']):
                    logging.info(f"Found D435 device: {device_name}, Serial: {serial_number}")
                    return serial_number
                    
            return None
            
        except Exception as e:
            logging.error(f"Error finding D435 serial: {e}")
            return None


HandCameraCallbacks = {
    'RealSense': RealSense435Camera,
}
