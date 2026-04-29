import time
import math
import functools
import numpy as np
import cv2
from PIL import Image
import os
import logging
import json
import open3d as o3d

try:
    import pyzed.sl as sl
except ImportError:
    sl = None

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

from .base_camera import BaseCamera

from utils.config import get_config, get_camera_config, resolve_project_path


class BasicMonoCamera(BaseCamera):
    def __init__(self, config_file=None, is_warmup=True, target_serial=None):
        super(BasicMonoCamera, self).__init__()

        cam_config = get_camera_config('realsense_d455')

        cur_path = os.path.dirname(os.path.abspath(__file__))
        self.camparas_file = cam_config.paras_file if cam_config else os.path.join(cur_path, "assets/calibration/external_camera/realsense455_paras.json")
        self.max_depth = cam_config.max_depth if cam_config else 10000
        self.fps = cam_config.fps if cam_config else 30
        self.type = "external"

        with open(self.camparas_file, 'r') as f:
            cam_paras_dict = json.load(f)
        left_cam_paras = cam_paras_dict["left"]

        self.R = np.array(left_cam_paras["R"])
        self.T = np.array(left_cam_paras["T"])
        self.K = np.array(left_cam_paras["K"])
        self.D = np.array(left_cam_paras["D"])
        self.w = left_cam_paras["w"]
        self.h = left_cam_paras["h"]

        self.rgb_image = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.depth_map = np.zeros((self.h, self.w), dtype=np.float32)

    def capture_image(self):
        logging.info("Capturing image from BasicMonoCamera.")
        return self.rgb_image, self.depth_map

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
        self.max_depth = config_paras["max_depth"]
        self.fps = config_paras["fps"]
        self.type = config_paras["type"]



class ZEDCamera(BaseCamera):
    def __init__(self, config_file=None, is_warmup=True):
        super(ZEDCamera, self).__init__()

        if is_warmup and sl is None:
            raise ImportError("pyzed is required to use ZEDCamera with is_warmup=True.")

        cam_config = get_camera_config('zed')

        cur_path = os.path.dirname(os.path.abspath(__file__))
        self.camparas_file = cam_config.paras_file if cam_config else os.path.join(cur_path, "assets/calibration/external_camera/zed_paras.json")
        self.max_depth = cam_config.max_depth if cam_config else 5000
        self.fps = cam_config.fps if cam_config else 30
        self.depth_mode = cam_config.depth_mode if cam_config else "ULTRA"

        self.cam_settings = cam_config.settings if cam_config and cam_config.settings else None

        with open(self.camparas_file, 'r') as f:
            cam_paras_dict = json.load(f)
        left_cam_paras = cam_paras_dict["left"]

        self.R = np.array(left_cam_paras["R"])
        self.T = np.array(left_cam_paras["T"])
        self.K = np.array(left_cam_paras["K"])
        self.D = np.array(left_cam_paras["D"])
        self.w = left_cam_paras["w"]
        self.h = left_cam_paras["h"]
        self.type = "external"

        self.left_image = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.right_image = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.depth_map = np.zeros((self.h, self.w), dtype=np.float32)

        if is_warmup:
            self.Cam = sl.Camera()
            self.CamRuntimeParameters = sl.RuntimeParameters()

            confidence_threshold = self.cam_settings.confidence_threshold if self.cam_settings else 80
            self.CamRuntimeParameters.confidence_threshold = confidence_threshold
            self.CamRuntimeParameters.texture_confidence_threshold = 100

            init_parames = sl.InitParameters()
            if self.w == 1280 and self.h == 720:
                init_parames.camera_resolution = sl.RESOLUTION.HD720
            elif self.w == 1920 and self.h == 1080:
                init_parames.camera_resolution = sl.RESOLUTION.HD1080

            depth_mode_map = {
                "ULTRA": sl.DEPTH_MODE.ULTRA,
                "QUALITY": sl.DEPTH_MODE.QUALITY,
                "PERFORMANCE": sl.DEPTH_MODE.PERFORMANCE,
                "NEURAL": sl.DEPTH_MODE.NEURAL,
            }
            init_parames.depth_mode = depth_mode_map.get(self.depth_mode, sl.DEPTH_MODE.ULTRA)
            init_parames.camera_fps = self.fps
            init_parames.coordinate_units = sl.UNIT.MILLIMETER
            init_parames.depth_minimum_distance = 200
            init_parames.depth_maximum_distance = self.max_depth

            init_parames.depth_stabilization = 1

            if self.Cam.open(init_parames) != sl.ERROR_CODE.SUCCESS:
                print("Failed to open ZED camera")
                return

            exposure = self.cam_settings.exposure if self.cam_settings else 60
            brightness = self.cam_settings.brightness if self.cam_settings else 6
            contrast = self.cam_settings.contrast if self.cam_settings else 4
            gain = self.cam_settings.gain if self.cam_settings else 4

            self.Cam.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, exposure)
            self.Cam.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS, brightness)
            self.Cam.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST, contrast)
            self.Cam.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, gain)

            for i in range(30):
                rgb_image, depth_map = self.capture_image()

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
        self.max_depth = config_paras["max_depth"]
        self.fps = config_paras["fps"]
        self.type = config_paras["type"]

    def capture_image(self):
        raw_image = sl.Mat()
        raw_depth = sl.Mat()
        raw_dimg = sl.Mat()
        raw_pcd = sl.Mat()
        raw_right_image = sl.Mat()

        if self.Cam.grab(self.CamRuntimeParameters) == sl.ERROR_CODE.SUCCESS:
            self.Cam.retrieve_image(raw_image, sl.VIEW.LEFT)
            self.Cam.retrieve_measure(raw_depth, sl.MEASURE.DEPTH, sl.MEM.CPU)
            self.Cam.retrieve_image(raw_dimg, sl.VIEW.DEPTH)
            self.Cam.retrieve_measure(raw_pcd, sl.MEASURE.XYZBGRA, sl.MEM.CPU)
            self.Cam.retrieve_image(raw_right_image, sl.VIEW.RIGHT)

            CamBGR = raw_image.get_data()
            CamD = raw_depth.get_data()
            CamDIMG = raw_dimg.get_data()
            CamPCD = raw_pcd.get_data()
            CamBGR_right = raw_right_image.get_data()

            CamBGR = np.array(CamBGR)[..., :3]
            CamRGB = cv2.cvtColor(CamBGR, cv2.COLOR_BGR2RGB)
            CamD = np.array(CamD)
            
            CamD[np.isnan(CamD)] = 0
            CamD[np.isinf(CamD)] = 0
            CamD[CamD > self.max_depth] = 0
            CamD[CamD < 200] = 0
            
            CamD = cv2.medianBlur(CamD.astype(np.float32), 5)
            
            CamRGB_right = cv2.cvtColor(np.array(CamBGR_right)[..., :3], cv2.COLOR_BGR2RGB)

        else:
            CamRGB = None
            CamD = None
            CamDIMG = None
            CamPCD = None
            CamRGB_right = None
            print("External camera captures fail")

        self.rgb_image = CamRGB
        self.rgb_right_image = CamRGB_right
        self.depth_map = CamD        
        return self.rgb_image, self.depth_map
    
    def capture_two_images(self):
        raw_image = sl.Mat()
        raw_depth = sl.Mat()
        raw_dimg = sl.Mat()
        raw_pcd = sl.Mat()
        raw_right_image = sl.Mat()

        if self.Cam.grab(self.CamRuntimeParameters) == sl.ERROR_CODE.SUCCESS:
            self.Cam.retrieve_image(raw_image, sl.VIEW.LEFT)
            self.Cam.retrieve_measure(raw_depth, sl.MEASURE.DEPTH, sl.MEM.CPU)
            self.Cam.retrieve_image(raw_dimg, sl.VIEW.DEPTH)
            self.Cam.retrieve_measure(raw_pcd, sl.MEASURE.XYZBGRA, sl.MEM.CPU)
            self.Cam.retrieve_image(raw_right_image, sl.VIEW.RIGHT)

            CamBGR = raw_image.get_data()
            CamD = raw_depth.get_data()
            CamDIMG = raw_dimg.get_data()
            CamPCD = raw_pcd.get_data()
            CamBGR_right = raw_right_image.get_data()

            CamBGR = np.array(CamBGR)[..., :3]
            CamRGB = cv2.cvtColor(CamBGR, cv2.COLOR_BGR2RGB)
            CamD = np.array(CamD)
            
            CamD[np.isnan(CamD)] = 0
            CamD[np.isinf(CamD)] = 0
            CamD[CamD > self.max_depth] = 0
            CamD[CamD < 200] = 0
            
            CamD = cv2.medianBlur(CamD.astype(np.float32), 5)
            
            CamRGB_right = cv2.cvtColor(np.array(CamBGR_right)[..., :3], cv2.COLOR_BGR2RGB)

        else:
            CamRGB = None
            CamD = None
            CamDIMG = None
            CamPCD = None
            CamRGB_right = None
            print("External camera captures fail")

        self.left_image = CamRGB
        self.right_image = CamRGB_right
        self.depth_map = CamD        
        return self.left_image, self.right_image, self.depth_map

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

class RealSense455Camera(BaseCamera):
    def __init__(self, config_file=None, is_warmup=True, target_serial=None):
        super(RealSense455Camera, self).__init__()

        if is_warmup and rs is None:
            raise ImportError("pyrealsense2 is required to use RealSense455Camera with is_warmup=True.")

        cam_config = get_camera_config('realsense_d455')

        cur_path = os.path.dirname(os.path.abspath(__file__))
        self.camparas_file = cam_config.paras_file if cam_config else os.path.join(cur_path, "assets/calibration/external_camera/realsense455_paras.json")
        self.preset_file = os.path.join(cur_path, "assets/calibration/external_camera/realsense455_HighResHighAccuracyPreset.json")
        self.max_depth = cam_config.max_depth if cam_config else 10000
        self.fps = cam_config.fps if cam_config else 30
        self.type = "external"
        self.target_serial = target_serial

        with open(self.camparas_file, 'r') as f:
            cam_paras_dict = json.load(f)
        left_cam_paras = cam_paras_dict["left"]

        self.R = np.array(left_cam_paras["R"])
        self.T = np.array(left_cam_paras["T"])
        self.K = np.array(left_cam_paras["K"])
        self.D = np.array(left_cam_paras["D"])
        self.w = left_cam_paras["w"]
        self.h = left_cam_paras["h"]

        self.rgb_image = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.depth_map = np.zeros((self.h, self.w), dtype=np.float32)

        if is_warmup:
            self._initialize_camera()
            for i in range(30):
                rgb_image, depth_map = self.capture_image()

    def _find_d455_serial(self):
        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            
            for device in devices:
                device_name = device.get_info(rs.camera_info.name)
                serial_number = device.get_info(rs.camera_info.serial_number)
                product_id = device.get_info(rs.camera_info.product_id)
                
                if ("D455" in device_name or "455" in device_name or 
                    product_id in ['0x0B5C', '0x0B5B']):
                    logging.info(f"Found D455 device: {device_name}, Serial: {serial_number}")
                    return serial_number
                    
            return None
            
        except Exception as e:
            logging.error(f"Error finding D455 serial: {e}")
            return None

    def _initialize_camera(self):
        try:
            self.ctx = rs.context()
            devices = self.ctx.query_devices()
            
            if len(devices) == 0:
                raise RuntimeError("No RealSense devices found")
            
            if self.target_serial:
                target_device = None
                for device in devices:
                    serial_number = device.get_info(rs.camera_info.serial_number)
                    if serial_number == self.target_serial:
                        device_name = device.get_info(rs.camera_info.name)
                        logging.info(f"Found target device with serial {self.target_serial}: {device_name}")
                        if "D455" in device_name or "455" in device_name:
                            target_device = device
                            break
                        else:
                            raise RuntimeError(f"Device with serial {self.target_serial} is not a D455 device: {device_name}")
                
                if target_device is None:
                    raise RuntimeError(f"No device found with serial number: {self.target_serial}")
                
                self.device = target_device
            else:
                d455_device = None
                for device in devices:
                    device_name = device.get_info(rs.camera_info.name)
                    product_id = device.get_info(rs.camera_info.product_id)
                    
                    if ("D455" in device_name or "455" in device_name or 
                        product_id in ['0x0B5C', '0x0B5B']):
                        d455_device = device
                        serial_number = device.get_info(rs.camera_info.serial_number)
                        logging.info(f"Found RealSense D455 device: {device_name}, Serial: {serial_number}")
                        break
                    else:
                        logging.warning(f"Found non-D455 device: {device_name}, skipping...")
                
                if d455_device is None:
                    raise RuntimeError("No RealSense D455 device found. Please ensure D455 is connected.")
                
                self.device = d455_device
            
            self.pipeline = rs.pipeline(self.ctx)
            config = rs.config()
            
            config.enable_device(self.device.get_info(rs.camera_info.serial_number))
            
            config.enable_stream(rs.stream.color, self.w, self.h, rs.format.bgr8, self.fps)
            config.enable_stream(rs.stream.depth, self.w, self.h, rs.format.z16, self.fps)
            
            self.profile = self.pipeline.start(config)

            try:
                depth_sensor = self.profile.get_device().first_depth_sensor()
                color_sensor = self.profile.get_device().first_color_sensor()

                depth_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.depth))
                color_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.color))

                depth_intrinsics = depth_profile.get_intrinsics()
                color_intrinsics = color_profile.get_intrinsics()

                logging.info("Depth camera intrinsics:")
                logging.info(f"  Width: {depth_intrinsics.width}")
                logging.info(f"  Height: {depth_intrinsics.height}")
                logging.info(f"  Fx: {depth_intrinsics.fx}")
                logging.info(f"  Fy: {depth_intrinsics.fy}")
                logging.info(f"  Ppx: {depth_intrinsics.ppx}")
                logging.info(f"  Ppy: {depth_intrinsics.ppy}")
                logging.info(f"  Model: {depth_intrinsics.model}")
                logging.info(f"  Coeffs (distortion coefficients): {depth_intrinsics.coeffs}")

                logging.info("Color camera intrinsics:")
                logging.info(f"  Width: {color_intrinsics.width}")
                logging.info(f"  Height: {color_intrinsics.height}")
                logging.info(f"  Fx: {color_intrinsics.fx}")
                logging.info(f"  Fy: {color_intrinsics.fy}")
                logging.info(f"  Ppx: {color_intrinsics.ppx}")
                logging.info(f"  Ppy: {color_intrinsics.ppy}")
                logging.info(f"  Model: {color_intrinsics.model}")
                logging.info(f"  Coeffs (distortion coefficients): {color_intrinsics.coeffs}")

            except Exception as e:
                logging.error(f"Failed to get camera intrinsics: {e}")
            
            self._configure_depth_sensor()
            
            depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            logging.info(f"Depth Scale is: {self.depth_scale}")
            
            self.align = rs.align(rs.stream.color)
            
        except Exception as e:
            logging.error(f"Failed to initialize RealSense D455 camera: {e}")
            raise

    def _configure_depth_sensor(self):
        try:
            depth_sensor = self.profile.get_device().first_depth_sensor()
            
            depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
            
            depth_sensor.set_option(rs.option.exposure, 8500)
            
            depth_sensor.set_option(rs.option.laser_power, 240.0)
            
            depth_sensor.set_option(rs.option.gain, 16.0)
            
            color_sensor = self.profile.get_device().first_color_sensor()
            color_sensor.set_option(rs.option.enable_auto_white_balance, 1)
            
            logging.info("Depth sensor configured successfully")
            
        except Exception as e:
            logging.warning(f"Failed to configure depth sensor: {e}")

    def capture_image(self):
        try:
            frames = self.pipeline.wait_for_frames()
            
            aligned_frames = self.align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                logging.warning("Failed to get valid frames")
                return None, None
            
            CamBGR = np.array(color_frame.get_data())
            CamRGB = cv2.cvtColor(CamBGR, cv2.COLOR_BGR2RGB)
            
            CamD = np.array(depth_frame.get_data()) * self.depth_scale * 1000
            CamD[CamD > self.max_depth] = 0
            CamD[np.isnan(CamD)] = 0
            
        except Exception as e:
            logging.error(f"RealSense D455 camera capture error: {e}")
            CamRGB = None
            CamD = None

        self.rgb_image = CamRGB
        self.depth_map = CamD
        return self.rgb_image, self.depth_map

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
        self.preset_file = resolve_project_path(config_paras.get("preset_file", self.preset_file))
        self.max_depth = config_paras["max_depth"]
        self.fps = config_paras["fps"]
        self.type = config_paras["type"]

    def get_intrinsics(self):
        if hasattr(self, 'profile'):
            color_stream = self.profile.get_stream(rs.stream.color)
            intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            return {
                'width': intrinsics.width,
                'height': intrinsics.height,
                'fx': intrinsics.fx,
                'fy': intrinsics.fy,
                'ppx': intrinsics.ppx,
                'ppy': intrinsics.ppy,
                'coeffs': intrinsics.coeffs
            }
        return None

ExternCameraCallbacks = {
    'ZED': ZEDCamera,
    'RealSense455': RealSense455Camera,
    'BaseCamera': BaseCamera,
    'BasicMonoCamera': BasicMonoCamera,
}
