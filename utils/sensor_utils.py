import numpy as np
import math
from pathlib import Path
import sys
import time
import cv2
import threading
import os

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
GSROBOTICS_PATH = ROOT_DIR / "visions" / "gsrobotics"

if str(GSROBOTICS_PATH) not in sys.path:
    sys.path.insert(0, str(GSROBOTICS_PATH))

try:
    from utilities.gelsightmini import GelSightMini
    from utilities.marker_tracker import MarkerTracker
except ImportError as e:
    print(f"Warning: Could not import GelSightMini or MarkerTracker: {e}")
    print(f"Expected gsrobotics path: {GSROBOTICS_PATH}")
    pass

class ForceSensor:
    def __init__(self, threshold=5.0, device_index=0):
        self.threshold = threshold
        self.triggered = False
        self.current_iter = 0
        self.device_index = device_index
        
        self.cam = GelSightMini(target_width=320, target_height=240, border_fraction=0.15)
        devices = self.cam.get_device_list()
        if not devices:
            raise RuntimeError("No GelSight Mini devices found.")
            
        connected = False
        sorted_indices = sorted(devices.keys())
        
        for idx in sorted_indices:
            try:
                print(f"Trying device index {idx}...")
                self.cam.select_device(idx)
                self.cam.start()
                
                frame = None
                for _ in range(10):
                    frame = self.cam.update(0.0)
                    if frame is not None:
                        break
                    time.sleep(0.1)
                
                if frame is None:
                     print(f"Device {idx} opened but failed to return frames (timeout).")
                     continue

                self.device_index = idx
                connected = True
                print(f"Connected to device {idx}.")
                break
            except Exception as e:
                print(f"Failed to connect to device {idx}: {e}")
        
        if not connected:
            raise RuntimeError("Failed to connect to any GelSight Mini device.")
        
        self.is_tracking_initialized = False
        self._init_tracking()

        self.lock = threading.Lock()
        self.running = True
        self.current_force = np.zeros(3)
        self.latest_frame_data = None
        
        self.is_observing = False
        self.start_obs_val = 0.0
        self.accumulated_obs_val = 0.0
        self.accumulated_frames = 0

        self.save_vis_enabled = False
        self.save_vis_path = None
        self.vis_frame_count = 0
        self.internal_frame_count = 0

        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def _init_tracking(self):
        print("Initializing tracking... waiting for first frame.")
        first_frame = None
        while first_frame is None:
            first_frame = self.cam.update(0.0)
            if first_frame is None:
                time.sleep(0.01)
        
        img = np.float32(first_frame) / 255.0
        self.tracker = MarkerTracker(img)
        
        self.old_gray = cv2.cvtColor(first_frame, cv2.COLOR_RGB2GRAY)
        marker_centers = self.tracker.initial_marker_center
        self.Ox = marker_centers[:, 1]
        self.Oy = marker_centers[:, 0]
        self.nct = len(marker_centers)
        
        grid_spacing = float(self.tracker.grid_spacing)
        y_min = float(self.Oy.min())
        y_max = float(self.Oy.max())
        row_band = 0.5 * grid_spacing
        self.top_row_ids = np.where(self.Oy <= y_min + row_band)[0]
        self.bottom_row_ids = np.where(self.Oy >= y_max - row_band)[0]
        
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        
        self.p0 = np.stack([self.Ox, self.Oy], axis=1).reshape(-1, 1, 2).astype(np.float32)
        self.marker_ids = np.arange(self.nct, dtype=int)
        self.init_centers = np.stack([self.Ox, self.Oy], axis=1)
        
        self.is_tracking_initialized = True
        print(f"Tracking initialized with {self.nct} markers.")

    def _update_loop(self):
        while self.running:
            force, frame_data = self._read_force_internal()
            self.internal_frame_count += 1
            
            if self.is_observing:
                delta_force = force[2] - self.current_force[2]
                self.accumulated_obs_val = abs(delta_force) if abs(delta_force) > self.accumulated_obs_val else self.accumulated_obs_val
                print('******* delta_force: ', self.internal_frame_count, self.current_force[2], force[2], delta_force)
                print('******* accumulated_obs_val: ', self.accumulated_obs_val)
                print('******* accumulated_frames: ', self.accumulated_frames + 1)
                self.accumulated_frames += 1         
            with self.lock:
                self.current_force = force
                if frame_data is not None:
                    self.latest_frame_data = frame_data
            time.sleep(0.001)

    def _generate_vis_image(self, frame, curr_pts, ids):
        try:
            VIS_SCALE = 2.0 
            vis_h, vis_w = int(frame.shape[0] * VIS_SCALE), int(frame.shape[1] * VIS_SCALE)
            vis_img = cv2.resize(frame.copy(), (vis_w, vis_h), interpolation=cv2.INTER_LINEAR)
            
            init_pts = self.init_centers[ids]
            
            ARROW_BODY_SCALE = 3.0

            for i in range(len(curr_pts)):
                pt_start_orig = init_pts[i]
                pt_end_orig = curr_pts[i]
                
                diff = pt_end_orig - pt_start_orig
                
                pt_start_scaled = pt_start_orig * VIS_SCALE
                
                pt_end_scaled = pt_start_scaled + (diff * VIS_SCALE * ARROW_BODY_SCALE)
                
                start_pt = tuple(pt_start_scaled.astype(int))
                end_pt = tuple(pt_end_scaled.astype(int))
                
                color = (255, 0, 0) 
                
                thickness = 6
                
                vec_float = pt_end_scaled - pt_start_scaled
                length = np.linalg.norm(vec_float)
                
                MIN_VIS_LEN = 1.0
                
                if length > 1e-5:
                    angle = np.arctan2(vec_float[1], vec_float[0])
                    
                    if length < MIN_VIS_LEN:
                        length = MIN_VIS_LEN
                        end_pt_float = pt_start_scaled + (vec_float / np.linalg.norm(vec_float) * MIN_VIS_LEN)
                        end_pt = tuple(end_pt_float.astype(int))
                    
                    head_len = length * 0.35
                    head_width = max(thickness * 2.5, head_len * 0.8) 
                    
                    p_tip = np.array(end_pt)
                    
                    p_base_center_x = p_tip[0] - head_len * np.cos(angle)
                    p_base_center_y = p_tip[1] - head_len * np.sin(angle)
                    p_base_center = (int(p_base_center_x), int(p_base_center_y))
                    
                    cv2.line(vis_img, start_pt, p_base_center, color, thickness, cv2.LINE_AA)
                    
                    perp_x = -np.sin(angle)
                    perp_y = np.cos(angle)
                    
                    hw = head_width * 0.5
                    p1_x = p_base_center_x + hw * perp_x
                    p1_y = p_base_center_y + hw * perp_y
                    p2_x = p_base_center_x - hw * perp_x
                    p2_y = p_base_center_y - hw * perp_y
                    
                    triangle_cnt = np.array([p_tip, [p1_x, p1_y], [p2_x, p2_y]], dtype=np.int32)
                    cv2.fillPoly(vis_img, [triangle_cnt], color)
            
            return vis_img
        except Exception as e:
            print(f"Error generating visualization: {e}")
            return None

    def get_latest_vis_image(self):
        with self.lock:
            if not hasattr(self, 'latest_frame_data') or self.latest_frame_data is None:
                return None, 0
            data = self.latest_frame_data
            current_count = self.internal_frame_count
        
        vis_img = self._generate_vis_image(data['frame'], data['curr_pts'], data['ids'])
        if vis_img is not None:
            return cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR), current_count
        return None, current_count

    def _read_force_internal(self):
        if not self.is_tracking_initialized:
            return np.zeros(3), None
            
        frame = self.cam.update(0.0)
        if frame is None:
            return np.zeros(3), None
            
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            self.old_gray, frame_gray, self.p0, None, **self.lk_params
        )
        
        if p1 is None or st is None:
            self.old_gray = frame_gray
            return np.zeros(3), None
        
        valid_mask = st.flatten() == 1
        if not np.any(valid_mask):
            self.old_gray = frame_gray
            return np.zeros(3), None
        
        good_new = p1[valid_mask].reshape(-1, 2)
        marker_ids_valid = self.marker_ids[valid_mask]
        
        self.p0 = good_new.reshape(-1, 1, 2)
        self.marker_ids = marker_ids_valid
        self.old_gray = frame_gray.copy()
        
        top_mask = np.isin(marker_ids_valid, self.top_row_ids)
        bot_mask = np.isin(marker_ids_valid, self.bottom_row_ids)
        
        def get_mean_dx(mask):
            if not np.any(mask):
                return 0.0
            ids = marker_ids_valid[mask]
            init_pts = self.init_centers[ids]
            curr_pts = good_new[mask]
            disp = curr_pts - init_pts
            disp_len = np.abs(disp[:, 0])

            return disp[np.argmax(disp_len), 0]
        
        top_dx = get_mean_dx(top_mask)
        bot_dx = get_mean_dx(bot_mask)
        
        rotation_metric = bot_dx - top_dx
        
        force = np.array([top_dx, bot_dx, rotation_metric])
        
        frame_data = {
            'frame': frame.copy(),
            'curr_pts': good_new.copy(),
            'ids': marker_ids_valid.copy()
        }

        if self.save_vis_enabled:
            try:
                vis_img = self._generate_vis_image(frame, good_new, marker_ids_valid)
                if vis_img is not None:
                    vis_img_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
                    filename = os.path.join(self.save_vis_path, f"gelsight_{self.vis_frame_count:06d}.png")
                    cv2.imwrite(filename, vis_img_bgr)
                    self.vis_frame_count += 1
            except Exception as e:
                print(f"Error saving visualization: {e}")

        return force, frame_data

    def read_force(self):
        with self.lock:
            return self.current_force.copy()

    def start_obs_rot(self):
        with self.lock:
            self.is_observing = True
            self.start_obs_val = self.current_force[2]
            self.accumulated_obs_val = 0.0
            self.accumulated_frames = 0
            self.triggered = False


    def end_obs_rot(self):
        with self.lock:
            self.is_observing = False


    def enable_vis_saving(self, save_path):
        with self.lock:
            self.save_vis_path = save_path
            os.makedirs(self.save_vis_path, exist_ok=True)
            self.save_vis_enabled = True
            
    def disable_vis_saving(self):
        with self.lock:
            self.save_vis_enabled = False

    def is_threshold_exceeded(self):
        with self.lock:
            val = self.accumulated_obs_val
            thresh = self.threshold
            exceeded = val > thresh
            self.triggered = exceeded
            self.accumulated_obs_val = 0.0
            self.accumulated_frames = 0
            return exceeded, val, thresh

    def increment_iter(self):
        self.current_iter += 1

    def reset_iter(self):
        self.current_iter = 0
        
    def close(self):
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        
        if hasattr(self, 'cam') and self.cam.camera:
            self.cam.camera.release()

def main():
    print("ForceSensor Demo")
    print("Make sure the GelSight Mini is connected...")

    try:
        sensor = ForceSensor(threshold=2.0)
        print("Sensor initialized successfully. Waiting for tracking to stabilize...")
        time.sleep(2)

        print("\n--- Start rotation observation (start_obs_rot) ---")
        print("Rotate the object touching the sensor surface during the next 5 seconds...")
        sensor.start_obs_rot()

        for i in range(5):
            force = sensor.read_force()
            print(f"[{i+1}s] Current reading (Top, Bot, Rot): {force}")
            time.sleep(1)

        sensor.end_obs_rot()
        print("--- End rotation observation (end_obs_rot) ---")

        exceeded, val, thresh = sensor.is_threshold_exceeded()
        if exceeded:
            print(f"Result: threshold triggered. (current value: {val:.4f}, threshold: {thresh})")
            print("Significant rotation detected.")
        else:
            print(f"Result: threshold not triggered. (current value: {val:.4f}, threshold: {thresh})")
            print("Rotation is within the allowed range.")

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if 'sensor' in locals():
            sensor.close()
            print("Sensor closed.")

if __name__ == "__main__":
    main()
