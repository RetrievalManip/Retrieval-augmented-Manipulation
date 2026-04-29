import numpy as np
import viser
import threading
import time
import logging
from typing import Optional, Dict, Any, List, Tuple

class PointCloudViewer:
    def __init__(self, port: int = 8080, host: str = "localhost", auto_start: bool = True):
        self.server = viser.ViserServer(host=host, port=port)
        self.point_clouds = {}
        self.is_running = False
        self.thread = None
        
        if auto_start:
            self.start()
            
        logging.info(f"Point cloud viewer initialized. Visit http://{host}:{port} to view.")
        
    def start(self):
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._run_server, daemon=True)
            self.thread.start()
            logging.info("Viewer server started in background thread.")
            
    def _run_server(self):
        while self.is_running:
            time.sleep(0.1)
            
    def stop(self):
        self.is_running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None
        logging.info("Viewer server stopped.")
            
    def add_point_cloud(self, 
                       points: np.ndarray, 
                       colors: Optional[np.ndarray] = None, 
                       name: str = "point_cloud",
                       point_size: float = 2.0,
                       reset_view: bool = False):
        points = np.array(points).reshape(-1, 3)
        
        if colors is not None:
            colors = np.array(colors).reshape(-1, 3)
            if colors.max() > 1.0:
                colors = colors / 255.0
        else:
            colors = np.ones_like(points) * 0.7
            
        if name in self.point_clouds:
            self.remove_point_cloud(name)
            
        pc = self.server.add_point_cloud(
            name=name,
            points=points,
            colors=colors,
            point_size=point_size,
        )
        
        self.point_clouds[name] = pc
        
        if reset_view:
            self.server.camera.reset_view()
            
        return pc
        
    def remove_point_cloud(self, name: str):
        if name in self.point_clouds:
            self.server.remove_point_cloud(name)
            del self.point_clouds[name]
            return True
        return False
    
    def clear(self):
        for name in list(self.point_clouds.keys()):
            self.remove_point_cloud(name)
            
    def add_transform_axes(self, name: str, 
                          position: List[float] = [0, 0, 0], 
                          orientation: List[float] = [0, 0, 0, 1],
                          scale: float = 0.1):
        frame = self.server.add_frame(
            name=name,
            wxyz=orientation,
            position=position,
            axes_length=scale,
            axes_radius=scale * 0.1,
        )
        return frame
    
    def __del__(self):
        self.stop()
