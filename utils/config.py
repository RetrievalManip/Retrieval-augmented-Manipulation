
import os
import yaml
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_CONFIG_PATH = os.environ.get(
    "FAIRINO_CONFIG",
    os.path.join(PROJECT_ROOT, "config", "config.yaml")
)

_config_instance = None
_config_lock = threading.Lock()


@dataclass
class APIConfig:
    api_key: str = ""
    base_url: str = ""
    model_name: str = ""
    max_tokens: int = 2000
    temperature: float = 1.0


@dataclass
class GroundingConfig:
    sam2_checkpoint: str = ""
    sam2_model_config: str = ""
    grounding_dino_config: str = ""
    grounding_dino_checkpoint: str = ""
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    device: str = "cuda:0"

@dataclass
class RAMConfig:
    checkpoint: str = ""
    template_dir: str = ""
    ram_k: int = 5
    foundation_model: str = "dinov2-b14"
    feat_layer: List[int] = field(default_factory=lambda: [7, 9, 11])
    img_size: int = 224
    n_pts: int = 1024
    device: str = "cuda:0"


@dataclass
class CameraSettings:
    confidence_threshold: int = 80
    exposure: int = 60
    brightness: int = 6
    contrast: int = 4
    gain: int = 4


@dataclass
class CameraConfig:
    paras_file: str = ""
    max_depth: int = 5000
    min_depth: int = 0
    fps: int = 30
    depth_mode: str = ""
    settings: Optional[CameraSettings] = None


@dataclass
class CameraRecordingConfig:
    buffer_size: int = 5
    fps: int = 2


@dataclass
class GripperConfig:
    idx: int = 1
    offset: int = 20
    close_width: int = 5


@dataclass
class RobotConfig:
    ip: str = ""
    reset_pos: List[float] = field(default_factory=list)
    workspace: List[float] = field(default_factory=list)
    gripper: GripperConfig = field(default_factory=GripperConfig)


@dataclass
class PlannerConfig:
    map_size: int = 300
    workspace: List[float] = field(default_factory=list)
    table_height: int = -100


@dataclass
class ObjectCategoriesConfig:
    static_seg_ram: List[str] = field(default_factory=list)
    dynamic_seg_only: List[str] = field(default_factory=list)
    dynamic_seg_ram: List[str] = field(default_factory=list)


@dataclass
class SubtaskCacheConfig:
    enabled: bool = False
    cached_objects: List[str] = field(default_factory=list)


@dataclass
class TaskConfig:
    default_prompt: str = ""
    output_dir: str = ""
    use_timestamped_run_dir: bool = False
    batch_reuse_perception_snapshot: bool = False
    max_rounds: int = 10
    vlm_execution_mode: str = "online_serial"
    vlm_parallel_workers: int = 2
    batch_require_independent_subtasks: bool = True
    action_execution_mode: str = "serial_from_json"
    execution_state_mode: str = "internal_only"
    object_categories: ObjectCategoriesConfig = field(default_factory=ObjectCategoriesConfig)
    known_objects: List[Dict] = field(default_factory=list)
    subtask_cache: SubtaskCacheConfig = field(default_factory=SubtaskCacheConfig)


class Config:

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        self.config_path = config_path
        self._raw_config: Dict[str, Any] = {}

        self._load_config()

        self._parse_config()

    def _load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._raw_config = yaml.safe_load(f)

    def _parse_config(self):
        project = self._raw_config.get('project', {})
        self.root_dir = self._resolve_path(project.get('root_dir', PROJECT_ROOT))

        api = self._raw_config.get('api', {})
        profiles = api.get('profiles', None)

        if profiles:
            active = api.get('active_profile', '')
            if active not in profiles:
                available = list(profiles.keys())
                raise ValueError(
                    f"API active_profile '{active}' not found. "
                    f"Available profiles: {available}"
                )
            api_data = profiles[active]
        else:
            api_data = api

        self.api = APIConfig(
            api_key=api_data.get('api_key', ''),
            base_url=api_data.get('base_url', ''),
            model_name=api_data.get('model_name', ''),
            max_tokens=api_data.get('max_tokens', 2000),
            temperature=api_data.get('temperature', 1.0)
        )

        vision = self._raw_config.get('vision', {})
        grounding = vision.get('grounding', {})
        self.grounding = GroundingConfig(
            sam2_checkpoint=self._resolve_path(grounding.get('sam2_checkpoint', '')),
            sam2_model_config=self._resolve_path(grounding.get('sam2_model_config', '')),
            grounding_dino_config=self._resolve_path(grounding.get('grounding_dino_config', '')),
            grounding_dino_checkpoint=self._resolve_path(grounding.get('grounding_dino_checkpoint', '')),
            box_threshold=grounding.get('box_threshold', 0.35),
            text_threshold=grounding.get('text_threshold', 0.25),
            device=grounding.get('device', 'cuda:0')
        )

        ram = vision.get('ram', {})
        self.ram = RAMConfig(
            checkpoint=self._resolve_path(ram.get('checkpoint', '')),
            template_dir=self._resolve_path(ram.get('template_dir', '')),
            ram_k=ram.get('ram_k', 5),
            foundation_model=ram.get('foundation_model', 'dinov2-b14'),
            feat_layer=ram.get('feat_layer', [7, 9, 11]),
            img_size=ram.get('img_size', 224),
            n_pts=ram.get('n_pts', 1024),
            device=ram.get('device', 'cuda:0')
        )

        cameras = self._raw_config.get('cameras', {})
        self.cameras: Dict[str, CameraConfig] = {}

        for cam_name, cam_cfg in cameras.items():
            settings = None
            if 'settings' in cam_cfg:
                settings = CameraSettings(
                    confidence_threshold=cam_cfg['settings'].get('confidence_threshold', 80),
                    exposure=cam_cfg['settings'].get('exposure', 60),
                    brightness=cam_cfg['settings'].get('brightness', 6),
                    contrast=cam_cfg['settings'].get('contrast', 4),
                    gain=cam_cfg['settings'].get('gain', 4)
                )

            self.cameras[cam_name] = CameraConfig(
                paras_file=self._resolve_path(cam_cfg.get('paras_file', '')),
                max_depth=cam_cfg.get('max_depth', 5000),
                min_depth=cam_cfg.get('min_depth', 0),
                fps=cam_cfg.get('fps', 30),
                depth_mode=cam_cfg.get('depth_mode', ''),
                settings=settings
            )

        robot = self._raw_config.get('robot', {})
        gripper_cfg = robot.get('gripper', {})
        self.robot = RobotConfig(
            ip=robot.get('ip', ''),
            reset_pos=robot.get('reset_pos', []),
            workspace=robot.get('workspace', []),
            gripper=GripperConfig(
                idx=gripper_cfg.get('idx', 1),
                offset=gripper_cfg.get('offset', 20),
                close_width=gripper_cfg.get('close_width', 5)
            )
        )

        planner = self._raw_config.get('planner', {})
        self.planner = PlannerConfig(
            map_size=planner.get('map_size', 300),
            workspace=planner.get('workspace', []),
            table_height=planner.get('table_height', -100)
        )

        task = self._raw_config.get('task', {})
        obj_cats = task.get('object_categories', {})
        object_categories = ObjectCategoriesConfig(
            static_seg_ram=obj_cats.get('static_seg_ram', []),
            dynamic_seg_only=obj_cats.get('dynamic_seg_only', []),
            dynamic_seg_ram=obj_cats.get('dynamic_seg_ram', [])
        )

        subtask_cache_data = task.get('subtask_cache', {})
        subtask_cache = SubtaskCacheConfig(
            enabled=subtask_cache_data.get('enabled', False),
            cached_objects=subtask_cache_data.get('cached_objects', [])
        )

        self.task = TaskConfig(
            default_prompt=task.get('default_prompt', ''),
            output_dir=self._resolve_path(task.get('output_dir', 'outputs')),
            use_timestamped_run_dir=task.get('use_timestamped_run_dir', False),
            batch_reuse_perception_snapshot=task.get('batch_reuse_perception_snapshot', False),
            max_rounds=task.get('max_rounds', 10),
            vlm_execution_mode=task.get('vlm_execution_mode', 'online_serial'),
            vlm_parallel_workers=task.get('vlm_parallel_workers', 2),
            batch_require_independent_subtasks=task.get('batch_require_independent_subtasks', True),
            action_execution_mode=task.get('action_execution_mode', 'serial_from_json'),
            execution_state_mode=task.get('execution_state_mode', 'internal_only'),
            object_categories=object_categories,
            known_objects=task.get('known_objects', []),
            subtask_cache=subtask_cache,
        )

    def _resolve_path(self, value: str) -> str:
        if not value:
            return value
        return value.replace("${PROJECT_ROOT}", PROJECT_ROOT)

    def get_camera_config(self, camera_name: str) -> Optional[CameraConfig]:
        return self.cameras.get(camera_name)

    def get_raw(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self._raw_config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value


def get_config(config_path: str = DEFAULT_CONFIG_PATH) -> Config:
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = Config(config_path)
    return _config_instance


def reload_config(config_path: str = DEFAULT_CONFIG_PATH) -> Config:
    global _config_instance
    with _config_lock:
        _config_instance = Config(config_path)
    return _config_instance


def get_api_config() -> APIConfig:
    return get_config().api


def get_grounding_config() -> GroundingConfig:
    return get_config().grounding

def get_ram_config() -> RAMConfig:
    return get_config().ram


def get_robot_config() -> RobotConfig:
    return get_config().robot


def get_planner_config() -> PlannerConfig:
    return get_config().planner


def get_task_config() -> TaskConfig:
    return get_config().task
    

def get_object_categories_config() -> ObjectCategoriesConfig:
    return get_config().task.object_categories


def get_subtask_cache_config() -> SubtaskCacheConfig:
    return get_config().task.subtask_cache


def get_camera_config(camera_name: str) -> Optional[CameraConfig]:
    return get_config().get_camera_config(camera_name)


def get_camera_recording_config() -> CameraRecordingConfig:
    config = get_config()
    recording_data = config._raw_config.get('cameras', {}).get('recording', {})
    return CameraRecordingConfig(
        buffer_size=recording_data.get('buffer_size', 5),
        fps=recording_data.get('fps', 2)
    )


def resolve_project_path(value: str) -> str:
    if not value:
        return value
    return value.replace("${PROJECT_ROOT}", PROJECT_ROOT)
