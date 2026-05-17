import sys
import os
cur_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_path)

import time
import math
import functools
import numpy as np
import json
import cv2
from PIL import Image
import logging
from contextlib import nullcontext

import torch
from torchvision.ops import box_convert
import supervision as sv
import grounding_dino.groundingdino.datasets.transforms as T

from .base_grounding import BaseGrounding
from grounding_dino.groundingdino.util.inference import load_model, predict
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from utils.config import get_grounding_config

class DinoGrounding(BaseGrounding):
    def __init__(self, config_file=None, device=None, sam2_predictor=None):
        """
        Initialize DinoGrounding.

        Args:
            config_file: Path to configuration JSON file (DEPRECATED - use config/config.yaml)
            device: Device to run models on (cuda:0, cpu, etc.)
            sam2_predictor: Optional external SAM2ImagePredictor instance for sharing.
                           If provided, SAM2 will not be initialized internally.
        """
        # Load configuration from unified config
        grounding_config = get_grounding_config()

        # Use config values (absolute paths from config.yaml)
        self.sam2_checkpoint = grounding_config.sam2_checkpoint
        self.sam2_model_config = grounding_config.sam2_model_config
        self.grounding_dino_config = grounding_config.grounding_dino_config
        self.grounding_dino_checkpoint = grounding_config.grounding_dino_checkpoint

        requested_device = device if device is not None else grounding_config.device
        if not requested_device:
            requested_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_device = torch.device(requested_device)
        if torch_device.type == "cuda" and not torch.cuda.is_available():
            logging.warning(
                "Configured CUDA device %s is unavailable; falling back to CPU for grounding.",
                requested_device,
            )
            torch_device = torch.device("cpu")
        elif torch_device.type == "cuda":
            cuda_count = torch.cuda.device_count()
            cuda_index = torch_device.index if torch_device.index is not None else 0
            if cuda_index >= cuda_count:
                logging.warning(
                    "Configured CUDA device %s is unavailable; only %d CUDA device(s) detected. "
                    "Falling back to cuda:0.",
                    requested_device,
                    cuda_count,
                )
                torch_device = torch.device("cuda:0")
        self.device = str(torch_device)
        self.box_threshold = grounding_config.box_threshold
        self.text_threshold = grounding_config.text_threshold

        logging.info("==> Grounding configuration loaded from unified config.yaml")

        # Initialize SAM2 (use external instance if provided)
        if sam2_predictor is not None:
            # Use externally provided SAM2 predictor (for sharing with other modules)
            self.sam2_predictor = sam2_predictor
            self.sam2_model = None  # Not owned by this instance
            self._external_sam2 = True
            logging.info("==> DinoGrounding using external SAM2 predictor (shared instance)")
        else:
            # build SAM2 image predictor
            # SAM2 config/checkpoint paths are expected to follow the upstream
            # Grounded-SAM-2 setup described in README.md.
            self.sam2_model = build_sam2(self.sam2_model_config, self.sam2_checkpoint, device=self.device)
            self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
            self._external_sam2 = False
            logging.info("==> DinoGrounding initialized internal SAM2 instance")

        # build grounding dino model
        self.grounding_model = load_model(
            model_config_path=self.grounding_dino_config,
            model_checkpoint_path=self.grounding_dino_checkpoint,
            device=self.device
        )

    def restore_config(self, config_file):
        """
        Restore the configuration file
        :param config_file: path to the configuration file
        :return: None
        """
        config_paras = {
            "sam2_checkpoint": self.sam2_checkpoint,
            "sam2_model_config": self.sam2_model_config,
            "grounding_dino_config": self.grounding_dino_config,
            "grounding_dino_checkpoint": self.grounding_dino_checkpoint,
            "box_threshold": self.box_threshold,
            "text_threshold": self.text_threshold,
        }
        # write config_paras to a json file
        with open(config_file, 'w') as f:
            json.dump(config_paras, f, indent=4)
        logging.info("Configuration saved to %s", config_file)

    def read_config(self, config_file):
        with open(config_file, 'r') as f:
            config_paras = json.load(f)
        self.sam2_checkpoint = config_paras["sam2_checkpoint"]
        self.sam2_model_config = config_paras["sam2_model_config"]
        self.grounding_dino_config = config_paras["grounding_dino_config"]
        self.grounding_dino_checkpoint = config_paras["grounding_dino_checkpoint"]
        self.box_threshold = config_paras["box_threshold"]
        self.text_threshold = config_paras["text_threshold"]
        return 0

    def reset(self):
        """
        Reset the grounding model
        :return: None
        """
        self.sam2_model = build_sam2(self.sam2_model_config, self.sam2_checkpoint, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

        # build grounding dino model
        self.grounding_model = load_model(
            model_config_path=self.grounding_dino_config,
            model_checkpoint_path=self.grounding_dino_checkpoint,
            device=self.device
        )
        logging.info("==> Grounding model reset successfully")
        return 0

    def grounding(self, image_rgb, text_prompt):
        """
        Grounding the object in the image
        :param image_rgb: image in RGB format
        :param text_prompt: text prompt for grounding
        :return: bounding boxes and labels
            detections.xyxy: (n, 4)
            detections.mask: (n, h, w)
            detections.class_id: (n, )
        """

        # setup the input image and text prompt for SAM 2 and Grounding DINO
        # VERY important: text queries need to be lowercased + end with a dot

        # initial
        text = text_prompt

        if text[-1] != ".":
            text += "."
        image_source, image = self._load_data_from_rgb(image_rgb)

        self.sam2_predictor.set_image(image_source.copy())

        device_type = torch.device(self.device).type
        autocast_context = nullcontext()
        if device_type == "cuda":
            cuda_index = torch.device(self.device).index or 0
            cuda_major = torch.cuda.get_device_properties(cuda_index).major
            autocast_dtype = torch.bfloat16 if cuda_major >= 8 else torch.float16
            autocast_context = torch.autocast(device_type=device_type, dtype=autocast_dtype)
        with autocast_context:
            if device_type == "cuda" and torch.cuda.is_available():
                cuda_index = torch.device(self.device).index or 0
                if torch.cuda.get_device_properties(cuda_index).major >= 8:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True

            logging.info("SAM text prompt: %s", text)
            boxes, confidences, labels = predict(
                model=self.grounding_model,
                image=image,
                caption=text,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device=self.device
            )

        # process the box prompt for SAM 2
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        if len(input_boxes) == 0:
            masks = np.zeros((0, h, w), dtype=bool)
        else:
            masks, scores, logits = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )

        """
        Post-process the output of the model to get the masks, scores, and logits for visualization
        """
        # convert the shape to (n, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)


        confidences = confidences.detach().cpu().numpy().tolist()
        class_names = labels
        class_ids = np.arange(len(class_names), dtype=np.int64)

        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence
            in zip(class_names, confidences)
        ]

        detections = sv.Detections(
            xyxy=input_boxes,  # (n, 4)
            mask=masks.astype(bool),  # (n, h, w)
            class_id=class_ids  # (n,)
        )

        # Return detections with class_names for batch detection matching - Modified 2026-01-18
        # Store class_names as an attribute for downstream use
        detections.class_names = class_names  # List of detected object names

        return detections

    @staticmethod
    def _load_data_from_rgb(image_rgb):
        """Compatibility shim for GroundingDINO builds that no longer expose load_data()."""
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_source = np.asarray(image_rgb).copy()
        image_pil = Image.fromarray(image_source.astype(np.uint8)).convert("RGB")
        image_transformed, _ = transform(image_pil, None)
        return image_source, image_transformed
