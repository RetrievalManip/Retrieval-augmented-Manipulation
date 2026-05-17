from abc import ABC, abstractmethod

import torch
from torchvision.ops import box_convert
import supervision as sv


class BaseGrounding():
    def __init__(self):
        pass

    @abstractmethod
    def restore_config(self, config_file):
        """
        Restore the configuration file
        :param config_file: path to the configuration file
        :return: None
        """
        pass

    @abstractmethod
    def read_config(self, config_file):
        """
        Read the configuration file
        :param config_file: path to the configuration file
        :return: None
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the grounding model
        :return: None
        """
        pass

    @abstractmethod
    def grounding(self, image_bgr, text_prompt):
        """
        Grounding the object in the image
        :param image_bgr: image in BGR format
        :param text_prompt: text prompt
        :return: None
        """
        h = 480
        w = 640

        detections = sv.Detections(
            xyxy=torch.tensor([[0, 0, 0, 0]]),
            mask=torch.zeros((1, h, w)),
            confidence=torch.tensor([0]),
            class_id=torch.tensor([0]),
        )
        return detections
