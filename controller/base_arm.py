import time
import math
import functools
import numpy as np
from abc import ABC, abstractmethod

class BaseArm():
    def __init__(self, config_file=None):
        pass
    
    @abstractmethod
    def reset(self):
        raise NotImplementedError("Subclasses should implement this method")

    @abstractmethod
    def restore_config(self, config_file: str):
        raise NotImplementedError("Subclasses should implement this method")

    @abstractmethod
    def read_config(self, config_file: str):
        raise NotImplementedError("Subclasses should implement this method")

    @abstractmethod
    def moveto(self):
        raise NotImplementedError("Subclasses should implement this method")
    
    @abstractmethod
    def get_status(self):
        raise NotImplementedError("Subclasses should implement this method")
    
    @abstractmethod
    def act_gripper(self, status: int):
        raise NotImplementedError("Subclasses should implement this method")


